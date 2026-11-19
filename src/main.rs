use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::bail;
use bytes::Bytes;
use clap::Parser;
use futures_util::StreamExt;
use http::HeaderMap;
use serde_json::{Value, json};
use tracing_subscriber::EnvFilter;

use zero_opencode::cli::{Cli, Command};
use zero_opencode::config::AppConfig;
use zero_opencode::opencode::executor::OpenCodeExecutor;
use zero_opencode::opencode::models::{FreeModel, ModelCatalog};
use zero_opencode::opencode::session::SessionManager;
use zero_opencode::opencode::tls_client::build_opencode_client;
use zero_opencode::opencode::transform::transform_chat_request;
use zero_opencode::proxy::ProxyRuntime;
use zero_opencode::proxy::health::{ProbeEndpoints, run_probe_wave};
use zero_opencode::server::{self, handlers::AppState};
use zero_opencode::stream::aggregator::aggregate_sse_to_json;
use zero_opencode::stream::passthrough::into_normalized_sse_stream;
use zero_opencode::stream::sse_parser::{SseLineParser, extract_sse_data};

fn init_tracing(level: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let config = AppConfig::load(cli.config.as_deref())?;
    init_tracing(&config.logging.level);

    let base_client = build_opencode_client(
        config.opencode.proxy.as_deref(),
        config.opencode.timeout_secs,
    )?;
    let sessions = Arc::new(SessionManager::new(Duration::from_secs(
        config.opencode.session_ttl_secs,
    )));
    let cmd = cli.command.unwrap_or(Command::Serve);
    let start_background = matches!(cmd, Command::Serve | Command::Test { .. });

    let proxy_runtime = if config.opencode.proxy_pool.enabled {
        Some(
            ProxyRuntime::initialize(
                &config,
                &base_client,
                Arc::clone(&sessions),
                start_background,
            )
            .await?,
        )
    } else {
        None
    };

    let mut catalog = ModelCatalog::new(
        base_client.clone(),
        Duration::from_secs(config.opencode.models_ttl_secs),
    );
    let executor = if let Some(rt) = &proxy_runtime {
        catalog = catalog.with_proxy_pool(Arc::clone(&rt.pool), rt.pool_client.clone());
        OpenCodeExecutor::new_pooled(
            base_client.clone(),
            rt.pool_client.clone(),
            Arc::clone(&rt.pool),
            Arc::clone(&sessions),
            config.opencode.retry.clone(),
        )
    } else {
        OpenCodeExecutor::new(
            base_client.clone(),
            Arc::clone(&sessions),
            config.opencode.retry.clone(),
        )
    };

    let result = match cmd {
        Command::Serve => {
            let state = Arc::new(AppState {
                config,
                catalog,
                executor,
                proxy_pool: proxy_runtime.as_ref().map(|r| Arc::clone(&r.pool)),
                started_at: Instant::now(),
            });
            server::serve(state).await
        }
        Command::Models { json: as_json } => run_models_cmd(&catalog, as_json).await,
        Command::Test { model } => {
            run_test_cmd(
                &config,
                &catalog,
                &executor,
                proxy_runtime.as_ref(),
                model.as_deref(),
            )
            .await
        }
        Command::Proxies { json, probe, limit } => {
            run_proxies_cmd(
                &config,
                proxy_runtime.as_ref(),
                &base_client,
                &sessions,
                json,
                probe,
                limit,
            )
            .await
        }
    };

    if let Some(rt) = proxy_runtime {
        rt.shutdown().await;
    }

    result
}

async fn run_models_cmd(catalog: &ModelCatalog, as_json: bool) -> anyhow::Result<()> {
    let models = catalog.list().await?;
    if as_json {
        let json_list: Vec<Value> = models.iter().map(FreeModel::to_api_json).collect();
        println!("{}", serde_json::to_string_pretty(&json_list)?);
        return Ok(());
    }

    if models.is_empty() {
        println!("No free models returned by OpenCode.");
        return Ok(());
    }

    println!(
        "{:<30} {:<10} {:<10} {:<26} THINKING LEVELS",
        "ID", "CONTEXT", "MAX OUT", "CAPABILITIES"
    );
    println!("{}", "-".repeat(110));
    for m in &models {
        let ctx = m
            .context_window
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into());
        let out = m
            .max_output_tokens
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into());
        let caps = m.capability_tags();
        let efforts = m
            .thinking_levels
            .as_ref()
            .map(|v| v.join(","))
            .unwrap_or_else(|| "-".into());
        println!(
            "{:<30} {:<10} {:<10} {:<26} {}",
            m.id, ctx, out, caps, efforts
        );
    }
    println!("\nTotal available free models: {}", models.len());
    Ok(())
}

async fn run_test_cmd(
    config: &AppConfig,
    catalog: &ModelCatalog,
    executor: &OpenCodeExecutor,
    proxy_runtime: Option<&ProxyRuntime>,
    requested_model: Option<&str>,
) -> anyhow::Result<()> {
    if let Some(rt) = proxy_runtime {
        let stats = rt.pool.stats();
        let chain_note = stats
            .chained_via
            .as_deref()
            .map(|c| format!(" (chained via {c})"))
            .unwrap_or_default();
        if stats.healthy == 0 {
            let wait_secs = config.opencode.proxy_pool.health_check.startup_wait_secs;
            println!(
                "Warming proxy pool{} ({} candidates, waiting up to {wait_secs}s for first healthy proxy)...",
                chain_note, stats.total
            );
            let ready = rt.pool.wait_for_healthy(Duration::from_secs(wait_secs)).await;
            if !ready {
                println!(
                    "No healthy proxy verified within {wait_secs}s yet; live executor will keep probing/switching up to failover_timeout."
                );
            }
        } else {
            println!(
                "Proxy pool ready{}: {} healthy / {} total",
                chain_note, stats.healthy, stats.total
            );
        }
    }

    let models = catalog.list().await?;
    let selected: Option<FreeModel> = if let Some(m) = requested_model {
        catalog.get(m).await?
    } else {
        ["big-pickle", "mimo-v2.6-flash-free", "deepseek-v4.1-flash-free"]
            .iter()
            .find_map(|pref| models.iter().find(|m| m.id == *pref).cloned())
            .or_else(|| models.first().cloned())
    };

    let target_model = requested_model
        .map(ToString::to_string)
        .or_else(|| selected.as_ref().map(|m| m.id.clone()))
        .ok_or_else(|| anyhow::anyhow!("no free models available to test"))?;

    println!("Testing OpenCode Free model: {target_model}");

    // 1. Non-streaming (forced upstream SSE -> aggregated JSON)
    let t0 = Instant::now();
    let mut json_body = json!({
        "model": target_model,
        "stream": false,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Reply with just: OK"}]
    });
    let session = executor
        .sessions()
        .resolve_session(&HeaderMap::new(), &json_body);
    let transformed = transform_chat_request(&mut json_body, &target_model, selected.as_ref());
    let resp = executor
        .execute_chat_with_body(
            &HeaderMap::new(),
            Some(&json_body),
            Bytes::from(serde_json::to_vec(&json_body)?),
            &session,
        )
        .await?;
    let used_proxy_1 = resp.used_proxy.clone();

    if !resp.status.is_success() {
        let (status, _, err_bytes) = resp.into_bytes().await?;
        bail!(
            "non-streaming test failed with HTTP {}: {}",
            status.as_u16(),
            String::from_utf8_lossy(&err_bytes)
        );
    }

    let aggregated = aggregate_sse_to_json(
        resp.body,
        &transformed.model,
        &transformed.rename_map,
        config.streaming.keep_reasoning_on_json,
    )
    .await?;
    let json_text = aggregated["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    let json_reasoning = aggregated["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .unwrap_or("");
    println!(
        "[PASS] non-streaming ({:?}{}): content={:?}{}",
        t0.elapsed(),
        used_proxy_1
            .as_deref()
            .map(|p| format!(" via {p}"))
            .unwrap_or_default(),
        json_text,
        if json_reasoning.is_empty() {
            String::new()
        } else {
            format!(" (reasoning={} chars)", json_reasoning.len())
        }
    );

    // 2. Streaming (SSE passthrough)
    let t1 = Instant::now();
    let mut sse_body = json!({
        "model": target_model,
        "stream": true,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Reply with just: OK"}]
    });
    let session2 = executor
        .sessions()
        .resolve_session(&HeaderMap::new(), &sse_body);
    let transformed2 = transform_chat_request(&mut sse_body, &target_model, selected.as_ref());
    let resp2 = executor
        .execute_chat_with_body(
            &HeaderMap::new(),
            Some(&sse_body),
            Bytes::from(serde_json::to_vec(&sse_body)?),
            &session2,
        )
        .await?;
    let used_proxy_2 = resp2.used_proxy.clone();

    if !resp2.status.is_success() {
        let (status, _, err_bytes) = resp2.into_bytes().await?;
        bail!(
            "streaming test failed with HTTP {}: {}",
            status.as_u16(),
            String::from_utf8_lossy(&err_bytes)
        );
    }

    let mut sse_stream = into_normalized_sse_stream(
        resp2.body,
        transformed2.rename_map,
        transformed2.model,
    );
    let mut parser = SseLineParser::new();
    let mut content_acc = String::new();
    let mut reasoning_acc = String::new();
    let mut saw_done = false;
    while let Some(chunk) = sse_stream.next().await {
        let bytes = chunk?;
        for line in parser.push_chunk(&bytes) {
            if let Some(payload) = extract_sse_data(&line) {
                if payload == "[DONE]" {
                    saw_done = true;
                } else if let Ok(v) = serde_json::from_str::<Value>(payload) {
                    if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                        content_acc.push_str(c);
                    }
                    if let Some(r) = v["choices"][0]["delta"]["reasoning_content"].as_str() {
                        reasoning_acc.push_str(r);
                    }
                }
            }
        }
    }
    if !saw_done {
        bail!("streaming test ended without data: [DONE]");
    }
    println!(
        "[PASS] streaming ({:?}{}): content={:?}, reasoning={} chars, saw [DONE]",
        t1.elapsed(),
        used_proxy_2
            .as_deref()
            .map(|p| format!(" via {p}"))
            .unwrap_or_default(),
        content_acc,
        reasoning_acc.len()
    );

    Ok(())
}

async fn run_proxies_cmd(
    config: &AppConfig,
    existing_rt: Option<&ProxyRuntime>,
    base_client: &wreq::Client,
    sessions: &Arc<SessionManager>,
    as_json: bool,
    probe: bool,
    limit: usize,
) -> anyhow::Result<()> {
    let pp = &config.opencode.proxy_pool;
    if existing_rt.is_none() && pp.sources.is_empty() && pp.static_proxies.is_empty() {
        bail!("no opencode.proxy_pool.sources or static_proxies configured");
    }

    let owned_rt = if existing_rt.is_none() {
        Some(ProxyRuntime::initialize(config, base_client, Arc::clone(sessions), false).await?)
    } else {
        None
    };

    let rt = existing_rt
        .or(owned_rt.as_ref())
        .expect("runtime initialized");

    let exec_res: anyhow::Result<()> = async {
        let load = &rt.load_summary;
        if load.sources_total > 0 && load.sources_succeeded == 0 && load.proxies.is_empty() {
            bail!(
                "failed to fetch all {} configured proxy source URL(s)",
                load.sources_total
            );
        }

        if probe {
            run_probe_wave(
                &rt.pool,
                &rt.probe_client,
                &rt.pool_client,
                sessions,
                &ProbeEndpoints::default(),
            )
            .await;
        }

        let stats = rt.pool.stats();
        let items = rt.pool.snapshot_proxies(limit);

        if as_json {
            let out = json!({
                "stats": stats,
                "protocol_counts": {
                    "socks5": load.protocol_counts.socks5,
                    "socks4": load.protocol_counts.socks4,
                    "http": load.protocol_counts.http,
                    "https": load.protocol_counts.https,
                },
                "sources": {
                    "total": load.sources_total,
                    "succeeded": load.sources_succeeded,
                    "failed": load.sources_failed,
                },
                "proxies": items,
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
            return Ok(());
        }

        println!(
            "Proxy pool: enabled={}, total={} (static={}, remote={}), chained_via={}",
            stats.enabled,
            stats.total,
            load.static_count,
            load.remote_count,
            stats.chained_via.as_deref().unwrap_or("none"),
        );
        println!(
            "Protocols: socks5={}, socks4={}, http={}, https={} | Sources ok={}/{}",
            load.protocol_counts.socks5,
            load.protocol_counts.socks4,
            load.protocol_counts.http,
            load.protocol_counts.https,
            load.sources_succeeded,
            load.sources_total,
        );
        println!(
            "Health: healthy={}, untested={}, cooldown={}, dead={}, hop1_healthy={}",
            stats.healthy, stats.untested, stats.cooldown, stats.dead, stats.hop1_healthy,
        );
        println!();
        println!(
            "{:<6} {:<24} {:<32} {:<10} {:<8} DETAIL",
            "ID", "STATUS", "PROXY", "RTT(ms)", "STATIC"
        );
        println!("{}", "-".repeat(100));
        for item in &items {
            let rtt = item
                .ewma_rtt_ms
                .or(item.rtt_ms)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into());
            let detail = match (item.cooldown_remaining_secs, item.last_error.as_deref()) {
                (Some(s), Some(e)) => format!("reset {s}s | {e}"),
                (Some(s), None) => format!("reset {s}s"),
                (None, Some(e)) => e.to_string(),
                (None, None) => String::new(),
            };
            println!(
                "{:<6} {:<24} {:<32} {:<10} {:<8} {}",
                item.id, item.status, item.proxy, rtt, item.is_static, detail
            );
        }
        Ok(())
    }
    .await;

    if let Some(rt_owned) = owned_rt {
        rt_owned.shutdown().await;
    }

    exec_res
}
