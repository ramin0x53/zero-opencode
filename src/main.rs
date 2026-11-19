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

    let client = build_opencode_client(
        config.opencode.proxy.as_deref(),
        config.opencode.timeout_secs,
    )?;
    let catalog = ModelCatalog::new(
        client.clone(),
        Duration::from_secs(config.opencode.models_ttl_secs),
    );
    let sessions = Arc::new(SessionManager::new(Duration::from_secs(
        config.opencode.session_ttl_secs,
    )));
    let executor = OpenCodeExecutor::new(client, sessions, config.opencode.retry.clone());

    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => {
            let state = Arc::new(AppState {
                config,
                catalog,
                executor,
                started_at: Instant::now(),
            });
            server::serve(state).await?;
        }
        Command::Models { json: as_json } => {
            run_models_cmd(&catalog, as_json).await?;
        }
        Command::Test { model } => {
            run_test_cmd(&config, &catalog, &executor, model.as_deref()).await?;
        }
    }

    Ok(())
}

async fn run_models_cmd(catalog: &ModelCatalog, as_json: bool) -> anyhow::Result<()> {
    let models = catalog.list().await?;
    if as_json {
        println!("{}", serde_json::to_string_pretty(&models)?);
        return Ok(());
    }

    if models.is_empty() {
        println!("No free models returned by OpenCode.");
        return Ok(());
    }

    println!(
        "{:<32} {:<10} {:<10} {:<18} THINKING LEVELS",
        "ID", "CONTEXT", "MAX OUT", "LANE"
    );
    println!("{}", "-".repeat(92));
    for m in &models {
        let ctx = m
            .context_window
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into());
        let out = m
            .max_output_tokens
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into());
        let efforts = m
            .thinking_levels
            .as_ref()
            .map(|v| v.join(","))
            .unwrap_or_else(|| "-".into());
        println!(
            "{:<32} {:<10} {:<10} {:<18} {}",
            m.id, ctx, out, m.recommended_api, efforts
        );
    }
    println!("\nTotal available free models: {}", models.len());
    Ok(())
}

async fn run_test_cmd(
    config: &AppConfig,
    catalog: &ModelCatalog,
    executor: &OpenCodeExecutor,
    requested_model: Option<&str>,
) -> anyhow::Result<()> {
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
        .execute_chat(
            &HeaderMap::new(),
            Bytes::from(serde_json::to_vec(&json_body)?),
            &session,
        )
        .await?;

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
        "[PASS] non-streaming ({:?}): content={:?}{}",
        t0.elapsed(),
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
        .execute_chat(
            &HeaderMap::new(),
            Bytes::from(serde_json::to_vec(&sse_body)?),
            &session2,
        )
        .await?;

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
        "[PASS] streaming ({:?}): content={:?}, reasoning={} chars, saw [DONE]",
        t1.elapsed(),
        content_acc,
        reasoning_acc.len()
    );

    Ok(())
}
