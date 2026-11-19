use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use http::{HeaderMap, StatusCode};
use serde_json::{Value, json};
use tokio::sync::{Semaphore, watch};
use tracing::{debug, info, warn};
use wreq::Client;

use crate::opencode::executor::{
    OPENCODE_CHAT_URL, build_upstream_headers, is_html_response, is_overload_text,
    parse_retry_after,
};
use crate::opencode::models::{FreeModel, OPENCODE_MODELS_URL, filter_free_models_json};
use crate::opencode::session::SessionManager;
use crate::opencode::tls_client::resolve_wreq_proxy;
use crate::opencode::transform::transform_chat_request;
use crate::proxy::chain::ChainBridge;
use crate::proxy::endpoint::ProxyEndpoint;
use crate::proxy::loader::fetch_all_proxies;
use crate::proxy::pool::ProxyPool;
use crate::stream::sse_parser::{SseLineParser, extract_sse_data};

const PREFERRED_PROBE_MODELS: &[&str] = &[
    "big-pickle",
    "mimo-v2.6-flash-free",
    "deepseek-v4.1-flash-free",
];

/// Hardcoded upstream endpoints for two-stage health checks.
#[derive(Debug, Clone)]
pub struct ProbeEndpoints {
    pub models_url: String,
    pub chat_url: String,
}

impl Default for ProbeEndpoints {
    fn default() -> Self {
        Self {
            models_url: OPENCODE_MODELS_URL.to_string(),
            chat_url: OPENCODE_CHAT_URL.to_string(),
        }
    }
}

#[cfg(test)]
impl ProbeEndpoints {
    pub fn for_tests(models_url: impl Into<String>, chat_url: impl Into<String>) -> Self {
        Self {
            models_url: models_url.into(),
            chat_url: chat_url.into(),
        }
    }
}

/// Result of Stage 1 probe (`GET /zen/v1/models`).
pub enum Stage1Result {
    Ok { rtt_ms: u64, probe_model: Box<FreeModel> },
    Hop1Down,
    RateLimited { retry_after: Option<Duration>, detail: String },
    Overloaded,
    Forbidden(String),
    Dead(String),
}

/// Result of Stage 2 probe (`POST /inference/v1/chat/completions`).
pub enum Stage2Result {
    Healthy { rtt_ms: u64 },
    Hop1Down,
    RateLimited { retry_after: Option<Duration>, detail: String },
    Overloaded,
    Forbidden(String),
    Dead(String),
}

fn select_probe_model(models: &[FreeModel]) -> Option<FreeModel> {
    PREFERRED_PROBE_MODELS
        .iter()
        .find_map(|pref| models.iter().find(|m| m.id == *pref).cloned())
        .or_else(|| models.first().cloned())
}

/// Stage 1: Verify proxy reachability, TLS integrity, and OpenCode free model catalog availability.
pub async fn probe_stage1(
    client: &Client,
    proxy: &ProxyEndpoint,
    bridge: Option<&Arc<ChainBridge>>,
    models_url: &str,
) -> Stage1Result {
    let wreq_proxy = match resolve_wreq_proxy(proxy, bridge) {
        Ok(p) => p,
        Err(e) => return Stage1Result::Dead(e.to_string()),
    };

    let started = Instant::now();
    let res = match client.get(models_url).proxy(wreq_proxy).send().await {
        Ok(r) => r,
        Err(err) => {
            if bridge.is_some_and(|b| b.is_hop1_failing()) {
                return Stage1Result::Hop1Down;
            }
            return Stage1Result::Dead(format!("stage1 transport error: {err}"));
        }
    };

    let rtt_ms = started.elapsed().as_millis() as u64;
    let status = res.status();
    let headers = res.headers().clone();
    let body = match res.text().await {
        Ok(t) => t,
        Err(err) => return Stage1Result::Dead(format!("stage1 read error: {err}")),
    };

    if status == StatusCode::PROXY_AUTHENTICATION_REQUIRED {
        return Stage1Result::Dead("HTTP 407 proxy authentication required".to_string());
    }
    if is_html_response(&headers, body.as_bytes()) {
        return Stage1Result::Dead("stage1 returned HTML (WAF/captive portal)".to_string());
    }
    if status == StatusCode::FORBIDDEN {
        return Stage1Result::Forbidden("stage1 HTTP 403".to_string());
    }
    if status == StatusCode::TOO_MANY_REQUESTS {
        if is_overload_text(&body) {
            return Stage1Result::Overloaded;
        }
        return Stage1Result::RateLimited {
            retry_after: parse_retry_after(&headers),
            detail: "stage1 HTTP 429".to_string(),
        };
    }
    if !status.is_success() {
        return Stage1Result::Dead(format!("stage1 HTTP {}", status.as_u16()));
    }

    let Ok(raw_json) = serde_json::from_str::<Value>(&body) else {
        return Stage1Result::Dead("stage1 returned invalid JSON".to_string());
    };
    if !raw_json.get("data").is_some_and(Value::is_array) {
        return Stage1Result::Dead("stage1 JSON missing data array".to_string());
    }

    let Ok(models) = filter_free_models_json(&body) else {
        return Stage1Result::Dead("stage1 model catalog parse error".to_string());
    };
    let Some(probe_model) = select_probe_model(&models) else {
        return Stage1Result::Dead("stage1 returned zero active free models".to_string());
    };

    Stage1Result::Ok {
        rtt_ms,
        probe_model: Box::new(probe_model),
    }
}

/// Stage 2: Verify the proxy IP has not already hit OpenCode's free inference limit.
pub async fn probe_stage2(
    client: &Client,
    proxy: &ProxyEndpoint,
    bridge: Option<&Arc<ChainBridge>>,
    chat_url: &str,
    probe_model: &FreeModel,
    sessions: &SessionManager,
    stage_timeout: Duration,
) -> Stage2Result {
    let wreq_proxy = match resolve_wreq_proxy(proxy, bridge) {
        Ok(p) => p,
        Err(e) => return Stage2Result::Dead(e.to_string()),
    };

    let mut body = json!({
        "model": probe_model.id,
        "stream": true,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}]
    });
    transform_chat_request(&mut body, &probe_model.id, Some(probe_model));
    let body_bytes = match serde_json::to_vec(&body) {
        Ok(b) => b,
        Err(e) => return Stage2Result::Dead(e.to_string()),
    };

    let sid = sessions.generate_session_id();
    let headers = build_upstream_headers(&HeaderMap::new(), &sid);

    let started = Instant::now();
    let res = match client
        .post(chat_url)
        .proxy(wreq_proxy)
        .headers(headers)
        .body(body_bytes)
        .send()
        .await
    {
        Ok(r) => r,
        Err(err) => {
            if bridge.is_some_and(|b| b.is_hop1_failing()) {
                return Stage2Result::Hop1Down;
            }
            return Stage2Result::Dead(format!("stage2 transport error: {err}"));
        }
    };

    let status = res.status();
    let resp_headers = res.headers().clone();

    if !status.is_success() {
        let text = res.text().await.unwrap_or_default();
        if status == StatusCode::PROXY_AUTHENTICATION_REQUIRED {
            return Stage2Result::Dead("stage2 HTTP 407".to_string());
        }
        if is_html_response(&resp_headers, text.as_bytes()) {
            return Stage2Result::Dead("stage2 returned HTML".to_string());
        }
        if status == StatusCode::FORBIDDEN {
            return Stage2Result::Forbidden("stage2 HTTP 403".to_string());
        }
        if status == StatusCode::TOO_MANY_REQUESTS {
            if is_overload_text(&text) {
                return Stage2Result::Overloaded;
            }
            return Stage2Result::RateLimited {
                retry_after: parse_retry_after(&resp_headers),
                detail: "stage2 HTTP 429 quota limit".to_string(),
            };
        }
        return Stage2Result::Dead(format!("stage2 HTTP {}", status.as_u16()));
    }

    if is_html_response(&resp_headers, &[]) {
        return Stage2Result::Dead("stage2 200 returned text/html".to_string());
    }

    // Peek just the first non-empty SSE chunk to verify OpenCode did not embed an error event
    // inside a 200 stream, then drop immediately without waiting for full generation.
    let mut stream = res.bytes_stream();
    let deadline = tokio::time::Instant::now() + stage_timeout;
    let mut parser = SseLineParser::new();

    loop {
        let chunk_opt = match tokio::time::timeout_at(deadline, stream.next()).await {
            Ok(item) => item,
            Err(_) => {
                return Stage2Result::Dead("stage2 timed out waiting for first SSE chunk".into());
            }
        };

        match chunk_opt {
            None => {
                if let Some(last) = parser.finish()
                    && let Some(res) = classify_stage2_sse_line(&last, &resp_headers)
                {
                    return res;
                }
                return Stage2Result::Dead("stage2 upstream closed with empty stream".into());
            }
            Some(Err(err)) => {
                return Stage2Result::Dead(format!("stage2 stream error: {err}"));
            }
            Some(Ok(bytes)) => {
                if bytes.is_empty() {
                    continue;
                }
                if is_html_response(&resp_headers, &bytes) {
                    return Stage2Result::Dead("stage2 stream returned HTML".to_string());
                }
                let lines = parser.push_chunk(&bytes);
                for line in &lines {
                    if let Some(res) = classify_stage2_sse_line(line, &resp_headers) {
                        return res;
                    }
                }
                // Check whether any complete data line arrived without error.
                if lines.iter().any(|l| extract_sse_data(l).is_some_and(|p| !p.is_empty())) {
                    return Stage2Result::Healthy {
                        rtt_ms: started.elapsed().as_millis() as u64,
                    };
                }
            }
        }
    }
}

fn classify_stage2_sse_line(line: &str, headers: &HeaderMap) -> Option<Stage2Result> {
    let payload = extract_sse_data(line)?;
    if payload.is_empty() || payload == "[DONE]" {
        return None;
    }
    let val = serde_json::from_str::<Value>(payload).ok()?;
    let err_val = val.get("error").filter(|v| !v.is_null())?;
    let err_str = err_val.to_string();
    let lower = err_str.to_ascii_lowercase();
    let status = err_val.get("status").and_then(Value::as_u64).unwrap_or(0);

    if is_overload_text(&err_str) {
        return Some(Stage2Result::Overloaded);
    }
    if status == 429
        || lower.contains("freeusagelimiterror")
        || lower.contains("rate limit")
        || lower.contains("too many requests")
        || lower.contains("quota exceeded")
    {
        return Some(Stage2Result::RateLimited {
            retry_after: parse_retry_after(headers),
            detail: "stage2 SSE 429 limit".to_string(),
        });
    }
    if status == 403 || lower.contains("freetiererror") || lower.contains("forbidden") {
        return Some(Stage2Result::Forbidden("stage2 SSE 403".to_string()));
    }
    Some(Stage2Result::Dead(format!("stage2 SSE error: {err_str}")))
}

/// Probe `Untested` proxies concurrently using Stage 1 + Stage 2 until
/// `healthy >= target_healthy` or `Untested` candidates are exhausted.
pub async fn run_probe_wave(
    pool: &Arc<ProxyPool>,
    probe_client: &Client,
    pool_client: &Client,
    sessions: &Arc<SessionManager>,
    endpoints: &ProbeEndpoints,
) {
    let cfg = pool.config().clone();
    let concurrency = cfg.health_check.concurrency.max(1);
    let target_healthy = cfg.health_check.target_healthy;
    let stage_timeout = Duration::from_secs(cfg.health_check.timeout_secs);
    let bridge = pool.bridge().cloned();

    let sem = Arc::new(Semaphore::new(concurrency));

    loop {
        if pool.healthy_count() >= target_healthy {
            break;
        }
        if let Some(b) = &bridge
            && b.is_hop1_failing()
            && !b.check_hop1_reachable().await
        {
            warn!(
                error = ?b.last_hop1_error(),
                "chain proxy (Hop 1) unreachable; pausing probe wave"
            );
            break;
        }

        let batch = pool.take_untested_batch(concurrency);
        if batch.is_empty() {
            break;
        }

        let mut handles = Vec::with_capacity(batch.len());
        for ep in batch {
            let permit = sem.clone().acquire_owned().await.expect("semaphore open");
            let pool_c = Arc::clone(pool);
            let probe_cli = probe_client.clone();
            let live_cli = pool_client.clone();
            let sess = Arc::clone(sessions);
            let br = bridge.clone();
            let ep_urls = endpoints.clone();
            let hot_count = cfg.health_check.hot_standby_count;

            handles.push(tokio::spawn(async move {
                let _permit = permit;

                // Stop early if another worker already reached target_healthy.
                if pool_c.healthy_count() >= target_healthy {
                    pool_c.release_probe(ep.id);
                    return;
                }
                if br.as_ref().is_some_and(|b| b.is_hop1_failing()) {
                    pool_c.release_probe(ep.id);
                    return;
                }

                match probe_stage1(&probe_cli, &ep, br.as_ref(), &ep_urls.models_url).await {
                    Stage1Result::Ok { probe_model, .. } => {
                        if pool_c.healthy_count() >= target_healthy {
                            pool_c.release_probe(ep.id);
                            return;
                        }
                        match probe_stage2(
                            &probe_cli,
                            &ep,
                            br.as_ref(),
                            &ep_urls.chat_url,
                            &probe_model,
                            &sess,
                            stage_timeout,
                        )
                        .await
                        {
                            Stage2Result::Healthy { rtt_ms } => {
                                pool_c.report_success(ep.id, rtt_ms);
                                debug!(proxy = %ep.redacted(), rtt_ms, "proxy verified healthy (Stage 1 + Stage 2)");
                                if pool_c.is_top_hot_standby(ep.id, hot_count) {
                                    // Pre-warm TLS + chain tunnel in pool_client so failover is instant.
                                    let _ = probe_stage1(
                                        &live_cli,
                                        &ep,
                                        br.as_ref(),
                                        &ep_urls.models_url,
                                    )
                                    .await;
                                }
                            }
                            Stage2Result::Hop1Down => pool_c.release_probe(ep.id),
                            Stage2Result::RateLimited {
                                retry_after,
                                detail,
                            } => pool_c.report_rate_limited(ep.id, retry_after, detail),
                            Stage2Result::Overloaded => pool_c.report_overloaded(ep.id),
                            Stage2Result::Forbidden(detail) => {
                                pool_c.report_forbidden(ep.id, detail)
                            }
                            Stage2Result::Dead(detail) => pool_c.report_dead(ep.id, detail),
                        }
                    }
                    Stage1Result::Hop1Down => pool_c.release_probe(ep.id),
                    Stage1Result::RateLimited {
                        retry_after,
                        detail,
                    } => pool_c.report_rate_limited(ep.id, retry_after, detail),
                    Stage1Result::Overloaded => pool_c.report_overloaded(ep.id),
                    Stage1Result::Forbidden(detail) => pool_c.report_forbidden(ep.id, detail),
                    Stage1Result::Dead(detail) => pool_c.report_dead(ep.id, detail),
                }
            }));
        }

        for h in handles {
            let _ = h.await;
        }
    }
}

/// Spawn the background prober, hot-standby keep-alive warmer, and list refresher tasks.
pub fn spawn_background_tasks(
    pool: Arc<ProxyPool>,
    base_client: Client,
    probe_client: Client,
    pool_client: Client,
    sessions: Arc<SessionManager>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Vec<tokio::task::JoinHandle<()>> {
    let endpoints = ProbeEndpoints::default();
    let cfg = pool.config().clone();

    // 1. Demand-driven two-stage prober loop
    let prober_handle = {
        let pool = Arc::clone(&pool);
        let base_client = base_client.clone();
        let probe_client = probe_client.clone();
        let pool_client = pool_client.clone();
        let sessions = Arc::clone(&sessions);
        let endpoints = endpoints.clone();
        let cfg = cfg.clone();
        let mut rx = shutdown_rx.clone();

        tokio::spawn(async move {
            let mut last_emergency_refresh: Option<Instant> = None;
            run_probe_wave(&pool, &probe_client, &pool_client, &sessions, &endpoints).await;

            loop {
                tokio::select! {
                    changed = rx.changed() => {
                        if changed.is_err() || *rx.borrow() {
                            break;
                        }
                    }
                    _ = pool.needs_probe.notified() => {}
                    _ = tokio::time::sleep(Duration::from_secs(10)) => {}
                }

                if *rx.borrow() {
                    break;
                }

                pool.sweep_expired_cooldowns();

                if pool.healthy_count() < cfg.health_check.target_healthy {
                    if pool.untested_count() == 0
                        && pool.healthy_count() == 0
                        && !cfg.sources.is_empty()
                        && last_emergency_refresh
                            .is_none_or(|t| t.elapsed() >= Duration::from_secs(60))
                    {
                        last_emergency_refresh = Some(Instant::now());
                        if let Ok(loaded) = fetch_all_proxies(&base_client, &cfg).await {
                            let rep = pool.replace_proxies(loaded.proxies);
                            debug!(?rep, "emergency proxy list refresh finished");
                        }
                    }
                    if pool.untested_count() == 0 && pool.healthy_count() == 0 {
                        pool.recycle_old_dead(Duration::from_secs(600));
                    }
                    if pool.untested_count() > 0 {
                        run_probe_wave(
                            &pool,
                            &probe_client,
                            &pool_client,
                            &sessions,
                            &endpoints,
                        )
                        .await;
                    }
                }
            }
        })
    };

    // 2. Hot-standby keep-alive warmer + periodic Stage 1 liveness re-checks
    let warmer_handle = {
        let pool = Arc::clone(&pool);
        let probe_client = probe_client.clone();
        let pool_client = pool_client.clone();
        let endpoints = endpoints.clone();
        let cfg = cfg.clone();
        let mut rx = shutdown_rx.clone();

        tokio::spawn(async move {
            let keepalive_dur = Duration::from_secs(cfg.health_check.keepalive_interval_secs);
            let liveness_dur = Duration::from_secs(cfg.health_check.interval_secs);
            let mut keepalive_tick = tokio::time::interval(keepalive_dur);
            let mut liveness_tick = tokio::time::interval(liveness_dur);
            keepalive_tick.tick().await;
            liveness_tick.tick().await;

            loop {
                tokio::select! {
                    changed = rx.changed() => {
                        if changed.is_err() || *rx.borrow() {
                            break;
                        }
                    }
                    _ = keepalive_tick.tick() => {
                        let standbys = pool.top_standbys_for_keepalive(
                            cfg.health_check.hot_standby_count,
                            keepalive_dur,
                        );
                        probe_stage1_batch(&pool, &pool_client, standbys, &endpoints.models_url).await;
                    }
                    _ = liveness_tick.tick() => {
                        let stale = pool.stale_healthy_for_recheck(
                            liveness_dur,
                            cfg.health_check.concurrency,
                        );
                        probe_stage1_batch(&pool, &probe_client, stale, &endpoints.models_url).await;
                    }
                }
            }
        })
    };

    // 3. Periodic source list refresher (replaces old list in memory)
    let refresher_handle = {
        let pool = Arc::clone(&pool);
        let base_client = base_client.clone();
        let cfg = cfg.clone();

        tokio::spawn(async move {
            let interval = Duration::from_secs(cfg.refresh_interval_secs);
            let mut tick = tokio::time::interval(interval);
            tick.tick().await;

            loop {
                tokio::select! {
                    changed = shutdown_rx.changed() => {
                        if changed.is_err() || *shutdown_rx.borrow() {
                            break;
                        }
                    }
                    _ = tick.tick() => {
                        match fetch_all_proxies(&base_client, &cfg).await {
                            Ok(loaded) => {
                                let rep = pool.replace_proxies(loaded.proxies);
                                if rep.aborted_empty {
                                    warn!("proxy pool refresh yielded 0 usable proxies; keeping previous pool");
                                } else {
                                    info!(
                                        total = rep.total,
                                        kept = rep.kept,
                                        added = rep.added,
                                        removed = rep.removed,
                                        "refreshed proxy pool list"
                                    );
                                }
                            }
                            Err(err) => {
                                warn!(error = %err, "periodic proxy pool refresh failed");
                            }
                        }
                    }
                }
            }
        })
    };

    vec![prober_handle, warmer_handle, refresher_handle]
}

async fn probe_stage1_batch(
    pool: &Arc<ProxyPool>,
    client: &Client,
    proxies: Vec<ProxyEndpoint>,
    models_url: &str,
) {
    if proxies.is_empty() {
        return;
    }
    let bridge = pool.bridge().cloned();
    let sem = Arc::new(Semaphore::new(pool.config().health_check.concurrency.max(1)));
    let mut handles = Vec::with_capacity(proxies.len());

    for ep in proxies {
        let permit = sem.clone().acquire_owned().await.expect("semaphore open");
        let pool_c = Arc::clone(pool);
        let cli = client.clone();
        let br = bridge.clone();
        let url = models_url.to_string();

        handles.push(tokio::spawn(async move {
            let _permit = permit;
            match probe_stage1(&cli, &ep, br.as_ref(), &url).await {
                Stage1Result::Ok { rtt_ms, .. } => pool_c.report_liveness_ok(ep.id, rtt_ms),
                Stage1Result::Hop1Down => pool_c.release_probe(ep.id),
                Stage1Result::RateLimited {
                    retry_after,
                    detail,
                } => pool_c.report_rate_limited(ep.id, retry_after, detail),
                Stage1Result::Overloaded => pool_c.report_overloaded(ep.id),
                Stage1Result::Forbidden(detail) => pool_c.report_forbidden(ep.id, detail),
                Stage1Result::Dead(detail) => pool_c.report_dead(ep.id, detail),
            }
        }));
    }

    for h in handles {
        let _ = h.await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProxyPoolConfig;
    use crate::opencode::tls_client::{build_pool_client, build_probe_client};
    use crate::proxy::endpoint::{ParsedProxy, ProxyId};
    use axum::Router;
    use axum::http::header::CONTENT_TYPE;
    use axum::routing::{get, post};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::{TcpListener, TcpStream};

    async fn spawn_mock_http_proxy(mode: &'static str) -> std::net::SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((client, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut reader = BufReader::new(client);
                    let mut req_line = String::new();
                    if reader.read_line(&mut req_line).await.is_err() {
                        return;
                    }
                    let mut parts = req_line.split_whitespace();
                    let method = parts.next().unwrap_or("");
                    let target = parts.next().unwrap_or("").to_string();
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).await.is_err()
                            || line == "\r\n"
                            || line.is_empty()
                        {
                            break;
                        }
                    }
                    let mut client = reader.into_inner();
                    if mode == "reject_407" {
                        let _ = client
                            .write_all(b"HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\n\r\n")
                            .await;
                        return;
                    }
                    if method.eq_ignore_ascii_case("CONNECT") {
                        if let Ok(mut remote) = TcpStream::connect(&target).await {
                            let _ = client
                                .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                                .await;
                            let _ = tokio::io::copy_bidirectional(&mut client, &mut remote).await;
                        }
                    } else if let Ok(parsed_url) = url::Url::parse(&target) {
                        let host = parsed_url.host_str().unwrap_or("127.0.0.1");
                        let Some(port) = parsed_url.port_or_known_default() else {
                            return;
                        };
                        if let Ok(mut remote) = TcpStream::connect((host, port)).await {
                            let path = parsed_url.path();
                            let req = format!(
                                "{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
                            );
                            let _ = remote.write_all(req.as_bytes()).await;
                            let mut body = Vec::new();
                            let _ = remote.read_to_end(&mut body).await;
                            let _ = client.write_all(&body).await;
                        }
                    }
                });
            }
        });
        addr
    }

    async fn spawn_mock_opencode_pair() -> (String, String, Arc<AtomicUsize>) {
        let stage2_calls = Arc::new(AtomicUsize::new(0));
        let calls_ref = Arc::clone(&stage2_calls);

        let app = Router::new()
            .route(
                "/zen/v1/models",
                get(|| async {
                    r#"{"object":"list","data":[{"id":"big-pickle","type":"chat","cost":{"text":{"input":{"price":0},"output":{"price":0}}}}]}"#
                }),
            )
            .route(
                "/inference/v1/chat/completions",
                post(move || {
                    let count = calls_ref.fetch_add(1, Ordering::Relaxed) + 1;
                    async move {
                        if count == 1 {
                            (
                                StatusCode::OK,
                                [(CONTENT_TYPE, "text/event-stream")],
                                "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
                            )
                                .into_response_tuple()
                        } else {
                            (
                                StatusCode::TOO_MANY_REQUESTS,
                                [(CONTENT_TYPE, "application/json")],
                                r#"{"error":{"type":"FreeUsageLimitError"}}"#,
                            )
                                .into_response_tuple()
                        }
                    }
                }),
            );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (
            format!("http://{addr}/zen/v1/models"),
            format!("http://{addr}/inference/v1/chat/completions"),
            stage2_calls,
        )
    }

    trait TupleResp {
        fn into_response_tuple(self) -> axum::response::Response;
    }

    impl TupleResp for (StatusCode, [(axum::http::HeaderName, &'static str); 1], &'static str) {
        fn into_response_tuple(self) -> axum::response::Response {
            use axum::response::IntoResponse;
            (self.0, self.1, self.2).into_response()
        }
    }

    #[tokio::test]
    async fn two_stage_probe_promotes_healthy_cools_rate_limited_and_kills_407() {
        let (models_url, chat_url, _calls) = spawn_mock_opencode_pair().await;
        let listener_ok1 = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr_ok1 = listener_ok1.local_addr().unwrap();
        let listener_ok2 = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr_ok2 = listener_ok2.local_addr().unwrap();
        for l in [listener_ok1, listener_ok2] {
            tokio::spawn(async move {
                while let Ok((mut client, _)) = l.accept().await {
                    tokio::spawn(async move {
                        let mut hdr = [0u8; 2];
                        if client.read_exact(&mut hdr).await.is_err() {
                            return;
                        }
                        let mut methods = vec![0u8; hdr[1] as usize];
                        let _ = client.read_exact(&mut methods).await;
                        let _ = client.write_all(&[0x05, 0x00]).await;
                        let mut req = [0u8; 4];
                        if client.read_exact(&mut req).await.is_err() {
                            return;
                        }
                        let host = match req[3] {
                            0x01 => {
                                let mut ip = [0u8; 4];
                                let _ = client.read_exact(&mut ip).await;
                                std::net::IpAddr::V4(std::net::Ipv4Addr::from(ip)).to_string()
                            }
                            0x03 => {
                                let len = client.read_u8().await.unwrap() as usize;
                                let mut d = vec![0u8; len];
                                let _ = client.read_exact(&mut d).await;
                                String::from_utf8(d).unwrap()
                            }
                            _ => return,
                        };
                        let mut pb = [0u8; 2];
                        let _ = client.read_exact(&mut pb).await;
                        let port = u16::from_be_bytes(pb);
                        if let Ok(mut remote) = TcpStream::connect((host.as_str(), port)).await {
                            let _ = client
                                .write_all(&[0x05, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0])
                                .await;
                            let _ = tokio::io::copy_bidirectional(&mut client, &mut remote).await;
                        }
                    });
                }
            });
        }
        let addr_407 = spawn_mock_http_proxy("reject_407").await;

        let mut cfg = ProxyPoolConfig {
            enabled: true,
            ..Default::default()
        };
        cfg.health_check.concurrency = 1; // deterministic sequential order: ok1, ok2, 407
        cfg.health_check.target_healthy = 5;

        let ep2 = ParsedProxy::parse_explicit(&format!("socks5h://{addr_ok2}"), true).unwrap();
        let pool = ProxyPool::new(
            cfg,
            vec![
                ParsedProxy::parse_explicit(&format!("socks5h://{addr_ok1}"), true).unwrap(),
                ep2.clone(),
                ParsedProxy::parse_explicit(&format!("http://{addr_407}"), true).unwrap(),
            ],
            None,
        );
        let probe_client = build_probe_client(5).unwrap();
        let pool_client = build_pool_client(4, 30).unwrap();
        let sessions = Arc::new(SessionManager::new(Duration::from_secs(7200)));
        let endpoints = ProbeEndpoints::for_tests(models_url.clone(), chat_url);

        run_probe_wave(&pool, &probe_client, &pool_client, &sessions, &endpoints).await;

        let stats = pool.stats();
        assert_eq!(stats.healthy, 1);
        assert_eq!(stats.cooldown, 1);
        assert_eq!(stats.dead, 1);

        // Running Stage 1 liveness check must NEVER clear Proxy 2's RateLimited cooldown.
        let p2 = ep2.into_endpoint(ProxyId(2));
        probe_stage1_batch(&pool, &probe_client, vec![p2], &models_url).await;
        assert_eq!(pool.stats().cooldown, 1);
        assert_eq!(pool.stats().healthy, 1);
    }
}
