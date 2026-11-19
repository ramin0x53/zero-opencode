use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use std::task::{Context as TaskContext, Poll};
use std::time::{Duration, Instant};

use anyhow::anyhow;
use bytes::Bytes;
use futures_util::stream::FuturesUnordered;
use futures_util::{Stream, StreamExt};
use http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, USER_AGENT};
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use regex::Regex;
use serde_json::Value;
use wreq::Client;

use crate::config::RetryConfig;
use crate::opencode::session::{OPENCODE_SESSION_HEADER, ResolvedSession, SessionManager};
use crate::opencode::tls_client::resolve_wreq_proxy;
use crate::proxy::chain::ChainBridge;
use crate::proxy::endpoint::{ProxyEndpoint, ProxyId};
use crate::proxy::pool::ProxyPool;
use crate::stream::sse_parser::{SseLineParser, extract_sse_data};

pub const OPENCODE_CHAT_URL: &str = "https://opencode.ai/inference/v1/chat/completions";
pub const OPENCODE_UA: &str = "opencode/latest/2.0.26/cli";
pub const OPENCODE_CLIENT: &str = "cli";

const MIN_RETRY_AFTER_SECS: u64 = 30;
const MAX_RETRY_AFTER_SECS: u64 = 3600;

static VALID_OPENCODE_UA_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^opencode/[a-z0-9-]+/(\d+)\.\d+\.\d+/[a-z0-9-]+$").expect("valid regex")
});
static OVERLOAD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)overloaded").expect("valid regex"));

/// Typed error returned when pooled failover exhausts its deadline without finding
/// a working proxy (mapped to HTTP 503 Service Unavailable in server handlers).
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct PoolExhaustedError {
    pub message: String,
}

/// Streaming or buffered response body from upstream.
pub enum UpstreamBody {
    Stream(Pin<Box<dyn Stream<Item = anyhow::Result<Bytes>> + Send>>),
    Buffered(Bytes),
}

/// Response returned by an upstream transport attempt.
pub struct UpstreamResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: UpstreamBody,
    /// Redacted proxy URL (`scheme://host:port`) that served this response, if pooled.
    pub used_proxy: Option<String>,
}

impl UpstreamResponse {
    /// Collect the entire body into `Bytes` (used by non-2xx or buffered paths).
    pub async fn into_bytes(self) -> anyhow::Result<(StatusCode, HeaderMap, Bytes)> {
        let bytes = match self.body {
            UpstreamBody::Buffered(b) => b,
            UpstreamBody::Stream(mut stream) => {
                let mut acc = Vec::new();
                while let Some(chunk) = stream.next().await {
                    acc.extend_from_slice(&chunk?);
                }
                Bytes::from(acc)
            }
        };
        Ok((self.status, self.headers, bytes))
    }
}

/// Stream wrapper that replays already-peeked head bytes before reading the rest
/// of the underlying upstream stream.
struct PrependStream {
    head: std::collections::VecDeque<Bytes>,
    tail: Pin<Box<dyn Stream<Item = anyhow::Result<Bytes>> + Send>>,
}

impl PrependStream {
    fn new(
        head: Vec<Bytes>,
        tail: Pin<Box<dyn Stream<Item = anyhow::Result<Bytes>> + Send>>,
    ) -> Self {
        Self {
            head: head.into_iter().filter(|b| !b.is_empty()).collect(),
            tail,
        }
    }
}

impl Stream for PrependStream {
    type Item = anyhow::Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        if let Some(chunk) = self.head.pop_front() {
            return Poll::Ready(Some(Ok(chunk)));
        }
        self.tail.as_mut().poll_next(cx)
    }
}

/// Abstraction over the HTTP client so retry & session-eviction logic can be unit tested offline.
pub trait UpstreamTransport: Send + Sync {
    fn post<'a>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
        body: Bytes,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<UpstreamResponse>> + Send + 'a>>;
}

/// Proxy-aware upstream transport used by pooled execution and unit tests.
pub trait RoutedTransport: Send + Sync {
    fn post_via<'a>(
        &'a self,
        proxy: Option<&'a ProxyEndpoint>,
        url: &'a str,
        headers: HeaderMap,
        body: Bytes,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<UpstreamResponse>> + Send + 'a>>;

    fn is_hop1_failing(&self) -> bool {
        false
    }
}

/// Production `wreq` BoringSSL transport (single client, non-pooled).
pub struct WreqTransport {
    client: Client,
}

impl WreqTransport {
    pub fn new(client: Client) -> Self {
        Self { client }
    }
}

impl UpstreamTransport for WreqTransport {
    fn post<'a>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
        body: Bytes,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<UpstreamResponse>> + Send + 'a>> {
        Box::pin(send_via_client(&self.client, None, url, headers, body, None))
    }
}

/// Production `wreq` BoringSSL transport routing each attempt through either a selected
/// pool proxy (chained via `ChainBridge` when configured) or `base_client`.
pub struct PooledWreqTransport {
    pool_client: Client,
    base_client: Client,
    bridge: Option<Arc<ChainBridge>>,
}

impl PooledWreqTransport {
    pub fn new(
        pool_client: Client,
        base_client: Client,
        bridge: Option<Arc<ChainBridge>>,
    ) -> Self {
        Self {
            pool_client,
            base_client,
            bridge,
        }
    }
}

impl RoutedTransport for PooledWreqTransport {
    fn post_via<'a>(
        &'a self,
        proxy: Option<&'a ProxyEndpoint>,
        url: &'a str,
        headers: HeaderMap,
        body: Bytes,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<UpstreamResponse>> + Send + 'a>> {
        Box::pin(async move {
            match proxy {
                Some(ep) => {
                    let wreq_proxy = resolve_wreq_proxy(ep, self.bridge.as_ref())?;
                    send_via_client(
                        &self.pool_client,
                        Some(wreq_proxy),
                        url,
                        headers,
                        body,
                        Some(ep.redacted()),
                    )
                    .await
                }
                None => {
                    send_via_client(&self.base_client, None, url, headers, body, None).await
                }
            }
        })
    }

    fn is_hop1_failing(&self) -> bool {
        self.bridge
            .as_ref()
            .is_some_and(|b| b.is_hop1_failing())
    }
}

async fn send_via_client(
    client: &Client,
    proxy: Option<wreq::Proxy>,
    url: &str,
    headers: HeaderMap,
    body: Bytes,
    used_proxy: Option<String>,
) -> anyhow::Result<UpstreamResponse> {
    let mut req = client.post(url).headers(headers).body(body);
    if let Some(p) = proxy {
        req = req.proxy(p);
    }
    let res = req.send().await?;
    let status = res.status();
    let resp_headers = res.headers().clone();
    if !status.is_success() {
        let bytes = res.bytes().await?;
        return Ok(UpstreamResponse {
            status,
            headers: resp_headers,
            body: UpstreamBody::Buffered(bytes),
            used_proxy,
        });
    }
    let stream = res
        .bytes_stream()
        .map(|chunk| chunk.map_err(anyhow::Error::from));
    Ok(UpstreamResponse {
        status,
        headers: resp_headers,
        body: UpstreamBody::Stream(Box::pin(stream)),
        used_proxy,
    })
}

/// Returns `true` when text indicates OpenCode's transient load-shedding `429 overloaded`.
pub fn is_overload_text(text: &str) -> bool {
    OVERLOAD_RE.is_match(text)
}

/// Returns `true` when headers or leading body bytes look like HTML (captive portal / WAF block).
pub fn is_html_response(headers: &HeaderMap, body_prefix: &[u8]) -> bool {
    if let Some(ct) = headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok())
        && ct.to_ascii_lowercase().contains("text/html")
    {
        return true;
    }
    let trimmed = String::from_utf8_lossy(body_prefix);
    let lower = trimmed.trim_start().to_ascii_lowercase();
    lower.starts_with("<!doctype") || lower.starts_with("<html")
}

/// Parse `Retry-After` or `x-ratelimit-reset-after` headers into a bounded cooldown duration.
pub fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let raw = headers
        .get("retry-after")
        .or_else(|| headers.get("x-ratelimit-reset-after"))
        .and_then(|v| v.to_str().ok())?
        .trim();
    if raw.is_empty() {
        return None;
    }
    let num = raw.parse::<f64>().ok()?;
    if !num.is_finite() || num <= 0.0 {
        return None;
    }
    // Large values (> 1 day in seconds) are assumed to be milliseconds.
    let secs = if num > 86_400.0 {
        (num / 1000.0).ceil() as u64
    } else {
        num.ceil() as u64
    };
    Some(Duration::from_secs(
        secs.clamp(MIN_RETRY_AFTER_SECS, MAX_RETRY_AFTER_SECS),
    ))
}

/// Returns `true` when `ua` matches OpenCode v2+ (`opencode/<channel>/<major>=2.../client`).
pub fn has_valid_opencode_version(ua: &str) -> bool {
    let Some(caps) = VALID_OPENCODE_UA_RE.captures(ua.trim()) else {
        return false;
    };
    caps.get(1)
        .and_then(|m| m.as_str().parse::<u32>().ok())
        .is_some_and(|major| major >= 2)
}

/// Build the OpenCode v2 upstream headers for a given session ID.
/// Never forwards `x-opencode-request` (v1 header rejected with 403 upstream).
pub fn build_upstream_headers(incoming: &HeaderMap, session_id: &str) -> HeaderMap {
    let mut out = HeaderMap::new();
    out.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    out.insert(AUTHORIZATION, HeaderValue::from_static("Bearer public"));

    let downstream_ua = incoming
        .get(USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let ua_val = if has_valid_opencode_version(downstream_ua) {
        HeaderValue::from_str(downstream_ua.trim())
            .unwrap_or_else(|_| HeaderValue::from_static(OPENCODE_UA))
    } else {
        HeaderValue::from_static(OPENCODE_UA)
    };
    out.insert(USER_AGENT, ua_val);

    let client_header = incoming
        .get("x-opencode-client")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| HeaderValue::from_str(s).ok())
        .unwrap_or_else(|| HeaderValue::from_static(OPENCODE_CLIENT));
    out.insert(HeaderName::from_static("x-opencode-client"), client_header);

    let project_header = incoming
        .get("x-opencode-project")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| HeaderValue::from_str(s).ok())
        .unwrap_or_else(|| HeaderValue::from_static("global"));
    out.insert(
        HeaderName::from_static("x-opencode-project"),
        project_header,
    );

    if let Ok(session_val) = HeaderValue::from_str(session_id) {
        for name in [
            OPENCODE_SESSION_HEADER,
            "x-opencode-session-id",
            "x-session-affinity",
            "x-session-id",
        ] {
            out.insert(HeaderName::from_static(name), session_val.clone());
        }
    }

    out.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
    out
}

enum AttemptOutcome {
    Success(UpstreamResponse),
    ClientError(UpstreamResponse),
    RateLimited {
        retry_after: Option<Duration>,
        detail: String,
    },
    Forbidden(String),
    Overloaded(String),
    Dead(String),
    Hijacked(String),
    TransientError(String),
}

struct AttemptReport {
    proxy: ProxyEndpoint,
    session: ResolvedSession,
    elapsed_ms: u64,
    outcome: AttemptOutcome,
    fallback_response: Option<UpstreamResponse>,
}

/// Executes requests against OpenCode Free with session management, overload retries,
/// and optional rotating proxy-pool failover with speculative hedging.
#[derive(Clone)]
pub struct OpenCodeExecutor {
    transport: Arc<dyn UpstreamTransport>,
    routed: Option<Arc<dyn RoutedTransport>>,
    pool: Option<Arc<ProxyPool>>,
    sessions: Arc<SessionManager>,
    retry: RetryConfig,
    chat_url: String,
}

impl OpenCodeExecutor {
    pub fn new(client: Client, sessions: Arc<SessionManager>, retry: RetryConfig) -> Self {
        Self::with_transport(
            Arc::new(WreqTransport::new(client)),
            sessions,
            retry,
            OPENCODE_CHAT_URL,
        )
    }

    pub fn with_transport(
        transport: Arc<dyn UpstreamTransport>,
        sessions: Arc<SessionManager>,
        retry: RetryConfig,
        chat_url: impl Into<String>,
    ) -> Self {
        Self {
            transport,
            routed: None,
            pool: None,
            sessions,
            retry,
            chat_url: chat_url.into(),
        }
    }

    /// Construct an executor wired to a `ProxyPool` and `PooledWreqTransport`.
    pub fn new_pooled(
        base_client: Client,
        pool_client: Client,
        pool: Arc<ProxyPool>,
        sessions: Arc<SessionManager>,
        retry: RetryConfig,
    ) -> Self {
        let bridge = pool.bridge().cloned();
        let routed: Arc<dyn RoutedTransport> =
            Arc::new(PooledWreqTransport::new(pool_client, base_client.clone(), bridge));
        Self::with_pooled_transport(
            Arc::new(WreqTransport::new(base_client)),
            routed,
            pool,
            sessions,
            retry,
            OPENCODE_CHAT_URL,
        )
    }

    pub fn with_pooled_transport(
        base_transport: Arc<dyn UpstreamTransport>,
        routed: Arc<dyn RoutedTransport>,
        pool: Arc<ProxyPool>,
        sessions: Arc<SessionManager>,
        retry: RetryConfig,
        chat_url: impl Into<String>,
    ) -> Self {
        Self {
            transport: base_transport,
            routed: Some(routed),
            pool: Some(pool),
            sessions,
            retry,
            chat_url: chat_url.into(),
        }
    }

    pub fn sessions(&self) -> &SessionManager {
        &self.sessions
    }

    pub fn proxy_pool(&self) -> Option<&Arc<ProxyPool>> {
        self.pool.as_ref()
    }

    /// Execute an upstream OpenCode chat completion request.
    /// When `proxy_pool` is configured, routes through healthy proxies with first-frame
    /// validation, retry-until-success switching, and speculative hedging.
    pub async fn execute_chat(
        &self,
        incoming_headers: &HeaderMap,
        body_bytes: Bytes,
        session: &ResolvedSession,
    ) -> anyhow::Result<UpstreamResponse> {
        self.execute_chat_with_body(incoming_headers, None, body_bytes, session)
            .await
    }

    /// Execute an upstream OpenCode chat completion request, optionally passing the
    /// parsed JSON request body for per-proxy session hint extraction.
    pub async fn execute_chat_with_body(
        &self,
        incoming_headers: &HeaderMap,
        body_value: Option<&Value>,
        body_bytes: Bytes,
        fallback_session: &ResolvedSession,
    ) -> anyhow::Result<UpstreamResponse> {
        if let (Some(pool), Some(routed)) = (&self.pool, &self.routed) {
            return self
                .execute_chat_pooled(
                    pool,
                    routed,
                    incoming_headers,
                    body_value,
                    body_bytes,
                    fallback_session,
                )
                .await;
        }

        self.execute_chat_single(incoming_headers, body_bytes, fallback_session)
            .await
    }

    async fn execute_chat_single(
        &self,
        incoming_headers: &HeaderMap,
        body_bytes: Bytes,
        session: &ResolvedSession,
    ) -> anyhow::Result<UpstreamResponse> {
        let headers = build_upstream_headers(incoming_headers, &session.id);
        let max_attempts = self.retry.max_attempts.max(1);

        for attempt in 1..=max_attempts {
            let resp = self
                .transport
                .post(&self.chat_url, headers.clone(), body_bytes.clone())
                .await?;

            if resp.status == StatusCode::FORBIDDEN {
                if !session.is_native && self.sessions.forget_session(&session.id) {
                    tracing::debug!(
                        session_id = %session.id,
                        "OpenCode 403: dropped cached session so next request re-mints"
                    );
                }
                return Ok(resp);
            }

            if resp.status == StatusCode::TOO_MANY_REQUESTS {
                let (status, resp_headers, raw_body) = resp.into_bytes().await?;
                let text = String::from_utf8_lossy(&raw_body);
                if is_overload_text(&text) && attempt < max_attempts {
                    let base = self
                        .retry
                        .base_delay_ms
                        .saturating_mul(attempt as u64)
                        .min(self.retry.max_delay_ms);
                    let jitter = rand::random_range(0..250);
                    let delay = Duration::from_millis(base + jitter);
                    tracing::debug!(
                        attempt,
                        max_attempts,
                        delay_ms = delay.as_millis(),
                        "OpenCode free lane overloaded (429), retrying with same session"
                    );
                    tokio::time::sleep(delay).await;
                    continue;
                }

                if !is_overload_text(&text)
                    && !session.is_native
                    && self.sessions.forget_session(&session.id)
                {
                    tracing::debug!(
                        session_id = %session.id,
                        "OpenCode 429: dropped cached session so next request re-mints"
                    );
                }

                return Ok(UpstreamResponse {
                    status,
                    headers: resp_headers,
                    body: UpstreamBody::Buffered(raw_body),
                    used_proxy: None,
                });
            }

            return Ok(resp);
        }

        unreachable!("loop returns by attempt == max_attempts")
    }

    async fn execute_chat_pooled(
        &self,
        pool: &Arc<ProxyPool>,
        routed: &Arc<dyn RoutedTransport>,
        incoming_headers: &HeaderMap,
        body_value: Option<&Value>,
        body_bytes: Bytes,
        fallback_session: &ResolvedSession,
    ) -> anyhow::Result<UpstreamResponse> {
        let sel = &pool.config().selection;
        let first_byte_timeout = Duration::from_secs(sel.first_byte_timeout_secs);
        let hedge_delay = sel.hedge_delay_ms.map(Duration::from_millis);
        let deadline = sel
            .failover_timeout_secs
            .map(|secs| Instant::now() + Duration::from_secs(secs));

        let empty_json = Value::Null;
        let session_body = body_value.unwrap_or(&empty_json);

        let mut excluded: HashSet<ProxyId> = HashSet::new();
        let mut overload_retried: HashSet<ProxyId> = HashSet::new();
        let mut in_flight: FuturesUnordered<Pin<Box<dyn Future<Output = AttemptReport> + Send + '_>>> =
            FuturesUnordered::new();
        let mut last_launch_at: Option<Instant> = None;
        let mut primary_of_hedge: Option<ProxyId> = None;
        let mut last_fallback_response: Option<UpstreamResponse> = None;
        let mut last_error: Option<String> = None;

        loop {
            if let Some(dl) = deadline
                && Instant::now() >= dl
            {
                break;
            }

            // Ensure at least one attempt is running when possible.
            if in_flight.is_empty() {
                if let Some(proxy) = pool.select_proxy(&excluded) {
                    excluded.insert(proxy.id);
                    primary_of_hedge = Some(proxy.id);
                    last_launch_at = Some(Instant::now());
                    let session = if body_value.is_some() {
                        self.sessions.resolve_session_for_proxy(
                            incoming_headers,
                            session_body,
                            Some(proxy.id),
                        )
                    } else if fallback_session.is_native {
                        fallback_session.clone()
                    } else {
                        self.sessions.resolve_session_for_proxy(
                            incoming_headers,
                            session_body,
                            Some(proxy.id),
                        )
                    };
                    let headers = build_upstream_headers(incoming_headers, &session.id);
                    in_flight.push(Box::pin(attempt_single_proxy(
                        Arc::clone(routed),
                        proxy,
                        session,
                        self.chat_url.clone(),
                        headers,
                        body_bytes.clone(),
                        first_byte_timeout,
                    )));
                } else {
                    // No currently healthy proxy outside excluded: wake prober and wait cleanly.
                    pool.trigger_probe();
                    let wait_dur = match deadline {
                        Some(dl) => {
                            let rem = dl.saturating_duration_since(Instant::now());
                            if rem.is_zero() {
                                break;
                            }
                            rem.min(Duration::from_secs(1))
                        }
                        None => Duration::from_secs(1),
                    };
                    pool.wait_for_unexcluded_healthy(&excluded, wait_dur).await;
                    continue;
                }
            }

            // Compute Option A speculative hedge sleep (fire 2nd standby after hedge_delay_ms
            // if only 1 request is in flight and haven't stalled indefinitely).
            let hedge_fut: Pin<Box<dyn Future<Output = ()> + Send>> =
                if let (Some(hd), Some(launched)) = (hedge_delay, last_launch_at)
                    && in_flight.len() < 2
                {
                    let target_instant = launched + hd;
                    let bounded = match deadline {
                        Some(dl) if target_instant > dl => dl,
                        _ => target_instant,
                    };
                    Box::pin(tokio::time::sleep_until(tokio::time::Instant::from_std(
                        bounded,
                    )))
                } else {
                    Box::pin(std::future::pending::<()>())
                };

            let deadline_fut: Pin<Box<dyn Future<Output = ()> + Send>> = match deadline {
                Some(dl) => Box::pin(tokio::time::sleep_until(tokio::time::Instant::from_std(dl))),
                None => Box::pin(std::future::pending::<()>()),
            };

            tokio::select! {
                Some(report) = in_flight.next() => {
                    let pid = report.proxy.id;
                    let was_concurrent = !in_flight.is_empty();
                    if let Some(fb) = report.fallback_response {
                        last_fallback_response = Some(fb);
                    }
                    match report.outcome {
                        AttemptOutcome::Success(resp) => {
                            pool.report_success(pid, report.elapsed_ms);
                            if was_concurrent && primary_of_hedge.is_some_and(|p| p != pid) {
                                pool.record_hedged_win();
                            }
                            pool.promote_active(pid);
                            return Ok(resp);
                        }
                        AttemptOutcome::ClientError(resp) => {
                            // Request-scoped 4xx: do not penalize healthy proxy.
                            return Ok(resp);
                        }
                        AttemptOutcome::Overloaded(detail) => {
                            last_error = Some(detail);
                            if in_flight.is_empty() && overload_retried.insert(pid) {
                                // One fast overload retry on the same proxy before rotating.
                                let jitter = Duration::from_millis(350 + rand::random_range(0..250));
                                let can_sleep = deadline.is_none_or(|dl| Instant::now() + jitter < dl);
                                if can_sleep {
                                    tokio::time::sleep(jitter).await;
                                    last_launch_at = Some(Instant::now());
                                    let headers = build_upstream_headers(incoming_headers, &report.session.id);
                                    in_flight.push(Box::pin(attempt_single_proxy(
                                        Arc::clone(routed),
                                        report.proxy,
                                        report.session,
                                        self.chat_url.clone(),
                                        headers,
                                        body_bytes.clone(),
                                        first_byte_timeout,
                                    )));
                                    continue;
                                }
                            }
                            pool.report_overloaded(pid);
                        }
                        AttemptOutcome::RateLimited { retry_after, detail } => {
                            tracing::debug!(proxy = %report.proxy.redacted(), %detail, "proxy hit rate limit; switching");
                            last_error = Some(detail.clone());
                            if !report.session.is_native {
                                self.sessions.forget_session(&report.session.id);
                            }
                            pool.report_rate_limited(pid, retry_after, detail);
                        }
                        AttemptOutcome::Forbidden(detail) => {
                            tracing::debug!(proxy = %report.proxy.redacted(), %detail, "proxy hit 403; switching");
                            last_error = Some(detail.clone());
                            if !report.session.is_native {
                                self.sessions.forget_session(&report.session.id);
                            }
                            pool.report_forbidden(pid, detail);
                        }
                        AttemptOutcome::Dead(detail) | AttemptOutcome::Hijacked(detail) => {
                            tracing::debug!(proxy = %report.proxy.redacted(), %detail, "proxy dead/hijacked; switching");
                            last_error = Some(detail.clone());
                            pool.report_dead(pid, detail);
                        }
                        AttemptOutcome::TransientError(detail) => {
                            tracing::debug!(proxy = %report.proxy.redacted(), %detail, "proxy transport/5xx error; switching");
                            last_error = Some(detail.clone());
                            if routed.is_hop1_failing() {
                                pool.release_probe(pid);
                            } else {
                                pool.report_transport_error(pid, detail);
                            }
                        }
                    }
                }
                _ = hedge_fut => {
                    if let Some(standby) = pool.select_proxy(&excluded) {
                        tracing::debug!(
                            standby = %standby.redacted(),
                            "launching speculative hedged proxy attempt"
                        );
                        excluded.insert(standby.id);
                        last_launch_at = Some(Instant::now());
                        let session = self.sessions.resolve_session_for_proxy(
                            incoming_headers,
                            session_body,
                            Some(standby.id),
                        );
                        let headers = build_upstream_headers(incoming_headers, &session.id);
                        in_flight.push(Box::pin(attempt_single_proxy(
                            Arc::clone(routed),
                            standby,
                            session,
                            self.chat_url.clone(),
                            headers,
                            body_bytes.clone(),
                            first_byte_timeout,
                        )));
                    } else {
                        // Avoid spinning tight if no standby is currently ready
                        last_launch_at = None;
                        pool.trigger_probe();
                    }
                }
                _ = deadline_fut => {
                    break;
                }
            }
        }

        // Exhausted deadline without finding a working proxy.
        if sel.fallback_to_base {
            tracing::warn!("proxy pool failover exhausted; falling back to base client");
            let session = self.sessions.resolve_session(incoming_headers, session_body);
            let headers = build_upstream_headers(incoming_headers, &session.id);
            return routed
                .post_via(None, &self.chat_url, headers, body_bytes)
                .await;
        }

        if let Some(fb) = last_fallback_response {
            // Preserve concrete upstream limit payload only if explicitly helpful in logs,
            // while surfacing a clear 503 PoolExhaustedError with pool state.
            drop(fb);
        }

        let stats = pool.stats();
        let reset_hint = pool
            .earliest_cooldown_remaining()
            .map(|d| format!("; earliest cooldown reset in {}s", d.as_secs()))
            .unwrap_or_default();
        let err_hint = last_error
            .map(|e| format!(" (last error: {e})"))
            .unwrap_or_default();

        Err(anyhow!(PoolExhaustedError {
            message: format!(
                "no healthy proxy succeeded before failover timeout (total={}, healthy={}, cooldown={}, dead={}, hop1_healthy={}{}{})",
                stats.total,
                stats.healthy,
                stats.cooldown,
                stats.dead,
                stats.hop1_healthy,
                reset_hint,
                err_hint,
            ),
        }))
    }
}

async fn attempt_single_proxy(
    routed: Arc<dyn RoutedTransport>,
    proxy: ProxyEndpoint,
    session: ResolvedSession,
    chat_url: String,
    headers: HeaderMap,
    body_bytes: Bytes,
    first_byte_timeout: Duration,
) -> AttemptReport {
    let started = Instant::now();
    let resp = match routed.post_via(Some(&proxy), &chat_url, headers, body_bytes).await {
        Ok(r) => r,
        Err(err) => {
            return AttemptReport {
                proxy,
                session,
                elapsed_ms: started.elapsed().as_millis() as u64,
                outcome: AttemptOutcome::TransientError(err.to_string()),
                fallback_response: None,
            };
        }
    };

    let status = resp.status;
    if !status.is_success() {
        let resp_headers = resp.headers.clone();
        let (_, _, raw_err) = match resp.into_bytes().await {
            Ok(parts) => parts,
            Err(err) => {
                return AttemptReport {
                    proxy,
                    session,
                    elapsed_ms: started.elapsed().as_millis() as u64,
                    outcome: AttemptOutcome::TransientError(err.to_string()),
                    fallback_response: None,
                };
            }
        };

        let text = String::from_utf8_lossy(&raw_err).into_owned();
        let detail = format!("HTTP {}: {}", status.as_u16(), truncate_msg(&text, 160));
        let outcome = if status == StatusCode::PROXY_AUTHENTICATION_REQUIRED {
            AttemptOutcome::Dead(detail)
        } else if is_html_response(&resp_headers, &raw_err) {
            AttemptOutcome::Hijacked(detail)
        } else if status == StatusCode::FORBIDDEN {
            AttemptOutcome::Forbidden(detail)
        } else if status == StatusCode::TOO_MANY_REQUESTS {
            if is_overload_text(&text) {
                AttemptOutcome::Overloaded(detail)
            } else {
                AttemptOutcome::RateLimited {
                    retry_after: parse_retry_after(&resp_headers),
                    detail,
                }
            }
        } else if matches!(
            status,
            StatusCode::BAD_REQUEST
                | StatusCode::UNAUTHORIZED
                | StatusCode::NOT_FOUND
                | StatusCode::UNPROCESSABLE_ENTITY
        ) && serde_json::from_slice::<Value>(&raw_err).is_ok()
        {
            return AttemptReport {
                elapsed_ms: started.elapsed().as_millis() as u64,
                outcome: AttemptOutcome::ClientError(UpstreamResponse {
                    status,
                    headers: resp_headers,
                    body: UpstreamBody::Buffered(raw_err),
                    used_proxy: Some(proxy.redacted()),
                }),
                proxy,
                session,
                fallback_response: None,
            };
        } else {
            AttemptOutcome::TransientError(detail)
        };

        return AttemptReport {
            elapsed_ms: started.elapsed().as_millis() as u64,
            outcome,
            fallback_response: Some(UpstreamResponse {
                status,
                headers: resp_headers,
                body: UpstreamBody::Buffered(raw_err),
                used_proxy: Some(proxy.redacted()),
            }),
            proxy,
            session,
        };
    }

    let (outcome, fallback_response) =
        peek_and_validate_stream(resp, &proxy, first_byte_timeout).await;
    AttemptReport {
        proxy,
        session,
        elapsed_ms: started.elapsed().as_millis() as u64,
        outcome,
        fallback_response,
    }
}

async fn peek_and_validate_stream(
    resp: UpstreamResponse,
    proxy: &ProxyEndpoint,
    first_byte_timeout: Duration,
) -> (AttemptOutcome, Option<UpstreamResponse>) {
    if is_html_response(&resp.headers, &[]) {
        return (
            AttemptOutcome::Hijacked("200 response had text/html Content-Type".to_string()),
            None,
        );
    }

    let status = resp.status;
    let headers = resp.headers;
    let redacted = proxy.redacted();

    match resp.body {
        UpstreamBody::Buffered(bytes) => {
            if is_html_response(&headers, &bytes) {
                return (
                    AttemptOutcome::Hijacked("200 buffered body was HTML".to_string()),
                    None,
                );
            }
            let mut parser = SseLineParser::new();
            let mut lines = parser.push_chunk(&bytes);
            if let Some(last) = parser.finish() {
                lines.push(last);
            }
            match inspect_sse_lines(&lines, &headers) {
                SsePeekDecision::Valid => (
                    AttemptOutcome::Success(UpstreamResponse {
                        status,
                        headers,
                        body: UpstreamBody::Buffered(bytes),
                        used_proxy: Some(redacted),
                    }),
                    None,
                ),
                SsePeekDecision::NeedMore => (
                    AttemptOutcome::TransientError("empty upstream SSE response".to_string()),
                    None,
                ),
                SsePeekDecision::ClientError => (
                    AttemptOutcome::ClientError(UpstreamResponse {
                        status,
                        headers,
                        body: UpstreamBody::Buffered(bytes),
                        used_proxy: Some(redacted),
                    }),
                    None,
                ),
                SsePeekDecision::Fail(outcome) => (outcome, None),
            }
        }
        UpstreamBody::Stream(mut stream) => {
            let mut parser = SseLineParser::new();
            let mut head_chunks: Vec<Bytes> = Vec::new();
            let mut prefix_sample: Vec<u8> = Vec::new();
            let deadline = tokio::time::Instant::now() + first_byte_timeout;

            loop {
                let next_res = tokio::time::timeout_at(deadline, stream.next()).await;
                match next_res {
                    Err(_) => {
                        return (
                            AttemptOutcome::TransientError(format!(
                                "timed out waiting {}s for first SSE frame",
                                first_byte_timeout.as_secs()
                            )),
                            None,
                        );
                    }
                    Ok(Some(Err(err))) => {
                        return (
                            AttemptOutcome::TransientError(format!(
                                "stream read error before first SSE frame: {err}"
                            )),
                            None,
                        );
                    }
                    Ok(Some(Ok(chunk))) => {
                        if chunk.is_empty() {
                            continue;
                        }
                        if prefix_sample.len() < 128 {
                            let need = (128 - prefix_sample.len()).min(chunk.len());
                            prefix_sample.extend_from_slice(&chunk[..need]);
                            if is_html_response(&headers, &prefix_sample) {
                                return (
                                    AttemptOutcome::Hijacked(
                                        "captive portal / HTML payload in SSE stream".to_string(),
                                    ),
                                    None,
                                );
                            }
                        }

                        head_chunks.push(chunk.clone());
                        let lines = parser.push_chunk(&chunk);
                        match inspect_sse_lines(&lines, &headers) {
                            SsePeekDecision::Valid => {
                                let combined = PrependStream::new(head_chunks, stream);
                                return (
                                    AttemptOutcome::Success(UpstreamResponse {
                                        status,
                                        headers,
                                        body: UpstreamBody::Stream(Box::pin(combined)),
                                        used_proxy: Some(redacted),
                                    }),
                                    None,
                                );
                            }
                            SsePeekDecision::NeedMore => continue,
                            SsePeekDecision::ClientError => {
                                let combined = PrependStream::new(head_chunks, stream);
                                return (
                                    AttemptOutcome::ClientError(UpstreamResponse {
                                        status,
                                        headers,
                                        body: UpstreamBody::Stream(Box::pin(combined)),
                                        used_proxy: Some(redacted),
                                    }),
                                    None,
                                );
                            }
                            SsePeekDecision::Fail(outcome) => return (outcome, None),
                        }
                    }
                    Ok(None) => {
                        let tail = parser.finish().into_iter().collect::<Vec<_>>();
                        match inspect_sse_lines(&tail, &headers) {
                            SsePeekDecision::Valid => {
                                let combined = PrependStream::new(head_chunks, stream);
                                return (
                                    AttemptOutcome::Success(UpstreamResponse {
                                        status,
                                        headers,
                                        body: UpstreamBody::Stream(Box::pin(combined)),
                                        used_proxy: Some(redacted),
                                    }),
                                    None,
                                );
                            }
                            SsePeekDecision::ClientError => {
                                let combined = PrependStream::new(head_chunks, stream);
                                return (
                                    AttemptOutcome::ClientError(UpstreamResponse {
                                        status,
                                        headers,
                                        body: UpstreamBody::Stream(Box::pin(combined)),
                                        used_proxy: Some(redacted),
                                    }),
                                    None,
                                );
                            }
                            SsePeekDecision::Fail(outcome) => return (outcome, None),
                            SsePeekDecision::NeedMore => {
                                return (
                                    AttemptOutcome::TransientError(
                                        "upstream closed stream before producing any SSE data"
                                            .to_string(),
                                    ),
                                    None,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

enum SsePeekDecision {
    Valid,
    NeedMore,
    ClientError,
    Fail(AttemptOutcome),
}

fn inspect_sse_lines(lines: &[String], headers: &HeaderMap) -> SsePeekDecision {
    for line in lines {
        let Some(payload) = extract_sse_data(line) else {
            continue;
        };
        if payload.is_empty() {
            continue;
        }
        if payload == "[DONE]" {
            return SsePeekDecision::Valid;
        }
        let lower_payload = payload.trim_start().to_ascii_lowercase();
        if lower_payload.starts_with("<html") || lower_payload.starts_with("<!doctype") {
            return SsePeekDecision::Fail(AttemptOutcome::Hijacked(
                "HTML returned inside SSE data".to_string(),
            ));
        }
        if let Ok(val) = serde_json::from_str::<Value>(payload) {
            if let Some(err_val) = val.get("error").filter(|v| !v.is_null()) {
                return classify_sse_error(err_val, headers);
            }
            return SsePeekDecision::Valid;
        }
        // Non-JSON non-HTML SSE payload
        return SsePeekDecision::Valid;
    }
    SsePeekDecision::NeedMore
}

fn classify_sse_error(err_val: &Value, headers: &HeaderMap) -> SsePeekDecision {
    let err_str = err_val.to_string();
    let lower = err_str.to_ascii_lowercase();
    let status = err_val.get("status").and_then(Value::as_u64).unwrap_or(0);

    if is_overload_text(&err_str) {
        return SsePeekDecision::Fail(AttemptOutcome::Overloaded(truncate_msg(&err_str, 160)));
    }
    if status == 429
        || lower.contains("freeusagelimiterror")
        || lower.contains("rate limit")
        || lower.contains("too many requests")
        || lower.contains("quota exceeded")
    {
        return SsePeekDecision::Fail(AttemptOutcome::RateLimited {
            retry_after: parse_retry_after(headers),
            detail: truncate_msg(&err_str, 160),
        });
    }
    if status == 403 || lower.contains("freetiererror") || lower.contains("forbidden") {
        return SsePeekDecision::Fail(AttemptOutcome::Forbidden(truncate_msg(&err_str, 160)));
    }
    if matches!(status, 400 | 401 | 404 | 422) {
        return SsePeekDecision::ClientError;
    }
    SsePeekDecision::Fail(AttemptOutcome::TransientError(truncate_msg(&err_str, 160)))
}

fn truncate_msg(s: &str, max: usize) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= max {
        trimmed.to_string()
    } else {
        let prefix: String = trimmed.chars().take(max).collect();
        format!("{prefix}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ProxyPoolConfig, ProxySelectionConfig};
    use crate::proxy::endpoint::ParsedProxy;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct MockCall {
        headers: HeaderMap,
    }

    struct MockTransport {
        responses: Mutex<Vec<(StatusCode, &'static str)>>,
        calls: Mutex<Vec<MockCall>>,
    }

    impl MockTransport {
        fn new(responses: Vec<(StatusCode, &'static str)>) -> Self {
            Self {
                responses: Mutex::new(responses),
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    impl UpstreamTransport for MockTransport {
        fn post<'a>(
            &'a self,
            _url: &'a str,
            headers: HeaderMap,
            _body: Bytes,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<UpstreamResponse>> + Send + 'a>> {
            Box::pin(async move {
                self.calls.lock().unwrap().push(MockCall { headers });
                let (status, text) = {
                    let mut guard = self.responses.lock().unwrap();
                    if guard.len() > 1 {
                        guard.remove(0)
                    } else {
                        guard[0]
                    }
                };
                Ok(UpstreamResponse {
                    status,
                    headers: HeaderMap::new(),
                    body: UpstreamBody::Buffered(Bytes::from_static(text.as_bytes())),
                    used_proxy: None,
                })
            })
        }
    }

    #[derive(Clone)]
    enum ScriptedAction {
        Respond {
            status: StatusCode,
            content_type: &'static str,
            body: &'static str,
            delay_ms: u64,
        },
        StreamChunks {
            chunks: Vec<&'static str>,
            delay_ms: u64,
        },
        TransportErr(&'static str, u64),
    }

    struct ScriptedRoutedTransport {
        scripts: Mutex<HashMap<ProxyId, Vec<ScriptedAction>>>,
        calls: Mutex<Vec<(ProxyId, String)>>,
    }

    impl ScriptedRoutedTransport {
        fn new(scripts: HashMap<ProxyId, Vec<ScriptedAction>>) -> Self {
            Self {
                scripts: Mutex::new(scripts),
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    impl RoutedTransport for ScriptedRoutedTransport {
        fn post_via<'a>(
            &'a self,
            proxy: Option<&'a ProxyEndpoint>,
            _url: &'a str,
            headers: HeaderMap,
            _body: Bytes,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<UpstreamResponse>> + Send + 'a>> {
            let pid = proxy.map(|p| p.id).unwrap_or(ProxyId(0));
            let sid = headers
                .get("x-opencode-session")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();
            Box::pin(async move {
                self.calls.lock().unwrap().push((pid, sid));
                let action = {
                    let mut guard = self.scripts.lock().unwrap();
                    let list = guard.get_mut(&pid).expect("missing script for proxy");
                    if list.len() > 1 {
                        list.remove(0)
                    } else {
                        list[0].clone()
                    }
                };
                match action {
                    ScriptedAction::Respond {
                        status,
                        content_type,
                        body,
                        delay_ms,
                    } => {
                        if delay_ms > 0 {
                            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        }
                        let mut hdrs = HeaderMap::new();
                        hdrs.insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
                        Ok(UpstreamResponse {
                            status,
                            headers: hdrs,
                            body: UpstreamBody::Buffered(Bytes::from_static(body.as_bytes())),
                            used_proxy: proxy.map(ProxyEndpoint::redacted),
                        })
                    }
                    ScriptedAction::StreamChunks { chunks, delay_ms } => {
                        if delay_ms > 0 {
                            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        }
                        let s = futures_util::stream::iter(
                            chunks
                                .into_iter()
                                .map(|c| Ok(Bytes::from_static(c.as_bytes()))),
                        );
                        let mut hdrs = HeaderMap::new();
                        hdrs.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
                        Ok(UpstreamResponse {
                            status: StatusCode::OK,
                            headers: hdrs,
                            body: UpstreamBody::Stream(Box::pin(s)),
                            used_proxy: proxy.map(ProxyEndpoint::redacted),
                        })
                    }
                    ScriptedAction::TransportErr(msg, delay_ms) => {
                        if delay_ms > 0 {
                            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        }
                        Err(anyhow::anyhow!(msg))
                    }
                }
            })
        }
    }

    fn make_pooled_executor(
        proxies: Vec<&str>,
        scripts: HashMap<ProxyId, Vec<ScriptedAction>>,
        selection: ProxySelectionConfig,
    ) -> (OpenCodeExecutor, Arc<ProxyPool>, Arc<ScriptedRoutedTransport>) {
        let mut cfg = ProxyPoolConfig {
            enabled: true,
            selection,
            ..Default::default()
        };
        cfg.health_check.target_healthy = proxies.len().max(1);
        let parsed: Vec<ParsedProxy> = proxies
            .into_iter()
            .map(|s| ParsedProxy::parse_explicit(s, false).unwrap())
            .collect();
        let pool = ProxyPool::new(cfg, parsed, None);
        let routed = Arc::new(ScriptedRoutedTransport::new(scripts));
        let base_transport = Arc::new(MockTransport::new(vec![(
            StatusCode::OK,
            "data: {\"choices\":[{\"delta\":{\"content\":\"base\"}}]}\n\n",
        )]));
        let sessions = Arc::new(SessionManager::new(Duration::from_secs(7200)));
        let exec = OpenCodeExecutor::with_pooled_transport(
            base_transport,
            routed.clone(),
            Arc::clone(&pool),
            sessions,
            RetryConfig::default(),
            OPENCODE_CHAT_URL,
        );
        (exec, pool, routed)
    }

    #[test]
    fn validates_and_builds_v2_headers_without_x_opencode_request() {
        let mut incoming = HeaderMap::new();
        incoming.insert(
            USER_AGENT,
            HeaderValue::from_static("opencode/1.18.31 ai-sdk/4.0.40"),
        );
        incoming.insert(
            HeaderName::from_static("x-opencode-request"),
            HeaderValue::from_static("msg_123"),
        );
        let session = "ses_f534dfae8ffeCy4Ee4tLWNygDc";
        let out = build_upstream_headers(&incoming, session);

        assert_eq!(out.get(USER_AGENT).unwrap(), OPENCODE_UA);
        assert_eq!(out.get(AUTHORIZATION).unwrap(), "Bearer public");
        assert_eq!(out.get("x-opencode-client").unwrap(), "cli");
        assert_eq!(out.get("x-opencode-project").unwrap(), "global");
        assert_eq!(out.get("x-opencode-session").unwrap(), session);
        assert_eq!(out.get("x-opencode-session-id").unwrap(), session);
        assert_eq!(out.get("x-session-affinity").unwrap(), session);
        assert_eq!(out.get("x-session-id").unwrap(), session);
        assert!(out.get("x-opencode-request").is_none());

        let mut v2_in = HeaderMap::new();
        v2_in.insert(
            USER_AGENT,
            HeaderValue::from_static("opencode/latest/2.0.29/desktop"),
        );
        let out_v2 = build_upstream_headers(&v2_in, session);
        assert_eq!(
            out_v2.get(USER_AGENT).unwrap(),
            "opencode/latest/2.0.29/desktop"
        );
    }

    #[tokio::test]
    async fn retries_transient_overload_429_keeping_same_session() {
        let overload = r#"{"error":{"type":"server_error","message":"The inference service is overloaded. Please try again later."}}"#;
        let transport = Arc::new(MockTransport::new(vec![
            (StatusCode::TOO_MANY_REQUESTS, overload),
            (StatusCode::TOO_MANY_REQUESTS, overload),
            (StatusCode::OK, "data: ok\n\n"),
        ]));
        let sessions = Arc::new(SessionManager::new(Duration::from_secs(7200)));
        let exec = OpenCodeExecutor::with_transport(
            transport.clone(),
            sessions.clone(),
            RetryConfig {
                max_attempts: 8,
                base_delay_ms: 1,
                max_delay_ms: 2,
            },
            OPENCODE_CHAT_URL,
        );

        let resolved = sessions.resolve_session(&HeaderMap::new(), &json!({}));
        let res = exec
            .execute_chat(&HeaderMap::new(), Bytes::from_static(b"{}"), &resolved)
            .await
            .unwrap();
        let (status, _, body) = res.into_bytes().await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body.as_ref(), b"data: ok\n\n");

        let calls = transport.calls.lock().unwrap();
        assert_eq!(calls.len(), 3);
        let s0 = calls[0].headers.get("x-opencode-session").unwrap();
        assert_eq!(calls[1].headers.get("x-opencode-session").unwrap(), s0);
        assert_eq!(calls[2].headers.get("x-opencode-session").unwrap(), s0);
    }

    #[tokio::test]
    async fn stops_at_max_attempts_preserving_429_body() {
        let overload = r#"{"error":{"message":"The inference service is overloaded."}}"#;
        let transport = Arc::new(MockTransport::new(vec![(
            StatusCode::TOO_MANY_REQUESTS,
            overload,
        )]));
        let sessions = Arc::new(SessionManager::new(Duration::from_secs(7200)));
        let exec = OpenCodeExecutor::with_transport(
            transport.clone(),
            sessions.clone(),
            RetryConfig {
                max_attempts: 3,
                base_delay_ms: 1,
                max_delay_ms: 2,
            },
            OPENCODE_CHAT_URL,
        );

        let resolved = sessions.resolve_session(&HeaderMap::new(), &json!({}));
        let res = exec
            .execute_chat(&HeaderMap::new(), Bytes::from_static(b"{}"), &resolved)
            .await
            .unwrap();
        let (status, _, body) = res.into_bytes().await.unwrap();
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(body.as_ref(), overload.as_bytes());
        assert_eq!(transport.calls.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn does_not_retry_free_usage_429_and_evicts_minted_session() {
        let quota_err = r#"{"type":"error","error":{"type":"FreeUsageLimitError"}}"#;
        let transport = Arc::new(MockTransport::new(vec![
            (StatusCode::TOO_MANY_REQUESTS, quota_err),
            (StatusCode::OK, "data: ok\n\n"),
        ]));
        let sessions = Arc::new(SessionManager::new(Duration::from_secs(7200)));
        let exec = OpenCodeExecutor::with_transport(
            transport.clone(),
            sessions.clone(),
            RetryConfig {
                max_attempts: 8,
                base_delay_ms: 1,
                max_delay_ms: 2,
            },
            OPENCODE_CHAT_URL,
        );

        let first = sessions.resolve_session(&HeaderMap::new(), &json!({}));
        let res = exec
            .execute_chat(&HeaderMap::new(), Bytes::from_static(b"{}"), &first)
            .await
            .unwrap();
        assert_eq!(res.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(transport.calls.lock().unwrap().len(), 1);

        // Next resolution must mint a fresh session ID.
        let second = sessions.resolve_session(&HeaderMap::new(), &json!({}));
        assert_ne!(first.id, second.id);
    }

    #[tokio::test]
    async fn evicts_minted_session_on_403_but_never_native_session() {
        let transport = Arc::new(MockTransport::new(vec![(
            StatusCode::FORBIDDEN,
            r#"{"error":"FreeTierError"}"#,
        )]));
        let sessions = Arc::new(SessionManager::new(Duration::from_secs(7200)));
        let exec = OpenCodeExecutor::with_transport(
            transport,
            sessions.clone(),
            RetryConfig::default(),
            OPENCODE_CHAT_URL,
        );

        let minted = sessions.resolve_session(&HeaderMap::new(), &json!({}));
        let _ = exec
            .execute_chat(&HeaderMap::new(), Bytes::from_static(b"{}"), &minted)
            .await
            .unwrap();
        let re_minted = sessions.resolve_session(&HeaderMap::new(), &json!({}));
        assert_ne!(minted.id, re_minted.id);

        let native_id = "ses_0123456789abABCDEFGHIJKLMN";
        let mut native_headers = HeaderMap::new();
        native_headers.insert(
            "x-opencode-session",
            HeaderValue::from_static(native_id),
        );
        let native = sessions.resolve_session(&native_headers, &json!({}));
        assert!(native.is_native);
        let _ = exec
            .execute_chat(&native_headers, Bytes::from_static(b"{}"), &native)
            .await
            .unwrap();
        let native_again = sessions.resolve_session(&native_headers, &json!({}));
        assert_eq!(native_again.id, native_id);
    }

    #[tokio::test]
    async fn pooled_failover_switches_seamlessly_on_429_403_html_and_sse_in_stream_errors() {
        let mut scripts = HashMap::new();
        // Proxy 1: HTTP 429 FreeUsageLimitError
        scripts.insert(
            ProxyId(1),
            vec![ScriptedAction::Respond {
                status: StatusCode::TOO_MANY_REQUESTS,
                content_type: "application/json",
                body: r#"{"error":{"type":"FreeUsageLimitError"}}"#,
                delay_ms: 0,
            }],
        );
        // Proxy 2: 200 OK but in-stream SSE error caught by first-frame peek
        scripts.insert(
            ProxyId(2),
            vec![ScriptedAction::StreamChunks {
                chunks: vec![r#"data: {"error":{"type":"FreeUsageLimitError","status":429}}"#],
                delay_ms: 0,
            }],
        );
        // Proxy 3: 200 OK captive portal HTML hijack
        scripts.insert(
            ProxyId(3),
            vec![ScriptedAction::Respond {
                status: StatusCode::OK,
                content_type: "text/html",
                body: "<html>captive portal</html>",
                delay_ms: 0,
            }],
        );
        // Proxy 4: 403 Forbidden
        scripts.insert(
            ProxyId(4),
            vec![ScriptedAction::Respond {
                status: StatusCode::FORBIDDEN,
                content_type: "application/json",
                body: r#"{"error":"Forbidden"}"#,
                delay_ms: 0,
            }],
        );
        // Proxy 5: clean SSE stream split across two chunks
        scripts.insert(
            ProxyId(5),
            vec![ScriptedAction::StreamChunks {
                chunks: vec![
                    "data: {\"id\":\"chatcmpl-ok\",\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n",
                    "data: {\"id\":\"chatcmpl-ok\",\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]\n\n",
                ],
                delay_ms: 0,
            }],
        );

        let (exec, pool, routed) = make_pooled_executor(
            vec![
                "socks5://1.1.1.1:1080",
                "socks5://2.2.2.2:1080",
                "socks5://3.3.3.3:1080",
                "socks5://4.4.4.4:1080",
                "socks5://5.5.5.5:1080",
            ],
            scripts,
            ProxySelectionConfig {
                hedge_delay_ms: None,
                ..Default::default()
            },
        );
        for (i, rtt) in [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)] {
            pool.report_success(ProxyId(i), rtt);
        }

        let fb = exec.sessions().resolve_session(&HeaderMap::new(), &json!({}));
        let res = exec
            .execute_chat_with_body(
                &HeaderMap::new(),
                Some(&json!({"messages":[]})),
                Bytes::from_static(b"{}"),
                &fb,
            )
            .await
            .unwrap();

        assert_eq!(res.status, StatusCode::OK);
        assert_eq!(res.used_proxy.as_deref(), Some("socks5://5.5.5.5:1080"));
        let (_, _, raw) = res.into_bytes().await.unwrap();
        let text = String::from_utf8(raw.to_vec()).unwrap();
        assert!(text.contains("hel"));
        assert!(text.contains("lo"));
        assert!(text.contains("[DONE]"));

        // Verify distinct sessions were sent across proxies and Proxy 3 is Dead
        let calls = routed.calls.lock().unwrap();
        assert_eq!(calls.len(), 5);
        assert_ne!(calls[0].1, calls[4].1);
        let stats = pool.stats();
        assert_eq!(stats.dead, 1);
        assert_eq!(stats.cooldown, 3);
        assert_eq!(stats.active_proxy.as_deref(), Some("socks5://5.5.5.5:1080"));
    }

    #[tokio::test]
    async fn option_a_speculative_hedge_wins_fast_over_stalled_proxy() {
        let mut scripts = HashMap::new();
        // Proxy 1 stalls 3000ms
        scripts.insert(
            ProxyId(1),
            vec![ScriptedAction::StreamChunks {
                chunks: vec!["data: {\"choices\":[{\"delta\":{\"content\":\"slow\"}}]}\n\n"],
                delay_ms: 3000,
            }],
        );
        // Proxy 2 answers in 5ms
        scripts.insert(
            ProxyId(2),
            vec![ScriptedAction::StreamChunks {
                chunks: vec!["data: {\"choices\":[{\"delta\":{\"content\":\"fast\"}}]}\n\ndata: [DONE]\n\n"],
                delay_ms: 5,
            }],
        );

        let (exec, pool, _) = make_pooled_executor(
            vec!["socks5://1.1.1.1:1080", "socks5://2.2.2.2:1080"],
            scripts,
            ProxySelectionConfig {
                hedge_delay_ms: Some(40),
                failover_timeout_secs: Some(5),
                ..Default::default()
            },
        );
        pool.report_success(ProxyId(1), 10);
        pool.report_success(ProxyId(2), 20);

        let t0 = Instant::now();
        let fb = exec.sessions().resolve_session(&HeaderMap::new(), &json!({}));
        let res = exec
            .execute_chat_with_body(&HeaderMap::new(), Some(&json!({})), Bytes::from_static(b"{}"), &fb)
            .await
            .unwrap();
        assert!(t0.elapsed() < Duration::from_millis(500));
        assert_eq!(res.used_proxy.as_deref(), Some("socks5://2.2.2.2:1080"));
        assert_eq!(pool.stats().hedged_wins, 1);
        assert_eq!(pool.stats().active_proxy.as_deref(), Some("socks5://2.2.2.2:1080"));
    }

    #[tokio::test]
    async fn pooled_waits_for_newly_probed_proxy_when_bench_is_empty() {
        let mut scripts = HashMap::new();
        scripts.insert(
            ProxyId(1),
            vec![ScriptedAction::StreamChunks {
                chunks: vec!["data: {\"choices\":[{\"delta\":{\"content\":\"ready\"}}]}\n\ndata: [DONE]\n\n"],
                delay_ms: 0,
            }],
        );
        let (exec, pool, _) = make_pooled_executor(
            vec!["socks5://9.9.9.9:1080"],
            scripts,
            ProxySelectionConfig {
                hedge_delay_ms: None,
                failover_timeout_secs: Some(5),
                ..Default::default()
            },
        );
        // Proxy 1 starts Untested (healthy == 0). Promote it in 40ms.
        let pool_bg = Arc::clone(&pool);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(40)).await;
            pool_bg.report_success(ProxyId(1), 25);
        });

        let fb = exec.sessions().resolve_session(&HeaderMap::new(), &json!({}));
        let res = exec
            .execute_chat_with_body(&HeaderMap::new(), Some(&json!({})), Bytes::from_static(b"{}"), &fb)
            .await
            .unwrap();
        assert_eq!(res.status, StatusCode::OK);
        assert_eq!(res.used_proxy.as_deref(), Some("socks5://9.9.9.9:1080"));
    }

    #[tokio::test]
    async fn request_scoped_400_returns_immediately_without_penalizing_proxy() {
        let mut scripts = HashMap::new();
        scripts.insert(
            ProxyId(1),
            vec![ScriptedAction::Respond {
                status: StatusCode::BAD_REQUEST,
                content_type: "application/json",
                body: r#"{"error":{"type":"invalid_request_error","message":"bad model"}}"#,
                delay_ms: 0,
            }],
        );
        let (exec, pool, routed) = make_pooled_executor(
            vec!["socks5://1.1.1.1:1080", "socks5://2.2.2.2:1080"],
            scripts,
            ProxySelectionConfig::default(),
        );
        pool.report_success(ProxyId(1), 10);
        pool.report_success(ProxyId(2), 20);

        let fb = exec.sessions().resolve_session(&HeaderMap::new(), &json!({}));
        let res = exec
            .execute_chat_with_body(&HeaderMap::new(), Some(&json!({})), Bytes::from_static(b"{}"), &fb)
            .await
            .unwrap();
        assert_eq!(res.status, StatusCode::BAD_REQUEST);
        assert_eq!(routed.calls.lock().unwrap().len(), 1);
        assert_eq!(pool.stats().healthy, 2);
        assert_eq!(pool.stats().cooldown, 0);
    }

    #[tokio::test]
    async fn transport_error_and_double_overload_rotate_to_standby() {
        let mut scripts = HashMap::new();
        scripts.insert(
            ProxyId(1),
            vec![ScriptedAction::TransportErr("tcp reset", 0)],
        );
        scripts.insert(
            ProxyId(2),
            vec![
                ScriptedAction::Respond {
                    status: StatusCode::TOO_MANY_REQUESTS,
                    content_type: "application/json",
                    body: r#"{"error":{"message":"The inference service is overloaded."}}"#,
                    delay_ms: 0,
                },
                ScriptedAction::Respond {
                    status: StatusCode::TOO_MANY_REQUESTS,
                    content_type: "application/json",
                    body: r#"{"error":{"message":"The inference service is overloaded."}}"#,
                    delay_ms: 0,
                },
            ],
        );
        scripts.insert(
            ProxyId(3),
            vec![ScriptedAction::StreamChunks {
                chunks: vec!["data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"],
                delay_ms: 0,
            }],
        );
        let (exec, pool, routed) = make_pooled_executor(
            vec![
                "socks5://1.1.1.1:1080",
                "socks5://2.2.2.2:1080",
                "socks5://3.3.3.3:1080",
            ],
            scripts,
            ProxySelectionConfig {
                hedge_delay_ms: None,
                ..Default::default()
            },
        );
        pool.report_success(ProxyId(1), 10);
        pool.report_success(ProxyId(2), 20);
        pool.report_success(ProxyId(3), 30);

        let fb = exec.sessions().resolve_session(&HeaderMap::new(), &json!({}));
        let res = exec
            .execute_chat_with_body(&HeaderMap::new(), Some(&json!({})), Bytes::from_static(b"{}"), &fb)
            .await
            .unwrap();
        assert_eq!(res.status, StatusCode::OK);
        assert_eq!(res.used_proxy.as_deref(), Some("socks5://3.3.3.3:1080"));
        // Proxy 1 (1 call) -> Proxy 2 (2 overload calls) -> Proxy 3 (1 call)
        assert_eq!(routed.calls.lock().unwrap().len(), 4);
    }
}
