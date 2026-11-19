use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, USER_AGENT};
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use regex::Regex;
use wreq::Client;

use crate::config::RetryConfig;
use crate::opencode::session::{OPENCODE_SESSION_HEADER, ResolvedSession, SessionManager};

pub const OPENCODE_CHAT_URL: &str = "https://opencode.ai/inference/v1/chat/completions";
pub const OPENCODE_UA: &str = "opencode/latest/2.0.26/cli";
pub const OPENCODE_CLIENT: &str = "cli";

static VALID_OPENCODE_UA_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^opencode/[a-z0-9-]+/(\d+)\.\d+\.\d+/[a-z0-9-]+$").expect("valid regex")
});
static OVERLOAD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)overloaded").expect("valid regex"));

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

/// Abstraction over the HTTP client so retry & session-eviction logic can be unit tested offline.
pub trait UpstreamTransport: Send + Sync {
    fn post<'a>(
        &'a self,
        url: &'a str,
        headers: HeaderMap,
        body: Bytes,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<UpstreamResponse>> + Send + 'a>>;
}

/// Production `wreq` BoringSSL transport.
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
        Box::pin(async move {
            let res = self
                .client
                .post(url)
                .headers(headers)
                .body(body)
                .send()
                .await?;
            let status = res.status();
            let resp_headers = res.headers().clone();
            if status == StatusCode::TOO_MANY_REQUESTS {
                let bytes = res.bytes().await?;
                return Ok(UpstreamResponse {
                    status,
                    headers: resp_headers,
                    body: UpstreamBody::Buffered(bytes),
                });
            }
            let stream = res
                .bytes_stream()
                .map(|chunk| chunk.map_err(anyhow::Error::from));
            Ok(UpstreamResponse {
                status,
                headers: resp_headers,
                body: UpstreamBody::Stream(Box::pin(stream)),
            })
        })
    }
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

/// Executes requests against OpenCode Free with session management and overload 429 retries.
#[derive(Clone)]
pub struct OpenCodeExecutor {
    transport: Arc<dyn UpstreamTransport>,
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
            sessions,
            retry,
            chat_url: chat_url.into(),
        }
    }

    pub fn sessions(&self) -> &SessionManager {
        &self.sessions
    }

    /// Execute an upstream OpenCode chat completion request, handling transient `429 overloaded`
    /// retries with backoff and dropping spent sessions on `403` / non-overload `429`.
    pub async fn execute_chat(
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
                if OVERLOAD_RE.is_match(&text) && attempt < max_attempts {
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

                if !OVERLOAD_RE.is_match(&text)
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
                });
            }

            return Ok(resp);
        }

        unreachable!("loop returns by attempt == max_attempts")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
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
                })
            })
        }
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
}
