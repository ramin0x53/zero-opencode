pub mod dashboard;
pub mod error;
pub mod handlers;
pub mod metrics;

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use axum::Router;
use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, post};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::server::error::ApiError;
use crate::server::handlers::AppState;

async fn bearer_auth(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let Some(expected) = state
        .config
        .server
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return Ok(next.run(req).await);
    };

    let headers = req.headers();
    let bearer = headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer ").or_else(|| s.strip_prefix("bearer ")))
        .map(str::trim);
    let x_api_key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(str::trim);

    if bearer == Some(expected) || x_api_key == Some(expected) {
        Ok(next.run(req).await)
    } else {
        Err(ApiError::unauthorized("Invalid or missing API key"))
    }
}

/// Build the standalone router for the embedded dashboard (never attaches permissive CORS).
pub fn build_dashboard_app(state: Arc<AppState>) -> Router {
    dashboard::build_dashboard_router(state.clone())
        .with_state(state)
        .layer(TraceLayer::new_for_http())
}

/// Build the Axum router with OpenAI-compatible routes, optional Bearer auth,
/// API-scoped CORS, optional embedded dashboard (when sharing the main listener),
/// and tracing.
pub fn build_router(state: Arc<AppState>) -> Router {
    let authed_routes = Router::new()
        .route("/v1/models", get(handlers::list_models))
        .route("/models", get(handlers::list_models))
        .route("/v1/models/{id}", get(handlers::get_model))
        .route("/models/{id}", get(handlers::get_model))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/chat/completions", post(handlers::chat_completions))
        .route_layer(middleware::from_fn_with_state(state.clone(), bearer_auth));

    let mut api_router = Router::new()
        .route("/health", get(handlers::health))
        .merge(authed_routes);

    if state.config.server.cors {
        api_router = api_router.layer(CorsLayer::permissive());
    }

    let mut router = api_router;
    if state.config.dashboard.enabled && !state.config.dashboard_uses_separate_listener() {
        router = router.merge(dashboard::build_dashboard_router(state.clone()));
    }

    router
        .with_state(state)
        .layer(TraceLayer::new_for_http())
}

async fn bind_listener(host: &str, port: u16, label: &str) -> anyhow::Result<(TcpListener, SocketAddr)> {
    let trimmed = host.trim();
    let clean_host = trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(trimmed);
    let listener = TcpListener::bind((clean_host, port))
        .await
        .with_context(|| format!("failed to bind {label} on {clean_host}:{port}"))?;
    let local_addr = listener
        .local_addr()
        .with_context(|| format!("failed to read local_addr for {label}"))?;
    Ok((listener, local_addr))
}

async fn wait_for_shutdown(mut rx: tokio::sync::watch::Receiver<bool>) {
    while !*rx.borrow_and_update() {
        if rx.changed().await.is_err() {
            break;
        }
    }
}

async fn serve_dual<F>(
    api_listener: TcpListener,
    dash_listener: TcpListener,
    state: Arc<AppState>,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let watcher_tx = shutdown_tx.clone();
    let watcher = tokio::spawn(async move {
        shutdown.await;
        let _ = watcher_tx.send(true);
    });

    let api_app = build_router(Arc::clone(&state));
    let dash_app = build_dashboard_app(state);

    let api_tx = shutdown_tx.clone();
    let api_rx = shutdown_rx.clone();
    let api_fut = async move {
        let res = axum::serve(api_listener, api_app)
            .with_graceful_shutdown(wait_for_shutdown(api_rx))
            .await
            .context("API HTTP server error");
        let _ = api_tx.send(true);
        res
    };

    let dash_tx = shutdown_tx;
    let dash_fut = async move {
        let res = axum::serve(dash_listener, dash_app)
            .with_graceful_shutdown(wait_for_shutdown(shutdown_rx))
            .await
            .context("dashboard HTTP server error");
        let _ = dash_tx.send(true);
        res
    };

    let (api_res, dash_res) = tokio::join!(api_fut, dash_fut);
    watcher.abort();
    api_res?;
    dash_res?;
    Ok(())
}

/// Start the HTTP server (and optional dedicated dashboard listener) and block until Ctrl-C / SIGTERM.
pub async fn serve(state: Arc<AppState>) -> anyhow::Result<()> {
    let (api_listener, api_addr) =
        bind_listener(&state.config.server.host, state.config.server.port, "API server").await?;

    if state.config.dashboard.enabled && state.config.dashboard_uses_separate_listener() {
        let (dash_listener, dash_addr) = bind_listener(
            state.config.dashboard_host(),
            state.config.dashboard_port(),
            "dashboard server",
        )
        .await?;

        info!(addr = %api_addr, "zero-opencode listening");
        info!(
            addr = %dash_addr,
            url = %format!("http://{dash_addr}/dashboard"),
            "embedded dashboard listening on dedicated address"
        );
        return serve_dual(api_listener, dash_listener, state, shutdown_signal()).await;
    }

    info!(addr = %api_addr, "zero-opencode listening");
    if state.config.dashboard.enabled {
        info!(url = %format!("http://{api_addr}/dashboard"), "embedded dashboard enabled");
    }

    let app = build_router(state);
    axum::serve(api_listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("HTTP server error")?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("shutting down gracefully");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    use axum::body::Body;
    use axum::http::{HeaderMap, StatusCode};
    use bytes::Bytes;
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use crate::config::{AppConfig, RetryConfig};
    use crate::opencode::executor::{
        OPENCODE_CHAT_URL, OpenCodeExecutor, UpstreamBody, UpstreamResponse, UpstreamTransport,
    };
    use crate::opencode::models::ModelCatalog;
    use crate::opencode::session::SessionManager;
    use crate::opencode::tls_client::build_opencode_client;

    use crate::server::dashboard::DashboardAuth;
    use crate::server::metrics::MetricsStore;

    struct CaptureTransport {
        sent_bodies: Mutex<Vec<Value>>,
        sse_payload: &'static str,
    }

    impl UpstreamTransport for CaptureTransport {
        fn post<'a>(
            &'a self,
            _url: &'a str,
            _headers: HeaderMap,
            body: Bytes,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<UpstreamResponse>> + Send + 'a>> {
            Box::pin(async move {
                let parsed: Value = serde_json::from_slice(&body).unwrap();
                self.sent_bodies.lock().unwrap().push(parsed);
                Ok(UpstreamResponse {
                    status: StatusCode::OK,
                    headers: HeaderMap::new(),
                    body: UpstreamBody::Buffered(Bytes::from_static(self.sse_payload.as_bytes())),
                    used_proxy: None,
                })
            })
        }
    }

    fn make_test_state(
        api_key: Option<&str>,
        sse_payload: &'static str,
    ) -> (Arc<AppState>, Arc<CaptureTransport>) {
        let mut cfg = AppConfig::default();
        cfg.server.api_key = api_key.map(ToString::to_string);
        let client = build_opencode_client(None, 30).unwrap();
        let catalog = ModelCatalog::new(client.clone(), Duration::from_secs(600));
        let sessions = Arc::new(SessionManager::new(Duration::from_secs(7200)));
        let transport = Arc::new(CaptureTransport {
            sent_bodies: Mutex::new(Vec::new()),
            sse_payload,
        });
        let executor = OpenCodeExecutor::with_transport(
            transport.clone(),
            sessions,
            RetryConfig::default(),
            OPENCODE_CHAT_URL,
        );
        let metrics = MetricsStore::new(cfg.dashboard.max_request_history);
        let dashboard_auth = DashboardAuth::from_config(&cfg.dashboard);
        let state = Arc::new(AppState {
            config: cfg,
            catalog,
            executor,
            proxy_pool: None,
            metrics,
            dashboard_auth,
            base_client: client,
            started_at: Instant::now(),
        });
        (state, transport)
    }

    #[tokio::test]
    async fn health_is_public_while_v1_enforces_api_key() {
        let (state, _) = make_test_state(Some("secret-123"), "");
        let app = build_router(state);

        let health_res = app
            .clone()
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health_res.status(), StatusCode::OK);

        let unauth = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

        let bad_req_authed = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret-123")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":""}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(bad_req_authed.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn chat_completions_handles_non_stream_and_restores_terminal_tool() {
        let sse = concat!(
            "data: {\"id\":\"chatcmpl-123456\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"run_command\",\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl-123456\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let (state, transport) = make_test_state(None, sse);
        let app = build_router(state);

        let req_body = json!({
            "model": "big-pickle",
            "stream": false,
            "messages": [{"role": "user", "content": "list files"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "run command",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}
                }
            }]
        });

        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let json_res: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            json_res["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "terminal"
        );

        let sent = transport.sent_bodies.lock().unwrap();
        assert_eq!(sent[0]["stream"], true);
        let sent_tools = sent[0]["tools"].as_array().unwrap();
        assert!(sent_tools.iter().any(|t| t["function"]["name"] == "read"));
        assert!(sent_tools.iter().any(|t| t["function"]["name"] == "shell"));
        assert!(!sent_tools.iter().any(|t| {
            t["function"]["name"]
                .as_str()
                .is_some_and(|n| n.to_lowercase().contains("terminal"))
        }));
    }

    #[tokio::test]
    async fn chat_completions_streams_normalized_sse() {
        let sse = concat!(
            "data: {\"id\":\"chat\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning\":\"thinking...\"}}]}\n\n",
            "data: {\"id\":\"chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"OK\"},\"finish_reason\":\"stop\"}]}\n\n"
        );
        let (state, _) = make_test_state(None, sse);
        let app = build_router(state);

        let req_body = json!({
            "model": "mimo-v2.6-flash-free",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        });

        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::OK);
        let text = String::from_utf8(
            res.into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(text.contains("\"reasoning_content\":\"thinking...\""));
        assert!(text.contains("\"content\":\"OK\""));
        assert!(text.ends_with("data: [DONE]\n\n"));
    }

    #[tokio::test]
    async fn models_endpoint_returns_9router_capabilities_and_opencode_metadata() {
        let raw = r#"{
            "object": "list",
            "data": [{
                "id": "mimo-v2.6-flash-free",
                "name": "MiMo-V2.6-Flash Free",
                "canonical_id": "xiaomi/mimo-v2.6-flash",
                "type": "chat",
                "description": "Multimodal model",
                "modalities": {"input": ["text", "image/jpeg", "audio/mpeg", "video/mp4"], "output": ["text"]},
                "limit": {"context": 200000, "output": 32000},
                "capabilities": {"tools": {"supported": true}, "reasoning": {"supported": true}},
                "cost": {"text": {"input": {"price": 0}, "output": {"price": 0}}},
                "recommended_api": "chat-completions"
            }]
        }"#;
        let models = crate::opencode::models::filter_free_models_json(raw).unwrap();
        let (state, _) = make_test_state(None, "");
        state.catalog.seed(models).await;
        let app = build_router(state);

        let list_res = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(list_res.status(), StatusCode::OK);
        let list_bytes = list_res.into_body().collect().await.unwrap().to_bytes();
        let list_json: Value = serde_json::from_slice(&list_bytes).unwrap();
        assert_eq!(list_json["object"], "list");
        let item = &list_json["data"][0];
        assert_eq!(item["id"], "mimo-v2.6-flash-free");
        assert_eq!(item["canonical_id"], "xiaomi/mimo-v2.6-flash");
        assert_eq!(item["description"], "Multimodal model");
        assert_eq!(item["context_length"], 200_000);
        assert_eq!(item["max_completion_tokens"], 32_000);
        assert_eq!(item["capabilities"]["vision"], true);
        assert_eq!(item["capabilities"]["audioInput"], true);
        assert_eq!(item["capabilities"]["videoInput"], true);
        assert!(item["modalities"].is_object());
        assert!(item["cost"].is_object());

        let single_res = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models/mimo-v2.6-flash-free")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(single_res.status(), StatusCode::OK);

        let missing_res = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models/unknown-model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn health_includes_proxy_pool_stats_when_pool_is_present() {
        use crate::config::ProxyPoolConfig;
        use crate::proxy::endpoint::{ParsedProxy, ProxyId};
        use crate::proxy::pool::ProxyPool;

        let pp_cfg = ProxyPoolConfig {
            enabled: true,
            ..Default::default()
        };
        let pool = ProxyPool::new(
            pp_cfg,
            vec![ParsedProxy::parse_explicit("socks5://1.2.3.4:1080", false).unwrap()],
            None,
        );
        pool.report_success(ProxyId(1), 42);

        let mut cfg = AppConfig::default();
        cfg.opencode.proxy_pool.enabled = true;
        let client = build_opencode_client(None, 30).unwrap();
        let catalog = ModelCatalog::new(client.clone(), Duration::from_secs(600));
        let sessions = Arc::new(SessionManager::new(Duration::from_secs(7200)));
        let transport = Arc::new(CaptureTransport {
            sent_bodies: Mutex::new(Vec::new()),
            sse_payload: "",
        });
        let executor = OpenCodeExecutor::with_transport(
            transport,
            sessions,
            RetryConfig::default(),
            OPENCODE_CHAT_URL,
        );
        let metrics = MetricsStore::new(cfg.dashboard.max_request_history);
        let dashboard_auth = DashboardAuth::from_config(&cfg.dashboard);
        let state = Arc::new(AppState {
            config: cfg,
            catalog,
            executor,
            proxy_pool: Some(pool),
            metrics,
            dashboard_auth,
            base_client: client,
            started_at: Instant::now(),
        });

        let app = build_router(state);
        let res = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["proxy_pool"]["enabled"], true);
        assert_eq!(body["proxy_pool"]["healthy"], 1);
        assert_eq!(body["proxy_pool"]["total"], 1);
        assert_eq!(body["proxy_pool"]["hop1_healthy"], true);
    }

    #[tokio::test]
    async fn chat_completions_returns_503_when_proxy_pool_exhausted() {
        use crate::config::ProxyPoolConfig;
        use crate::opencode::executor::RoutedTransport;
        use crate::proxy::endpoint::ProxyEndpoint;
        use crate::proxy::pool::ProxyPool;

        struct AlwaysEmptyTransport;
        impl RoutedTransport for AlwaysEmptyTransport {
            fn post_via<'a>(
                &'a self,
                _proxy: Option<&'a ProxyEndpoint>,
                _url: &'a str,
                _headers: HeaderMap,
                _body: Bytes,
            ) -> Pin<Box<dyn Future<Output = anyhow::Result<UpstreamResponse>> + Send + 'a>> {
                Box::pin(async move { anyhow::bail!("unreachable") })
            }
        }

        let mut pp_cfg = ProxyPoolConfig {
            enabled: true,
            ..Default::default()
        };
        pp_cfg.selection.failover_timeout_secs = Some(1);
        pp_cfg.selection.fallback_to_base = false;

        // No proxies ever become healthy -> PoolExhaustedError -> 503
        let pool = ProxyPool::new(pp_cfg, vec![], None);
        let mut cfg = AppConfig::default();
        cfg.opencode.proxy_pool.enabled = true;

        let client = build_opencode_client(None, 30).unwrap();
        let catalog = ModelCatalog::new(client.clone(), Duration::from_secs(600));
        let sessions = Arc::new(SessionManager::new(Duration::from_secs(7200)));
        let base_transport = Arc::new(CaptureTransport {
            sent_bodies: Mutex::new(Vec::new()),
            sse_payload: "",
        });
        let executor = OpenCodeExecutor::with_pooled_transport(
            base_transport,
            Arc::new(AlwaysEmptyTransport),
            Arc::clone(&pool),
            sessions,
            RetryConfig::default(),
            OPENCODE_CHAT_URL,
        );
        let metrics = MetricsStore::new(cfg.dashboard.max_request_history);
        let dashboard_auth = DashboardAuth::from_config(&cfg.dashboard);
        let state = Arc::new(AppState {
            config: cfg,
            catalog,
            executor,
            proxy_pool: Some(pool),
            metrics,
            dashboard_auth,
            base_client: client,
            started_at: Instant::now(),
        });

        let app = build_router(state);
        let req_body = json!({
            "model": "big-pickle",
            "stream": false,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["code"], "proxy_pool_exhausted");
    }

    fn make_dashboard_state(sse_payload: &'static str) -> Arc<AppState> {
        use crate::config::ProxyPoolConfig;
        use crate::proxy::endpoint::{ParsedProxy, ProxyId};
        use crate::proxy::pool::ProxyPool;

        let mut cfg = AppConfig::default();
        cfg.dashboard.enabled = true;
        cfg.dashboard.username = "ops".to_string();
        cfg.dashboard.password = "s3cr3t!".to_string();
        cfg.opencode.proxy_pool = ProxyPoolConfig {
            enabled: true,
            ..Default::default()
        };

        let pool = ProxyPool::new(
            cfg.opencode.proxy_pool.clone(),
            vec![
                ParsedProxy::parse_explicit("socks5://alice:pw@1.2.3.4:1080", false).unwrap(),
                ParsedProxy::parse_explicit("http://5.6.7.8:8080", false).unwrap(),
            ],
            None,
        );
        pool.report_success(ProxyId(1), 35);
        pool.report_rate_limited(ProxyId(2), Some(Duration::from_secs(120)), "429 limit");

        let client = build_opencode_client(None, 30).unwrap();
        let catalog = ModelCatalog::new(client.clone(), Duration::from_secs(600));
        let sessions = Arc::new(SessionManager::new(Duration::from_secs(7200)));
        let transport = Arc::new(CaptureTransport {
            sent_bodies: Mutex::new(Vec::new()),
            sse_payload,
        });
        let executor = OpenCodeExecutor::with_transport(
            transport,
            sessions,
            RetryConfig::default(),
            OPENCODE_CHAT_URL,
        );
        let metrics = MetricsStore::new(cfg.dashboard.max_request_history);
        let dashboard_auth = DashboardAuth::from_config(&cfg.dashboard);

        Arc::new(AppState {
            config: cfg,
            catalog,
            executor,
            proxy_pool: Some(pool),
            metrics,
            dashboard_auth,
            base_client: client,
            started_at: Instant::now(),
        })
    }

    #[tokio::test]
    async fn dashboard_disabled_by_default_returns_404() {
        let (state, _) = make_test_state(None, "");
        let app = build_router(state);
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/dashboard")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn dashboard_serves_nonce_csp_headers_and_enforces_auth_csrf_logout_and_metrics() {
        let sse = "data: {\"id\":\"chatcmpl-123456\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let state = make_dashboard_state(sse);
        let app = build_router(state);

        // 1. Root redirects to /dashboard
        let root_res = app
            .clone()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(root_res.status(), StatusCode::SEE_OTHER);
        assert_eq!(root_res.headers().get("location").unwrap(), "/dashboard");

        // 2. HTML serves nonce CSP, security headers, and no unreplaced placeholder or permissive CORS
        let html_res = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/dashboard")
                    .header("origin", "https://evil.example")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(html_res.status(), StatusCode::OK);
        assert!(html_res.headers().get("access-control-allow-origin").is_none());
        assert_eq!(html_res.headers().get("x-frame-options").unwrap(), "DENY");
        assert_eq!(
            html_res.headers().get("x-content-type-options").unwrap(),
            "nosniff"
        );
        assert_eq!(
            html_res.headers().get("cross-origin-resource-policy").unwrap(),
            "same-origin"
        );
        let csp = html_res
            .headers()
            .get("content-security-policy")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(csp.contains("require-trusted-types-for 'script'"));
        assert!(csp.contains("frame-ancestors 'none'"));
        let html_bytes = html_res.into_body().collect().await.unwrap().to_bytes();
        let html = String::from_utf8(html_bytes.to_vec()).unwrap();
        assert!(!html.contains("__CSP_NONCE__"));

        // 3. Unauthenticated overview returns 401; session probe returns authenticated=false
        let unauth_overview = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/dashboard/api/overview")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauth_overview.status(), StatusCode::UNAUTHORIZED);

        // 4. Cross-site login POST is rejected by Fetch Metadata check
        let cross_site = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/dashboard/api/login")
                    .header("sec-fetch-site", "cross-site")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"username":"ops","password":"s3cr3t!"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cross_site.status(), StatusCode::FORBIDDEN);

        // 5. Wrong password returns 401; valid credentials return 200 + HttpOnly SameSite=Lax cookie
        let bad_login = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/dashboard/api/login")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"username":"ops","password":"wrong"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(bad_login.status(), StatusCode::UNAUTHORIZED);

        let good_login = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/dashboard/api/login")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"username":"ops","password":"s3cr3t!"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(good_login.status(), StatusCode::OK);
        let set_cookie = good_login
            .headers()
            .get("set-cookie")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(set_cookie.contains("HttpOnly"));
        assert!(set_cookie.contains("SameSite=Lax"));
        let cookie_pair = set_cookie.split(';').next().unwrap().to_string();

        // 6. Execute a chat request (with sensitive user prompt) & check overview records metrics without leaking prompt
        let chat_res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"big-pickle","stream":false,"messages":[{"role":"user","content":"TOP_SECRET_PROMPT"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(chat_res.status(), StatusCode::OK);

        let overview_res = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/dashboard/api/overview")
                    .header("cookie", &cookie_pair)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(overview_res.status(), StatusCode::OK);
        assert!(overview_res.headers().get("access-control-allow-origin").is_none());
        let ov_bytes = overview_res.into_body().collect().await.unwrap().to_bytes();
        let ov_text = String::from_utf8(ov_bytes.to_vec()).unwrap();
        assert!(!ov_text.contains("TOP_SECRET_PROMPT"));
        assert!(!ov_text.contains("alice"));
        let ov: Value = serde_json::from_str(&ov_text).unwrap();
        assert_eq!(ov["metrics"]["summary"]["chat_requests"], 1);
        assert_eq!(ov["metrics"]["summary"]["success_requests"], 1);
        assert_eq!(ov["proxy_pool"]["healthy"], 1);
        assert_eq!(ov["proxy_pool"]["cooldown"], 1);

        // 7. Filter proxies endpoint
        let proxies_res = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/dashboard/api/proxies?status=healthy")
                    .header("cookie", &cookie_pair)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(proxies_res.status(), StatusCode::OK);
        let px_bytes = proxies_res.into_body().collect().await.unwrap().to_bytes();
        let px: Value = serde_json::from_slice(&px_bytes).unwrap();
        assert_eq!(px["total_matching"], 1);
        assert_eq!(px["proxies"][0]["proxy"], "socks5://1.2.3.4:1080");

        // 8. Logout sends Clear-Site-Data and revokes session token
        let logout_res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/dashboard/api/logout")
                    .header("cookie", &cookie_pair)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(logout_res.status(), StatusCode::OK);
        let clear_hdr = logout_res
            .headers()
            .get("clear-site-data")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(clear_hdr.contains("\"cookies\""));
        assert!(clear_hdr.contains("\"storage\""));
        assert!(clear_hdr.contains("\"cache\""));

        let post_logout = app
            .oneshot(
                Request::builder()
                    .uri("/dashboard/api/overview")
                    .header("cookie", &cookie_pair)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(post_logout.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn dashboard_login_rate_limits_repeated_failures() {
        let state = make_dashboard_state("");
        let app = build_router(state);

        for _ in 0..10 {
            let res = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/dashboard/api/login")
                        .header("content-type", "application/json")
                        .body(Body::from(r#"{"username":"ops","password":"bad"}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        }

        let limited = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/dashboard/api/login")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"username":"ops","password":"s3cr3t!"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(limited.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn dashboard_separate_listener_isolates_routes_and_shares_state() {
        let sse = "data: {\"id\":\"chatcmpl-sep123\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let base_state = make_dashboard_state(sse);
        let mut cfg = base_state.config.clone();
        cfg.dashboard.host = Some("127.0.0.1".to_string());
        cfg.dashboard.port = Some(8081);
        let state = Arc::new(AppState {
            config: cfg,
            catalog: base_state.catalog.clone(),
            executor: base_state.executor.clone(),
            proxy_pool: base_state.proxy_pool.clone(),
            metrics: base_state.metrics.clone(),
            dashboard_auth: base_state.dashboard_auth.clone(),
            base_client: base_state.base_client.clone(),
            started_at: base_state.started_at,
        });

        let api_app = build_router(Arc::clone(&state));
        let dash_app = build_dashboard_app(Arc::clone(&state));

        // API router must not expose / or /dashboard when dedicated listener is configured
        let api_dash = api_app
            .clone()
            .oneshot(Request::builder().uri("/dashboard").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(api_dash.status(), StatusCode::NOT_FOUND);

        let api_root = api_app
            .clone()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(api_root.status(), StatusCode::NOT_FOUND);

        // API router still serves /health and /v1/chat/completions
        let api_health = api_app
            .clone()
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(api_health.status(), StatusCode::OK);

        let chat_res = api_app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"big-pickle","stream":false,"messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(chat_res.status(), StatusCode::OK);

        // Dashboard app must serve / and /dashboard, but NOT /health or /v1/chat/completions
        let dash_root = dash_app
            .clone()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(dash_root.status(), StatusCode::SEE_OTHER);

        let dash_html = dash_app
            .clone()
            .oneshot(Request::builder().uri("/dashboard").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(dash_html.status(), StatusCode::OK);

        let dash_health = dash_app
            .clone()
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(dash_health.status(), StatusCode::NOT_FOUND);

        let dash_chat = dash_app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(dash_chat.status(), StatusCode::NOT_FOUND);

        // Shared AppState: login via dashboard app sees API request metrics and listener metadata
        let login = dash_app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/dashboard/api/login")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"username":"ops","password":"s3cr3t!"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(login.status(), StatusCode::OK);
        let cookie_pair = login
            .headers()
            .get("set-cookie")
            .unwrap()
            .to_str()
            .unwrap()
            .split(';')
            .next()
            .unwrap()
            .to_string();

        let ov_res = dash_app
            .oneshot(
                Request::builder()
                    .uri("/dashboard/api/overview")
                    .header("cookie", cookie_pair)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ov_res.status(), StatusCode::OK);
        let ov_bytes = ov_res.into_body().collect().await.unwrap().to_bytes();
        let ov: Value = serde_json::from_slice(&ov_bytes).unwrap();
        assert_eq!(ov["metrics"]["summary"]["chat_requests"], 1);
        assert_eq!(ov["dashboard"]["host"], "127.0.0.1");
        assert_eq!(ov["dashboard"]["port"], 8081);
        assert_eq!(ov["dashboard"]["separate_listener"], true);
    }

    async fn http_get_status(addr: SocketAddr, path: &str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let req = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
        stream.write_all(req.as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await.unwrap();
        let raw = String::from_utf8_lossy(&buf);
        raw.lines().next().unwrap_or("").to_string()
    }

    #[tokio::test]
    async fn serve_dual_binds_separate_api_and_dashboard_listeners_and_shuts_down() {
        let (api_listener, api_addr) = bind_listener("127.0.0.1", 0, "test api").await.unwrap();
        let (dash_listener, dash_addr) = bind_listener("127.0.0.1", 0, "test dashboard")
            .await
            .unwrap();

        let base_state = make_dashboard_state("");
        let mut cfg = base_state.config.clone();
        cfg.server.host = "127.0.0.1".to_string();
        cfg.server.port = api_addr.port();
        cfg.dashboard.enabled = true;
        cfg.dashboard.host = Some("127.0.0.1".to_string());
        cfg.dashboard.port = Some(dash_addr.port());
        assert!(cfg.dashboard_uses_separate_listener());

        let state = Arc::new(AppState {
            config: cfg,
            catalog: base_state.catalog.clone(),
            executor: base_state.executor.clone(),
            proxy_pool: base_state.proxy_pool.clone(),
            metrics: base_state.metrics.clone(),
            dashboard_auth: base_state.dashboard_auth.clone(),
            base_client: base_state.base_client.clone(),
            started_at: base_state.started_at,
        });

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let server_task = tokio::spawn(async move {
            serve_dual(api_listener, dash_listener, state, async {
                let _ = rx.await;
            })
            .await
        });

        // Both listeners are already bound before serve_dual starts accepting
        assert!(http_get_status(api_addr, "/health").await.contains("200 OK"));
        assert!(
            http_get_status(api_addr, "/dashboard")
                .await
                .contains("404 Not Found")
        );
        assert!(
            http_get_status(dash_addr, "/dashboard")
                .await
                .contains("200 OK")
        );
        assert!(
            http_get_status(dash_addr, "/health")
                .await
                .contains("404 Not Found")
        );

        let _ = tx.send(());
        let joined = tokio::time::timeout(Duration::from_secs(5), server_task)
            .await
            .expect("serve_dual should shut down cleanly within 5s")
            .expect("server task should not panic");
        joined.expect("serve_dual should return Ok(())");
    }
}
