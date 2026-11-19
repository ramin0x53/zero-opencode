pub mod error;
pub mod handlers;

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

/// Build the Axum router with OpenAI-compatible routes, optional Bearer auth,
/// CORS, and tracing.
pub fn build_router(state: Arc<AppState>) -> Router {
    let authed_routes = Router::new()
        .route("/v1/models", get(handlers::list_models))
        .route("/models", get(handlers::list_models))
        .route("/v1/models/{id}", get(handlers::get_model))
        .route("/models/{id}", get(handlers::get_model))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/chat/completions", post(handlers::chat_completions))
        .route_layer(middleware::from_fn_with_state(state.clone(), bearer_auth));

    let mut router = Router::new()
        .route("/health", get(handlers::health))
        .merge(authed_routes)
        .with_state(state.clone())
        .layer(TraceLayer::new_for_http());

    if state.config.server.cors {
        router = router.layer(CorsLayer::permissive());
    }

    router
}

/// Start the HTTP server and block until Ctrl-C / SIGTERM.
pub async fn serve(state: Arc<AppState>) -> anyhow::Result<()> {
    let addr_str = format!("{}:{}", state.config.server.host, state.config.server.port);
    let addr: SocketAddr = addr_str
        .parse()
        .with_context(|| format!("invalid bind address: {addr_str}"))?;

    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;

    info!(%addr, "zero-opencode listening");

    let app = build_router(state);
    axum::serve(listener, app)
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
        let catalog = ModelCatalog::new(client, Duration::from_secs(600));
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
        let state = Arc::new(AppState {
            config: cfg,
            catalog,
            executor,
            proxy_pool: None,
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
        let catalog = ModelCatalog::new(client, Duration::from_secs(600));
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
        let state = Arc::new(AppState {
            config: cfg,
            catalog,
            executor,
            proxy_pool: Some(pool),
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
        let catalog = ModelCatalog::new(client, Duration::from_secs(600));
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
        let state = Arc::new(AppState {
            config: cfg,
            catalog,
            executor,
            proxy_pool: Some(pool),
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
}
