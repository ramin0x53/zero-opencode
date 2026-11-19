use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::header::{CACHE_CONTROL, CONNECTION, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use serde_json::{Value, json};

use crate::config::AppConfig;
use crate::opencode::executor::{OpenCodeExecutor, PoolExhaustedError};
use crate::opencode::models::{FreeModel, ModelCatalog};
use crate::opencode::transform::transform_chat_request;
use crate::proxy::pool::ProxyPool;
use crate::server::dashboard::DashboardAuth;
use crate::server::error::{ApiError, forward_upstream_error};
use crate::server::metrics::{GuardedStream, MetricsStore, NewRequestRecord};
use crate::stream::aggregator::{AggregatorError, aggregate_sse_to_json};
use crate::stream::passthrough::into_normalized_sse_stream;

/// Shared server state across handlers.
pub struct AppState {
    pub config: AppConfig,
    pub catalog: ModelCatalog,
    pub executor: OpenCodeExecutor,
    pub proxy_pool: Option<Arc<ProxyPool>>,
    pub metrics: Arc<MetricsStore>,
    pub dashboard_auth: Arc<DashboardAuth>,
    pub base_client: wreq::Client,
    pub started_at: Instant,
}

fn format_model_object(m: &FreeModel) -> Value {
    m.to_api_json()
}

/// `GET /health`
pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut payload = json!({
        "status": "ok",
        "uptime_secs": state.started_at.elapsed().as_secs(),
        "cached_models": state.catalog.cached_count(),
    });
    if let Some(pool) = &state.proxy_pool
        && let Some(obj) = payload.as_object_mut()
    {
        obj.insert(
            "proxy_pool".to_string(),
            serde_json::to_value(pool.stats()).unwrap_or(Value::Null),
        );
    }
    Json(payload)
}

/// `GET /v1/models` and `GET /models`
pub async fn list_models(State(state): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    let started = Instant::now();
    match state.catalog.list().await {
        Ok(models) => {
            state.metrics.record_request(NewRequestRecord {
                method: "GET",
                path: "/v1/models".to_string(),
                model: None,
                stream: None,
                status: 200,
                duration_ms: started.elapsed().as_millis() as u64,
                proxy: None,
                error: None,
                is_chat: false,
                is_models: true,
                prompt_tokens: 0,
                completion_tokens: 0,
            });
            let data: Vec<Value> = models.iter().map(format_model_object).collect();
            Ok(Json(json!({
                "object": "list",
                "data": data,
            })))
        }
        Err(e) => {
            let msg = format!("failed to fetch OpenCode models: {e}");
            state.metrics.record_request(NewRequestRecord {
                method: "GET",
                path: "/v1/models".to_string(),
                model: None,
                stream: None,
                status: 502,
                duration_ms: started.elapsed().as_millis() as u64,
                proxy: None,
                error: Some(msg.clone()),
                is_chat: false,
                is_models: true,
                prompt_tokens: 0,
                completion_tokens: 0,
            });
            Err(ApiError::bad_gateway(msg))
        }
    }
}

/// `GET /v1/models/{id}` and `GET /models/{id}`
pub async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let started = Instant::now();
    let path = format!("/v1/models/{id}");
    match state.catalog.get(&id).await {
        Ok(Some(model)) => {
            state.metrics.record_request(NewRequestRecord {
                method: "GET",
                path,
                model: Some(model.id.clone()),
                stream: None,
                status: 200,
                duration_ms: started.elapsed().as_millis() as u64,
                proxy: None,
                error: None,
                is_chat: false,
                is_models: true,
                prompt_tokens: 0,
                completion_tokens: 0,
            });
            Ok(Json(format_model_object(&model)))
        }
        Ok(None) => {
            let msg = format!("The model `{id}` does not exist");
            state.metrics.record_request(NewRequestRecord {
                method: "GET",
                path,
                model: Some(id),
                stream: None,
                status: 404,
                duration_ms: started.elapsed().as_millis() as u64,
                proxy: None,
                error: Some(msg.clone()),
                is_chat: false,
                is_models: true,
                prompt_tokens: 0,
                completion_tokens: 0,
            });
            Err(ApiError::not_found(msg))
        }
        Err(e) => {
            let msg = format!("failed to fetch OpenCode models: {e}");
            state.metrics.record_request(NewRequestRecord {
                method: "GET",
                path,
                model: Some(id),
                stream: None,
                status: 502,
                duration_ms: started.elapsed().as_millis() as u64,
                proxy: None,
                error: Some(msg.clone()),
                is_chat: false,
                is_models: true,
                prompt_tokens: 0,
                completion_tokens: 0,
            });
            Err(ApiError::bad_gateway(msg))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn record_chat_error(
    metrics: &MetricsStore,
    model: Option<String>,
    stream: Option<bool>,
    status: u16,
    started: Instant,
    proxy: Option<String>,
    error: String,
) {
    metrics.record_request(NewRequestRecord {
        method: "POST",
        path: "/v1/chat/completions".to_string(),
        model,
        stream,
        status,
        duration_ms: started.elapsed().as_millis() as u64,
        proxy,
        error: Some(error),
        is_chat: true,
        is_models: false,
        prompt_tokens: 0,
        completion_tokens: 0,
    });
}

/// `POST /v1/chat/completions` and `POST /chat/completions`
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body_bytes: Bytes,
) -> Result<Response, ApiError> {
    let started = Instant::now();
    let in_flight_guard = state.metrics.start_in_flight();

    let mut body: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("invalid JSON body: {e}");
            record_chat_error(&state.metrics, None, None, 400, started, None, msg.clone());
            return Err(ApiError::bad_request(msg));
        }
    };

    if !body.is_object() {
        let msg = "request body must be a JSON object".to_string();
        record_chat_error(&state.metrics, None, None, 400, started, None, msg.clone());
        return Err(ApiError::bad_request(msg));
    }

    let raw_model = match body
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(m) => m.to_string(),
        None => {
            let msg = "`model` must be a non-empty string".to_string();
            record_chat_error(&state.metrics, None, None, 400, started, None, msg.clone());
            return Err(ApiError::bad_request(msg));
        }
    };

    let client_wants_stream = body
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if !body.get("messages").is_some_and(Value::is_array) {
        let msg = "`messages` must be an array".to_string();
        record_chat_error(
            &state.metrics,
            Some(raw_model),
            Some(client_wants_stream),
            400,
            started,
            None,
            msg.clone(),
        );
        return Err(ApiError::bad_request(msg));
    }

    let model_meta = state.catalog.get(&raw_model).await.ok().flatten();
    let session = state.executor.sessions().resolve_session(&headers, &body);
    let transformed = transform_chat_request(&mut body, &raw_model, model_meta.as_ref());

    let serialized = serde_json::to_vec(&body)
        .map_err(|e| ApiError::bad_request(format!("failed to serialize request: {e}")))?;

    let upstream = state
        .executor
        .execute_chat_with_body(&headers, Some(&body), Bytes::from(serialized), &session)
        .await
        .map_err(|e| {
            let err = if e.downcast_ref::<PoolExhaustedError>().is_some() {
                ApiError::service_unavailable(e.to_string())
            } else {
                ApiError::bad_gateway(format!("upstream request failed: {e}"))
            };
            record_chat_error(
                &state.metrics,
                Some(transformed.model.clone()),
                Some(client_wants_stream),
                err.status.as_u16(),
                started,
                None,
                err.message.clone(),
            );
            err
        })?;

    let used_proxy = upstream.used_proxy.clone();

    if !upstream.status.is_success() {
        let (status, _, raw_err) = upstream
            .into_bytes()
            .await
            .map_err(|e| {
                let msg = format!("failed reading upstream error: {e}");
                record_chat_error(
                    &state.metrics,
                    Some(transformed.model.clone()),
                    Some(client_wants_stream),
                    502,
                    started,
                    used_proxy.clone(),
                    msg.clone(),
                );
                ApiError::bad_gateway(msg)
            })?;
        record_chat_error(
            &state.metrics,
            Some(transformed.model),
            Some(client_wants_stream),
            status.as_u16(),
            started,
            used_proxy,
            String::from_utf8_lossy(&raw_err).into_owned(),
        );
        return Ok(forward_upstream_error(status, &raw_err));
    }

    if client_wants_stream {
        state.metrics.record_request(NewRequestRecord {
            method: "POST",
            path: "/v1/chat/completions".to_string(),
            model: Some(transformed.model.clone()),
            stream: Some(true),
            status: 200,
            duration_ms: started.elapsed().as_millis() as u64,
            proxy: used_proxy,
            error: None,
            is_chat: true,
            is_models: false,
            prompt_tokens: 0,
            completion_tokens: 0,
        });
        let sse_stream = into_normalized_sse_stream(
            upstream.body,
            transformed.rename_map,
            transformed.model,
        );
        let guarded_stream = GuardedStream::new(sse_stream, in_flight_guard);
        let mut resp = Response::new(Body::from_stream(guarded_stream));
        *resp.status_mut() = StatusCode::OK;
        let resp_headers = resp.headers_mut();
        resp_headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );
        resp_headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        resp_headers.insert(CONNECTION, HeaderValue::from_static("keep-alive"));
        resp_headers.insert(
            HeaderName::from_static("x-accel-buffering"),
            HeaderValue::from_static("no"),
        );
        return Ok(resp);
    }

    let json_resp = aggregate_sse_to_json(
        upstream.body,
        &transformed.model,
        &transformed.rename_map,
        state.config.streaming.keep_reasoning_on_json,
    )
    .await
    .map_err(|err| {
        let api_err = match err {
            AggregatorError::UpstreamError { status, body } => ApiError {
                status,
                message: body
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("upstream SSE error")
                    .to_string(),
                error_type: body
                    .get("error")
                    .and_then(|e| e.get("type"))
                    .and_then(Value::as_str)
                    .unwrap_or("upstream_error")
                    .to_string(),
                code: body
                    .get("error")
                    .and_then(|e| e.get("code"))
                    .and_then(Value::as_str)
                    .map(ToString::to_string),
            },
            other => ApiError::bad_gateway(other.to_string()),
        };
        record_chat_error(
            &state.metrics,
            Some(transformed.model.clone()),
            Some(false),
            api_err.status.as_u16(),
            started,
            used_proxy.clone(),
            api_err.message.clone(),
        );
        api_err
    })?;

    let prompt_tokens = json_resp
        .get("usage")
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion_tokens = json_resp
        .get("usage")
        .and_then(|u| u.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    state.metrics.record_request(NewRequestRecord {
        method: "POST",
        path: "/v1/chat/completions".to_string(),
        model: Some(transformed.model),
        stream: Some(false),
        status: 200,
        duration_ms: started.elapsed().as_millis() as u64,
        proxy: used_proxy,
        error: None,
        is_chat: true,
        is_models: false,
        prompt_tokens,
        completion_tokens,
    });

    Ok(Json(json_resp).into_response())
}
