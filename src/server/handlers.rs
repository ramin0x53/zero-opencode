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
use crate::server::error::{ApiError, forward_upstream_error};
use crate::stream::aggregator::{AggregatorError, aggregate_sse_to_json};
use crate::stream::passthrough::into_normalized_sse_stream;

/// Shared server state across handlers.
pub struct AppState {
    pub config: AppConfig,
    pub catalog: ModelCatalog,
    pub executor: OpenCodeExecutor,
    pub proxy_pool: Option<Arc<ProxyPool>>,
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
    let models = state
        .catalog
        .list()
        .await
        .map_err(|e| ApiError::bad_gateway(format!("failed to fetch OpenCode models: {e}")))?;

    let data: Vec<Value> = models.iter().map(format_model_object).collect();
    Ok(Json(json!({
        "object": "list",
        "data": data,
    })))
}

/// `GET /v1/models/{id}` and `GET /models/{id}`
pub async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let found = state
        .catalog
        .get(&id)
        .await
        .map_err(|e| ApiError::bad_gateway(format!("failed to fetch OpenCode models: {e}")))?;

    match found {
        Some(model) => Ok(Json(format_model_object(&model))),
        None => Err(ApiError::not_found(format!("The model `{id}` does not exist"))),
    }
}

/// `POST /v1/chat/completions` and `POST /chat/completions`
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body_bytes: Bytes,
) -> Result<Response, ApiError> {
    let mut body: Value = serde_json::from_slice(&body_bytes)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))?;

    if !body.is_object() {
        return Err(ApiError::bad_request("request body must be a JSON object"));
    }

    let raw_model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::bad_request("`model` must be a non-empty string"))?
        .to_string();

    if !body.get("messages").is_some_and(Value::is_array) {
        return Err(ApiError::bad_request("`messages` must be an array"));
    }

    let client_wants_stream = body
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);

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
            if e.downcast_ref::<PoolExhaustedError>().is_some() {
                ApiError::service_unavailable(e.to_string())
            } else {
                ApiError::bad_gateway(format!("upstream request failed: {e}"))
            }
        })?;

    if !upstream.status.is_success() {
        let (status, _, raw_err) = upstream
            .into_bytes()
            .await
            .map_err(|e| ApiError::bad_gateway(format!("failed reading upstream error: {e}")))?;
        return Ok(forward_upstream_error(status, &raw_err));
    }

    if client_wants_stream {
        let sse_stream = into_normalized_sse_stream(
            upstream.body,
            transformed.rename_map,
            transformed.model,
        );
        let mut resp = Response::new(Body::from_stream(sse_stream));
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
    .map_err(|err| match err {
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
    })?;

    Ok(Json(json_resp).into_response())
}
