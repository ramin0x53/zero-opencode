use axum::Json;
use axum::http::StatusCode;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

/// OpenAI-compatible JSON error payload.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub error_type: String,
    pub code: Option<String>,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            error_type: "invalid_request_error".to_string(),
            code: None,
        }
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: message.into(),
            error_type: "authentication_error".to_string(),
            code: Some("invalid_api_key".to_string()),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
            error_type: "invalid_request_error".to_string(),
            code: Some("model_not_found".to_string()),
        }
    }

    pub fn bad_gateway(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: message.into(),
            error_type: "upstream_error".to_string(),
            code: None,
        }
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            error_type: "service_unavailable".to_string(),
            code: Some("proxy_pool_exhausted".to_string()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
            }
        });
        (self.status, Json(body)).into_response()
    }
}

/// Forward a non-2xx upstream response, preserving valid JSON error envelopes
/// or wrapping plain text in OpenAI's `{ "error": ... }` structure.
pub fn forward_upstream_error(status: StatusCode, raw: &[u8]) -> Response {
    if let Ok(val) = serde_json::from_slice::<Value>(raw)
        && val.is_object()
    {
        return (status, [(CONTENT_TYPE, "application/json")], raw.to_vec()).into_response();
    }
    let text = String::from_utf8_lossy(raw).trim().to_string();
    let msg = if text.is_empty() {
        format!("upstream returned HTTP {}", status.as_u16())
    } else {
        text
    };
    ApiError {
        status,
        message: msg,
        error_type: "upstream_error".to_string(),
        code: Some(status.as_u16().to_string()),
    }
    .into_response()
}
