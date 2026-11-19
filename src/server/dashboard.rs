use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Query, Request, State};
use axum::http::header::{AUTHORIZATION, CACHE_CONTROL, CONTENT_TYPE, COOKIE, SET_COOKIE, VARY};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::config::DashboardConfig;
use crate::proxy::endpoint::ParsedProxy;
use crate::proxy::loader::fetch_all_proxies;
use crate::proxy::pool::ProxyInfo;
use crate::server::handlers::AppState;

const DASHBOARD_HTML_TEMPLATE: &str = include_str!("dashboard.html");
const SESSION_COOKIE_NAME: &str = "zero_opencode_session";
const MAX_SESSIONS: usize = 64;
const LOGIN_RATE_LIMIT_MAX: usize = 10;
const LOGIN_RATE_LIMIT_WINDOW: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoginError {
    InvalidCredentials,
    RateLimited,
}

/// Manages dashboard credentials, brute-force rate limiting, and in-memory session cookies.
pub struct DashboardAuth {
    enabled: bool,
    username: String,
    password: String,
    session_ttl: Duration,
    sessions: Mutex<HashMap<String, Instant>>,
    failed_logins: Mutex<VecDeque<Instant>>,
}

impl DashboardAuth {
    pub fn from_config(cfg: &DashboardConfig) -> Arc<Self> {
        Arc::new(Self {
            enabled: cfg.enabled,
            username: cfg.username.clone(),
            password: cfg.password.clone(),
            session_ttl: Duration::from_secs(cfg.session_ttl_secs),
            sessions: Mutex::new(HashMap::new()),
            failed_logins: Mutex::new(VecDeque::new()),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn username(&self) -> &str {
        &self.username
    }

    pub fn session_ttl_secs(&self) -> u64 {
        self.session_ttl.as_secs()
    }

    /// Verify username & password in constant time and mint a 256-bit session token.
    pub fn verify_credentials(&self, user: &str, pass: &str) -> Result<String, LoginError> {
        if !self.enabled {
            return Err(LoginError::InvalidCredentials);
        }

        let now = Instant::now();
        {
            let mut fails = self.failed_logins.lock().expect("failed_logins lock");
            while fails
                .front()
                .is_some_and(|t| now.saturating_duration_since(*t) >= LOGIN_RATE_LIMIT_WINDOW)
            {
                fails.pop_front();
            }
            if fails.len() >= LOGIN_RATE_LIMIT_MAX {
                return Err(LoginError::RateLimited);
            }
        }

        let user_ok = constant_time_eq(user.as_bytes(), self.username.as_bytes());
        let pass_ok = constant_time_eq(pass.as_bytes(), self.password.as_bytes());
        if !(user_ok & pass_ok) {
            let mut fails = self.failed_logins.lock().expect("failed_logins lock");
            fails.push_back(now);
            return Err(LoginError::InvalidCredentials);
        }

        {
            let mut fails = self.failed_logins.lock().expect("failed_logins lock");
            fails.clear();
        }

        let token_bytes: [u8; 32] = rand::random();
        let token = hex::encode(token_bytes);
        let expiry = now + self.session_ttl;

        let mut sessions = self.sessions.lock().expect("sessions lock");
        sessions.retain(|_, exp| *exp > now);
        if sessions.len() >= MAX_SESSIONS
            && let Some(oldest_key) = sessions
                .iter()
                .min_by_key(|(_, exp)| **exp)
                .map(|(k, _)| k.clone())
        {
            sessions.remove(&oldest_key);
        }
        sessions.insert(token.clone(), expiry);
        Ok(token)
    }

    pub fn validate_session(&self, token: &str) -> bool {
        if !self.enabled || token.is_empty() {
            return false;
        }
        let now = Instant::now();
        let mut sessions = self.sessions.lock().expect("sessions lock");
        sessions.retain(|_, exp| *exp > now);
        let token_bytes = token.as_bytes();
        let mut matched = false;
        for stored in sessions.keys() {
            matched |= constant_time_eq(stored.as_bytes(), token_bytes);
        }
        matched
    }

    pub fn revoke_session(&self, token: &str) {
        if token.is_empty() {
            return;
        }
        let mut sessions = self.sessions.lock().expect("sessions lock");
        sessions.remove(token);
    }

    pub fn is_authenticated(&self, headers: &HeaderMap) -> bool {
        if !self.enabled {
            return false;
        }
        if let Some(token) = extract_session_cookie(headers)
            && self.validate_session(token)
        {
            return true;
        }
        if let Some((u, p)) = extract_basic_auth(headers) {
            let u_ok = constant_time_eq(u.as_bytes(), self.username.as_bytes());
            let p_ok = constant_time_eq(p.as_bytes(), self.password.as_bytes());
            return u_ok & p_ok;
        }
        false
    }
}

/// Build the dashboard Axum sub-router (only mounted when `dashboard.enabled` is true).
pub fn build_dashboard_router(state: Arc<AppState>) -> Router<Arc<AppState>> {
    let authed_api = Router::new()
        .route("/dashboard/api/overview", get(api_overview))
        .route("/dashboard/api/proxies", get(api_proxies))
        .route("/dashboard/api/proxies/probe", post(api_proxies_probe))
        .route("/dashboard/api/proxies/refresh", post(api_proxies_refresh))
        .route_layer(middleware::from_fn_with_state(state, require_dashboard_auth));

    Router::new()
        .route("/", get(redirect_root))
        .route("/dashboard", get(serve_dashboard_html))
        .route("/dashboard/", get(serve_dashboard_html))
        .route("/dashboard/api/session", get(api_session))
        .route("/dashboard/api/login", post(api_login))
        .route("/dashboard/api/logout", post(api_logout))
        .merge(authed_api)
}

async fn redirect_root() -> Response {
    let mut resp = Redirect::to("/dashboard").into_response();
    apply_dashboard_security_headers(resp.headers_mut(), None);
    resp
}

async fn serve_dashboard_html() -> Response {
    let nonce_bytes: [u8; 16] = rand::random();
    let nonce = hex::encode(nonce_bytes);
    let html = DASHBOARD_HTML_TEMPLATE.replace("__CSP_NONCE__", &nonce);
    let csp = format!(
        "default-src 'none'; script-src 'nonce-{nonce}' 'strict-dynamic'; style-src 'nonce-{nonce}'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'; require-trusted-types-for 'script'"
    );

    let mut resp = (
        StatusCode::OK,
        [(CONTENT_TYPE, "text/html; charset=utf-8")],
        html,
    )
        .into_response();
    apply_dashboard_security_headers(resp.headers_mut(), Some(&csp));
    resp
}

async fn api_session(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let authed = state.dashboard_auth.is_authenticated(&headers);
    let body = json!({
        "authenticated": authed,
        "username": if authed { Some(state.dashboard_auth.username()) } else { None },
    });
    json_response(StatusCode::OK, body)
}

async fn api_login(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Some(resp) = check_same_origin_post(&headers) {
        return resp;
    }

    let Some((username, password)) = parse_login_payload(&headers, &body) else {
        return json_response(StatusCode::BAD_REQUEST, json!({"error": "Invalid login payload"}));
    };

    match state
        .dashboard_auth
        .verify_credentials(&username, &password)
    {
        Ok(token) => {
            let mut resp = json_response(
                StatusCode::OK,
                json!({
                    "ok": true,
                    "username": state.dashboard_auth.username(),
                }),
            );
            let secure_attr = if is_https_request(&headers) {
                "; Secure"
            } else {
                ""
            };
            let cookie = format!(
                "{SESSION_COOKIE_NAME}={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age={}{secure_attr}",
                state.dashboard_auth.session_ttl_secs()
            );
            if let Ok(val) = HeaderValue::from_str(&cookie) {
                resp.headers_mut().insert(SET_COOKIE, val);
            }
            resp
        }
        Err(LoginError::RateLimited) => json_response(
            StatusCode::TOO_MANY_REQUESTS,
            json!({"error": "Too many failed login attempts; try again in 60s"}),
        ),
        Err(LoginError::InvalidCredentials) => json_response(
            StatusCode::UNAUTHORIZED,
            json!({"error": "Invalid username or password"}),
        ),
    }
}

async fn api_logout(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Some(resp) = check_same_origin_post(&headers) {
        return resp;
    }
    if let Some(token) = extract_session_cookie(&headers) {
        state.dashboard_auth.revoke_session(token);
    }
    let mut resp = json_response(StatusCode::OK, json!({"ok": true}));
    let secure_attr = if is_https_request(&headers) {
        "; Secure"
    } else {
        ""
    };
    let expired =
        format!("{SESSION_COOKIE_NAME}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0{secure_attr}");
    if let Ok(val) = HeaderValue::from_str(&expired) {
        resp.headers_mut().insert(SET_COOKIE, val);
    }
    resp.headers_mut().insert(
        HeaderName::from_static("clear-site-data"),
        HeaderValue::from_static(r#""cookies", "storage", "cache""#),
    );
    resp
}

async fn require_dashboard_auth(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    if !state.dashboard_auth.is_authenticated(req.headers()) {
        return json_response(StatusCode::UNAUTHORIZED, json!({"error": "Unauthorized"}));
    }
    next.run(req).await
}

async fn api_overview(State(state): State<Arc<AppState>>) -> Response {
    let cfg = &state.config;
    let pool_cfg = &cfg.opencode.proxy_pool;
    let chain_proxy = cfg
        .opencode
        .proxy
        .as_deref()
        .and_then(|raw| ParsedProxy::parse_explicit(raw, true).ok())
        .map(|p| p.redacted());

    let (pool_stats_json, proxies) = match &state.proxy_pool {
        Some(pool) => {
            let mut stats_val = serde_json::to_value(pool.stats()).unwrap_or(Value::Null);
            let earliest = pool.earliest_cooldown_remaining().map(|d| d.as_secs());
            if let Some(obj) = stats_val.as_object_mut() {
                obj.insert(
                    "earliest_cooldown_secs".to_string(),
                    earliest.map_or(Value::Null, |s| json!(s)),
                );
            }
            (stats_val, pool.snapshot_proxies(100))
        }
        None => (Value::Null, Vec::new()),
    };

    let models: Vec<Value> = state
        .catalog
        .cached_snapshot()
        .into_iter()
        .map(|m| {
            json!({
                "id": m.id,
                "name": m.name,
                "context_window": m.context_window,
                "max_output": m.max_output_tokens,
                "capabilities": m.capability_tags(),
                "thinking_levels": m.thinking_levels,
            })
        })
        .collect();

    let metrics = state.metrics.snapshot();

    json_response(
        StatusCode::OK,
        json!({
            "server": {
                "host": cfg.server.host,
                "port": cfg.server.port,
                "uptime_secs": state.started_at.elapsed().as_secs(),
                "api_auth_enabled": cfg.server.api_key.as_deref().is_some_and(|s| !s.trim().is_empty()),
                "cors_enabled": cfg.server.cors,
            },
            "dashboard": {
                "host": cfg.dashboard_host(),
                "port": cfg.dashboard_port(),
                "separate_listener": cfg.dashboard_uses_separate_listener(),
            },
            "config": {
                "chain_proxy": chain_proxy,
                "timeout_secs": cfg.opencode.timeout_secs,
                "strategy": pool_cfg.selection.strategy,
                "hedge_delay_ms": pool_cfg.selection.hedge_delay_ms,
                "failover_timeout_secs": pool_cfg.selection.failover_timeout_secs,
                "target_healthy": pool_cfg.health_check.target_healthy,
                "fallback_to_base": pool_cfg.selection.fallback_to_base,
            },
            "metrics": metrics,
            "proxy_pool": pool_stats_json,
            "proxies": proxies,
            "models": models,
        }),
    )
}

#[derive(Debug, Deserialize)]
struct ProxiesQuery {
    status: Option<String>,
    q: Option<String>,
    limit: Option<usize>,
}

async fn api_proxies(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ProxiesQuery>,
) -> Response {
    let Some(pool) = &state.proxy_pool else {
        return json_response(
            StatusCode::OK,
            json!({
                "stats": Value::Null,
                "total_matching": 0,
                "proxies": [],
            }),
        );
    };

    let status_filter = query
        .status
        .as_deref()
        .map(str::trim)
        .unwrap_or("all")
        .to_ascii_lowercase();
    let search = query
        .q
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_default();
    let limit = query.limit.unwrap_or(200).clamp(1, 2000);

    let all = pool.snapshot_proxies(usize::MAX);
    let filtered: Vec<ProxyInfo> = all
        .into_iter()
        .filter(|p| match status_filter.as_str() {
            "healthy" => p.status == "healthy",
            "cooldown" => p.status.starts_with("cooldown"),
            "untested" => p.status == "untested",
            "dead" => p.status == "dead",
            _ => true,
        })
        .filter(|p| {
            if search.is_empty() {
                return true;
            }
            p.proxy.to_ascii_lowercase().contains(&search)
                || p.protocol.to_ascii_lowercase().contains(&search)
                || p.status.to_ascii_lowercase().contains(&search)
                || p.last_error
                    .as_deref()
                    .is_some_and(|e| e.to_ascii_lowercase().contains(&search))
        })
        .collect();

    let total_matching = filtered.len();
    let proxies: Vec<ProxyInfo> = filtered.into_iter().take(limit).collect();

    let mut stats_val = serde_json::to_value(pool.stats()).unwrap_or(Value::Null);
    let earliest = pool.earliest_cooldown_remaining().map(|d| d.as_secs());
    if let Some(obj) = stats_val.as_object_mut() {
        obj.insert(
            "earliest_cooldown_secs".to_string(),
            earliest.map_or(Value::Null, |s| json!(s)),
        );
    }

    json_response(
        StatusCode::OK,
        json!({
            "stats": stats_val,
            "total_matching": total_matching,
            "proxies": proxies,
        }),
    )
}

async fn api_proxies_probe(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Some(resp) = check_same_origin_post(&headers) {
        return resp;
    }
    let Some(pool) = &state.proxy_pool else {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"error": "Proxy pool is disabled"}),
        );
    };
    pool.trigger_probe();
    json_response(
        StatusCode::OK,
        json!({
            "ok": true,
            "message": "Probe wave triggered",
            "stats": pool.stats(),
        }),
    )
}

async fn api_proxies_refresh(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Some(resp) = check_same_origin_post(&headers) {
        return resp;
    }
    let Some(pool) = &state.proxy_pool else {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"error": "Proxy pool is disabled"}),
        );
    };
    match fetch_all_proxies(&state.base_client, &state.config.opencode.proxy_pool).await {
        Ok(loaded) => {
            let summary = pool.replace_proxies(loaded.proxies);
            json_response(
                StatusCode::OK,
                json!({
                    "ok": true,
                    "sources_succeeded": loaded.sources_succeeded,
                    "sources_failed": loaded.sources_failed,
                    "replace": {
                        "total": summary.total,
                        "kept": summary.kept,
                        "added": summary.added,
                        "removed": summary.removed,
                        "aborted_empty": summary.aborted_empty,
                    },
                    "stats": pool.stats(),
                }),
            )
        }
        Err(err) => json_response(
            StatusCode::BAD_GATEWAY,
            json!({"error": format!("Failed to refresh proxy lists: {err}")}),
        ),
    }
}

fn json_response(status: StatusCode, body: Value) -> Response {
    let mut resp = (status, Json(body)).into_response();
    apply_dashboard_security_headers(resp.headers_mut(), None);
    resp
}

fn apply_dashboard_security_headers(headers: &mut HeaderMap, csp: Option<&str>) {
    headers.insert(
        HeaderName::from_static("x-frame-options"),
        HeaderValue::from_static("DENY"),
    );
    headers.insert(
        HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        HeaderName::from_static("referrer-policy"),
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );
    headers.insert(
        HeaderName::from_static("permissions-policy"),
        HeaderValue::from_static(
            "camera=(), geolocation=(), microphone=(), focus-without-user-activation=()",
        ),
    );
    headers.insert(
        HeaderName::from_static("cross-origin-opener-policy"),
        HeaderValue::from_static("same-origin-allow-popups"),
    );
    headers.insert(
        HeaderName::from_static("cross-origin-resource-policy"),
        HeaderValue::from_static("same-origin"),
    );
    headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(
        VARY,
        HeaderValue::from_static("Sec-Fetch-Dest, Sec-Fetch-Mode, Sec-Fetch-Site"),
    );
    if let Some(policy) = csp
        && let Ok(val) = HeaderValue::from_str(policy)
    {
        headers.insert(HeaderName::from_static("content-security-policy"), val);
    }
}

fn check_same_origin_post(headers: &HeaderMap) -> Option<Response> {
    if let Some(site) = headers
        .get("sec-fetch-site")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_ascii_lowercase())
        && !matches!(site.as_str(), "same-origin" | "none")
    {
        return Some(json_response(
            StatusCode::FORBIDDEN,
            json!({"error": "Cross-site POST forbidden"}),
        ));
    }
    None
}

fn is_https_request(headers: &HeaderMap) -> bool {
    headers
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("https"))
}

fn parse_login_payload(headers: &HeaderMap, body: &[u8]) -> Option<(String, String)> {
    let ct = headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();

    if ct.contains("application/x-www-form-urlencoded") {
        let mut user = None;
        let mut pass = None;
        for (k, v) in url::form_urlencoded::parse(body) {
            if k == "username" {
                user = Some(v.into_owned());
            } else if k == "password" {
                pass = Some(v.into_owned());
            }
        }
        return Some((user?, pass?));
    }

    let parsed: Value = serde_json::from_slice(body).ok()?;
    let user = parsed.get("username")?.as_str()?.to_string();
    let pass = parsed.get("password")?.as_str()?.to_string();
    Some((user, pass))
}

fn extract_session_cookie(headers: &HeaderMap) -> Option<&str> {
    let raw = headers.get(COOKIE)?.to_str().ok()?;
    for part in raw.split(';') {
        let trimmed = part.trim();
        if let Some((name, val)) = trimmed.split_once('=')
            && name.trim() == SESSION_COOKIE_NAME
        {
            return Some(val.trim());
        }
    }
    None
}

fn extract_basic_auth(headers: &HeaderMap) -> Option<(String, String)> {
    let auth = headers.get(AUTHORIZATION)?.to_str().ok()?.trim();
    let encoded = auth
        .strip_prefix("Basic ")
        .or_else(|| auth.strip_prefix("basic "))?
        .trim();
    let decoded = decode_base64(encoded)?;
    let text = String::from_utf8(decoded).ok()?;
    let (u, p) = text.split_once(':')?;
    Some((u.to_string(), p.to_string()))
}

fn decode_base64(input: &str) -> Option<Vec<u8>> {
    let clean: Vec<u8> = input
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();
    if !clean.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity((clean.len() / 4) * 3);
    let (chunks, _) = clean.as_chunks::<4>();
    for chunk in chunks {
        let v0 = b64_val(chunk[0])?;
        let v1 = b64_val(chunk[1])?;
        let pad2 = chunk[2] == b'=';
        let pad3 = chunk[3] == b'=';
        if pad2 && !pad3 {
            return None;
        }
        let v2 = if pad2 { 0 } else { b64_val(chunk[2])? };
        let v3 = if pad3 { 0 } else { b64_val(chunk[3])? };
        let triple =
            ((v0 as u32) << 18) | ((v1 as u32) << 12) | ((v2 as u32) << 6) | (v3 as u32);
        out.push(((triple >> 16) & 0xFF) as u8);
        if !pad2 {
            out.push(((triple >> 8) & 0xFF) as u8);
        }
        if !pad3 {
            out.push((triple & 0xFF) as u8);
        }
    }
    Some(out)
}

const fn b64_val(b: u8) -> Option<u8> {
    match b {
        b'A'..=b'Z' => Some(b - b'A'),
        b'a'..=b'z' => Some(b - b'a' + 26),
        b'0'..=b'9' => Some(b - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let max_len = a.len().max(b.len());
    let mut diff = (a.len() ^ b.len()) as u8;
    for i in 0..max_len {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= x ^ y;
    }
    diff == 0
}
