use std::collections::VecDeque;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use http::HeaderMap;
use regex::Regex;
use serde_json::Value;
use sha2::{Digest, Sha256};

const MAX_SESSION_LENGTH: usize = 256;
const MAX_TRANSLATED_SESSIONS: usize = 5000;
const MAX_STABLE_SESSIONS: usize = 1000;
const ASSISTANT_MIN_LEN: usize = 50;
const ASSISTANT_CAP_LEN: usize = 50;
const SESSION_PREFIX_MASK: u64 = (1u64 << 48) - 1;
const BASE62_CHARS: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

pub const OPENCODE_SESSION_HEADER: &str = "x-opencode-session";

static OPENCODE_SESSION_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$").expect("valid regex"));

static CLAUDE_CODE_SESSION_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"_session_([a-f0-9-]+)$").expect("valid regex"));

const SESSION_HEADER_KEYS: &[&str] = &[
    "x-session-id",
    "session-id",
    "session_id",
    "x-amp-thread-id",
    "x-client-request-id",
];

/// Result of resolving an upstream OpenCode session ID for a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSession {
    pub id: String,
    /// True when the caller supplied a valid `ses_...` header directly.
    pub is_native: bool,
}

#[derive(Debug, Clone)]
struct SessionEntry {
    key: String,
    session_id: String,
    last_used: Instant,
}

/// Simple bounded LRU+TTL store preserving insertion/access order.
#[derive(Debug)]
struct SessionCache {
    max_entries: usize,
    ttl: Duration,
    entries: VecDeque<SessionEntry>,
}

impl SessionCache {
    fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            max_entries,
            ttl,
            entries: VecDeque::new(),
        }
    }

    fn evict_expired(&mut self, now: Instant) {
        self.entries
            .retain(|entry| now.saturating_duration_since(entry.last_used) <= self.ttl);
    }

    fn get_and_touch(&mut self, key: &str, now: Instant) -> Option<String> {
        self.evict_expired(now);
        let idx = self.entries.iter().position(|e| e.key == key)?;
        let mut entry = self.entries.remove(idx)?;
        entry.last_used = now;
        let id = entry.session_id.clone();
        self.entries.push_back(entry);
        Some(id)
    }

    fn insert(&mut self, key: String, session_id: String, now: Instant) {
        self.evict_expired(now);
        if let Some(idx) = self.entries.iter().position(|e| e.key == key) {
            self.entries.remove(idx);
        }
        while self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(SessionEntry {
            key,
            session_id,
            last_used: now,
        });
    }

    fn remove_by_session_id(&mut self, session_id: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.session_id != session_id);
        self.entries.len() < before
    }
}

#[derive(Debug, Default)]
struct GeneratorState {
    last_timestamp_ms: u64,
    counter: u64,
}

/// Manages OpenCode free-tier `ses_...` session generation, caching, and invalidation.
pub struct SessionManager {
    gen_state: Mutex<GeneratorState>,
    translated: Mutex<SessionCache>,
    stable: Mutex<SessionCache>,
}

impl SessionManager {
    pub fn new(ttl: Duration) -> Self {
        Self {
            gen_state: Mutex::new(GeneratorState::default()),
            translated: Mutex::new(SessionCache::new(MAX_TRANSLATED_SESSIONS, ttl)),
            stable: Mutex::new(SessionCache::new(MAX_STABLE_SESSIONS, ttl)),
        }
    }

    /// Generate a new `ses_<12 hex><14 base62>` session ID at the current wall clock time.
    pub fn generate_session_id(&self) -> String {
        self.generate_session_id_at(current_time_ms())
    }

    /// Generate a new `ses_<12 hex><14 base62>` session ID for a specific millisecond timestamp.
    pub fn generate_session_id_at(&self, timestamp_ms: u64) -> String {
        let counter_val = {
            let mut state = self.gen_state.lock().expect("session generator lock");
            if timestamp_ms != state.last_timestamp_ms {
                state.last_timestamp_ms = timestamp_ms;
                state.counter = 0;
            }
            // Upstream rejects large counters (>=100); keep within 1..=16 even under bursts.
            if state.counter >= 16 {
                state.counter = 0;
            }
            state.counter += 1;
            state.counter
        };
        format!(
            "ses_{}{}",
            descending_time_hex(timestamp_ms, counter_val),
            unstable_random()
        )
    }

    /// Translate an arbitrary client session key into a memoized, current-time `ses_...` ID.
    pub fn translate_session_id(&self, hint: &str) -> String {
        let trimmed = hint.trim();
        if is_valid_opencode_session(trimmed) {
            return trimmed.to_string();
        }
        let key = format!("generic\0{trimmed}");
        let now = Instant::now();
        let mut cache = self.translated.lock().expect("translated sessions lock");
        if let Some(existing) = cache.get_and_touch(&key, now) {
            return existing;
        }
        let session_id = format!(
            "ses_{}{}",
            descending_time_hex(current_time_ms(), 1),
            unstable_random()
        );
        cache.insert(key, session_id.clone(), now);
        session_id
    }

    /// Return or mint a stable session ID for a downstream caller identity.
    pub fn stable_session_id(&self, headers: &HeaderMap) -> String {
        let key = identity_key(headers);
        let now = Instant::now();
        let mut cache = self.stable.lock().expect("stable sessions lock");
        if let Some(existing) = cache.get_and_touch(&key, now) {
            return existing;
        }
        drop(cache);

        let session_id = self.generate_session_id();
        let mut cache = self.stable.lock().expect("stable sessions lock");
        if let Some(existing) = cache.get_and_touch(&key, now) {
            return existing;
        }
        cache.insert(key, session_id.clone(), now);
        session_id
    }

    /// Resolve the upstream OpenCode session ID for an incoming request.
    pub fn resolve_session(&self, headers: &HeaderMap, body: &Value) -> ResolvedSession {
        if let Some(incoming) = header_value(headers, OPENCODE_SESSION_HEADER) {
            if is_valid_opencode_session(&incoming) {
                return ResolvedSession {
                    id: incoming,
                    is_native: true,
                };
            }
            return ResolvedSession {
                id: self.translate_session_id(&incoming),
                is_native: false,
            };
        }

        if let Some(hint) = extract_client_session_hint(headers, body) {
            return ResolvedSession {
                id: self.translate_session_id(&hint),
                is_native: false,
            };
        }

        ResolvedSession {
            id: self.stable_session_id(headers),
            is_native: false,
        }
    }

    /// Drop a rejected/spent minted session ID by value so the next request re-mints.
    /// Never affects native client-supplied sessions because they are not stored here.
    pub fn forget_session(&self, session_id: &str) -> bool {
        if session_id.is_empty() {
            return false;
        }
        let mut removed = false;
        {
            let mut stable = self.stable.lock().expect("stable sessions lock");
            removed |= stable.remove_by_session_id(session_id);
        }
        {
            let mut translated = self.translated.lock().expect("translated sessions lock");
            removed |= translated.remove_by_session_id(session_id);
        }
        removed
    }
}

pub fn is_valid_opencode_session(value: &str) -> bool {
    OPENCODE_SESSION_RE.is_match(value.trim())
}

fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn descending_time_hex(time_ms: u64, counter_value: u64) -> String {
    let combined = time_ms.wrapping_mul(0x1000).wrapping_add(counter_value);
    let inverted = (!combined) & SESSION_PREFIX_MASK;
    format!("{inverted:012x}")
}

fn unstable_random() -> String {
    let mut out = String::with_capacity(14);
    for _ in 0..14 {
        let idx = rand::random_range(0..62);
        out.push(BASE62_CHARS[idx] as char);
    }
    out
}

fn normalize_session(raw: Option<&str>) -> Option<String> {
    let v = raw?.trim();
    if v.is_empty() || v.len() > MAX_SESSION_LENGTH {
        return None;
    }
    Some(v.to_string())
}

fn header_value(headers: &HeaderMap, key: &str) -> Option<String> {
    let val = headers.get(key)?.to_str().ok()?;
    normalize_session(Some(val))
}

fn identity_key(headers: &HeaderMap) -> String {
    let auth = headers
        .get("authorization")
        .or_else(|| headers.get("x-api-key"))
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !auth.is_empty() {
        let digest = hex::encode(Sha256::digest(auth.as_bytes()));
        return format!("opencode:auth:{}", &digest[..32]);
    }
    "opencode:default".to_string()
}

fn extract_claude_code_session(user_id: Option<&str>) -> Option<String> {
    let user_id = user_id?.trim();
    if user_id.is_empty() {
        return None;
    }
    if let Some(caps) = CLAUDE_CODE_SESSION_RE.captures(user_id) {
        return normalize_session(caps.get(1).map(|m| m.as_str()));
    }
    if user_id.starts_with('{')
        && let Ok(val) = serde_json::from_str::<Value>(user_id)
        && let Some(sid) = val.get("session_id").and_then(Value::as_str)
    {
        return normalize_session(Some(sid));
    }
    None
}

fn extract_client_session_hint(headers: &HeaderMap, body: &Value) -> Option<String> {
    let meta_user_id = body
        .get("metadata")
        .and_then(|m| m.get("user_id"))
        .and_then(Value::as_str);

    if let Some(claude) = extract_claude_code_session(meta_user_id)
        .or_else(|| header_value(headers, "x-claude-code-session-id"))
    {
        return Some(format!("claude:{claude}"));
    }

    for key in SESSION_HEADER_KEYS {
        if let Some(v) = header_value(headers, key) {
            return Some(v);
        }
    }

    for field in ["prompt_cache_key", "session_id", "conversation_id"] {
        if let Some(v) = normalize_session(body.get(field).and_then(Value::as_str)) {
            return Some(v);
        }
    }

    if let Some(v) = normalize_session(meta_user_id) {
        return Some(v);
    }

    assistant_text_hint(body)
}

fn assistant_text_hint(body: &Value) -> Option<String> {
    let items = body
        .get("messages")
        .and_then(Value::as_array)
        .or_else(|| body.get("input").and_then(Value::as_array))?;

    let mut text = String::new();
    for item in items {
        if item.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        match item.get("content") {
            Some(Value::String(s)) => text.push_str(s),
            Some(Value::Array(parts)) => {
                for part in parts {
                    if let Some(s) = part
                        .get("text")
                        .or_else(|| part.get("output"))
                        .and_then(Value::as_str)
                    {
                        text.push_str(s);
                    }
                }
            }
            _ => {}
        }
        if text.chars().count() >= ASSISTANT_CAP_LEN {
            break;
        }
    }

    if text.chars().count() < ASSISTANT_MIN_LEN {
        return None;
    }

    let prefix: String = text.chars().take(ASSISTANT_CAP_LEN).collect();
    let digest = hex::encode(Sha256::digest(format!("opencode:{prefix}").as_bytes()));
    Some(format!("assistant:{}", &digest[..16]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;
    use serde_json::json;

    fn counter_of(id: &str) -> u64 {
        let hex_part = &id[4..16];
        let val = u64::from_str_radix(hex_part, 16).expect("valid hex");
        (!val) & 0xfff
    }

    fn time_of(id: &str) -> u64 {
        let hex_part = &id[4..16];
        let val = u64::from_str_radix(hex_part, 16).expect("valid hex");
        ((!val) & SESSION_PREFIX_MASK) >> 12
    }

    #[test]
    fn generates_canonical_30_char_session_ids() {
        let mgr = SessionManager::new(Duration::from_secs(7200));
        for _ in 0..25 {
            let id = mgr.generate_session_id();
            assert_eq!(id.len(), 30);
            assert!(is_valid_opencode_session(&id));
            assert!((1..=16).contains(&counter_of(&id)));
        }
    }

    #[test]
    fn keeps_counter_small_even_under_bursts_in_same_ms() {
        let mgr = SessionManager::new(Duration::from_secs(7200));
        for _ in 0..40 {
            let id = mgr.generate_session_id_at(1_700_000_000_000);
            assert!(is_valid_opencode_session(&id));
            assert!((1..=16).contains(&counter_of(&id)));
        }
    }

    #[test]
    fn translates_with_current_time_and_memoizes() {
        let mgr = SessionManager::new(Duration::from_secs(7200));
        let mask_36 = (1u64 << 36) - 1;
        let before = current_time_ms() & mask_36;
        let first = mgr.translate_session_id("claude:550e8400-e29b-41d4-a716-446655440000");
        let after = current_time_ms() & mask_36;

        assert!(is_valid_opencode_session(&first));
        assert_eq!(counter_of(&first), 1);
        let decoded_time = time_of(&first);
        assert!(decoded_time >= before && decoded_time <= after);

        // Memoized reuse for same conversation, different for another
        assert_eq!(
            mgr.translate_session_id("claude:550e8400-e29b-41d4-a716-446655440000"),
            first
        );
        assert_ne!(mgr.translate_session_id("claude:other"), first);
    }

    #[test]
    fn preserves_valid_native_opencode_session_header() {
        let mgr = SessionManager::new(Duration::from_secs(7200));
        let valid = "ses_f534dfae8ffeCy4Ee4tLWNygDc";
        let mut headers = HeaderMap::new();
        headers.insert("X-OpenCode-Session", HeaderValue::from_str(&format!(" {valid} ")).unwrap());

        let resolved = mgr.resolve_session(&headers, &json!({}));
        assert_eq!(
            resolved,
            ResolvedSession {
                id: valid.to_string(),
                is_native: true
            }
        );

        // Native session is never forgotten
        assert!(!mgr.forget_session(valid));
    }

    #[test]
    fn translates_invalid_opencode_session_header() {
        let mgr = SessionManager::new(Duration::from_secs(7200));
        let mut headers = HeaderMap::new();
        headers.insert("x-opencode-session", HeaderValue::from_static("invalid-uuid"));

        let resolved = mgr.resolve_session(&headers, &json!({}));
        assert!(!resolved.is_native);
        assert_ne!(resolved.id, "invalid-uuid");
        assert!(is_valid_opencode_session(&resolved.id));
    }

    #[test]
    fn reuses_and_isolates_stable_sessions_by_auth_header() {
        let mgr = SessionManager::new(Duration::from_secs(7200));
        let mut h1 = HeaderMap::new();
        h1.insert("authorization", HeaderValue::from_static("Bearer key-a"));
        let mut h2 = HeaderMap::new();
        h2.insert("authorization", HeaderValue::from_static("Bearer key-b"));

        let a1 = mgr.resolve_session(&h1, &json!({"messages": []}));
        let a2 = mgr.resolve_session(&h1, &json!({"messages": []}));
        let b1 = mgr.resolve_session(&h2, &json!({"messages": []}));

        assert_eq!(a1.id, a2.id);
        assert_ne!(a1.id, b1.id);
        assert!(is_valid_opencode_session(&a1.id));
    }

    #[test]
    fn derives_stable_session_from_long_assistant_history() {
        let mgr = SessionManager::new(Duration::from_secs(7200));
        let long_text = "This assistant message is definitely longer than fifty characters in length.";
        let body1 = json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": long_text},
                {"role": "user", "content": "follow up"}
            ]
        });
        let body2 = json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": long_text},
                {"role": "user", "content": "second follow up"}
            ]
        });
        let s1 = mgr.resolve_session(&HeaderMap::new(), &body1);
        let s2 = mgr.resolve_session(&HeaderMap::new(), &body2);
        assert_eq!(s1.id, s2.id);
    }

    #[test]
    fn forget_session_drops_minted_id_by_value() {
        let mgr = SessionManager::new(Duration::from_secs(7200));
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer key-forget"));

        let first = mgr.resolve_session(&headers, &json!({}));
        assert!(mgr.forget_session(&first.id));
        assert!(!mgr.forget_session(&first.id));

        let second = mgr.resolve_session(&headers, &json!({}));
        assert_ne!(first.id, second.id);
        assert!(is_valid_opencode_session(&second.id));
    }
}
