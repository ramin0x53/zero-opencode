use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{SystemTime, UNIX_EPOCH};

use futures_util::Stream;
use serde::Serialize;

const MAX_ERROR_LEN: usize = 160;

/// Single recorded API request entry (never stores prompts, responses, or credentials).
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RequestRecord {
    pub id: u64,
    pub timestamp_ms: u64,
    pub method: String,
    pub path: String,
    pub model: Option<String>,
    pub stream: Option<bool>,
    pub status: u16,
    pub duration_ms: u64,
    pub proxy: Option<String>,
    pub error: Option<String>,
}

/// Input payload for recording a completed (or stream-started) API request.
#[derive(Debug, Clone)]
pub struct NewRequestRecord {
    pub method: &'static str,
    pub path: String,
    pub model: Option<String>,
    pub stream: Option<bool>,
    pub status: u16,
    pub duration_ms: u64,
    pub proxy: Option<String>,
    pub error: Option<String>,
    pub is_chat: bool,
    pub is_models: bool,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug, Clone, Default)]
struct ModelMetricsAcc {
    requests: u64,
    successes: u64,
    errors: u64,
    total_latency_ms: u64,
    last_used_ms: u64,
}

/// Aggregated per-model request statistics.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelMetricsSummary {
    pub model: String,
    pub requests: u64,
    pub successes: u64,
    pub errors: u64,
    pub avg_latency_ms: u64,
    pub last_used_ms: u64,
}

/// Global request and token counters.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MetricsSummary {
    pub total_requests: u64,
    pub chat_requests: u64,
    pub models_requests: u64,
    pub streaming_requests: u64,
    pub non_streaming_requests: u64,
    pub success_requests: u64,
    pub client_error_requests: u64,
    pub server_error_requests: u64,
    pub in_flight_requests: u64,
    pub avg_latency_ms: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// Point-in-time metrics & recent request snapshot for the dashboard API.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub summary: MetricsSummary,
    pub models: Vec<ModelMetricsSummary>,
    pub recent_requests: Vec<RequestRecord>,
}

#[derive(Debug)]
struct MetricsInner {
    max_history: usize,
    recent: VecDeque<RequestRecord>,
    by_model: HashMap<String, ModelMetricsAcc>,
}

/// Thread-safe in-memory metrics and bounded recent request ring buffer.
pub struct MetricsStore {
    next_id: AtomicU64,
    total_requests: AtomicU64,
    chat_requests: AtomicU64,
    models_requests: AtomicU64,
    streaming_requests: AtomicU64,
    non_streaming_requests: AtomicU64,
    success_requests: AtomicU64,
    client_error_requests: AtomicU64,
    server_error_requests: AtomicU64,
    in_flight_requests: AtomicU64,
    total_latency_ms: AtomicU64,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    inner: Mutex<MetricsInner>,
}

impl MetricsStore {
    pub fn new(max_history: usize) -> Arc<Self> {
        Arc::new(Self {
            next_id: AtomicU64::new(1),
            total_requests: AtomicU64::new(0),
            chat_requests: AtomicU64::new(0),
            models_requests: AtomicU64::new(0),
            streaming_requests: AtomicU64::new(0),
            non_streaming_requests: AtomicU64::new(0),
            success_requests: AtomicU64::new(0),
            client_error_requests: AtomicU64::new(0),
            server_error_requests: AtomicU64::new(0),
            in_flight_requests: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
            inner: Mutex::new(MetricsInner {
                max_history: max_history.max(1),
                recent: VecDeque::with_capacity(max_history.min(256)),
                by_model: HashMap::new(),
            }),
        })
    }

    /// Increment `in_flight_requests` and decrement automatically when the guard drops.
    pub fn start_in_flight(self: &Arc<Self>) -> InFlightGuard {
        self.in_flight_requests.fetch_add(1, Ordering::Relaxed);
        InFlightGuard {
            store: Arc::clone(self),
        }
    }

    pub fn record_request(&self, entry: NewRequestRecord) {
        let now_ms = current_time_ms();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        self.total_requests.fetch_add(1, Ordering::Relaxed);
        if entry.is_chat {
            self.chat_requests.fetch_add(1, Ordering::Relaxed);
        }
        if entry.is_models {
            self.models_requests.fetch_add(1, Ordering::Relaxed);
        }
        match entry.stream {
            Some(true) => {
                self.streaming_requests.fetch_add(1, Ordering::Relaxed);
            }
            Some(false) => {
                self.non_streaming_requests.fetch_add(1, Ordering::Relaxed);
            }
            None => {}
        }

        if (200..300).contains(&entry.status) {
            self.success_requests.fetch_add(1, Ordering::Relaxed);
        } else if (400..500).contains(&entry.status) {
            self.client_error_requests.fetch_add(1, Ordering::Relaxed);
        } else if entry.status >= 500 {
            self.server_error_requests.fetch_add(1, Ordering::Relaxed);
        }

        self.total_latency_ms
            .fetch_add(entry.duration_ms, Ordering::Relaxed);
        if entry.prompt_tokens > 0 {
            self.prompt_tokens
                .fetch_add(entry.prompt_tokens, Ordering::Relaxed);
        }
        if entry.completion_tokens > 0 {
            self.completion_tokens
                .fetch_add(entry.completion_tokens, Ordering::Relaxed);
        }

        let sanitized_err = entry.error.map(|e| truncate_error(&e, MAX_ERROR_LEN));
        let model_name = entry.model.clone();

        let rec = RequestRecord {
            id,
            timestamp_ms: now_ms,
            method: entry.method.to_string(),
            path: entry.path,
            model: entry.model,
            stream: entry.stream,
            status: entry.status,
            duration_ms: entry.duration_ms,
            proxy: entry.proxy,
            error: sanitized_err,
        };

        let mut guard = self.inner.lock().expect("MetricsStore lock");
        if guard.recent.len() >= guard.max_history {
            guard.recent.pop_back();
        }
        guard.recent.push_front(rec);

        if entry.is_chat
            && let Some(model) = model_name
        {
            let acc = guard.by_model.entry(model).or_default();
            acc.requests += 1;
            if (200..300).contains(&entry.status) {
                acc.successes += 1;
            } else {
                acc.errors += 1;
            }
            acc.total_latency_ms = acc.total_latency_ms.saturating_add(entry.duration_ms);
            acc.last_used_ms = now_ms;
        }
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ms.load(Ordering::Relaxed);
        let prompt_tokens = self.prompt_tokens.load(Ordering::Relaxed);
        let completion_tokens = self.completion_tokens.load(Ordering::Relaxed);

        let summary = MetricsSummary {
            total_requests,
            chat_requests: self.chat_requests.load(Ordering::Relaxed),
            models_requests: self.models_requests.load(Ordering::Relaxed),
            streaming_requests: self.streaming_requests.load(Ordering::Relaxed),
            non_streaming_requests: self.non_streaming_requests.load(Ordering::Relaxed),
            success_requests: self.success_requests.load(Ordering::Relaxed),
            client_error_requests: self.client_error_requests.load(Ordering::Relaxed),
            server_error_requests: self.server_error_requests.load(Ordering::Relaxed),
            in_flight_requests: self.in_flight_requests.load(Ordering::Relaxed),
            avg_latency_ms: total_latency.checked_div(total_requests).unwrap_or(0),
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        };

        let guard = self.inner.lock().expect("MetricsStore lock");
        let mut models: Vec<ModelMetricsSummary> = guard
            .by_model
            .iter()
            .map(|(model, acc)| ModelMetricsSummary {
                model: model.clone(),
                requests: acc.requests,
                successes: acc.successes,
                errors: acc.errors,
                avg_latency_ms: acc.total_latency_ms.checked_div(acc.requests).unwrap_or(0),
                last_used_ms: acc.last_used_ms,
            })
            .collect();
        models.sort_by(|a, b| b.requests.cmp(&a.requests).then_with(|| a.model.cmp(&b.model)));

        let recent_requests: Vec<RequestRecord> = guard.recent.iter().cloned().collect();

        MetricsSnapshot {
            summary,
            models,
            recent_requests,
        }
    }
}

/// RAII guard tracking open requests (including active SSE streams).
pub struct InFlightGuard {
    store: Arc<MetricsStore>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.store.in_flight_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Stream wrapper keeping an `InFlightGuard` alive until the SSE stream ends or drops.
pub struct GuardedStream<S> {
    inner: S,
    _guard: InFlightGuard,
}

impl<S> GuardedStream<S> {
    pub fn new(inner: S, guard: InFlightGuard) -> Self {
        Self {
            inner,
            _guard: guard,
        }
    }
}

impl<S: Stream + Unpin> Stream for GuardedStream<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn truncate_error(s: &str, max: usize) -> String {
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
    use futures_util::StreamExt;

    #[test]
    fn bounds_ring_buffer_and_aggregates_per_model() {
        let store = MetricsStore::new(2);
        for i in 1..=3 {
            store.record_request(NewRequestRecord {
                method: "POST",
                path: "/v1/chat/completions".to_string(),
                model: Some(if i < 3 {
                    "big-pickle".to_string()
                } else {
                    "mimo-v2.6-flash-free".to_string()
                }),
                stream: Some(i == 3),
                status: if i == 2 { 503 } else { 200 },
                duration_ms: i * 100,
                proxy: Some("socks5://1.2.3.4:1080".to_string()),
                error: if i == 2 {
                    Some("pool exhausted".to_string())
                } else {
                    None
                },
                is_chat: true,
                is_models: false,
                prompt_tokens: 10,
                completion_tokens: 5,
            });
        }

        let snap = store.snapshot();
        assert_eq!(snap.summary.total_requests, 3);
        assert_eq!(snap.summary.chat_requests, 3);
        assert_eq!(snap.summary.success_requests, 2);
        assert_eq!(snap.summary.server_error_requests, 1);
        assert_eq!(snap.summary.streaming_requests, 1);
        assert_eq!(snap.summary.non_streaming_requests, 2);
        assert_eq!(snap.summary.avg_latency_ms, 200);
        assert_eq!(snap.summary.total_tokens, 45);
        // Ring buffer capped at 2, newest first
        assert_eq!(snap.recent_requests.len(), 2);
        assert_eq!(snap.recent_requests[0].id, 3);
        assert_eq!(snap.recent_requests[1].id, 2);
        // Per-model summary sorted descending by request count
        assert_eq!(snap.models.len(), 2);
        assert_eq!(snap.models[0].model, "big-pickle");
        assert_eq!(snap.models[0].requests, 2);
        assert_eq!(snap.models[0].successes, 1);
        assert_eq!(snap.models[0].errors, 1);
    }

    #[tokio::test]
    async fn guarded_stream_decrements_in_flight_after_completion() {
        let store = MetricsStore::new(10);
        let guard = store.start_in_flight();
        assert_eq!(store.snapshot().summary.in_flight_requests, 1);

        let inner = futures_util::stream::iter(vec![Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"hi"))]);
        let mut guarded = GuardedStream::new(inner, guard);
        assert_eq!(store.snapshot().summary.in_flight_requests, 1);
        let _ = guarded.next().await;
        drop(guarded);
        assert_eq!(store.snapshot().summary.in_flight_requests, 0);
    }
}
