use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::Notify;

use crate::config::{ProxyPoolConfig, ProxyStrategy};
use crate::proxy::chain::ChainBridge;
use crate::proxy::endpoint::{ParsedProxy, ProxyEndpoint, ProxyId, ProxyKey};

const MAX_TRANSPORT_BACKOFF_SECS: u64 = 900;

/// Why a proxy is temporarily placed on cooldown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CooldownReason {
    RateLimited,
    Forbidden,
    Overloaded,
    TransientError,
}

/// Lifecycle state of a proxy inside `ProxyPool`.
#[derive(Debug, Clone)]
pub enum ProxyState {
    Untested,
    Healthy {
        rtt_ms: u64,
        ewma_rtt_ms: u64,
        last_ok: Instant,
        ok_count: u64,
    },
    Cooldown {
        until: Instant,
        reason: CooldownReason,
        prev_rtt_ms: Option<u64>,
        last_error: String,
    },
    Dead {
        failed_at: Instant,
        last_error: String,
    },
}

#[derive(Debug, Clone)]
struct ProxyEntry {
    endpoint: ProxyEndpoint,
    state: ProxyState,
    last_checked: Option<Instant>,
    consecutive_failures: u32,
    backoff_level: u32,
}

/// Redacted view of a proxy entry for CLI display (`zero-opencode proxies`).
#[derive(Debug, Clone, Serialize)]
pub struct ProxyInfo {
    pub id: u64,
    pub proxy: String,
    pub protocol: String,
    pub status: &'static str,
    pub rtt_ms: Option<u64>,
    pub ewma_rtt_ms: Option<u64>,
    pub ok_count: u64,
    pub is_static: bool,
    pub cooldown_remaining_secs: Option<u64>,
    pub last_error: Option<String>,
}

/// Redacted summary exposed on `/health` and CLI status output.
#[derive(Debug, Clone, Serialize)]
pub struct ProxyPoolStats {
    pub enabled: bool,
    pub chained_via: Option<String>,
    pub hop1_healthy: bool,
    pub total: usize,
    pub healthy: usize,
    pub untested: usize,
    pub cooldown: usize,
    pub dead: usize,
    pub active_proxy: Option<String>,
    pub active_rtt_ms: Option<u64>,
    pub switches_total: u64,
    pub hedged_wins: u64,
}

/// Summary returned by `ProxyPool::replace_proxies`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplaceSummary {
    pub total: usize,
    pub kept: usize,
    pub added: usize,
    pub removed: usize,
    pub aborted_empty: bool,
}

#[derive(Debug)]
struct PoolInner {
    next_id: u64,
    entries: HashMap<ProxyId, ProxyEntry>,
    key_to_id: HashMap<ProxyKey, ProxyId>,
    order: Vec<ProxyId>,
    active_proxy: Option<ProxyId>,
    probing_inflight: HashSet<ProxyId>,
}

/// Thread-safe state machine managing proxy selection, cooldowns, replacement refreshes,
/// and prober/waiter coordination.
pub struct ProxyPool {
    cfg: ProxyPoolConfig,
    bridge: Option<Arc<ChainBridge>>,
    inner: Mutex<PoolInner>,
    pub needs_probe: Arc<Notify>,
    pub healthy_ready: Arc<Notify>,
    switches_total: AtomicU64,
    hedged_wins: AtomicU64,
    rr_cursor: AtomicU64,
}

impl ProxyPool {
    pub fn new(
        cfg: ProxyPoolConfig,
        initial_proxies: Vec<ParsedProxy>,
        bridge: Option<Arc<ChainBridge>>,
    ) -> Arc<Self> {
        let pool = Arc::new(Self {
            cfg,
            bridge,
            inner: Mutex::new(PoolInner {
                next_id: 1,
                entries: HashMap::new(),
                key_to_id: HashMap::new(),
                order: Vec::new(),
                active_proxy: None,
                probing_inflight: HashSet::new(),
            }),
            needs_probe: Arc::new(Notify::new()),
            healthy_ready: Arc::new(Notify::new()),
            switches_total: AtomicU64::new(0),
            hedged_wins: AtomicU64::new(0),
            rr_cursor: AtomicU64::new(0),
        });
        if !initial_proxies.is_empty() {
            pool.replace_proxies(initial_proxies);
        }
        pool
    }

    pub fn config(&self) -> &ProxyPoolConfig {
        &self.cfg
    }

    pub fn bridge(&self) -> Option<&Arc<ChainBridge>> {
        self.bridge.as_ref()
    }

    /// Replace the in-memory proxy list with `new_list`, dropping removed entries,
    /// carrying over existing health/cooldown state for overlapping proxies, and adding
    /// newly discovered proxies as `Untested`.
    pub fn replace_proxies(&self, new_list: Vec<ParsedProxy>) -> ReplaceSummary {
        let (summary, endpoints_for_bridge, should_probe, has_healthy) = {
            let mut guard = self.inner.lock().expect("ProxyPool lock");

            if new_list.is_empty() && !guard.entries.is_empty() {
                return ReplaceSummary {
                    total: guard.entries.len(),
                    kept: guard.entries.len(),
                    added: 0,
                    removed: 0,
                    aborted_empty: true,
                };
            }

            let mut old_entries = std::mem::take(&mut guard.entries);
            let old_key_to_id = std::mem::take(&mut guard.key_to_id);
            guard.order.clear();

            let mut new_entries = HashMap::with_capacity(new_list.len());
            let mut new_key_to_id = HashMap::with_capacity(new_list.len());
            let mut new_order = Vec::with_capacity(new_list.len());
            let mut kept = 0usize;
            let mut added = 0usize;

            for parsed in new_list {
                let key = parsed.key();
                if new_key_to_id.contains_key(&key) {
                    continue;
                }
                if let Some(&existing_id) = old_key_to_id.get(&key)
                    && let Some(mut existing) = old_entries.remove(&existing_id)
                {
                    existing.endpoint.is_static = parsed.is_static;
                    existing.endpoint.auth = parsed.auth;
                    new_key_to_id.insert(key, existing_id);
                    new_order.push(existing_id);
                    new_entries.insert(existing_id, existing);
                    kept += 1;
                    continue;
                }

                let id = ProxyId(guard.next_id);
                guard.next_id = guard.next_id.saturating_add(1);
                let ep = parsed.into_endpoint(id);
                new_key_to_id.insert(key, id);
                new_order.push(id);
                new_entries.insert(
                    id,
                    ProxyEntry {
                        endpoint: ep,
                        state: ProxyState::Untested,
                        last_checked: None,
                        consecutive_failures: 0,
                        backoff_level: 0,
                    },
                );
                added += 1;
            }

            let removed = old_entries.len();
            guard.entries = new_entries;
            guard.key_to_id = new_key_to_id;
            guard.order = new_order;
            let mut inflight = std::mem::take(&mut guard.probing_inflight);
            inflight.retain(|id| guard.entries.contains_key(id));
            guard.probing_inflight = inflight;

            if let Some(active_id) = guard.active_proxy {
                let still_healthy = guard
                    .entries
                    .get(&active_id)
                    .is_some_and(|e| matches!(e.state, ProxyState::Healthy { .. }));
                if !still_healthy {
                    guard.active_proxy = None;
                }
            }

            let healthy_count = guard
                .entries
                .values()
                .filter(|e| matches!(e.state, ProxyState::Healthy { .. }))
                .count();

            let endpoints: Vec<ProxyEndpoint> = if self.bridge.is_some() {
                guard.entries.values().map(|e| e.endpoint.clone()).collect()
            } else {
                Vec::new()
            };

            let summary = ReplaceSummary {
                total: guard.entries.len(),
                kept,
                added,
                removed,
                aborted_empty: false,
            };
            (
                summary,
                endpoints,
                healthy_count < self.cfg.health_check.target_healthy,
                healthy_count > 0,
            )
        };

        if let Some(bridge) = &self.bridge {
            bridge.replace_endpoints(endpoints_for_bridge);
        }
        if should_probe {
            self.needs_probe.notify_one();
        }
        if has_healthy {
            self.healthy_ready.notify_waiters();
        }
        summary
    }

    /// Promote any expired cooldowns (`RateLimited`/`Forbidden`/`TransientError` -> `Untested`,
    /// `Overloaded` -> `Healthy` with prior RTT) and return whether probing is needed.
    pub fn sweep_expired_cooldowns(&self) -> bool {
        let now = Instant::now();
        let (woke_probe, has_healthy) = {
            let mut guard = self.inner.lock().expect("ProxyPool lock");
            let changed = promote_expired_cooldowns(&mut guard, now);
            let healthy = guard
                .entries
                .values()
                .filter(|e| matches!(e.state, ProxyState::Healthy { .. }))
                .count();
            (
                changed && healthy < self.cfg.health_check.target_healthy,
                healthy > 0,
            )
        };
        if woke_probe {
            self.needs_probe.notify_one();
        }
        if has_healthy {
            self.healthy_ready.notify_waiters();
        }
        woke_probe
    }

    /// Select an eligible healthy proxy not in `excluded` according to `selection.strategy`.
    pub fn select_proxy(&self, excluded: &HashSet<ProxyId>) -> Option<ProxyEndpoint> {
        let now = Instant::now();
        let mut wake_probe = false;
        let picked = {
            let mut guard = self.inner.lock().expect("ProxyPool lock");
            if promote_expired_cooldowns(&mut guard, now) {
                wake_probe = true;
            }

            let healthy_ids: Vec<(ProxyId, u64)> = guard
                .entries
                .iter()
                .filter_map(|(&id, entry)| match entry.state {
                    ProxyState::Healthy { ewma_rtt_ms, .. } if !excluded.contains(&id) => {
                        Some((id, ewma_rtt_ms))
                    }
                    _ => None,
                })
                .collect();

            if healthy_ids.is_empty() {
                None
            } else {
                let chosen_id = match self.cfg.selection.strategy {
                    ProxyStrategy::Sticky => {
                        if let Some(active) = guard.active_proxy
                            && !excluded.contains(&active)
                            && guard
                                .entries
                                .get(&active)
                                .is_some_and(|e| matches!(e.state, ProxyState::Healthy { .. }))
                        {
                            active
                        } else {
                            let best = healthy_ids
                                .iter()
                                .min_by_key(|&&(id, ewma)| (ewma, id))
                                .expect("non-empty healthy_ids")
                                .0;
                            if guard.active_proxy.is_none() {
                                guard.active_proxy = Some(best);
                            }
                            best
                        }
                    }
                    ProxyStrategy::LowestLatency => {
                        healthy_ids
                            .iter()
                            .min_by_key(|&&(id, ewma)| (ewma, id))
                            .expect("non-empty healthy_ids")
                            .0
                    }
                    ProxyStrategy::RoundRobin => {
                        let mut sorted: Vec<ProxyId> =
                            healthy_ids.iter().map(|&(id, _)| id).collect();
                        sorted.sort_unstable();
                        let idx = (self.rr_cursor.fetch_add(1, Ordering::Relaxed) as usize)
                            % sorted.len();
                        sorted[idx]
                    }
                };
                guard.entries.get(&chosen_id).map(|e| e.endpoint.clone())
            }
        };

        if wake_probe {
            self.needs_probe.notify_one();
        }
        picked
    }

    /// Atomically promote a winning proxy as `active_proxy` (counts switch if changed).
    pub fn promote_active(&self, winner_id: ProxyId) {
        let mut guard = self.inner.lock().expect("ProxyPool lock");
        let is_healthy = guard
            .entries
            .get(&winner_id)
            .is_some_and(|e| matches!(e.state, ProxyState::Healthy { .. }));
        if !is_healthy {
            return;
        }
        if let Some(prev) = guard.active_proxy
            && prev != winner_id
        {
            self.switches_total.fetch_add(1, Ordering::Relaxed);
        }
        guard.active_proxy = Some(winner_id);
    }

    pub fn record_hedged_win(&self) {
        self.hedged_wins.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a successful request or Stage 2 probe, marking the proxy `Healthy`.
    pub fn report_success(&self, id: ProxyId, rtt_ms: u64) {
        let now = Instant::now();
        {
            let mut guard = self.inner.lock().expect("ProxyPool lock");
            guard.probing_inflight.remove(&id);
            let Some(entry) = guard.entries.get_mut(&id) else {
                return;
            };
            let (new_ewma, new_ok_count) = match entry.state {
                ProxyState::Healthy {
                    ewma_rtt_ms,
                    ok_count,
                    ..
                } => ((ewma_rtt_ms * 7 + rtt_ms * 3) / 10, ok_count + 1),
                ProxyState::Cooldown {
                    prev_rtt_ms: Some(prev),
                    ..
                } => ((prev * 7 + rtt_ms * 3) / 10, 1),
                _ => (rtt_ms, 1),
            };
            entry.state = ProxyState::Healthy {
                rtt_ms,
                ewma_rtt_ms: new_ewma,
                last_ok: now,
                ok_count: new_ok_count,
            };
            entry.last_checked = Some(now);
            entry.consecutive_failures = 0;
            entry.backoff_level = 0;
        }
        self.healthy_ready.notify_waiters();
    }

    /// Update liveness timestamp and RTT without clearing an active cooldown.
    pub fn report_liveness_ok(&self, id: ProxyId, rtt_ms: u64) {
        let now = Instant::now();
        let mut guard = self.inner.lock().expect("ProxyPool lock");
        guard.probing_inflight.remove(&id);
        let Some(entry) = guard.entries.get_mut(&id) else {
            return;
        };
        entry.last_checked = Some(now);
        if let ProxyState::Healthy {
            ewma_rtt_ms,
            ok_count,
            ..
        } = entry.state
        {
            entry.state = ProxyState::Healthy {
                rtt_ms,
                ewma_rtt_ms: (ewma_rtt_ms * 7 + rtt_ms * 3) / 10,
                last_ok: now,
                ok_count: ok_count + 1,
            };
        }
    }

    pub fn report_rate_limited(
        &self,
        id: ProxyId,
        retry_after: Option<Duration>,
        detail: impl Into<String>,
    ) {
        let cooldown = retry_after
            .unwrap_or_else(|| Duration::from_secs(self.cfg.selection.cooldown_rate_limit_secs));
        self.apply_cooldown(id, CooldownReason::RateLimited, cooldown, detail.into());
    }

    pub fn report_forbidden(&self, id: ProxyId, detail: impl Into<String>) {
        let cooldown = Duration::from_secs(self.cfg.selection.cooldown_forbidden_secs);
        self.apply_cooldown(id, CooldownReason::Forbidden, cooldown, detail.into());
    }

    pub fn report_overloaded(&self, id: ProxyId) {
        let cooldown = Duration::from_secs(self.cfg.selection.cooldown_overload_secs);
        self.apply_cooldown(
            id,
            CooldownReason::Overloaded,
            cooldown,
            "429 overloaded".to_string(),
        );
    }

    fn apply_cooldown(
        &self,
        id: ProxyId,
        reason: CooldownReason,
        duration: Duration,
        detail: String,
    ) {
        let now = Instant::now();
        let need_probe = {
            let mut guard = self.inner.lock().expect("ProxyPool lock");
            guard.probing_inflight.remove(&id);
            let Some(entry) = guard.entries.get_mut(&id) else {
                return;
            };
            let prev_rtt_ms = match entry.state {
                ProxyState::Healthy { ewma_rtt_ms, .. } => Some(ewma_rtt_ms),
                ProxyState::Cooldown { prev_rtt_ms, .. } => prev_rtt_ms,
                _ => None,
            };
            entry.state = ProxyState::Cooldown {
                until: now + duration,
                reason,
                prev_rtt_ms,
                last_error: detail,
            };
            entry.last_checked = Some(now);
            if guard.active_proxy == Some(id) {
                guard.active_proxy = None;
                self.switches_total.fetch_add(1, Ordering::Relaxed);
            }
            healthy_count_locked(&guard) < self.cfg.health_check.target_healthy
        };
        if need_probe {
            self.needs_probe.notify_one();
        }
    }

    /// Record a transport/timeout/5xx error; escalates to `Dead` after `max_consecutive_failures`.
    pub fn report_transport_error(&self, id: ProxyId, detail: impl Into<String>) {
        let now = Instant::now();
        let detail = detail.into();
        let need_probe = {
            let mut guard = self.inner.lock().expect("ProxyPool lock");
            guard.probing_inflight.remove(&id);
            let Some(entry) = guard.entries.get_mut(&id) else {
                return;
            };
            entry.consecutive_failures = entry.consecutive_failures.saturating_add(1);
            entry.backoff_level = entry.backoff_level.saturating_add(1);
            entry.last_checked = Some(now);

            if entry.consecutive_failures >= self.cfg.selection.max_consecutive_failures {
                entry.state = ProxyState::Dead {
                    failed_at: now,
                    last_error: detail,
                };
            } else {
                let shift = entry.backoff_level.saturating_sub(1).min(10);
                let delay_secs = self
                    .cfg
                    .selection
                    .cooldown_error_base_secs
                    .saturating_mul(1u64 << shift)
                    .min(MAX_TRANSPORT_BACKOFF_SECS);
                entry.state = ProxyState::Cooldown {
                    until: now + Duration::from_secs(delay_secs),
                    reason: CooldownReason::TransientError,
                    prev_rtt_ms: None,
                    last_error: detail,
                };
            }

            if guard.active_proxy == Some(id) {
                guard.active_proxy = None;
                self.switches_total.fetch_add(1, Ordering::Relaxed);
            }
            healthy_count_locked(&guard) < self.cfg.health_check.target_healthy
        };
        if need_probe {
            self.needs_probe.notify_one();
        }
    }

    /// Immediately mark a proxy `Dead` (e.g. probe failure, `407`, or HTML hijack).
    pub fn report_dead(&self, id: ProxyId, detail: impl Into<String>) {
        let now = Instant::now();
        let need_probe = {
            let mut guard = self.inner.lock().expect("ProxyPool lock");
            guard.probing_inflight.remove(&id);
            let Some(entry) = guard.entries.get_mut(&id) else {
                return;
            };
            entry.state = ProxyState::Dead {
                failed_at: now,
                last_error: detail.into(),
            };
            entry.last_checked = Some(now);
            if guard.active_proxy == Some(id) {
                guard.active_proxy = None;
                self.switches_total.fetch_add(1, Ordering::Relaxed);
            }
            healthy_count_locked(&guard) < self.cfg.health_check.target_healthy
        };
        if need_probe {
            self.needs_probe.notify_one();
        }
    }

    /// Release an in-flight probe lock without changing state (used if Hop 1 went down).
    pub fn release_probe(&self, id: ProxyId) {
        let mut guard = self.inner.lock().expect("ProxyPool lock");
        guard.probing_inflight.remove(&id);
    }

    /// Take up to `max` `Untested` proxies not currently in flight (prioritizing `static_proxies`).
    pub fn take_untested_batch(&self, max: usize) -> Vec<ProxyEndpoint> {
        let mut guard = self.inner.lock().expect("ProxyPool lock");
        let order = guard.order.clone();
        let mut static_batch = Vec::new();
        let mut remote_batch = Vec::new();

        for id in order {
            if static_batch.len() + remote_batch.len() >= max {
                break;
            }
            if guard.probing_inflight.contains(&id) {
                continue;
            }
            if let Some(entry) = guard.entries.get(&id)
                && matches!(entry.state, ProxyState::Untested)
            {
                if entry.endpoint.is_static {
                    static_batch.push(entry.endpoint.clone());
                } else {
                    remote_batch.push(entry.endpoint.clone());
                }
            }
        }

        static_batch.extend(remote_batch);
        static_batch.truncate(max);
        for ep in &static_batch {
            guard.probing_inflight.insert(ep.id);
        }
        static_batch
    }

    /// Return up to `n` fastest `Healthy` standbys that have been idle at least `min_idle`
    /// for lightweight keep-alive warming.
    pub fn top_standbys_for_keepalive(
        &self,
        n: usize,
        min_idle: Duration,
    ) -> Vec<ProxyEndpoint> {
        if n == 0 {
            return Vec::new();
        }
        let now = Instant::now();
        let mut guard = self.inner.lock().expect("ProxyPool lock");
        let mut candidates: Vec<(u64, ProxyEndpoint)> = guard
            .entries
            .iter()
            .filter_map(|(&id, entry)| {
                if guard.probing_inflight.contains(&id) {
                    return None;
                }
                match entry.state {
                    ProxyState::Healthy {
                        ewma_rtt_ms,
                        last_ok,
                        ..
                    } if now.saturating_duration_since(last_ok) >= min_idle => {
                        Some((ewma_rtt_ms, entry.endpoint.clone()))
                    }
                    _ => None,
                }
            })
            .collect();
        candidates.sort_by_key(|(ewma, ep)| (*ewma, ep.id));
        candidates.truncate(n);
        let out: Vec<ProxyEndpoint> = candidates.into_iter().map(|(_, ep)| ep).collect();
        for ep in &out {
            guard.probing_inflight.insert(ep.id);
        }
        out
    }

    /// Return up to `max` `Healthy` proxies whose last check is older than `min_idle`.
    pub fn stale_healthy_for_recheck(
        &self,
        min_idle: Duration,
        max: usize,
    ) -> Vec<ProxyEndpoint> {
        let now = Instant::now();
        let mut guard = self.inner.lock().expect("ProxyPool lock");
        let mut out = Vec::new();
        for (&id, entry) in &guard.entries {
            if out.len() >= max {
                break;
            }
            if guard.probing_inflight.contains(&id) {
                continue;
            }
            if let ProxyState::Healthy { last_ok, .. } = entry.state {
                let checked = entry.last_checked.unwrap_or(last_ok);
                if now.saturating_duration_since(checked) >= min_idle {
                    out.push(entry.endpoint.clone());
                }
            }
        }
        for ep in &out {
            guard.probing_inflight.insert(ep.id);
        }
        out
    }

    /// If pool is completely dry (`healthy == 0 && untested == 0`), recycle `Dead` entries
    /// older than `min_age` back to `Untested` as a last-resort recovery.
    pub fn recycle_old_dead(&self, min_age: Duration) -> usize {
        let now = Instant::now();
        let recycled = {
            let mut guard = self.inner.lock().expect("ProxyPool lock");
            let has_ready_or_untested = guard.entries.values().any(|e| {
                matches!(
                    e.state,
                    ProxyState::Healthy { .. } | ProxyState::Untested
                )
            });
            if has_ready_or_untested {
                return 0;
            }
            let mut count = 0usize;
            for entry in guard.entries.values_mut() {
                if let ProxyState::Dead { failed_at, .. } = entry.state
                    && now.saturating_duration_since(failed_at) >= min_age
                {
                    entry.state = ProxyState::Untested;
                    entry.consecutive_failures = 0;
                    entry.backoff_level = 0;
                    count += 1;
                }
            }
            count
        };
        if recycled > 0 {
            self.needs_probe.notify_one();
        }
        recycled
    }

    pub fn trigger_probe(&self) {
        self.needs_probe.notify_one();
    }

    pub fn healthy_count(&self) -> usize {
        let guard = self.inner.lock().expect("ProxyPool lock");
        healthy_count_locked(&guard)
    }

    pub fn untested_count(&self) -> usize {
        let guard = self.inner.lock().expect("ProxyPool lock");
        guard
            .entries
            .values()
            .filter(|e| matches!(e.state, ProxyState::Untested))
            .count()
    }

    /// Check whether `id` is currently among the top `n` lowest-latency healthy proxies.
    pub fn is_top_hot_standby(&self, id: ProxyId, n: usize) -> bool {
        if n == 0 {
            return false;
        }
        let guard = self.inner.lock().expect("ProxyPool lock");
        let mut healthy: Vec<(u64, ProxyId)> = guard
            .entries
            .iter()
            .filter_map(|(&pid, e)| match e.state {
                ProxyState::Healthy { ewma_rtt_ms, .. } => Some((ewma_rtt_ms, pid)),
                _ => None,
            })
            .collect();
        healthy.sort_unstable();
        healthy.into_iter().take(n).any(|(_, pid)| pid == id)
    }

    pub fn earliest_cooldown_remaining(&self) -> Option<Duration> {
        let now = Instant::now();
        let guard = self.inner.lock().expect("ProxyPool lock");
        guard
            .entries
            .values()
            .filter_map(|e| match e.state {
                ProxyState::Cooldown { until, .. } => {
                    Some(until.saturating_duration_since(now))
                }
                _ => None,
            })
            .min()
    }

    /// Returns `true` when at least one `Healthy` proxy exists outside `excluded`.
    pub fn has_unexcluded_healthy(&self, excluded: &HashSet<ProxyId>) -> bool {
        let guard = self.inner.lock().expect("ProxyPool lock");
        guard.entries.iter().any(|(id, e)| {
            !excluded.contains(id) && matches!(e.state, ProxyState::Healthy { .. })
        })
    }

    /// Wait up to `timeout` for at least one `Healthy` proxy to become available.
    pub async fn wait_for_healthy(&self, timeout: Duration) -> bool {
        let empty = HashSet::new();
        self.wait_for_unexcluded_healthy(&empty, timeout).await
    }

    /// Wait up to `timeout` for at least one `Healthy` proxy not in `excluded` to become available.
    pub async fn wait_for_unexcluded_healthy(
        &self,
        excluded: &HashSet<ProxyId>,
        timeout: Duration,
    ) -> bool {
        let start = Instant::now();
        loop {
            let notified = self.healthy_ready.notified();
            if self.has_unexcluded_healthy(excluded) {
                return true;
            }
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return self.has_unexcluded_healthy(excluded);
            }
            let remaining = timeout.saturating_sub(elapsed);
            let slice = remaining.min(Duration::from_secs(1));
            self.sweep_expired_cooldowns();
            if self.has_unexcluded_healthy(excluded) {
                return true;
            }
            let _ = tokio::time::timeout(slice, notified).await;
        }
    }

    pub fn stats(&self) -> ProxyPoolStats {
        let now = Instant::now();
        let guard = self.inner.lock().expect("ProxyPool lock");
        let mut healthy = 0usize;
        let mut untested = 0usize;
        let mut cooldown = 0usize;
        let mut dead = 0usize;

        for entry in guard.entries.values() {
            match entry.state {
                ProxyState::Healthy { .. } => healthy += 1,
                ProxyState::Untested => untested += 1,
                ProxyState::Cooldown { until, .. } if now >= until => untested += 1,
                ProxyState::Cooldown { .. } => cooldown += 1,
                ProxyState::Dead { .. } => dead += 1,
            }
        }

        let (active_proxy, active_rtt_ms) = match guard
            .active_proxy
            .and_then(|id| guard.entries.get(&id))
        {
            Some(entry) => {
                let rtt = match entry.state {
                    ProxyState::Healthy { ewma_rtt_ms, .. } => Some(ewma_rtt_ms),
                    _ => None,
                };
                (Some(entry.endpoint.redacted()), rtt)
            }
            None => (None, None),
        };

        ProxyPoolStats {
            enabled: self.cfg.enabled,
            chained_via: self.bridge.as_ref().map(|b| b.chained_via()),
            hop1_healthy: self.bridge.as_ref().is_none_or(|b| !b.is_hop1_failing()),
            total: guard.entries.len(),
            healthy,
            untested,
            cooldown,
            dead,
            active_proxy,
            active_rtt_ms,
            switches_total: self.switches_total.load(Ordering::Relaxed),
            hedged_wins: self.hedged_wins.load(Ordering::Relaxed),
        }
    }

    /// Return redacted snapshot of proxies sorted by health & latency.
    pub fn snapshot_proxies(&self, limit: usize) -> Vec<ProxyInfo> {
        let now = Instant::now();
        let guard = self.inner.lock().expect("ProxyPool lock");
        let mut items: Vec<(u8, u64, ProxyInfo)> = guard
            .entries
            .values()
            .map(|entry| {
                let (rank, sort_rtt, status, rtt_ms, ewma_rtt_ms, ok_count, cd_rem, last_error) =
                    match &entry.state {
                        ProxyState::Healthy {
                            rtt_ms,
                            ewma_rtt_ms,
                            ok_count,
                            ..
                        } => (
                            0u8,
                            *ewma_rtt_ms,
                            "healthy",
                            Some(*rtt_ms),
                            Some(*ewma_rtt_ms),
                            *ok_count,
                            None,
                            None,
                        ),
                        ProxyState::Cooldown {
                            until,
                            reason,
                            last_error,
                            ..
                        } => {
                            let label = match reason {
                                CooldownReason::RateLimited => "cooldown:rate_limited",
                                CooldownReason::Forbidden => "cooldown:forbidden",
                                CooldownReason::Overloaded => "cooldown:overloaded",
                                CooldownReason::TransientError => "cooldown:error",
                            };
                            (
                                1u8,
                                u64::MAX,
                                label,
                                None,
                                None,
                                0,
                                Some(until.saturating_duration_since(now).as_secs()),
                                Some(last_error.clone()),
                            )
                        }
                        ProxyState::Untested => {
                            (2u8, u64::MAX, "untested", None, None, 0, None, None)
                        }
                        ProxyState::Dead { last_error, .. } => (
                            3u8,
                            u64::MAX,
                            "dead",
                            None,
                            None,
                            0,
                            None,
                            Some(last_error.clone()),
                        ),
                    };

                (
                    rank,
                    sort_rtt,
                    ProxyInfo {
                        id: entry.endpoint.id.0,
                        proxy: entry.endpoint.redacted(),
                        protocol: entry.endpoint.protocol.scheme_str().to_string(),
                        status,
                        rtt_ms,
                        ewma_rtt_ms,
                        ok_count,
                        is_static: entry.endpoint.is_static,
                        cooldown_remaining_secs: cd_rem,
                        last_error,
                    },
                )
            })
            .collect();

        items.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.id.cmp(&b.2.id)));
        items.into_iter().take(limit).map(|(_, _, i)| i).collect()
    }
}

fn healthy_count_locked(guard: &PoolInner) -> usize {
    guard
        .entries
        .values()
        .filter(|e| matches!(e.state, ProxyState::Healthy { .. }))
        .count()
}

fn promote_expired_cooldowns(guard: &mut PoolInner, now: Instant) -> bool {
    let mut moved_to_untested = false;
    for entry in guard.entries.values_mut() {
        let replacement = match &entry.state {
            ProxyState::Cooldown {
                until,
                reason,
                prev_rtt_ms,
                ..
            } if now >= *until => match reason {
                CooldownReason::Overloaded => {
                    if let Some(rtt) = prev_rtt_ms {
                        Some(ProxyState::Healthy {
                            rtt_ms: *rtt,
                            ewma_rtt_ms: *rtt,
                            last_ok: now,
                            ok_count: 1,
                        })
                    } else {
                        moved_to_untested = true;
                        Some(ProxyState::Untested)
                    }
                }
                CooldownReason::RateLimited
                | CooldownReason::Forbidden
                | CooldownReason::TransientError => {
                    moved_to_untested = true;
                    Some(ProxyState::Untested)
                }
            },
            _ => None,
        };
        if let Some(new_state) = replacement {
            entry.state = new_state;
        }
    }
    moved_to_untested
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(raw: &str) -> ParsedProxy {
        ParsedProxy::parse_explicit(raw, false).unwrap()
    }

    #[test]
    fn sticky_lowest_latency_and_round_robin_selection() {
        let mut cfg = ProxyPoolConfig {
            enabled: true,
            ..Default::default()
        };
        cfg.selection.strategy = ProxyStrategy::Sticky;

        let pool = ProxyPool::new(
            cfg.clone(),
            vec![
                p("socks5://1.1.1.1:1080"),
                p("socks5://2.2.2.2:1080"),
                p("socks5://3.3.3.3:1080"),
            ],
            None,
        );
        pool.report_success(ProxyId(1), 150);
        pool.report_success(ProxyId(2), 50);
        pool.report_success(ProxyId(3), 100);

        let empty = HashSet::new();
        // Sticky picks lowest latency first and sticks to it
        let first = pool.select_proxy(&empty).unwrap();
        assert_eq!(first.id, ProxyId(2));
        // Even if ProxyId(1) becomes much faster later (EWMA drops below 50ms), sticky keeps ProxyId(2)
        for _ in 0..5 {
            pool.report_success(ProxyId(1), 1);
        }
        assert_eq!(pool.select_proxy(&empty).unwrap().id, ProxyId(2));

        // Exclusion skips active without unpinning active unless promoted
        let mut excl = HashSet::new();
        excl.insert(ProxyId(2));
        assert_eq!(pool.select_proxy(&excl).unwrap().id, ProxyId(1));

        // RoundRobin rotates evenly
        let mut rr_cfg = cfg;
        rr_cfg.selection.strategy = ProxyStrategy::RoundRobin;
        let rr_pool = ProxyPool::new(
            rr_cfg,
            vec![p("socks5://1.1.1.1:1080"), p("socks5://2.2.2.2:1080")],
            None,
        );
        rr_pool.report_success(ProxyId(1), 50);
        rr_pool.report_success(ProxyId(2), 50);
        assert_eq!(rr_pool.select_proxy(&empty).unwrap().id, ProxyId(1));
        assert_eq!(rr_pool.select_proxy(&empty).unwrap().id, ProxyId(2));
        assert_eq!(rr_pool.select_proxy(&empty).unwrap().id, ProxyId(1));
    }

    #[test]
    fn full_replace_drops_missing_carries_over_state_and_guards_empty_refresh() {
        let cfg = ProxyPoolConfig {
            enabled: true,
            ..Default::default()
        };
        let pool = ProxyPool::new(
            cfg,
            vec![
                p("socks5://1.1.1.1:1080"),
                p("socks5://2.2.2.2:1080"),
                p("socks5://3.3.3.3:1080"),
            ],
            None,
        );
        pool.report_success(ProxyId(1), 40);
        pool.report_rate_limited(ProxyId(2), Some(Duration::from_secs(600)), "429");
        pool.report_dead(ProxyId(3), "dead");
        assert_eq!(pool.select_proxy(&HashSet::new()).unwrap().id, ProxyId(1));

        // Outage guard: empty refresh does NOT wipe pool
        let empty_res = pool.replace_proxies(vec![]);
        assert!(empty_res.aborted_empty);
        assert_eq!(pool.stats().total, 3);

        // Replace with overlap on 2.2.2.2 (cooldown carried) + new 4.4.4.4, dropping 1.1.1.1 & 3.3.3.3
        let rep = pool.replace_proxies(vec![
            p("socks5://2.2.2.2:1080"),
            p("http://4.4.4.4:8080"),
        ]);
        assert!(!rep.aborted_empty);
        assert_eq!(rep.total, 2);
        assert_eq!(rep.kept, 1);
        assert_eq!(rep.added, 1);
        assert_eq!(rep.removed, 2);

        let stats = pool.stats();
        assert_eq!(stats.healthy, 0);
        assert_eq!(stats.cooldown, 1);
        assert_eq!(stats.untested, 1);
        assert_eq!(stats.active_proxy, None);
    }

    #[test]
    fn cooldown_expiry_and_consecutive_failures_to_dead() {
        let mut cfg = ProxyPoolConfig {
            enabled: true,
            ..Default::default()
        };
        cfg.selection.max_consecutive_failures = 2;
        cfg.selection.cooldown_error_base_secs = 1;
        let pool = ProxyPool::new(
            cfg,
            vec![p("socks5://1.1.1.1:1080"), p("socks5://2.2.2.2:1080")],
            None,
        );
        pool.report_success(ProxyId(1), 50);

        // Zero-duration rate limit expires immediately back to Untested (requires re-probe)
        pool.report_rate_limited(ProxyId(1), Some(Duration::from_millis(0)), "limit");
        assert_eq!(pool.select_proxy(&HashSet::new()), None);
        assert_eq!(pool.untested_count(), 2);

        // Liveness OK never clears an active cooldown
        pool.report_rate_limited(ProxyId(1), Some(Duration::from_secs(300)), "limit");
        pool.report_liveness_ok(ProxyId(1), 20);
        assert_eq!(pool.stats().cooldown, 1);

        // 2 consecutive transport errors mark ProxyId(2) Dead
        pool.report_transport_error(ProxyId(2), "timeout 1");
        assert_eq!(pool.stats().dead, 0);
        pool.report_transport_error(ProxyId(2), "timeout 2");
        assert_eq!(pool.stats().dead, 1);
    }

    #[tokio::test]
    async fn wait_for_unexcluded_healthy_sleeps_instead_of_busy_returning() {
        let cfg = ProxyPoolConfig {
            enabled: true,
            ..Default::default()
        };
        let pool = ProxyPool::new(
            cfg,
            vec![p("socks5://1.1.1.1:1080"), p("socks5://2.2.2.2:1080")],
            None,
        );
        pool.report_success(ProxyId(1), 25);

        let mut excluded = HashSet::new();
        excluded.insert(ProxyId(1));

        // Only ProxyId(1) is healthy and it is excluded -> must actually sleep the timeout.
        let t0 = Instant::now();
        let ready = pool
            .wait_for_unexcluded_healthy(&excluded, Duration::from_millis(80))
            .await;
        assert!(!ready);
        assert!(t0.elapsed() >= Duration::from_millis(70));
    }
}
