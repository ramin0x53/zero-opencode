pub mod chain;
pub mod endpoint;
pub mod health;
pub mod loader;
pub mod pool;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use tokio::sync::watch;
use tracing::info;
use wreq::Client;

use crate::config::AppConfig;
use crate::opencode::session::SessionManager;
use crate::opencode::tls_client::{build_pool_client, build_probe_client};
use crate::proxy::chain::ChainBridge;
use crate::proxy::endpoint::{ParsedProxy, ProxyId};
use crate::proxy::health::spawn_background_tasks;
use crate::proxy::loader::{LoadResult, fetch_all_proxies};
use crate::proxy::pool::ProxyPool;

/// Lifecycle manager for the proxy pool, optional `ChainBridge`, and background workers.
pub struct ProxyRuntime {
    pub pool: Arc<ProxyPool>,
    pub pool_client: Client,
    pub probe_client: Client,
    pub bridge: Option<Arc<ChainBridge>>,
    pub load_summary: LoadResult,
    shutdown_tx: watch::Sender<bool>,
    handles: Vec<tokio::task::JoinHandle<()>>,
}

impl ProxyRuntime {
    /// Load initial proxies (through `base_client` -> chained when `opencode.proxy` is set),
    /// start `ChainBridge` when `opencode.proxy` is configured, and optionally start
    /// background prober / keep-alive warmer / refresher tasks.
    pub async fn initialize(
        config: &AppConfig,
        base_client: &Client,
        sessions: Arc<SessionManager>,
        start_background: bool,
    ) -> anyhow::Result<Self> {
        let pool_cfg = &config.opencode.proxy_pool;

        let bridge = if let Some(raw_chain) = config
            .opencode
            .proxy
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let base_ep = ParsedProxy::parse_explicit(raw_chain, true)
                .context("invalid opencode.proxy")?
                .into_endpoint(ProxyId(0));
            let b = ChainBridge::start(
                base_ep,
                Duration::from_secs(pool_cfg.selection.connect_timeout_secs),
            )
            .await?;
            info!(
                chained_via = %b.chained_via(),
                bridge_addr = %b.local_addr(),
                "universal proxy chaining active"
            );
            Some(b)
        } else {
            None
        };

        let probe_client = build_probe_client(pool_cfg.health_check.timeout_secs)?;
        let pool_client = build_pool_client(
            pool_cfg.selection.connect_timeout_secs,
            config.opencode.timeout_secs,
        )?;

        let load_summary = fetch_all_proxies(base_client, pool_cfg).await?;
        info!(
            total = load_summary.proxies.len(),
            static_proxies = load_summary.static_count,
            remote_proxies = load_summary.remote_count,
            socks5 = load_summary.protocol_counts.socks5,
            socks4 = load_summary.protocol_counts.socks4,
            http = load_summary.protocol_counts.http,
            https = load_summary.protocol_counts.https,
            sources_ok = load_summary.sources_succeeded,
            sources_failed = load_summary.sources_failed,
            chained = bridge.is_some(),
            "loaded unique proxy pool candidates"
        );

        let pool = ProxyPool::new(pool_cfg.clone(), load_summary.proxies.clone(), bridge.clone());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let handles = if start_background {
            spawn_background_tasks(
                Arc::clone(&pool),
                base_client.clone(),
                probe_client.clone(),
                pool_client.clone(),
                sessions,
                shutdown_rx,
            )
        } else {
            Vec::new()
        };

        Ok(Self {
            pool,
            pool_client,
            probe_client,
            bridge,
            load_summary,
            shutdown_tx,
            handles,
        })
    }

    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);
        if let Some(bridge) = &self.bridge {
            bridge.shutdown();
        }
        for handle in self.handles {
            handle.abort();
            let _ = handle.await;
        }
    }
}
