use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use wreq::tls::TlsOptions;
use wreq::{Client, Proxy};

use crate::proxy::chain::ChainBridge;
use crate::proxy::endpoint::{ParsedProxy, ProxyEndpoint};

/// BoringSSL/Chrome-family cipher list accepted by opencode.ai's free tier gate
/// (JA4 cipher hash `5b57614c22b0`). Must remain verbatim.
pub const OPENCODE_CIPHER_LIST: &str = concat!(
    "TLS_AES_128_GCM_SHA256:",
    "TLS_AES_256_GCM_SHA384:",
    "TLS_CHACHA20_POLY1305_SHA256:",
    "ECDHE-ECDSA-AES128-GCM-SHA256:",
    "ECDHE-RSA-AES128-GCM-SHA256:",
    "ECDHE-ECDSA-AES256-GCM-SHA384:",
    "ECDHE-RSA-AES256-GCM-SHA384:",
    "ECDHE-ECDSA-CHACHA20-POLY1305:",
    "ECDHE-RSA-CHACHA20-POLY1305:",
    "ECDHE-ECDSA-AES128-SHA:",
    "ECDHE-RSA-AES128-SHA:",
    "ECDHE-ECDSA-AES256-SHA:",
    "ECDHE-RSA-AES256-SHA:",
    "AES128-GCM-SHA256:",
    "AES256-GCM-SHA384:",
    "AES128-SHA:",
    "AES256-SHA"
);

/// Build a BoringSSL-backed `wreq::Client` configured with OpenCode's accepted
/// TLS cipher suite list and optional HTTP/HTTPS/SOCKS4/SOCKS5 upstream proxy.
///
/// Uses `connect_timeout` + `read_timeout` (idle/stall timeout per chunk)
/// instead of a whole-request `timeout` so long reasoning SSE streams are not
/// aborted mid-generation.
pub fn build_opencode_client(proxy: Option<&str>, timeout_secs: u64) -> anyhow::Result<Client> {
    let tls = TlsOptions::builder()
        .cipher_list(OPENCODE_CIPHER_LIST)
        .build();

    let mut builder = Client::builder()
        .tls_options(tls)
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(timeout_secs));

    if let Some(proxy_url) = proxy.map(str::trim).filter(|s| !s.is_empty()) {
        // Normalize SOCKS4/SOCKS5 to remote DNS resolution (`socks4a`/`socks5h`).
        let normalized = ParsedProxy::parse_explicit(proxy_url, true)
            .map(|p| {
                p.into_endpoint(crate::proxy::endpoint::ProxyId(0))
                    .to_wreq_url()
            })
            .unwrap_or_else(|_| proxy_url.to_string());
        let parsed = Proxy::all(&normalized)
            .with_context(|| format!("invalid upstream proxy URL: {proxy_url}"))?;
        builder = builder.proxy(parsed);
    }

    builder
        .build()
        .context("failed to build wreq BoringSSL HTTP client")
}

/// Build a dedicated BoringSSL `wreq::Client` for parallel proxy probing.
/// Disables idle connection pooling (`pool_max_idle_per_host(0)`) to avoid FD bloat
/// when scanning thousands of candidate proxies.
pub fn build_probe_client(timeout_secs: u64) -> anyhow::Result<Client> {
    let tls = TlsOptions::builder()
        .cipher_list(OPENCODE_CIPHER_LIST)
        .build();

    Client::builder()
        .tls_options(tls)
        .connect_timeout(Duration::from_secs(timeout_secs))
        .timeout(Duration::from_secs(timeout_secs))
        .pool_max_idle_per_host(0)
        .build()
        .context("failed to build wreq probe client")
}

/// Build a shared BoringSSL `wreq::Client` for live proxy-pool requests and hot-standby
/// keep-alive warming, with bounded idle pooling per proxy.
pub fn build_pool_client(
    connect_timeout_secs: u64,
    read_timeout_secs: u64,
) -> anyhow::Result<Client> {
    let tls = TlsOptions::builder()
        .cipher_list(OPENCODE_CIPHER_LIST)
        .build();

    Client::builder()
        .tls_options(tls)
        .connect_timeout(Duration::from_secs(connect_timeout_secs))
        .read_timeout(Duration::from_secs(read_timeout_secs))
        .pool_max_idle_per_host(2)
        .pool_max_size(64)
        .pool_idle_timeout(Duration::from_secs(90))
        .build()
        .context("failed to build wreq pool client")
}

/// Resolve the `wreq::Proxy` for an endpoint, routing through `ChainBridge` whenever
/// `opencode.proxy` chaining is active.
pub fn resolve_wreq_proxy(
    endpoint: &ProxyEndpoint,
    bridge: Option<&Arc<ChainBridge>>,
) -> anyhow::Result<Proxy> {
    match bridge {
        Some(b) => b.wreq_proxy_for(endpoint.id),
        None => Proxy::all(endpoint.to_wreq_url())
            .with_context(|| format!("invalid proxy URL for {}", endpoint.redacted())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy::endpoint::ProxyId;

    #[test]
    fn cipher_list_matches_verbatim_17_ciphers() {
        let ciphers: Vec<&str> = OPENCODE_CIPHER_LIST.split(':').collect();
        assert_eq!(ciphers.len(), 17);
        assert_eq!(ciphers[0], "TLS_AES_128_GCM_SHA256");
        assert_eq!(ciphers[16], "AES256-SHA");
    }

    #[test]
    fn builds_direct_client_successfully() {
        let client = build_opencode_client(None, 120);
        assert!(client.is_ok());
    }

    #[test]
    fn builds_proxied_client_successfully() {
        let client = build_opencode_client(Some("socks5://127.0.0.1:1080"), 120);
        assert!(client.is_ok());
    }

    #[test]
    fn builds_probe_and_pool_clients_successfully() {
        assert!(build_probe_client(6).is_ok());
        assert!(build_pool_client(4, 120).is_ok());
        let ep = ParsedProxy::parse_explicit("socks5://1.2.3.4:1080", false)
            .unwrap()
            .into_endpoint(ProxyId(1));
        assert!(resolve_wreq_proxy(&ep, None).is_ok());
    }
}
