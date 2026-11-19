use std::time::Duration;

use anyhow::Context;
use wreq::tls::TlsOptions;
use wreq::{Client, Proxy};

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
/// TLS cipher suite list and optional HTTP/HTTPS/SOCKS5 proxy.
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
        let parsed = Proxy::all(proxy_url)
            .with_context(|| format!("invalid upstream proxy URL: {proxy_url}"))?;
        builder = builder.proxy(parsed);
    }

    builder
        .build()
        .context("failed to build wreq BoringSSL HTTP client")
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
