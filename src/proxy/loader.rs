use std::collections::HashSet;
use std::time::Duration;

use anyhow::bail;
use futures_util::StreamExt;
use tracing::warn;
use url::Url;
use wreq::Client;

use crate::config::ProxyPoolConfig;
use crate::proxy::endpoint::{
    ParsedProxy, ProxyProtocol, deduplicate_proxies, parse_proxy_list,
};

const SOURCE_FETCH_TIMEOUT_SECS: u64 = 15;
const MAX_SOURCE_BODY_BYTES: usize = 10 * 1024 * 1024;
const SOURCE_FETCH_CONCURRENCY: usize = 8;

/// Breakdown of loaded unique proxies by protocol family.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProtocolCounts {
    pub socks5: usize,
    pub socks4: usize,
    pub http: usize,
    pub https: usize,
}

impl ProtocolCounts {
    pub fn from_proxies(proxies: &[ParsedProxy]) -> Self {
        let mut counts = Self::default();
        for p in proxies {
            match p.protocol {
                ProxyProtocol::Socks5 | ProxyProtocol::Socks5h => counts.socks5 += 1,
                ProxyProtocol::Socks4 | ProxyProtocol::Socks4a => counts.socks4 += 1,
                ProxyProtocol::Http => counts.http += 1,
                ProxyProtocol::Https => counts.https += 1,
            }
        }
        counts
    }
}

/// Summary of loading and deduplicating proxies from `static_proxies` and `sources.*`.
#[derive(Debug, Clone)]
pub struct LoadResult {
    pub proxies: Vec<ParsedProxy>,
    pub static_count: usize,
    pub remote_count: usize,
    pub sources_total: usize,
    pub sources_succeeded: usize,
    pub sources_failed: usize,
    pub protocol_counts: ProtocolCounts,
}

fn redact_url_for_log(raw: &str) -> String {
    if let Ok(mut parsed) = Url::parse(raw) {
        if !parsed.username().is_empty() || parsed.password().is_some() {
            let _ = parsed.set_username("");
            let _ = parsed.set_password(None);
        }
        return parsed.to_string();
    }
    "<invalid-url>".to_string()
}

async fn fetch_one_source(
    client: &Client,
    url: &str,
    default_protocol: Option<ProxyProtocol>,
) -> anyhow::Result<Vec<ParsedProxy>> {
    let resp = client
        .get(url)
        .timeout(Duration::from_secs(SOURCE_FETCH_TIMEOUT_SECS))
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        bail!("HTTP {}", status.as_u16());
    }

    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if body.len().saturating_add(chunk.len()) > MAX_SOURCE_BODY_BYTES {
            bail!("response exceeded {} bytes limit", MAX_SOURCE_BODY_BYTES);
        }
        body.extend_from_slice(&chunk);
    }

    let text = String::from_utf8_lossy(&body);
    Ok(parse_proxy_list(&text, default_protocol, true))
}

fn shuffle_in_place<T>(items: &mut [T]) {
    if items.len() > 1 {
        for i in (1..items.len()).rev() {
            let j = rand::random_range(0..=i);
            items.swap(i, j);
        }
    }
}

/// Fetch all configured proxy source URLs concurrently using `base_client` (which
/// automatically tunnels through `opencode.proxy` when configured), merge with
/// `static_proxies`, deduplicate, and shuffle unique entries.
pub async fn fetch_all_proxies(
    base_client: &Client,
    cfg: &ProxyPoolConfig,
) -> anyhow::Result<LoadResult> {
    let mut static_items = Vec::new();
    for raw in &cfg.static_proxies {
        static_items.push(ParsedProxy::parse_explicit(raw, true)?);
    }
    let mut static_items = deduplicate_proxies(static_items);
    let static_keys: HashSet<_> = static_items.iter().map(ParsedProxy::key).collect();

    let mut source_jobs: Vec<(String, Option<ProxyProtocol>)> = Vec::new();
    for url in &cfg.sources.socks5 {
        source_jobs.push((url.trim().to_string(), Some(ProxyProtocol::Socks5h)));
    }
    for url in &cfg.sources.socks4 {
        source_jobs.push((url.trim().to_string(), Some(ProxyProtocol::Socks4a)));
    }
    for url in &cfg.sources.http {
        source_jobs.push((url.trim().to_string(), Some(ProxyProtocol::Http)));
    }
    for url in &cfg.sources.mixed {
        source_jobs.push((url.trim().to_string(), None));
    }

    let sources_total = source_jobs.len();

    let results = futures_util::stream::iter(source_jobs.into_iter().map(
        |(url, default_proto)| async move {
            let res = fetch_one_source(base_client, &url, default_proto).await;
            (url, res)
        },
    ))
    .buffer_unordered(SOURCE_FETCH_CONCURRENCY)
    .collect::<Vec<_>>()
    .await;

    let mut sources_succeeded = 0usize;
    let mut sources_failed = 0usize;
    let mut remote_raw = Vec::new();

    for (url, res) in results {
        match res {
            Ok(parsed) => {
                sources_succeeded += 1;
                remote_raw.extend(parsed);
            }
            Err(err) => {
                sources_failed += 1;
                warn!(
                    url = %redact_url_for_log(&url),
                    error = %err,
                    "failed to fetch proxy list source"
                );
            }
        }
    }

    // Deduplicate remote candidates and exclude duplicates already in static_proxies.
    let mut remote_items: Vec<ParsedProxy> = deduplicate_proxies(remote_raw)
        .into_iter()
        .filter(|item| !static_keys.contains(&item.key()))
        .collect();

    // Shuffle unique candidates after deduplication.
    shuffle_in_place(&mut static_items);
    shuffle_in_place(&mut remote_items);

    let mut proxies = Vec::with_capacity(static_items.len() + remote_items.len());
    proxies.extend(static_items);
    proxies.extend(remote_items);

    let static_count = proxies.iter().filter(|p| p.is_static).count();
    let remote_count = proxies.len().saturating_sub(static_count);
    let protocol_counts = ProtocolCounts::from_proxies(&proxies);

    Ok(LoadResult {
        proxies,
        static_count,
        remote_count,
        sources_total,
        sources_succeeded,
        sources_failed,
        protocol_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProxySourcesConfig;
    use crate::opencode::tls_client::build_opencode_client;
    use axum::Router;
    use axum::http::StatusCode;
    use axum::routing::get;
    use tokio::net::TcpListener;

    async fn spawn_mock_sources_server() -> String {
        let app = Router::new()
            .route(
                "/socks5.txt",
                get(|| async { "1.1.1.1:1080\n2.2.2.2:1080\n127.0.0.1:1080\n" }),
            )
            .route(
                "/socks4.txt",
                get(|| async { "3.3.3.3:4145\n1.1.1.1:1080\n" }),
            )
            .route(
                "/mixed.json",
                get(|| async { r#"["http://4.4.4.4:8080","socks5://1.1.1.1:1080"]"# }),
            )
            .route("/broken", get(|| async { StatusCode::INTERNAL_SERVER_ERROR }));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn fetches_deduplicates_and_keeps_all_including_private_while_tolerating_failures() {
        let base = spawn_mock_sources_server().await;
        let client = build_opencode_client(None, 10).unwrap();

        let cfg = ProxyPoolConfig {
            enabled: true,
            sources: ProxySourcesConfig {
                socks5: vec![format!("{base}/socks5.txt"), format!("{base}/broken")],
                socks4: vec![format!("{base}/socks4.txt")],
                http: vec![],
                mixed: vec![format!("{base}/mixed.json")],
            },
            static_proxies: vec!["socks5://127.0.0.1:9999".to_string()],
            ..Default::default()
        };

        let result = fetch_all_proxies(&client, &cfg).await.unwrap();
        assert_eq!(result.sources_total, 4);
        assert_eq!(result.sources_succeeded, 3);
        assert_eq!(result.sources_failed, 1);
        assert_eq!(result.static_count, 1);
        assert_eq!(result.remote_count, 6);
        // 1 static (127.0.0.1:9999) + all 6 unique remote endpoints including private 127.0.0.1:1080:
        // socks5h://1.1.1.1:1080, socks5h://2.2.2.2:1080, socks5h://127.0.0.1:1080,
        // socks4a://3.3.3.3:4145, socks4a://1.1.1.1:1080, http://4.4.4.4:8080
        assert_eq!(result.proxies.len(), 7);
        assert_eq!(result.proxies[0].host, "127.0.0.1");
        assert!(result.proxies[0].is_static);
        // Remote 127.0.0.1:1080 is now kept as a non-static entry
        assert!(
            result
                .proxies
                .iter()
                .any(|p| p.port == 1080 && p.host == "127.0.0.1" && !p.is_static)
        );
    }

    #[tokio::test]
    async fn shuffles_unique_loaded_proxies() {
        let mut list_a = String::new();
        let mut list_b = String::new();
        for i in 1..=20u16 {
            list_a.push_str(&format!("1.1.1.{i}:1080\n"));
            if i <= 10 {
                // Duplicates across sources to verify deduplication before shuffling
                list_b.push_str(&format!("1.1.1.{i}:1080\n"));
            }
        }

        let app = Router::new()
            .route("/a.txt", get(move || async move { list_a.clone() }))
            .route("/b.txt", get(move || async move { list_b.clone() }));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let base = format!("http://{addr}");

        let client = build_opencode_client(None, 10).unwrap();
        let cfg = ProxyPoolConfig {
            enabled: true,
            sources: ProxySourcesConfig {
                socks5: vec![format!("{base}/a.txt"), format!("{base}/b.txt")],
                ..Default::default()
            },
            ..Default::default()
        };

        let first = fetch_all_proxies(&client, &cfg).await.unwrap();
        assert_eq!(first.proxies.len(), 20);
        let first_order: Vec<String> = first.proxies.iter().map(ParsedProxy::redacted).collect();
        let first_set: HashSet<String> = first_order.iter().cloned().collect();
        assert_eq!(first_set.len(), 20);

        let mut saw_different_order = false;
        for _ in 0..5 {
            let next = fetch_all_proxies(&client, &cfg).await.unwrap();
            assert_eq!(next.proxies.len(), 20);
            let next_order: Vec<String> = next.proxies.iter().map(ParsedProxy::redacted).collect();
            let next_set: HashSet<String> = next_order.iter().cloned().collect();
            assert_eq!(next_set, first_set);
            if next_order != first_order {
                saw_different_order = true;
                break;
            }
        }
        assert!(
            saw_different_order,
            "expected fetch_all_proxies to shuffle unique loaded proxies across runs"
        );
    }
}
