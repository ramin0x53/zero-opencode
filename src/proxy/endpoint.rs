use std::collections::HashSet;
use std::fmt;
use std::net::{IpAddr, Ipv6Addr};

use anyhow::{Context, bail};
use percent_encoding::{AsciiSet, CONTROLS, percent_decode_str, utf8_percent_encode};
use serde::Serialize;
use serde_json::Value;
use url::Url;

const USERINFO_ENCODE_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'/')
    .add(b':')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'@')
    .add(b'[')
    .add(b'\\')
    .add(b']')
    .add(b'^')
    .add(b'`')
    .add(b'{')
    .add(b'|')
    .add(b'}');

/// Monotonic identifier assigned to a proxy endpoint within the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct ProxyId(pub u64);

impl fmt::Display for ProxyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Supported upstream proxy protocols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ProxyProtocol {
    Http,
    Https,
    Socks4,
    Socks4a,
    Socks5,
    Socks5h,
}

impl ProxyProtocol {
    /// Parse a proxy scheme case-insensitively.
    pub fn parse_scheme(scheme: &str) -> Option<Self> {
        match scheme.trim().to_ascii_lowercase().as_str() {
            "http" => Some(Self::Http),
            "https" => Some(Self::Https),
            "socks4" => Some(Self::Socks4),
            "socks4a" => Some(Self::Socks4a),
            "socks5" => Some(Self::Socks5),
            "socks5h" => Some(Self::Socks5h),
            _ => None,
        }
    }

    pub const fn scheme_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Https => "https",
            Self::Socks4 => "socks4",
            Self::Socks4a => "socks4a",
            Self::Socks5 => "socks5",
            Self::Socks5h => "socks5h",
        }
    }

    /// Normalize SOCKS variants to remote DNS resolution (`socks5h` / `socks4a`).
    pub const fn to_remote_dns(self) -> Self {
        match self {
            Self::Socks5 => Self::Socks5h,
            Self::Socks4 => Self::Socks4a,
            other => other,
        }
    }

    /// Canonical protocol used for deduplication (`socks5` == `socks5h`, `socks4` == `socks4a`).
    pub const fn dedup_protocol(self) -> Self {
        self.to_remote_dns()
    }

    pub const fn is_socks(self) -> bool {
        matches!(
            self,
            Self::Socks4 | Self::Socks4a | Self::Socks5 | Self::Socks5h
        )
    }
}

impl fmt::Display for ProxyProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.scheme_str())
    }
}

/// Optional authentication credentials for an upstream proxy.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ProxyAuth {
    pub username: String,
    pub password: String,
}

impl fmt::Debug for ProxyAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProxyAuth")
            .field("username", &"[REDACTED]")
            .field("password", &"[REDACTED]")
            .finish()
    }
}

/// Canonical deduplication key for a proxy endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProxyKey {
    pub protocol: ProxyProtocol,
    pub host: String,
    pub port: u16,
    pub auth: Option<ProxyAuth>,
}

/// Parsed proxy before receiving a pool-assigned `ProxyId`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedProxy {
    pub protocol: ProxyProtocol,
    pub host: String,
    pub port: u16,
    pub auth: Option<ProxyAuth>,
    pub is_static: bool,
}

impl ParsedProxy {
    /// Parse an explicit proxy URL (requires scheme, host, and port > 0).
    /// Allows private/loopback addresses (for `opencode.proxy` and `static_proxies`).
    pub fn parse_explicit(raw: &str, is_static: bool) -> anyhow::Result<Self> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            bail!("proxy URL must not be empty");
        }
        if !trimmed.contains("://") {
            bail!("proxy URL must include an explicit scheme (e.g. socks5://, socks4://, http://)");
        }
        let parsed = Url::parse(trimmed)
            .with_context(|| format!("invalid proxy URL syntax: {}", redact_url_str(trimmed)))?;
        let protocol = ProxyProtocol::parse_scheme(parsed.scheme()).ok_or_else(|| {
            anyhow::anyhow!(
                "unsupported proxy scheme `{}` (expected http, https, socks4, socks4a, socks5, socks5h)",
                parsed.scheme()
            )
        })?;
        let host = normalize_host(parsed.host_str().unwrap_or(""))?;
        let port = parsed
            .port()
            .filter(|&p| p > 0)
            .ok_or_else(|| anyhow::anyhow!("proxy URL must specify a valid non-zero port"))?;

        if !parsed.path().is_empty() && parsed.path() != "/" {
            bail!("proxy URL must not contain a path");
        }
        if parsed.query().is_some() || parsed.fragment().is_some() {
            bail!("proxy URL must not contain query or fragment parameters");
        }

        let auth = extract_url_auth(&parsed);
        Ok(Self {
            protocol,
            host,
            port,
            auth,
            is_static,
        })
    }

    pub fn key(&self) -> ProxyKey {
        ProxyKey {
            protocol: self.protocol.dedup_protocol(),
            host: self.host.clone(),
            port: self.port,
            auth: self.auth.clone(),
        }
    }

    pub fn into_endpoint(self, id: ProxyId) -> ProxyEndpoint {
        ProxyEndpoint {
            id,
            protocol: self.protocol,
            host: self.host,
            port: self.port,
            auth: self.auth,
            is_static: self.is_static,
        }
    }

    pub fn redacted(&self) -> String {
        format_redacted(self.protocol, &self.host, self.port)
    }
}

/// Active proxy endpoint registered in the pool and `ChainBridge`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyEndpoint {
    pub id: ProxyId,
    pub protocol: ProxyProtocol,
    pub host: String,
    pub port: u16,
    pub auth: Option<ProxyAuth>,
    pub is_static: bool,
}

impl ProxyEndpoint {
    pub fn key(&self) -> ProxyKey {
        ProxyKey {
            protocol: self.protocol.dedup_protocol(),
            host: self.host.clone(),
            port: self.port,
            auth: self.auth.clone(),
        }
    }

    /// Credential-free `scheme://host:port` string safe for logs, API responses, and CLI output.
    pub fn redacted(&self) -> String {
        format_redacted(self.protocol, &self.host, self.port)
    }

    /// Build a `wreq`-compatible proxy URL using remote-DNS SOCKS schemes (`socks5h` / `socks4a`).
    pub fn to_wreq_url(&self) -> String {
        let scheme = self.protocol.to_remote_dns().scheme_str();
        let host = bracket_host(&self.host);
        match &self.auth {
            Some(auth) => {
                let user = utf8_percent_encode(&auth.username, USERINFO_ENCODE_SET);
                let pass = utf8_percent_encode(&auth.password, USERINFO_ENCODE_SET);
                format!("{scheme}://{user}:{pass}@{host}:{}", self.port)
            }
            None => format!("{scheme}://{host}:{}", self.port),
        }
    }
}

fn format_redacted(protocol: ProxyProtocol, host: &str, port: u16) -> String {
    format!("{}://{}:{port}", protocol.scheme_str(), bracket_host(host))
}

fn bracket_host(host: &str) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.to_string()
    }
}

fn redact_url_str(raw: &str) -> String {
    if let Ok(mut parsed) = Url::parse(raw)
        && (!parsed.username().is_empty() || parsed.password().is_some())
    {
        let _ = parsed.set_username("");
        let _ = parsed.set_password(None);
        return parsed.to_string();
    }
    "<invalid-proxy-url>".to_string()
}

fn extract_url_auth(url: &Url) -> Option<ProxyAuth> {
    let raw_user = url.username();
    let raw_pass = url.password();
    if raw_user.is_empty() && raw_pass.is_none() {
        return None;
    }
    let username = percent_decode_str(raw_user).decode_utf8_lossy().into_owned();
    let password = raw_pass
        .map(|p| percent_decode_str(p).decode_utf8_lossy().into_owned())
        .unwrap_or_default();
    Some(ProxyAuth { username, password })
}

fn normalize_host(raw: &str) -> anyhow::Result<String> {
    let trimmed = raw
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_end_matches('.');
    if trimmed.is_empty() {
        bail!("proxy host must not be empty");
    }
    if trimmed.chars().any(|c| c.is_whitespace() || c.is_control()) {
        bail!("proxy host contains invalid whitespace or control characters");
    }
    if trimmed.starts_with('.') {
        bail!("proxy host must not start with a dot");
    }
    if let Ok(ip) = trimmed.parse::<IpAddr>() {
        return Ok(ip.to_string());
    }
    // Hostname sanity check (allow DNS labels with letters, digits, hyphen, underscore, dot)
    if !trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
    {
        bail!("proxy host `{trimmed}` contains unsupported characters");
    }
    Ok(trimmed.to_ascii_lowercase())
}

fn is_disallowed_remote_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_multicast()
                || v4.is_broadcast()
                || v4.is_documentation()
        }
        IpAddr::V6(v6) => is_disallowed_ipv6(v6),
    }
}

fn is_disallowed_ipv6(v6: Ipv6Addr) -> bool {
    if let Some(mapped) = v6.to_ipv4_mapped() {
        return is_disallowed_remote_ip(IpAddr::V4(mapped));
    }
    let segs = v6.segments();
    let is_ula = (segs[0] & 0xfe00) == 0xfc00;
    let is_link_local = (segs[0] & 0xffc0) == 0xfe80;
    v6.is_loopback() || v6.is_unspecified() || v6.is_multicast() || is_ula || is_link_local
}

/// Check whether a normalized host is allowed for remote untrusted proxy lists.
pub fn is_allowed_host(host: &str, allow_private: bool) -> bool {
    if allow_private {
        return true;
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        return !is_disallowed_remote_ip(ip);
    }
    let lower = host.to_ascii_lowercase();
    !(lower == "localhost"
        || lower.ends_with(".localhost")
        || lower.ends_with(".local")
        || lower.ends_with(".internal"))
}

fn strip_line_comment(line: &str) -> &str {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
        return "";
    }
    let mut end = trimmed.len();
    for marker in [" #", "\t#", " //", "\t//"] {
        if let Some(idx) = trimmed[..end].find(marker) {
            end = idx;
        }
    }
    trimmed[..end].trim()
}

/// Parse a single proxy line (text list entry) into a `ParsedProxy`.
pub fn parse_proxy_line(
    raw_line: &str,
    default_protocol: Option<ProxyProtocol>,
    allow_private: bool,
) -> Option<ParsedProxy> {
    let cleaned = strip_line_comment(raw_line);
    if cleaned.is_empty() {
        return None;
    }

    // Take first whitespace token (handles trailing annotations like `1.2.3.4:1080 US elite`).
    let token = cleaned.split_whitespace().next()?.trim_end_matches(',');
    if token.is_empty() {
        return None;
    }

    let parsed = if token.contains("://") {
        ParsedProxy::parse_explicit(token, false).ok()?
    } else {
        let proto = default_protocol?;
        // Check `ip:port:user:pass` before adding scheme (IPv4/hostname without brackets).
        if !token.contains('@') && !token.contains('[') {
            let parts: Vec<&str> = token.split(':').collect();
            if parts.len() == 4 {
                let port: u16 = parts[1].parse().ok()?;
                if port == 0 {
                    return None;
                }
                let host = normalize_host(parts[0]).ok()?;
                let auth = if !parts[2].is_empty() || !parts[3].is_empty() {
                    Some(ProxyAuth {
                        username: parts[2].to_string(),
                        password: parts[3].to_string(),
                    })
                } else {
                    None
                };
                let candidate = ParsedProxy {
                    protocol: proto,
                    host,
                    port,
                    auth,
                    is_static: false,
                };
                return is_allowed_host(&candidate.host, allow_private).then_some(candidate);
            }
        }
        let candidate_url = format!("{}://{token}", proto.scheme_str());
        ParsedProxy::parse_explicit(&candidate_url, false).ok()?
    };

    is_allowed_host(&parsed.host, allow_private).then_some(parsed)
}

/// Parse either plain-text proxy lines or JSON feeds (`["..."]` / `{"data":[...]}`).
pub fn parse_proxy_list(
    raw: &str,
    default_protocol: Option<ProxyProtocol>,
    allow_private: bool,
) -> Vec<ParsedProxy> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    if (trimmed.starts_with('[') || trimmed.starts_with('{'))
        && let Ok(val) = serde_json::from_str::<Value>(trimmed)
    {
        return parse_json_proxy_list(&val, default_protocol, allow_private);
    }

    trimmed
        .lines()
        .filter_map(|line| parse_proxy_line(line, default_protocol, allow_private))
        .collect()
}

fn parse_json_proxy_list(
    val: &Value,
    default_protocol: Option<ProxyProtocol>,
    allow_private: bool,
) -> Vec<ParsedProxy> {
    let items: Option<&Vec<Value>> = match val {
        Value::Array(arr) => Some(arr),
        Value::Object(map) => map
            .get("data")
            .or_else(|| map.get("proxies"))
            .or_else(|| map.get("list"))
            .and_then(Value::as_array),
        _ => None,
    };

    let Some(items) = items else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for item in items {
        match item {
            Value::String(s) => {
                if let Some(p) = parse_proxy_line(s, default_protocol, allow_private) {
                    out.push(p);
                }
            }
            Value::Object(obj) => {
                parse_json_proxy_object(obj, default_protocol, allow_private, &mut out);
            }
            _ => {}
        }
    }
    out
}

fn parse_json_proxy_object(
    obj: &serde_json::Map<String, Value>,
    default_protocol: Option<ProxyProtocol>,
    allow_private: bool,
    out: &mut Vec<ParsedProxy>,
) {
    if let Some(url_str) = obj.get("url").or_else(|| obj.get("proxy")).and_then(Value::as_str)
        && let Some(p) = parse_proxy_line(url_str, default_protocol, allow_private)
    {
        out.push(p);
        return;
    }

    let Some(raw_host) = obj
        .get("ip")
        .or_else(|| obj.get("host"))
        .and_then(Value::as_str)
    else {
        return;
    };
    let Ok(host) = normalize_host(raw_host) else {
        return;
    };
    if !is_allowed_host(&host, allow_private) {
        return;
    }

    let port_val = obj.get("port");
    let port: Option<u16> = match port_val {
        Some(Value::Number(n)) => n.as_u64().and_then(|v| u16::try_from(v).ok()),
        Some(Value::String(s)) => s.trim().parse::<u16>().ok(),
        _ => None,
    };
    let Some(port) = port.filter(|&p| p > 0) else {
        return;
    };

    let auth = match (
        obj.get("username")
            .or_else(|| obj.get("user"))
            .and_then(Value::as_str),
        obj.get("password")
            .or_else(|| obj.get("pass"))
            .and_then(Value::as_str),
    ) {
        (Some(u), Some(p)) if !u.is_empty() || !p.is_empty() => Some(ProxyAuth {
            username: u.to_string(),
            password: p.to_string(),
        }),
        (Some(u), None) if !u.is_empty() => Some(ProxyAuth {
            username: u.to_string(),
            password: String::new(),
        }),
        _ => None,
    };

    let mut protocols: Vec<ProxyProtocol> = Vec::new();
    if let Some(arr) = obj.get("protocols").and_then(Value::as_array) {
        for p in arr {
            if let Some(proto) = p.as_str().and_then(ProxyProtocol::parse_scheme) {
                protocols.push(proto);
            }
        }
    } else if let Some(s) = obj
        .get("protocol")
        .or_else(|| obj.get("type"))
        .and_then(Value::as_str)
        && let Some(proto) = ProxyProtocol::parse_scheme(s)
    {
        protocols.push(proto);
    }

    if protocols.is_empty()
        && let Some(def) = default_protocol
    {
        protocols.push(def);
    }

    for proto in protocols {
        out.push(ParsedProxy {
            protocol: proto,
            host: host.clone(),
            port,
            auth: auth.clone(),
            is_static: false,
        });
    }
}

/// Deduplicate parsed proxies preserving order (`static_proxies` kept first when placed first).
pub fn deduplicate_proxies(items: Vec<ParsedProxy>) -> Vec<ParsedProxy> {
    let mut seen = HashSet::with_capacity(items.len());
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if seen.insert(item.key()) {
            out.push(item);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_text_formats_comments_and_trailing_metadata() {
        let raw = "\
# header comment\r\n
// another comment\r\n
1.2.3.4:1080\r\n
5.6.7.8:8080 US elite\r\n
user1:p@ss!@9.10.11.12:3128\r\n
13.14.15.16:9050:myuser:mypass\r\n
[2001:db8::1]:1080\r\n
http://17.18.19.20:8888 # inline comment\r\n
bad_line_no_port\r\n
21.22.23.24:0\r\n
";
        let list = parse_proxy_list(raw, Some(ProxyProtocol::Socks5h), false);
        assert_eq!(list.len(), 6);
        assert_eq!(list[0].redacted(), "socks5h://1.2.3.4:1080");
        assert_eq!(list[1].redacted(), "socks5h://5.6.7.8:8080");
        assert_eq!(
            list[2].auth,
            Some(ProxyAuth {
                username: "user1".into(),
                password: "p@ss!".into()
            })
        );
        assert_eq!(list[3].redacted(), "socks5h://13.14.15.16:9050");
        assert_eq!(
            list[3].auth,
            Some(ProxyAuth {
                username: "myuser".into(),
                password: "mypass".into()
            })
        );
        assert_eq!(list[4].redacted(), "socks5h://[2001:db8::1]:1080");
        assert_eq!(list[5].redacted(), "http://17.18.19.20:8888");
    }

    #[test]
    fn mixed_source_requires_explicit_scheme_and_json_parses() {
        let mixed_raw = "1.2.3.4:1080\nsocks4://1.2.3.4:4145\nsocks5://5.6.7.8:1080\n";
        let mixed = parse_proxy_list(mixed_raw, None, false);
        assert_eq!(mixed.len(), 2);
        assert_eq!(mixed[0].protocol, ProxyProtocol::Socks4);
        assert_eq!(mixed[1].protocol, ProxyProtocol::Socks5);

        let geonode = r#"{"data":[{"ip":"1.1.1.1","port":"1080","protocols":["socks5","socks4"]}]}"#;
        let json_list = parse_proxy_list(geonode, None, false);
        assert_eq!(json_list.len(), 2);
        assert_eq!(json_list[0].redacted(), "socks5://1.1.1.1:1080");
        assert_eq!(json_list[1].redacted(), "socks4://1.1.1.1:1080");
    }

    #[test]
    fn deduplicates_preserving_protocol_distinction_and_normalizing_dns() {
        let items = vec![
            ParsedProxy::parse_explicit("socks5://1.2.3.4:1080", false).unwrap(),
            ParsedProxy::parse_explicit("socks5h://1.2.3.4:1080", false).unwrap(),
            ParsedProxy::parse_explicit("http://1.2.3.4:1080", false).unwrap(),
            ParsedProxy::parse_explicit("socks4://1.2.3.4:1080", false).unwrap(),
            ParsedProxy::parse_explicit("socks4a://1.2.3.4:1080", false).unwrap(),
        ];
        let deduped = deduplicate_proxies(items);
        assert_eq!(deduped.len(), 3);
        let ep = deduped[0].clone().into_endpoint(ProxyId(1));
        assert_eq!(ep.to_wreq_url(), "socks5h://1.2.3.4:1080");
    }

    #[test]
    fn blocks_private_and_loopback_ips_from_remote_sources_unless_allowed() {
        let raw = "\
127.0.0.1:1080\n\
10.0.0.5:1080\n\
192.168.1.10:1080\n\
172.16.0.1:1080\n\
169.254.1.1:1080\n\
0.0.0.0:1080\n\
[::1]:1080\n\
localhost:1080\n\
8.8.8.8:1080\n\
";
        let strict = parse_proxy_list(raw, Some(ProxyProtocol::Socks5h), false);
        assert_eq!(strict.len(), 1);
        assert_eq!(strict[0].host, "8.8.8.8");

        let relaxed = parse_proxy_list(raw, Some(ProxyProtocol::Socks5h), true);
        assert_eq!(relaxed.len(), 9);

        // Explicit static_proxies always allow loopback/private IPs.
        let static_local = ParsedProxy::parse_explicit("socks5://127.0.0.1:10808", true).unwrap();
        assert!(static_local.is_static);
    }

    #[test]
    fn redacts_credentials_in_display_and_debug() {
        let ep = ParsedProxy::parse_explicit("socks5h://alice:supersecret@1.2.3.4:1080", false)
            .unwrap()
            .into_endpoint(ProxyId(7));
        assert_eq!(ep.redacted(), "socks5h://1.2.3.4:1080");
        assert!(!format!("{ep:?}").contains("supersecret"));
        assert!(!format!("{ep:?}").contains("alice"));
        assert_eq!(ep.to_wreq_url(), "socks5h://alice:supersecret@1.2.3.4:1080");
    }
}
