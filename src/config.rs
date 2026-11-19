use std::fs;
use std::path::Path;

use anyhow::{Context, bail};
use serde::Deserialize;
use url::Url;

use crate::proxy::endpoint::ParsedProxy;

/// Top-level application configuration loaded from YAML.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub opencode: OpenCodeConfig,
    pub streaming: StreamingConfig,
    pub logging: LoggingConfig,
}

/// Local HTTP server settings.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub cors: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            api_key: None,
            cors: true,
        }
    }
}

/// Upstream OpenCode Free transport and caching settings.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct OpenCodeConfig {
    pub proxy: Option<String>,
    pub timeout_secs: u64,
    pub models_ttl_secs: u64,
    pub session_ttl_secs: u64,
    pub retry: RetryConfig,
    pub proxy_pool: ProxyPoolConfig,
}

impl Default for OpenCodeConfig {
    fn default() -> Self {
        Self {
            proxy: None,
            timeout_secs: 120,
            models_ttl_secs: 600,
            session_ttl_secs: 7200,
            retry: RetryConfig::default(),
            proxy_pool: ProxyPoolConfig::default(),
        }
    }
}

/// Configuration for the health-checked rotating proxy pool.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ProxyPoolConfig {
    pub enabled: bool,
    pub sources: ProxySourcesConfig,
    pub static_proxies: Vec<String>,
    pub refresh_interval_secs: u64,
    pub max_proxies: Option<usize>,
    pub allow_private_proxies: bool,
    pub health_check: ProxyHealthCheckConfig,
    pub selection: ProxySelectionConfig,
}

impl Default for ProxyPoolConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sources: ProxySourcesConfig::default(),
            static_proxies: Vec::new(),
            refresh_interval_secs: 900,
            max_proxies: Some(2000),
            allow_private_proxies: false,
            health_check: ProxyHealthCheckConfig::default(),
            selection: ProxySelectionConfig::default(),
        }
    }
}

/// Remote URLs providing proxy lists grouped by default protocol.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ProxySourcesConfig {
    pub socks5: Vec<String>,
    pub socks4: Vec<String>,
    pub http: Vec<String>,
    pub mixed: Vec<String>,
}

impl ProxySourcesConfig {
    pub fn is_empty(&self) -> bool {
        self.socks5.is_empty()
            && self.socks4.is_empty()
            && self.http.is_empty()
            && self.mixed.is_empty()
    }
}

/// Probing & hot-standby warming settings (probe target URLs are hardcoded).
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ProxyHealthCheckConfig {
    pub concurrency: usize,
    pub timeout_secs: u64,
    pub interval_secs: u64,
    pub keepalive_interval_secs: u64,
    pub hot_standby_count: usize,
    pub target_healthy: usize,
    pub startup_wait_secs: u64,
}

impl Default for ProxyHealthCheckConfig {
    fn default() -> Self {
        Self {
            concurrency: 25,
            timeout_secs: 6,
            interval_secs: 180,
            keepalive_interval_secs: 45,
            hot_standby_count: 3,
            target_healthy: 20,
            startup_wait_secs: 15,
        }
    }
}

/// Proxy selection strategy among verified healthy proxies.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProxyStrategy {
    #[default]
    Sticky,
    LowestLatency,
    RoundRobin,
}

/// Live failover, hedging, and cooldown parameters for pooled requests.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ProxySelectionConfig {
    pub strategy: ProxyStrategy,
    pub connect_timeout_secs: u64,
    pub first_byte_timeout_secs: u64,
    pub failover_timeout_secs: Option<u64>,
    pub hedge_delay_ms: Option<u64>,
    pub cooldown_rate_limit_secs: u64,
    pub cooldown_forbidden_secs: u64,
    pub cooldown_overload_secs: u64,
    pub cooldown_error_base_secs: u64,
    pub max_consecutive_failures: u32,
    pub fallback_to_base: bool,
}

impl Default for ProxySelectionConfig {
    fn default() -> Self {
        Self {
            strategy: ProxyStrategy::Sticky,
            connect_timeout_secs: 4,
            first_byte_timeout_secs: 12,
            failover_timeout_secs: Some(90),
            hedge_delay_ms: Some(2000),
            cooldown_rate_limit_secs: 600,
            cooldown_forbidden_secs: 900,
            cooldown_overload_secs: 15,
            cooldown_error_base_secs: 60,
            max_consecutive_failures: 3,
            fallback_to_base: false,
        }
    }
}

/// Retry schedule for transient upstream `429 overloaded` responses.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct RetryConfig {
    pub max_attempts: usize,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 8,
            base_delay_ms: 800,
            max_delay_ms: 4000,
        }
    }
}

/// Streaming and SSE-to-JSON response formatting options.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct StreamingConfig {
    pub keep_reasoning_on_json: bool,
}

/// Logging configuration.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct LoggingConfig {
    pub level: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
        }
    }
}

impl AppConfig {
    /// Load configuration from an explicit path, `./config.yaml` if present, or defaults.
    pub fn load(path: Option<&Path>) -> anyhow::Result<Self> {
        let cfg = match path {
            Some(p) => Self::from_file(p)?,
            None => {
                let default_path = Path::new("config.yaml");
                if default_path.exists() {
                    Self::from_file(default_path)?
                } else {
                    Self::default()
                }
            }
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse and validate configuration from a YAML file path.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read config file {}", path.display()))?;
        Self::from_yaml_str(&raw)
            .with_context(|| format!("failed to parse config file {}", path.display()))
    }

    /// Parse and validate configuration from a YAML string.
    pub fn from_yaml_str(raw: &str) -> anyhow::Result<Self> {
        let value: yaml_serde::Value = yaml_serde::from_str(raw)?;
        let cfg: Self = if value.is_null() {
            Self::default()
        } else {
            yaml_serde::from_value(value)?
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate invariants across loaded config sections.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.server.host.trim().is_empty() {
            bail!("server.host must not be empty");
        }
        if self.server.port == 0 {
            bail!("server.port must be greater than 0");
        }
        if self.opencode.timeout_secs == 0 {
            bail!("opencode.timeout_secs must be greater than 0");
        }
        if self.opencode.retry.max_attempts == 0 {
            bail!("opencode.retry.max_attempts must be at least 1");
        }
        if let Some(proxy) = &self.opencode.proxy {
            if proxy.trim().is_empty() {
                bail!("opencode.proxy must be null or a non-empty URL");
            }
            ParsedProxy::parse_explicit(proxy, true).context("invalid opencode.proxy")?;
        }
        self.validate_proxy_pool()?;
        Ok(())
    }

    fn validate_proxy_pool(&self) -> anyhow::Result<()> {
        let pool = &self.opencode.proxy_pool;
        for (kind, list) in [
            ("socks5", &pool.sources.socks5),
            ("socks4", &pool.sources.socks4),
            ("http", &pool.sources.http),
            ("mixed", &pool.sources.mixed),
        ] {
            for raw_url in list {
                let trimmed = raw_url.trim();
                let parsed = Url::parse(trimmed).with_context(|| {
                    format!("invalid URL in opencode.proxy_pool.sources.{kind}: `{trimmed}`")
                })?;
                if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
                    bail!(
                        "opencode.proxy_pool.sources.{kind} URL must be http:// or https:// with a host: `{trimmed}`"
                    );
                }
            }
        }
        for raw_static in &pool.static_proxies {
            ParsedProxy::parse_explicit(raw_static, true).with_context(|| {
                format!("invalid entry in opencode.proxy_pool.static_proxies: `{raw_static}`")
            })?;
        }
        if pool.enabled && pool.sources.is_empty() && pool.static_proxies.is_empty() {
            bail!(
                "opencode.proxy_pool.enabled is true, but both sources.* and static_proxies are empty"
            );
        }
        if pool.refresh_interval_secs == 0 {
            bail!("opencode.proxy_pool.refresh_interval_secs must be greater than 0");
        }
        if pool.max_proxies == Some(0) {
            bail!("opencode.proxy_pool.max_proxies must be null or greater than 0");
        }
        let hc = &pool.health_check;
        if hc.concurrency == 0 {
            bail!("opencode.proxy_pool.health_check.concurrency must be at least 1");
        }
        if hc.timeout_secs == 0 {
            bail!("opencode.proxy_pool.health_check.timeout_secs must be greater than 0");
        }
        if hc.interval_secs == 0 {
            bail!("opencode.proxy_pool.health_check.interval_secs must be greater than 0");
        }
        if hc.keepalive_interval_secs == 0 {
            bail!(
                "opencode.proxy_pool.health_check.keepalive_interval_secs must be greater than 0"
            );
        }
        if hc.target_healthy == 0 {
            bail!("opencode.proxy_pool.health_check.target_healthy must be at least 1");
        }
        let sel = &pool.selection;
        if sel.connect_timeout_secs == 0 {
            bail!("opencode.proxy_pool.selection.connect_timeout_secs must be greater than 0");
        }
        if sel.first_byte_timeout_secs == 0 {
            bail!("opencode.proxy_pool.selection.first_byte_timeout_secs must be greater than 0");
        }
        if sel.failover_timeout_secs == Some(0) {
            bail!("opencode.proxy_pool.selection.failover_timeout_secs must be null or > 0");
        }
        if sel.hedge_delay_ms == Some(0) {
            bail!("opencode.proxy_pool.selection.hedge_delay_ms must be null or > 0");
        }
        if sel.max_consecutive_failures == 0 {
            bail!("opencode.proxy_pool.selection.max_consecutive_failures must be at least 1");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_example_config_file() {
        let raw = include_str!("../config.example.yaml");
        let cfg = AppConfig::from_yaml_str(raw).expect("example config should parse");
        assert_eq!(cfg, AppConfig::default());
    }

    #[test]
    fn parses_minimal_partial_override() {
        let yaml = r#"
server:
  port: 9090
  api_key: "secret-token"
"#;
        let cfg = AppConfig::from_yaml_str(yaml).expect("partial yaml should parse");
        assert_eq!(cfg.server.host, "127.0.0.1");
        assert_eq!(cfg.server.port, 9090);
        assert_eq!(cfg.server.api_key.as_deref(), Some("secret-token"));
        assert_eq!(cfg.opencode.retry.max_attempts, 8);
    }

    #[test]
    fn parses_comments_only_yaml_as_defaults() {
        let yaml = "# just a comment\n";
        let cfg = AppConfig::from_yaml_str(yaml).expect("empty/comment yaml should default");
        assert_eq!(cfg, AppConfig::default());
    }

    #[test]
    fn rejects_unknown_fields_including_models() {
        let yaml = r#"
opencode:
  models:
    include: ["big-pickle"]
"#;
        assert!(AppConfig::from_yaml_str(yaml).is_err());
    }

    #[test]
    fn errors_when_explicit_config_file_missing() {
        let err = AppConfig::load(Some(Path::new("/nonexistent/does_not_exist.yaml")))
            .expect_err("missing explicit -c file must error");
        assert!(err.to_string().contains("does_not_exist.yaml"));
    }

    #[test]
    fn rejects_invalid_values() {
        assert!(AppConfig::from_yaml_str("server:\n  port: 0\n").is_err());
        assert!(AppConfig::from_yaml_str("server:\n  host: \"   \"\n").is_err());
        assert!(
            AppConfig::from_yaml_str("opencode:\n  retry:\n    max_attempts: 0\n").is_err()
        );
        assert!(
            AppConfig::from_yaml_str("opencode:\n  proxy: \"not-a-url\"\n").is_err()
        );
    }

    #[test]
    fn parses_proxy_pool_overrides_and_null_optionals() {
        let yaml = r#"
opencode:
  proxy: "socks5://127.0.0.1:10808"
  proxy_pool:
    enabled: true
    sources:
      socks5: ["https://example.com/socks5.txt"]
      http: ["https://example.com/http.txt"]
    static_proxies:
      - "socks5h://user:pass@10.0.0.2:1080"
    max_proxies: null
    selection:
      strategy: "round_robin"
      failover_timeout_secs: null
      hedge_delay_ms: null
"#;
        let cfg = AppConfig::from_yaml_str(yaml).expect("valid proxy_pool yaml");
        let pp = &cfg.opencode.proxy_pool;
        assert!(pp.enabled);
        assert_eq!(pp.sources.socks5, vec!["https://example.com/socks5.txt"]);
        assert_eq!(pp.max_proxies, None);
        assert_eq!(pp.selection.strategy, ProxyStrategy::RoundRobin);
        assert_eq!(pp.selection.failover_timeout_secs, None);
        assert_eq!(pp.selection.hedge_delay_ms, None);
        assert_eq!(pp.health_check.concurrency, 25);
    }

    #[test]
    fn rejects_invalid_proxy_pool_configs() {
        // enabled without sources or static_proxies
        assert!(
            AppConfig::from_yaml_str("opencode:\n  proxy_pool:\n    enabled: true\n").is_err()
        );
        // bad source URL scheme
        assert!(
            AppConfig::from_yaml_str(
                "opencode:\n  proxy_pool:\n    sources:\n      socks5: [\"ftp://bad.example/list\"]\n"
            )
            .is_err()
        );
        // static proxy missing scheme
        assert!(
            AppConfig::from_yaml_str(
                "opencode:\n  proxy_pool:\n    static_proxies: [\"1.2.3.4:1080\"]\n"
            )
            .is_err()
        );
        // zero limits / timeouts
        assert!(
            AppConfig::from_yaml_str(
                "opencode:\n  proxy_pool:\n    max_proxies: 0\n"
            )
            .is_err()
        );
        assert!(
            AppConfig::from_yaml_str(
                "opencode:\n  proxy_pool:\n    selection:\n      hedge_delay_ms: 0\n"
            )
            .is_err()
        );
    }
}
