use std::fs;
use std::path::Path;

use anyhow::{Context, bail};
use serde::Deserialize;

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
}

impl Default for OpenCodeConfig {
    fn default() -> Self {
        Self {
            proxy: None,
            timeout_secs: 120,
            models_ttl_secs: 600,
            session_ttl_secs: 7200,
            retry: RetryConfig::default(),
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
        if let Some(proxy) = &self.opencode.proxy
            && proxy.trim().is_empty()
        {
            bail!("opencode.proxy must be null or a non-empty URL");
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
    }
}
