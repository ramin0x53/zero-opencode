use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::warn;
use wreq::Client;

pub const OPENCODE_MODELS_URL: &str = "https://opencode.ai/zen/v1/models";

/// Free OpenCode models that do not carry the `-free` suffix.
const KNOWN_FREE_MODELS: &[&str] = &["big-pickle"];

/// Models known to be dead/unavailable upstream.
const DEAD_FREE_MODELS: &[&str] = &["deepseek-v4-flash-free"];

/// Normalized representation of an active OpenCode free model.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FreeModel {
    pub id: String,
    pub name: String,
    pub created: u64,
    pub owned_by: String,
    pub recommended_api: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_levels: Option<Vec<String>>,
    #[serde(skip)]
    pub force_auto_tool_choice: bool,
}

/// Parsed model specification after stripping optional `oc/` / `opencode/`
/// prefix and `(level)` thinking suffix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub base_id: String,
    pub thinking_level: Option<String>,
}

/// Parse `"oc/deepseek-v4.1-flash-free(max)"` -> `("deepseek-v4.1-flash-free", Some("max"))`.
pub fn parse_model_spec(raw: &str) -> ModelSpec {
    let trimmed = raw.trim();
    let without_prefix = trimmed
        .strip_prefix("oc/")
        .or_else(|| trimmed.strip_prefix("opencode/"))
        .unwrap_or(trimmed);

    if let Some(stripped) = without_prefix.strip_suffix(')')
        && let Some(open_idx) = stripped.rfind('(')
    {
        let base = stripped[..open_idx].trim();
        let level = stripped[open_idx + 1..].trim();
        if !base.is_empty() && !level.is_empty() && !level.contains('(') {
            return ModelSpec {
                base_id: base.to_string(),
                thinking_level: Some(level.to_string()),
            };
        }
    }

    ModelSpec {
        base_id: without_prefix.trim().to_string(),
        thinking_level: None,
    }
}

#[derive(Debug, Deserialize, Default)]
struct RawModelsResponse {
    #[serde(default)]
    data: Vec<RawCatalogModel>,
}

#[derive(Debug, Deserialize, Default)]
struct RawCatalogModel {
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    owned_by: Option<String>,
    #[serde(rename = "type", default)]
    model_type: Option<String>,
    #[serde(default)]
    recommended_api: Option<String>,
    #[serde(default)]
    limit: Option<RawLimit>,
    #[serde(default)]
    capabilities: Option<RawCapabilities>,
    #[serde(default)]
    cost: Option<RawCost>,
}

#[derive(Debug, Deserialize, Default)]
struct RawLimit {
    #[serde(default)]
    context: Option<u64>,
    #[serde(default)]
    output: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
struct RawCapabilities {
    #[serde(default)]
    reasoning: Option<RawReasoning>,
}

#[derive(Debug, Deserialize, Default)]
struct RawReasoning {
    #[serde(default)]
    supported: Option<bool>,
    #[serde(default)]
    effort: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Default)]
struct RawCost {
    #[serde(default)]
    text: Option<RawTextCost>,
}

#[derive(Debug, Deserialize, Default)]
struct RawTextCost {
    #[serde(default)]
    input: Option<RawPrice>,
    #[serde(default)]
    output: Option<RawPrice>,
}

#[derive(Debug, Deserialize, Default)]
struct RawPrice {
    #[serde(default)]
    price: Option<f64>,
}

fn is_zero_cost(cost: Option<&RawCost>) -> bool {
    let Some(text) = cost.and_then(|c| c.text.as_ref()) else {
        return false;
    };
    let inp = text.input.as_ref().and_then(|p| p.price);
    let out = text.output.as_ref().and_then(|p| p.price);
    matches!((inp, out), (Some(i), Some(o)) if i == 0.0 && o == 0.0)
}

/// Filter and normalize the raw `/zen/v1/models` JSON response into active free models.
pub fn filter_free_models_json(raw_json: &str) -> anyhow::Result<Vec<FreeModel>> {
    let parsed: RawModelsResponse =
        serde_json::from_str(raw_json).context("failed to parse OpenCode models catalog JSON")?;
    Ok(filter_raw_models(parsed.data))
}

fn filter_raw_models(raw_models: Vec<RawCatalogModel>) -> Vec<FreeModel> {
    let mut out = Vec::new();
    for raw in raw_models {
        let id = raw.id.trim();
        if id.is_empty() || DEAD_FREE_MODELS.contains(&id) {
            continue;
        }
        let model_type = raw.model_type.as_deref().unwrap_or("chat");
        if model_type != "chat" {
            continue;
        }
        let free = is_zero_cost(raw.cost.as_ref())
            || id.ends_with("-free")
            || KNOWN_FREE_MODELS.contains(&id);
        if !free {
            continue;
        }

        let mut context_window = raw.limit.as_ref().and_then(|l| l.context);
        let max_output_tokens = raw.limit.as_ref().and_then(|l| l.output);
        let mut thinking_levels = raw
            .capabilities
            .as_ref()
            .and_then(|c| c.reasoning.as_ref())
            .filter(|r| r.supported.unwrap_or(false))
            .and_then(|r| r.effort.clone())
            .filter(|v| !v.is_empty());

        let mut force_auto_tool_choice = false;

        // Curated overrides from 9router (capabilities.js / thinkingLevels.js / opencode.js)
        match id {
            "deepseek-v4.1-flash-free" => {
                if let Some(ctx) = context_window {
                    context_window = Some(ctx.min(330_000));
                }
                thinking_levels = Some(vec![
                    "low".to_string(),
                    "high".to_string(),
                    "max".to_string(),
                ]);
            }
            "mimo-v2.6-flash-free" => {
                thinking_levels = Some(vec![
                    "none".to_string(),
                    "minimal".to_string(),
                    "low".to_string(),
                    "medium".to_string(),
                    "high".to_string(),
                    "xhigh".to_string(),
                    "max".to_string(),
                ]);
            }
            "muse-spark-1.3-contributor-free" => {
                force_auto_tool_choice = true;
            }
            _ => {}
        }

        out.push(FreeModel {
            id: id.to_string(),
            name: raw
                .name
                .filter(|n| !n.trim().is_empty())
                .unwrap_or_else(|| id.to_string()),
            created: raw.created.unwrap_or(0),
            owned_by: raw
                .owned_by
                .filter(|o| !o.trim().is_empty())
                .unwrap_or_else(|| "opencode".to_string()),
            recommended_api: raw
                .recommended_api
                .filter(|r| !r.trim().is_empty())
                .unwrap_or_else(|| "chat-completions".to_string()),
            context_window,
            max_output_tokens,
            thinking_levels,
            force_auto_tool_choice,
        });
    }
    out
}

#[derive(Debug, Clone)]
struct CachedModels {
    models: Vec<FreeModel>,
    fetched_at: Instant,
}

/// Caching live catalog client for OpenCode's free models.
#[derive(Clone)]
pub struct ModelCatalog {
    client: Client,
    url: String,
    ttl: Duration,
    cache: Arc<RwLock<Option<CachedModels>>>,
}

impl ModelCatalog {
    pub fn new(client: Client, ttl: Duration) -> Self {
        Self::with_url(client, OPENCODE_MODELS_URL, ttl)
    }

    pub fn with_url(client: Client, url: impl Into<String>, ttl: Duration) -> Self {
        Self {
            client,
            url: url.into(),
            ttl,
            cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Return all active free models, refreshing from upstream if cache is expired.
    pub async fn list(&self) -> anyhow::Result<Vec<FreeModel>> {
        {
            let guard = self.cache.read().await;
            if let Some(cached) = guard.as_ref()
                && cached.fetched_at.elapsed() < self.ttl
            {
                return Ok(cached.models.clone());
            }
        }

        match self.fetch_fresh().await {
            Ok(models) => {
                let mut guard = self.cache.write().await;
                *guard = Some(CachedModels {
                    models: models.clone(),
                    fetched_at: Instant::now(),
                });
                Ok(models)
            }
            Err(err) => {
                let guard = self.cache.read().await;
                if let Some(stale) = guard.as_ref() {
                    warn!(
                        error = %err,
                        "failed to refresh OpenCode model catalog; serving stale cache"
                    );
                    return Ok(stale.models.clone());
                }
                Err(err)
            }
        }
    }

    /// Look up a free model by raw request ID (strips optional `oc/` prefix and `(level)` suffix).
    pub async fn get(&self, raw_id: &str) -> anyhow::Result<Option<FreeModel>> {
        let spec = parse_model_spec(raw_id);
        let models = self.list().await?;
        Ok(models.into_iter().find(|m| m.id == spec.base_id))
    }

    /// Best-effort cached model count for `/health` without blocking on network.
    pub fn cached_count(&self) -> usize {
        self.cache
            .try_read()
            .ok()
            .and_then(|g| g.as_ref().map(|c| c.models.len()))
            .unwrap_or(0)
    }

    async fn fetch_fresh(&self) -> anyhow::Result<Vec<FreeModel>> {
        let res = self
            .client
            .get(&self.url)
            .send()
            .await
            .with_context(|| format!("GET {} failed", self.url))?;
        let status = res.status();
        let body = res
            .text()
            .await
            .with_context(|| format!("failed reading body from {}", self.url))?;
        if !status.is_success() {
            bail!("{} returned HTTP {}: {}", self.url, status.as_u16(), body);
        }
        filter_free_models_json(&body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_spec_prefixes_and_thinking_suffixes() {
        assert_eq!(
            parse_model_spec("big-pickle"),
            ModelSpec {
                base_id: "big-pickle".into(),
                thinking_level: None
            }
        );
        assert_eq!(
            parse_model_spec("oc/deepseek-v4.1-flash-free(max)"),
            ModelSpec {
                base_id: "deepseek-v4.1-flash-free".into(),
                thinking_level: Some("max".into())
            }
        );
        assert_eq!(
            parse_model_spec("opencode/mimo-v2.6-flash-free(xhigh)"),
            ModelSpec {
                base_id: "mimo-v2.6-flash-free".into(),
                thinking_level: Some("xhigh".into())
            }
        );
    }

    #[test]
    fn filters_paid_dead_and_non_chat_models_and_applies_overrides() {
        let sample = r#"{
            "object": "list",
            "data": [
                {
                    "id": "big-pickle",
                    "name": "Big Pickle",
                    "created": 1795077329,
                    "owned_by": "opencode",
                    "type": "chat",
                    "limit": {"context": 200000, "output": 32000},
                    "capabilities": {"reasoning": {"supported": true}},
                    "cost": {"text": {"input": {"price": 0}, "output": {"price": 0}}},
                    "recommended_api": "chat-completions"
                },
                {
                    "id": "deepseek-v4.1-flash-free",
                    "name": "DeepSeek V4.1 Flash Free",
                    "type": "chat",
                    "limit": {"context": 1000000, "output": 32000},
                    "capabilities": {"reasoning": {"supported": true, "effort": ["low", "high", "max"]}},
                    "cost": {"text": {"input": {"price": 0}, "output": {"price": 0}}}
                },
                {
                    "id": "mimo-v2.6-flash-free",
                    "type": "chat",
                    "limit": {"context": 200000, "output": 32000},
                    "capabilities": {"reasoning": {"supported": true}},
                    "cost": {"text": {"input": {"price": 0}, "output": {"price": 0}}}
                },
                {
                    "id": "deepseek-v4-flash-free",
                    "type": "chat",
                    "cost": {"text": {"input": {"price": 0}, "output": {"price": 0}}}
                },
                {
                    "id": "claude-opus-5",
                    "type": "chat",
                    "cost": {"text": {"input": {"price": 15.0}, "output": {"price": 75.0}}}
                },
                {
                    "id": "jev-1.13-free",
                    "type": "systemone",
                    "cost": {"text": {"input": {"price": 0}, "output": {"price": 0}}}
                }
            ]
        }"#;

        let models = filter_free_models_json(sample).expect("sample should parse");
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "big-pickle",
                "deepseek-v4.1-flash-free",
                "mimo-v2.6-flash-free"
            ]
        );

        assert_eq!(models[1].context_window, Some(330_000));
        assert_eq!(
            models[1].thinking_levels.as_deref(),
            Some(&["low".into(), "high".into(), "max".into()][..])
        );
        assert_eq!(
            models[2].thinking_levels.as_deref(),
            Some(
                &[
                    "none".into(),
                    "minimal".into(),
                    "low".into(),
                    "medium".into(),
                    "high".into(),
                    "xhigh".into(),
                    "max".into(),
                ][..]
            )
        );
    }
}
