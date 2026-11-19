use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use serde::Serialize;
use serde_json::{Map, Value, json};
use tokio::sync::RwLock;
use tracing::warn;
use wreq::Client;

pub const OPENCODE_MODELS_URL: &str = "https://opencode.ai/zen/v1/models";

/// Free OpenCode models that do not carry the `-free` suffix.
const KNOWN_FREE_MODELS: &[&str] = &["big-pickle"];

/// Models known to be dead/unavailable upstream.
const DEAD_FREE_MODELS: &[&str] = &["deepseek-v4-flash-free"];

const DEFAULT_CONTEXT_WINDOW: u64 = 200_000;
const DEFAULT_MAX_OUTPUT: u64 = 64_000;

/// 9router-standard capabilities object enriched with OpenCode's native flags.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelCapabilities {
    // Input modalities (9router standard)
    pub vision: bool,
    pub pdf: bool,
    #[serde(rename = "audioInput")]
    pub audio_input: bool,
    #[serde(rename = "videoInput")]
    pub video_input: bool,
    // Output modalities (9router standard)
    #[serde(rename = "imageOutput")]
    pub image_output: bool,
    #[serde(rename = "audioOutput")]
    pub audio_output: bool,
    // Features (9router standard)
    pub search: bool,
    pub tools: bool,
    pub reasoning: bool,
    #[serde(rename = "thinkingFormat")]
    pub thinking_format: Option<String>,
    #[serde(rename = "thinkingCanDisable")]
    pub thinking_can_disable: bool,
    #[serde(rename = "thinkingRange")]
    pub thinking_range: Option<Value>,
    #[serde(rename = "thinkingEffortSupported")]
    pub thinking_effort_supported: bool,
    // Limits (9router camelCase standard)
    #[serde(rename = "contextWindow")]
    pub context_window: u64,
    #[serde(rename = "maxOutput")]
    pub max_output: u64,
    // Additional OpenCode native capability flags (preserved when present)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_output: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_updates: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort_updates: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_toggle: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_strict: Option<bool>,
}

/// Normalized representation of an active OpenCode free model, retaining the
/// full upstream OpenCode catalog entry for `/v1/models`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeModel {
    pub id: String,
    pub name: String,
    pub created: u64,
    pub owned_by: String,
    pub recommended_api: String,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub thinking_levels: Option<Vec<String>>,
    pub capabilities: ModelCapabilities,
    pub raw: Map<String, Value>,
    pub force_auto_tool_choice: bool,
}

impl FreeModel {
    /// Serialize into the combined 9router-standard + full OpenCode metadata JSON shape.
    pub fn to_api_json(&self) -> Value {
        let mut map = self.raw.clone();
        map.insert("id".to_string(), Value::String(self.id.clone()));
        map.insert("object".to_string(), Value::String("model".to_string()));
        map.insert("created".to_string(), json!(self.created));
        map.insert("owned_by".to_string(), Value::String(self.owned_by.clone()));
        map.insert("name".to_string(), Value::String(self.name.clone()));
        map.insert(
            "recommended_api".to_string(),
            Value::String(self.recommended_api.clone()),
        );
        map.insert(
            "capabilities".to_string(),
            serde_json::to_value(&self.capabilities).expect("ModelCapabilities serializes"),
        );
        map.insert(
            "context_length".to_string(),
            json!(self.capabilities.context_window),
        );
        map.insert(
            "max_completion_tokens".to_string(),
            json!(self.capabilities.max_output),
        );
        if let Some(levels) = &self.thinking_levels
            && !levels.is_empty()
        {
            map.insert("thinking_levels".to_string(), json!(levels));
        } else {
            map.remove("thinking_levels");
        }
        Value::Object(map)
    }

    /// Concise summary tags for CLI table display.
    pub fn capability_tags(&self) -> String {
        let mut tags = Vec::new();
        if self.capabilities.vision {
            tags.push("vision");
        }
        if self.capabilities.audio_input {
            tags.push("audio");
        }
        if self.capabilities.video_input {
            tags.push("video");
        }
        if self.capabilities.pdf {
            tags.push("pdf");
        }
        if self.capabilities.tools {
            tags.push("tools");
        }
        if self.capabilities.reasoning {
            tags.push("reasoning");
        }
        if self.capabilities.structured_output == Some(true) {
            tags.push("json");
        }
        if tags.is_empty() {
            "text".to_string()
        } else {
            tags.join(",")
        }
    }
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

fn is_zero_cost(raw: &Map<String, Value>) -> bool {
    let Some(text) = raw
        .get("cost")
        .and_then(|c| c.get("text"))
        .and_then(Value::as_object)
    else {
        return false;
    };
    let inp = text
        .get("input")
        .and_then(|i| i.get("price"))
        .and_then(Value::as_f64);
    let out = text
        .get("output")
        .and_then(|o| o.get("price"))
        .and_then(Value::as_f64);
    matches!((inp, out), (Some(i), Some(o)) if i == 0.0 && o == 0.0)
}

fn string_list(val: Option<&Value>) -> Vec<String> {
    val.and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn has_modality(list: &[String], prefix: &str, exact: &[&str]) -> bool {
    list.iter().any(|item| {
        let lower = item.to_ascii_lowercase();
        exact.iter().any(|e| lower == *e) || lower.starts_with(prefix)
    })
}

/// Filter and normalize the raw `/zen/v1/models` JSON response into active free models.
pub fn filter_free_models_json(raw_json: &str) -> anyhow::Result<Vec<FreeModel>> {
    let parsed: Value =
        serde_json::from_str(raw_json).context("failed to parse OpenCode models catalog JSON")?;
    let items = parsed
        .get("data")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Ok(filter_raw_models(items))
}

fn filter_raw_models(raw_models: Vec<Value>) -> Vec<FreeModel> {
    let mut out = Vec::new();
    for item in raw_models {
        let Value::Object(mut raw) = item else {
            continue;
        };
        let id = raw
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or("")
            .to_string();
        if id.is_empty() || DEAD_FREE_MODELS.contains(&id.as_str()) {
            continue;
        }
        let model_type = raw.get("type").and_then(Value::as_str).unwrap_or("chat");
        if model_type != "chat" {
            continue;
        }
        let free = is_zero_cost(&raw)
            || id.ends_with("-free")
            || KNOWN_FREE_MODELS.contains(&id.as_str());
        if !free {
            continue;
        }

        let name = raw
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(&id)
            .to_string();
        let created = raw.get("created").and_then(Value::as_u64).unwrap_or(0);
        let owned_by = raw
            .get("owned_by")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("opencode")
            .to_string();
        let recommended_api = raw
            .get("recommended_api")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("chat-completions")
            .to_string();

        let mod_input = string_list(raw.get("modalities").and_then(|m| m.get("input")));
        let mod_output = string_list(raw.get("modalities").and_then(|m| m.get("output")));

        let vision = has_modality(&mod_input, "image/", &["image"]);
        let pdf = has_modality(&mod_input, "application/pdf", &["pdf", "application/pdf"]);
        let audio_input = has_modality(&mod_input, "audio/", &["audio"]);
        let video_input = has_modality(&mod_input, "video/", &["video"]);
        let image_output = has_modality(&mod_output, "image/", &["image"]);
        let audio_output = has_modality(&mod_output, "audio/", &["audio"]);

        let raw_caps = raw.get("capabilities").and_then(Value::as_object);
        let tools = raw_caps
            .and_then(|c| c.get("tools"))
            .and_then(|t| t.get("supported"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let reasoning_obj = raw_caps
            .and_then(|c| c.get("reasoning"))
            .and_then(Value::as_object);
        let reasoning = reasoning_obj
            .and_then(|r| r.get("supported"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let reasoning_toggle = reasoning_obj
            .and_then(|r| r.get("toggle"))
            .and_then(Value::as_bool);
        let structured_output = raw_caps
            .and_then(|c| c.get("structured_output"))
            .and_then(Value::as_bool);
        let temperature = raw_caps
            .and_then(|c| c.get("temperature"))
            .and_then(Value::as_bool);
        let top_p = raw_caps.and_then(|c| c.get("top_p")).and_then(Value::as_bool);
        let top_k = raw_caps.and_then(|c| c.get("top_k")).and_then(Value::as_bool);
        let system_updates = raw_caps
            .and_then(|c| c.get("system_updates"))
            .and_then(Value::as_bool);
        let effort_updates = raw_caps
            .and_then(|c| c.get("effort_updates"))
            .and_then(Value::as_bool);

        let mut thinking_levels = if reasoning {
            let levels = string_list(reasoning_obj.and_then(|r| r.get("effort")));
            if levels.is_empty() { None } else { Some(levels) }
        } else {
            None
        };

        let mut context_window = raw
            .get("limit")
            .and_then(|l| l.get("context"))
            .and_then(Value::as_u64);
        let max_output_tokens = raw
            .get("limit")
            .and_then(|l| l.get("output"))
            .and_then(Value::as_u64);

        let mut force_auto_tool_choice = false;

        // Curated overrides from 9router (capabilities.js / thinkingLevels.js / opencode.js)
        match id.as_str() {
            "deepseek-v4.1-flash-free" => {
                if let Some(ctx) = context_window {
                    let capped = ctx.min(330_000);
                    context_window = Some(capped);
                    if let Some(limit_obj) = raw.get_mut("limit").and_then(Value::as_object_mut) {
                        limit_obj.insert("context".to_string(), json!(capped));
                    }
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

        let thinking_format = if !reasoning {
            None
        } else if id == "deepseek-v4.1-flash-free" {
            Some("deepseek".to_string())
        } else {
            Some("openai".to_string())
        };

        let thinking_can_disable = match id.as_str() {
            "glm-5.3-flash-free" => false,
            "mimo-v2.6-flash-free" => true,
            _ => reasoning_toggle.unwrap_or_else(|| {
                thinking_levels
                    .as_ref()
                    .is_none_or(|levels| levels.iter().any(|l| l == "none"))
            }),
        };

        let thinking_effort_supported = thinking_levels.as_ref().is_some_and(|v| !v.is_empty());

        let api_tools = raw
            .get("api")
            .and_then(|a| a.get(&recommended_api))
            .and_then(|r| r.get("capabilities"))
            .and_then(|c| c.get("tools"))
            .and_then(Value::as_object);
        let tool_choice = api_tools
            .map(|t| string_list(t.get("choice")))
            .filter(|v| !v.is_empty());
        let tool_strict = api_tools.and_then(|t| t.get("strict")).and_then(Value::as_bool);

        let capabilities = ModelCapabilities {
            vision,
            pdf,
            audio_input,
            video_input,
            image_output,
            audio_output,
            search: false,
            tools,
            reasoning,
            thinking_format,
            thinking_can_disable,
            thinking_range: None,
            thinking_effort_supported,
            context_window: context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW),
            max_output: max_output_tokens.unwrap_or(DEFAULT_MAX_OUTPUT),
            structured_output,
            temperature,
            top_p,
            top_k,
            system_updates,
            effort_updates,
            reasoning_toggle,
            tool_choice,
            tool_strict,
        };

        out.push(FreeModel {
            id,
            name,
            created,
            owned_by,
            recommended_api,
            context_window,
            max_output_tokens,
            thinking_levels,
            capabilities,
            raw,
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

    /// Pre-seed the catalog cache (useful for offline unit/router tests).
    pub async fn seed(&self, models: Vec<FreeModel>) {
        let mut guard = self.cache.write().await;
        *guard = Some(CachedModels {
            models,
            fetched_at: Instant::now(),
        });
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
    fn filters_models_derives_modalities_and_preserves_all_opencode_metadata() {
        let sample = r#"{
            "object": "list",
            "data": [
                {
                    "id": "big-pickle",
                    "name": "Big Pickle",
                    "created": 1795077329,
                    "owned_by": "opencode",
                    "type": "chat",
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "limit": {"context": 200000, "output": 32000},
                    "capabilities": {"tools": {"supported": true}, "reasoning": {"supported": true}, "structured_output": true},
                    "cost": {"text": {"input": {"price": 0}, "output": {"price": 0}}},
                    "recommended_api": "chat-completions"
                },
                {
                    "id": "deepseek-v4.1-flash-free",
                    "name": "DeepSeek V4.1 Flash Free",
                    "canonical_id": "deepseek/deepseek-v4.1-flash-1017",
                    "description": "DeepSeek model",
                    "family": "deepseek-flash",
                    "open_weights": true,
                    "release_date": "2026-10-17",
                    "last_updated": "2026-10-17",
                    "knowledge": "2025-05",
                    "type": "chat",
                    "modalities": {"input": ["text", "image/jpeg", "application/pdf"], "output": ["text"]},
                    "limit": {"context": 1000000, "output": 32000},
                    "capabilities": {
                        "tools": {"supported": true},
                        "reasoning": {"supported": true, "toggle": true, "effort": ["low", "high", "max"]},
                        "structured_output": true,
                        "temperature": true
                    },
                    "cost": {"text": {"input": {"price": 0}, "output": {"price": 0}}},
                    "api": {
                        "chat-completions": {
                            "base_url": "https://opencode.ai/inference/openai/v1",
                            "capabilities": {"tools": {"choice": ["auto", "required"]}}
                        }
                    },
                    "future_opencode_field": "kept-intact"
                },
                {
                    "id": "mimo-v2.6-flash-free",
                    "type": "chat",
                    "modalities": {
                        "input": ["text", "image/png", "audio/mpeg", "video/mp4"],
                        "output": ["text"]
                    },
                    "limit": {"context": 200000, "output": 32000},
                    "capabilities": {"tools": {"supported": true}, "reasoning": {"supported": true}},
                    "cost": {"text": {"input": {"price": 0}, "output": {"price": 0}}},
                    "api": {
                        "chat-completions": {
                            "capabilities": {"tools": {"strict": true, "choice": ["auto"]}}
                        }
                    }
                },
                {
                    "id": "longcat-2.5-free",
                    "type": "chat",
                    "modalities": {"input": ["text", "image"], "output": ["text", "image"]},
                    "limit": {"context": 1000000, "output": 262144},
                    "capabilities": {"tools": {"supported": true}, "reasoning": {"supported": true, "toggle": true}},
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
                "mimo-v2.6-flash-free",
                "longcat-2.5-free"
            ]
        );

        // big-pickle: text-only vision=false
        assert!(!models[0].capabilities.vision);
        assert!(models[0].capabilities.reasoning);

        // deepseek-v4.1-flash-free: capped context consistently, mime vision + pdf, raw metadata kept
        let ds_json = models[1].to_api_json();
        assert_eq!(ds_json["context_length"], 330_000);
        assert_eq!(ds_json["max_completion_tokens"], 32_000);
        assert_eq!(ds_json["limit"]["context"], 330_000);
        assert_eq!(ds_json["capabilities"]["contextWindow"], 330_000);
        assert_eq!(ds_json["capabilities"]["maxOutput"], 32_000);
        assert_eq!(ds_json["capabilities"]["vision"], true);
        assert_eq!(ds_json["capabilities"]["pdf"], true);
        assert_eq!(ds_json["capabilities"]["thinkingFormat"], "deepseek");
        assert_eq!(ds_json["capabilities"]["thinkingCanDisable"], true);
        assert_eq!(ds_json["capabilities"]["thinkingEffortSupported"], true);
        assert_eq!(ds_json["capabilities"]["structured_output"], true);
        assert_eq!(ds_json["capabilities"]["tool_choice"], json!(["auto", "required"]));
        assert_eq!(ds_json["thinking_levels"], json!(["low", "high", "max"]));
        assert_eq!(ds_json["canonical_id"], "deepseek/deepseek-v4.1-flash-1017");
        assert_eq!(ds_json["family"], "deepseek-flash");
        assert_eq!(ds_json["open_weights"], true);
        assert_eq!(ds_json["future_opencode_field"], "kept-intact");
        assert!(ds_json["modalities"].is_object());
        assert!(ds_json["cost"].is_object());
        assert!(ds_json["api"].is_object());

        // mimo-v2.6-flash-free: vision + audio + video + tool_strict
        let mimo_json = models[2].to_api_json();
        assert_eq!(mimo_json["capabilities"]["vision"], true);
        assert_eq!(mimo_json["capabilities"]["audioInput"], true);
        assert_eq!(mimo_json["capabilities"]["videoInput"], true);
        assert_eq!(mimo_json["capabilities"]["tool_strict"], true);
        assert_eq!(
            mimo_json["thinking_levels"],
            json!(["none", "minimal", "low", "medium", "high", "xhigh", "max"])
        );

        // longcat-2.5-free: bare "image" input and output
        assert!(models[3].capabilities.vision);
        assert!(models[3].capabilities.image_output);
    }
}
