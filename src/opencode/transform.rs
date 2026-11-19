use serde_json::{Value, json};

use crate::opencode::fingerprint::{RenameMap, apply_fingerprint_tools};
use crate::opencode::models::{FreeModel, parse_model_spec};

/// Output of transforming a downstream OpenAI Chat Completions request body
/// for OpenCode's free tier.
#[derive(Debug, Clone)]
pub struct TransformedRequest {
    pub model: String,
    pub rename_map: RenameMap,
}

/// Transform a Chat Completions JSON request body in place for OpenCode Free.
pub fn transform_chat_request(
    body: &mut Value,
    raw_model: &str,
    model_meta: Option<&FreeModel>,
) -> TransformedRequest {
    let spec = parse_model_spec(raw_model);
    let base_model = spec.base_id;

    if let Some(obj) = body.as_object_mut() {
        obj.insert("model".to_string(), Value::String(base_model.clone()));
        if let Some(level) = spec.thinking_level {
            obj.insert("reasoning_effort".to_string(), Value::String(level));
        }
        // Zen/inference rejects stream=false on free models with 403 FreeTierError.
        obj.insert("stream".to_string(), Value::Bool(true));

        normalize_thinking_fields(obj, &base_model);

        let force_auto = model_meta.is_some_and(|m| m.force_auto_tool_choice)
            || base_model == "muse-spark-1.3-contributor-free";
        if force_auto
            && let Some(choice) = obj.get("tool_choice")
            && choice.as_str() != Some("auto")
        {
            obj.insert("tool_choice".to_string(), Value::String("auto".to_string()));
        }
    }

    let rename_map = apply_fingerprint_tools(body, false);
    inject_reasoning_content(body);

    TransformedRequest {
        model: base_model,
        rename_map,
    }
}

fn normalize_thinking_fields(obj: &mut serde_json::Map<String, Value>, base_model: &str) {
    // Strip Claude-style adaptive thinking marker rather than turning it into "auto",
    // which OpenCode's chat lane rejects with 400 "Invalid option".
    if obj
        .get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(Value::as_str)
        == Some("adaptive")
    {
        obj.remove("thinking");
    }

    if obj.get("reasoning_effort").and_then(Value::as_str) == Some("auto") {
        obj.remove("reasoning_effort");
    }

    match base_model {
        "deepseek-v4.1-flash-free" => {
            if let Some(effort) = obj
                .get("reasoning_effort")
                .and_then(Value::as_str)
                .map(|s| s.trim().to_ascii_lowercase())
            {
                match effort.as_str() {
                    "none" => {
                        obj.remove("reasoning_effort");
                        obj.insert("thinking".to_string(), json!({ "type": "disabled" }));
                    }
                    "minimal" | "low" => {
                        obj.insert(
                            "reasoning_effort".to_string(),
                            Value::String("low".to_string()),
                        );
                    }
                    "medium" | "high" => {
                        obj.insert(
                            "reasoning_effort".to_string(),
                            Value::String("high".to_string()),
                        );
                    }
                    "xhigh" | "max" => {
                        obj.insert(
                            "reasoning_effort".to_string(),
                            Value::String("max".to_string()),
                        );
                    }
                    _ => {}
                }
            }
        }
        "glm-5.3-flash-free" => {
            // OpenCode serves glm-5.3-flash-free with reasoning_effort; sending a z.ai
            // `thinking` object causes 400 unknown field "thinking", and GLM 5.3 cannot disable thinking.
            obj.remove("thinking");
            if obj
                .get("reasoning_effort")
                .and_then(Value::as_str)
                .is_some_and(|e| e.eq_ignore_ascii_case("none"))
            {
                obj.remove("reasoning_effort");
            }
        }
        "mimo-v2.6-flash-free" => {
            // OpenRouter-backed MiMo route takes plain OpenAI reasoning_effort verbatim
            // (none|minimal|low|medium|high|xhigh|max) and ignores/rejects zai/deepseek `thinking`.
            obj.remove("thinking");
        }
        _ => {}
    }
}

/// Ensure every prior assistant message carries a non-empty `reasoning_content`
/// placeholder so Zen multi-turn reasoning endpoints do not return 400.
pub fn inject_reasoning_content(body: &mut Value) {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    for msg in messages {
        let Some(obj) = msg.as_object_mut() else {
            continue;
        };
        if obj.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let has_non_empty = obj
            .get("reasoning_content")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty());
        if !has_non_empty {
            obj.insert("reasoning_content".to_string(), Value::String(" ".to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forces_stream_true_and_strips_oc_prefix_and_suffix() {
        let mut body = json!({
            "model": "oc/mimo-v2.6-flash-free(xhigh)",
            "stream": false,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let transformed = transform_chat_request(&mut body, "oc/mimo-v2.6-flash-free(xhigh)", None);
        assert_eq!(transformed.model, "mimo-v2.6-flash-free");
        assert_eq!(body["model"], "mimo-v2.6-flash-free");
        assert_eq!(body["stream"], true);
        assert_eq!(body["reasoning_effort"], "xhigh");
    }

    #[test]
    fn drops_adaptive_thinking_and_auto_effort() {
        let mut body = json!({
            "model": "mimo-v2.6-flash-free",
            "thinking": { "type": "adaptive" },
            "reasoning_effort": "auto",
            "messages": [{"role": "user", "content": "hi"}]
        });
        transform_chat_request(&mut body, "mimo-v2.6-flash-free", None);
        assert!(body.get("thinking").is_none());
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn deepseek_v4_1_flash_free_maps_efforts_and_none_to_disabled_thinking() {
        for (input, expected) in [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "high"),
            ("high", "high"),
            ("xhigh", "max"),
            ("max", "max"),
        ] {
            let mut body = json!({
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": input
            });
            transform_chat_request(&mut body, "deepseek-v4.1-flash-free", None);
            assert_eq!(body["reasoning_effort"], expected);
        }

        let mut off_body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "none"
        });
        transform_chat_request(&mut off_body, "deepseek-v4.1-flash-free", None);
        assert!(off_body.get("reasoning_effort").is_none());
        assert_eq!(off_body["thinking"], json!({"type": "disabled"}));
    }

    #[test]
    fn glm_5_3_flash_free_drops_thinking_object_and_none_effort() {
        let mut body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": { "type": "enabled" },
            "reasoning_effort": "high"
        });
        transform_chat_request(&mut body, "glm-5.3-flash-free", None);
        assert!(body.get("thinking").is_none());
        assert_eq!(body["reasoning_effort"], "high");

        let mut none_body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "none"
        });
        transform_chat_request(&mut none_body, "glm-5.3-flash-free", None);
        assert!(none_body.get("reasoning_effort").is_none());
    }

    #[test]
    fn injects_reasoning_content_placeholder_on_assistant_turns() {
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "1"},
                {"role": "assistant", "content": "2"},
                {"role": "assistant", "content": "3", "reasoning_content": "real"},
                {"role": "user", "content": "4"}
            ]
        });
        transform_chat_request(&mut body, "big-pickle", None);
        assert_eq!(body["messages"][2]["reasoning_content"], " ");
        assert_eq!(body["messages"][3]["reasoning_content"], "real");
    }
}
