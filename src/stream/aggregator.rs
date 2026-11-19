use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use futures_util::StreamExt;
use http::StatusCode;
use serde_json::{Map, Value, json};

use crate::opencode::executor::UpstreamBody;
use crate::opencode::fingerprint::{RenameMap, restore_tool_names};
use crate::stream::passthrough::extract_reasoning_text;
use crate::stream::sse_parser::{SseLineParser, extract_sse_data};

/// Errors produced while aggregating an upstream SSE response into JSON.
#[derive(Debug, thiserror::Error)]
pub enum AggregatorError {
    #[error("upstream stream read failed: {0}")]
    Transport(#[from] anyhow::Error),
    #[error("upstream returned empty or invalid SSE stream")]
    EmptyStream,
    #[error("upstream SSE error ({status})")]
    UpstreamError { status: StatusCode, body: Value },
}

#[derive(Debug, Default)]
struct ToolCallAcc {
    id: String,
    name: String,
    arguments: String,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// Aggregate an upstream OpenAI-style SSE byte stream into a single non-streaming
/// `chat.completion` JSON response.
pub async fn aggregate_sse_to_json(
    upstream: UpstreamBody,
    fallback_model: &str,
    rename_map: &RenameMap,
    keep_reasoning_on_json: bool,
) -> Result<Value, AggregatorError> {
    let mut parser = SseLineParser::new();
    let mut lines = Vec::new();

    match upstream {
        UpstreamBody::Buffered(bytes) => {
            lines.extend(parser.push_chunk(&bytes));
            if let Some(last) = parser.finish() {
                lines.push(last);
            }
        }
        UpstreamBody::Stream(mut stream) => {
            while let Some(chunk) = stream.next().await {
                let bytes = chunk?;
                lines.extend(parser.push_chunk(&bytes));
            }
            if let Some(last) = parser.finish() {
                lines.push(last);
            }
        }
    }

    aggregate_lines(&lines, fallback_model, rename_map, keep_reasoning_on_json)
}

/// Pure SSE-line aggregator (also directly unit testable).
pub fn aggregate_lines(
    lines: &[String],
    fallback_model: &str,
    rename_map: &RenameMap,
    keep_reasoning_on_json: bool,
) -> Result<Value, AggregatorError> {
    let mut chunks: Vec<Value> = Vec::new();
    let mut stream_error: Option<Value> = None;

    for line in lines {
        let Some(payload) = extract_sse_data(line) else {
            continue;
        };
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        let Ok(val) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        if let Some(err) = val.get("error") {
            stream_error = Some(err.clone());
        } else {
            chunks.push(val);
        }
    }

    if let Some(err) = stream_error {
        let code = err
            .get("status")
            .and_then(Value::as_u64)
            .and_then(|s| u16::try_from(s).ok())
            .filter(|&s| (400..=599).contains(&s))
            .and_then(|s| StatusCode::from_u16(s).ok())
            .unwrap_or(StatusCode::BAD_GATEWAY);
        return Err(AggregatorError::UpstreamError {
            status: code,
            body: json!({ "error": err }),
        });
    }

    if chunks.is_empty() {
        return Err(AggregatorError::EmptyStream);
    }

    let first = &chunks[0];
    let id = first
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("chatcmpl-{}", now_ms()));
    let created = first
        .get("created")
        .and_then(Value::as_u64)
        .unwrap_or_else(now_secs);
    let model = first
        .get("model")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(fallback_model)
        .to_string();

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: BTreeMap<u64, ToolCallAcc> = BTreeMap::new();
    let mut finish_reason: Option<String> = None;
    let mut usage: Option<Value> = None;

    for chunk in &chunks {
        if let Some(u) = chunk.get("usage").filter(|u| u.is_object()) {
            usage = Some(u.clone());
        }
        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
        else {
            continue;
        };
        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str)
            && !fr.is_empty()
        {
            finish_reason = Some(fr.to_string());
        }

        let Some(delta) = choice.get("delta") else {
            continue;
        };
        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            content.push_str(text);
        }
        let r = extract_reasoning_text(delta);
        if !r.is_empty() {
            reasoning.push_str(&r);
        }

        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tc in calls {
                let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
                let entry = tool_calls.entry(idx).or_default();
                if let Some(tc_id) = tc.get("id").and_then(Value::as_str)
                    && !tc_id.is_empty()
                {
                    entry.id = tc_id.to_string();
                }
                if let Some(fn_obj) = tc.get("function") {
                    if let Some(name) = fn_obj.get("name").and_then(Value::as_str) {
                        entry.name.push_str(name);
                    }
                    if let Some(args) = fn_obj.get("arguments").and_then(Value::as_str) {
                        entry.arguments.push_str(args);
                    }
                }
            }
        }
    }

    let mut message = Map::new();
    message.insert("role".to_string(), Value::String("assistant".to_string()));
    if content.is_empty() && !tool_calls.is_empty() {
        message.insert("content".to_string(), Value::Null);
    } else {
        message.insert("content".to_string(), Value::String(content.clone()));
    }

    // Preserve reasoning_content when explicitly configured OR when content is empty
    // (reasoning-only completion where token budget ran out during thinking).
    if !reasoning.is_empty() && (keep_reasoning_on_json || content.is_empty()) {
        message.insert("reasoning_content".to_string(), Value::String(reasoning));
    }

    if !tool_calls.is_empty() {
        let stamp = now_ms();
        let arr: Vec<Value> = tool_calls
            .into_iter()
            .map(|(idx, acc)| {
                let call_id = if acc.id.is_empty() {
                    format!("call_{stamp}_{idx}")
                } else {
                    acc.id
                };
                json!({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": acc.name,
                        "arguments": acc.arguments,
                    }
                })
            })
            .collect();
        message.insert("tool_calls".to_string(), Value::Array(arr));
    }

    let resolved_finish = finish_reason.unwrap_or_else(|| {
        if message.contains_key("tool_calls") {
            "tool_calls".to_string()
        } else {
            "stop".to_string()
        }
    });

    let mut result = json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": Value::Object(message),
            "finish_reason": resolved_finish
        }]
    });
    if let Some(u) = usage {
        result
            .as_object_mut()
            .expect("object")
            .insert("usage".to_string(), u);
    }

    restore_tool_names(&mut result, rename_map);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_lines(chunks: &[Value]) -> Vec<String> {
        let mut out: Vec<String> = chunks
            .iter()
            .map(|c| format!("data: {}", serde_json::to_string(c).unwrap()))
            .collect();
        out.push("data: [DONE]".to_string());
        out
    }

    #[test]
    fn aggregates_content_and_strips_reasoning_when_content_non_empty_by_default() {
        let lines = to_lines(&[
            json!({"id":"c1","created":10,"model":"mimo-v2.6-flash-free","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning":"17*23"}}]}),
            json!({"id":"c1","created":10,"model":"mimo-v2.6-flash-free","choices":[{"index":0,"delta":{"content":"391"},"finish_reason":"stop"}]}),
            json!({"id":"c1","created":10,"model":"mimo-v2.6-flash-free","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}),
        ]);

        let stripped = aggregate_lines(&lines, "fallback", &RenameMap::new(), false).unwrap();
        assert_eq!(stripped["choices"][0]["message"]["content"], "391");
        assert!(stripped["choices"][0]["message"].get("reasoning_content").is_none());
        assert_eq!(stripped["usage"]["total_tokens"], 8);

        let kept = aggregate_lines(&lines, "fallback", &RenameMap::new(), true).unwrap();
        assert_eq!(kept["choices"][0]["message"]["reasoning_content"], "17*23");
    }

    #[test]
    fn preserves_reasoning_when_content_is_empty() {
        let lines = to_lines(&[
            json!({"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning_details":[{"text":"thinking only"}]}}]}),
        ]);
        let res = aggregate_lines(&lines, "big-pickle", &RenameMap::new(), false).unwrap();
        assert_eq!(res["choices"][0]["message"]["content"], "");
        assert_eq!(
            res["choices"][0]["message"]["reasoning_content"],
            "thinking only"
        );
    }

    #[test]
    fn aggregates_tool_calls_and_restores_aliased_names() {
        let mut map = RenameMap::new();
        map.insert("run_command".into(), "terminal".into());

        let lines = to_lines(&[
            json!({"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"run_command","arguments":"{\"cmd\":"}}]}}]}),
            json!({"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"pwd\"}"}}]},"finish_reason":"tool_calls"}]}),
        ]);

        let res = aggregate_lines(&lines, "big-pickle", &map, false).unwrap();
        let msg = &res["choices"][0]["message"];
        assert!(msg["content"].is_null());
        assert_eq!(msg["tool_calls"][0]["id"], "call_1");
        assert_eq!(msg["tool_calls"][0]["function"]["name"], "terminal");
        assert_eq!(
            msg["tool_calls"][0]["function"]["arguments"],
            r#"{"cmd":"pwd"}"#
        );
    }
}
