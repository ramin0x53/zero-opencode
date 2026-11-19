use std::collections::HashMap;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::{Value, json};

use crate::opencode::executor::UpstreamBody;
use crate::opencode::fingerprint::{RenameMap, restore_tool_names};
use crate::stream::sse_parser::{SseLineParser, extract_sse_data};

/// Extract reasoning text across OpenAI-compatible vendor fields:
/// - `reasoning_content` (standard GLM/Qwen/DeepSeek/Kimi)
/// - `reasoning` (string alias)
/// - `reasoning_details[]` (OpenRouter / MiMo alias)
pub fn extract_reasoning_text(delta: &Value) -> String {
    let Some(obj) = delta.as_object() else {
        return String::new();
    };
    if let Some(rc) = obj.get("reasoning_content").and_then(Value::as_str)
        && !rc.is_empty()
    {
        return rc.to_string();
    }
    if let Some(r) = obj.get("reasoning").and_then(Value::as_str)
        && !r.is_empty()
    {
        return r.to_string();
    }
    if let Some(details) = obj.get("reasoning_details").and_then(Value::as_array) {
        let mut out = String::new();
        for item in details {
            if let Some(s) = item.as_str() {
                out.push_str(s);
            } else if let Some(s) = item
                .get("text")
                .or_else(|| item.get("content"))
                .and_then(Value::as_str)
            {
                out.push_str(s);
            }
        }
        return out;
    }
    String::new()
}

/// Add canonical `delta.reasoning_content` when only vendor aliases are present,
/// leaving the original alias fields intact.
pub fn lift_reasoning_aliases(delta: &mut Value) -> bool {
    let Some(obj) = delta.as_object_mut() else {
        return false;
    };
    if obj
        .get("reasoning_content")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
    {
        return false;
    }
    let text = extract_reasoning_text(&Value::Object(obj.clone()));
    if text.is_empty() {
        return false;
    }
    obj.insert("reasoning_content".to_string(), Value::String(text));
    true
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn fix_invalid_id(parsed: &mut Value) {
    let Some(obj) = parsed.as_object_mut() else {
        return;
    };
    let should_fix = obj
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|id| id == "chat" || id == "completion" || id.len() < 8);
    if should_fix {
        let fallback = obj
            .get("extend_fields")
            .and_then(|e| e.get("requestId").or_else(|| e.get("traceId")))
            .and_then(Value::as_str)
            .map(ToString::to_string)
            .unwrap_or_else(|| now_ms().to_string());
        obj.insert("id".to_string(), Value::String(format!("chatcmpl-{fallback}")));
    }
}

fn has_valuable_content(chunk: &Value) -> bool {
    if chunk.get("usage").is_some_and(|u| u.is_object()) {
        return true;
    }
    let Some(choice) = chunk
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
    else {
        return true;
    };
    if choice
        .get("finish_reason")
        .is_some_and(|f| !f.is_null() && f.as_str() != Some(""))
    {
        return true;
    }
    let Some(delta) = choice.get("delta") else {
        return true;
    };
    let has_content = delta
        .get("content")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty());
    let has_reasoning = delta
        .get("reasoning_content")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty());
    let has_tool_calls = delta
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty());
    let has_role = delta
        .get("role")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty());

    has_content || has_reasoning || has_tool_calls || has_role
}

/// Stateful per-stream SSE chunk normalizer matching 9router's OpenAI passthrough.
#[derive(Debug)]
pub struct ChunkNormalizer {
    seen_tool_call_ids: HashMap<u64, String>,
    rename_map: RenameMap,
    done_sent: bool,
    fallback_model: String,
    seq: u64,
}

impl ChunkNormalizer {
    pub fn new(rename_map: RenameMap, fallback_model: impl Into<String>) -> Self {
        Self {
            seen_tool_call_ids: HashMap::new(),
            rename_map,
            done_sent: false,
            fallback_model: fallback_model.into(),
            seq: 0,
        }
    }

    /// Normalize a single SSE line and return formatted `data: ...\n\n` output if emit-worthy.
    pub fn normalize_line(&mut self, line: &str) -> Option<String> {
        let payload = extract_sse_data(line)?;
        if payload.is_empty() {
            return None;
        }
        if payload == "[DONE]" {
            if self.done_sent {
                return None;
            }
            self.done_sent = true;
            return Some("data: [DONE]\n\n".to_string());
        }

        let mut parsed: Value = serde_json::from_str(payload).ok()?;

        // Pass upstream SSE error payloads directly to the client.
        if parsed.get("error").is_some() {
            return serde_json::to_string(&parsed)
                .ok()
                .map(|s| format!("data: {s}\n\n"));
        }

        fix_invalid_id(&mut parsed);

        if let Some(obj) = parsed.as_object_mut() {
            obj.remove("prompt_filter_results");
            if obj.get("choices").is_some() {
                if !obj.get("object").is_some_and(|v| v.as_str().is_some_and(|s| !s.is_empty())) {
                    obj.insert(
                        "object".to_string(),
                        Value::String("chat.completion.chunk".to_string()),
                    );
                }
                if !obj.get("created").is_some_and(|v| v.is_number()) {
                    obj.insert("created".to_string(), json!(now_secs()));
                }
                if !obj.get("model").is_some_and(|v| v.as_str().is_some_and(|s| !s.is_empty())) {
                    obj.insert("model".to_string(), Value::String(self.fallback_model.clone()));
                }
            }
        }

        if let Some(choices) = parsed.get_mut("choices").and_then(Value::as_array_mut) {
            for choice in choices {
                let Some(cobj) = choice.as_object_mut() else {
                    continue;
                };
                cobj.remove("content_filter_results");

                if let Some(delta) = cobj.get_mut("delta") {
                    lift_reasoning_aliases(delta);

                    if let Some(dobj) = delta.as_object_mut() {
                        if dobj
                            .get("tool_calls")
                            .and_then(Value::as_array)
                            .is_some_and(Vec::is_empty)
                        {
                            dobj.remove("tool_calls");
                        } else if let Some(calls) =
                            dobj.get_mut("tool_calls").and_then(Value::as_array_mut)
                        {
                            for call in calls {
                                let Some(call_obj) = call.as_object_mut() else {
                                    continue;
                                };
                                let idx = call_obj.get("index").and_then(Value::as_u64).unwrap_or(0);
                                let existing_id = call_obj
                                    .get("id")
                                    .and_then(Value::as_str)
                                    .filter(|s| !s.is_empty())
                                    .map(ToString::to_string);
                                if let Some(id) = existing_id {
                                    self.seen_tool_call_ids.insert(idx, id);
                                } else {
                                    let id = self
                                        .seen_tool_call_ids
                                        .entry(idx)
                                        .or_insert_with(|| {
                                            self.seq += 1;
                                            format!("call_{}_{}_{}", now_ms(), idx, self.seq)
                                        })
                                        .clone();
                                    call_obj.insert("id".to_string(), Value::String(id));
                                }
                            }
                        }
                    }
                }
            }
        }

        restore_tool_names(&mut parsed, &self.rename_map);

        if !has_valuable_content(&parsed) {
            return None;
        }

        serde_json::to_string(&parsed)
            .ok()
            .map(|s| format!("data: {s}\n\n"))
    }

    /// Emit a single terminal `[DONE]` frame if upstream ended without one.
    pub fn finish_sse(&mut self) -> Option<String> {
        if self.done_sent {
            None
        } else {
            self.done_sent = true;
            Some("data: [DONE]\n\n".to_string())
        }
    }
}

/// Convert an upstream byte stream into a normalized SSE `Bytes` stream.
pub fn into_normalized_sse_stream(
    upstream: UpstreamBody,
    rename_map: RenameMap,
    fallback_model: String,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    let raw_stream: Pin<Box<dyn Stream<Item = anyhow::Result<Bytes>> + Send>> = match upstream {
        UpstreamBody::Stream(s) => s,
        UpstreamBody::Buffered(b) => Box::pin(futures_util::stream::once(async move { Ok(b) })),
    };

    let stream = futures_util::stream::unfold(
        (
            raw_stream,
            SseLineParser::new(),
            ChunkNormalizer::new(rename_map, fallback_model),
            false,
        ),
        |(mut rx, mut parser, mut normalizer, done)| async move {
            if done {
                return None;
            }

            loop {
                match rx.next().await {
                    Some(Ok(chunk)) => {
                        let mut out = String::new();
                        for line in parser.push_chunk(&chunk) {
                            if let Some(formatted) = normalizer.normalize_line(&line) {
                                out.push_str(&formatted);
                            }
                        }
                        if !out.is_empty() {
                            return Some((
                                Ok(Bytes::from(out)),
                                (rx, parser, normalizer, false),
                            ));
                        }
                    }
                    Some(Err(err)) => {
                        return Some((
                            Err(std::io::Error::other(err.to_string())),
                            (rx, parser, normalizer, true),
                        ));
                    }
                    None => {
                        let mut out = String::new();
                        if let Some(last) = parser.finish()
                            && let Some(formatted) = normalizer.normalize_line(&last)
                        {
                            out.push_str(&formatted);
                        }
                        if let Some(done_frame) = normalizer.finish_sse() {
                            out.push_str(&done_frame);
                        }
                        if out.is_empty() {
                            return None;
                        }
                        return Some((
                            Ok(Bytes::from(out)),
                            (rx, parser, normalizer, true),
                        ));
                    }
                }
            }
        },
    );

    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifts_reasoning_and_reasoning_details_keeping_originals() {
        let mut d1 = json!({ "content": "", "reasoning": "step 1" });
        assert!(lift_reasoning_aliases(&mut d1));
        assert_eq!(d1["reasoning_content"], "step 1");
        assert_eq!(d1["reasoning"], "step 1");

        let mut d2 = json!({
            "content": "",
            "reasoning_details": [{"text": "ab"}, {"content": "c"}]
        });
        assert!(lift_reasoning_aliases(&mut d2));
        assert_eq!(d2["reasoning_content"], "abc");

        let mut d3 = json!({ "reasoning_content": "kept", "reasoning": "ignored" });
        assert!(!lift_reasoning_aliases(&mut d3));
        assert_eq!(d3["reasoning_content"], "kept");
    }

    #[test]
    fn ensures_tool_call_ids_strips_empty_tool_calls_and_restores_names() {
        let mut rename = RenameMap::new();
        rename.insert("run_command".into(), "terminal".into());
        let mut norm = ChunkNormalizer::new(rename, "big-pickle");

        // Chunk with empty tool_calls array + reasoning alias
        let c1 = r#"data: {"id":"chat","choices":[{"index":0,"delta":{"role":"assistant","reasoning":"think","tool_calls":[]}}]}"#;
        let out1 = norm.normalize_line(c1).unwrap();
        let v1: Value = serde_json::from_str(out1.trim_start_matches("data: ").trim()).unwrap();
        assert_eq!(v1["object"], "chat.completion.chunk");
        assert!(v1["id"].as_str().unwrap().starts_with("chatcmpl-"));
        assert!(v1["choices"][0]["delta"].get("tool_calls").is_none());
        assert_eq!(v1["choices"][0]["delta"]["reasoning_content"], "think");

        // Tool call fragment missing id on index 0, plus another fragment reusing index 0
        let c2 = r#"data: {"id":"chatcmpl-123456","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"run_command","arguments":"{\"cmd\":"}}]}}]}"#;
        let c3 = r#"data: {"id":"chatcmpl-123456","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls\"}"}}]}}]}"#;

        let out2 = norm.normalize_line(c2).unwrap();
        let out3 = norm.normalize_line(c3).unwrap();
        let v2: Value = serde_json::from_str(out2.trim_start_matches("data: ").trim()).unwrap();
        let v3: Value = serde_json::from_str(out3.trim_start_matches("data: ").trim()).unwrap();

        let id2 = v2["choices"][0]["delta"]["tool_calls"][0]["id"]
            .as_str()
            .unwrap();
        let id3 = v3["choices"][0]["delta"]["tool_calls"][0]["id"]
            .as_str()
            .unwrap();
        assert!(!id2.is_empty());
        assert_eq!(id2, id3);
        assert_eq!(
            v2["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "terminal"
        );
    }

    #[test]
    fn deduplicates_done_sentinel() {
        let mut norm = ChunkNormalizer::new(RenameMap::new(), "big-pickle");
        assert_eq!(norm.normalize_line("data: [DONE]").as_deref(), Some("data: [DONE]\n\n"));
        assert!(norm.normalize_line("data: [DONE]").is_none());
        assert!(norm.finish_sse().is_none());
    }
}
