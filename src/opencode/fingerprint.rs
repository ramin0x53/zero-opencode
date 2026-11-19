use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value, json};

/// Fingerprint tool names required by OpenCode's free-tier body gate.
pub const OPENCODE_FINGERPRINT_TOOLS: [&str; 2] = ["read", "shell"];

const GATED_TOOL_ALIAS: &str = "run_command";
const ENV_TAG: &str = "<env>";
const ENV_HEADING: &str =
    "Here is some useful information about the environment you are running in:";

static GATED_TOOL_NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("(?i)terminal").expect("valid regex"));

/// Map of `sent_tool_name -> original_caller_tool_name`.
pub type RenameMap = HashMap<String, String>;

fn read_tool_description() -> &'static str {
    "Read the contents of a file or directory. Supports text files, images, and PDFs. Images and PDFs are presented directly to the model. Each text line is prefixed by its 1-based line number as <line>: <content>. The prefix is for reference and is not part of the file content. Directory entries are returned one per line. Use offset and limit to read large files or directories in sections. Prefer one larger read over many small slices, and use grep to find specific content in large files."
}

fn read_tool_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File or directory to read"
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "The line or directory entry to start reading from (1-based)"
            },
            "limit": {
                "type": "integer",
                "minimum": 0,
                "description": "The maximum number of lines or directory entries to read (defaults to and capped at 2000)"
            }
        },
        "required": ["path"],
        "additionalProperties": false
    })
}

fn shell_tool_description() -> &'static str {
    "Execute a shell command and return its output. Commands run on Linux using bash. Quote file paths containing spaces or special characters. Prefer dedicated tools over shell commands when possible. When output is large, the full result is saved to a file and a truncated preview is returned. Rely on automatic truncation unless filtering the output is more useful. Commands accept an optional timeout, background commands have no timeout by default. Background commands return immediately, and you will be notified when they complete."
}

fn shell_tool_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command string to execute"
            },
            "workdir": {
                "type": "string",
                "description": "Working directory to execute the command in. Defaults to the current working directory. When possible, avoid changing directories in the command and set the working directory here instead."
            },
            "timeout": {
                "type": "integer",
                "minimum": 0,
                "description": "Timeout in milliseconds. Set to 0 to disable the timeout. Defaults to 120000 for foreground commands. Background commands have no timeout by default."
            },
            "background": {
                "type": "boolean",
                "description": "Run the command in the background and return immediately (useful for dev servers and long-running builds). You do not need to use '&' at the end of the command when using this parameter. You will be notified when it completes. DO NOT poll for completion."
            }
        },
        "required": ["command"],
        "additionalProperties": false
    })
}

fn fingerprint_definition(key: &str) -> Option<(&'static str, Value)> {
    match key {
        "read" => Some((read_tool_description(), read_tool_parameters())),
        "shell" => Some((shell_tool_description(), shell_tool_parameters())),
        _ => None,
    }
}

/// Return `"read"` or `"shell"` when `name` (trimmed) matches an exact fingerprint name.
pub fn fingerprint_tool_key(name: &str) -> Option<&'static str> {
    match name.trim() {
        "read" => Some("read"),
        "shell" => Some("shell"),
        _ => None,
    }
}

fn tool_name_of(tool: &Value) -> String {
    let Some(obj) = tool.as_object() else {
        return String::new();
    };
    if let Some(name) = obj.get("name").and_then(Value::as_str) {
        let trimmed = name.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Some(name) = obj
        .get("function")
        .and_then(Value::as_object)
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
    {
        return name.trim().to_string();
    }
    String::new()
}

fn tool_def_obj(tool: &Value) -> Option<&Map<String, Value>> {
    let obj = tool.as_object()?;
    if let Some(fn_obj) = obj.get("function").and_then(Value::as_object) {
        Some(fn_obj)
    } else {
        Some(obj)
    }
}

fn is_fingerprint_definition(tool: &Value, key: &str) -> bool {
    let Some((want_desc, want_params)) = fingerprint_definition(key) else {
        return false;
    };
    let Some(def) = tool_def_obj(tool) else {
        return false;
    };
    def.get("description").and_then(Value::as_str) == Some(want_desc)
        && def.get("parameters") == Some(&want_params)
}

fn can_normalize_in_place(tool: &Value, key: &str) -> bool {
    let Some((_, want_params)) = fingerprint_definition(key) else {
        return false;
    };
    let Some(def) = tool_def_obj(tool) else {
        return false;
    };
    let Some(declared_props) = def
        .get("parameters")
        .and_then(|p| p.get("properties"))
        .and_then(Value::as_object)
    else {
        return false;
    };
    let Some(wanted_props) = want_params.get("properties").and_then(Value::as_object) else {
        return false;
    };
    let Some(wanted_req) = want_params.get("required").and_then(Value::as_array) else {
        return false;
    };

    if declared_props.keys().any(|k| !wanted_props.contains_key(k)) {
        return false;
    }
    wanted_req.iter().all(|req| {
        req.as_str()
            .is_some_and(|name| declared_props.contains_key(name))
    })
}

fn with_tool_name(tool: &Value, name: &str) -> Value {
    let mut cloned = tool.clone();
    if let Some(obj) = cloned.as_object_mut() {
        if let Some(fn_obj) = obj.get_mut("function").and_then(Value::as_object_mut) {
            fn_obj.insert("name".to_string(), Value::String(name.to_string()));
        } else {
            obj.insert("name".to_string(), Value::String(name.to_string()));
        }
    }
    cloned
}

fn with_fingerprint_definition(tool: &Value, key: &str) -> Value {
    let Some((desc, params)) = fingerprint_definition(key) else {
        return tool.clone();
    };
    let mut cloned = tool.clone();
    if let Some(obj) = cloned.as_object_mut() {
        if let Some(fn_obj) = obj.get_mut("function").and_then(Value::as_object_mut) {
            fn_obj.insert("description".to_string(), Value::String(desc.to_string()));
            fn_obj.insert("parameters".to_string(), params);
        } else {
            obj.insert("description".to_string(), Value::String(desc.to_string()));
            obj.insert("parameters".to_string(), params);
        }
    }
    cloned
}

fn next_available_alias(base: &str, taken: &mut HashSet<String>) -> String {
    let mut alias = base.to_string();
    let mut i = 2u32;
    while taken.contains(&alias) {
        alias = format!("{base}{i}");
        i += 1;
    }
    taken.insert(alias.clone());
    alias
}

/// Normalize compatible caller `read`/`shell` definitions in place and alias
/// colliding/gated tool names (`terminal` -> `run_command`, incompatible `read` -> `read_client`).
pub fn conceal_fingerprint_tool_names(tools: &[Value]) -> (Vec<Value>, RenameMap) {
    let mut map = RenameMap::new();
    if tools.is_empty() {
        return (Vec::new(), map);
    }

    let mut taken: HashSet<String> = tools.iter().map(tool_name_of).collect();
    let mut seen_fp: HashSet<&'static str> = HashSet::new();
    let mut out = Vec::with_capacity(tools.len());

    for tool in tools {
        if !tool.is_object() {
            out.push(tool.clone());
            continue;
        }

        let name = tool_name_of(tool);
        if !name.is_empty() && GATED_TOOL_NAME_RE.is_match(&name) {
            let alias = next_available_alias(GATED_TOOL_ALIAS, &mut taken);
            map.insert(alias.clone(), name);
            out.push(with_tool_name(tool, &alias));
            continue;
        }

        let Some(key) = fingerprint_tool_key(&name) else {
            out.push(tool.clone());
            continue;
        };

        if !seen_fp.insert(key) {
            continue;
        }

        if is_fingerprint_definition(tool, key) {
            out.push(tool.clone());
            continue;
        }

        if can_normalize_in_place(tool, key) {
            out.push(with_fingerprint_definition(tool, key));
            continue;
        }

        let base_alias = format!("{key}_client");
        let alias = next_available_alias(&base_alias, &mut taken);
        map.insert(alias.clone(), key.to_string());
        out.push(with_tool_name(tool, &alias));
    }

    (out, map)
}

/// Append OpenCode's exact `read` and `shell` tool definitions when missing.
/// Returns the names of tools appended.
pub fn append_missing_fingerprint_tools(tools: &mut Vec<Value>, flat: bool) -> Vec<&'static str> {
    let mut appended = Vec::new();
    for name in OPENCODE_FINGERPRINT_TOOLS {
        let already_present = tools
            .iter()
            .any(|t| fingerprint_tool_key(&tool_name_of(t)) == Some(name));
        if already_present {
            continue;
        }
        let Some((description, parameters)) = fingerprint_definition(name) else {
            continue;
        };
        let entry = if flat {
            json!({
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters,
                "strict": false
            })
        } else {
            json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            })
        };
        tools.push(entry);
        appended.push(name);
    }
    appended
}

fn aliases_by_original(map: &RenameMap) -> HashMap<&str, &str> {
    map.iter()
        .map(|(alias, original)| (original.as_str(), alias.as_str()))
        .collect()
}

/// Point an explicit `tool_choice` at the alias of a renamed caller tool.
pub fn retarget_tool_choice(body: &mut Value, map: &RenameMap) {
    if map.is_empty() {
        return;
    }
    let aliases = aliases_by_original(map);
    let Some(choice) = body.get_mut("tool_choice").and_then(Value::as_object_mut) else {
        return;
    };

    if let Some(name) = choice.get("name").and_then(Value::as_str).map(str::trim)
        && let Some(alias) = aliases.get(name)
    {
        choice.insert("name".to_string(), Value::String((*alias).to_string()));
        return;
    }

    if let Some(fn_obj) = choice.get_mut("function").and_then(Value::as_object_mut)
        && let Some(name) = fn_obj.get("name").and_then(Value::as_str).map(str::trim)
        && let Some(alias) = aliases.get(name)
    {
        fn_obj.insert("name".to_string(), Value::String((*alias).to_string()));
    }
}

/// Rewrite prior-turn assistant tool call names to match aliased declarations.
pub fn alias_history_tool_calls(body: &mut Value, map: &RenameMap) {
    if map.is_empty() {
        return;
    }
    let aliases = aliases_by_original(map);

    if let Some(input) = body.get_mut("input").and_then(Value::as_array_mut) {
        for item in input {
            let Some(obj) = item.as_object_mut() else {
                continue;
            };
            if obj.get("type").and_then(Value::as_str) == Some("function_call")
                && let Some(name) = obj.get("name").and_then(Value::as_str)
                && let Some(alias) = aliases.get(name)
            {
                obj.insert("name".to_string(), Value::String((*alias).to_string()));
            }
        }
    }

    if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
        for msg in messages {
            let Some(tool_calls) = msg.get_mut("tool_calls").and_then(Value::as_array_mut) else {
                continue;
            };
            for call in tool_calls {
                if let Some(fn_obj) = call.get_mut("function").and_then(Value::as_object_mut)
                    && let Some(name) = fn_obj.get("name").and_then(Value::as_str)
                    && let Some(alias) = aliases.get(name)
                {
                    fn_obj.insert("name".to_string(), Value::String((*alias).to_string()));
                }
            }
        }
    }
}

fn system_carries_env(val: &Value) -> bool {
    match val {
        Value::String(s) => s.contains(ENV_TAG),
        Value::Array(blocks) => blocks.iter().any(|b| {
            b.get("text")
                .and_then(Value::as_str)
                .is_some_and(|t| t.contains(ENV_TAG))
        }),
        _ => false,
    }
}

fn append_system_text(val: Option<&Value>, text: &str) -> Value {
    match val {
        Some(Value::String(existing)) => {
            if existing.trim().is_empty() {
                Value::String(text.to_string())
            } else {
                Value::String(format!("{existing}\n\n{text}"))
            }
        }
        Some(Value::Array(blocks)) => {
            let mut next = blocks.clone();
            for block in next.iter_mut().rev() {
                if let Some(obj) = block.as_object_mut()
                    && let Some(existing) = obj.get("text").and_then(Value::as_str)
                {
                    let updated = if existing.trim().is_empty() {
                        text.to_string()
                    } else {
                        format!("{existing}\n\n{text}")
                    };
                    obj.insert("text".to_string(), Value::String(updated));
                    return Value::Array(next);
                }
            }
            next.push(json!({ "type": "text", "text": text }));
            Value::Array(next)
        }
        _ => Value::Array(vec![json!({ "type": "text", "text": text })]),
    }
}

fn note_unavailable_tools(body: &mut Value, flat: bool, names: &[&str]) {
    if names.is_empty() {
        return;
    }
    let list = names
        .iter()
        .map(|n| format!("`{n}`"))
        .collect::<Vec<_>>()
        .join(" or ");
    let note = if names.len() > 1 {
        format!(
            "Do not call the {list} tools; they are placeholders that do nothing. Use the other tools you have been given instead."
        )
    } else {
        format!(
            "Do not call the {list} tool; it is a placeholder that does nothing. Use the other tools you have been given instead."
        )
    };

    if flat {
        let updated = match body.get("instructions").and_then(Value::as_str) {
            Some(existing) if !existing.trim().is_empty() => format!("{existing}\n\n{note}"),
            _ => note,
        };
        if let Some(obj) = body.as_object_mut() {
            obj.insert("instructions".to_string(), Value::String(updated));
        }
        return;
    }

    if body.get("system").is_some() {
        let updated = append_system_text(body.get("system"), &note);
        if let Some(obj) = body.as_object_mut() {
            obj.insert("system".to_string(), updated);
        }
        return;
    }

    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    if let Some(first) = messages.first_mut()
        && first.get("role").and_then(Value::as_str) == Some("system")
    {
        let updated = match first.get("content") {
            Some(Value::String(s)) => {
                if s.trim().is_empty() {
                    Value::String(note)
                } else {
                    Value::String(format!("{s}\n\n{note}"))
                }
            }
            other => append_system_text(other, &note),
        };
        if let Some(obj) = first.as_object_mut() {
            obj.insert("content".to_string(), updated);
        }
    } else {
        messages.insert(0, json!({ "role": "system", "content": note }));
    }
}

fn node_platform_name() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        other => other,
    }
}

/// Build the CLI `<env>` block matching OpenCode CLI's runtime format.
pub fn build_env_fingerprint_prompt() -> String {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let tmp_dir = std::env::temp_dir().join("opencode");
    let platform = node_platform_name();
    [
        ENV_HEADING.to_string(),
        ENV_TAG.to_string(),
        format!("  Working directory: {cwd}"),
        format!("  Workspace root folder: {cwd}"),
        "  Is directory a git repo: no".to_string(),
        format!("  Platform: {platform}"),
        format!(
            "  Prefer {} over generic system temporary directories such as /tmp; it is pre-created and approved for external access.",
            tmp_dir.display()
        ),
        "</env>".to_string(),
    ]
    .join("\n")
}

/// Append the `<env>` system prompt marker if not already present.
pub fn apply_env_fingerprint(body: &mut Value, flat: bool) -> bool {
    let Some(obj) = body.as_object_mut() else {
        return false;
    };
    let env_prompt = build_env_fingerprint_prompt();

    if flat {
        if let Some(instr) = obj.get("instructions") {
            let Some(s) = instr.as_str() else {
                return false;
            };
            if s.contains(ENV_TAG) {
                return false;
            }
            let updated = if s.trim().is_empty() {
                env_prompt
            } else {
                format!("{s}\n\n{env_prompt}")
            };
            obj.insert("instructions".to_string(), Value::String(updated));
        } else {
            obj.insert("instructions".to_string(), Value::String(env_prompt));
        }
        return true;
    }

    if let Some(sys) = obj.get("system") {
        if system_carries_env(sys) {
            return false;
        }
        let updated = match sys {
            Value::String(s) => {
                if s.trim().is_empty() {
                    Value::String(env_prompt)
                } else {
                    Value::String(format!("{s}\n\n{env_prompt}"))
                }
            }
            _ => append_system_text(Some(sys), &env_prompt),
        };
        obj.insert("system".to_string(), updated);
        return true;
    }

    let Some(messages) = obj.get_mut("messages").and_then(Value::as_array_mut) else {
        return false;
    };
    if let Some(idx) = messages
        .iter()
        .position(|m| m.get("role").and_then(Value::as_str) == Some("system"))
    {
        if let Some(content) = messages[idx].get("content")
            && system_carries_env(content)
        {
            return false;
        }
        let updated = match messages[idx].get("content") {
            Some(Value::String(s)) => {
                if s.trim().is_empty() {
                    Value::String(env_prompt)
                } else {
                    Value::String(format!("{s}\n\n{env_prompt}"))
                }
            }
            other => append_system_text(other, &env_prompt),
        };
        if let Some(msg_obj) = messages[idx].as_object_mut() {
            msg_obj.insert("content".to_string(), updated);
        }
        return true;
    }

    messages.insert(0, json!({ "role": "system", "content": env_prompt }));
    true
}

/// Apply the full request-side OpenCode free-tier fingerprint:
/// - conceal/normalize colliding or `terminal`-gated caller tools
/// - append exact `read`/`shell` definitions if absent
/// - retarget `tool_choice` and history tool calls
/// - inject placeholder note (when tools were appended) and `<env>` marker
pub fn apply_fingerprint_tools(body: &mut Value, flat: bool) -> RenameMap {
    let Some(obj) = body.as_object_mut() else {
        return RenameMap::new();
    };

    let incoming_tools = obj
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let (mut next_tools, map) = conceal_fingerprint_tool_names(&incoming_tools);
    let appended = append_missing_fingerprint_tools(&mut next_tools, flat);
    obj.insert("tools".to_string(), Value::Array(next_tools));

    retarget_tool_choice(body, &map);
    alias_history_tool_calls(body, &map);

    if !appended.is_empty() {
        note_unavailable_tools(body, flat, &appended);
    }

    apply_env_fingerprint(body, flat);

    if flat && !body.get("tool_choice").is_some_and(|v| !v.is_null()) {
        body.as_object_mut()
            .expect("object")
            .insert("tool_choice".to_string(), Value::String("auto".to_string()));
    }

    map
}

/// Restore aliased caller tool names in supported Chat/Responses/Claude payloads.
/// Returns `true` if any tool name was modified.
pub fn restore_tool_names(payload: &mut Value, map: &RenameMap) -> bool {
    if map.is_empty() {
        return false;
    }

    match payload {
        Value::Array(items) => {
            let mut changed = false;
            for item in items {
                changed |= restore_tool_names(item, map);
            }
            changed
        }
        Value::Object(obj) => {
            let mut changed = false;

            // Claude streaming content_block_start event
            if obj.get("type").and_then(Value::as_str) == Some("content_block_start")
                && let Some(block) = obj.get_mut("content_block").and_then(Value::as_object_mut)
                && block.get("type").and_then(Value::as_str) == Some("tool_use")
                && let Some(name) = block.get("name").and_then(Value::as_str)
                && let Some(orig) = map.get(name)
            {
                block.insert("name".to_string(), Value::String(orig.clone()));
                changed = true;
            }

            // Claude non-streaming message content
            if let Some(content) = obj.get_mut("content").and_then(Value::as_array_mut) {
                for block in content {
                    if let Some(bobj) = block.as_object_mut()
                        && bobj.get("type").and_then(Value::as_str) == Some("tool_use")
                        && let Some(name) = bobj.get("name").and_then(Value::as_str)
                        && let Some(orig) = map.get(name)
                    {
                        bobj.insert("name".to_string(), Value::String(orig.clone()));
                        changed = true;
                    }
                }
            }

            // OpenAI Chat Completions (both streaming delta and JSON message)
            if let Some(choices) = obj.get_mut("choices").and_then(Value::as_array_mut) {
                for choice in choices {
                    let Some(cobj) = choice.as_object_mut() else {
                        continue;
                    };
                    for holder in ["delta", "message"] {
                        let Some(calls) = cobj
                            .get_mut(holder)
                            .and_then(|v| v.get_mut("tool_calls"))
                            .and_then(Value::as_array_mut)
                        else {
                            continue;
                        };
                        for call in calls {
                            if let Some(fn_obj) =
                                call.get_mut("function").and_then(Value::as_object_mut)
                                && let Some(name) = fn_obj.get("name").and_then(Value::as_str)
                                && let Some(orig) = map.get(name)
                            {
                                fn_obj.insert("name".to_string(), Value::String(orig.clone()));
                                changed = true;
                            }
                        }
                    }
                }
            }

            // OpenAI Responses final JSON body
            if let Some(output) = obj.get_mut("output").and_then(Value::as_array_mut) {
                for item in output {
                    if let Some(iobj) = item.as_object_mut()
                        && iobj.get("type").and_then(Value::as_str) == Some("function_call")
                        && let Some(name) = iobj.get("name").and_then(Value::as_str)
                        && let Some(orig) = map.get(name)
                    {
                        iobj.insert("name".to_string(), Value::String(orig.clone()));
                        changed = true;
                    }
                }
            }

            // OpenAI Responses SSE item
            if let Some(item) = obj.get_mut("item").and_then(Value::as_object_mut)
                && item.get("type").and_then(Value::as_str) == Some("function_call")
                && let Some(name) = item.get("name").and_then(Value::as_str)
                && let Some(orig) = map.get(name)
            {
                item.insert("name".to_string(), Value::String(orig.clone()));
                changed = true;
            }

            changed
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_names(body: &Value) -> Vec<String> {
        body.get("tools")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(tool_name_of)
            .collect()
    }

    fn chat_tools(names: &[&str]) -> Vec<Value> {
        names
            .iter()
            .map(|n| json!({ "type": "function", "function": { "name": n } }))
            .collect()
    }

    fn flat_tools(names: &[&str]) -> Vec<Value> {
        names
            .iter()
            .map(|n| json!({ "type": "function", "name": n }))
            .collect()
    }

    #[test]
    fn appends_read_and_shell_preserving_case_variants() {
        let mut body = json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": chat_tools(&["Bash", "Read", "Edit"])
        });
        let map = apply_fingerprint_tools(&mut body, false);
        assert!(map.is_empty());
        assert_eq!(tool_names(&body), vec!["Bash", "Read", "Edit", "read", "shell"]);
        let sys = body["messages"][0]["content"].as_str().unwrap();
        assert!(sys.contains("Do not call the `read` or `shell` tools;"));
        assert!(sys.contains("<env>"));
    }

    #[test]
    fn normalizes_compatible_windows_shell_in_place_without_aliasing() {
        let windows_shell = json!({
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Execute a shell command and return its output. Commands run on Windows using pwsh.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" },
                        "workdir": { "type": "string" },
                        "timeout": { "type": "integer" },
                        "background": { "type": "boolean" }
                    },
                    "required": ["command"]
                }
            }
        });
        let mut body = json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [windows_shell]
        });
        let map = apply_fingerprint_tools(&mut body, false);
        assert!(map.is_empty());
        assert_eq!(tool_names(&body), vec!["shell", "read"]);
        assert!(
            body["tools"][0]["function"]["description"]
                .as_str()
                .unwrap()
                .contains("Commands run on Linux using bash.")
        );
        let sys = body["messages"][0]["content"].as_str().unwrap();
        assert!(sys.contains("Do not call the `read` tool; it is a placeholder"));
    }

    #[test]
    fn aliases_incompatible_fingerprint_name_avoiding_collisions() {
        let mut body = json!({
            "tools": flat_tools(&["read", "read_client", "Edit"])
        });
        let map = apply_fingerprint_tools(&mut body, true);
        assert_eq!(
            tool_names(&body),
            vec!["read_client2", "read_client", "Edit", "read", "shell"]
        );
        assert_eq!(map.get("read_client2").map(String::as_str), Some("read"));
    }

    #[test]
    fn drops_duplicate_fingerprint_declarations() {
        let mut body = json!({
            "tools": flat_tools(&["shell", "shell"])
        });
        apply_fingerprint_tools(&mut body, true);
        assert_eq!(tool_names(&body), vec!["shell_client", "read", "shell"]);
    }

    #[test]
    fn aliases_terminal_substring_retargets_choice_and_history_and_restores_response() {
        let mut body = json!({
            "tools": chat_tools(&["read_file", "my_terminal_thing", "Terminal"]),
            "tool_choice": { "type": "function", "function": { "name": "my_terminal_thing" } },
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        { "id": "c1", "type": "function", "function": { "name": "my_terminal_thing", "arguments": "{}" } }
                    ]
                },
                { "role": "tool", "tool_call_id": "c1", "content": "ok" }
            ]
        });
        let map = apply_fingerprint_tools(&mut body, false);
        let sent = tool_names(&body);
        assert!(!sent.iter().any(|n| n.to_lowercase().contains("terminal")));
        assert_eq!(
            body["tool_choice"]["function"]["name"].as_str(),
            Some("run_command")
        );
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["name"].as_str(),
            Some("run_command")
        );

        let mut resp = json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": { "name": "run_command", "arguments": "{}" }
                    }]
                }
            }]
        });
        assert!(restore_tool_names(&mut resp, &map));
        assert_eq!(
            resp["choices"][0]["delta"]["tool_calls"][0]["function"]["name"].as_str(),
            Some("my_terminal_thing")
        );
    }

    #[test]
    fn env_fingerprint_is_idempotent_and_preserves_existing_system_prompt() {
        let mut body = json!({
            "messages": [
                { "role": "system", "content": "Be concise." },
                { "role": "user", "content": "hi" }
            ]
        });
        assert!(apply_env_fingerprint(&mut body, false));
        let first = body["messages"][0]["content"].as_str().unwrap().to_string();
        assert!(first.starts_with("Be concise.\n\n"));
        assert!(first.contains("<env>"));
        assert!(!apply_env_fingerprint(&mut body, false));
    }

    #[test]
    fn never_panics_on_malformed_tools() {
        for bad in [
            vec![Value::Null, json!(42), json!([])],
            vec![json!({}), json!({ "name": "" })],
        ] {
            let (out, _map) = conceal_fingerprint_tool_names(&bad);
            assert_eq!(out.len(), bad.len());
        }
    }
}
