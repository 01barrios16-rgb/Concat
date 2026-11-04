// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah

//! The AI providers the Copilot talks to, behind one call.
//!
//! Two wire formats cover them all: Anthropic's Messages API, and the
//! chat-completions format OpenAI made and almost everyone else speaks -
//! Ollama and LM Studio on this machine, and OpenRouter, Groq, Together or
//! any other server that says it is "OpenAI compatible". A conversation is
//! kept here in a shape of its own ([`Message`]) and written out in
//! whichever format the chosen provider reads, so switching providers in
//! the middle of a chat keeps the chat.
//!
//! Blocking, on the caller's thread: the Copilot's worker runs one turn at
//! a time and has nothing else to do while it waits.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// How long a connection to a provider may take to open.
const CONNECT: Duration = Duration::from_secs(15);
/// How long a provider may think before its answer starts. A local model
/// on a modest GPU can take a while over forty tool schemas.
const READ: Duration = Duration::from_secs(300);
/// The longest answer asked for, in tokens.
const MAX_TOKENS: u32 = 4096;

/// Which provider, and so which wire format and which defaults.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    /// Claude, through Anthropic's Messages API.
    #[default]
    Anthropic,
    /// OpenAI's own API.
    Openai,
    /// Ollama on this machine, through its OpenAI-compatible endpoint.
    Ollama,
    /// Any other OpenAI-compatible server, at the address given.
    Custom,
}

impl Provider {
    /// The providers in the order the pane's picker lists them.
    pub const ALL: [Provider; 4] = [
        Provider::Anthropic,
        Provider::Openai,
        Provider::Ollama,
        Provider::Custom,
    ];

    /// The picker's index of this provider.
    pub fn index(self) -> i32 {
        Provider::ALL
            .iter()
            .position(|provider| *provider == self)
            .unwrap_or(0) as i32
    }

    /// The provider at the picker's `index`.
    pub fn at(index: i32) -> Provider {
        Provider::ALL
            .get(index.max(0) as usize)
            .copied()
            .unwrap_or_default()
    }

    /// The model asked for when none is named.
    pub fn default_model(self) -> &'static str {
        match self {
            Provider::Anthropic => "claude-sonnet-5-5",
            Provider::Openai => "gpt-5.5",
            Provider::Ollama => "qwen3:8b",
            Provider::Custom => "",
        }
    }

    /// Where the provider is when no address is given.
    pub fn default_base_url(self) -> &'static str {
        match self {
            Provider::Anthropic => "https://api.anthropic.com",
            Provider::Openai => "https://api.openai.com/v1",
            Provider::Ollama => "http://localhost:11434/v1",
            Provider::Custom => "",
        }
    }

    /// Whether a call without a key is refused before it is sent.
    pub fn needs_key(self) -> bool {
        matches!(self, Provider::Anthropic | Provider::Openai)
    }
}

/// What the Copilot is set to talk to. Saved as `copilot.json` in the
/// config folder; see `copilot::settings_path`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    pub provider: Provider,
    /// Empty for the provider's default.
    pub model: String,
    /// Empty for a provider that takes none.
    pub api_key: String,
    /// Empty for the provider's default address.
    pub base_url: String,
}

impl Settings {
    /// The settings a call is made with: what was saved, with the common
    /// slips put right. A key says whose it is - Anthropic's start
    /// `sk-ant-`, OpenAI's `sk-` - so a key pasted under the wrong provider
    /// (Ollama, which takes none) is sent where it belongs; and a model
    /// typed the way people say it ("opus 5.5", "Claude Sonnet 5.5") is
    /// written the way the API names it (`claude-opus-5-5`). A model that
    /// belongs to another provider than the one the key opens is dropped
    /// for that provider's default.
    pub fn resolved(&self) -> Settings {
        let mut resolved = self.clone();
        resolved.model = normalize_model(&self.model);
        let key = self.api_key.trim();
        let claude = resolved.model.starts_with("claude");
        let openai = is_openai_model(&resolved.model);
        let custom_address = !self.base_url.trim().is_empty()
            && Provider::ALL
                .iter()
                .all(|provider| provider.default_base_url() != self.base_url.trim());
        let wanted = if key.starts_with("sk-ant-") {
            Some(Provider::Anthropic)
        } else if key.starts_with("sk-")
            && matches!(self.provider, Provider::Ollama | Provider::Anthropic)
            && !custom_address
        {
            Some(Provider::Openai)
        } else {
            None
        };
        if let Some(provider) = wanted.filter(|provider| *provider != self.provider) {
            log::info!(
                "copilot: the key is {provider:?}'s; using it instead of {:?}",
                self.provider
            );
            resolved.provider = provider;
            if !custom_address {
                resolved.base_url.clear();
            }
        }
        let mismatched = match resolved.provider {
            Provider::Anthropic => !resolved.model.is_empty() && !claude,
            Provider::Openai => claude,
            Provider::Ollama | Provider::Custom => false,
        };
        if mismatched || (resolved.provider == Provider::Openai && !openai && claude) {
            resolved.model.clear();
        }
        resolved
    }

    /// The model to ask for.
    pub fn model(&self) -> String {
        let model = self.model.trim();
        if model.is_empty() {
            self.provider.default_model().to_owned()
        } else {
            model.to_owned()
        }
    }

    /// The address to call, without a trailing slash.
    pub fn base_url(&self) -> String {
        let base = self.base_url.trim();
        let base = if base.is_empty() {
            self.provider.default_base_url()
        } else {
            base
        };
        base.trim_end_matches('/').to_owned()
    }

    /// Why a call cannot be made as these settings stand, if it cannot.
    pub fn problem(&self) -> Option<Problem> {
        if self.provider.needs_key() && self.api_key.trim().is_empty() {
            return Some(Problem::Key);
        }
        if self.base_url().is_empty() {
            return Some(Problem::Address);
        }
        if self.model().is_empty() {
            return Some(Problem::Model);
        }
        None
    }
}

/// What is missing from the settings for a call to be made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Problem {
    /// The provider takes a key and none is given.
    Key,
    /// No address, and the provider has no default.
    Address,
    /// No model, and the provider has no default.
    Model,
}

/// A model named the way people say it, written the way the API names it:
/// "opus 5.5", "Claude Opus 5.5" and "claude-opus-5.5" are all
/// `claude-opus-5-5`. Anything else is kept as typed, trimmed.
pub fn normalize_model(model: &str) -> String {
    let typed = model.trim();
    let lower = typed.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    let words: Vec<&str> = match words.first() {
        Some(&"claude") => words[1..].to_vec(),
        _ => words,
    };
    let family = words.first().copied().unwrap_or_default();
    let numbered = words.len() >= 2
        && words[1..]
            .iter()
            .all(|word| word.chars().all(|c| c.is_ascii_digit()));
    if matches!(family, "opus" | "sonnet" | "haiku" | "fable") && numbered {
        return format!("claude-{}", words.join("-"));
    }
    typed.to_owned()
}

/// Whether a model is one of OpenAI's own.
fn is_openai_model(model: &str) -> bool {
    let model = model.to_lowercase();
    model.starts_with("gpt")
        || model.starts_with("chatgpt")
        || ["o1", "o3", "o4"]
            .iter()
            .any(|prefix| model.starts_with(prefix))
}

/// A tool the model may call.
#[derive(Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// Its arguments' JSON Schema, an object.
    pub schema: Value,
}

/// One call the model asked for.
#[derive(Clone, Debug)]
pub struct Call {
    /// The provider's id for it, which its result must name.
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// What one call came back with.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// The call's id.
    pub id: String,
    /// The tool's name; the OpenAI format does not need it, Ollama likes it.
    pub name: String,
    pub content: String,
    pub is_error: bool,
}

/// One step of a conversation, in no provider's format.
#[derive(Clone, Debug)]
pub enum Message {
    /// What the person typed.
    User(String),
    /// What the model said, and the calls it asked for.
    Assistant { text: String, calls: Vec<Call> },
    /// The answers to the calls of the message before.
    Results(Vec<Outcome>),
}

/// One answer from the model.
#[derive(Clone, Debug, Default)]
pub struct Turn {
    pub text: String,
    pub calls: Vec<Call>,
}

/// Sends the conversation to the provider `settings` names and returns its
/// next turn.
pub fn chat(
    settings: &Settings,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
) -> Result<Turn, String> {
    let settings = &settings.resolved();
    match settings.provider {
        Provider::Anthropic => anthropic(settings, system, messages, tools),
        Provider::Openai | Provider::Ollama | Provider::Custom => {
            openai(settings, system, messages, tools)
        }
    }
}

// ── Anthropic ───────────────────────────────────────────────────────────

fn anthropic(
    settings: &Settings,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
) -> Result<Turn, String> {
    let body = json!({
        "model": settings.model(),
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": anthropic_messages(messages),
        "tools": tools
            .iter()
            .map(|tool| json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.schema,
            }))
            .collect::<Vec<_>>(),
    });
    let url = format!("{}/v1/messages", settings.base_url());
    let key = settings.api_key.trim().to_owned();
    let headers = [
        ("x-api-key", key.as_str()),
        ("anthropic-version", "2023-06-01"),
    ];
    let text = concat_host::net::post_json_text(&url, CONNECT, READ, &headers, &body.to_string())?;
    let reply: Value = serde_json::from_str(&text).map_err(|error| format!("not JSON: {error}"))?;
    anthropic_turn(&reply)
}

/// The conversation in Anthropic's format, with turns of the same role
/// run together: a request stopped after its tools ran ends on their
/// results, which are the user's turn, and the next message is the user's
/// too.
fn anthropic_messages(messages: &[Message]) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::new();
    for entry in anthropic_entries(messages) {
        match merged.last_mut() {
            Some(last) if last["role"] == entry["role"] => {
                let blocks = entry["content"].as_array().cloned().unwrap_or_default();
                if let Some(content) = last["content"].as_array_mut() {
                    content.extend(blocks);
                }
            }
            _ => merged.push(entry),
        }
    }
    merged
}

fn anthropic_entries(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .map(|message| match message {
            Message::User(text) => json!({
                "role": "user",
                "content": [{ "type": "text", "text": text }],
            }),
            Message::Assistant { text, calls } => {
                let mut content = Vec::new();
                if !text.trim().is_empty() {
                    content.push(json!({ "type": "text", "text": text }));
                }
                for call in calls {
                    content.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": object_or_empty(&call.arguments),
                    }));
                }
                if content.is_empty() {
                    content.push(json!({ "type": "text", "text": "…" }));
                }
                json!({ "role": "assistant", "content": content })
            }
            Message::Results(outcomes) => json!({
                "role": "user",
                "content": outcomes
                    .iter()
                    .map(|outcome| json!({
                        "type": "tool_result",
                        "tool_use_id": outcome.id,
                        "content": outcome.content,
                        "is_error": outcome.is_error,
                    }))
                    .collect::<Vec<_>>(),
            }),
        })
        .collect()
}

fn anthropic_turn(reply: &Value) -> Result<Turn, String> {
    if let Some(error) = reply.get("error") {
        return Err(error["message"]
            .as_str()
            .unwrap_or("the provider refused")
            .to_owned());
    }
    let mut turn = Turn::default();
    for block in reply["content"].as_array().into_iter().flatten() {
        match block["type"].as_str() {
            Some("text") => {
                if !turn.text.is_empty() {
                    turn.text.push('\n');
                }
                turn.text
                    .push_str(block["text"].as_str().unwrap_or_default());
            }
            Some("tool_use") => turn.calls.push(Call {
                id: block["id"].as_str().unwrap_or_default().to_owned(),
                name: block["name"].as_str().unwrap_or_default().to_owned(),
                arguments: block["input"].clone(),
            }),
            _ => {}
        }
    }
    Ok(turn)
}

// ── OpenAI-compatible ───────────────────────────────────────────────────

fn openai(
    settings: &Settings,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
) -> Result<Turn, String> {
    let mut wire = vec![json!({ "role": "system", "content": system })];
    wire.extend(openai_messages(messages));
    let mut body = json!({
        "model": settings.model(),
        "messages": wire,
        "tools": tools
            .iter()
            .map(|tool| json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                },
            }))
            .collect::<Vec<_>>(),
    });
    // OpenAI's own newer models refuse `max_tokens` and take
    // `max_completion_tokens`; servers that copy the older format read the
    // first.
    let limit = if settings.provider == Provider::Openai {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };
    body[limit] = json!(MAX_TOKENS * 4);
    let url = format!("{}/chat/completions", settings.base_url());
    let key = settings.api_key.trim();
    let bearer = format!("Bearer {key}");
    let headers: Vec<(&str, &str)> = if key.is_empty() {
        Vec::new()
    } else {
        vec![("authorization", bearer.as_str())]
    };
    let text = concat_host::net::post_json_text(&url, CONNECT, READ, &headers, &body.to_string())?;
    let reply: Value = serde_json::from_str(&text).map_err(|error| format!("not JSON: {error}"))?;
    openai_turn(&reply)
}

fn openai_messages(messages: &[Message]) -> Vec<Value> {
    let mut wire = Vec::new();
    for message in messages {
        match message {
            Message::User(text) => wire.push(json!({ "role": "user", "content": text })),
            Message::Assistant { text, calls } => {
                let mut entry = json!({ "role": "assistant", "content": text });
                if !calls.is_empty() {
                    entry["tool_calls"] = calls
                        .iter()
                        .map(|call| {
                            json!({
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": object_or_empty(&call.arguments).to_string(),
                                },
                            })
                        })
                        .collect();
                }
                wire.push(entry);
            }
            // One message per result: the format answers each call apart.
            Message::Results(outcomes) => {
                for outcome in outcomes {
                    wire.push(json!({
                        "role": "tool",
                        "tool_call_id": outcome.id,
                        "name": outcome.name,
                        "content": outcome.content,
                    }));
                }
            }
        }
    }
    wire
}

fn openai_turn(reply: &Value) -> Result<Turn, String> {
    if let Some(error) = reply.get("error").filter(|error| !error.is_null()) {
        return Err(error["message"]
            .as_str()
            .or_else(|| error.as_str())
            .unwrap_or("the provider refused")
            .to_owned());
    }
    let message = &reply["choices"][0]["message"];
    if message.is_null() {
        return Err("the provider sent no answer".to_owned());
    }
    let mut turn = Turn {
        text: message["content"].as_str().unwrap_or_default().to_owned(),
        calls: Vec::new(),
    };
    for (index, call) in message["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let function = &call["function"];
        // A string of JSON in OpenAI's format; Ollama has sent the object
        // itself.
        let arguments = match &function["arguments"] {
            Value::String(text) if text.trim().is_empty() => Value::Object(Map::new()),
            Value::String(text) => {
                serde_json::from_str(text).unwrap_or_else(|_| json!({ "_unparsed": text }))
            }
            other => other.clone(),
        };
        let id = call["id"]
            .as_str()
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("call_{index}"));
        turn.calls.push(Call {
            id,
            name: function["name"].as_str().unwrap_or_default().to_owned(),
            arguments,
        });
    }
    Ok(turn)
}

/// The arguments as an object, for formats that take nothing else.
fn object_or_empty(arguments: &Value) -> Value {
    if arguments.is_object() {
        arguments.clone()
    } else {
        Value::Object(Map::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_anthropic_answer_reads_its_text_and_its_calls() {
        let reply = json!({
            "content": [
                { "type": "text", "text": "Cutting it." },
                { "type": "tool_use", "id": "t1", "name": "clips_split", "input": { "time": 2.5 } },
            ],
        });
        let turn = anthropic_turn(&reply).unwrap();
        assert_eq!(turn.text, "Cutting it.");
        assert_eq!(turn.calls.len(), 1);
        assert_eq!(turn.calls[0].arguments["time"], 2.5);
    }

    #[test]
    fn an_openai_answer_parses_string_arguments_and_names_missing_ids() {
        let reply = json!({
            "choices": [{ "message": {
                "content": null,
                "tool_calls": [
                    { "function": { "name": "timeline_read", "arguments": "" } },
                    { "id": "x", "function": { "name": "clip_trim", "arguments": "{\"clipId\":\"c1\"}" } },
                ],
            }}],
        });
        let turn = openai_turn(&reply).unwrap();
        assert_eq!(turn.calls[0].id, "call_0");
        assert!(turn.calls[0].arguments.as_object().unwrap().is_empty());
        assert_eq!(turn.calls[1].arguments["clipId"], "c1");
    }

    #[test]
    fn results_become_one_tool_message_each_in_the_openai_format() {
        let messages = vec![
            Message::User("hi".into()),
            Message::Assistant {
                text: String::new(),
                calls: vec![
                    Call {
                        id: "a".into(),
                        name: "x".into(),
                        arguments: json!({}),
                    },
                    Call {
                        id: "b".into(),
                        name: "y".into(),
                        arguments: json!({}),
                    },
                ],
            },
            Message::Results(vec![
                Outcome {
                    id: "a".into(),
                    name: "x".into(),
                    content: "ok".into(),
                    is_error: false,
                },
                Outcome {
                    id: "b".into(),
                    name: "y".into(),
                    content: "ok".into(),
                    is_error: false,
                },
            ]),
        ];
        let wire = openai_messages(&messages);
        assert_eq!(wire.len(), 4);
        assert_eq!(wire[2]["role"], "tool");
        assert_eq!(wire[3]["tool_call_id"], "b");
    }

    #[test]
    fn anthropic_runs_two_user_turns_together() {
        let messages = vec![
            Message::User("cut it".into()),
            Message::Assistant {
                text: String::new(),
                calls: vec![Call {
                    id: "a".into(),
                    name: "x".into(),
                    arguments: json!({}),
                }],
            },
            Message::Results(vec![Outcome {
                id: "a".into(),
                name: "x".into(),
                content: "Stopped".into(),
                is_error: true,
            }]),
            Message::User("now the titles".into()),
        ];
        let wire = anthropic_messages(&messages);
        assert_eq!(wire.len(), 3);
        assert_eq!(wire[2]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn models_said_aloud_become_api_names() {
        assert_eq!(normalize_model("opus 5.5"), "claude-opus-5-5");
        assert_eq!(normalize_model(" Claude Sonnet 5.5 "), "claude-sonnet-5-5");
        assert_eq!(normalize_model("claude-haiku-5.5"), "claude-haiku-5-5");
        assert_eq!(normalize_model("gpt-5.5"), "gpt-5.5");
        assert_eq!(normalize_model("qwen3:8b"), "qwen3:8b");
    }

    #[test]
    fn a_key_under_the_wrong_provider_is_sent_where_it_belongs() {
        let settings = Settings {
            provider: Provider::Ollama,
            model: "opus 5.5".into(),
            api_key: "sk-ant-api03-xyz".into(),
            base_url: String::new(),
        };
        let resolved = settings.resolved();
        assert_eq!(resolved.provider, Provider::Anthropic);
        assert_eq!(resolved.model, "claude-opus-5-5");
        assert_eq!(resolved.base_url(), "https://api.anthropic.com");

        let openai = Settings {
            provider: Provider::Ollama,
            model: "opus 5.5".into(),
            api_key: "sk-proj-abc".into(),
            base_url: String::new(),
        }
        .resolved();
        assert_eq!(openai.provider, Provider::Openai);
        assert_eq!(openai.model(), Provider::Openai.default_model());

        let local = Settings {
            provider: Provider::Ollama,
            ..Settings::default()
        }
        .resolved();
        assert_eq!(local.provider, Provider::Ollama);
    }

    #[test]
    fn a_key_is_asked_for_only_where_one_is_needed() {
        let mut settings = Settings::default();
        assert_eq!(settings.problem(), Some(Problem::Key));
        settings.provider = Provider::Ollama;
        assert_eq!(settings.problem(), None);
        assert_eq!(settings.base_url(), "http://localhost:11434/v1");
        settings.provider = Provider::Custom;
        assert_eq!(settings.problem(), Some(Problem::Address));
    }
}
