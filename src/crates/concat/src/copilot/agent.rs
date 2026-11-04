// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah

//! One request to the Copilot, start to finish, on a worker thread: ask
//! the model, run the tools it calls against the open project, hand it
//! what they did, and go round until it answers in words - or until it has
//! gone round [`ROUNDS`] times, or the person pressed Stop.
//!
//! The tools are the MCP server's ([`concat_mcp::tools`]): the same names,
//! descriptions and schemas an outside agent gets, so what the Copilot can
//! do grows with the server and needs no second list kept in step. Only
//! the project's `path` is taken out of their arguments - there is one
//! project, the one on screen - and put back before the call.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use concat_mcp::tools::{Context, Toolbox};
use serde_json::{Value, json};

use super::backend::Live;
use super::provider::{self, Call, Message, Outcome, Settings, Tool};

/// The most times the model is asked again after its tools ran, in one
/// request. A long edit - captions over a minute of speech, a cut every few
/// seconds - is a handful of rounds; past this it is going in circles.
const ROUNDS: usize = 60;

/// The longest tool result handed back, in characters. A timeline summary
/// of a long edit can run to tens of thousands; the model needs the shape
/// and the ids, and every character is paid for on every later round.
const RESULT_CHARS: usize = 14_000;

/// The MCP tools the Copilot is given. Not the ones that open, close or
/// export projects, which are the person's to do, nor the raw `edit_apply`,
/// whose nested schema some providers reject and whose every command has
/// a tool of its own here.
const ALLOWED: &[&str] = &[
    "timeline_read",
    "project_save",
    "media_probe",
    "media_import",
    "edit_undo",
    "edit_redo",
    "clip_add",
    "clips_add_sequence",
    "clip_trim",
    "clips_split",
    "clips_move",
    "clips_remove",
    "clip_update",
    "clip_mask",
    "clip_stabilize",
    "clip_frame",
    "timeline_background",
    "timeline_linkage",
    "clip_transform",
    "clip_speed",
    "title_add",
    "title_style_words",
    "title_clear_word_styles",
    "captions_add",
    "keyframe_set",
    "keyframe_clear",
    "effect_layer_add",
    "effect_key_set",
    "track_add",
    "effects_list",
    "template_list",
];

/// Tools whose `path` is a file of the person's, not the project's.
const PATH_IS_A_FILE: &[&str] = &["media_probe"];

/// What the worker tells the window as it goes.
pub enum Event {
    /// A tool is about to run: its title, for the chat's step line.
    Step(String),
    /// The request is over. The conversation as it now stands, to send
    /// with the next request; the model's last words, if it had any; and
    /// what went wrong, if something did.
    Finished {
        history: Vec<Message>,
        reply: String,
        failure: Option<String>,
    },
}

/// The toolbox, and its tools in the provider's terms, made once.
fn tools() -> &'static (Toolbox, Vec<Tool>) {
    static TOOLS: OnceLock<(Toolbox, Vec<Tool>)> = OnceLock::new();
    TOOLS.get_or_init(|| {
        let toolbox = Toolbox::new();
        let tools = ALLOWED
            .iter()
            .filter_map(|name| {
                let (tool, schema) = toolbox.find(name)?;
                let mut schema = schema.clone();
                if !PATH_IS_A_FILE.contains(name) {
                    drop_path(&mut schema);
                }
                Some(Tool {
                    name: tool.name.to_owned(),
                    description: tool.description.to_owned(),
                    schema,
                })
            })
            .chain(super::window_tools::tools())
            .collect();
        (toolbox, tools)
    })
}

/// What is on screen as the request starts: the timeline's summary (the
/// same one timeline_read gives) and the playhead and selection, so the
/// model starts from the edit instead of spending a round asking for it.
fn screen_state(
    toolbox: &Toolbox,
    backend: &mut Live,
    project: Option<&str>,
    stop: &AtomicBool,
) -> String {
    let Some(path) = project else {
        return "(no project open)".to_owned();
    };
    let timeline = toolbox
        .call(
            "timeline_read",
            json!({ "path": path }),
            Context {
                backend,
                cancelled: stop,
                progress: None,
            },
        )
        .map(|outcome| {
            outcome
                .content
                .iter()
                .filter_map(|block| block["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    let owned = path.to_owned();
    let view = super::backend::on_window(owned, |studio| Ok(studio.live_context()))
        .map(|context| context.to_string())
        .unwrap_or_default();
    clip_text(format!("Timeline: {timeline}\nView: {view}"))
}

/// A tool result cut to what the model needs.
fn clip_text(mut content: String) -> String {
    if content.chars().count() > RESULT_CHARS {
        content = content.chars().take(RESULT_CHARS).collect();
        content.push_str("\n… (cut short; call timeline_read for the rest)");
    }
    content
}

/// Takes the project's `path` out of a tool's schema.
fn drop_path(schema: &mut Value) {
    if let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut) {
        properties.remove("path");
    }
    if let Some(required) = schema.get_mut("required").and_then(Value::as_array_mut) {
        required.retain(|field| field.as_str() != Some("path"));
    }
}

/// What the model is told before anything else: who it is, the project,
/// what is on screen right now, and how short-form edits are made.
fn system_prompt(project: Option<&str>, state: &str) -> String {
    let project = match project {
        Some(path) => format!("The project open in the window is the folder {path}."),
        None => "No project is open in the window yet; editing tools will fail until the \
                 person opens or creates one; tell them so."
            .to_owned(),
    };
    format!(
        "You are the Copilot inside Concat, a video editor, with full control of the \
         project the person has open. Every tool call is applied at once on their timeline as \
         an undoable step (Ctrl+Z).\n\
         {project}\n\n\
         WHAT IS ON SCREEN NOW (read at the start of this request):\n{state}\n\n\
         HOW TO WORK\n\
         - Act, do not lecture. When the person asks for an edit, make it with the tools, \
         then say briefly what you did. Never answer with instructions for them to follow \
         when a tool can do it.\n\
         - Use the ids above or from timeline_read; never invent ids. Call timeline_read \
         again after big changes to get fresh ids and times.\n\
         - Times are seconds on the timeline; positions in the frame are fractions.\n\
         - \"This clip\", \"the selected one\", \"here\" mean the selection and the \
         playhead above.\n\
         - Broad creative requests (\"make it viral\", \"edit it for TikTok\", \
         \"aggressive\", \"more dynamic\") are not ambiguous: carry out the recipe below \
         fully, choosing sensible values yourself. Ask a question only when you truly \
         cannot proceed (no project, no clips).\n\
         - Use effects_list to find effect, filter and transition ids before applying them.\n\
         - Files: the person gives a full path; media_import then clip_add.\n\
         - Opening/closing projects is the person's to do with the window's buttons.\n\n\
         RECIPE: AGGRESSIVE TIKTOK / SHORTS / REELS EDIT (\"agresivo\", \"dinámico\", \
         \"viral\")\n\
         1. Vertical frame: project_set_video to 1080x1920 at the timeline's rate (keep \
         30 or 60 fps), then clip_frame each video clip with framing \"fill\" so it covers \
         the frame.\n\
         2. Pace: remove_silences on every clip with voice (thresholdDb -35, minSilence 0.3; \
         for gameplay with constant sound use -24 and 0.25). Trim slow intros so the first \
         second already has action (clip_trim, ripple). Long shots: clips_split every \
         2-4 s at the best moments and drop dead parts (clips_remove with ripple).\n\
         3. Hook: a big bold title_add in the first 0-2 s (3-6 words, CAPS, punchy, in the \
         person's language), near the top third.\n\
         4. Energy: punch-in zooms on key moments with keyframe_set on scale (1.0 -> 1.15-1.3 \
         over 0.1-0.2 s, back after), small speed bumps with clip_speed (1.1-1.5x) on slow \
         parts, quick transitions or flash/shake/glitch effects from effects_list between \
         cuts. Do not overdo one effect: vary them.\n\
         5. Captions: auto_captions on the main voice clip (big, centred-low captions are \
         what short-form viewers expect).\n\
         6. End: finish on a strong moment; no slow fade-out.\n\
         Do every step you can with the tools, then summarise in 2-4 lines what changed and \
         what the person could still add (music, their own footage). Export only if asked \
         (export_video).\n\n\
         Answer in the language the person writes in (Spanish if they write Spanish)."
    )
}

/// Runs one request. `history` is the conversation so far, ending with the
/// person's new message.
pub fn run(
    settings: &Settings,
    mut history: Vec<Message>,
    project: Option<String>,
    stop: &AtomicBool,
    report: &dyn Fn(Event),
) {
    let (toolbox, tools) = tools();
    let mut backend = Live::default();
    let state = screen_state(toolbox, &mut backend, project.as_deref(), stop);
    let system = system_prompt(project.as_deref(), &state);
    let mut reply = String::new();
    let mut failure = None;

    for _ in 0..ROUNDS {
        if stop.load(Ordering::Relaxed) {
            failure = Some("copilot.stopped".to_owned());
            break;
        }
        let turn = match provider::chat(settings, &system, &history, tools) {
            Ok(turn) => turn,
            Err(error) => {
                failure = Some(error);
                break;
            }
        };
        reply = turn.text.trim().to_owned();
        let calls = turn.calls;
        history.push(Message::Assistant {
            text: turn.text,
            calls: calls.clone(),
        });
        if calls.is_empty() {
            break;
        }
        let mut outcomes = Vec::with_capacity(calls.len());
        for call in calls {
            if stop.load(Ordering::Relaxed) {
                outcomes.push(Outcome {
                    id: call.id,
                    name: call.name,
                    content: "Stopped by the person before this ran.".to_owned(),
                    is_error: true,
                });
                continue;
            }
            let title = toolbox
                .find(&call.name)
                .map(|(tool, _)| tool.title)
                .or_else(|| super::window_tools::title(&call.name))
                .unwrap_or(call.name.as_str())
                .to_owned();
            report(Event::Step(title));
            outcomes.push(call_tool(
                toolbox,
                &mut backend,
                project.as_deref(),
                call,
                stop,
            ));
        }
        history.push(Message::Results(outcomes));
        // The last round's calls ran, but the model never saw what they did.
        reply.clear();
    }
    use concat_mcp::Backend as _;
    backend.finish();
    report(Event::Finished {
        history,
        reply,
        failure,
    });
}

/// Runs one call and words its result for the model.
fn call_tool(
    toolbox: &Toolbox,
    backend: &mut Live,
    project: Option<&str>,
    call: Call,
    stop: &AtomicBool,
) -> Outcome {
    if super::window_tools::has(&call.name) {
        let result =
            super::window_tools::call(&call.name, &call.arguments, project.unwrap_or_default());
        let (content, is_error) = match result {
            Ok(text) => (text, false),
            Err(error) => (error, true),
        };
        return Outcome {
            id: call.id,
            name: call.name,
            content: clip_text(content),
            is_error,
        };
    }
    let allowed = ALLOWED.contains(&call.name.as_str());
    let mut arguments = match call.arguments {
        Value::Object(fields) => Value::Object(fields),
        _ => json!({}),
    };
    if !PATH_IS_A_FILE.contains(&call.name.as_str()) {
        arguments["path"] = json!(project.unwrap_or_default());
    }
    let outcome = if allowed {
        toolbox.call(
            &call.name,
            arguments,
            Context {
                backend,
                cancelled: stop,
                progress: None,
            },
        )
    } else {
        None
    };
    let Some(outcome) = outcome else {
        return Outcome {
            id: call.id,
            content: format!("There is no tool called {}.", call.name),
            name: call.name,
            is_error: true,
        };
    };
    let mut content: String = outcome
        .content
        .iter()
        .filter_map(|block| block["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    if content.chars().count() > RESULT_CHARS {
        content = content.chars().take(RESULT_CHARS).collect();
        content.push_str("\n… (cut short; call timeline_read for the rest)");
    }
    Outcome {
        id: call.id,
        name: call.name,
        content,
        is_error: outcome.is_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_allowed_tool_is_one_the_server_has() {
        let (toolbox, tools) = tools();
        for name in ALLOWED {
            assert!(toolbox.find(name).is_some(), "the MCP server has no {name}");
        }
        assert_eq!(
            tools.len(),
            ALLOWED.len() + super::super::window_tools::tools().len()
        );
    }

    #[test]
    fn the_project_path_is_taken_out_of_every_schema_but_a_files() {
        let (_, tools) = tools();
        for tool in tools {
            let has_path = tool.schema["properties"].get("path").is_some();
            assert_eq!(
                has_path,
                PATH_IS_A_FILE.contains(&tool.name.as_str()),
                "{}",
                tool.name
            );
            assert_eq!(tool.schema["type"], "object", "{}", tool.name);
        }
    }
}
