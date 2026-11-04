// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah

//! The Copilot pane: a chat with an AI model that edits the open project.
//!
//! The person writes what they want ("cut the silences", "put a title on
//! the first three seconds"); a worker thread sends it to the provider the
//! pane is set to ([`provider`]), runs the edits the model asks for on the
//! timeline ([`agent`], through [`backend`]), and the chat shows each step
//! and the model's answer. Every edit is an ordinary undo step.
//!
//! This file is the event-loop side: the chat's lines, the conversation
//! kept between requests, the settings and where they are saved, and the
//! wiring of the `CopilotChat` global's callbacks.

pub mod agent;
mod backend;
pub mod provider;
mod window_tools;

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};

use crate::host::Shell;
use crate::i18n::t;
use crate::ui::{App, CopilotChat, CopilotLine};
use agent::Event;
use provider::{Message, Problem, Provider, Settings};

/// The settings file, in the config folder beside the preferences.
const SETTINGS_FILE: &str = "copilot.json";

/// The most lines the chat keeps; older ones scroll away for good.
const LINES: usize = 400;

/// The pane's state, on the event-loop thread.
struct State {
    lines: Rc<VecModel<CopilotLine>>,
    /// The conversation the next request continues.
    history: Vec<Message>,
    settings: Settings,
    /// Where `settings` is saved.
    file: PathBuf,
    /// The running request's stop flag, while one runs.
    running: Option<Arc<AtomicBool>>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

fn with_state<R>(body: impl FnOnce(&mut State) -> R) -> Option<R> {
    STATE.with(|slot| slot.borrow_mut().as_mut().map(body))
}

/// Reads the saved settings, hands the pane its model and wires its
/// callbacks. Once, after the window is made.
pub fn install(app: &App, config: PathBuf) {
    let file = config.join(SETTINGS_FILE);
    let settings: Settings = std::fs::read_to_string(&file)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    let lines = Rc::new(VecModel::<CopilotLine>::default());

    let chat = app.global::<CopilotChat>();
    chat.set_lines(ModelRc::from(lines.clone()));
    chat.set_default_models(ModelRc::from(Rc::new(VecModel::from(
        Provider::ALL
            .iter()
            .map(|provider| SharedString::from(provider.default_model()))
            .collect::<Vec<_>>(),
    ))));
    chat.set_default_urls(ModelRc::from(Rc::new(VecModel::from(
        Provider::ALL
            .iter()
            .map(|provider| SharedString::from(provider.default_base_url()))
            .collect::<Vec<_>>(),
    ))));
    show_settings(app, &settings);

    STATE.with(|slot| {
        *slot.borrow_mut() = Some(State {
            lines,
            history: Vec::new(),
            settings,
            file,
            running: None,
        });
    });

    let weak = app.as_weak();
    chat.on_send(move |text| {
        if let Some(app) = weak.upgrade() {
            send(&app, text.as_str());
        }
    });
    chat.on_stop(|| {
        with_state(|state| {
            if let Some(stop) = &state.running {
                stop.store(true, Ordering::Relaxed);
            }
        });
    });
    let weak = app.as_weak();
    chat.on_clear(move || {
        with_state(|state| {
            if state.running.is_none() {
                state.history.clear();
                state.lines.set_vec(Vec::new());
            }
        });
        if let Some(app) = weak.upgrade() {
            app.global::<CopilotChat>().set_status(SharedString::new());
        }
    });
    let weak = app.as_weak();
    chat.on_save_settings(move |provider, model, key, url| {
        let settings = Settings {
            provider: Provider::at(provider),
            model: model.trim().to_owned(),
            api_key: key.trim().to_owned(),
            base_url: url.trim().to_owned(),
        };
        with_state(|state| {
            state.settings = settings.clone();
            if let Err(error) = save(&state.file, &settings) {
                log::warn!("copilot: the settings could not be saved: {error}");
            }
        });
        if let Some(app) = weak.upgrade() {
            show_settings(&app, &settings);
        }
    });
}

/// Puts `settings` in the pane's fields.
fn show_settings(app: &App, settings: &Settings) {
    let chat = app.global::<CopilotChat>();
    chat.set_provider(settings.provider.index());
    chat.set_model(settings.model.as_str().into());
    chat.set_api_key(settings.api_key.as_str().into());
    chat.set_base_url(settings.base_url.as_str().into());
    chat.set_model_shown(settings.model().into());
    chat.set_configured(settings.problem().is_none());
}

/// Writes `settings` where [`install`] reads them, readable by this user
/// alone where the system can say so: there is a key in it.
fn save(file: &std::path::Path, settings: &Settings) -> std::io::Result<()> {
    if let Some(folder) = file.parent() {
        std::fs::create_dir_all(folder)?;
    }
    let text = serde_json::to_string_pretty(settings).map_err(std::io::Error::other)?;
    std::fs::write(file, text)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(file, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Adds a line to the chat.
fn push(state: &State, role: &str, text: &str) {
    if state.lines.row_count() >= LINES {
        state.lines.remove(0);
    }
    state.lines.push(CopilotLine {
        role: role.into(),
        text: text.into(),
    });
}

/// Starts a request.
fn send(app: &App, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    // The project on screen, read now: the worker cannot reach the window's
    // state except through the event loop, and this is the event loop.
    let mut project = None;
    Shell::with(|shell, _| project = shell.studio.borrow().live_path());

    let started = with_state(|state| {
        if state.running.is_some() {
            return None;
        }
        push(state, "user", text);
        if let Some(problem) = state.settings.problem() {
            let sentence = match problem {
                Problem::Key => t("copilot.needsKey"),
                Problem::Address => t("copilot.needsAddress"),
                Problem::Model => t("copilot.needsModel"),
            };
            push(state, "error", &sentence);
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        state.running = Some(stop.clone());
        let mut history = state.history.clone();
        history.push(Message::User(text.to_owned()));
        Some((state.settings.clone(), history, stop))
    })
    .flatten();
    let Some((settings, history, stop)) = started else {
        return;
    };

    let chat = app.global::<CopilotChat>();
    chat.set_busy(true);
    chat.set_status(t("copilot.thinking").into());

    let spawned = std::thread::Builder::new()
        .name("copilot".to_owned())
        .spawn(move || {
            agent::run(&settings, history, project, &stop, &|event| {
                let _ = slint::invoke_from_event_loop(move || heard(event));
            });
        });
    if let Err(error) = spawned {
        with_state(|state| {
            state.running = None;
            push(state, "error", &error.to_string());
        });
        chat.set_busy(false);
        chat.set_status(SharedString::new());
    }
}

/// What the worker said, on the event loop.
fn heard(event: Event) {
    let finished = matches!(event, Event::Finished { .. });
    with_state(|state| match event {
        Event::Step(title) => push(state, "tool", &title),
        Event::Finished {
            history,
            reply,
            failure,
        } => {
            state.history = history;
            state.running = None;
            if !reply.is_empty() {
                push(state, "assistant", &reply);
            }
            match failure.as_deref() {
                Some("copilot.stopped") => push(state, "error", &t("copilot.stopped")),
                Some(error) => {
                    push(state, "error", &format!("{} {error}", t("copilot.failed")));
                }
                None => {}
            }
        }
    });
    Shell::with(|_, app| {
        let chat = app.global::<CopilotChat>();
        if finished {
            chat.set_busy(false);
            chat.set_status(SharedString::new());
        } else {
            chat.set_status(t("copilot.working").into());
        }
        chat.set_scroll_token(chat.get_scroll_token() + 1);
    });
}
