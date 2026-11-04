// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah

//! The third [`Backend`] the MCP module's notes foresaw: the window's own
//! open project ("live mode", #337).
//!
//! The Copilot's tools are the MCP server's tools, unchanged; only where
//! their requests go differs. A request that edits or reads the project is
//! carried to the event loop, applied to the window's [`Studio`] exactly as
//! a gesture would be - one undo step, the panes republished - and the
//! engine's view comes back across a channel. A request that needs no
//! project (a file's probe, the effect catalogue, the templates) goes to an
//! API of this thread's own, which shares nothing with the window.
//! Anything that would open, close or export a project is refused with a
//! sentence the model can act on: the Copilot works on what is on screen.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use concat_api::{Api, ApiError, ErrorCode, Request, Response};
use concat_host::EditorView;
use concat_mcp::Backend;
use serde_json::{Value, json};

use crate::host::Shell;
use crate::studio::Studio;

/// How long an edit may take on the event loop before the Copilot gives up
/// waiting for it. An edit is a moment; this only stops a hung window from
/// hanging the worker with it.
const PATIENCE: Duration = Duration::from_secs(60);

/// The window's open project, as a backend.
#[derive(Default)]
pub struct Live {
    /// For the requests that need no project; made on first use.
    spare: Option<Api>,
}

impl Live {
    fn spare(&mut self) -> Result<&mut Api, ApiError> {
        if self.spare.is_none() {
            self.spare = Some(Api::new(Arc::new(|_| {})).map_err(ApiError::failed)?);
        }
        Ok(self.spare.as_mut().expect("made above"))
    }
}

impl Backend for Live {
    fn call(&mut self, request: Request) -> Result<Value, ApiError> {
        match request {
            Request::ProjectGet { path } => on_window(path, |studio| studio.live_view().map(view)),
            Request::EditApply { path, command } => {
                on_window(path, move |studio| studio.live_apply(*command).map(view))
            }
            Request::EditUndo { path } => on_window(path, |studio| studio.live_undo().map(view)),
            Request::EditRedo { path } => on_window(path, |studio| studio.live_redo().map(view)),
            Request::MediaImport { path, file } => {
                on_window(path, move |studio| studio.live_import(&file).map(view))
            }
            Request::ProjectSave { path, .. } => on_window(path, |studio| {
                studio.live_save().map(|()| json!({ "saved": true }))
            }),
            request @ (Request::Version
            | Request::MediaProbe { .. }
            | Request::CatalogueList { .. }
            | Request::TemplateList) => match self.spare()?.dispatch(request) {
                Response::Result(reply) => serde_json::to_value(&reply).map_err(|error| {
                    ApiError::failed(format!("the reply did not encode: {error}"))
                }),
                Response::Error(error) => Err(error),
            },
            _ => Err(ApiError::new(
                ErrorCode::Refused,
                "the Copilot edits the project open in the window; opening, closing, \
                 exporting and previewing projects are done by the person with the window's own \
                 controls",
            )),
        }
    }

    fn describe(&self) -> Value {
        json!({ "mode": "live" })
    }

    fn finish(&mut self) {
        if let Some(api) = self.spare.as_mut() {
            api.finish();
        }
    }
}

/// An editor view as the API's JSON.
pub(super) fn view(view: EditorView) -> Value {
    serde_json::to_value(&view).unwrap_or(Value::Null)
}

/// Runs `body` against the window's state on the event loop, republishes
/// the panes, and waits for its answer.
pub(super) fn on_window(
    path: String,
    body: impl FnOnce(&mut Studio) -> Result<Value, String> + Send + 'static,
) -> Result<Value, ApiError> {
    let (sender, answer) = mpsc::channel();
    slint::invoke_from_event_loop(move || {
        Shell::with(|shell, app| {
            // Check on the event loop, immediately before the operation: the
            // person may have switched projects while the model was thinking.
            let result = {
                let mut studio = shell.studio.borrow_mut();
                ensure_project(&path, studio.live_path().as_deref())
                    .and_then(|()| body(&mut studio))
            };
            shell.studio.borrow().publish(&app, &shell.models);
            let _ = sender.send(result);
        });
        // A window that is gone drops the sender unsent, which the wait
        // below reads as the window having closed.
    })
    .map_err(|error| ApiError::failed(format!("the window could not be reached: {error}")))?;
    match answer.recv_timeout(PATIENCE) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(refusal)) => Err(ApiError::new(ErrorCode::Refused, refusal)),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            Err(ApiError::failed("the window took too long to answer"))
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(ApiError::failed("the window has closed")),
    }
}

/// A late model reply must never edit a different project with reused ids.
fn ensure_project(requested: &str, open: Option<&str>) -> Result<(), String> {
    if open == Some(requested) {
        Ok(())
    } else {
        Err(
            "the project changed or was closed while the Copilot was working; \
             send a new request for the project now open"
                .to_owned(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::ensure_project;

    #[test]
    fn late_replies_are_refused_after_switching_or_closing_the_project() {
        assert!(ensure_project("project-a", Some("project-a")).is_ok());
        assert!(ensure_project("project-a", Some("project-b")).is_err());
        assert!(ensure_project("project-a", None).is_err());
        assert!(ensure_project("", Some("project-b")).is_err());
    }
}
