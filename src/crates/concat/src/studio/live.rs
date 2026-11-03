// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah

//! The open project as the Copilot edits it ("live mode", #337): the same
//! commands the window's own gestures make, through the same bookkeeping
//! as [`Studio::apply`], but with the engine's view - or its refusal, as a
//! sentence - handed back to the caller instead of a toast, because the
//! caller is a model that has to read what happened to decide what next.

use concat_host::EditorView;
use concat_host::media;
use concat_project::Command;

use super::Studio;

impl Studio {
    /// The open project's folder, if one is open.
    pub fn live_path(&self) -> Option<String> {
        self.session.as_ref().map(|session| session.path().to_owned())
    }

    /// The open project as the engine sees it.
    pub fn live_view(&self) -> Result<EditorView, String> {
        self.session
            .as_ref()
            .map(|session| session.view())
            .ok_or_else(|| "no project is open in the window".to_owned())
    }

    /// Applies `command` to the open project as one undo step.
    pub fn live_apply(&mut self, command: Command) -> Result<EditorView, String> {
        self.end_burst();
        log::info!("copilot edit: {}", super::command_name(&command));
        self.flush_commit();
        self.echo = None;
        self.last_commit = None;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| "no project is open in the window".to_owned())?;
        let before = session.video().color_space;
        let view = session.apply(command)?;
        let after = session.video().color_space;
        self.after_change();
        self.tell_of_hdr(before, after);
        Ok(view)
    }

    /// Steps the open project's history back, as the window's Undo does.
    pub fn live_undo(&mut self) -> Result<EditorView, String> {
        self.undo();
        self.live_view()
    }

    /// Steps it forward again.
    pub fn live_redo(&mut self) -> Result<EditorView, String> {
        self.redo();
        self.live_view()
    }

    /// Probes `file` and puts it in the open project's bin.
    pub fn live_import(&mut self, file: &str) -> Result<EditorView, String> {
        let item = media::probe(file)?.to_new_media();
        self.live_apply(Command::AddMedia { item })
    }

    /// Saves the open project, as Ctrl+S does.
    pub fn live_save(&mut self) -> Result<(), String> {
        if self.session.is_none() {
            return Err("no project is open in the window".to_owned());
        }
        self.save(true);
        Ok(())
    }
}
