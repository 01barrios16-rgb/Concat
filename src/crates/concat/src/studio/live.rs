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
        self.session
            .as_ref()
            .map(|session| session.path().to_owned())
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

    /// What the person is looking at besides the edit: the playhead, the
    /// selection, the frame size. Handed to the model with every request
    /// so "this clip" and "here" mean something.
    pub fn live_context(&self) -> serde_json::Value {
        let video = self.session.as_ref().map(|session| session.video());
        serde_json::json!({
            "playheadSeconds": self.playhead,
            "selectedClipIds": self.selection,
            "frame": video.map(|video| serde_json::json!({
                "width": video.width,
                "height": video.height,
            })),
        })
    }

    /// The facts about one clip a sound analysis needs: its media file and
    /// how the clip maps the file onto the timeline.
    pub fn live_clip_source(&self, clip_id: &str) -> Result<ClipSource, String> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| "no project is open in the window".to_owned())?;
        let project = session.project();
        let clip = project
            .active()
            .clips
            .iter()
            .find(|clip| clip.id == clip_id)
            .ok_or_else(|| format!("there is no clip {clip_id} on the active timeline"))?;
        if clip.speed_curve.is_some() {
            return Err("this clip plays on a speed curve; take the curve off first".to_owned());
        }
        let media = project
            .media
            .iter()
            .find(|item| item.id == clip.media_id)
            .ok_or_else(|| "this clip has no media file (a title or a shape)".to_owned())?;
        Ok(ClipSource {
            path: media.path.clone(),
            stream: clip.audio_stream.map(|stream| stream as usize),
            start: clip.start,
            duration: clip.duration,
            source_start: clip.source_start,
            speed: if clip.speed > 0.0 { clip.speed } else { 1.0 },
        })
    }

    /// Cuts the timeline spans `spans` (seconds, any order) out of the clip
    /// `clip_id` and closes the gaps, as one undo step: split at each end,
    /// remove the piece between, ripple. Returns the view and how many
    /// pieces went.
    pub fn live_cut_spans(
        &mut self,
        clip_id: &str,
        spans: &[(f64, f64)],
    ) -> Result<(EditorView, usize), String> {
        const GESTURE: &str = "copilot-cut";
        self.end_burst();
        self.flush_commit();
        self.echo = None;
        self.last_commit = None;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| "no project is open in the window".to_owned())?;
        let mut spans: Vec<(f64, f64)> = spans
            .iter()
            .copied()
            .filter(|(from, to)| to > from)
            .collect();
        // Latest first: a cut and its ripple only move what comes after it,
        // so every earlier span is still where it was measured.
        spans.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut removed = 0;
        let mut outcome = Ok(());
        for (from, to) in spans {
            let clip_start = session
                .project()
                .active()
                .clips
                .iter()
                .find(|clip| clip.id == clip_id)
                .map(|clip| clip.start);
            let Some(clip_start) = clip_start else {
                break;
            };
            let step = (|| {
                session.apply_within(
                    Some(GESTURE),
                    Command::SplitClips {
                        clip_ids: vec![clip_id.to_owned()],
                        time: to,
                    },
                )?;
                // At the clip's very head the clip itself is the silence.
                let piece = if from <= clip_start + 1.0 / 60.0 {
                    Some(clip_id.to_owned())
                } else {
                    session
                        .apply_within(
                            Some(GESTURE),
                            Command::SplitClips {
                                clip_ids: vec![clip_id.to_owned()],
                                time: from,
                            },
                        )?
                        .created_id
                };
                if let Some(piece) = piece {
                    session.apply_within(
                        Some(GESTURE),
                        Command::RemoveClips {
                            clip_ids: vec![piece],
                            ripple: true,
                        },
                    )?;
                    return Ok::<bool, String>(true);
                }
                Ok(false)
            })();
            match step {
                Ok(true) => removed += 1,
                Ok(false) => {}
                Err(error) => {
                    outcome = Err(error);
                    break;
                }
            }
        }
        session.end_gesture();
        self.after_change();
        outcome?;
        Ok((self.live_view()?, removed))
    }

    /// Moves the playhead, as a click on the ruler does.
    pub fn live_seek(&mut self, seconds: f64) -> Result<EditorView, String> {
        self.seek(seconds.max(0.0) as f32);
        self.live_view()
    }

    /// Selects exactly these clips, as clicking them would; unknown ids are
    /// left out.
    pub fn live_select(&mut self, ids: Vec<String>) -> Result<Vec<String>, String> {
        let view = self.live_view()?;
        let known: Vec<String> = ids
            .into_iter()
            .filter(|id| {
                view.project
                    .active()
                    .clips
                    .iter()
                    .any(|clip| &clip.id == id)
            })
            .collect();
        self.selection = known.clone();
        self.after_change();
        Ok(known)
    }

    /// Starts the Captions sheet's transcription over a clip's voice, as
    /// its Generate button does; the captions land when it finishes.
    pub fn live_auto_captions(&mut self, clip_id: Option<String>) -> Result<String, String> {
        use crate::panes::Msg;
        use crate::panes::captions::CaptionsMsg;
        self.live_view()?;
        match clip_id {
            Some(id) => self.handle(Msg::Captions(CaptionsMsg::OpenOnClip(id))),
            None => self.handle(Msg::Captions(CaptionsMsg::Open)),
        }
        self.handle(Msg::Captions(CaptionsMsg::VoiceToggled(true)));
        self.handle(Msg::Captions(CaptionsMsg::Begin));
        Ok(
            "Transcription started in the Captions sheet. The captions are added to the \
            timeline by themselves when it finishes (seconds to a few minutes); call \
            timeline_read later to see them. If no speech model is installed the sheet \
            says so and the person installs one in Settings > Models."
                .to_owned(),
        )
    }

    /// Starts an export with the Export sheet's settings, as its Export
    /// button does, under `name` when one is given.
    pub fn live_export(&mut self, name: Option<String>) -> Result<String, String> {
        use crate::panes::Msg;
        use crate::panes::export::ExportMsg;
        self.live_view()?;
        self.handle(Msg::Export(ExportMsg::Open));
        if let Some(name) = name.filter(|name| !name.trim().is_empty()) {
            self.handle(Msg::Export(ExportMsg::NameEdited(name)));
        }
        self.handle(Msg::Export(ExportMsg::Start));
        Ok(
            "The export started in the Export sheet, at the timeline's frame size and \
            rate, into the folder the sheet shows; the person sees its progress there."
                .to_owned(),
        )
    }
}

/// One clip's media and timing, for an analysis that reads the file.
#[derive(Clone, Debug)]
pub struct ClipSource {
    pub path: String,
    pub stream: Option<usize>,
    pub start: f64,
    pub duration: f64,
    pub source_start: f64,
    pub speed: f64,
}
