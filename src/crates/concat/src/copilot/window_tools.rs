// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah

//! The Copilot's tools that the MCP server does not have, because they
//! are the window's: the playhead and the selection, the Captions sheet's
//! transcription, the Export sheet, and a cut that listens to the sound -
//! silences found in a clip's audio and taken out as one undo step.
//!
//! Each one runs on the event loop through [`super::backend::on_window`],
//! except the listening, which reads the file on the Copilot's worker so
//! the window never waits on a decode.

use serde_json::{Value, json};

use super::backend::{on_window, view};
use super::provider::Tool;

/// The tools, in the provider's terms.
pub fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "remove_silences".to_owned(),
            description: "Listens to a clip's sound and cuts out every pause quieter than \
                `thresholdDb` that lasts at least `minSilence` seconds, closing the gaps \
                (ripple), as ONE undo step. The tool for 'cut the silences', 'make it \
                faster/more dynamic', jump cuts, aggressive TikTok pacing. Works on video or \
                audio clips with sound; for loud gameplay with constant noise raise \
                thresholdDb (e.g. -25). Returns how many pauses were cut and the new \
                timeline."
                .to_owned(),
            schema: json!({
                "type": "object",
                "properties": {
                    "clipId": { "type": "string", "description": "The clip to listen to." },
                    "thresholdDb": { "type": "number", "default": -35, "minimum": -80, "maximum": -5,
                        "description": "Below this loudness (dBFS) is a pause. -35 for speech; -25 to -20 for gameplay with music or game noise." },
                    "minSilence": { "type": "number", "default": 0.35, "minimum": 0.1, "maximum": 5,
                        "description": "Shortest pause cut, in seconds. 0.25-0.35 is aggressive, 0.6 is gentle." },
                    "padding": { "type": "number", "default": 0.06, "minimum": 0, "maximum": 0.5,
                        "description": "Seconds of each pause kept beside the sound so words are not clipped." },
                },
                "required": ["clipId"],
            }),
        },
        Tool {
            name: "auto_captions".to_owned(),
            description: "Transcribes a clip's speech with the built-in Whisper model and adds \
                the captions as titles on the timeline, exactly as the Captions sheet does. \
                Runs in the background: the captions appear by themselves when it finishes. \
                Without clipId it takes the selected clip, or the first one with sound."
                .to_owned(),
            schema: json!({
                "type": "object",
                "properties": {
                    "clipId": { "type": "string", "description": "The clip whose voice is captioned." },
                },
            }),
        },
        Tool {
            name: "export_video".to_owned(),
            description: "Starts exporting the active timeline to a video file with the Export \
                sheet's settings (the timeline's frame size and rate, H.264), into the folder \
                the sheet shows. Only when the person asks to export or render."
                .to_owned(),
            schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "The file's name, without extension." },
                },
            }),
        },
        Tool {
            name: "playhead_seek".to_owned(),
            description: "Moves the playhead (and the monitor) to a time on the timeline, so \
                the person sees that moment."
                .to_owned(),
            schema: json!({
                "type": "object",
                "properties": {
                    "time": { "type": "number", "minimum": 0, "description": "Seconds on the timeline." },
                },
                "required": ["time"],
            }),
        },
        Tool {
            name: "select_clips".to_owned(),
            description: "Selects exactly these clips in the window, so the person sees which \
                ones you mean and the inspector shows them. An empty list clears the selection."
                .to_owned(),
            schema: json!({
                "type": "object",
                "properties": {
                    "clipIds": { "type": "array", "items": { "type": "string" } },
                },
                "required": ["clipIds"],
            }),
        },
    ]
}

/// The name a person is shown for one of these tools' steps.
pub fn title(name: &str) -> Option<&'static str> {
    Some(match name {
        "remove_silences" => "Cut the silences",
        "auto_captions" => "Generate captions",
        "export_video" => "Export the video",
        "playhead_seek" => "Move the playhead",
        "select_clips" => "Select clips",
        _ => return None,
    })
}

/// Whether `name` is one of these tools.
pub fn has(name: &str) -> bool {
    matches!(
        name,
        "remove_silences" | "auto_captions" | "export_video" | "playhead_seek" | "select_clips"
    )
}

/// Runs one of these tools: its result as text, or why it could not.
pub fn call(name: &str, arguments: &Value, project: &str) -> Result<String, String> {
    let path = project.to_owned();
    let text = |key: &str| arguments[key].as_str().map(str::to_owned);
    let number = |key: &str, default: f64| arguments[key].as_f64().unwrap_or(default);
    let refused = |error: concat_api::ApiError| error.message;
    match name {
        "remove_silences" => {
            let clip = text("clipId").ok_or("clipId is required")?;
            let threshold = number("thresholdDb", -35.0).clamp(-80.0, -5.0);
            let shortest = number("minSilence", 0.35).clamp(0.1, 5.0);
            let padding = number("padding", 0.06).clamp(0.0, 0.5);
            let id = clip.clone();
            let source = on_window(path.clone(), move |studio| {
                studio.live_clip_source(&id).map(|source| {
                    json!({
                        "path": source.path,
                        "stream": source.stream,
                        "start": source.start,
                        "duration": source.duration,
                        "sourceStart": source.source_start,
                        "speed": source.speed,
                    })
                })
            })
            .map_err(refused)?;
            let spans = silences(&source, threshold, shortest, padding)?;
            if spans.is_empty() {
                return Ok(format!(
                    "No pause quieter than {threshold} dB and longer than {shortest} s in \
                     that clip; nothing was cut. Try a higher thresholdDb for noisy audio."
                ));
            }
            let found = spans.len();
            let total: f64 = spans.iter().map(|(from, to)| to - from).sum();
            let reply = on_window(path, move |studio| {
                studio
                    .live_cut_spans(&clip, &spans)
                    .map(|(edited, removed)| json!({ "removed": removed, "view": view(edited) }))
            })
            .map_err(refused)?;
            let removed = reply["removed"].as_u64().unwrap_or(0);
            let summary = concat_mcp::summary::view(&reply["view"]);
            Ok(format!(
                "Cut {removed} of {found} pauses ({total:.1} s of silence) as one undo step.\n{}",
                serde_json::to_string(&summary).unwrap_or_default()
            ))
        }
        "auto_captions" => {
            let clip = text("clipId");
            on_window(path, move |studio| {
                studio.live_auto_captions(clip).map(Value::String)
            })
            .map(|reply| reply.as_str().unwrap_or_default().to_owned())
            .map_err(refused)
        }
        "export_video" => {
            let name = text("name");
            on_window(path, move |studio| {
                studio.live_export(name).map(Value::String)
            })
            .map(|reply| reply.as_str().unwrap_or_default().to_owned())
            .map_err(refused)
        }
        "playhead_seek" => {
            let time = number("time", 0.0);
            on_window(path, move |studio| {
                studio.live_seek(time).map(|_| json!(time))
            })
            .map(|_| format!("The playhead is at {time:.2} s."))
            .map_err(refused)
        }
        "select_clips" => {
            let ids: Vec<String> = arguments["clipIds"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|id| id.as_str().map(str::to_owned))
                .collect();
            on_window(path, move |studio| {
                studio.live_select(ids).map(|known| json!(known))
            })
            .map(|known| format!("Selected: {known}"))
            .map_err(refused)
        }
        _ => Err(format!("There is no tool called {name}.")),
    }
}

/// How finely the sound is read, in windows a second.
const WINDOWS: u32 = 50;

/// The pauses in the clip `source` describes, as timeline spans with
/// `padding` of each left beside the sound.
fn silences(
    source: &Value,
    threshold_db: f64,
    shortest: f64,
    padding: f64,
) -> Result<Vec<(f64, f64)>, String> {
    let path = source["path"].as_str().ok_or("the clip has no file")?;
    let stream = source["stream"].as_u64().map(|stream| stream as usize);
    let start = source["start"].as_f64().unwrap_or(0.0);
    let duration = source["duration"].as_f64().unwrap_or(0.0);
    let source_start = source["sourceStart"].as_f64().unwrap_or(0.0);
    let speed = source["speed"]
        .as_f64()
        .filter(|speed| *speed > 0.0)
        .unwrap_or(1.0);

    let peaks = concat_media::peaks::extract(std::path::Path::new(path), WINDOWS, stream)
        .map_err(|error| format!("the clip's sound could not be read: {error}"))?;
    let floor = 10f64.powf(threshold_db / 20.0) as f32;
    let step = 1.0 / f64::from(WINDOWS);
    // The clip's span of the file, window by window.
    let from = source_start;
    let to = source_start + duration * speed;
    let mut spans = Vec::new();
    let mut quiet_since: Option<f64> = None;
    let mut at = from;
    while at < to {
        let (low, high) = peaks.extremes(at as f32, (at + step) as f32);
        let loud = high.max(-low) >= floor;
        match (loud, quiet_since) {
            (false, None) => quiet_since = Some(at),
            (true, Some(since)) => {
                spans.push((since, at));
                quiet_since = None;
            }
            _ => {}
        }
        at += step;
    }
    if let Some(since) = quiet_since {
        spans.push((since, to));
    }
    Ok(spans
        .into_iter()
        .filter(|(a, b)| b - a >= shortest)
        .map(|(a, b)| {
            // Keep a breath beside the sound, but not at the clip's own ends.
            let a = if a <= from + step { a } else { a + padding };
            let b = if b >= to - step { b } else { b - padding };
            (start + (a - from) / speed, start + (b - from) / speed)
        })
        .filter(|(a, b)| b - a >= 0.05)
        .collect())
}
