# The app

```bash
npm run ui          # or: node cli/videodoc.ts ui [--port 4317] [--no-open]
```

This serves the app on localhost and opens it in your default browser. Use Chrome or Edge:
analysis needs WebGPU. Reading and searching docs that already exist work in any browser.
Everything runs in the page; nothing is uploaded.

## Adding videos

- **Open folder** (Chrome and Edge): pick a folder; every video in it and its subfolders is
  listed. Docs the CLI or the app wrote before are loaded, so those videos are ready at once.
  New docs, keyframe images and face reviews are saved next to the videos. This is the same
  collection the CLI and the MCP server use.
- **Add videos**, or drop files anywhere on the page: works in any browser. Drop a video's
  `.videodoc.json` (and `.redact.json`) with it to reuse them. Docs then live in the tab;
  **Download doc** saves one.

Videos without a doc show how long analysis will take; **Analyze** (one video) or **Analyze all**
queues them. They are analyzed one at a time, with each step shown as it runs (Listening →
Transcribing → Finding shots). You can read and search the others meanwhile. **Cancel** takes a
queued video off the queue, or stops the running job within a few seconds; nothing half-done is
kept (a cancelled blurred copy is not saved, and a file created for it is removed).

**What to analyze** (top right) chooses the layers for new analyses: speech, shots and keyframes
always; the transcript (on by default; needed for search); who speaks (speakers, off by default; about a tenth of the video's length more; paragraphs and search hits then say who speaks); faces (off by default; about doubles
the time); license plates (off by default; found anyway when you review or blur). **Add layers**
on an analyzed video computes the ones ticked and missing. Under them, a line says what runs (one
pass over the sound, one over the frames for shots and every detector, then key pictures) and
how long an hour of video takes with what is ticked.

### Presets

A [pipeline preset](pipelines.md) adds track layers on your own detectors (vehicles, logos,
screens) and settings to what the app analyzes; it is the same file the CLI takes with
`--pipeline`.

- **Load a preset…** in What to analyze: choose the preset's `.json` together with the model
  files it names (or drop them on the page). Model paths in the preset are matched to the chosen
  files by name.
- **In a folder**: a `videodoc.pipeline.json` at the top of an opened folder is used for every
  video in it; its model paths are read relative to the folder.
- The preset ticks the layers it lists and adds its track layers as checkboxes (slow models are
  marked); its settings are listed. × stops using it.
- **Save as preset** saves what is ticked, with the preset's models, track layers and settings:
  as `videodoc.pipeline.json` at the top of the opened folder, or as a download.
- Analyze all then runs it on every video not analyzed yet. A track layer's results are reused
  as long as its model file and settings are unchanged.

## Reading

Selecting a video shows the player beside its reading view: the same Markdown an agent gets,
with a picture for each shot and each silent stretch. Click a time, a picture or a shot heading
to jump the video there; while it plays, the paragraph being said is highlighted. **Copy
Markdown** puts the text on the clipboard (to paste into a chat or notes).

## Searching

Type in the search box (or press `/`) to search what is said in every loaded video. Hits show the
video, the time, the passage with the words marked, and the nearest picture; click one to open
the video at the moment the words are said. When nothing matches, the app suggests queries from
the transcripts (names and places that come up).

## Review and redaction

**Review faces and plates** (on a video with both; on one without, it finds what is missing
first) turns the player into a review tool:

- Boxes over the video show what the copy will cover (plates in cyan), with the blur previewed
  live. **Preview** off shows the boxes only.
- **Cover** ticks which of the video's other track layers are covered besides faces: plates
  (on by default) and a preset's layers (off until ticked), each in its own color.
- **Click a box** to leave it visible (a face that may be shown, a sign taken for a plate); click
  again to cover it.
- **Drag a box** over anything the detector missed. Step forward and drag it again where it moved
  to, and the box follows. The selected box's bar has **Starts here**, **Ends here** and
  **Delete**.
- The panel lists faces, then plates, shot by shot, longest first, with thumbnails; click one to
  jump to it. Crowded shots list 24, then more on request.
- **Cover with** picks blur, pixelate or solid boxes.
- **Spoken words**, above the shots: list words and phrases to bleep, one per line (the names the
  transcript repeats are offered as buttons), tick **Strong language too**, and pick bleep or
  silence. Each match is listed with its time; click the time to hear it (the preview mutes the
  player over it and plays the tone) or **Leave audible**. A video transcribed before word times
  existed offers **Transcribe again with word times**.
- Keys: Space plays, ← → step a frame (Shift: 10), `[` `]` previous / next shot, Esc deselects,
  Delete removes a hand-drawn box.

The review is saved next to the video as `<video>.redact.json` (folder mode) and used by
**Blur faces and plates…** (**Blur and bleep…** with words listed), which makes the copy: next to the video in folder mode, else where you choose
(or as a download), with its redaction log. When the copy was saved to a file, **Watch it** in the
toast that follows plays it. See [redaction](redaction.md) for what is covered and the file formats.

## Notes

- Analysis runs in the page itself (a worker is planned): keep the tab open, and a laptop
  plugged in, while it works.
- Dropped videos and their docs are gone when the tab closes, unless downloaded; folders keep
  everything on disk.
- Models are served by `videodoc ui` from `public/models`; there is no hosted version yet.
