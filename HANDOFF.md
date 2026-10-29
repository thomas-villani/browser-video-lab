# Hand-off (2026-10-28)

Where things stand, for picking the work up in a fresh session. The long-lived records are
**NOTES.md** (every measurement, lessons, checklist at the end), **DESIGN.md** (library design
and roadmap) and **README.md** (how to run everything). This file is the bridge: what was
being done, what is next, and the traps.

## The project in one paragraph

A free, browser-only video understanding runtime: WebCodecs (mediabunny) → WebGPU → ONNX
Runtime Web, with dependency-free TS schedulers/trackers. Spike started 2026-10-25; the library
is named **videodoc** (decided 2026-10-26): analyze a video once into a saved, versioned JSON
document of time-indexed layers (`<video>.videodoc.json`), then query it many times. Agents are
first-class users: CLI and (soon) MCP share one implementation with the browser API. Redaction
is the first app; markdown "read the video", search, counting and summaries follow.

The user built a large semantic video search system for a prosecutor's office; their
heuristics are design inputs: transcript first (≈80% of the value), divide-and-conquer
sampling, cheap models screen and expensive models confirm, scene changes from a cheap frame
difference, denser visual analysis where nobody speaks.

## State (2026-10-28)

- Branch `main`, everything committed and pushed (last: `e61e8ac`, browser test suite). Commit
  and push only when the user asks.
- Built: roadmap steps 1-6 (doc + planner, keyframes + markdown, MCP server, search, the app,
  redaction of faces, plates and spoken words with the review UI) and the browser test suite
  (step 8's first part). Captions (step 7) not started.
- Layers: `speech` (Silero VAD), `transcript` (whisper-base_timestamped, word times),
  `scenes`, `keyframes` (pure), `faces` (YuNet + adaptive sampling + tile refinement), `plates`
  (YOLOv9-t, same decode as faces). Defaults in `PRODUCERS` (`src/doc/plan.ts`).
- Tests: `npm test` 59 pure-TS tests (seconds); `npm run test:browser` 9 tests in headless Edge
  (~4 min; `npm run fixtures` first; docs/lab.md "Tests"). Run both before and after step 9.
- Audience (user, 2026-10-28): creators (bodycam/dashcam channels, podcasts, tutorials), not
  institutions; DESIGN.md "Audience" and roadmap steps 10-16.
- Power overlay still "Best performance" (restore: `powercfg /overlaysetactive
  00000000-0000-0000-0000-000000000000` when the user says).

## Step 9 starting points (pluggable models and a pipeline builder)

- Detector models: `ModelSpec` (`src/models.ts`: url, input geometry, normalization, `decode` to
  boxes), the `MODELS` table (`yunetFace`, `yoloPlate`, ...), looked up by name in
  `src/runtime/detector.ts`. Several heads share one decode via `multiStream`
  (`src/runtime/stream.ts`), used by `computeLayers` in `src/analyze.ts`.
- Layers: `PRODUCERS` (`src/doc/plan.ts`: defaults, help, `pass`, `needs`); `plan()` resolves
  dependencies and passes and hashes configs (the hash includes the model name, not the file).
- Transcript model: `transcript.model` (any `_timestamped` Whisper repo under `public/models`).
- Redaction covers `REDACT_KINDS` (faces, plates) in `src/doc/redact.ts`; custom track layers
  should join that.
- DESIGN.md step 9 has the design sketch (registry, decoder presets, custom track layers,
  presets as `--pipeline preset.json`, app builder on the planner, not a graph editor).

## Key numbers (details in NOTES.md)

- True recall vs labels (walk, 861 faces): YuNet 640 alone misses 50%; with tile refinement
  every 2nd frame **8.1% missed**, 65% precision, ~30 fps on Iris Xe. RetinaFace 1080p misses 17%.
- Pipeline speed (640×384 input, AC power): music video 201 fps, walk 72 fps, briefing 353 fps
  (without refinement). Transcript: whisper-base 5.3x real time, small 2.1x.
- videodoc runs: briefing speech+scenes 66 s; cached re-run 0.1 s (no browser); transcript
  added incrementally 161 s; music video all layers ~90 s after the refinement fix.

## Done: roadmap step 2 up to captions (2026-10-27)

- `keyframes` layer (`src/doc/keyframes.ts`): pure, `pass: 'pure'`, derived in Node by
  `deriveLayers` (`src/doc/plan.ts`); now in the CLI's default `analyze` layers.
- `markdown` module (`src/doc/markdown.ts`, `toMarkdown`) and `videodoc markdown <video>
  [--images] [--out f.md]`: analyzes missing layers, renders shots / timed paragraphs / silent
  gaps / face counts; `--images` seeks to keyframes (`frameImages` in `src/analyze.ts`) and
  writes `<video>.keyframes/<frame>.jpg` (gitignored).
- Measurements and findings: NOTES.md "videodoc step 2".

## Done: roadmap step 3, MCP server (2026-10-27)

- `cli/mcp.ts`: stdio server, tools `markdown` (can embed keyframes as image content),
  `analyze`, `info`, `layers`, `check`; schemas generated from `PRODUCERS`; progress
  notifications; cancellation closes the compute page. `npm run mcp`.
- `cli/lib.ts`: commands shared with the CLI; `warmBrowser` (one browser + page, serialized,
  idle close) vs `oneShotBrowser` (CLI). `scripts/lib/harness.mjs` gained `openBrowser`.
- Measurements: NOTES.md "videodoc step 3". Warm calls skip ~1.7 s of browser start.
- Not yet: model sessions kept alive between calls (each call re-creates Whisper/YuNet).

## Done: roadmap step 4, transcript search (2026-10-27)

- `src/doc/search.ts`: BM25 over overlapping ~20 s transcript windows, crude stemming,
  stopwords, phrase bonus, overlapping/touching hits collapsed; pure TS (web app can reuse).
- `videodoc search <query> <paths...>` and MCP `search`; `analyze <folder>`; `markdown --from/--to`
  (MCP `from`/`to`) to read around a hit. 42 tests pass.
- New footage `meeting.mp4`, `talkshow.mp4`, `bodycam.mp4` analyzed (docs next to them);
  measurements in NOTES.md "videodoc step 4".
- Next for search: fuzzy matching for ASR errors; then embeddings.

## Done: roadmap step 5, the app (2026-10-27)

- `app.html` + `src/app/` (`library.ts`, `render.ts`, `main.ts`, `style.css`), `videodoc ui`
  (`npm run ui`; harness gained `serve()`). Open folder (File System Access, shares docs with
  the CLI) or add/drop files; analyze in the page; reading view = toMarkdown rendered to HTML
  beside a player; search across loaded docs; hits seek to `at` and mark the terms.
- Silent videos now analyze (empty speech/transcript). 43 tests pass.
- **Open folder is not tested automatically**: check it by hand in Chrome/Edge.
- First-run guidance: welcome screen (Open a folder / Choose videos, three steps), toasts (what
  is ready, "N not analyzed yet, takes about X" with an Analyze button; done; failed), analysis
  progress as steps (Listening → Transcribing → Finding shots, `analysisSteps`/`stepOf` in
  `render.ts`), queue position, estimates (`estimate`), search placeholder and suggested
  queries (`suggestions`: names said mid-sentence), one-time seek tip, plain names in the
  "What to analyze" menu. 44 tests pass.

## Done: roadmap step 6 part 1, redaction (2026-10-27)

- `src/doc/redact.ts` (pure): `RedactionCursor` (regions per frame, swept in decode order;
  margin; hold within the shot; held boxes over a live face dropped; `keep` ids),
  `redactionLog` / `formatLog` (`videodoc-redactions/0`: settings, totals, every track).
- `src/redact.ts` (browser): `redact(input, doc, stream?)` via mediabunny `Conversion`
  (`process` hook, H.264 at 1.25x the source bitrate, audio copied). Styles blur / pixelate /
  fill, drawn per strength level, not per face (crowd: 3 -> 29 fps).
- `videodoc redact` (`--style --margin --hold --keep --out`), MCP `redact`, app **Blur faces…**
  (folder: saved next to the video; files: save dialog, or download). The page writes the
  output to Node with `PUT /@file/out?pos=` (harness), so large files stream.
- Not yet: plates, audio redaction.

## Done: roadmap step 6 part 2, the review UI (2026-10-28)

- `src/doc/redact.ts`: `RedactPlan` (`videodoc-redact/0`, saved as `<video>.redact.json`:
  options, keep ids, `added` hand tracks with negative ids), `checkPlan` (keep ids dropped with
  a warning when the faces layer's configHash changed), `handTrack` / `setBox` / `setEnd`
  (drawing a box again further along moves the carried samples, so the box follows the face).
  `RedactionCursor.regions(frame)` returns track ids; going back restarts the sweep.
- `src/app/review.ts`: canvas over the `<video>` (letterbox-aware), redrawn per presented
  frame (`requestVideoFrameCallback`), preview drawn with the same `coverer` as the output;
  own transport bar (native controls off while reviewing); click = keep/unkeep, drag = add or
  move a hand box; faces per shot, longest first, 24 then pages of 120; thumbnails cropped
  from the player as faces pass and from a hidden second `<video>` for list items in view.
- `videodoc redact` / MCP `redact` use the plan when present (`--no-plan` / `plan: false`).

## Done: README rewrite and docs/ (2026-10-28)

README.md now leads with motivation and use cases; user documentation lives in `docs/`
(getting started, app, CLI, MCP, format, library API, redaction, and `lab.md` for the
benchmarks and evaluation tools that used to fill the README). Keep it current with changes.
`check` no longer fails when the optional whisper-small is absent.

## Done: license plates (2026-10-28)

- Detector: YOLOv9-t-640 from open-image-models (`public/models/yolo_plate_t640.onnx`, NMS
  inside the model), chosen over OpenCV's LPD-YuNet on three clips (NOTES.md). `ModelSpec`
  gained `pad` (gray letterbox); readback skips empty outputs.
- `src/runtime/stream.ts`: `multiStream(source, heads, opts)` runs several detectors over one
  decode, each with its own adaptive sampling (`Head.maxStride` must divide the shared one);
  `adaptiveStream` is the one-head wrapper. Detector runs are serialized across all sessions:
  two ORT WebGPU sessions running at once hang (the first faces+plates run never left frame 0).
- `plates` layer (`PRODUCERS.plates`): hysteresis (`lowScore`), and `linkTracks` with
  `maxShift` (match by predicted motion, not IoU: passing plates move most of their width per
  frame) and `maxGap` (bridge missed frames). Dashcam: 363 -> 92 tracks, median 1 -> 6 frames.
- Redaction: `cover` (default faces and plates), `keepPlates`, `Region.kind`, log `detectors`
  per layer and `layer` per track, plan `plates` hash; CLI `--keep-plates`, `--cover`; MCP
  likewise. Review: tracks keyed `f:<id>` / `p:<id>`, plates in cyan, listed after faces.
  The app finds plates whenever it reviews or blurs.
- Trap: `check` now requires the plate model (redaction covers plates by default).

## Done: audio redaction (2026-10-28)

- Transcript default is now `onnx-community/whisper-base_timestamped` with `transcript.words`
  (word times from cross-attention; the plain onnx-community exports output logits only).
  `TranscriptItem.words: [t0, t1, text][]`; segments are rebuilt from words (sentence ends,
  pauses of 1 s). Word ends are clamped to their VAD chunk.
- `src/doc/words.ts`: `findWords` (inflections, `*`, "exact", phrases; hyphenated words split
  into tokens sharing times; a match inside a longer one dropped), `PROFANITY`, `suggestNames`.
- `src/doc/redact.ts`: options `words`, `profanity`, `sound`, `lead` 0.4, `tail` 0.2,
  `keepWords` (keys `<item>.<word>`); `audioSpans`, `audioCover`; `cover: []` allowed (faces
  required only when covered); log `audio` section and `detectors.transcript`; plan
  `transcript` hash (stale `keepWords` dropped), `addedAudio`, `faces` optional.
- `src/redact.ts`: `soundOver` audio `process` hook (1 kHz at -10 dBFS or silence, 8 ms fades);
  video copied when nothing visual is covered (`frames` 0). CLI `--words`, `--words-file`,
  `--profanity`, `--sound`, `--lead`, `--tail`, `--keep-words`, `--cover none`; MCP likewise.
- App review: Spoken words section (word box, name suggestions, matches to play or leave
  audible, preview mutes the player and plays the tone); "Transcribe again with word times" for
  old transcripts (`Entry.wantWords`).
- Trap: every doc made before today has a transcript with the old config hash, so the next
  command that wants `transcript` (markdown, search-prep, the app's analysis) transcribes again.
- Trap: the preview server serves `dist/`: with `--no-build`, new files in `public/` are missing.

## Done: browser test suite (2026-10-28)

- `npm run test:browser` (`tests/browser/`, `--test-concurrency=1`): analysis, redaction and app
  tests in headless Edge, ~4 min; `npm test` now runs only `tests/*.test.ts` (pure TS).
- Fixtures: `tests/fixtures/tts-words.mp4` (+ `.tsv` ground truth, `.txt` script, `tts.ps1`)
  committed; `npm run fixtures` (`scripts/fixtures.mjs`) cuts walk-crowd, dashcam-plates,
  rickroll-cuts from `public/media` into `tests/fixtures/local/` (gitignored).
- `tests/browser/baseline.json`: pinned figures and layer hashes; `UPDATE_BASELINE=1` rewrites.
  `near()` fails outside tolerance, `identical()` reports changed output, `speed()` reports only.
- Search hits' `at` now uses word times (half a second before the word); found by the app test.
- Trap: the browser tests copy fixtures to a fresh temp folder each run, so nothing is cached;
  docs next to fixtures are never read.

## Next

Pluggable models and a pipeline builder in the app (DESIGN.md step 9, user direction
2026-10-28); run `npm run test:browser` before and after (it reworks how models load).
Test suite follow-ups (DESIGN.md step 8): an accuracy tier on the full labeled footage, the MCP
server over a real browser. Audio follow-ups (DESIGN.md step 6): sound-alike name matching, drawing audio
spans in the app, highlights in the reading view, audio-only review. Lead application (decided 2026-10-27) stays search
and read. Smaller follow-ups for the app: hosted build with a model registry, analysis in a
worker, persistence for dropped files. Ideas noted in DESIGN.md (not scheduled): audio-only
files (lectures, podcasts) and transcript-driven editing (filler words, cuts).

## Later: captions

Captions with a small in-browser VLM (Florence-2 / SmolVLM class via transformers.js) over the
keyframe images, as a `captions` layer (`{ keyframe, text }`) whose pass is a seek pass;
extract images and caption in the same pass. `toMarkdown` then renders each caption in italics
after its image (DESIGN.md sketch). Needs AC power and the benchmark overlay for timings.

Open issues found in step 1 (also in DESIGN.md open questions / NOTES.md):
- Silero finds no speech in sung vocals → empty transcript on music; needs a lyrics mode.
- Face tracks are linked by IoU only and fragment under fast motion (median 13 frames on the
  music video). Fine for redaction/per-frame counts, not for "who is this".
- Refinement is on by default; only faces < 120 px trigger tiles (close-ups were being cut
  into duplicate faces). Batching a frame's tiles into one call is a backlog item.

## Conventions

- Commands: `npm test`, `npm run typecheck`, `npm run videodoc -- <cmd>`, `npm run pipeline`,
  `npm run transcribe`, `npm run eval:labels`, `npm run eval:refine`. Python only through
  `uv run ...` (not installed globally).
- Node 24 runs `.ts` directly (type stripping): Node-loaded TS (`src/doc`, `src/track`,
  `scripts/*.ts`, `cli/`, `tests/`) must import with explicit `.ts` extensions and use only
  erasable syntax (no enums or parameter properties). Browser code is bundled by Vite and
  imports without extensions.
- Headless tools: JSON/JSONL on stdout, progress and human tables on stderr, config echoed in
  results. Errors say how to fix the problem.
- American English; comments explain why; match the surrounding terse style.
- Benchmark only on AC power with the "Best performance" power overlay (battery skewed results
  2–3x). The overlay was switched on 2026-10-25; restore with
  `powercfg /overlaysetactive 00000000-0000-0000-0000-000000000000` when done.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Traps already hit

- **Git Bash rewrites `/media/x.mp4`** into `C:/Program Files/Git/media/x.mp4`. Prefix with
  `MSYS_NO_PATHCONV=1` (e.g. `MSYS_NO_PATHCONV=1 npm run pipeline -- --video /media/walk.mp4`).
- PowerShell: `git commit -F -` does not read a here-string from the pipeline; write the
  message to a temp file. `2>&1 >$null` swallows everything.
- `.gitignore`: a `!` negation must come after the rule it overrides (`*.jsonl` is ignored
  globally; `!results/labels/*.jsonl` is at the end).
- `npm run` changes cwd to the repo root; the CLI resolves user paths against `INIT_CWD`.
- The harness builds into `dist/` and copies `public/` (≈1 GB of models) on every build; the
  CLI rebuilds only when sources are newer than `dist/analyze.html`.
- Media and models are gitignored (`public/media/*.mp4`, `public/models/*.onnx`, Whisper dirs);
  `*.videodoc.json` outputs and `*.keyframes/` folders are gitignored too.
- MCP server: stdout is the protocol channel. Nothing in `cli/lib.ts` may print to stdout
  (`cli/mcp.ts` redirects `console.log` to stderr as a guard).
- The Bash tool collapses `\\` to `\` even inside quoted heredocs: write scripts containing
  backslash escapes with the Write tool.
- Python on Windows writes cp1252 and CRLF by default: in edit scripts use
  `read_text(encoding='utf-8')` and `write_bytes(s.encode('utf-8'))`.
- Node's type stripping rejects constructor parameter properties (`constructor(private o: X)`):
  declare the field and assign it.
- mediabunny `new Quality(n)` with a bare number is not a bitrate (the encoder failed with
  "Encoding error"); use `new Quality({ bitrate })`.
- `pan1080.mp4` is a pan over a face-detection demo image: every face has boxes and landmark
  dots painted on, so YuNet misses many. Do not judge detection or redaction quality on it.
