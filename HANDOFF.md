# Hand-off (2026-10-27)

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

## State

- Branch `main`; see `git log` / `git status` for what is pushed. Push only when the user asks.
- Roadmap steps 1-3 are done except captions (status below): `src/doc/` (schema `videodoc/0`,
  planner, tracks, keyframes, markdown, hashing, serialization), `src/analyze.ts`
  (`computeLayers`, browser `analyze()`, `frameImages`), `cli/lib.ts` (the commands),
  `cli/videodoc.ts` (CLI: `check`, `analyze`, `markdown`, `info`, `layers`), `cli/mcp.ts` (MCP
  server, same commands), `analyze.html` + `src/analyze-page.ts` (the compute page they drive).
  43 tests pass (4 of them drive the MCP server over stdio); `tsc --noEmit` clean. The app:
  `app.html` + `src/app/`, served by `videodoc ui`.
- Layers today: `speech` (Silero VAD), `transcript` (Whisper base via transformers.js; needs
  speech), `scenes` (GPU thumbnail scene score), `faces` (YuNet 640 + adaptive bisection +
  native-resolution tile refinement; needs scenes). Defaults live in `PRODUCERS`
  (`src/doc/plan.ts`) and are the best measured settings.
- Human face labels: 100 of 309 frames done (`results/labels/faces.jsonl`, committed). The
  user may label more (rickroll, briefing, walk dense stretches) in `label.html` via
  `npm run dev`.

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

## Next: redaction (step 6)

Lead application decided 2026-10-27: search and read your videos privately; redaction follows
(output encoding, redaction log, plates, review UI). Smaller follow-ups for the app: hosted
build with a model registry, analysis in a worker, persistence for dropped files. Ideas noted
in DESIGN.md (not scheduled): audio-only files (lectures, podcasts) and transcript-driven
editing (filler words, cuts).

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
