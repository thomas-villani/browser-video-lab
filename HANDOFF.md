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

- Branch `main`. Last commit `9a32bfb` "videodoc step 1" is **not pushed** (earlier ones are).
  Push only when the user asks.
- Roadmap step 1 is done: `src/doc/` (schema `videodoc/0`, planner, tracks, hashing,
  serialization), `src/analyze.ts` (`computeLayers`, browser `analyze()`), `cli/videodoc.ts`
  (`check`, `analyze`, `info`, `layers`; `--set layer.key=v`, `--dry-run`), `analyze.html` +
  `src/analyze-page.ts` (the compute page the CLI drives). 30 tests pass; `tsc --noEmit` clean.
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

## Next: roadmap step 2, keyframes + markdown

Agreed direction (DESIGN.md "Modules" and "Roadmap"): a `keyframes` layer chosen from
scenes + speech (scene starts, silent gaps, coverage so long shots get more than one), and a
`markdown` module that renders the doc as the agent-facing "read the video" view (sketch in
DESIGN.md), working without captions or an LLM. Keyframe JPEGs need a seek-based pass
(selected frames only, not a full decode). Then captions with a small in-browser VLM
(Florence-2 / SmolVLM class), then the MCP server with a warm browser (step 3).

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
  `*.videodoc.json` outputs are gitignored too.
