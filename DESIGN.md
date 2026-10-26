# Library design (draft)

How the spike's building blocks become a library. Status: proposal; names are placeholders
(`videolab`). Decisions so far:

- **Analyze once, query many times.** Analysis produces a saved, versioned document; every
  feature (markdown, search, summaries, counting, redaction) reads from it.
- **Agents are first-class.** The CLI and MCP server are primary surfaces, not afterthoughts,
  and share one implementation with the browser API.
- **Browser-first compute.** Everything runs on WebGPU/wasm in a browser (or headless Chromium);
  no server, no Python, no native addons.

## Layers

```
@videolab/core     decode, ingest, detector, adaptive scheduler, scenes, VAD, ASR, model registry
@videolab/doc      VideoDoc schema (versioned JSON Schema), planner, load/save, refs
modules            markdown · search · redact · count · summarize
surfaces           JS API (worker by default) · CLI · MCP server · demo web app
```

- **core** is today's `src/runtime` + `src/track` + `src/scene.ts` + `src/transcript.ts`,
  hardened. Public, for power users; most callers never touch it.
- **doc** owns the data model and the planner. Small and dependency-free (loads in Node
  without a browser, so the CLI can read and query docs instantly).
- **modules** are separate entry points so a redaction app does not ship Whisper. Each declares
  the layers it needs; the planner computes missing ones.
- **Models are never bundled.** A registry fetches on first use, caches in the browser's
  private file storage (OPFS), reports sizes up front, and accepts self-hosted URLs
  (offline / air-gapped deployments).

## The VideoDoc

One JSON document per video: time-indexed **layers** on a common timeline (seconds, plus frame
indices where frames matter). Saved next to the video as `<video>.videolab.json`, with large
binary data (embeddings) in a sidecar `<video>.videolab.bin`.

```ts
interface VideoDoc {
  schema: 'videolab.doc/0';
  source: {
    name: string;
    /** Cheap identity: size + SHA-256 of the first and last MiB + duration. Lets a doc be
     *  matched to its video without hashing gigabytes in the browser. */
    fingerprint: string;
    duration: number; width: number; height: number; fps: number;
    video: { codec: string }; audio?: { codec: string; sampleRate: number; channels: number };
  };
  layers: Partial<{
    speech: Layer<{ t0: number; t1: number }>;                 // VAD spans
    transcript: Layer<{ t0: number; t1: number; text: string; words?: Word[] }>;
    scenes: Layer<{ t0: number; t1: number; f0: number; f1: number; score: number }>;
    keyframes: Layer<{ t: number; frame: number; scene: number; why: 'scene' | 'silence' | 'coverage' }>;
    captions: Layer<{ keyframe: number; text: string }>;
    faces: Layer<Track>;                                        // also: plates, objects
    objects: Layer<Track & { label: string }>;
    embeddings: Layer<{ ref: Ref; offset: number }>;           // vectors live in the sidecar
  }>;
}

interface Layer<T> {
  producer: { module: string; version: string; model?: string; config: object; configHash: string };
  /** What was analyzed: the whole video, or [t0, t1] ranges (partial / incremental analysis). */
  coverage: [number, number][];
  createdAt: string; ms: number; device: 'webgpu' | 'wasm';
  items: T[];
}

/** A tracked object: sampled boxes (frame, x, y, w, h, score); boxes between samples are
 *  linearly interpolated. Compact for long videos; per-frame boxes are derived on demand. */
interface Track { id: number; f0: number; f1: number; samples: number[][] }

/** Pointer to an item in another layer, e.g. "transcript/12", "keyframes/3". */
type Ref = `${string}/${number}`;
```

Rules:

- **Layers are append/replace, never edited in place.** Re-running with the same config is a
  no-op (`configHash` match); a different config replaces the layer.
- **Provenance is mandatory.** Every layer says what produced it, with what settings, on what
  device, and what it covered. Silent fallbacks (wasm instead of WebGPU, a smaller model) show up
  here and in the CLI summary.
- **Refs, not copies.** Captions point at keyframes, embeddings at transcript chunks or
  keyframes; search results resolve refs back to time ranges.
- **Schema is versioned and published** (JSON Schema + an `llms.txt` describing it), so agents
  can read docs without the library.

## Planner

Each producer declares `needs` and `produces`:

| Producer | Needs | Pass |
|---|---|---|
| speech (Silero VAD) | — | audio |
| transcript (Whisper) | speech | audio |
| scenes (GPU thumbnails) | — | frames |
| faces / objects (adaptive + refine) | (scenes) | frames |
| keyframes (transcript-guided) | scenes, speech | — (pure) |
| captions (small VLM) | keyframes | seek (selected frames only) |
| embeddings (text + image encoders) | transcript, keyframes, captions | seek + text |

`video.analyze(['faces', 'scenes', 'transcript'])` resolves dependencies, skips layers already
present with a matching config, and **groups every frame consumer into one decode pass** (decode
and transfer are a large share of browser cost). Audio is decoded separately (cheap). Selected-
frame consumers seek instead of decoding everything — the "cheap screening, expensive
confirmation" rule from the bodycam work, applied structurally.

`video.estimate(plan)` runs the same resolution without computing: models to download (MB),
passes, and expected time from a per-device calibration (device class × measured fps).

## Modules

| Module | Reads | Output | Notes |
|---|---|---|---|
| **markdown** | transcript, scenes, keyframes, captions? | Markdown (+ keyframe JPEGs) | Works without captions or an LLM; captions enrich it. The agent-facing "read the video" view. |
| **search** | embeddings (+ transcript text) | ranked hits with time ranges and refs | Hybrid: BM25 over transcript/captions + vector similarity. Query-time text encoding runs in the browser too. |
| **redact** | faces (+ plates) | encoded video + redaction log (JSON) | Needs output encoding (WebCodecs `VideoEncoder` + mediabunny output), audio passthrough. |
| **count** | objects / faces tracks | counts per label, per scene, over time | Mostly aggregation over tracks. |
| **summarize** | markdown | text | The only LLM-dependent module; takes a pluggable `llm(prompt) => text` (WebLLM, a bring-your-own-key API, or the calling agent). |

The core never depends on an LLM. Agents usually get markdown and summarize it themselves.

Markdown shape (sketch):

```markdown
# briefing.mp4 — 14:23, 1920×1080, 30 fps
<!-- videolab.doc/0 · transcript: whisper-small · scenes: 3 · faces: 2 tracks -->

## Scene 1 · 00:00–06:12
![00:00:04](briefing.keyframes/0001.jpg) *A man at a podium in front of two flags.*

**[00:04]** Good morning. Today we are announcing…
**[00:31]** …

> *(silent 02:10–02:45)* ![02:20](briefing.keyframes/0004.jpg) *Slides showing a bar chart.*
```

## Surfaces

**JS API.** One-line happy path; every knob reachable; heavy work in a worker by default.

```ts
import { open } from 'videolab';
import { toMarkdown } from 'videolab/markdown';

const video = await open(file);                      // File | URL | ReadableStream
const doc = await video.analyze(['transcript', 'scenes', 'faces'], { onProgress, signal });
await video.save(doc);                               // or doc.toJSON()
const md = toMarkdown(doc);
```

**CLI.** A Node process that drives a persistent headless Chromium (the existing Playwright
harness, generalized). Video files are served to the page by a local range-request server, never
uploaded. stdout: results (JSON / JSONL / Markdown); stderr: progress and human summaries.

```
videolab check                                   # WebGPU? adapter, browsers, model cache
videolab analyze talk.mp4 --layers transcript,scenes,faces   # writes talk.mp4.videolab.json
videolab estimate talk.mp4 --layers captions
videolab markdown talk.mp4 > talk.md             # analyzes missing layers first
videolab search talk.mp4 "budget cuts" --top 5
videolab redact bodycam.mp4 -o redacted.mp4 --log redactions.json
videolab models list|pull|clear
```

Commands that only read a doc (`markdown` from complete layers, `search` over cached
embeddings except the query vector) never need the GPU and return instantly.

**MCP server.** stdio, same implementation as the CLI, one tool per command (`analyze`,
`markdown`, `search`, `redact`, `estimate`, `check`). Long runs report MCP progress
notifications and return the doc path; the browser stays warm between calls (persistent
profile, models loaded once) so an agent's second query is fast.

**Demo web app.** Drag-and-drop a video, pick features, watch layers fill in, download the doc
and outputs. Also the regression harness's UI.

## Developer-experience rules

- **One line to start, no knob hidden.** Defaults are the measured best settings; the building
  blocks stay public.
- **Progress and cancellation everywhere** (`onProgress`, `AbortSignal`).
- **Explicit capability reporting.** `check()` before running; fallbacks recorded in provenance
  and printed; never silently slower or worse.
- **Errors say how to fix it.** "WebGPU unavailable in this browser; use Chrome/Edge 113+, or
  pass `device: 'wasm'` (≈6x slower)".
- **Deterministic and inspectable.** Same input + config → same doc; config echoed in every
  result; JSONL for machines, tables for people.
- **Self-describing for agents.** `--help` and MCP tool schemas generated from one definition;
  `llms.txt` covers the CLI, the doc schema, and the markdown format.

## Roadmap

1. **Doc + planner** wrapping today's blocks (speech, transcript, scenes, faces) into
   `analyze()`; save/load; CLI `analyze` / `check` on the existing harness.
2. **Keyframes + markdown module**: transcript-guided keyframe selection (scene starts, silent
   gaps, coverage), markdown without captions; then captions with a small VLM.
3. **MCP server** with a warm browser.
4. **Redact module**: output encoding, redaction log, plates.
5. **Search**: text + image embeddings, hybrid ranking.
6. Worker-by-default runtime, single shared GPU device (detector + Whisper), packaging/publishing.

## Open questions

- Package name and scope.
- Track storage: sampled + interpolated (compact) vs. per-frame (simpler consumers). Leaning
  sampled, with a `boxesAt(frame)` helper.
- Sidecar format for embeddings: raw float16 + offsets, or a small Parquet/Arrow file.
- Long-video incremental analysis: resume a partially covered layer after an interrupted run.
- Which captioning VLM fits in-browser budgets (Florence-2 / SmolVLM class via transformers.js).
