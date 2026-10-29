# Library design (draft)

How the spike's building blocks become a library, **videodoc** (named after its data type;
decided 2026-10-26). Status: roadmap step 1 is built (`src/doc`, `src/analyze.ts`,
`cli/videodoc.ts`), step 2 up to captions (`keyframes` layer, `markdown` module and command),
and step 3 (`cli/mcp.ts`); the rest is proposal. Decisions so far:

- **Analyze once, query many times.** Analysis produces a saved, versioned document; every
  feature (markdown, search, summaries, counting, redaction) reads from it.
- **Agents are first-class.** The CLI and MCP server are primary surfaces, not afterthoughts,
  and share one implementation with the browser API.
- **Browser-first compute.** Everything runs on WebGPU/wasm in a browser (or headless Chromium);
  no server, no Python, no native addons.

## Layers

```
videodoc/core      decode, ingest, detector, adaptive scheduler, scenes, VAD, ASR, model registry
videodoc/doc       VideoDoc schema (versioned JSON Schema), planner, load/save, refs
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
indices where frames matter). Saved next to the video as `<video>.videodoc.json`, with large
binary data (embeddings) in a sidecar `<video>.videodoc.bin`. Types: `src/doc/schema.ts`.
Serialized with one layer item per line (`formatDoc`), so docs stay greppable and diffs small.

```ts
interface VideoDoc {
  schema: 'videodoc/0';
  source: {
    name: string;
    /** Cheap identity: "v0:<size>:<SHA-256 of the first and last MiB>". Lets a doc be matched
     *  to its video without decoding it or hashing gigabytes (Node and browser compute it alike). */
    fingerprint: string;
    size: number; duration: number; width: number; height: number; fps: number; frames: number;
    video: { codec: string }; audio?: { codec: string; sampleRate: number; channels: number };
  };
  layers: Partial<{
    speech: Layer<{ t0: number; t1: number }>;                 // VAD spans
    transcript: Layer<{ t0: number; t1: number; text: string; words?: Word[] }>;
    scenes: Layer<{ t0: number; t1: number; f0: number; f1: number; score: number }>;
    keyframes: Layer<{ t: number; frame: number; scene: number; why: 'scene' | 'silence' | 'coverage' }>;
    captions: Layer<{ keyframe: number; text: string }>;
    faces: Layer<Track>;
    plates: Layer<Track>;                                       // also later: objects
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

/** A tracked object: sampled boxes [frame, x, y, w, h]; boxes between samples are linearly
 *  interpolated (`boxesAt`), within max(1 px, 2%) of the boxes the pipeline produced. */
interface Track { id: number; f0: number; f1: number; samples: number[][] }

/** Pointer to an item in another layer, e.g. "transcript/12", "keyframes/3". */
type Ref = `${string}/${number}`;
```

Rules:

- **Layers are append/replace, never edited in place.** Re-running with the same config is a
  no-op (`configHash` match); a different config replaces the layer. A layer's hash covers its
  config and the hashes of the layers it was computed from, so changing the scene threshold
  also recomputes faces (its cuts came from scenes).
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
| keyframes (shots, silences, coverage) | scenes, speech | pure (Node, no decode) |
| captions (small VLM) | keyframes | seek (selected frames only) |
| embeddings (text + image encoders) | transcript, keyframes, captions | seek + text |

Built: `src/doc/plan.ts` (`PRODUCERS` holds each layer's needs, pass, defaults and help;
the CLI's `layers` command and `--help` are generated from it). Pure layers (`pass: 'pure'`) are
derived from other layers by `deriveLayers`, in Node or the browser alike.
`analyze(file, ['faces', 'scenes', 'transcript'])` resolves dependencies, skips layers already
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
| **redact** | faces, plates (+ scenes) | encoded video + redaction log (JSON) | Built: mediabunny `Conversion` with a per-frame `process` hook, audio copied. Regions = track boxes grown by a margin, held 0.3 s within the shot; drawn per strength level (one downscaled copy of the frame per level, one clipped draw). A review (`RedactPlan`, from the app) adds hand-drawn tracks and keep ids (per layer: `keep`, `keepPlates`); `cover` picks the layers. |
| **count** | objects / faces tracks | counts per label, per scene, over time | Mostly aggregation over tracks. |
| **summarize** | markdown | text | The only LLM-dependent module; takes a pluggable `llm(prompt) => text` (WebLLM, a bring-your-own-key API, or the calling agent). |

The core never depends on an LLM. Agents usually get markdown and summarize it themselves.

Markdown shape (sketch):

```markdown
# briefing.mp4 — 14:23, 1920×1080, 30 fps
<!-- videodoc/0 · transcript: whisper-small · scenes: 3 · faces: 2 tracks -->

## Scene 1 · 00:00–06:12
![00:00:04](briefing.keyframes/0001.jpg) *A man at a podium in front of two flags.*

**[00:04]** Good morning. Today we are announcing…
**[00:31]** …

> *(silent 02:10–02:45)* ![02:20](briefing.keyframes/0004.jpg) *Slides showing a bar chart.*
```

## Surfaces

**JS API.** One-line happy path; every knob reachable; heavy work in a worker by default.

```ts
import { open } from 'videodoc';
import { toMarkdown } from 'videodoc/markdown';

const video = await open(file);                      // File | URL | ReadableStream
const doc = await video.analyze(['transcript', 'scenes', 'faces'], { onProgress, signal });
await video.save(doc);                               // or doc.toJSON()
const md = toMarkdown(doc);
```

**CLI.** A Node process that drives a persistent headless Chromium (the existing Playwright
harness, generalized). Video files are served to the page by a local range-request server, never
uploaded. stdout: results (JSON / JSONL / Markdown); stderr: progress and human summaries.

```
videodoc check                                   # WebGPU? adapter, browsers, model cache
videodoc analyze talk.mp4 --layers transcript,scenes,faces   # writes talk.mp4.videodoc.json
videodoc info talk.mp4                           # summarize a saved doc, no browser
videodoc layers                                  # layers, needs, settings + defaults (JSON)
videodoc estimate talk.mp4 --layers captions
videodoc markdown talk.mp4 > talk.md             # analyzes missing layers first
videodoc search talk.mp4 "budget cuts" --top 5
videodoc redact bodycam.mp4 -o redacted.mp4 --log redactions.json
videodoc models list|pull|clear
```

Commands that only read a doc (`markdown` from complete layers, `search` over cached
embeddings except the query vector) never need the GPU and return instantly.

**MCP server.** stdio, same implementation as the CLI (`cli/lib.ts`), one tool per command
(built: `markdown`, `search`, `analyze`, `redact`, `info`, `layers`, `check`; later `estimate`).
Tool schemas are generated from `PRODUCERS`, like the CLI help. Long runs report MCP progress
notifications; cancellation closes the compute page, which stops the work. The browser and
compute page stay warm between calls, so an agent's second query skips browser start and page
load. Not yet: models loaded once (each call still creates its Whisper/YuNet sessions; files
come from the browser's HTTP cache) and a persistent browser profile.

**Demo web app** (built: `app.html`, `videodoc ui`). Open a folder (File System Access: docs
and keyframes read from and saved next to the videos, shared with the CLI and MCP server) or
drop videos (docs in memory, downloadable); analyze in the browser; read the Markdown view
rendered beside a player (timestamps, keyframes and hits seek); search across everything
loaded. Reading and searching work without WebGPU.

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

1. ~~**Doc + planner** wrapping today's blocks (speech, transcript, scenes, faces) into
   `analyze()`; save/load; CLI `analyze` / `check` on the existing harness.~~ Done
   2026-10-26 (plus `info`, `layers`, `--set`, `--dry-run`).
2. **Keyframes + markdown module**: ~~keyframe selection (scene starts, silent gaps,
   coverage), markdown without captions, seek-based keyframe JPEGs~~ (done 2026-10-27:
   `src/doc/keyframes.ts`, `src/doc/markdown.ts`, `videodoc markdown [--images]`); next,
   captions with a small VLM.
3. ~~**MCP server** with a warm browser.~~ Done 2026-10-27 (`cli/mcp.ts`, shared `cli/lib.ts`).
   Follow-up: keep model sessions alive in the warm page.
4. **Search across a collection**, transcript-first: ~~a folder of videos and their docs, keyword
   (BM25) ranking over transcripts, hits resolved to time ranges and keyframes; CLI command and
   MCP tool; `analyze <folder>`; `markdown --from/--to` to read around a hit~~ (built
   2026-10-27: `src/doc/search.ts`). Next (after step 9): fuzzy matching for ASR errors with
   Double Metaphone sound keys, indexed as a second, lower-weighted BM25 field so exact
   spellings rank first, and offered for names and rare words only (short words collide);
   then embeddings (text, then image) and hybrid ranking, fusing BM25 and vector ranks
   (reciprocal rank fusion) rather than scores. Keyword stays the backbone: in the user's
   experiments keyword beat vector alone, and only hybrid beat keyword.
5. **Demo web app: read and search**, statically hosted: ~~drop in videos, watch layers fill in,
   read the Markdown view, search across them. Nothing leaves the machine.~~ Built 2026-10-27
   (`app.html`, `src/app/`, `videodoc ui`). Not yet: hosted build with a model registry (models
   are served from `public/models`), analysis in a worker, OPFS persistence for dropped files.
   First-run guidance added the same day: welcome screen with the two ways in, toasts that say
   what is ready and what to analyze (with a time estimate), step-by-step analysis progress,
   search placeholder and suggested queries from the transcripts, a one-time seek tip.
6. **Redact module**: ~~output encoding, redaction log~~ built 2026-10-27 (`src/doc/redact.ts`
   regions and log, `src/redact.ts` decode/cover/encode, `videodoc redact`, MCP `redact`, the
   app's Blur faces). ~~Review UI~~ built 2026-10-28 (`src/app/review.ts`: regions over the
   player with a live preview of the cover, keep a face by clicking it, add a missed face by
   dragging a box, faces listed per shot with thumbnails; saved as `<video>.redact.json`, a
   `RedactPlan`, which `videodoc redact` and MCP `redact` also use). ~~Plates~~ built
   2026-10-28: a `plates` layer (YOLOv9-t from open-image-models, chosen over OpenCV's
   LPD-YuNet, NOTES.md), computed in the same decode as faces (`multiStream`: several detector
   heads, each with its own adaptive sampling), linked by predicted motion; redaction and the
   review cover both. ~~Audio redaction~~ built 2026-10-28: the transcript stores word times
   (`whisper-base_timestamped`, cross-attention alignment; `transcript.words`), `src/doc/words.ts`
   matches words and phrases (inflections, `*` wildcards, "exact" quotes, an opt-in
   strong-language list) and suggests the transcript's recurring names; matches are covered
   from `lead` before to `tail` after (word starts run late, NOTES.md) with a 1 kHz bleep or
   silence, crossfaded, in the same Conversion as the video (`cover: []` copies the video).
   Settings live in the review plan (`words`, `profanity`, `sound`, `lead`, `tail`,
   `keepWords`, `addedAudio`); the log lists every match. The app's review has a Spoken words
   section (word list, suggestions, matches to play or leave audible, a live preview that mutes
   the player and plays the tone). 8.1% of faces are still missed against human labels, some
   plates, and misspelled names, so a person must check the result.

   Not yet: matching names by sound (Whisper wrote "Okafor" as "Ackover"; today a person picks
   the transcript's spelling; Double Metaphone keys shared with search, step 4, with every
   sound-alike match listed for review; a phoneme recognizer such as wav2vec2-espeak only if
   real footage shows names garbled beyond what sound keys catch), drawing audio spans by hand in the app (`addedAudio` is in the
   plan and the CLI honors it), highlighting matches in the reading view, audio-only review in
   the app (the review needs faces), audio-only input files.
7. Captions with a small VLM, pulled in where they visibly improve search or the Markdown view.
   Together with **on-screen text (OCR)**: slides, screen recordings, whiteboards, documents
   held up to a camera. Plan in NOTES.md ("On-screen text: plan", 2026-10-28): a text detector
   on every keyframe screens, recognition runs only where it found text. Captions matter most
   for scenes without speech (bodycam stretches, b-roll, silent tutorials), where the
   transcript says nothing; caption those first.
8. Worker-by-default runtime, single shared GPU device (detector + Whisper), packaging/publishing.
   **Test suite first** (user direction, 2026-10-28: catch breakage as features land and
   dependencies change). Today `npm test` covers the pure TS (doc, planner, tracking, matching)
   and the CLI/MCP plumbing; nothing checks the browser half (decode, WebGPU models, encode,
   the app) or accuracy. Needed: browser tests in headless Edge on small fixtures (every layer
   computes, outputs are plausible, redaction round-trips, the app's main flows), and a
   regression suite with pinned metrics and tolerances on the measured footage (face recall
   against the hand labels, transcript accuracy, word-time error, cut accuracy, speed), so an
   onnxruntime-web, transformers.js or mediabunny upgrade that shifts results shows up.
9. **Pluggable models and pipelines** (user direction, 2026-10-28). Library: let people plug in
   their own models without forking. A frame layer is already a `ModelSpec` (input geometry,
   normalization, a `decode` to boxes) plus tracking settings, and `multiStream` runs any
   number of detector heads over one decode; what is missing is a registry, so a layer's
   producer can name a model that the caller registers (`registerModel(spec)`, or a URL to an
   ONNX file plus a decoder preset: YOLO end-to-end, YuNet, DETR-style), and custom track
   layers (e.g. `objects`, `logos`, `screens`) that redaction and the review can cover like
   faces and plates. The planner's config hash already includes the model name; it should
   include a hash of the model file too. App: a straightforward way to build a processing
   pipeline for a video or a folder (choose layers and models, see the order and the decode
   passes they share, the time estimate, save it as a preset, run it on many videos), built on
   the planner (`plan()` already resolves dependencies and passes) rather than a free-form
   graph editor. Presets double as CLI/MCP input (`--pipeline preset.json`).
10. **Speakers** (diarization): segmentation (pyannote segmentation-3.0, ONNX) + speaker
    embeddings (WeSpeaker / ECAPA class) + clustering, speakers assigned per word from the word
    times. Diarization errs (overlap, short turns, noisy bodycam audio, the wearer's voice
    sounding unlike everyone else's; one person split in two is the common failure), so design
    for correction: speakers are cluster ids, names a separate user-edited map that survives
    re-analysis; merge, split from a point, reassign a turn, saved like the review plan. Each
    speaker keeps a voice embedding, so naming someone once suggests that name for matching
    voices in other videos (suggest, never apply silently). Uses: speaker labels in the
    transcript, Markdown and caption exports, search filtered by speaker, redaction of
    everything one person says (bleep or voice disguise). A good first test of step 9's
    registry: two models and a new layer.
11. **Sound events and tone**: an audio tagger (YAMNet / AST / CLAP class) for gunshots, sirens,
    shouting, breaking glass, dogs, alarms, music, applause, as a time-indexed layer that search
    and Markdown show; CLAP-style audio-text embeddings also allow describing a sound in a
    query. Speech emotion only if cheap and honest: small models exist (wav2vec2 / emotion2vec
    class), but categorical emotions trained on acted speech generalize poorly; arousal
    (raised voices, agitation) is more robust and is what bodycam viewers want. Measure on
    real footage before showing a label.
12. **Visual search**: CLIP-class image-text embeddings (MobileCLIP / SigLIP-class, fast in
    the browser) on keyframes and sampled frames, joining step 4's hybrid ranking ("red truck",
    "handcuffs", "whiteboard diagram"). Storage in the embeddings sidecar.
13. **Translation**: Whisper's translate task (to English) or a small MT model for other
    targets, stored beside the original transcript with the same times; search either.
14. **Editing in the app**, so people do not bounce to another editor: trim, cut, split and
    join clips, export a clip around a search hit (redacted or not). Transcript-driven editing:
    delete words or sentences in the transcript and the video is cut there (word times exist),
    with fillers ("um", "uh", false starts) and long pauses highlighted to remove (needs a
    pass that keeps disfluencies; Whisper drops most "um"s). Cuts are an edit list over the
    source (non-destructive), exported as EDL/FCPXML or rendered: stream copy between
    keyframes, re-encoding only the GOPs at each cut (mediabunny), so a render stays fast.
    Auto-clipping for social media: candidate clips from long videos (a topic in the
    transcript, a sound event, a search hit, step 16's narration), cropped to vertical with the
    speaker kept in frame (face tracks exist), burned-in captions from the word times.
15. **Incidents with several cameras**: group the bodycam and dashcam videos of one event, align
    them by audio cross-correlation (they share sirens, voices, radio), play and search them on
    one timeline, redact and export together.
16. **Narration**: a small LLM (sub-2B, WebGPU via transformers.js or WebLLM) reads a region's
    transcript, captions, sound events and speakers and writes a summary, a title and labels
    (classification against a user's list) for clips, scenes or the whole video: chapters for
    YouTube descriptions, "what happens here" for long silent stretches. Measure faithfulness
    against hand summaries; agents over MCP already do this with bigger models, so it is for
    people without one.

Use cases beyond creators worth designing for (2026-10-28): journalists and open-source
investigators (verify footage, who said what across many clips), researchers coding
interviews and focus groups (speakers + search + labels), teachers clipping lectures, family
archives (visual search), meeting and community-stream recordings, accessibility captions with
speakers and sound events.

Ideas for later applications (2026-10-27, not scheduled):

- **Audio files** (lectures, podcasts, voice memos): the audio pass (VAD, Whisper), transcript,
  Markdown and search already work without frames; needs audio-only input in `analyze()`
  (no frames pass, no keyframes) and a player without video in the app.

**Lead application (decided 2026-10-27): search and read your videos privately.** It is more
generally useful than redaction and shows every core claim: browser-only and private, analyze
once / query many, agent-first, transcript-first. Redaction follows as the privacy showcase.
Test footage should match the targets (meetings, interviews, bodycam-like first-person video:
several speakers, long silent stretches), not only the briefing / music video / street walk.

**Audience (user direction, 2026-10-28): creators, not institutions.** Prosecutors and
agencies have storage and compliance requirements that rule out a homebrew tool. The likely
users are YouTube, Instagram and TikTok creators who do not want another subscription:
bodycam and dashcam channels (several videos per incident, faces, plates and names to redact,
gunshots and sirens to find), video podcasts and tutorial channels (has this been covered
before, and how; clips for shorts), and anyone who edits from the transcript.

## Open questions

- Package scope: `videodoc` was free on npm on 2026-10-26 (alternative considered: framebase).
- Track storage: sampled + interpolated, with `boxesAt(frame)` (built). Open: face tracks are
  linked from per-frame boxes by IoU only, so fast motion fragments them (music video: median
  track 13 frames); good enough for redaction and counting per frame, not yet for "who is
  this". Plate tracks use predicted motion and gap bridging (`maxShift`, `maxGap` in
  `linkTracks`); worth trying on faces too, measured against the hand labels.
- Plates: the 640 px detector input misses plates under about 40 px wide in 1080p; native
  tiles (as for faces) would help distant plates at a cost. Plates' adaptive sampling rarely
  agrees across 8 frames in traffic (plates move fast), so it detects most frames: about real
  time on a busy street. A motion-compensated agreement test would cut that.
- Speech gating: Silero VAD finds no speech in sung vocals (music video: 0 spans), so the
  transcript is empty there. Needs a lyrics mode (lower threshold or no gating).
- Sidecar format for embeddings: raw float16 + offsets, or a small Parquet/Arrow file.
- Long-video incremental analysis: resume a partially covered layer after an interrupted run.
- Which captioning VLM fits in-browser budgets (Florence-2 / SmolVLM class via transformers.js).
- Keyframe images are files next to the video (`<video>.keyframes/<frame>.jpg`), not doc data;
  the image size is not part of any cache key (delete the folder to re-extract). Captions will
  need the images in the browser anyway: extract and caption in one seek pass.
- Markdown on quick-cut video lists many shots without a frame (shorter than `minGap`, 2 s).
  Merging runs of such shots into one section may read better; wait for captions to judge.
