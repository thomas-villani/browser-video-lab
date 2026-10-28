# Library API

videodoc is two layers of TypeScript:

- **`src/doc/`**: the data model and everything that reads it. Pure TypeScript with no
  dependencies and no browser APIs: parse and save docs, plan analysis, render Markdown, search,
  query face tracks, compute redaction regions, logs and review plans. Runs in Node, a worker or
  a page.
- **Browser compute** (`src/analyze.ts`, `src/redact.ts`): decode with WebCodecs (mediabunny),
  run models on WebGPU (ONNX Runtime Web, transformers.js), produce layers and redacted video.
  Needs a Chromium browser with WebGPU.

The package is not on npm yet: import from the repository. Node 24 runs the `.ts` files
directly; import `src/doc` with `.ts` extensions there. Browser code is bundled (the repo uses
Vite).

## Reading docs (Node or browser)

```ts
import { readFileSync } from 'node:fs';
import { parseDoc, docPathFor, summarize, toMarkdown, search, boxesAt } from './browser-video-lab/src/doc/index.ts';

const doc = parseDoc(readFileSync(docPathFor('talk.mp4'), 'utf8'), 'talk.mp4.videodoc.json');

summarize(doc);             // per layer: items, ms, device, model, configHash, figures
toMarkdown(doc);            // the "read the video" view
toMarkdown(doc, { from: 600, to: 900, image: (k) => `talk.mp4.keyframes/${String(k.frame).padStart(6, '0')}.jpg` });

const hits = search([doc], 'budget cuts', { top: 5 });
// [{ doc: 0, t0, t1, at, score, matched: ['budget', 'cut'], snippet: '…**budget** **cuts**…', keyframe?: 3, i0, i1, text }]

const faces = doc.layers.faces?.items ?? [];
boxesAt(faces, 1200);       // [{ id, box: [x, y, w, h] }] on frame 1200
```

### Docs

| Function | |
|---|---|
| `parseDoc(text, where?)` | Parse and check a saved doc; throws `DocError` saying what is wrong and how to fix it. |
| `formatDoc(doc)` | Serialize, one layer item per line. |
| `newDoc(source)` | An empty doc for a `SourceInfo`. |
| `withLayers(doc, layers)` | A new doc with layers added or replaced (never edited in place). |
| `summarize(doc)` | Per-layer figures, as `videodoc info` prints them. |
| `docPathFor(video)` | `<video>.videodoc.json`. |
| `fingerprint(size, read)` | The source fingerprint, given the size and a `read(start, end)` of bytes. |

Types: `VideoDoc`, `SourceInfo`, `Layer<T>`, `Producer`, `SpeechItem`, `TranscriptItem`,
`SceneItem`, `KeyframeItem`, `Track`, `Box`, `LayerName`, `LAYER_NAMES`. See [the format](format.md).

### Markdown

`toMarkdown(doc, opts?)` with `MarkdownOptions`:

| Option | |
|---|---|
| `image(k, index)` | Link for a keyframe's image (relative to where the Markdown will live); undefined leaves it out. |
| `from`, `to` | Render only this window, seconds. |
| `pause` | Start a new paragraph after a pause this long (default 2 s). |
| `silence` | Mark silent gaps at least this long (default: the keyframes layer's, else 4 s). |

Also `fmtTime(t, long?)` (`83.4` → `"01:23"`) and `maxFaces(tracks, f0, f1)`.

### Search

`search(docs, query, opts?)` ranks transcript passages across any number of docs (BM25 over
stemmed words, stopwords dropped; passages with the query's words in order score 1.5x). The index is built per call: about
25 ms for 27,000 words.

| Option | Default | |
|---|---|---|
| `top` | 10 | hits to return |
| `window` | 20 | passage length, seconds |
| `gap` | 2 | hits in one video closer than this (seconds) count as overlapping; the best is kept |
| `k1`, `b` | 1.2, 0.75 | BM25 parameters |

A `SearchHit` has `doc` (index into `docs`), `t0`/`t1` (the passage), `at` (where the first
matched word is said: seek there), `score`, `matched` (stems), `snippet` (matches in `**bold**`),
`keyframe` (index of the nearest keyframe) and `i0`/`i1` (transcript items spanned). Helpers:
`terms(text)`, `stem(word)`, `highlight(text, stems)`.

### Tracks

| Function | |
|---|---|
| `trackBox(track, frame)` | The interpolated box on a frame, or undefined outside the track. |
| `boxesAt(tracks, frame)` | All boxes on a frame, with track ids. |
| `linkTracks(frames, cuts?, opts?)` | Per-frame boxes to compact tracks (IoU matching, never across cuts). |

### Planning

```ts
import { plan, parseSettings, resolve, PRODUCERS } from './src/doc/index.ts';

resolve(['keyframes']);   // ['speech', 'scenes', 'keyframes']: dependencies, in order
const p = await plan(doc, ['transcript', 'faces'], parseSettings(['faces.every=4']));
p.layers;   // [{ name, status: 'cached' | 'compute', config, configHash }]
p.passes;   // { audio: ['transcript'], frames: ['faces'] }: what has to be decoded
```

`PRODUCERS` holds each layer's dependencies, pass, description, defaults and help text (the CLI
help and MCP schemas are generated from it). `deriveLayers(doc, plannedLayers)` computes the
pure layers (keyframes) from layers already in the doc. Errors are `PlanError`s that list the
valid names.

## Analyzing (browser)

```ts
import { analyze, frameImages } from './browser-video-lab/src/analyze';

const doc = await analyze(file, ['transcript', 'scenes', 'keyframes'], {
  doc: previous,                       // reuse layers from a saved doc of the same file
  settings: { transcript: { language: 'de' } },
  onProgress: ({ pass, stage, done, total }) => console.log(pass, stage, done, total),
  signal: controller.signal,
});
```

- `input` is a `File`/`Blob` or a URL (fetched with range requests, so large files are fine).
- Dependencies are added, cached layers reused (when `doc` is the same file, by fingerprint),
  audio decoded once and video decoded once for every frame layer. Pure layers are derived last.
- Progress reports `pass` (`probe`, `audio`, `frames`), a human `stage` and `done`/`total`.
- `force: true` recomputes the requested layers.

`frameImages(input, times, { maxSide = 768, quality = 0.85 })` returns JPEG blobs of the frames
at the given times, seeking to each (no full decode): how the keyframe images are made.

`computeLayers(input, plannedLayers, opts)` is the lower-level call the CLI uses after planning in
Node: it computes exactly the planned layers and returns them with the probe results.

### Hosting the browser code

The compute code loads models and runtimes by URL from the page's origin:

| URL | Contents |
|---|---|
| `/models/silero_vad.onnx`, `/models/yunet_face_dyn.onnx` | VAD and face models |
| `/models/onnx-community/whisper-*/` | Whisper (transformers.js `localModelPath`); never fetched from the Hub |
| `/ort/`, `/ort-hf/` | ONNX Runtime Web wasm files (copied by `npm install`) |

Serve the page **cross-origin isolated** (`Cross-Origin-Opener-Policy: same-origin`,
`Cross-Origin-Embedder-Policy: require-corp`) so wasm runs multithreaded; `vite.config.ts` and
the CLI's harness do this. A configurable model registry is on the roadmap.

## Redacting (browser)

```ts
import { redact } from './browser-video-lab/src/redact';
import { newPlan, handTrack, redactionLog, formatLog } from './browser-video-lab/src/doc/index.ts';

const plan = newPlan(doc, { style: 'blur', margin: 0.15, hold: 0.3, keep: [12] });
plan.added.push(handTrack(-1, 300, [100, 80, 60, 70], 360));   // a face the detector missed

const out = await (await fileHandle.createWritable());          // or omit for an in-memory MP4
const r = await redact(file, doc, out, { ...plan.options, added: plan.added, onProgress, signal });
// r: { frames, ms, buffer? }; r.frames must equal doc.source.frames
const log = formatLog(redactionLog(doc, plan.options, { name: 'out.mp4', frames: r.frames, ms: r.ms }, plan.added));
```

`redact(input, doc, stream?, opts)` checks the doc's fingerprint against the input, re-encodes
H.264 at 1.25x the source bitrate, copies the audio, and writes to `stream` (any
`WritableStream` of mediabunny chunks, e.g. a `FileSystemWritableFileStream`) or returns the MP4
in `buffer`. `coverer(canvas, style)` is the drawing function it uses, exported for previews.

The regions themselves are pure and can be computed in Node:

| Function | |
|---|---|
| `cursorFor(doc, options, added?)` | A `RedactionCursor` for the doc's faces plus hand tracks. |
| `cursor.regions(frame)` / `cursor.at(frame)` | Regions on a frame (`{ id, box, held }`) / just the boxes. Fast in increasing frame order; going back restarts the sweep. |
| `redactionLog(doc, options, output?, added?)` | The log, totals computed by sweeping every frame. `formatLog` serializes it. |
| `newPlan`, `checkPlan`, `parsePlan`, `formatPlan`, `planPathFor` | The review plan (`<video>.redact.json`). |
| `handTrack`, `setBox`, `setEnd` | Build and edit hand-drawn tracks. |
| `REDACT_DEFAULTS`, `REDACT_STYLES`, `REDACTION_CAVEAT`, `parseKeep` | |

See [redaction](redaction.md) for how regions are chosen.

## Lower-level building blocks

`src/runtime/` holds the pieces analysis is made of, usable on their own: `VideoSource` (indexed
frames over mediabunny), `Detector` (any frame or GPU texture, crops, any input size, pipelined),
`FrameIngest` (GPU frame retention with thumbnails), `adaptiveStream` (decode once, cut-aware
adaptive detection), `decodeAudio`, `Vad`, `Transcriber`. `src/track/` has a ByteTrack port,
Kalman filter, Hungarian assignment and sparse-detection strategies, all dependency-free. These
are less settled than `src/doc`; [the lab](lab.md) shows them in use.
