# The VideoDoc format

A VideoDoc is one JSON file per video, `<video>.videodoc.json`, saved next to it. It holds
**layers**: lists of time-indexed items (speech spans, transcript segments, shots, keyframes,
face tracks), each with a record of what produced it. Analysis writes layers; everything else
reads them. The TypeScript types are in `src/doc/schema.ts`; this page describes schema
`videodoc/0`.

## Top level

```ts
interface VideoDoc {
  schema: 'videodoc/0';
  source: SourceInfo;
  layers: Partial<{ speech; transcript; scenes; keyframes; faces; plates }>;
}

interface SourceInfo {
  name: string;          // file name, for display
  fingerprint: string;   // "v0:<size>:<SHA-256 of the first and last MiB>"
  size: number;          // bytes
  duration: number;      // seconds
  width: number; height: number;
  fps: number;           // average; frame f is at about f / fps seconds
  frames: number;
  video: { codec: string };                                        // e.g. "avc1.640028"
  audio?: { codec: string; sampleRate: number; channels: number };
}
```

The **fingerprint** matches a doc to its video without decoding it or hashing gigabytes: the
file size plus a hash of its first and last mebibyte, computed the same way in Node and the
browser. A doc whose fingerprint does not match the file is ignored (analysis starts fresh)
and redaction refuses it.

Times are seconds (floats); frame indices count decoded frames in presentation order from 0.

## Layers

Every layer has the same envelope:

```ts
interface Layer<T> {
  producer: {
    module: string;          // the layer name
    version: string;         // videodoc version that wrote it
    model?: string;          // e.g. "onnx-community/whisper-base_timestamped"
    config: Record<string, number | string | boolean>;   // every setting, defaults included
    configHash: string;      // hash of config + the hashes of the layers it was computed from
  };
  coverage: [number, number][];   // what was analyzed, as [t0, t1] second ranges
  createdAt: string;              // ISO time
  ms: number;                     // wall time of the pass that produced it
  device: 'webgpu' | 'wasm' | 'cpu';   // 'cpu': derived from other layers, no model
  items: T[];
}
```

| Layer | Item | Needs | Pass | Produced by |
|---|---|---|---|---|
| `speech` | `{ t0, t1 }` | — | audio | Silero VAD v5 (wasm) |
| `transcript` | `{ t0, t1, text, words? }` (`words`: `[t0, t1, text][]`) | speech | audio | Whisper via transformers.js (WebGPU), speech spans only; word times from cross-attention |
| `scenes` | `{ t0, t1, f0, f1, score }` (a shot; frames inclusive) | — | frames | GPU thumbnails, ffmpeg-style scene score |
| `keyframes` | `{ t, frame, scene, why: 'scene' \| 'silence' \| 'coverage' }` | scenes, speech | pure | derived in Node or the browser, no decode |
| `faces` | `Track` (below) | scenes | frames | YuNet with adaptive sampling and native-resolution tiles |
| `plates` | `Track` (below) | scenes | frames | YOLOv9-t license plates (open-image-models), adaptive sampling, motion-linked |

**Passes.** Audio is decoded once for speech and transcript. Every frame layer shares one video
decode. Pure layers are computed from other layers without touching the video.

**Keyframes** are the frames worth looking at: one 0.5 s into every shot, one in the middle of
every silent gap of 4 s or more (nobody is saying what is on screen there), and fill-ins so no
stretch of a shot goes 30 s without one. `scene` indexes the scenes layer.

**Tracks.** A face (or plate) track covers frames `[f0, f1]` with sampled boxes; ids are per
layer:

```ts
interface Track { id: number; f0: number; f1: number; samples: number[][] }  // [frame, x, y, w, h]
```

Boxes are in source pixels; between samples they are linearly interpolated (`trackBox(t, frame)`
and `boxesAt(tracks, frame)` in `src/doc/tracks.ts`). Samples are kept only where interpolation
would drift more than max(1 px, 2% of the box) from what the detector found, so a long, steady
face costs a few samples. Tracks never cross a shot boundary. Face tracks are linked by box
overlap only, so they fragment under fast motion: fine for covering and counting faces, not for
identifying people across a video. Plate tracks are linked by where their motion predicts them
and bridge a few missed frames (interpolated), since plates are small and move fast. Ids are
stable for the same input and settings.

## Settings

Every setting has a measured default; override with `--set layer.key=value` (CLI), `set` (MCP)
or `settings` (library). `videodoc layers` prints this table as JSON.

| Setting | Default | Meaning |
|---|---|---|
| `speech.threshold` | 0.5 | speech probability that starts a span |
| `speech.minSilence` | 0.3 | seconds below threshold − 0.15 that end a span |
| `speech.minSpeech` | 0.25 | drop spans shorter than this, seconds |
| `speech.pad` | 0.2 | grow each span on both sides, seconds |
| `transcript.model` | `onnx-community/whisper-base_timestamped` | Whisper repo under `public/models` (`whisper-small_timestamped`: ~3x slower, fewer errors; word times need a `_timestamped` export) |
| `transcript.words` | `true` | word times (audio redaction needs them; about a third slower) |
| `transcript.encoder` / `.decoder` | `fp32` / `q4` | ONNX dtypes |
| `transcript.language` | `en` | spoken language code |
| `transcript.device` | `webgpu` | `webgpu` or `wasm` |
| `scenes.threshold` | 0.15 | scene score at or above which a frame starts a new shot |
| `keyframes.offset` | 0.5 | seconds into a shot for its frame (skips fades at the cut) |
| `keyframes.silence` | 4 | silent gaps at least this long get a frame, seconds |
| `keyframes.minGap` | 2 | drop candidates closer than this to a kept frame (shot starts win), seconds |
| `keyframes.maxGap` | 30 | longest stretch of a shot without a frame, seconds |
| `faces.longSide` | 640 | detector input long side for the full-frame pass, px |
| `faces.maxStride` | 16 | coarsest spacing between detected frames (sampling adapts to motion and cuts) |
| `faces.minIou` | 0.3 | overlap for two detections to count as the same face |
| `faces.minScore` | 0.5 | detection score threshold |
| `faces.refine` | true | also detect native-resolution tiles where small faces were seen |
| `faces.tile` | `640x384` | refinement tile size |
| `faces.every` | 2 | refine every Nth frame |
| `faces.trigger` | 0.3 | reduced-pass score that marks a tile for refinement |
| `faces.small` | 120 | only faces shorter than this (px) mark tiles; 0 = any |
| `plates.maxStride` | 8 | coarsest spacing between detected frames; must divide `faces.maxStride` when both run |
| `plates.minIou` | 0.2 | overlap for detections at both ends of a sampling interval to count as the same plate |
| `plates.minScore` | 0.4 | detection score threshold |
| `plates.lowScore` | 0.2 | a plate seen at `minScore` at one end still matches this weak a detection at the other |
| `plates.maxShift` | 1 | link a plate to a track whose motion puts it at most this many plate sizes away |
| `plates.maxGap` | 6 | frames a plate track may go undetected and continue (interpolated); 0 = none |

## Caching: analyze once

Layers are replaced, never edited. Before computing, the planner (`plan()` in `src/doc/plan.ts`)
hashes each requested layer's settings together with the hashes of the layers it depends on:

- same hash as the layer in the doc: **cached**, nothing to do;
- different hash (a setting changed, or an input layer did): **compute** and replace it.

So repeating a request is free, `--set scenes.threshold=0.2` recomputes scenes and also faces
and keyframes (they were computed from the old shots), and `--force` recomputes regardless. With
nothing to compute, the CLI and MCP server never start a browser.

**Provenance** is always there: the model, every setting, the device, how long it took. A
fallback (wasm instead of WebGPU, another model) shows in the doc and in the CLI summary.

## Serialization

`formatDoc` writes one layer item per line:

```
{"schema": "videodoc/0",
 "source": {"name":"bodycam.mp4","size":376418296,"fingerprint":"v0:376418296:093a…","duration":1055.417, …},
 "layers": {
  "speech": {"producer":{…},"coverage":[[0,1055.417]],"createdAt":"…","ms":10800,"device":"wasm",
    "items": [
      {"t0":0.61,"t1":9.33},
      {"t0":10.02,"t1":14.9},
```

Docs stay compact for long videos (17 minutes of bodycam with transcript and 1,238 face tracks: 0.3 MB), greppable,
and diffs stay small when a layer is replaced. `parseDoc` checks the schema, the source and
each layer's envelope, and says how to fix what is wrong. Keyframe images are not in the doc:
they are files in `<video>.keyframes/` named by frame number.

## The Markdown view

`toMarkdown(doc)` (CLI `markdown`) renders whatever layers the doc has:

- a title, a facts line (duration, size, fps, shots, speech %, words, face tracks) and a
  provenance comment;
- one `## Shot n · start–end` section per shot (with `· up to n faces` when faces were found;
  no heading numbers when the video is one shot);
- `**[mm:ss]**` paragraphs of speech, broken at pauses of 2 s or more and after a keyframe
  once the sentence ends;
- `*(no speech a–b)*` for silent gaps of 4 s or more;
- `![mm:ss](path)` for keyframes when images are supplied.

Times are `mm:ss`, or `h:mm:ss` for videos of an hour or more. With `from`/`to`, only shots and
paragraphs overlapping the window are rendered, under a note saying it is an excerpt.

## Other files

The review (`<video>.redact.json`, `videodoc-redact/0`) and the redaction log
(`<out>.redactions.json`, `videodoc-redactions/0`) are described in [redaction](redaction.md).
