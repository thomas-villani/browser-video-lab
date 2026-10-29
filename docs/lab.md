# The lab

videodoc grew out of a spike measuring what a browser can do with video: WebCodecs decode,
zero-copy WebGPU preprocessing and ONNX Runtime Web inference. The benchmarks, evaluation
scripts and labeling tools behind every number in [NOTES.md](../NOTES.md) are still here; this
page is how to run them. None of it is needed to use videodoc.

## Extra models and media

Beyond the models in [getting started](getting-started.md), the benchmarks use:

| File | Source | License |
|---|---|---|
| `public/models/yunet_plate.onnx` | huggingface.co/opencv/license_plate_detection_yunet | see repo LICENSE |
| `public/models/rfdetr_nano.onnx`, `rfdetr_nano_fp16.onnx` | huggingface.co/onnx-community/rfdetr_nano-ONNX | Apache-2.0 |
| `public/models/yunet_face.onnx` | the original fixed-shape YuNet (input to `tools/dynamic_yunet.py`) | MIT |
| `public/media/pan1080.mp4` | any 1080p H.264 clip | — |

Test footage used in NOTES.md (meetings, a talk show, bodycam, a street walk, a music video)
was fetched with `yt-dlp` into `public/media/` and is not committed.

## Benchmark page

Interactive: `npm run dev`, then open the printed URL. Preset links are on the page; every
option is a URL query parameter (see the header of `src/main.ts`).

Headless (for agents and CI):

```bash
npm run bench -- --model yunet --path gpu --outloc batch --depth 2 --stride 1,10
```

- **stdout**: one JSON result per run (JSONL): config, adapter, fps, per-stage ms stats,
  first-frame detections.
- **stderr**: progress and a summary table.
- Comma-separated values form a matrix. All options are listed in the header of `scripts/bench.mjs`.
- `--screenshot` saves `results/<name>.png` showing first-frame detections.

For stable numbers, run on AC power with the performance power mode, and compare medians of
several runs.

## Single-pass adaptive pipeline

```bash
npm run pipeline -- --video /media/<clip>.mp4 --longside 640 --dump
```

Decodes the video once; every frame is retained on the GPU for one keyframe window and
thumbnailed for scene-cut scoring; the adaptive scheduler picks which frames the detector
runs on (`pipeline.html` in the browser, same URL params). stdout: one JSON summary per run
(calls %, input megapixels/frame, fps, cuts); `--dump` writes per-frame boxes, sampled
frames, and scene scores to `results/<clip>-pipeline-<name>.json`. Score it with:

```bash
node scripts/eval-track.ts --dump results/<clip>.dets.json --pred results/<clip>-pipeline-<name>.json --pads 0.15
node scripts/eval-cuts.ts --pred results/<clip>-pipeline-<name>.json --ref results/<clip>.cuts.txt
```

## Transcript (Whisper + VAD in the browser)

```bash
npm run transcribe -- --video /media/<clip>.mp4 [--seconds 120] [--model onnx-community/whisper-small]
```

Decodes the audio track (16 kHz mono), finds speech with Silero VAD, and transcribes only
the speech spans with Whisper on WebGPU (transformers.js). stdout: one JSON summary per run
(speed vs real time, words, speech coverage %, silent gaps); segments and speech spans go to
`results/<clip>-transcript-<name>.json`. Score against reference captions (e.g. YouTube
auto-captions via `yt-dlp --write-auto-subs --sub-format vtt --skip-download`):

```bash
node scripts/eval-transcript.ts --pred results/<clip>-transcript-<name>.json --ref results/ref/<clip>.en.vtt
```

## Face labels (human ground truth)

Model-proposed, human-verified labels, so detectors and strategies can be scored against
people rather than against the dense YuNet pass:

```bash
npm run label:select      # choose ~300 frames (seeded) and extract JPEGs to public/label/
npm run label:bootstrap   # RetinaFace-R50 at native resolution + flip (Python via uv, CPU)
npm run label:tasks       # merge RetinaFace + YuNet candidates, most uncertain frames first
npm run dev               # then open http://localhost:5173/label.html
npm run eval:labels       # score YuNet / RetinaFace / --pred pipeline runs against the labels
```

The review page saves every frame to `results/labels/faces.jsonl` (append-only; last line
per frame wins) through a dev-server endpoint, and resumes at the first unlabeled frame.
Label definition: any visible face, including profile and angled; not the back of a head;
faces under ~10 px tall are marked uncertain (ignored in scoring).

The labels, the frame list (`public/label/frames.json`) and the RetinaFace bootstrap
(`public/label/retina.json`) are committed; with the clips in place, `label:select` re-extracts
the same JPEGs and you can skip `label:bootstrap`.

## Tracking / sparse-detection evaluation

```bash
# 1. Dump every frame's detections (low threshold keeps weak boxes for ByteTrack/hysteresis)
npm run bench -- --model yunet --path gpu --outloc batch --depth 2 --frames 0 --dump --score 0.3 --video /media/<clip>.mp4
# 2. Optional: scene cuts via ffmpeg (writes results/<clip>.cuts.txt, one frame index per line)
npm run cuts -- public/media/<clip>.mp4
# 3. Evaluate strategies against the dense pass (JSONL on stdout, table on stderr)
npm run eval:track -- --dump results/<clip>-<name>.dets.json --cuts results/<clip>.cuts.txt --modes hold,offline,adaptive
npm test   # tracker unit tests (node:test)
```

## Tests

| Command | What | Time |
|---|---|---|
| `npm test` | Pure TS: doc, planner, tracking, search, word matching, redaction regions and log, CLI and MCP plumbing (`tests/*.test.ts`) | seconds |
| `npm run test:browser` | Headless Edge with WebGPU on the fixtures (`tests/browser/`): every layer, redaction, keyframe images, the app | ~4 min |

The browser tests need the models (`videodoc check`), ffmpeg, and the fixtures:
`tests/fixtures/tts-words.mp4` is synthetic speech (Windows TTS, with each word's true start in
`tts-words.tsv`; `tts.ps1` says how it was made) and committed; the others are 10-12 s cuts of the
measured footage, made by `npm run fixtures` into `tests/fixtures/local/` (gitignored; tests on a
missing fixture are skipped and say so).

- `analyze.test.ts`: speech and transcript against the TTS ground truth (words matched, word-start
  error, words a bleep covers whole); scenes, faces, plates, keyframes on a crowd, traffic and a
  music video (scenes tile the video, tracks stay in the frame, a re-run is cached).
- `redact.test.ts`: a blurred copy has the source's frames and length and few faces left to
  detect; bleeps carry a 1 kHz tone over every logged span and nowhere else, with the video
  stream copied byte for byte; keyframe JPEGs are linked from the Markdown.
- `app.test.ts`: add, analyze, read, search and seek from a hit; review a crowd and download the
  copy and its log; no page errors.

Figures (counts, word timing, faces left after blurring) are pinned in
`tests/browser/baseline.json` with tolerances; on one machine they repeat exactly, so each layer's
output is also hashed and a changed hash is reported (not failed). A figure outside its tolerance
fails with the baseline value; when the change is intended (a better model, new defaults), run
`UPDATE_BASELINE=1 npm run test:browser` and commit `baseline.json` with the change. Speeds are
reported against the baseline, never failed (power mode and load move them). Run the browser tests
before and after upgrading onnxruntime-web, transformers.js or mediabunny.

## Repository layout

- `app.html` + `src/app/` — the app (`library.ts`: folders and files, docs, keyframes and reviews on disk; `render.ts`: Markdown to HTML; `review.ts`: face review over the player; `main.ts`: UI)
- `cli/lib.ts` — the videodoc commands (shared); `cli/videodoc.ts` — CLI; `cli/mcp.ts` — MCP server (stdio, warm browser);
  `analyze.html` + `src/analyze-page.ts` — the compute page they drive
- `src/doc/` — the VideoDoc: schema, planner (`PRODUCERS`), keyframe selection, Markdown view, transcript search (BM25), redaction regions and log, track linking/compression, hashing, load/save (pure TS)
- `src/redact.ts` — `redact()`: decode, cover regions (blur, pixelate, fill), encode MP4 with mediabunny `Conversion` (audio copied)
- `src/analyze.ts` — `computeLayers` (one audio pass, one shared video decode), the browser `analyze()`, `frameImages` (seek-based keyframe JPEGs)
- `src/main.ts` — benchmark loop (decode → preprocess → infer → readback → decode outputs), in-flight ring
- `src/pipeline.ts` + `scripts/pipeline.mjs` — single-pass adaptive pipeline page and its headless runner
- `src/runtime/` — reusable building blocks:
  `detector.ts` (`Detector`: any frame or retained texture, optional crop and input size, pipelined),
  `source.ts` (`VideoSource`: indexed frames over mediabunny),
  `ingest.ts` (`FrameIngest`: GPU frame retention + thumbnails, one readback per window),
  `stream.ts` (`adaptiveStream`: decode once, cut-aware adaptive detection)
- `src/scene.ts` — ffmpeg-style scene score from thumbnails
- `src/runtime/audio.ts` (`decodeAudio`: mono PCM at 16 kHz), `vad.ts` (Silero VAD), `transcribe.ts`
  (`Transcriber`: Whisper via transformers.js, whole-file or per speech span)
- `src/transcript.ts` — VAD spans, Whisper chunk packing, speech coverage and silent gaps (pure TS)
- `src/transcribe-page.ts` + `scripts/transcribe.mjs` — transcript page and its headless runner
- `scripts/eval-transcript.ts` — WER against a reference WebVTT; `scripts/fetch-model.mjs` — local HF model download
- `src/preprocess.ts` — `GpuPreprocessor` (VideoFrame or GPUTexture, crop, any input size) and `CpuPreprocessor` baseline
- `src/readback.ts` — per-tensor vs batched single-map GPU readback
- `src/models.ts` — model specs + output decoders (YuNet, RF-DETR), NMS
- `scripts/bench.mjs` — headless matrix runner for the bench page; `scripts/lib/harness.mjs` — shared build/serve/browser harness
- `scripts/eval-cuts.ts` — scene-cut precision/recall vs reference cuts
- `tools/dynamic_yunet.py` — make YuNet's input size dynamic
- Labeling kit: `scripts/label-select.mjs`, `tools/bootstrap_faces.py`, `scripts/label-tasks.mjs`,
  `label.html` + `src/label.ts` (review page; label store in `vite.config.ts`), `scripts/eval-labels.ts`
- `src/track/` — dependency-free TS (runs in browser, worker, and Node):
  `bytetrack.ts` (ByteTrack port), `kalman.ts`, `assignment.ts` (Hungarian),
  `sparse.ts` (hold/online/offline sparse detection, scene-aware), `adaptive.ts`
  (divide-and-conquer scheduler), `coverage.ts` (redaction coverage metrics)
- `scripts/eval-track.ts` — strategy evaluation against a dense detection dump
- `tests/` — `node:test` unit tests
