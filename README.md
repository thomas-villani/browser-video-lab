# browser-video-lab

Spike for a browser-only video understanding runtime: WebCodecs decode → zero-copy WebGPU
preprocessing → ONNX Runtime Web inference, with no server. Findings and design notes are in
[NOTES.md](NOTES.md).

## Setup

```bash
npm install            # also copies ORT wasm files into public/ort
```

Models and media are not committed. Fetch them into `public/`:

| File | Source | License |
|---|---|---|
| `public/models/yunet_face.onnx` | huggingface.co/opencv/face_detection_yunet (`face_detection_yunet_2023mar.onnx`) | MIT |
| `public/models/yunet_plate.onnx` | huggingface.co/opencv/license_plate_detection_yunet | see repo LICENSE |
| `public/models/rfdetr_nano.onnx`, `rfdetr_nano_fp16.onnx` | huggingface.co/onnx-community/rfdetr_nano-ONNX | Apache-2.0 |
| `public/media/pan1080.mp4` | any 1080p H.264 clip | — |

Then relax YuNet's fixed 640x640 input so it runs at any multiple of 32 (outputs are
identical at 640x640; smaller inputs enable 640x384 for 16:9 frames and small ROI crops):

```bash
uv run --with onnx --with onnxruntime python tools/dynamic_yunet.py public/models/yunet_face.onnx public/models/yunet_face_dyn.onnx
```

## Run

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

### Single-pass adaptive pipeline

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

### Tracking / sparse-detection evaluation

```bash
# 1. Dump every frame's detections (low threshold keeps weak boxes for ByteTrack/hysteresis)
npm run bench -- --model yunet --path gpu --outloc batch --depth 2 --frames 0 --dump --score 0.3 --video /media/<clip>.mp4
# 2. Optional: scene cuts via ffmpeg (writes results/<clip>.cuts.txt, one frame index per line)
npm run cuts -- public/media/<clip>.mp4
# 3. Evaluate strategies against the dense pass (JSONL on stdout, table on stderr)
npm run eval:track -- --dump results/<clip>-<name>.dets.json --cuts results/<clip>.cuts.txt --modes hold,offline,adaptive
npm test   # tracker unit tests (node:test)
```

## Layout

- `src/main.ts` — benchmark loop (decode → preprocess → infer → readback → decode outputs), in-flight ring
- `src/pipeline.ts` + `scripts/pipeline.mjs` — single-pass adaptive pipeline page and its headless runner
- `src/runtime/` — reusable building blocks:
  `detector.ts` (`Detector`: any frame or retained texture, optional crop and input size, pipelined),
  `source.ts` (`VideoSource`: indexed frames over mediabunny),
  `ingest.ts` (`FrameIngest`: GPU frame retention + thumbnails, one readback per window),
  `stream.ts` (`adaptiveStream`: decode once, cut-aware adaptive detection)
- `src/scene.ts` — ffmpeg-style scene score from thumbnails
- `src/preprocess.ts` — `GpuPreprocessor` (VideoFrame or GPUTexture, crop, any input size) and `CpuPreprocessor` baseline
- `src/readback.ts` — per-tensor vs batched single-map GPU readback
- `src/models.ts` — model specs + output decoders (YuNet, RF-DETR), NMS
- `scripts/bench.mjs` — headless matrix runner for the bench page; `scripts/lib/harness.mjs` — shared build/serve/browser harness
- `scripts/eval-cuts.ts` — scene-cut precision/recall vs reference cuts
- `tools/dynamic_yunet.py` — make YuNet's input size dynamic
- `src/track/` — dependency-free TS (runs in browser, worker, and Node):
  `bytetrack.ts` (ByteTrack port), `kalman.ts`, `assignment.ts` (Hungarian),
  `sparse.ts` (hold/online/offline sparse detection, scene-aware), `adaptive.ts`
  (divide-and-conquer scheduler), `coverage.ts` (redaction coverage metrics)
- `scripts/eval-track.ts` — strategy evaluation against a dense detection dump
- `tests/` — `node:test` unit tests

## License

[MIT](LICENSE). Models and test media are not distributed with this repo; they keep their
own licenses (see the table under Setup).
