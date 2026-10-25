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

### Tracking / sparse-detection evaluation

```bash
# 1. Dump every frame's detections (low threshold keeps weak boxes for ByteTrack/hysteresis)
npm run bench -- --model yunet --path gpu --outloc batch --depth 2 --frames 0 --dump --score 0.3 --video /media/<clip>.mp4
# 2. Optional: scene cuts (one frame index per line)
ffmpeg -i public/media/<clip>.mp4 -an -vf "scale=320:-2,select='gt(scene,0.3)',showinfo" -f null - 2>&1 \
  | grep -o "pts_time:[0-9.]*" | cut -d: -f2 | awk '{printf "%d\n", $1*<fps>+0.5}' > results/<clip>.cuts.txt
# 3. Evaluate strategies against the dense pass (JSONL on stdout, table on stderr)
npm run eval:track -- --dump results/<name>.dets.json --cuts results/<clip>.cuts.txt --modes hold,offline,adaptive
npm test   # tracker unit tests (node:test)
```

## Layout

- `src/main.ts` — benchmark loop (decode → preprocess → infer → readback → decode outputs), in-flight ring
- `src/preprocess.ts` — `GpuPreprocessor` (importExternalTexture + WGSL) and `CpuPreprocessor` baseline
- `src/readback.ts` — per-tensor vs batched single-map GPU readback
- `src/models.ts` — model specs + output decoders (YuNet, RF-DETR), NMS
- `scripts/bench.mjs` — Playwright (Edge/Chrome) headless matrix runner
- `src/track/` — dependency-free TS (runs in browser, worker, and Node):
  `bytetrack.ts` (ByteTrack port), `kalman.ts`, `assignment.ts` (Hungarian),
  `sparse.ts` (hold/online/offline sparse detection, scene-aware), `adaptive.ts`
  (divide-and-conquer scheduler), `coverage.ts` (redaction coverage metrics)
- `scripts/eval-track.ts` — strategy evaluation against a dense detection dump
- `tests/` — `node:test` unit tests
