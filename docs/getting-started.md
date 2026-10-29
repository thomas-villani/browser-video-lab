# Getting started

## Requirements

- **Node 24 or later.** The CLI, MCP server and library run TypeScript directly (Node's type
  stripping); there is no build step for them.
- **Chrome or Edge** with WebGPU, for analysis and redaction. The CLI drives it headless (Edge
  by default, `--browser chrome` for Chrome); the app runs in it. WebGPU is on by default on
  Windows, macOS and ChromeOS; on Linux it may need `--enable-unsafe-webgpu`. Reading and
  searching docs that already exist works in any browser and needs no GPU.
- **Models**, about 200 MB, fetched once into `public/models/` (below). They are never
  downloaded at run time and nothing is sent anywhere.
- Python is only needed once, to prepare the face detector (through `uv`, which fetches what it
  needs).

## Install

```bash
git clone https://github.com/thomas-villani/browser-video-lab.git
cd browser-video-lab
npm install            # also copies ONNX Runtime's wasm files into public/ort
```

## Models

| File | What it does | Source | License |
|---|---|---|---|
| `public/models/silero_vad.onnx` | speech detection | github.com/snakers4/silero-vad (`src/silero_vad/data/silero_vad.onnx`) | MIT |
| `public/models/onnx-community/whisper-base_timestamped/` | transcript, with word times | `node scripts/fetch-model.mjs onnx-community/whisper-base_timestamped encoder_model decoder_model_merged_q4` | Apache-2.0 (OpenAI Whisper: MIT) |
| `public/models/yunet_face_dyn.onnx` | faces | huggingface.co/opencv/face_detection_yunet (`face_detection_yunet_2023mar.onnx`), then the step below | MIT |
| `public/models/yolo_plate_t640.onnx` | license plates (redaction) | `curl -L -o public/models/yolo_plate_t640.onnx https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx` | MIT (open-image-models; trained with the GPL-3.0 YOLOv9 code, training data not documented) |

YuNet ships with a fixed 640x640 input. Relax it so it runs at any multiple of 32 (outputs are
identical at 640x640; 16:9 frames then run at 640x384, and small crops for crowds work):

```bash
uv run --with onnx --with onnxruntime python tools/dynamic_yunet.py public/models/yunet_face.onnx public/models/yunet_face_dyn.onnx
```

Only the layers you compute need their models: speech and transcript need Silero and Whisper,
faces needs YuNet, shots need nothing. For a more accurate (and about 3x slower) transcript,
fetch `onnx-community/whisper-small_timestamped` the same way and pass
`--set transcript.model=onnx-community/whisper-small_timestamped`. Docs made before 2026-10-28
used `whisper-base` without word times; their transcript is made again the next time a command
needs it (videodoc caches by settings).

Then check:

```bash
npm run videodoc -- check
```

It prints JSON describing the browser, the WebGPU adapter, which codecs decode, and which model
files are present. When something is missing it says what and how to fix it (exit code 1).

## The first video

In the app:

```bash
npm run ui
```

Open a folder (or drop videos on the page), click **Analyze**, then read and search. See
[the app](app.md).

From the command line:

```bash
npm run videodoc -- analyze talk.mp4
```

This writes `talk.mp4.videodoc.json` next to the video, with progress on stderr and a JSON
summary on stdout. Then, with no GPU and in well under a second:

```bash
npm run videodoc -- markdown talk.mp4 > talk.md
npm run videodoc -- search "budget" talk.mp4
npm run videodoc -- info talk.mp4
```

`npm run` runs commands from the repository root, but paths you give are taken relative to the
directory you ran it from. To call the CLI directly: `node /path/to/browser-video-lab/cli/videodoc.ts`.

## Files videodoc writes

Everything sits next to the video, so a folder of videos is the whole database and the app,
the CLI and the MCP server share it:

| File | Written by | What it is |
|---|---|---|
| `<video>.videodoc.json` | analyze (all surfaces) | the doc: every layer, its settings and provenance ([format](format.md)) |
| `<video>.keyframes/<frame>.jpg` | markdown `--images`, the app | keyframe images, named by frame number; delete the folder to re-extract at another size |
| `<video>.redact.json` | the app's review | which faces and plates to leave visible, regions added by hand, the cover style, words to bleep ([redaction](redaction.md)) |
| `<name>.redacted.mp4` | redact | the redacted copy |
| `<name>.redacted.mp4.redactions.json` | redact | the redaction log: settings, totals, every face and plate covered, every word bleeped |

Deleting any of them is safe: they are recomputed when needed.

## Troubleshooting

- **"WebGPU unavailable"**: use Chrome or Edge 113+; on a laptop, plug in (some power modes
  disable the discrete GPU or slow the integrated one 2–3x). `check` shows the adapter found.
- **A video won't analyze**: `check` lists the codecs this browser decodes. H.264 and VP9 decode
  everywhere; HEVC depends on the OS; some MKVs need remuxing to MP4.
- **Analysis is slower than the numbers in the README**: those are on AC power with Windows'
  "Best performance" mode; battery mode was 2–3x slower on the test laptop.
- **A doc from another version**: `parseDoc` refuses schemas it does not know and says so;
  re-run analyze with `--force`.
