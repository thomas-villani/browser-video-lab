# Browser Video Lab — Spike Notes

Goal: decide whether a browser-only **video understanding runtime** is viable — WebCodecs
decode → WebGPU preprocessing → ONNX Runtime Web inference → tracking/indexing — with
redaction (faces/plates) as the first app and semantic video search close behind.

## Results (2026-10-25)

Hardware: Intel Iris Xe (gen-12lp, integrated), Edge headless, on AC power with the Windows
"Best performance" overlay. Video: 1080p H.264, 30 fps. Model: YuNet face detector (640×640).
Median of 3 runs, 300 frames each.

| Pipeline | fps | Notes |
|---|---:|---|
| Decode only (mediabunny + WebCodecs) | **980** | Hardware decode is never the bottleneck |
| WebGPU, naive (CPU preprocess, ORT returns CPU outputs) | 13 | ~60 ms in `run()` — mostly 12 serialized output readbacks |
| WebGPU, zero-copy preprocess, ORT CPU outputs | 13.5 | Preprocess 16 ms → 0.24 ms, but readback still dominates |
| WASM (CPU) baseline | 24 | Beats naive WebGPU! |
| WebGPU, zero-copy + **batched readback** | 48.5 | One staging buffer, one `mapAsync` |
| … + **2 frames in flight** | **61.6** | ~2× real time |
| … + detect every 3rd frame | 181 | 6× real time |
| … + detect every 10th frame | **557** | 18× real time — the tracker fills the gaps |

RF-DETR nano (384², transformer, 91 COCO classes): fp32 ~2.7 fps on Iris Xe — too heavy
for per-frame use on integrated GPUs; fine for keyframes. fp16 variant returned no detections
and logged EP fallback warnings — not investigated.

### Lessons

1. **GPU readback is the real enemy, not compute.** ORT maps each GPU output separately; YuNet
   has 12 outputs → 12 serialized round trips. Batching them into one map was worth ~4×.
   Next step: decode/threshold on GPU and read back only a compact candidate list.
2. **Naive WebGPU can lose to WASM.** The GPU only wins once data stays on it end to end.
3. **Zero-copy works**: `importExternalTexture(VideoFrame)` → WGSL compute → `GPUBuffer` →
   `ort.Tensor.fromGpuBuffer`, on ORT's own device (`await ort.env.webgpu.device`).
   Preprocessing 1080p→640² costs ~0.3 ms vs ~16–70 ms on CPU.
4. **Pipelining matters**: `session.run()` returns after submission when outputs stay on GPU,
   so a 2-deep ring of input/staging buffers overlaps CPU dispatch with GPU execution.
5. **Graph capture** (`enableGraphCapture`) made no measurable difference here.
6. **Sparse detection is the big lever.** Running the detector on a fraction of frames is
   10×+, which is exactly the "be clever" strategy below.
7. **Benchmark hygiene**: battery/power-saving mode changed results by 2–3×. Always run on AC
   with the performance overlay and report the median of several runs.
8. WGSL gotcha: `std` is a reserved word.

## Tracking & sparse detection (2026-10-25, session 2)

Test video: "Never Gonna Give You Up" (1080p H.264, 25 fps, 5,326 frames, 68 scene cuts ≈
one per 3 s, constant dancing/camera motion — near worst case for sparse detection).
Full-video YuNet pass: **61.8 fps** steady (87 s for 213 s of video). Ground truth =
YuNet run on every frame (score ≥ 0.6, 4,809 boxes). Note this is *pseudo*-GT: it measures
agreement with the dense detector, not with human-labeled faces.

Metric is redaction-oriented: per GT face, the fraction of its area covered by the best
predicted box (IoU is wrong for redaction: a bigger box that fully covers is a success).
"Miss" = < 50% covered (a leaked face). All rows below use 15% box padding.

| Strategy | Detector calls | Coverage | Miss rate |
|---|---:|---:|---:|
| Fixed stride 2, offline interp | 50% | 0.980 | 1.77% |
| Fixed stride 3, offline interp | 33% | 0.955 | 3.85% |
| Fixed stride 5, offline interp | 20% | 0.910 | 7.57% |
| Fixed stride 3, + scene-aware | 35% | 0.970 | 2.47% |
| Fixed stride 5, + scene-aware | 22% | 0.930 | 5.57% |
| **Adaptive bisection, max 4** | 38% | **0.994** | **0.58%** |
| **Adaptive bisection, max 8** | 30% | **0.986** | **1.16%** |
| Adaptive bisection, max 16 | 27% | 0.966 | 1.85% |

Lessons:

1. **Online ByteTrack is bad for coverage** (worse than just holding the last detections):
   new tracks need a confirming second hit and the Kalman estimate lags. It's an identity
   tool. For files, **offline interpolation** between keyframe detections is the right
   primitive; anchor on raw detections (not Kalman means), extrapolate track ends along
   their motion, and hold any unexplained detection so nothing is ever dropped.
2. **Scene cuts confirmed (design input #2)**: near-cut miss rates were 11–21%; resetting at
   cuts and always detecting each scene's first/last frame cut them to 1–6% for ~1.5% more
   detector calls. But most misses were elsewhere (motion), so cuts alone aren't enough.
3. **Adaptive divide-and-conquer wins (bodycam lesson)**: detect interval endpoints; if
   they agree (same faces, matched by IoU), interpolate; else bisect. Half the misses of
   fixed stride at fewer calls. Redaction-grade (~0.6% misses) at 38% of frames ≈ 2.7×
   fewer detector calls → ~160 fps effective on Iris Xe (6× real time).
4. **Score hysteresis didn't help** (~1% fewer calls): the subdivision floor (~23% calls even
   at max stride 32) is genuine motion, not detector flicker. Expect content-dependence:
   static/talking-head/bodycam-walking footage should be far sparser. Needs a second video.
5. Adaptive needs no ByteTrack at all — just endpoint matching. ByteTrack remains useful for
   stable IDs (e.g. "this person" in the redaction review UI, unblur-one-person workflows).

## Design input (from prior video-search work)

These shape the scheduler — the runtime's core idea is *spend compute where information is*:

1. **Transcript-guided sampling.** For search/Q&A the transcript usually wins. Where speech is
   present, visual sampling can be sparse; **silent stretches need denser image analysis**.
   → Whisper (or captions) runs first; its speech/no-speech timeline drives the visual budget.
2. **Cheap difference signal → scenes.** A cheap frame-difference model finds discontinuous
   jumps; treat each as a scene boundary and sample per scene (plus "important regions" where
   the difference is concentrated). → A GPU diff/histogram shader on decimated frames (we
   already have frames on GPU) feeds scene segmentation before any heavy model runs.
3. **Captions + fusion.** Production used a small VLM for detailed captions and an LLM to fuse
   transcript + captions into per-scene descriptions. In-browser options: small VLMs
   (e.g. SmolVLM / Florence-2 class via transformers.js) on scene keyframes only; fusion via a
   bring-your-own-key API or a local WebLLM model. Design the runtime to emit structured
   scene data so the fusion step is pluggable (browser, CLI, or the user's own agent).

### Divide and conquer (bodycam lessons)

Production workflow that worked (caveat: first-person moving bodycam footage, where scene
changes show up clearly as shifts in both semantic embeddings and overall frame content):

- **Coarse scan first**: sample n frames (typically 10–20) across the span.
- **Find low-coverage regions**: combine visual sampling with the transcript timeline; spans
  with little speech and few samples are "under-covered".
- **Recursively subdivide**: re-sample under-covered spans in smaller chunks, comparing
  neighboring embeddings; a large embedding distance marks a **semantic shift**, which
  becomes a dividing line. Recurse until shifts are localized or the budget is spent.
- **Cheap screening, expensive confirmation**: cheap models decide *where* to look; expensive
  models (VLM captions, heavy detectors) run only on the selected frames.
- **Transcript is king**: weight it heavily for search and understanding; the transcript
  alone typically reached **~80%** of full video-understanding performance.

→ Scheduler implication: the core abstraction is an **adaptive sampling budget over a
timeline**, not a fixed stride. Inputs: transcript coverage, cheap-signal deltas (pixel diff,
small-embedding distance). Output: a set of frames with a model tier per frame. Fixed-stride
detection + tracking (the redaction case) is the degenerate version of the same scheduler.
Redaction differs in one way: it needs *exhaustive* spatial coverage of faces on every frame,
so its sparsity comes from tracking/interpolation rather than from skipping content.

## LLM-friendly principles

Every tool here should be as usable by agents as by humans:

- **Headless first**: anything the UI does is reachable by URL params / CLI flags.
- **Machine-readable output**: JSONL on stdout, human summaries on stderr.
- **Self-describing**: config echoed in results; stable names; documented knobs in file headers.
- **Artifacts an LLM can read**: final outputs (scenes, tracks, redaction logs) as JSON +
  Markdown, not just video files.
- Plan an `llms.txt` and an MCP/CLI wrapper once there is a library API.

## Next steps

- [x] Real test footage (Rick Astley, via yt-dlp; gitignored).
- [x] TS ByteTrack port + offline/online sparse modes + coverage eval (`npm run eval:track`).
- [x] Adaptive bisection scheduler (`src/track/adaptive.ts`).
- [ ] Second test video of a different genre (talking head / bodycam-like walking) to
      measure content dependence of adaptive sampling.
- [ ] Wire the adaptive scheduler into the live browser pipeline (async `detect(frame)` via
      `VideoSampleSink.getSample`/`samplesAtTimestamps`; decode is cheap enough to seek).
- [ ] GPU frame-difference shader → scene cuts, replacing ffmpeg's scene score (design input #2);
      also usable as an extra "disagreement" trigger for bisection.
- [ ] GPU-side YuNet decode + compaction (score threshold → atomic append) to shrink readback.
- [ ] Human-labeled GT for a few hundred frames, to measure the detector itself, not just
      agreement with it.
- [ ] Plate model (YuNet LPD, prior-box decoding).
- [ ] Run the bench in a Worker (OffscreenCanvas) to keep the UI thread free.
- [ ] Re-test on a discrete GPU, a phone, and Firefox/Safari.
