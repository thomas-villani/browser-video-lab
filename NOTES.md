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

## Three-genre corroboration (2026-10-25, session 3)

The music video is near worst case for cuts, so two more clips (1080p H.264, 30 fps,
local-only via yt-dlp):

| Clip | Frames | Cuts | GT faces (≥0.6) | Character |
|---|---:|---:|---:|---|
| Rick Astley music video | 5,326 | 68 | 4,809 | fast cuts, dancing, ~1 face |
| Amsterdam street walk (POV) | 11,684 | 2 | 42,052 | continuous take, moving camera, ~3.6 faces/frame, constant entries/exits |
| AP White House briefing | 25,893 | 0 | 27,091 | fixed camera, podium speaker |

Full-video YuNet pass ran at 56–63 fps on all three. Results (15% padding, scene-aware):

| Clip | Fixed stride 5 | Fixed stride 10 | Adaptive max 8 | Adaptive max 32 |
|---|---|---|---|---|
| Music video | 22% calls, 5.6% miss | 12%, 15.8% | 30%, 1.2% | 24%, 5.0% |
| Street walk | 20%, **24.4%** | 10%, **49.6%** | 91%, 0.12% | 91%, 0.14% |
| Briefing | 20%, 0.6% | 10%, 1.7% | 14%, 0.06% | **5.3%, 0.11%** |

Lessons:

1. **Content dominates.** The same adaptive scheduler uses 5% of frames on a briefing and
   91% on a crowded street walk, while keeping misses ≤ 0.15% on both. Fixed strides fail
   unpredictably: fine on the briefing, catastrophic on the walk (24–50% misses).
2. **Adaptive is the right default for redaction**: it degrades *gracefully* toward
   dense detection when content demands it, instead of silently leaking faces.
3. **Crowds with a moving camera are the hard case.** With several faces per frame and
   constant entries/exits, interval endpoints almost never agree, so bisection runs down
   toward every frame. Loosening the matching (min IoU 0.1 + hysteresis 0.3) gets to ~73%
   of frames with 0.5% misses — a modest saving.
4. **Holding one-sided (entering/leaving) faces instead of bisecting** (`extendOneSided`)
   moves along the calls/misses trade-off curve rather than improving it (walk: 77% calls,
   0.65% miss — dominated by looser matching). Held boxes are static while pedestrians walk
   toward the camera. Kept as an option, off by default.
5. **Implication**: for crowd footage, savings must come from cheaper detector calls, not
   fewer. Ideas: ROI re-detection around known faces + frame borders (where entries
   happen), a smaller input size for "is anything new?" checks, full-frame detection only
   at intervals. That keeps the scheduler but changes the cost of each call.

## Runtime building blocks: single-pass pipeline (2026-10-26, session 4)

The adaptive scheduler now runs live in the browser (`npm run pipeline`, `src/runtime/`):
decode every frame once → GPU ingest (retain the frame for one keyframe window + 64 px
thumbnail) → per-window thumbnail readback → scene-cut scoring → bisection over retained
frames → detector only where needed. Laptop was **on battery**: fps values are lower bounds.

| Clip | Input | Calls | Miss (15% pad) | Cuts (ffmpeg) | Pipeline fps |
|---|---|---:|---:|---:|---:|
| Music video | 640×640 | 27.5% | 1.31% | 69 (68) | 104 |
| Music video | 640×384 | 27.5% | 1.31% | 69 | 142 |
| Street walk | 640×640 | 90.6% | 0.14% | 2 (2) | 44 |
| Street walk | 640×384 | 90.4% | 0.15% | 2 | 64 |
| Briefing | 640×640 | 8.3% | 0.06% | 0 (0) | — |
| Briefing | 640×384 | 8.3% | 0.16% | 0 | 213 |

Lessons:

1. **Live == offline.** Replaying the scheduler over the dense dump with the live run's cuts
   selects the same frames to within 0.1% (rickroll 1,466 shared of 1,467 / 1,469). The
   residue is the detector reading an 8-bit retained copy instead of the VideoFrame.
2. **GPU thumbnail cuts match ffmpeg.** ffmpeg's scene formula on area-averaged 64×36
   thumbnails needs threshold 0.15 (averaging lowers mafd vs ffmpeg's 0.3 on 320 px frames):
   F1 0.964 on the music video (66/68, same frames), 2/2 on the walk with no false cuts from
   camera motion, 0 on the briefing. Cost: one tiny dispatch per frame, one map per window.
3. **Dynamic input size is a free ~1.4x.** Relaxing YuNet's fixed 640×640 input
   (`tools/dynamic_yunet.py`, identical outputs at 640) lets 16:9 frames run at 640×384.
   Same calls and coverage on two clips; the briefing loses a little (0.06% → 0.16% misses),
   likely small distant faces.
4. **Decode once, detect anywhere in the window.** Retaining a window of frames on the GPU
   (17 × 1080p RGBA ≈ 140 MB) avoids re-decoding per bisection level, which would cost
   more than detection itself at ~5–10% call rates.
5. At low call rates the pipeline is now bounded by decode + ingest, not the model; next
   win there is overlapping the next window's decode with the current window's detection.

## Crowds: ROI tracking simulated, not adopted (2026-10-26, session 4)

`src/track/roi.ts`: full-frame detection only at window endpoints; faces that agree are
interpolated; the rest are followed with ROI crops (forward from the start, *backward* from
the end for entering faces — the window is retained on the GPU), with crops packed into
mosaic calls. Simulated with an **oracle** ROI detector (dense detections ≥60% inside the
crop), i.e. an upper bound. `eval-track --modes roi`, cost in model-input Mpx/frame.

| Walk (15% pad) | Mpx/frame | Miss | Area ratio |
|---|---:|---:|---:|
| Adaptive (default) | 0.226 | 0.10% | 1.94 |
| ROI, window 4, seeds ≥0.5 | 0.133 | 2.82% | 1.88 |
| ROI, window 4, seeds ≥0.3 | 0.159 | 0.59% | 2.80 |
| ROI, window 8, seeds ≥0.3 | 0.111 | 1.83% | 2.63 |

On the music video and briefing ROI tracking loses outright (adaptive is already sparse).

Lessons:

1. **Misses are score flicker, not lost tracks.** 76% of ROI-mode misses (window 4) were
   small faces (median 37 px tall) that score ≥0.6 mid-window but <0.5 at both endpoints.
   ROI size (expand 2–3) made no difference. Seeding from weak detections halves misses
   but over-blurs.
2. Even with a perfect ROI detector the saving is ~30% input pixels at ~6x the misses —
   the same frontier as loosening adaptive matching. Not worth the mosaic GPU path now.
3. **Crowds are already fast enough for files**: adaptive at 640×384 ran the walk at
   64 fps on battery (~2x real time). Graceful degradation to dense is acceptable.
4. **The metric's limit shows here.** GT is the dense detector itself; when it flickers on
   a small face, dense leaks that face too. Track gap-filling could beat "dense" against
   human labels — which makes human-labeled GT the prerequisite for further crowd work.

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
- [x] Second/third genre (street walk, press briefing): content dependence measured.
- [x] Cheaper detector calls for crowds: ROI tracking simulated (oracle); ~30% saving at 6x
      misses, not adopted. Crowd work now waits on human GT (score flicker, see above).
- [x] Live single-pass adaptive pipeline (`src/runtime/stream.ts`, `npm run pipeline`).
- [x] GPU thumbnail scene cuts (`src/runtime/ingest.ts`, `src/scene.ts`); F1 0.96 vs ffmpeg.
- [ ] Use the scene score as an extra bisection trigger (sub-threshold content change).
- [ ] Overlap next-window decode with current-window detection (double-buffered ingest).
- [ ] Transcript building block (Whisper via transformers.js/WebGPU) on the briefing clip.
- [ ] GPU-side YuNet decode + compaction (score threshold → atomic append) to shrink readback.
- [ ] **Human-labeled GT** for a few hundred frames (walk first), to measure the detector
      itself and track gap-filling, not just agreement with the dense pass.
- [ ] Plate model (YuNet LPD, prior-box decoding).
- [ ] Run the bench in a Worker (OffscreenCanvas) to keep the UI thread free.
- [ ] Re-test on a discrete GPU, a phone, and Firefox/Safari.
