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

## Transcript building block: Whisper + VAD in the browser (2026-10-26, session 4)

`npm run transcribe`: mediabunny audio decode → 16 kHz mono (OfflineAudioContext resampler)
→ Silero VAD v5 (wasm, ~40x real time) → speech spans packed into ≤28 s chunks → Whisper via
transformers.js on WebGPU, one call per chunk. Reference: YouTube auto-captions (also ASR,
so WER is relative, not absolute). Briefing clip, 14.4 min, laptop **on battery**.

| Run | Speed | WER |
|---|---:|---:|
| whisper-base, whole file (no VAD), first 2 min | 2.2x | 85.6% |
| whisper-base + VAD, first 2 min | 3.8x | 3.7% |
| whisper-small + VAD, first 2 min | 1.9x | 2.8% |
| whisper-base (q4 decoder) + VAD, full | 3.7x | 9.4% |
| whisper-base (fp32 decoder) + VAD, full | 3.5x | 8.6% |
| whisper-small (q4 decoder) + VAD, full | 1.7x | 6.8% |

Lessons:

1. **Never show Whisper silence.** Without VAD, base looped "Good morning" ~100 times over
   the 6.5 s silent lead-in, swallowed the next sentence, and its timestamps stretched across
   silences (99% "speech"). VAD gating: 85.6% → 3.7% WER *and* faster (no wasted decoding).
2. **VAD is the speech-coverage signal** (design input #1): speech onset 6.46 s vs captions
   6.56 s; 93.8% speech and 3 silent gaps ≥2 s on the briefing. Whisper's own timestamps
   are not usable for this.
3. Base has a residual failure: a chunk returned in exact 3 s segments with truncated words
   (fp32 decoder too); small transcribes the same audio cleanly. A `dropRepeats` guard
   removes the remaining repetition loops (10.3% → 9.4%).
4. Quality/speed knob: base ≈3.6x real time at ~9% WER, small ≈1.7x at ~7%, on an
   integrated GPU on battery. Both are fine for files; the transcript of a 15-minute
   briefing costs 4–8 minutes, i.e. about as much as dense face detection.
5. Speaker changes in YouTube captions are `&gt;&gt;`; scorers must strip WebVTT entities.

## Re-timing on AC power (2026-10-26)

Iris Xe (gen-12lp), Edge headless, AC + "Best performance" overlay. Raw output in
`results/retime-2026-10-26/` (local only). Same call counts and coverage as before.

| Run | 640×640 input | 640×384 input | (battery, 640×384) |
|---|---:|---:|---:|
| Dense YuNet, every frame (bench) | 55.3 fps | 88.4 fps | — |
| Pipeline: music video (27.5% calls) | 146 fps | **201 fps** | 142 |
| Pipeline: street walk (90% calls) | 56 fps | **72 fps** | 64 |
| Pipeline: briefing (8.3% calls) | 295 fps | **353 fps** | 213 |

| Transcript, full briefing (863 s) | Speed | (battery) |
|---|---:|---:|
| VAD + whisper-base | **5.3x** real time | 3.7x |
| VAD + whisper-small | **2.1x** real time | 1.7x |

So on an integrated laptop GPU: face redaction of a 30 fps file runs at 2.4–12x real time
depending on content (crowds slowest), and a full transcript at 2–5x real time.

## Human face labels: first 100 frames (2026-10-26)

Kit: `label:select` → `label:bootstrap` (RetinaFace-R50, native 1080p + flip, CPU, 37 min) →
`label:tasks` (merge with YuNet dense) → `label.html` review → `eval:labels`. "Face" = any
visible face incl. profile; not the back of a head; <~10 px, mannequins, posters → uncertain
(ignored in scoring). Labels: `results/labels/faces.jsonl` (tracked).

First session: the 100 hardest random **walk** frames (861 faces; 74 added by hand that neither
model proposed; 246 proposals rejected). 54 min, median 21 s/frame. Miss = <50% of the face
covered.

| Source | Miss | Precision | <20 px | 20–40 | 40–80 | ≥80 |
|---|---:|---:|---:|---:|---:|---:|
| YuNet 640, score ≥ 0.5 (pipeline setting) | **50%** | 85% | 94% | 51% | 12% | 7.5% |
| YuNet 640, score ≥ 0.3 | 35% | 59% | 80% | 27% | 8% | 4.5% |
| RetinaFace 1080p, ≥ 0.5 | **17%** | 94% | 37% | 12% | 8% | 3% |
| Adaptive pipeline (YuNet, pad 0.15) | 51% | 85% | 94% | 52% | 13% | 7.5% |

- The earlier walk "0.15% misses" measured the scheduler against YuNet's own dense pass, not
  recall. The scheduler itself costs <1 point (50.1 → 50.8%); **the detector is the gap**.
- Downscaling 1080p → 640 wide shrinks a 30 px face to ~18 px, below YuNet's useful range.
- RetinaFace numbers are optimistic: it seeded most candidates, so its misses only count where
  the reviewer added the face by hand.

**Resolution sweep** (walk, adaptive pipeline, YuNet ≥ 0.5, pad 0.15, AC power):

| Detector input | Miss | Precision | <20 px | 20–40 | 40–80 | ≥80 | Calls | Speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 640×384 | 50.8% | 85% | 94% | 52% | 13% | 7.5% | 90% | 72 fps |
| 960×544 | 27.5% | 83% | 64% | 20% | 9% | 4.5% | 97% | 46 fps |
| 1280×736 | 15.7% | 80% | 31% | 14% | 6% | 4.5% | 98% | 27 fps |
| 1920×1088 (native) | **7.9%** | 72% | 11% | 9% | 4% | 6% | 99% | 12.5 fps |

- Resolution alone closes most of the gap: native-resolution YuNet beats RetinaFace at 1080p
  (no padding) on these labels, at ~0.4x real time on Iris Xe.
- Cost scales with input pixels, and the scheduler saves almost nothing here (more small faces
  found → more disagreement → more calls). Next: spend resolution only where it is needed
  (tiles on crowd regions, or full-res only on frames the low-res pass flags as crowded).

**Tile refinement** (`src/runtime/refine.ts`, `--refine`): the 640 pass runs as before; every
Nth frame, native-resolution 640×384 tiles (12-tile grid, 32 px overlap) are detected wherever
the 640 pass saw a face (score ≥ 0.3, below the scheduler's 0.5), and those boxes are merged into
the frames in between (never across a cut). Simulated first with `npm run eval:refine`
(oracle: the native run's boxes inside active tiles); the real runs land within ~0.5 points.

| Walk, vs labels | Miss | <20 px | 20–40 | Precision | Tiles/frame | Mpx/frame | Speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| 640 only | 50.8% | 94% | 52% | 85% | 0 | 0.23 | 72 fps |
| Native 1920 | 7.9% | 11% | 8.6% | 72% | 0 | 2.07 | 12.5 fps |
| 640 + tiles every frame | 8.8% | 13% | 9.4% | 70% | 3.65 | 1.14 | 19.2 fps |
| 640 + tiles every 2nd | **8.0%** | 12% | 7.8% | 68% | 1.83 | 0.68 | **30.3 fps** |
| 640 + tiles every 4th | 8.5% | 13% | 8.8% | 63% | 0.91 | 0.45 | **44.8 fps** |

- Every 2nd frame matches native recall at 2.4x the speed; every 4th costs 0.6 points of recall
  for 3.6x. Held boxes drift, so precision falls with N (extra blur, not missed faces).
- Trigger threshold matters: tiles chosen from scores ≥ 0.5 only (simulated) miss 13–14%.
- 960×544 tiles cost more than the whole frame (overlap); grow > 0 buys nothing for its cost.
- Not yet measured on the music video / briefing (no labels there yet); on sparse content
  few tiles activate, so the cost should approach the 640 pass.

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
- [x] Transcript building block: Silero VAD + Whisper (transformers.js/WebGPU), speech coverage.
- [x] Transcript-guided sampling: VAD gaps + scene cuts drive keyframe selection (`keyframes` layer).
- [ ] One GPU device for detector + Whisper (transformers.js bundles its own onnxruntime-web).
- [ ] Word-level timestamps (Whisper cross-attention models) for precise search hits.
- [ ] GPU-side YuNet decode + compaction (score threshold → atomic append) to shrink readback.
- [~] **Human-labeled GT**: 100/309 frames (walk random). Remaining: music video, briefing,
      and the dense walk stretches (track gap-filling).
- [x] Small-face recall in crowds: native tiles where the 640 pass sees faces, every 2nd
      frame — 50.8% → 8.0% misses at 30 fps on the walk.
- [ ] Batch a frame's tiles into one detector call (dynamic batch dim or a tile mosaic).
- [~] Library shape: see DESIGN.md (VideoDoc, planner, modules, CLI + MCP). Step 1 (doc + planner + CLI `analyze`/`check`) done 2026-10-26; step 2 keyframes + markdown done 2026-10-27; next: captions.
- [x] Plate model: YOLOv9-t (open-image-models) over LPD-YuNet; `plates` layer, redaction and review (2026-10-28).
- [ ] Run the bench in a Worker (OffscreenCanvas) to keep the UI thread free.
- [ ] Re-test on a discrete GPU, a phone, and Firefox/Safari.

## videodoc step 1: doc + planner + CLI (2026-10-26)

`npm run videodoc -- analyze <video>` end to end (Iris Xe, Edge headless, AC):

| Run | Result | Time |
|---|---|---|
| Briefing (14:23): speech + scenes | 113 spans (92.9% speech), 1 shot | 66 s (scenes-only pass ≈530 fps) |
| Briefing: same again | all cached, no browser | 0.1 s |
| Briefing: + transcript (cached speech reused) | 192 segments, 1,960 words | 161 s (5.4x real time) |
| Music video (3:33): all four layers | 0 speech spans, 70 shots, faces | 123 s |

Findings:

1. **Refinement cut close-ups into extra faces.** A 640×384 tile sees part of a 560×744 face
   as a smaller face that failed the IoU ≥ 0.5 dedupe: on the music video face-frames doubled
   (4,947 → 9,846) and the frame pass took 3x as long (34.5 → 107.6 s). Fixes: only faces
   shorter than 120 px trigger tiles (`small`, default 120; simulated on walk: 8.2% misses at
   any size vs 8.2% at h < 120, 11.0% at h < 40), and a tile box mostly inside a kept box is a
   duplicate. After: music video 67 s, 5,845 face-frames (the rest are plausibly real small
   background faces; no labels there). Walk vs labels, real run: **8.1% missed** (was 8.0%),
   precision 65.3% (was 68%), 29.8 fps (was 30.3).
2. **Silero VAD finds no speech in sung vocals** (music video: 0 spans), so the transcript is
   empty. Needs a lyrics mode (lower threshold or no gating).
3. **Tracks fragment under fast motion**: linking per-frame boxes by IoU gives a median track of
   13 frames on the music video without refinement. Fine for redaction and per-frame counts.

## videodoc step 2: keyframes + markdown (2026-10-27)

`keyframes` is a pure layer (scenes + speech, no decode): a frame 0.5 s into every shot, one in
the middle of every silent gap of 4 s or more, and evenly spaced fill-ins so no stretch of a shot
goes 30 s without one. Shots shorter than 2 s lose their frame when a kept frame is within 2 s;
longer shots always keep theirs. `videodoc markdown` renders shots, timed transcript paragraphs
(new paragraph after a 2 s pause, or at the first sentence end after a keyframe), silent gaps and
optionally the keyframe images.

Iris Xe, Edge headless, AC + "Best performance" overlay; 3 runs each (spread ≤ 0.42 s), keyframe
folder deleted before every image run. `imageMs` = seeking + JPEG encoding inside the page;
the rest of an image run is the browser and preview server start (~1.7 s):

| Video | Keyframes | Text only | `--images` 768 px | `imageMs` | JPEGs (768 / 1536 px) |
|---|---|---|---|---|---|
| Briefing (14:23, 1 shot, 30 fps) | 30 (1 shot, 1 silence, 28 fill-ins) | 0.53 s, no browser | 5.26 s | 3.56 s (119 ms/frame) | 1.8 / 4.7 MB |
| Music video (3:33, 70 shots, 25 fps) | 47 (46 shots, 1 fill-in) | 0.53 s, no browser | 5.33 s | 3.58 s (76 ms/frame) | 1.6 / 4.2 MB |

1536 px images cost the same time as 768 px (3.59 vs 3.56 s; 3.57 vs 3.58 s). Inside one page,
1 frame takes 42–58 ms, 5 take 364 ms, 47 take 3.4–3.8 s: linear in frames, no fixed cost.

Findings:

1. **A keyframe image costs a seek: decoding from the preceding key frame.** Key frames
   (ffprobe): briefing every 5.33 s (160 frames), so ~80 frames decoded per image on average
   = 119 ms, i.e. ~680 fps of decode; JPEG size and encoding are negligible. The music video's
   key frames are denser (median gap 2.08 s) and it is cheaper per image. Still far below a full
   decode: briefing's 30 images in 3.6 s vs ~49 s to decode its 25,893 frames at ~530 fps.
   Idea (not built): snap fill-in and silence frames to the nearest key frame within a second
   or two — they are arbitrary times anyway — making those seeks nearly free.
2. **Keyframes in speech must wait for a sentence end.** Breaking the paragraph at the keyframe
   time split Whisper segments mid-sentence ("…the new White House" / "press secretary.").
3. Quick-cut video (music video) yields many shots with no frame of their own: 24 of 70 shots
   are under 2 s and near a kept frame. They still get a heading (with face counts).

## videodoc step 3: MCP server (2026-10-27)

`cli/mcp.ts` (stdio, `@modelcontextprotocol/sdk` 1.32.2, low-level `Server` with JSON Schemas
generated from `PRODUCERS`) over `cli/lib.ts`, the command code now shared with the CLI. One
warm browser + compute page, runs serialized. End to end through an SDK client, real videos
(Iris Xe, Edge headless, AC + "Best performance" overlay; single runs):

| Call | Time | Notes |
|---|---|---|
| `check`, cold (build fresh, browser start, page load) | 1.03 s | |
| `check`, warm | 4 ms | |
| `markdown` music video, extract 47 keyframes, embed 4 | 3.67 s | CLI: 5.33 s (browser start ~1.7 s saved); 7 progress messages; 4 images 23–50 KB |
| `markdown` music video again, images on disk | 11 ms | no browser |
| `markdown` briefing, extract 30 keyframes | 3.56 s | CLI: 5.26 s |
| `analyze` briefing scenes `force`, cancelled at 5 s | 5.01 s | page closed, doc untouched; 11 progress messages before |
| `check` right after the cancel | 244 ms | new compute page |
| `analyze` music video scenes `force` (5,326 frames) | 7.75 s | 70 shots as before (~690 fps); 16 progress messages |
| `markdown` briefing, text, cached | 7 ms | 11.8 kB |

Findings:

1. **Warm calls cost what the work costs.** The fixed ~1.7 s of browser and server start is
   paid once per session (and again after 10 min idle).
2. **Cancel by closing the page.** Aborting work inside the page would need abort checks in
   every stage; closing the page ends decode, GPU and wasm work at once, and the next call
   opens a fresh page in ~0.25 s.
3. MCP progress values must increase while our stages restart their counts: the server sends a
   running step number and puts the stage and count in the message, at most ~2 per second plus
   every stage change.

## videodoc step 4: transcript search across a collection (2026-10-27)

New footage closer to the targets (yt-dlp, gitignored, 1080p H.264): `meeting.mp4` (River
Falls school board, 65:09, 60 fps), `talkshow.mp4` (Meet the Press, 47:41, 148 shots),
`bodycam.mp4` (traffic stop bodycam compilation with narration, 17:35). Default layers
(speech, transcript whisper-base, scenes, keyframes), Iris Xe, Edge headless, AC + overlay:

| Video | Wall | Speech | Transcript | Scenes pass | Shots | Keyframes |
|---|---|---|---|---|---|---|
| bodycam (17:35) | 311 s (3.4x real time) | 83% | 251 s, 2,792 words | 44 s | 44 | 57 (21 shot, 10 silence, 26 fill-in) |
| talkshow (47:41) | 636 s (4.5x) | 95% | 473 s, 8,375 words | 132 s | 148 | 186 (141, 2, 43) |
| meeting (65:09) | 1,324 s (3.0x) | 96% | 869 s, 11,726 words | 418 s (234k frames, ~560 fps) | 12 | 136 (8, 5, 123) |

Whisper-base runs at 4.2–6.0x real time here; it dominates analysis everywhere except the
60 fps meeting, where decoding every frame for scenes costs half as much again. Idea (not
built): scenes at a lower sample rate for high-fps sources (every 2nd frame at 60 fps).

`videodoc search` over `public/media` (5 transcripts, ~27k words): **24–28 ms** per query
(index built per query), ~0.6 s per CLI call including Node start and loading the docs. Eleven
realistic queries ("reading intervention", "promotion versus placement third grade", "tariffs
Michigan", "Alaska Senate lawsuit", "Hawaii", "audio redacted", "pledge of allegiance", ...):
the top hit was a relevant passage every time (read against the transcripts; several queries
have more than one right answer, e.g. three "reading intervention" passages in the meeting).

Findings:

1. **Keyword search is already useful on meetings and talk shows**: their transcripts are
   dense (95% speech) and the vocabulary is distinctive (names, programs, places).
2. **ASR errors are real but partly absorbed**: "turn signal" still found the right passage
   ("You didn't use your **turd** signal") through the other word and a nearby "turn". Whisper
   wrote "plagues of Allegres" for the Pledge of Allegiance in the meeting intro (the pledge
   itself, recited 10 s later, was found). Fuzzy/phonetic matching is the next step.
3. **Narration compilations mix sources**: the bodycam video has a narrator, redacted audio
   and several cameras. Search handles it; per-speaker or per-source labels would help later.
4. Whisper's segments sometimes start mid-sentence, so snippets do too; acceptable for now.

## videodoc step 5: the app (2026-10-27)

`app.html` + `src/app/` and `videodoc ui`. Driven headless with Playwright (Edge, Iris Xe, AC +
overlay), adding videos through the file input (same path as drag-and-drop):

| Step | Time | Notes |
|---|---|---|
| Add bodycam + briefing (with their docs) + pan1080 | 0.1 s | docs matched to videos by name |
| Reading view of bodycam, extract 57 keyframes in the page | 6.3 s | 72 paragraphs; images appear as they are ready |
| Search "turn signal" across loaded docs | < 0.3 s | 9 hits with thumbnails |
| Click a hit ("prescription drug prices") | | player seeks to 3:17, paragraph highlighted, terms marked |
| Analyze pan1080 (0:20, no audio) in the page | 1 s | scenes + keyframes; empty speech/transcript |

Findings:

1. **Silent videos failed analysis** ("no audio track in the file"): screen recordings and
   muted exports are common. Now a video without audio gets empty speech and transcript layers,
   and Whisper is not loaded when there is no speech at all (also saves seconds on music).
2. **Seek to the first match, not the passage start.** Passages are 20 s windows that may start
   a sentence or two early; hits now carry `at`, the start of the first transcript segment with
   a matched word (also in CLI/MCP results).
3. Not tested automatically: **Open folder** (File System Access needs a real user gesture and
   a permission prompt). Needs a manual check in Chrome/Edge.

### First-run guidance (2026-10-27)

Driven headless (Edge, file chooser from the welcome screen): bodycam + briefing with docs,
rickroll and pan1080 without. Toast: "2 videos ready to read and search. 2 videos not analyzed
yet. Analyzing takes about 1 min in this browser. [Analyze 2]"; placeholder "Search what's said
in 2 videos, e.g. Fairfield"; a query with no hits offers Fairfield, Trump, Indiana, President.
Clicking Analyze 2: rickroll shows "Listening… → Transcribing → Finding shots" in the list and
the reading view, pan1080 "queued · 1 video ahead"; rickroll done in 13 s (estimate: under a
minute), pan1080 after. Estimates use 0.22x the duration for the transcript plus 0.05x for
shots (0.2x with faces), from the step-4 table; they are for an integrated GPU.
Found while testing: `[hidden]` lost to `display: grid` (the empty player showed under the
welcome screen), and the previous video's "extracting keyframes…" status carried over.

## videodoc step 6, part 1: redaction (2026-10-27)

`videodoc redact` = faces layer (computed if missing) -> regions per frame (`RedactionCursor`:
box + 15% margin, held 0.3 s within the shot) -> mediabunny `Conversion` with a `process` hook
(H.264 at 1.25x the source bitrate, audio copied). Iris Xe, Edge headless, AC + overlay:

| Video | Frames | Face tracks | Frames with regions | Redaction | Output vs source |
|---|---|---|---|---|---|
| pan1080 (0:20, painted-on demo boxes, 250+ faces/frame) | 600 | 13,225 | 600 (max 626 regions) | 20.9 s, **29 fps** (first version: 3 fps) | — |
| rickroll (3:33, 25 fps, close-ups, cuts) | 5,326 | 369 | 5,022 | 68 s, **78 fps** | 72 vs 84 MB, AAC copied |
| walk (6:29, 30 fps, crowded street) | 11,684 | 25,278 | 11,613 | 281 s, **42 fps** | 285 vs 225 MB, AAC copied |
| bodycam (17:35, 30 fps, night, from scratch) | 31,661 | 1,238 | 20,500 | faces ~330 s + redaction 414 s (**77 fps**); **752 s wall, 1.4x real time** | 462 vs 376 MB |

Findings:

1. **Draw per strength level, not per face.** One blur filter per face made a crowd frame cost
   hundreds of filtered draws (3 fps). Now regions are grouped into levels by size (blur
   radius ~ a fifth of the face: 6 * 2^k px, done as a 6 px blur on a 2^k-downscaled copy;
   pixelate: 4 * 2^k px blocks) and each level is one small copy plus one clipped draw of a
   path holding all its rounded rectangles: 10x faster on the crowd; ordinary videos run at
   70-80 fps, about the decode + encode cost.
2. **Holds must stop at cuts.** Holding a close-up's box 0.3 s past a cut left a large blob in
   the next shot (music video). Holds now stay within the track's shot (scenes layer).
3. **Fragmented tracks multiply held boxes.** IoU-linked tracks break under motion, and holding
   every fragment covered faces twice (crowd: 383k regions for 151k face-frames). A held box
   that overlaps a live track's box by > 30% (of the smaller box) is dropped: 279k regions.
4. Bitrate: `QUALITY_HIGH` gave 5 Mb/s on a 3.2 Mb/s source (1.7x the size); 1.25x the source
   bitrate keeps files near the original size (walk and bodycam are larger: their sources are
   below what the encoder settles at for 1080p30 noise and motion).
5. The region shape is a rounded rectangle (corner radius 30% of the shorter side), which
   covers the detector box whenever the margin is 10% or more.
6. Visual checks (frames extracted with ffmpeg, which the product does not use): faces covered
   in close-ups, a nighttime car interior and an arrest, a crowded street (small background
   faces too); over-blur where the detector fires on hands. pan1080 is not a fair test: its
   faces carry painted boxes and landmark dots, and YuNet misses many of them.
7. The app (download fallback, headless): Blur faces on pan1080 -> queued as a "Blurring faces"
   step, done in 25 s, MP4 + log downloaded, no page errors. The save dialog and folder writes
   need a manual check in Chrome/Edge.

## videodoc step 6, part 2: the review UI (2026-10-28)

The app's **Review faces**: regions drawn over the player from the doc (no decode), the cover
previewed live with the output's drawing code, click to keep, drag to add a face by hand,
faces listed per shot with thumbnails; the choices saved as `<video>.redact.json`. Iris Xe,
Edge headless, AC + overlay:

| Video | Faces (tracks) | Open review | Playback with preview (4 s) | Plan change (rebuild + list) |
|---|---|---|---|---|
| rickroll (3:33, 25 fps) | 369 | 0.5 s | 99 frames presented, 0 dropped | < 0.1 s |
| walk (6:29, 30 fps, ~48 regions/frame) | 25,278 | 0.6 s | 119 presented, 0 dropped (same with preview off) | ~0.5 s |

Findings:

1. The first list rendered every face of a shot and hid the rest with CSS: 25k buttons on the
   crowd, ~1 s per click. Now 24 per shot, then pages of 120 on request.
2. Thumbnails cropped only as faces pass on screen left most of the list gray. A hidden second
   `<video>` now seeks to each listed face in view (IntersectionObserver): ~10 thumbnails/s on
   1080p, and the person's place in the main player is untouched. Thumbnails are not blurred:
   the reviewer has to recognize who it is to decide.
3. Hand boxes: a face drawn at one frame is held still for 2 s (cut at the shot end); drawing
   it again further along moves the samples carried from before, so the box follows. Fixing an
   earlier frame leaves later drawings alone.
4. End to end: review in the app (keep #366, one hand face, download fallback) -> log with
   `kept: 1, added: 1`; the same as a plan file next to the video -> `videodoc redact` used it
   (pixelate, 81 fps); output frames checked with ffmpeg: hand box pixelated, kept face visible.
5. Not covered headless: Open folder + review saved to disk, and the save dialog.

## On-screen text (OCR): plan (2026-10-28, not built, nothing measured yet)

Goal: the text on screen in slide decks, lectures, screen recordings, whiteboards, documents
held up to a bodycam, burned-in captions; under each shot in the Markdown view and in search
(lecture slides often use words the speaker never says verbatim).

**Deciding where to read: let the text detector do it, not a slide classifier.** OCR is two
stages: detection (where is text; PaddleOCR's DB detector is a few MB, milliseconds per image)
and recognition (what does it say; heavier). Running detection on every keyframe costs almost
nothing (50–200 per video) and answers "is there text here" per shot for every genre, so no
presentation heuristic gates it. Recognition runs only on keyframes (and boxes) with text:
cheap screening, expensive confirmation, as in the bodycam work.

**What the existing layers still contribute** (inferred, not classified):
- *Static shots* (scene score near zero within the shot, long shots, hard cuts between them)
  are slides or a fixed camera. Within a static text shot, a small localized change marks a
  build (a bullet appearing): add a keyframe there, so each build is read once. Needs a
  per-shot motion figure from the frames pass (the scene scores are computed already, not
  kept).
- *Speech* near-continuous (> 90%) plus *faces* absent or one small track fixed in a corner
  (picture-in-picture presenter) plus static text shots: label the video a presentation in the
  Markdown header. A label for readers; it does not gate OCR.
- Dedupe: consecutive shots with the same text (a slide revisited) are read once and refer back.

**Recognition model, to measure on the Iris Xe before choosing:**
- PaddleOCR PP-OCR recognizer (ONNX, ~10 MB, ORT Web): small and fast, good on clean
  printed text; line by line from the detector's boxes.
- Florence-2 (base, ~230M params; transformers.js, WebGPU): `<OCR>` / `<OCR_WITH_REGION>` and
  captions from one model, so it would also deliver roadmap item 7. Heavier per image; fine if it
  runs at about a second per keyframe. Leaning this way if the speed holds.

**Shape:** a `text` layer, `{ keyframe, lines: { text, box, score }[] }` per keyframe with text,
from a seek pass over selected frames (like keyframe images: extract and read in one pass).
Markdown: the lines under the shot's image, as a quote block. Search: index the text with the
transcript, hits marked as on-screen text.

To check first: detector recall on small text at 768 px keyframes (may need the full-resolution
frame, which the seek pass can provide); recognition on stylized slide fonts; Florence-2 speed
and memory alongside Whisper in one page.

## License plates: choosing a detector (2026-10-28)

Python reference runs (ONNX Runtime CPU, one thread pool, same frames, scripts not kept) on
three clips: the night bodycam (106 frames, no readable plates), a 70mai rear dashcam in
Bangalore (30 frames every 2 s, plates mostly pre-masked by the uploader), and a US LPR camera
demo with a parking-lot camera and an NVR screen (56 frames every 6 s, Oklahoma plates, many
tiny plates in thumbnails). Hits were checked by eye on contact sheets.

| Model | Size | Input | CPU ms/frame | Real plates found | False hits |
|---|---|---|---|---|---|
| LPD-YuNet 2023mar (OpenCV zoo), dynamic, 1920 long side, score ≥ 0.85 | 0.4 MB | any ×64 | ~850 | most large ones; missed several plates YOLO found (e.g. the dashcam motorcycle plates) | constant: giant quads (several hundred px to wider than the frame) on most frames, vest patches, police lettering, shop signs, slide titles |
| YOLOv9-t-640 (open-image-models, end-to-end NMS), score ≥ 0.4 | 7.8 MB | 640² letterbox | ~180 | everything YuNet found that was a plate, plus more (motorcycles, small thumbnails down to ~40×30 px) | none seen on any of the three clips |
| YOLOv9-s-608, score ≥ 0.4 | 28.6 MB | 608² | ~600 | a few more thumbnail plates than t-640 | one (a hardware photo on a title card) |

Findings:

1. **LPD-YuNet is not usable as is.** Trained on CCPD (Chinese parking-lot plates), it treats
   any plate-shaped text as a plate and produces huge low-quality quads; a size filter would
   remove the giants but not the text hits or the misses.
2. **YOLOv9-t-640 wins:** higher recall, no false hits seen, 4-5x faster on CPU, and a plain
   conv net (Conv/Sigmoid/Concat plus one NonMaxSuppression at the end) that ORT Web runs.
   Input RGB 0-1, 640×640 letterbox (pad 114); output `[n, 7]` = batch, x1, y1, x2, y2, class,
   score, already NMSed.
3. Both still miss plates (angled, motion-blurred, small in the distance), so plates need the
   same treatment as faces: dense enough sampling, tracking with hold, and the review UI to add
   what the detector missed.
4. **License:** open-image-models is MIT and the weights are published under that repo's
   releases with no separate terms. They were trained with the YOLOv9 codebase (GPL-3.0); the
   training data is not documented. Worth a note in docs; not a blocker for local use.
5. Not yet measured: plates in real US driving footage (the NYC drive download stalled), and
   the model's speed on WebGPU in the browser.

## License plates: the plates layer in the browser (2026-10-28)

Iris Xe, Edge headless, AC, "Best performance" overlay. `videodoc analyze --layers plates`
(YOLOv9-t-640, maxStride 8, minScore 0.4):

| Clip | Frames | Plates pass | Tracks | Median track | Note |
|---|---|---|---|---|---|
| Dashcam, Bangalore, 1:00, busy traffic | 1,801 | 59 s (1.0x real time) | 92 | 6 frames | plates alone, scenes cached |
| Same, faces + plates in one decode | 1,801 | 88 s | 720 faces, 92 plates | | faces alone: 36 s, identical tracks |
| LPR camera demo, 5:39, parking lot + NVR screens | 10,156 | 129 s (2.6x faster than real time) | 46 | 50 frames | |

Findings:

1. **IoU linking shattered plate tracks.** First run: 499 tracks, median 1 frame, though the
   single-frame tracks were nearly all real plates. A passing car's plate (45 px wide) moves
   25-35 px per frame, so consecutive boxes overlap with IoU under 0.2. Hysteresis
   (`lowScore` 0.2) and bridging 6 missed frames gave 363 tracks (median still 1); linking by
   distance from the velocity-predicted box (`maxShift` 1 plate size) gave **92 tracks, median
   6, 34 single-frame** (close motorcycles jumping around, a few boxes at the frame edge).
   The redaction hold (0.3 s) covers each sighting either way.
2. **Two ORT WebGPU sessions running at once hang.** Faces + plates never left frame 0 until
   `session.run` was serialized across all detectors (it had been per detector). Cost: the
   heads' runs no longer overlap, so faces + plates (88 s) is about faces (36 s) + plates (53 s).
3. **Plates sampling is dense in traffic**: plates move too far in 8 frames for the endpoint
   test (minIou 0.2) to agree, so bisection goes down to most frames there. On calm footage
   (parking lot) it stays sparse: 2.6x faster than real time.
4. Our letterbox (top-left, gray 114 padding) scores like the training letterbox (centered):
   same plates found on 86 frames, scores within a few hundredths.
5. Redacting the dashcam clip: 720 face + 92 plate tracks on 1,544 of 1,801 frames, 39 s
   (46 fps); plates visibly blurred in sampled output frames.

## Audio redaction: word times and bleeps (2026-10-28)

Iris Xe, Edge headless, AC, "Best performance" overlay.

**Word times exist only in the `_timestamped` exports.** `onnx-community/whisper-base`'s decoder
outputs logits only; `whisper-base_timestamped` (same size: 82 MB encoder fp32 + 124 MB decoder
q4) adds `cross_attentions.*`, which transformers.js aligns to words (`return_timestamps:
'word'`). Briefing, first 120 s: ASR 23.7 s with segments, 31.6 s with words (+34%), same text.
Bodycam 17:35 with words: 513 s (2.1x real time; segments-only base was 5.3x on the briefing).

**Ground truth.** Windows TTS (System.Speech) speaking a 138-word script (57 s) with names,
profanity and numbers; `SpeakProgress` gives each word's start. Its `AudioPosition` counts at
16 kHz while writing 22.05 kHz, so true time = reported x 16000/22050; checked against energy
onsets after pauses: within 0.02 s on 22 of 23. 117 words matched to Whisper's output:

| Model | Start error p50 | p95 | max | min | Covered whole, lead 0.4 / tail 0.2 | lead 0.5 / tail 0.3 |
|---|---|---|---|---|---|---|
| base_timestamped, q4 decoder | +0.17 s | +0.37 | +0.48 | +0.06 | 110/117 | 116/117 |
| base_timestamped, fp32 decoder | +0.17 | +0.37 | +0.48 | +0.06 | 111/117 | 116/117 |
| small_timestamped, q4 decoder (1.1x real time) | +0.18 | +0.37 | +0.45 | +0.08 | 116/118 | 117/118 |

Word starts run late by a fairly steady 0.1-0.4 s whatever the model or quantization (the
alignment method, not the model), so the lead is the longer pad. On real speech (briefing,
24 onsets after pauses, measured from room-noise energy so rougher): median +0.07, range
-0.46 to +0.50. Ends spill into the following pause ("Trump" 23.30-24.80).

**Faithfulness.** Whisper wrote "damn", "shit", "bullshit", "hell", "bitch", "son of a bitch"
as spoken. Names suffered: "Okafor" -> "Ackover", "Lindqvist" -> "Lindquist", "Moore" ->
"more"; "three o'clock" -> "3 o 'clock"; the phone number as "555 -213 -4987". Hence matching
on the transcript's spelling (and name suggestions from it), and hyphen-split tokens.

**End to end.** TTS clip, `--cover none --profanity --words "Ackover,Daniel Reyes,Lindquist,Patel"`:
8 matches, 7.48 s; decoded output, 1 kHz detector (10 ms frames, >80% of power at 1 kHz):
every target word 100% toned, 7.48 s toned in all; audio pass 0.8 s (video copied). Bodycam
17:35, faces + 4 names + profanity: 10 matches, 9.88 s; tone runs in the output equal the
log's merged spans to the 10 ms frame; 31,661 frames in 467 s (68 fps, with the audio rewrite);
audio and video lengths unchanged (1055.42 / 1055.37 s).

## Browser test suite (2026-10-28)

Iris Xe, Edge headless, AC, "Best performance" overlay. `npm run test:browser`: 9 tests in
3 min 34 s (analysis 80 s, app 65 s, redaction 60 s, plus builds and browser starts).

- **Results repeat exactly** on one machine: three runs gave identical counts (tracks, face-frames,
  cuts, keyframes, words, word times), so the baseline pins them with tolerances (5% tracks, 3%
  face-frames, 3 words) and hashes each layer to report any change at all.
- **Pinned on the fixtures:** TTS 118/138 words matched, word-start error median +0.176 s, 114
  words covered whole by lead 0.4 / tail 0.2 (the full-clip measurement above: 110/117 with a
  stricter alignment); walk-crowd 635 face tracks, 4,102 face-frames in 300 frames, 2 cuts;
  dashcam 41 plate tracks, 278 plate-frames; rickroll 6 cuts, 21 face tracks; bleeping 6 matches,
  6.5 s, tone in >95% of every span's 10 ms frames and none outside.
- **Faces left after blurring** (walk-crowd, YuNet on the blurred copy): 189 face-frames of 4,102
  (4.6%). Cropped samples were all blurred faces: the detector still fires on a blurred head
  (head shape, hair, skin tone), and boxes off the source's (82 face-frames) were the same blurred
  faces with a shifted or larger box, not faces the source pass missed. So re-detecting on the
  output is a regression guard, not a way to find missed faces.
- **The suite found a bug on its first run:** a search hit's `at` was its transcript segment's
  start (0.25 s for "Daniel Reyes", said at 4.48 s), though the docs said "where the first matched
  word is said". With word times it is now half a second before the word (4.08 s).

## videodoc step 9, part 1: models as data, pipeline presets, track layers (2026-10-28)

- **Refactor check.** The browser suite before and after: every pinned figure and every layer
  hash (faces, plates, scenes, keyframes, transcript) is unchanged, so the data-driven decoders
  (`src/doc/models.ts`) reproduce the hand-written YuNet and plate decoders exactly. Saved docs
  still recompute faces and plates once: their config hash now includes the model's
  description and file fingerprint.
- **A custom track layer** (`heads`: a copy of YuNet beside the preset, named by a relative
  path, served to the page by the CLI) on rickroll-cuts: 10 tracks, test 3.1 s; re-run cached.
- **RF-DETR nano as a `vehicles` layer** (`examples/pipelines/dashcam.json`, labels
  car,truck,bus,motorcycle, minScore 0.5) with plates on dashcam-plates (10 s, 1080p30, AC,
  Best performance): frame pass 102 s (plates 41 tracks, unchanged; vehicles 59 tracks, 1250
  samples). The same clip with faces and plates takes ~22 s for the whole test, so RF-DETR
  (~3 fps, 108 MB) costs about 8x real time on its own: fine as a demo of the plumbing, too
  slow as the default object detector. A small YOLO (the `yolo` decoder) is the next thing to
  try; Ultralytics models are AGPL, so a permissive one (YOLOX, D-FINE, RT-DETRv2-S) is preferable.
- **sirv (vite preview) and `Range: bytes=0-0`:** answered with the whole file (an end of 0 is
  read as "to the end"), so a size probe started a 108 MB download. Probes now ask for
  `bytes=0-1`.
- **Range reads poisoned ORT's model fetch in the app.** Fingerprinting `/models/yunet_face_dyn.onnx`
  with range requests just before ORT fetched it whole made session creation fail ("protobuf
  parsing failed"): the browser reused the cached partial response. Found by the app review
  test (the CLI fingerprints in Node, so its tests passed). Range reads now use
  `cache: 'no-store'`.

## videodoc step 10, part 1: the speakers layer (2026-10-29)

Diarization in the browser: pyannote segmentation-3.0 (MIT, 6 MB, wasm) over 10 s windows, a
WeSpeaker ResNet34 voice embedding (CC BY 4.0, 26 MB, WebGPU) per window and local speaker
from the fbank frames where it speaks alone, then pyannote 3.1's clustering in TS
(`src/audio/diarize.ts`: centroid-linkage AHC at 0.7046, min cluster size 12, constrained
assignment, top-count reconstruction). Windows with no VAD speech are skipped.

**Benchmark: AMI test meetings** (`scripts/fetch-ami.mjs`: Mix-Headset audio, pyannote's
AMI-diarization-setup "only_words" RTTM and UEM; 4 meetings, 92 min, 4 speakers each).
`scripts/eval-speakers.ts`, DER weighted by reference speech, no collar, overlap scored
(pyannote's setting; pyannote 3.1 reports 18.8% on all 16 AMI test meetings). AC power,
Best performance, Iris Xe (gen-12lp):

| step | DER | missed | false alarm | confusion | speed (x real time) |
|---|---|---|---|---|---|
| 1 s (IS1009a only) | 20.6 (IS1009a) | | | | 4.3 |
| **2 s (default)** | **21.5** | 11.9 | 3.3 | 6.3 | 7.5-9.7 |
| 3 s | 22.3 | 11.6 | 3.8 | 6.9 | 10.6-12.1 |
| 5 s | 23.3 | 10.0 | 5.2 | 8.1 | 16.8-20.5 |

Per meeting at step 2: ES2004a 19.9 (5 speakers found / 4), IS1009a 21.9 (5/4), TS3003a 18.8
(3/4), EN2002a 24.1 (5/4). Most of the error is missed overlapped speech (EN2002a talks over
itself a lot); confusion is ~6%.

- **pyannote's tuned clustering values hold.** Thresholds 0.5-0.75 with min cluster size 6 or
  12: 0.6-0.75 at 12 are all 21.4-21.5 at step 2. Lower thresholds or size 6 leave many small
  clusters, which split or swallow speakers (TS3003a fell to 1-2 speakers at 0.5).
- **Where the time goes** (step 2, per 14 min meeting): segmentation 25 s (wasm, ~65 ms per
  window, batches of 16), fbank 10 s (transformers.js `WeSpeakerFeatureExtractor`), voice
  model 50 s (~70 ms per embedding on WebGPU). The voice model is compute-bound on the
  integrated GPU: wasm is 5x slower (420 ms); batching equal-length inputs (frames repeated up
  to multiples of 100 or 200) and padding to few distinct shapes gave no speedup; capping an
  embedding at 3 s of audio saved 20% and cost 4 points of DER. The fp16 export
  (`model_fp16.onnx`) does not load in ORT 1.30 (a Cast output typed float16 where float is
  expected).
- **Clustering cost**: a distance matrix updated by Lance-Williams (exact for centroid
  linkage; the same DER as recomputing centroids) clusters EN2002a's 2361 voices in 2.4 s in
  Node. At most 4000 voices are clustered (an even sample; 64 MB of distances), the rest
  assigned to the clusters, as pyannote's `max_num_embeddings` does.
- **Synthetic check** (`tests/fixtures/tts-dialog.mp4`: three Windows TTS voices, David, Zira
  and Mark (two male), 8 turns, 57 s; `tests/fixtures/tts-dialog.ps1` and
  `scripts/tts-dialog.mjs`): 3 speakers, every turn and every Markdown paragraph attributed to
  the right voice, DER 7.7% (no confusion, nothing missed; turns run ~0.15 s long at their
  edges). The first version of its reference counted TTS's padding and sentence pauses as
  speech (22% "missed"); the reference is now the audible stretches of each turn.

## videodoc step 10, part 2: speaker names and corrections (2026-10-29)

How far apart one person's voices are in separate recordings, against two people's
(`scripts/eval-voices.ts`, on the cached AMI inputs; no browser). Each meeting is cut into
parts that stand in for separate videos, each part diarized alone (default settings, step 2),
its speakers mapped to the reference's (DER mapping), and every pair of speakers in different
parts compared by the cosine distance of their voices (cluster centroids, as the layer stores):

| Parts per meeting (length) | Voices | Same person: median / p90 / max | Two people, same meeting: min / p10 / median | Different meetings: min |
|---|---|---|---|---|
| 2 (11.5 min) | 26 | 0.064 / 0.095 / 0.150 | 0.581 / 0.696 / 0.868 | 0.607 |
| 3 (7.7 min) | 34 | 0.057 / 0.105 / 0.183 | 0.539 / 0.674 / 0.845 | 0.602 |
| 4 (5.8 min) | 41 | 0.063 / 0.237 / 0.388 | 0.162 / 0.602 / 0.849 | 0.577 |
| 6 (3.8 min) | 58 | 0.077 / 0.236 / 0.371 | 0.294 / 0.545 / 0.844 | 0.537 |

- **Threshold 0.4** (`SAME_VOICE` in `src/doc/speakers.ts`), for suggesting a name from a voice
  named in another video and for moving edits to the speakers found again: it misses no
  same-person pair at any part length; it accepts 2.5% of two-people pairs at 5.8-minute parts
  and 3.9% at 3.8 minutes (none at 7.7 minutes and longer, none across meetings, i.e. other
  rooms and microphones). The few close two-people pairs are short parts where a found speaker
  mixes two people. Hence suggestions only: a person confirms each.
- Same room, same microphones is the hard case and the one measured: a podcast's host in every
  episode is the use. Across rooms nothing came closer than 0.54.
