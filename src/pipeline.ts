// Pipeline page: single-pass adaptive detection over a whole video (runtime/stream.ts).
// Configured by URL query params, like the bench page; the result is exposed as
// window.__BENCH_RESULT__ and rendered into <pre id="result">.
//
//   ?video=/media/rickroll.mp4  &model=yunet  &maxstride=16  &miniou=0.3  &minscore=0.5
//   &lowscore=0.5  &extend=0|1  &cut=0.15 (scene threshold, 0 = off)  &longside=640 (0 = model size)
//   &frames=0 (0 = all)  &depth=2  &maxside=1920  &dump=0|1
//
// With dump=1 the result also holds per-frame boxes, the sampled frames, and scene scores.
import { Detector } from './runtime/detector';
import { VideoSource } from './runtime/source';
import { adaptiveStream } from './runtime/stream';

declare global {
  interface Window {
    __BENCH_RESULT__?: unknown;
  }
}

function readConfig() {
  const q = new URLSearchParams(location.search);
  const num = (k: string, d: number) => (q.has(k) && q.get(k) !== '' ? Number(q.get(k)) : d);
  const minScore = num('minscore', 0.5);
  return {
    video: q.get('video') ?? '/media/rickroll.mp4',
    model: q.get('model') ?? 'yunet',
    maxStride: num('maxstride', 16),
    minIou: num('miniou', 0.3),
    minScore,
    lowScore: num('lowscore', minScore),
    extendOneSided: q.get('extend') === '1',
    cutThreshold: num('cut', 0.15),
    longSide: num('longside', 0) || undefined,
    frames: num('frames', 0) || Infinity,
    depth: num('depth', 2),
    maxSide: num('maxside', 1920),
    dump: q.get('dump') === '1',
  };
}

const r1 = (v: number) => Math.round(v * 10) / 10;

async function run(cfg: ReturnType<typeof readConfig>) {
  const status = document.getElementById('status')!;
  const wall0 = performance.now();
  // The detector keeps boxes down to the hysteresis threshold; the scheduler sorts strong from weak.
  const detector = await Detector.create({ model: cfg.model, depth: cfg.depth, score: Math.min(cfg.lowScore, cfg.minScore) });
  const sessionMs = performance.now() - wall0;
  const source = await VideoSource.open(cfg.video);

  const boxes: number[][][] = [];
  const sampled: number[] = [];
  const cuts: number[] = [];
  const sceneScores: number[] = [];
  const t0 = performance.now();
  for await (const r of adaptiveStream(source, detector, {
    ...cfg,
    cuts: undefined,
    detect: cfg.longSide ? { longSide: cfg.longSide } : undefined,
    ingest: { maxSide: cfg.maxSide },
    limit: cfg.frames,
  })) {
    if (r.sampled) sampled.push(r.frame);
    if (r.cut) cuts.push(r.frame);
    if (cfg.dump) {
      boxes.push(r.boxes.map((b) => b.map(r1)));
      sceneScores.push(Math.round(r.sceneScore * 1000) / 1000);
    }
    if (r.frame % 60 === 0) status.textContent = `frame ${r.frame}, ${sampled.length} detector calls`;
  }
  const ms = performance.now() - t0;
  const n = cfg.dump ? boxes.length : sampled.length && Math.max(...sampled) + 1;
  const { calls, inputPixels, submitMs, resultMs } = detector.stats;
  return {
    config: { ...cfg, frames: Number.isFinite(cfg.frames) ? cfg.frames : 0 },
    env: { userAgent: navigator.userAgent, adapter: detector.adapter, crossOriginIsolated },
    video: await source.meta(),
    sessionMs: r1(sessionMs),
    frames: n,
    detectorCalls: calls,
    callsPct: r1((100 * calls) / n),
    // Model-input megapixels per video frame: a cost proxy comparable across input sizes.
    inputMpxPerFrame: Math.round((inputPixels / n / 1e6) * 1000) / 1000,
    cuts,
    ms: r1(ms),
    // Video frames processed per wall-clock second (decode + ingest + scheduling + detection).
    fps: r1((n * 1000) / ms),
    detectorMs: { submit: r1(submitMs), result: r1(resultMs) },
    ...(cfg.dump && { boxes, sampled, sceneScores }),
  };
}

const cfg = readConfig();
const out = document.getElementById('result')!;
run(cfg)
  .then((r) => {
    window.__BENCH_RESULT__ = r;
    out.textContent = JSON.stringify({ ...r, boxes: undefined, sampled: undefined, sceneScores: undefined }, null, 2);
    document.getElementById('status')!.textContent = `done: ${r.fps} fps, ${r.callsPct}% detector calls`;
  })
  .catch((e: unknown) => {
    const error = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
    window.__BENCH_RESULT__ = { config: cfg, error };
    out.textContent = error;
    document.getElementById('status')!.textContent = 'failed';
  });
