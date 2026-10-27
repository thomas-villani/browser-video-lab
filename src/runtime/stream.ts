// Single-pass adaptive detection over a video file.
//
// Frames are decoded once, in order, in chunks of `maxStride` frames. Every frame is ingested
// (retained on the GPU + thumbnail). At each chunk boundary the chunk's thumbnails are read back
// in one map and scored for scene cuts; the chunk is then split into the same intervals the
// offline replay would use (`intervals`), and `bisect` decides which retained frames the
// detector runs on. Consecutive chunks share their boundary frame.
//
// With `refine`, every Nth frame is also detected in native-resolution tiles wherever the
// (reduced-resolution) pass saw a face, and those boxes are merged into nearby frames.
import type { GeometryOptions } from '../models';
import { SceneScorer } from '../scene';
import { bisect, intervals, type AdaptiveOptions, type Sampled } from '../track/adaptive';
import type { Box } from '../track/bytetrack';
import type { Detector } from './detector';
import { FrameIngest, type IngestOptions } from './ingest';
import { activeTiles, mergeBoxes, ownBoxes, refinedFor, tileGrid, type RefineOptions } from './refine';
import type { VideoSource } from './source';

export interface StreamOptions extends AdaptiveOptions {
  /** Scene score at or above which a frame starts a new scene; 0 disables cut detection. */
  cutThreshold: number;
  /** Detector geometry (e.g. `longSide` for a reduced input). */
  detect?: GeometryOptions;
  ingest?: IngestOptions;
  /** Native-resolution tile refinement (crowds). The detector's score threshold must be at or
   *  below `refine.trigger`. `every` should divide `maxStride`. */
  refine?: RefineOptions;
  /** Stop after this many frames. */
  limit?: number;
}

export interface FrameResult {
  frame: number;
  t: number;
  /** Boxes for this frame in source pixels: detected, interpolated, or held. */
  boxes: Box[];
  /** The detector ran on this frame. */
  sampled: boolean;
  sceneScore: number;
  /** This frame starts a new scene. */
  cut: boolean;
  /** Native-resolution tiles detected on this frame (refinement). */
  tiles: number;
}

interface Pending {
  frame: number;
  t: number;
  score: number;
}

export async function* adaptiveStream(
  source: VideoSource,
  detector: Detector,
  opts: StreamOptions,
): AsyncGenerator<FrameResult> {
  const S = opts.maxStride;
  const ingest = new FrameIngest(detector.device, source.width, source.height, S + 1, opts.ingest);
  const slot = (f: number) => f % (S + 1);
  // Retained frames are scaled down when the video exceeds ingest.maxSide; map boxes back.
  const toSrc = source.width / ingest.width;
  const scorer = new SceneScorer();
  const cache = new Map<number, Sampled>();
  const sampled = new Set<number>();
  let chunk: Pending[] = [];
  let scored = -1;

  const toBox = (d: { x: number; y: number; w: number; h: number }) => [d.x, d.y, d.w, d.h].map((v) => v * toSrc) as Box;
  // Raw reduced-pass detections (refinement triggers) and per-frame tile results.
  const raw = new Map<number, { box: Box; score: number }[]>();
  const refined = new Map<number, Box[]>();
  const tileCalls = new Map<number, number>();
  const cutFrames = new Set<number>();
  const r = opts.refine;
  const tiles = r ? tileGrid(source.width, source.height, r.tileW, r.tileH, r.overlap) : [];

  const detect = async (f: number) => {
    sampled.add(f);
    const dets = (await detector.detect(ingest.texture(slot(f)), opts.detect)).map((d) => ({ box: toBox(d), score: d.score }));
    raw.set(f, dets);
    return dets;
  };

  /** Detect frame f's active tiles at native resolution (relative to the retained texture). */
  const refine = async (f: number, current: Box[]) => {
    if (!r || refined.has(f)) return;
    if (!raw.has(f)) await detect(f);
    const triggers = [...current, ...raw.get(f)!.filter((d) => d.score >= r.trigger).map((d) => d.box)].filter((b) => !r.small || b[3] < r.small);
    const active = activeTiles(tiles, triggers, r.grow);
    const found = await Promise.all(
      active.map(async (t) => {
        const crop = { x: t[0] / toSrc, y: t[1] / toSrc, w: t[2] / toSrc, h: t[3] / toSrc };
        const dets = await detector.detect(ingest.texture(slot(f)), { crop, longSide: Math.max(crop.w, crop.h) });
        return ownBoxes(t, dets.filter((d) => d.score >= opts.minScore).map(toBox));
      }),
    );
    refined.set(f, mergeBoxes([], found.flat()));
    tileCalls.set(f, active.length);
  };
  const cutBetween = (a: number, b: number) => [...cutFrames].some((c) => c > a && c <= b);

  async function* flush(final: boolean): AsyncGenerator<FrameResult> {
    const a = chunk[0].frame;
    const g = chunk[chunk.length - 1].frame;
    const fresh = chunk.filter((p) => p.frame > scored);
    const thumbs = await ingest.thumbs(fresh.map((p) => slot(p.frame)));
    fresh.forEach((p, i) => (p.score = scorer.push(thumbs[i])));
    scored = g;
    const cutAt = (p: Pending) => opts.cutThreshold > 0 && p.frame > 0 && p.score >= opts.cutThreshold;
    const local = chunk.filter((p) => p.frame > a && cutAt(p)).map((p) => p.frame - a);

    const boxes: Box[][] = [];
    for (const [x, y] of intervals(g - a + 1, { maxStride: S, cuts: local })) {
      const gen = bisect(a + x, a + y, opts, cache);
      let step = gen.next();
      while (!step.done) step = gen.next(await Promise.all(step.value.map(detect)));
      step.value.forEach((b, i) => (boxes[x + i] = b));
    }
    for (const p of chunk) if (cutAt(p)) cutFrames.add(p.frame);
    if (r) {
      const due: Promise<void>[] = [];
      for (let f = a + ((r.every - (a % r.every)) % r.every); f <= g; f += r.every) due.push(refine(f, boxes[f - a]));
      await Promise.all(due);
    }

    const emit = final ? chunk : chunk.slice(0, -1);
    for (const [i, p] of emit.entries()) {
      const extra = r ? refinedFor(p.frame, r.every, g, cutBetween).flatMap((f) => refined.get(f) ?? []) : [];
      yield {
        frame: p.frame, t: p.t, boxes: mergeBoxes(boxes[i], extra), sampled: sampled.has(p.frame),
        sceneScore: p.score, cut: cutAt(p), tiles: tileCalls.get(p.frame) ?? 0,
      };
    }
    for (const f of cache.keys()) if (f < g) cache.delete(f);
    for (const m of [raw, refined, tileCalls]) for (const f of m.keys()) if (f < g) m.delete(f);
    for (const c of cutFrames) if (c < g) cutFrames.delete(c);
    chunk = [chunk[chunk.length - 1]];
  }

  try {
    for await (const { index, t, frame } of source.frames(opts.limit)) {
      ingest.put(frame, slot(index));
      chunk.push({ frame: index, t, score: 0 });
      if (index > 0 && index % S === 0) yield* flush(false);
    }
    if (chunk.length) yield* flush(true);
  } finally {
    ingest.destroy();
  }
}
