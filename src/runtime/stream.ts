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
//
// Several detectors (heads, e.g. faces and plates) can share the decode: multiStream.
import type { GeometryOptions } from '../models';
import { SceneScorer } from '../scene';
import { type AdaptiveOptions, bisect, intervals, type Sampled } from '../track/adaptive';
import type { Box } from '../track/bytetrack';
import type { Detector } from './detector';
import { FrameIngest, type IngestOptions } from './ingest';
import { activeTiles, mergeBoxes, ownBoxes, type RefineOptions, refinedFor, tileGrid } from './refine';
import type { VideoSource } from './source';

/** One detector over the shared decode, with its own sampling (and optional refinement). */
export interface Head extends Omit<AdaptiveOptions, 'maxStride' | 'cuts'> {
  detector: Detector;
  /** This head's coarsest spacing; must divide the shared `maxStride` (default: equal to it). */
  maxStride?: number;
  /** Detector geometry (e.g. `longSide` for a reduced input). */
  detect?: GeometryOptions;
  /** Native-resolution tile refinement (crowds). The detector's score threshold must be at or
   *  below `refine.trigger`. `every` should divide `maxStride`. */
  refine?: RefineOptions;
}

export interface MultiOptions {
  /** Coarsest spacing between detected frames, shared by every head (the chunk size). */
  maxStride: number;
  /** Scene score at or above which a frame starts a new scene; 0 disables cut detection. */
  cutThreshold: number;
  ingest?: IngestOptions;
  /** Stop after this many frames. */
  limit?: number;
}

export interface StreamOptions extends AdaptiveOptions, MultiOptions {
  detect?: GeometryOptions;
  refine?: RefineOptions;
}

/** One head's result on one frame. */
export interface HeadResult {
  /** Boxes in source pixels: detected, interpolated, or held. */
  boxes: Box[];
  /** The detector ran on this frame. */
  sampled: boolean;
  /** Native-resolution tiles detected on this frame (refinement). */
  tiles: number;
}

export interface MultiResult {
  frame: number;
  t: number;
  sceneScore: number;
  /** This frame starts a new scene. */
  cut: boolean;
  /** In the order of the heads given. */
  heads: HeadResult[];
}

export interface FrameResult extends HeadResult {
  frame: number;
  t: number;
  sceneScore: number;
  cut: boolean;
}

interface Pending {
  frame: number;
  t: number;
  score: number;
}

/** One detector over the video (see multiStream). */
export async function* adaptiveStream(source: VideoSource, detector: Detector, opts: StreamOptions): AsyncGenerator<FrameResult> {
  const { maxStride, cutThreshold, ingest, limit, cuts: _, ...head } = opts;
  for await (const r of multiStream(source, [{ ...head, detector }], { maxStride, cutThreshold, ingest, limit })) {
    yield { frame: r.frame, t: r.t, sceneScore: r.sceneScore, cut: r.cut, ...r.heads[0] };
  }
}

/**
 * Decode once and run every head on the frames its own sampling picks (faces and plates change
 * at different moments, so each bisects on its own). The retained frames live on the first
 * head's GPU device; ORT shares one device across sessions, so every head can read them.
 */
export async function* multiStream(source: VideoSource, heads: Head[], opts: MultiOptions): AsyncGenerator<MultiResult> {
  const S = opts.maxStride;
  const ingest = new FrameIngest(heads[0].detector.device, source.width, source.height, S + 1, opts.ingest);
  const slot = (f: number) => f % (S + 1);
  // Retained frames are scaled down when the video exceeds ingest.maxSide; map boxes back.
  const toSrc = source.width / ingest.width;
  const toBox = (d: { x: number; y: number; w: number; h: number }) => [d.x, d.y, d.w, d.h].map((v) => v * toSrc) as Box;
  const scorer = new SceneScorer();
  const cutFrames = new Set<number>();
  const cutBetween = (a: number, b: number) => [...cutFrames].some((c) => c > a && c <= b);
  let chunk: Pending[] = [];
  let scored = -1;

  const states = heads.map((h) => {
    const cache = new Map<number, Sampled>();
    const sampled = new Set<number>();
    // Raw reduced-pass detections (refinement triggers) and per-frame tile results.
    const raw = new Map<number, { box: Box; score: number }[]>();
    const refined = new Map<number, Box[]>();
    const tileCalls = new Map<number, number>();
    const r = h.refine;
    const tiles = r ? tileGrid(source.width, source.height, r.tileW, r.tileH, r.overlap) : [];
    const stride = h.maxStride ?? S;
    if (S % stride) throw new Error(`a head's maxStride (${stride}) must divide the shared one (${S})`);
    const adaptive: AdaptiveOptions = { ...h, maxStride: stride };

    const detect = async (f: number) => {
      sampled.add(f);
      const dets = (await h.detector.detect(ingest.texture(slot(f)), h.detect)).map((d) => ({ box: toBox(d), score: d.score }));
      raw.set(f, dets);
      return dets;
    };

    /** Detect frame f's active tiles at native resolution (relative to the retained texture). */
    const refine = async (f: number, current: Box[]) => {
      if (!r || refined.has(f)) return;
      if (!raw.has(f)) await detect(f);
      const triggers = [
        ...current,
        ...raw
          .get(f)!
          .filter((d) => d.score >= r.trigger)
          .map((d) => d.box),
      ].filter((b) => !r.small || b[3] < r.small);
      const active = activeTiles(tiles, triggers, r.grow);
      const found = await Promise.all(
        active.map(async (t) => {
          const crop = { x: t[0] / toSrc, y: t[1] / toSrc, w: t[2] / toSrc, h: t[3] / toSrc };
          const dets = await h.detector.detect(ingest.texture(slot(f)), { crop, longSide: Math.max(crop.w, crop.h) });
          return ownBoxes(t, dets.filter((d) => d.score >= h.minScore).map(toBox));
        }),
      );
      refined.set(f, mergeBoxes([], found.flat()));
      tileCalls.set(f, active.length);
    };

    /** Boxes for the chunk's frames a..g (indexed from a), detecting what the bisection asks for. */
    const solve = async (a: number, g: number, cuts: number[]): Promise<Box[][]> => {
      const boxes: Box[][] = [];
      for (const [x, y] of intervals(g - a + 1, { maxStride: stride, cuts })) {
        const gen = bisect(a + x, a + y, adaptive, cache);
        let step = gen.next();
        while (!step.done) step = gen.next(await Promise.all(step.value.map(detect)));
        step.value.forEach((b, i) => (boxes[x + i] = b));
      }
      if (r) {
        const due: Promise<void>[] = [];
        for (let f = a + ((r.every - (a % r.every)) % r.every); f <= g; f += r.every) due.push(refine(f, boxes[f - a]));
        await Promise.all(due);
      }
      return boxes;
    };

    const result = (f: number, boxes: Box[], g: number): HeadResult => {
      const extra = r ? refinedFor(f, r.every, g, cutBetween).flatMap((x) => refined.get(x) ?? []) : [];
      return { boxes: mergeBoxes(boxes, extra), sampled: sampled.has(f), tiles: tileCalls.get(f) ?? 0 };
    };

    const forget = (g: number) => {
      for (const f of cache.keys()) if (f < g) cache.delete(f);
      for (const m of [raw, refined, tileCalls]) for (const f of m.keys()) if (f < g) m.delete(f);
      for (const f of sampled) if (f < g) sampled.delete(f);
    };
    return { solve, result, forget };
  });

  async function* flush(final: boolean): AsyncGenerator<MultiResult> {
    const a = chunk[0].frame;
    const g = chunk[chunk.length - 1].frame;
    const fresh = chunk.filter((p) => p.frame > scored);
    const thumbs = await ingest.thumbs(fresh.map((p) => slot(p.frame)));
    fresh.forEach((p, i) => (p.score = scorer.push(thumbs[i])));
    scored = g;
    const cutAt = (p: Pending) => opts.cutThreshold > 0 && p.frame > 0 && p.score >= opts.cutThreshold;
    const local = chunk.filter((p) => p.frame > a && cutAt(p)).map((p) => p.frame - a);
    const boxes = await Promise.all(states.map((s) => s.solve(a, g, local)));
    for (const p of chunk) if (cutAt(p)) cutFrames.add(p.frame);

    const emit = final ? chunk : chunk.slice(0, -1);
    for (const [i, p] of emit.entries()) {
      yield { frame: p.frame, t: p.t, sceneScore: p.score, cut: cutAt(p), heads: states.map((s, h) => s.result(p.frame, boxes[h][i], g)) };
    }
    for (const s of states) s.forget(g);
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
