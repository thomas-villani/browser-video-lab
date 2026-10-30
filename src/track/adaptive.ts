// Adaptive (divide-and-conquer) sparse detection.
//
// Start with keyframes every `maxStride` frames (plus scene cuts). For each interval [a, b]:
// if the detections at a and b "agree" — same number of confident boxes, and a one-to-one
// matching where every pair overlaps — interpolate the matched boxes across the interval.
// Otherwise run the detector at the midpoint and recurse on both halves. Detector calls
// concentrate where things change (motion, entrances/exits, flicker) and stay sparse
// where the scene is stable.
//
// Decisions live in the `bisect` generator, which only asks for detections and never calls a
// detector itself, so the same logic drives a live async pipeline and an offline replay.

import { linearAssignment } from './assignment.ts';
import type { Box, Detection } from './bytetrack.ts';

export interface AdaptiveOptions {
  /** Initial keyframe spacing (the coarsest the scheduler will ever go). */
  maxStride: number;
  /** Minimum IoU for two endpoint boxes to count as the same object. */
  minIou: number;
  /** Detections below this score are ignored. */
  minScore: number;
  /**
   * Hysteresis: a confident box at one end may pair with a box this weak at the other end.
   * Stops a face whose score hovers around `minScore` from forcing needless subdivision.
   * Defaults to `minScore` (no hysteresis).
   */
  lowScore?: number;
  /** Scene-cut frame indices; always keyframes, and nothing interpolates across them. */
  cuts?: number[];
  /**
   * Redaction trade-off: when faces are unmatched at only ONE end of an interval (someone
   * entering or leaving), hold their boxes across the whole interval instead of subdividing.
   * Over-blurring briefly is cheap; detector calls are not. Unmatched faces at BOTH ends
   * (possibly one object that moved too far) still subdivide.
   */
  extendOneSided?: boolean;
}

export const DEFAULT_ADAPTIVE: AdaptiveOptions = { maxStride: 16, minIou: 0.3, minScore: 0.5 };

function iou(a: Box, b: Box): number {
  const w = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  const inter = w > 0 && h > 0 ? w * h : 0;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter || 1);
}

export interface Sampled {
  /** Boxes scoring at least minScore. */
  strong: Box[];
  /** Boxes scoring at least lowScore (a superset of strong). */
  weak: { box: Box; strong: boolean }[];
}

export interface Agreement {
  /** Matched endpoint pairs to interpolate. */
  pairs: [Box, Box][];
  /** One-sided (entering/leaving) boxes to hold across the interval (extendOneSided only). */
  held: Box[];
  /** Unmatched strong boxes at each end. */
  loneA: Box[];
  loneB: Box[];
}

/**
 * Match endpoint boxes one-to-one over the weak sets (IoU >= minIou). The endpoints agree
 * when every strong box on either side is matched — or, with `extendOneSided`, when the
 * unmatched strong boxes are all on one side. Returns null on disagreement, unless `always`
 * (then the caller handles the unmatched boxes itself).
 */
export function agree(a: Sampled, b: Sampled, minIou: number, extendOneSided: boolean, always = false): Agreement | null {
  if (!a.strong.length && !b.strong.length) return { pairs: [], held: [], loneA: [], loneB: [] };
  const cost = a.weak.map((x) => b.weak.map((y) => 1 - iou(x.box, y.box)));
  const { matches } = linearAssignment(cost, 1 - minIou, b.weak.length);
  const ma = new Set(matches.map((m) => m[0]));
  const mb = new Set(matches.map((m) => m[1]));
  const loneA = a.weak.filter((x, i) => x.strong && !ma.has(i)).map((x) => x.box);
  const loneB = b.weak.filter((y, j) => y.strong && !mb.has(j)).map((y) => y.box);
  if (loneA.length || loneB.length) {
    if (!always && (!extendOneSided || (loneA.length && loneB.length))) return null;
  }
  const pairs = matches.filter(([i, j]) => a.weak[i].strong || b.weak[j].strong).map(([i, j]) => [a.weak[i].box, b.weak[j].box] as [Box, Box]);
  return { pairs, held: [...loneA, ...loneB], loneA, loneB };
}

export interface AdaptiveResult {
  boxes: Box[][];
  /** Frames the detector ran on, ascending. */
  sampled: number[];
}

/** Detections reduced to what the scheduler compares. */
export function toSampled(dets: Detection[], opts: Pick<AdaptiveOptions, 'minScore' | 'lowScore'>): Sampled {
  const low = Math.min(opts.lowScore ?? opts.minScore, opts.minScore);
  const weak = dets.filter((d) => d.score >= low).map((d) => ({ box: d.box, strong: d.score >= opts.minScore }));
  return { weak, strong: weak.filter((w) => w.strong).map((w) => w.box) };
}

/**
 * The coarse keyframe grid as intervals to solve: keyframes at multiples of `maxStride`, the
 * last frame, and both sides of every cut; nothing spans a cut. A scene of one frame becomes
 * the degenerate interval [f, f]. Depends only on frame indices, so a streaming pipeline that
 * discovers cuts as it decodes builds the same grid as an offline replay.
 */
export function intervals(n: number, opts: Pick<AdaptiveOptions, 'maxStride' | 'cuts'>): [number, number][] {
  const cuts = [...new Set(opts.cuts ?? [])].filter((c) => c > 0 && c < n).sort((x, y) => x - y);
  const bounds = [0, ...cuts, n];
  const out: [number, number][] = [];
  for (let s = 0; s + 1 < bounds.length; s++) {
    const [start, end] = [bounds[s], bounds[s + 1] - 1];
    const keys = [start];
    for (let f = Math.ceil((start + 1) / opts.maxStride) * opts.maxStride; f < end; f += opts.maxStride) keys.push(f);
    if (end > start) keys.push(end);
    if (keys.length === 1) out.push([start, start]);
    for (let i = 0; i + 1 < keys.length; i++) out.push([keys[i], keys[i + 1]]);
  }
  return out;
}

/**
 * Solve one interval [a, b] by bisection. A generator, so the same decision logic serves a
 * synchronous replay and the async browser pipeline: it yields the frames it needs detected
 * (one batch per bisection level, which a live detector can pipeline), receives their
 * detections in the same order, and returns boxes for frames a..b. `cache` is shared across
 * intervals so a keyframe shared by two neighbors is detected once.
 */
export function* bisect(a: number, b: number, opts: AdaptiveOptions, cache: Map<number, Sampled>): Generator<number[], Box[][], Detection[][]> {
  const out: Box[][] = Array.from({ length: b - a + 1 }, () => []);
  let pending: [number, number][] = [[a, b]];
  let need = [...new Set([a, b])].filter((f) => !cache.has(f));
  for (;;) {
    if (need.length) {
      const dets = yield need;
      need.forEach((f, i) => cache.set(f, toSampled(dets[i], opts)));
    }
    const next: [number, number][] = [];
    for (const [x, y] of pending) {
      const X = cache.get(x)!;
      const Y = cache.get(y)!;
      out[x - a] = X.strong;
      out[y - a] = Y.strong;
      if (y - x <= 1) continue;
      const agreement = agree(X, Y, opts.minIou, opts.extendOneSided ?? false);
      if (agreement) {
        for (let f = x + 1; f < y; f++) {
          const t = (f - x) / (y - x);
          out[f - a] = agreement.pairs.map(([p, q]) => p.map((v, i) => v + (q[i] - v) * t) as Box);
          out[f - a].push(...agreement.held);
        }
      } else {
        const mid = (x + y) >> 1;
        next.push([x, mid], [mid, y]);
      }
    }
    if (!next.length) return out;
    pending = next;
    need = [...new Set(next.flat())].filter((f) => !cache.has(f));
  }
}

/** Synchronous driver for replaying recorded detections (evaluation, tests). */
export function adaptiveBoxes(n: number, detect: (frame: number) => Detection[], opts: AdaptiveOptions): AdaptiveResult {
  const cache = new Map<number, Sampled>();
  const out: Box[][] = Array.from({ length: n }, () => []);
  for (const [a, b] of intervals(n, opts)) {
    const gen = bisect(a, b, opts, cache);
    let step = gen.next();
    while (!step.done) step = gen.next(step.value.map(detect));
    step.value.forEach((boxes, i) => (out[a + i] = boxes));
  }
  return { boxes: out, sampled: [...cache.keys()].sort((x, y) => x - y) };
}
