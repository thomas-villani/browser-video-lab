// Adaptive (divide-and-conquer) sparse detection.
//
// Start with keyframes every `maxStride` frames (plus scene cuts). For each interval [a, b]:
// if the detections at a and b "agree" — same number of confident boxes, and a one-to-one
// matching where every pair overlaps — interpolate the matched boxes across the interval.
// Otherwise run the detector at the midpoint and recurse on both halves. Detector calls
// concentrate where things change (motion, entrances/exits, flicker) and stay sparse
// where the scene is stable.
//
// `detect(frame)` is the only contact with the detector, so the same scheduler drives a live
// pipeline (async detector in the browser) or a replay of recorded detections (evaluation).

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
}

export const DEFAULT_ADAPTIVE: AdaptiveOptions = { maxStride: 16, minIou: 0.3, minScore: 0.5 };

function iou(a: Box, b: Box): number {
  const w = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  const inter = w > 0 && h > 0 ? w * h : 0;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter || 1);
}

interface Sampled {
  /** Boxes scoring at least minScore. */
  strong: Box[];
  /** Boxes scoring at least lowScore (a superset of strong). */
  weak: { box: Box; strong: boolean }[];
}

/**
 * Match endpoint boxes one-to-one over the weak sets; the endpoints agree when every
 * strong box on either side is matched (with IoU >= minIou). Returns the matched pairs
 * that involve at least one strong box, or null on disagreement.
 */
function agree(a: Sampled, b: Sampled, minIou: number): [Box, Box][] | null {
  if (!a.strong.length && !b.strong.length) return [];
  const cost = a.weak.map((x) => b.weak.map((y) => 1 - iou(x.box, y.box)));
  const { matches } = linearAssignment(cost, 1 - minIou, b.weak.length);
  const ma = new Set(matches.map((m) => m[0]));
  const mb = new Set(matches.map((m) => m[1]));
  if (a.weak.some((x, i) => x.strong && !ma.has(i)) || b.weak.some((y, j) => y.strong && !mb.has(j))) return null;
  return matches
    .filter(([i, j]) => a.weak[i].strong || b.weak[j].strong)
    .map(([i, j]) => [a.weak[i].box, b.weak[j].box]);
}

export interface AdaptiveResult {
  boxes: Box[][];
  /** Frames the detector ran on, ascending. */
  sampled: number[];
}

/** Synchronous version for replaying recorded detections (evaluation, tests). */
export function adaptiveBoxes(n: number, detect: (frame: number) => Detection[], opts: AdaptiveOptions): AdaptiveResult {
  const low = Math.min(opts.lowScore ?? opts.minScore, opts.minScore);
  const cache = new Map<number, Sampled>();
  const sample = (f: number): Sampled => {
    let s = cache.get(f);
    if (!s) {
      const weak = detect(f)
        .filter((d) => d.score >= low)
        .map((d) => ({ box: d.box, strong: d.score >= opts.minScore }));
      s = { weak, strong: weak.filter((w) => w.strong).map((w) => w.box) };
      cache.set(f, s);
    }
    return s;
  };
  const out: Box[][] = Array.from({ length: n }, () => []);

  // Scene boundaries split the video into independent spans; each span gets a coarse grid.
  const bounds = [0, ...(opts.cuts ?? []).filter((c) => c > 0 && c < n), n];
  for (let s = 0; s + 1 < bounds.length; s++) {
    const [start, end] = [bounds[s], bounds[s + 1] - 1];
    if (end < start) continue;
    const grid: number[] = [];
    for (let f = start; f < end; f += opts.maxStride) grid.push(f);
    grid.push(end);

    const solve = (a: number, b: number) => {
      const A = sample(a);
      const B = sample(b);
      out[a] = A.strong;
      out[b] = B.strong;
      if (b - a <= 1) return;
      const pairs = agree(A, B, opts.minIou);
      if (pairs) {
        for (let f = a + 1; f < b; f++) {
          const t = (f - a) / (b - a);
          out[f] = pairs.map(([p, q]) => p.map((v, i) => v + (q[i] - v) * t) as Box);
        }
        return;
      }
      const mid = (a + b) >> 1;
      solve(a, mid);
      solve(mid, b);
    };
    for (let i = 0; i + 1 < grid.length; i++) solve(grid[i], grid[i + 1]);
    if (grid.length === 1) out[start] = sample(start).strong;
  }
  return { boxes: out, sampled: [...cache.keys()].sort((x, y) => x - y) };
}
