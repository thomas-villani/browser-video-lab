// Per-frame boxes <-> compact tracks. Linking: boxes in consecutive frames are matched by IoU
// (optimal assignment), never across a scene cut; with `maxShift`, by distance from where the
// track's velocity puts it instead (small, fast objects: a passing car's plate moves most of its
// own width per frame, so consecutive boxes barely overlap); with `maxGap`, a track survives
// that many frames without a match and the gap is interpolated. Compression: a track keeps only the samples
// needed for linear interpolation to reproduce every frame's box within a tolerance. Pure TS.
import { linearAssignment } from '../track/assignment.ts';
import type { Box, Track } from './schema.ts';

function iou(a: Box, b: Box): number {
  const w = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  const inter = w > 0 && h > 0 ? w * h : 0;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter || 1);
}

export interface TrackOptions {
  /** Minimum IoU for boxes in consecutive frames to continue a track. */
  minIou: number;
  /** Interpolation error allowed per coordinate: max(px, rel x box size). */
  tolerance: { px: number; rel: number };
  /** Frames a track may go unmatched and still continue (a detector that flickers); 0 = none. */
  maxGap?: number;
  /** Match by center distance from the track's predicted box (last box plus its velocity)
   *  instead of IoU: at most this many box sizes away, sizes within 2x. */
  maxShift?: number;
}

type Open = { id: number; f0: number; boxes: Box[]; miss: number };

/** Where track t's box should be in the next frame: its last box moved by its last velocity. */
function predict(t: Open): Box {
  const n = t.boxes.length;
  const last = t.boxes[n - 1];
  if (n < 2) return last;
  const p = t.boxes[n - 2];
  const k = t.miss + 1;
  return [last[0] + (last[0] - p[0]) * k, last[1] + (last[1] - p[1]) * k, last[2], last[3]];
}

/** Center distance in units of the predicted box's size; Infinity when the sizes differ over 2x. */
function shift(p: Box, b: Box): number {
  const ratio = (b[2] * b[3]) / (p[2] * p[3] || 1);
  if (ratio > 4 || ratio < 0.25) return Infinity;
  const d = Math.hypot(b[0] + b[2] / 2 - (p[0] + p[2] / 2), b[1] + b[3] / 2 - (p[1] + p[3] / 2));
  return d / Math.max(p[2], p[3], 1);
}

export const DEFAULT_TRACKS: TrackOptions = { minIou: 0.3, tolerance: { px: 1, rel: 0.02 } };

/** Link per-frame boxes (frame 0 first) into tracks; `cuts` are frames that start a new shot. */
export function linkTracks(frames: Box[][], cuts: number[] = [], opts: TrackOptions = DEFAULT_TRACKS): Track[] {
  const cutSet = new Set(cuts);
  const maxGap = opts.maxGap ?? 0;
  // `miss`: frames since the track's last box (its boxes end before them).
  const done: Open[] = [];
  let open: Open[] = [];
  const s = opts.maxShift;
  let nextId = 1;
  frames.forEach((boxes, f) => {
    const prev = cutSet.has(f) ? [] : open;
    if (prev !== open) done.push(...open);
    const cost = s
      ? prev.map((t) => {
        const p = predict(t);
        return boxes.map((b) => shift(p, b) / s);
      })
      : prev.map((t) => boxes.map((b) => 1 - iou(t.boxes[t.boxes.length - 1], b)));
    const a = linearAssignment(cost, s ? 1 : 1 - opts.minIou, boxes.length);
    const next: typeof done = [];
    for (const [ti, bi] of a.matches) {
      const t = prev[ti];
      const last = t.boxes[t.boxes.length - 1];
      for (let k = 1; k <= t.miss; k++) t.boxes.push(lerp(last, boxes[bi], k / (t.miss + 1)) as Box);
      t.boxes.push(boxes[bi]);
      t.miss = 0;
      next.push(t);
    }
    for (const ti of a.unmatchedRows) {
      if (prev[ti].miss < maxGap) {
        prev[ti].miss++;
        next.push(prev[ti]);
      } else done.push(prev[ti]);
    }
    for (const bi of a.unmatchedCols) next.push({ id: 0, f0: f, boxes: [boxes[bi]], miss: 0 });
    open = next;
  });
  done.push(...open);
  // Number tracks by first appearance, so ids are stable for the same input.
  return done
    .sort((x, y) => x.f0 - y.f0 || x.boxes[0][0] - y.boxes[0][0] || x.boxes[0][1] - y.boxes[0][1])
    .map((t) => ({ id: nextId++, f0: t.f0, f1: t.f0 + t.boxes.length - 1, samples: compress(t.f0, t.boxes, opts.tolerance) }));
}

const lerp = (a: number[], b: number[], k: number) => a.map((v, i) => v + (b[i] - v) * k);
const r1 = (v: number) => Math.round(v * 10) / 10;

/** Greedy: extend each segment while interpolating its endpoints reproduces every box inside. */
function compress(f0: number, boxes: Box[], tol: TrackOptions['tolerance']): number[][] {
  const fits = (a: number, b: number) => {
    for (let i = a + 1; i < b; i++) {
      const want = boxes[i];
      const got = lerp(boxes[a], boxes[b], (i - a) / (b - a));
      const limit = Math.max(tol.px, tol.rel * Math.max(want[2], want[3]));
      if (got.some((v, j) => Math.abs(v - want[j]) > limit)) return false;
    }
    return true;
  };
  const keep = [0];
  let a = 0;
  while (a < boxes.length - 1) {
    let b = a + 1;
    while (b + 1 < boxes.length && fits(a, b + 1)) b++;
    keep.push(b);
    a = b;
  }
  return keep.map((i) => [f0 + i, ...boxes[i].map(r1)]);
}

/** A track's box at `frame`, or undefined outside [f0, f1]. */
export function trackBox(t: Track, frame: number): Box | undefined {
  if (frame < t.f0 || frame > t.f1) return undefined;
  const s = t.samples;
  let lo = 0;
  let hi = s.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (s[mid][0] <= frame) lo = mid;
    else hi = mid;
  }
  const [a, b] = [s[lo], s[hi]];
  if (a[0] === frame || a === b) return a.slice(1) as Box;
  return lerp(a.slice(1), b.slice(1), (frame - a[0]) / (b[0] - a[0])) as Box;
}

/** All tracks' boxes at `frame`, with their track ids. */
export function boxesAt(tracks: Track[], frame: number): { id: number; box: Box }[] {
  const out: { id: number; box: Box }[] = [];
  for (const t of tracks) {
    const box = trackBox(t, frame);
    if (box) out.push({ id: t.id, box });
  }
  return out;
}
