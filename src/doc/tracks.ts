// Per-frame boxes <-> compact tracks. Linking: boxes in consecutive frames are matched by IoU
// (optimal assignment), never across a scene cut. Compression: a track keeps only the samples
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
}

export const DEFAULT_TRACKS: TrackOptions = { minIou: 0.3, tolerance: { px: 1, rel: 0.02 } };

/** Link per-frame boxes (frame 0 first) into tracks; `cuts` are frames that start a new shot. */
export function linkTracks(frames: Box[][], cuts: number[] = [], opts: TrackOptions = DEFAULT_TRACKS): Track[] {
  const cutSet = new Set(cuts);
  const done: { id: number; f0: number; boxes: Box[] }[] = [];
  let open: typeof done = [];
  let nextId = 1;
  frames.forEach((boxes, f) => {
    const prev = cutSet.has(f) ? [] : open;
    if (prev !== open) done.push(...open);
    const cost = prev.map((t) => boxes.map((b) => 1 - iou(t.boxes[t.boxes.length - 1], b)));
    const a = linearAssignment(cost, 1 - opts.minIou, boxes.length);
    const next: typeof done = [];
    for (const [ti, bi] of a.matches) {
      prev[ti].boxes.push(boxes[bi]);
      next.push(prev[ti]);
    }
    for (const ti of a.unmatchedRows) done.push(prev[ti]);
    for (const bi of a.unmatchedCols) next.push({ id: 0, f0: f, boxes: [boxes[bi]] });
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
