// Windowed bidirectional ROI tracking: a cheaper way to cover crowds.
//
// Full-frame detection runs only at window endpoints (every `window` frames, plus both sides
// of cuts). Faces that agree across a window are interpolated, exactly as in bisection. The
// rest are followed with small ROI detections instead of more full-frame calls:
//   - faces only at the start (leaving, or moving too fast to match) are tracked forward
//     until the ROI no longer finds them;
//   - faces only at the end (entering) are tracked *backward* from the end — possible because
//     the whole window is retained on the GPU — until lost or until they meet a forward track.
// Faces that appear and vanish entirely inside one window are missed; keep the window short
// relative to how long a face stays on screen.
//
// Like `bisect`, it is a generator: it yields batches of detection requests (full frame or
// crop) and receives detections, so one implementation serves replay and the live pipeline.
// All requests in a batch are independent, so a live detector can pack crops into one call.

import { type AdaptiveOptions, agree, intervals, type Sampled, toSampled } from './adaptive.ts';
import type { Box, Detection } from './bytetrack.ts';

export interface RoiOptions extends Pick<AdaptiveOptions, 'minIou' | 'minScore' | 'lowScore' | 'cuts'> {
  /** Full-frame detection every this many frames. */
  window: number;
  /** ROI side = expand x the larger side of the predicted face box. */
  expand: number;
  frameW: number;
  frameH: number;
}

export const DEFAULT_ROI: Omit<RoiOptions, 'frameW' | 'frameH'> = { window: 8, expand: 2.5, minIou: 0.3, minScore: 0.5 };

/** A detection request: the whole frame, or only a crop of it ([x, y, w, h] in frame pixels). */
export interface DetectRequest {
  frame: number;
  crop?: Box;
}

interface Track {
  box: Box;
  /** Per-frame motion of the box, in the direction of travel. */
  vel: Box;
}

function iou(a: Box, b: Box): number {
  const w = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  const inter = w > 0 && h > 0 ? w * h : 0;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter || 1);
}

const add = (a: Box, b: Box) => a.map((v, i) => v + b[i]) as Box;
const sub = (a: Box, b: Box) => a.map((v, i) => v - b[i]) as Box;

/** Square ROI around a box, clamped to the frame. */
export function roiFor([x, y, w, h]: Box, expand: number, frameW: number, frameH: number): Box {
  const side = Math.min(Math.max(w, h) * expand, frameW, frameH);
  const cx = Math.min(Math.max(x + w / 2, side / 2), frameW - side / 2);
  const cy = Math.min(Math.max(y + h / 2, side / 2), frameH - side / 2);
  return [cx - side / 2, cy - side / 2, side, side];
}

/**
 * Follow `tracks` frame by frame along `frames`, one batch of ROI requests per frame. A track
 * is dropped when its ROI holds no detection overlapping the prediction, or when `covered`
 * says another track already has it. Found boxes are written to out[frame - base].
 */
function* follow(
  tracks: Track[],
  frames: number[],
  base: number,
  opts: RoiOptions,
  out: Box[][],
  covered: (frame: number, box: Box) => boolean,
): Generator<DetectRequest[], void, Detection[][]> {
  const low = Math.min(opts.lowScore ?? opts.minScore, opts.minScore);
  for (const f of frames) {
    const live = tracks.filter((t) => !covered(f, add(t.box, t.vel)));
    if (!live.length) return;
    const predicted = live.map((t) => add(t.box, t.vel));
    const results = yield predicted.map((p) => ({ frame: f, crop: roiFor(p, opts.expand, opts.frameW, opts.frameH) }));
    tracks = [];
    live.forEach((t, i) => {
      let best: Detection | undefined;
      let bestIou = 0.1;
      for (const d of results[i]) {
        const o = iou(d.box, predicted[i]);
        if (d.score >= low && o > bestIou) [best, bestIou] = [d, o];
      }
      if (!best) return;
      out[f - base].push(best.box);
      tracks.push({ box: best.box, vel: sub(best.box, t.box) });
    });
  }
}

/**
 * Solve window [a, b]. `cache` holds full-frame detections shared between neighboring
 * windows. Returns boxes for frames a..b.
 */
export function* roiWindow(a: number, b: number, opts: RoiOptions, cache: Map<number, Sampled>): Generator<DetectRequest[], Box[][], Detection[][]> {
  const need = [...new Set([a, b])].filter((f) => !cache.has(f));
  if (need.length) {
    const dets = yield need.map((frame) => ({ frame }));
    // Everything above lowScore counts: small faces flicker around the threshold, and following
    // a weak face with ROI calls is cheap, while leaking it is not.
    const seed = { minScore: Math.min(opts.lowScore ?? opts.minScore, opts.minScore) };
    need.forEach((f, i) => cache.set(f, toSampled(dets[i], seed)));
  }
  const A = cache.get(a)!;
  const B = cache.get(b)!;
  const out: Box[][] = Array.from({ length: b - a + 1 }, () => []);
  out[0] = A.strong;
  out[b - a] = B.strong;
  if (b - a <= 1) return out;

  // Faces that agree across the window are interpolated; the rest are tracked.
  const { pairs, loneA, loneB } = agree(A, B, opts.minIou, false, true)!;
  for (let f = a + 1; f < b; f++) {
    const t = (f - a) / (b - a);
    out[f - a] = pairs.map(([p, q]) => p.map((v, i) => v + (q[i] - v) * t) as Box);
  }
  const interior = Array.from({ length: b - a - 1 }, (_, i) => a + 1 + i);
  const zero: Box = [0, 0, 0, 0];
  const forward: Box[][] = Array.from({ length: b - a + 1 }, () => []);
  yield* follow(
    loneA.map((box) => ({ box, vel: zero })),
    interior,
    a,
    opts,
    forward,
    () => false,
  );
  // Backward tracks stop where a forward track already holds the same face.
  const met = (f: number, box: Box) => forward[f - a].some((x) => iou(x, box) >= opts.minIou);
  const backward: Box[][] = Array.from({ length: b - a + 1 }, () => []);
  yield* follow(
    loneB.map((box) => ({ box, vel: zero })),
    interior.reverse(),
    a,
    opts,
    backward,
    met,
  );
  for (let i = 1; i < b - a; i++) out[i].push(...forward[i], ...backward[i]);
  return out;
}

export interface RoiResult {
  boxes: Box[][];
  fullCalls: number;
  /** ROI requests, i.e. crops that a mosaic would pack. */
  roiCrops: number;
  /** Batches of ROI requests when `group` windows advance in lockstep (= mosaic calls, if each fits). */
  roiBatches: number;
}

/**
 * Synchronous driver for replay. `group` windows advance in lockstep and their requests are
 * merged into one batch, as a live pipeline would pack them into one mosaic call.
 */
export function roiBoxes(n: number, detect: (req: DetectRequest) => Detection[], opts: RoiOptions, group = 4): RoiResult {
  const cache = new Map<number, Sampled>();
  const out: Box[][] = Array.from({ length: n }, () => []);
  const all = intervals(n, { maxStride: opts.window, cuts: opts.cuts });
  let [fullCalls, roiCrops, roiBatches] = [0, 0, 0];
  for (let g = 0; g < all.length; g += group) {
    const gens = all.slice(g, g + group).map(([a, b]) => ({ a, gen: roiWindow(a, b, opts, cache) }));
    let steps = gens.map(({ gen }) => gen.next());
    while (steps.some((s) => !s.done)) {
      const reqs = steps.flatMap((s) => (s.done ? [] : s.value));
      const crops = reqs.filter((r) => r.crop).length;
      fullCalls += reqs.length - crops;
      roiCrops += crops;
      if (crops) roiBatches++;
      steps = steps.map((s, i) => (s.done ? s : gens[i].gen.next(s.value.map(detect))));
    }
    steps.forEach((s, i) => s.done && s.value.forEach((boxes, j) => (out[gens[i].a + j] = boxes)));
  }
  return { boxes: out, fullCalls, roiCrops, roiBatches };
}
