// Sparse detection: run the detector on every `stride`-th frame (keyframes) and produce
// boxes for *every* frame. Three strategies:
//
//   hold    — no tracker: reuse the last keyframe's detections until the next keyframe.
//   online  — ByteTrack on keyframes; between keyframes, extrapolate each track along its
//             Kalman velocity. Causal: usable for live video.
//   offline — ByteTrack on keyframes; between two observations of the same track, linearly
//             interpolate. Uses the future, so it's only for files — which is exactly the
//             redaction case. Track ends are extended by one stride so the first/last
//             frames of an appearance are still covered.

import { type Box, ByteTracker, type ByteTrackOptions, type Detection } from './bytetrack.ts';

export type SparseMode = 'hold' | 'online' | 'offline';

export interface SparseOptions {
  stride: number;
  mode: SparseMode;
  /** Video frame rate; sets the tracker's memory to about one second. */
  fps: number;
  /** Grow every output box by this fraction of its size on each side (redaction safety margin). */
  pad?: number;
  tracker?: Partial<ByteTrackOptions>;
  /**
   * Scene-cut frame indices (first frame of each new scene). When given, each scene is
   * processed independently — the tracker resets and nothing interpolates across a cut —
   * and each scene's first and last frames are always keyframes.
   */
  cuts?: number[];
}

export interface SparseResult {
  boxes: Box[][];
  /** Frames the detector actually had to run on. */
  detectorCalls: number;
}

/** Sparse detection with optional scene awareness (see SparseOptions.cuts). */
export function sparse(frames: Detection[][], opts: SparseOptions): SparseResult {
  if (!opts.cuts?.length) {
    return { boxes: sparseBoxes(frames, opts), detectorCalls: Math.ceil(frames.length / opts.stride) };
  }
  const bounds = [0, ...opts.cuts.filter((c) => c > 0 && c < frames.length), frames.length];
  const boxes: Box[][] = [];
  let detectorCalls = 0;
  for (let i = 0; i + 1 < bounds.length; i++) {
    const scene = frames.slice(bounds[i], bounds[i + 1]);
    if (!scene.length) continue;
    boxes.push(...sparseBoxes(scene, { ...opts, cuts: undefined, lastKeyframe: true }));
    detectorCalls += Math.ceil(scene.length / opts.stride) + ((scene.length - 1) % opts.stride ? 1 : 0);
  }
  return { boxes, detectorCalls };
}

export function padBox([x, y, w, h]: Box, pad: number): Box {
  return [x - w * pad, y - h * pad, w * (1 + 2 * pad), h * (1 + 2 * pad)];
}

function iou(a: Box, b: Box): number {
  const w = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  const inter = w > 0 && h > 0 ? w * h : 0;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter || 1);
}

function lerp(a: Box, b: Box, s: number): Box {
  return a.map((v, i) => v + (b[i] - v) * s) as Box;
}

/**
 * `frames[i]` holds frame i's detections. Returns boxes for every frame.
 * `lastKeyframe` also runs the detector on the final frame (used per scene).
 */
export function sparseBoxes(frames: Detection[][], opts: SparseOptions & { lastKeyframe?: boolean }): Box[][] {
  const { stride, mode } = opts;
  const n = frames.length;
  const out: Box[][] = Array.from({ length: n }, () => []);
  const keyframes: number[] = [];
  for (let k = 0; k < n; k += stride) keyframes.push(k);
  if (opts.lastKeyframe && keyframes.at(-1) !== n - 1) keyframes.push(n - 1);
  const trackBuffer = Math.max(2, Math.round(opts.fps / stride));
  const minScore = opts.tracker?.trackThresh ?? 0.5;
  // Each keyframe owns the frames up to the next keyframe.
  const until = (i: number) => keyframes[i + 1] ?? n;

  if (mode === 'hold') {
    keyframes.forEach((k, i) => {
      const boxes = frames[k].filter((d) => d.score >= minScore).map((d) => d.box);
      for (let f = k; f < until(i); f++) out[f].push(...boxes);
    });
  } else if (mode === 'online') {
    const tracker = new ByteTracker({ trackBuffer, ...opts.tracker });
    for (const [i, k] of keyframes.entries()) {
      const tracks = tracker.update(frames[k]);
      const vel = tracker.velocities();
      for (let f = k; f < until(i); f++) {
        const s = (f - k) / stride;
        for (const t of tracks) {
          const v = vel.get(t.id) ?? [0, 0, 0, 0];
          out[f].push(t.box.map((b, i) => b + v[i] * s) as Box);
        }
      }
    }
  } else {
    // Offline: collect each track's observed keyframe boxes, then fill frames between them.
    const tracker = new ByteTracker({ trackBuffer, ...opts.tracker });
    const anchors = new Map<number, { frame: number; box: Box }[]>();
    const emit = (f: number, box: Box) => f >= 0 && f < n && out[f].push(box);
    for (const [i, k] of keyframes.entries()) {
      const observed = tracker.update(frames[k]).filter((t) => t.observed);
      // Anchor on the raw detection: at a keyframe it is the truth, while the Kalman
      // estimate lags until the filter has learned the velocity.
      for (const t of observed) {
        let list = anchors.get(t.id);
        if (!list) anchors.set(t.id, (list = []));
        list.push({ frame: k, box: t.detection ?? t.box });
      }
      // Confident detections no confirmed track explains (new or unconfirmed tracks, fast
      // movers that broke association) are held for the stride so nothing is dropped.
      for (const d of frames[k]) {
        if (d.score < minScore || observed.some((t) => iou(t.detection ?? t.box, d.box) >= 0.3)) continue;
        for (let f = k; f < until(i); f++) emit(f, d.box);
      }
    }
    const maxGap = trackBuffer * stride;
    for (const list of anchors.values()) {
      const linked = (i: number, j: number) => i >= 0 && j < list.length && list[j].frame - list[i].frame <= maxGap;
      list.forEach((a, i) => {
        // Segment start: extend backwards one stride, extrapolating along the next anchor's motion.
        if (!linked(i - 1, i)) {
          const ref = linked(i, i + 1) ? list[i + 1] : undefined;
          for (let f = a.frame - stride + 1; f < a.frame; f++) emit(f, ref ? lerp(a.box, ref.box, (f - a.frame) / (ref.frame - a.frame)) : a.box);
        }
        if (linked(i, i + 1)) {
          const next = list[i + 1];
          const span = next.frame - a.frame;
          for (let f = a.frame; f < next.frame; f++) emit(f, lerp(a.box, next.box, (f - a.frame) / span));
        } else {
          // Segment end: extend forwards one stride, extrapolating along the previous anchor's motion.
          const ref = linked(i - 1, i) ? list[i - 1] : undefined;
          for (let f = a.frame; f < a.frame + stride; f++) emit(f, ref ? lerp(ref.box, a.box, (f - ref.frame) / (a.frame - ref.frame)) : a.box);
        }
      });
    }
  }

  const pad = opts.pad ?? 0;
  return pad ? out.map((boxes) => boxes.map((b) => padBox(b, pad))) : out;
}
