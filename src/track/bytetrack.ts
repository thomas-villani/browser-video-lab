// ByteTrack (Zhang et al., ECCV 2022) in TypeScript — follows the reference
// byte_tracker.py: two-stage association (high-score detections first, then low-score
// detections against the leftover tracks), Kalman motion model, IoU costs.
//
// Pure TypeScript, no dependencies. Boxes are [x, y, w, h] top-left in pixels.

import { linearAssignment } from './assignment.ts';
import { fromXyah, initiate, predict, toXyah, update, type KalmanState } from './kalman.ts';

export type Box = [x: number, y: number, w: number, h: number];

export interface Detection {
  box: Box;
  score: number;
}

export interface ByteTrackOptions {
  /** Detections at or above this score are "high" (first association). */
  trackThresh: number;
  /** Detections above this (and below trackThresh) are "low" (second association). */
  lowThresh: number;
  /** Minimum score to start a new track (reference: trackThresh + 0.1). */
  newTrackThresh: number;
  /** IoU-distance limit (1 - IoU) for the first association. */
  matchThresh: number;
  /** Updates a lost track survives before removal. */
  trackBuffer: number;
  /** Multiply IoU by detection score in the first association (reference: on unless MOT20). */
  fuseScore: boolean;
}

export const DEFAULT_OPTIONS: ByteTrackOptions = {
  trackThresh: 0.5,
  lowThresh: 0.1,
  newTrackThresh: 0.6,
  matchThresh: 0.8,
  trackBuffer: 30,
  fuseScore: true,
};

type TrackState = 'tracked' | 'lost' | 'removed';

export interface TrackOutput {
  id: number;
  box: Box;
  score: number;
  /** True when this update matched a detection (false = coasting on prediction). */
  observed: boolean;
  /** The matched detection's raw box when observed (`box` is the Kalman-smoothed estimate). */
  detection?: Box;
}

class STrack {
  id = 0;
  state: TrackState = 'tracked';
  activated = false;
  kf?: KalmanState;
  box: Box;
  score: number;
  startFrame = 0;
  frame = 0;
  observed = false;

  constructor(det: Detection) {
    this.box = det.box;
    this.score = det.score;
  }

  /** Current box: the Kalman mean if initialized, else the detection box. */
  get tlwh(): Box {
    return this.kf ? (fromXyah(this.kf.mean.slice(0, 4)) as Box) : this.box;
  }

  /** Per-update velocity of [x, y, w, h] derived from the Kalman state. */
  get velocity(): Box {
    if (!this.kf) return [0, 0, 0, 0];
    const now = this.tlwh;
    const next = fromXyah(this.kf.mean.slice(0, 4).map((v, i) => v + this.kf!.mean[i + 4]));
    return [next[0] - now[0], next[1] - now[1], next[2] - now[2], next[3] - now[3]];
  }

  predict(): void {
    if (!this.kf) return;
    const mean = [...this.kf.mean];
    // Reference: a non-tracked track's height velocity is zeroed before predicting.
    if (this.state !== 'tracked') mean[7] = 0;
    this.kf = predict({ mean, cov: this.kf.cov });
  }

  activate(id: number, frame: number): void {
    this.id = id;
    this.kf = initiate(toXyah(this.box));
    this.state = 'tracked';
    // Only tracks born on the very first frame are trusted immediately.
    this.activated = frame === 1;
    this.frame = this.startFrame = frame;
    this.observed = true;
  }

  updateWith(det: Detection, frame: number): void {
    this.kf = update(this.kf!, toXyah(det.box));
    this.box = det.box;
    this.score = det.score;
    this.state = 'tracked';
    this.activated = true;
    this.frame = frame;
    this.observed = true;
  }
}

function iou(a: Box, b: Box): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]);
  const y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

function iouDistance(tracks: STrack[], dets: Detection[], fuse: boolean): number[][] {
  return tracks.map((t) => {
    const tb = t.tlwh;
    return dets.map((d) => 1 - iou(tb, d.box) * (fuse ? d.score : 1));
  });
}

export class ByteTracker {
  readonly options: ByteTrackOptions;
  private tracked: STrack[] = [];
  private lost: STrack[] = [];
  private frameId = 0;
  private nextId = 1;

  constructor(options: Partial<ByteTrackOptions> = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  /** Feed one frame's detections; returns the active (confirmed) tracks. */
  update(detections: Detection[]): TrackOutput[] {
    const o = this.options;
    const frame = ++this.frameId;
    const high = detections.filter((d) => d.score >= o.trackThresh).map((d) => new STrack(d));
    const low = detections
      .filter((d) => d.score > o.lowThresh && d.score < o.trackThresh)
      .map((d) => new STrack(d));

    for (const t of [...this.tracked, ...this.lost]) t.observed = false;
    const unconfirmed = this.tracked.filter((t) => !t.activated);
    const confirmed = this.tracked.filter((t) => t.activated);
    const pool = [...confirmed, ...this.lost];
    for (const t of pool) t.predict();

    const activated: STrack[] = [];
    const refound: STrack[] = [];
    const lostNow: STrack[] = [];
    const removed: STrack[] = [];
    const absorb = (t: STrack, d: STrack) => {
      const wasTracked = t.state === 'tracked';
      t.updateWith(d, frame);
      (wasTracked ? activated : refound).push(t);
    };

    // 1) High-score detections vs all confirmed + lost tracks.
    const a1 = linearAssignment(iouDistance(pool, high, o.fuseScore), o.matchThresh, high.length);
    for (const [ti, di] of a1.matches) absorb(pool[ti], high[di]);

    // 2) Low-score detections vs the still-tracked leftovers (the "BYTE" step).
    const leftovers = a1.unmatchedRows.map((i) => pool[i]).filter((t) => t.state === 'tracked');
    const a2 = linearAssignment(iouDistance(leftovers, low, false), 0.5, low.length);
    for (const [ti, di] of a2.matches) absorb(leftovers[ti], low[di]);
    for (const i of a2.unmatchedRows) {
      const t = leftovers[i];
      if (t.state !== 'lost') {
        t.state = 'lost';
        lostNow.push(t);
      }
    }

    // 3) Unconfirmed (one-frame-old) tracks vs remaining high detections.
    const remaining = a1.unmatchedCols.map((i) => high[i]);
    const a3 = linearAssignment(iouDistance(unconfirmed, remaining, o.fuseScore), 0.7, remaining.length);
    for (const [ti, di] of a3.matches) {
      unconfirmed[ti].updateWith(remaining[di], frame);
      activated.push(unconfirmed[ti]);
    }
    for (const i of a3.unmatchedRows) {
      unconfirmed[i].state = 'removed';
      removed.push(unconfirmed[i]);
    }

    // 4) Start new tracks from confident unmatched detections.
    for (const i of a3.unmatchedCols) {
      const t = remaining[i];
      if (t.score < o.newTrackThresh) continue;
      t.activate(this.nextId++, frame);
      activated.push(t);
    }

    // 5) Expire tracks lost for too long.
    for (const t of this.lost) {
      if (t.state === 'lost' && frame - t.frame > o.trackBuffer) {
        t.state = 'removed';
        removed.push(t);
      }
    }

    const tracked = [...this.tracked.filter((t) => t.state === 'tracked'), ...activated, ...refound];
    this.tracked = [...new Set(tracked)];
    const trackedSet = new Set(this.tracked);
    const removedSet = new Set(removed);
    this.lost = [...new Set([...this.lost, ...lostNow])].filter(
      (t) => t.state === 'lost' && !trackedSet.has(t) && !removedSet.has(t),
    );
    this.dedupe();

    return this.tracked
      .filter((t) => t.activated)
      .map((t) => ({
        id: t.id,
        box: t.tlwh,
        score: t.score,
        observed: t.observed,
        ...(t.observed && { detection: t.box }),
      }));
  }

  /** Lost tracks with their predicted boxes (useful for redaction, which prefers over-coverage). */
  lostTracks(): TrackOutput[] {
    return this.lost.map((t) => ({ id: t.id, box: t.tlwh, score: t.score, observed: false }));
  }

  /** Per-update velocity for each active track id (for sub-step extrapolation between keyframes). */
  velocities(): Map<number, Box> {
    return new Map(this.tracked.filter((t) => t.activated).map((t) => [t.id, t.velocity]));
  }

  /** Reference remove_duplicate_stracks: a tracked and a lost track overlapping heavily → keep the older one. */
  private dedupe(): void {
    const dropTracked = new Set<STrack>();
    const dropLost = new Set<STrack>();
    for (const a of this.tracked)
      for (const b of this.lost) {
        if (1 - iou(a.tlwh, b.tlwh) >= 0.15) continue;
        const ageA = a.frame - a.startFrame;
        const ageB = b.frame - b.startFrame;
        if (ageA > ageB) dropLost.add(b);
        else dropTracked.add(a);
      }
    this.tracked = this.tracked.filter((t) => !dropTracked.has(t));
    this.lost = this.lost.filter((t) => !dropLost.has(t));
  }
}
