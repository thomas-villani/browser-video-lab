// Speaker diarization, the pure part: who speaks when, from what the models said about each
// 10 s window. Follows pyannote.audio 3.1's pipeline (speaker-diarization-3.1), so its measured
// behavior carries over (NOTES.md):
//
// 1. Segmentation (pyannote segmentation-3.0) gives, per window, up to 3 local speakers per frame
//    (a powerset over {none, 1, 2, 3, 1+2, 1+3, 2+3}, decoded to 3 binary tracks).
// 2. Each active local speaker gets a voice embedding (WeSpeaker ResNet34) from the frames where
//    it speaks alone (all its frames when too few are clean).
// 3. The embeddings are clustered (agglomerative, centroid linkage on unit vectors, a distance
//    threshold; clusters smaller than minClusterSize join the nearest large one).
// 4. Each window's local speakers are assigned to distinct clusters (best total similarity).
// 5. Each frame keeps its most active clusters, as many as the windows say are speaking there.
//
// The browser part (models, features) is src/runtime/diarize.ts. Pure, Node-testable.

/** Segmentation model geometry at 16 kHz: 10 s windows, a frame every 270 samples, each
 *  covering 990 (its receptive field). */
export const SEG = { rate: 16000, window: 160000, step: 270, size: 990, frames: 589, speakers: 3 } as const;
/** Seconds between frames, and a frame's center from its start. */
export const FRAME_STEP = SEG.step / SEG.rate;
const FRAME_MID = SEG.size / SEG.rate / 2;

/** Powerset classes to local speakers: 0 none, 1-3 alone, 4-6 two at once. */
const POWERSET: number[][] = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]];

/** One window's segmentation: where it starts (seconds from the audio's start) and, per frame,
 *  which of its 3 local speakers are active (act[frame * 3 + s] = 0 or 1). */
export interface WindowSeg {
  start: number;
  act: Uint8Array;
}

export interface DiarizeOptions {
  /** Cluster distance threshold (Euclidean between unit vectors' centroids). */
  threshold: number;
  /** Clusters with fewer embeddings join the nearest larger one. */
  minClusterSize: number;
  /** At most this many speakers (0: as many as the threshold gives). */
  maxSpeakers: number;
}

/** pyannote/speaker-diarization-3.1's tuned values (for the same two models). */
export const DIARIZE_DEFAULTS: DiarizeOptions = { threshold: 0.7045654963945799, minClusterSize: 12, maxSpeakers: 0 };

/** At most this many voices are clustered (memory: their distance matrix, 4 bytes per pair). */
export const MAX_CLUSTERED = 4000;

/** A speaker found: turns (seconds, from the audio's start) and the voice (centroid, unit length). */
export interface Speaker {
  id: number;
  turns: [number, number][];
  voice: Float32Array;
}

/** Window start times (seconds) for `seconds` of audio at `step`, as pyannote slides them: whole
 *  windows, then one more (zero-padded) for the rest. */
export function windowStarts(seconds: number, step: number): number[] {
  const n = Math.round(seconds * SEG.rate);
  const win = SEG.window;
  const hop = Math.round(step * SEG.rate);
  const whole = n >= win ? Math.floor((n - win) / hop) + 1 : 0;
  const starts = Array.from({ length: whole }, (_, i) => (i * hop) / SEG.rate);
  if (n < win || (n - win) % hop > 0) starts.push((whole * hop) / SEG.rate);
  return starts;
}

/** Decode the model's logits [frames, 7] for one window into 3 binary local-speaker tracks
 *  (the most likely powerset class per frame). */
export function decodePowerset(logits: Float32Array, frames: number = SEG.frames): Uint8Array {
  const act = new Uint8Array(frames * SEG.speakers);
  for (let f = 0; f < frames; f++) {
    let best = 0;
    for (let c = 1; c < 7; c++) if (logits[f * 7 + c] > logits[f * 7 + best]) best = c;
    for (const s of POWERSET[best]) act[f * SEG.speakers + s] = 1;
  }
  return act;
}

/** Frames where local speaker s is active, and where it is the only one (for its embedding). */
export function speakerMask(act: Uint8Array, s: number): { mask: Uint8Array; clean: Uint8Array; active: number; alone: number } {
  const frames = act.length / SEG.speakers;
  const mask = new Uint8Array(frames);
  const clean = new Uint8Array(frames);
  let active = 0;
  let alone = 0;
  for (let f = 0; f < frames; f++) {
    if (!act[f * SEG.speakers + s]) continue;
    mask[f] = 1;
    active++;
    let n = 0;
    for (let k = 0; k < SEG.speakers; k++) n += act[f * SEG.speakers + k];
    if (n === 1) {
      clean[f] = 1;
      alone++;
    }
  }
  return { mask, clean, active, alone };
}

/** The mask an embedding is computed over: the clean frames when there are enough of them
 *  (pyannote: more than the embedding model's minimum input), else every active frame. */
export function embeddingMask(act: Uint8Array, s: number, minFrames: number): Uint8Array | undefined {
  const m = speakerMask(act, s);
  if (!m.active) return undefined;
  return m.alone > minFrames ? m.clean : m.mask;
}

/** Global frame index of window frame 0. */
const offsetOf = (w: WindowSeg) => Math.round(w.start / FRAME_STEP);

/** Instantaneous speaker count per global frame: the windows covering it, averaged, rounded. */
export function speakerCount(windows: WindowSeg[]): Uint8Array {
  const total = windows.length ? Math.max(...windows.map((w) => offsetOf(w) + w.act.length / SEG.speakers)) : 0;
  const sum = new Float32Array(total);
  const n = new Uint16Array(total);
  for (const w of windows) {
    const o = offsetOf(w);
    const frames = w.act.length / SEG.speakers;
    for (let f = 0; f < frames; f++) {
      let c = 0;
      for (let s = 0; s < SEG.speakers; s++) c += w.act[f * SEG.speakers + s];
      sum[o + f] += c;
      n[o + f]++;
    }
  }
  const count = new Uint8Array(total);
  for (let i = 0; i < total; i++) count[i] = n[i] ? Math.round(sum[i] / n[i]) : 0;
  return count;
}

const dot = (a: Float32Array, b: Float32Array) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};
const unit = (a: Float32Array) => {
  const n = Math.sqrt(dot(a, a)) || 1;
  return a.map((v) => v / n);
};
const cosine = (a: Float32Array, b: Float32Array) => 1 - dot(a, b) / (Math.sqrt(dot(a, a) * dot(b, b)) || 1);

/**
 * Agglomerative clustering with centroid linkage (scipy's `linkage(method='centroid')` +
 * `fcluster(criterion='distance')` on unit vectors): merge the two clusters whose centroids are
 * closest until the closest pair is farther apart than `threshold`, or (maxClusters) until at
 * most that many large clusters remain. Then clusters under `minClusterSize` join the large
 * cluster with the most similar centroid. Returns a cluster index per embedding, 0-based.
 */
export function cluster(embeddings: Float32Array[], o: DiarizeOptions): Int32Array {
  const n = embeddings.length;
  const labels = new Int32Array(n);
  if (n < 2) return labels;
  const dim = embeddings[0].length;
  const minSize = Math.min(o.minClusterSize, Math.max(1, Math.round(0.1 * n)));
  // Squared distances between cluster centroids (unit vectors to start: 2 - 2 cos), updated by
  // the Lance-Williams formula for centroid linkage as clusters merge; each cluster's nearest.
  const vecs = embeddings.map(unit);
  const sums = vecs.map((v) => Float32Array.from(v));
  const D = new Float32Array(n * n);
  for (let a = 0; a < n; a++)
    for (let b = a + 1; b < n; b++) {
      let s = 0;
      for (let i = 0; i < dim; i++) s += vecs[a][i] * vecs[b][i];
      D[a * n + b] = D[b * n + a] = Math.max(0, 2 - 2 * s);
    }
  const size = new Int32Array(n).fill(1);
  const alive = new Uint8Array(n).fill(1);
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const nn = new Int32Array(n).fill(-1);
  const nd = new Float64Array(n).fill(Infinity);
  const nearest = (a: number) => {
    nn[a] = -1;
    nd[a] = Infinity;
    for (let b = 0; b < n; b++) {
      if (b === a || !alive[b]) continue;
      const d = D[a * n + b];
      if (d < nd[a]) {
        nd[a] = d;
        nn[a] = b;
      }
    }
  };
  for (let a = 0; a < n; a++) nearest(a);
  const t2 = o.threshold * o.threshold;
  let clusters = n;
  const large = () => {
    let k = 0;
    for (let a = 0; a < n; a++) if (alive[a] && size[a] >= minSize) k++;
    return k;
  };
  while (clusters > 1) {
    let a = -1;
    for (let i = 0; i < n; i++) if (alive[i] && nn[i] >= 0 && (a < 0 || nd[i] < nd[a])) a = i;
    if (a < 0) break;
    // Past the threshold, keep merging only to get down to maxSpeakers.
    if (nd[a] > t2 && !(o.maxSpeakers > 0 && large() > o.maxSpeakers)) break;
    const b = nn[a];
    // Merge b into a: d(k, a+b)^2 = (na d(k,a)^2 + nb d(k,b)^2) / n - na nb d(a,b)^2 / n^2.
    const na = size[a];
    const nb = size[b];
    const m = na + nb;
    const dab = D[a * n + b];
    for (let k = 0; k < n; k++) {
      if (!alive[k] || k === a || k === b) continue;
      const d = Math.max(0, (na * D[k * n + a] + nb * D[k * n + b]) / m - (na * nb * dab) / (m * m));
      D[k * n + a] = D[a * n + k] = d;
    }
    for (let i = 0; i < dim; i++) sums[a][i] += sums[b][i];
    size[a] = m;
    alive[b] = 0;
    parent[b] = a;
    clusters--;
    nearest(a);
    for (let k = 0; k < n; k++) {
      if (!alive[k] || k === a) continue;
      if (nn[k] === a || nn[k] === b) nearest(k);
      else if (D[k * n + a] < nd[k]) {
        nd[k] = D[k * n + a];
        nn[k] = a;
      }
    }
  }
  const root = (i: number): number => (parent[i] === i ? i : (parent[i] = root(parent[i])));
  const roots = [...new Set(Array.from({ length: n }, (_, i) => root(i)))];
  const bigs = roots.filter((r) => size[r] >= minSize);
  const target = new Map<number, number>();
  if (!bigs.length) return labels;
  // Small clusters join the most similar large one (cosine between centroids).
  for (const r of roots) {
    if (size[r] >= minSize) target.set(r, r);
    else {
      let best = bigs[0];
      for (const g of bigs) if (cosine(sums[r], sums[g]) < cosine(sums[r], sums[best])) best = g;
      target.set(r, best);
    }
  }
  const index = new Map(bigs.map((r, i) => [r, i]));
  for (let i = 0; i < n; i++) labels[i] = index.get(target.get(root(i))!)!;
  return labels;
}

/** Best assignment of rows to distinct columns, maximizing the total score (rows beyond the
 *  column count stay unassigned: -1). For the small matrices here (3 local speakers). */
export function assignMax(score: number[][]): number[] {
  const rows = score.length;
  const cols = score[0]?.length ?? 0;
  const best = { total: -Infinity, pick: new Array<number>(rows).fill(-1) };
  const pick = new Array<number>(rows).fill(-1);
  const used = new Uint8Array(cols);
  const free = rows - Math.min(rows, cols);
  const go = (r: number, total: number, skipped: number) => {
    if (r === rows) {
      if (total > best.total) {
        best.total = total;
        best.pick = [...pick];
      }
      return;
    }
    for (let c = 0; c < cols; c++) {
      if (used[c]) continue;
      used[c] = 1;
      pick[r] = c;
      go(r + 1, total + score[r][c], skipped);
      used[c] = 0;
    }
    if (skipped < free) {
      pick[r] = -1;
      go(r + 1, total, skipped + 1);
    }
  };
  go(0, 0, 0);
  return best.pick;
}

/**
 * Who speaks when. `windows` are the segmented windows (in time order), `embeddings[w][s]` the
 * voice of window w's local speaker s (undefined when inactive or too short). Returns speakers in
 * order of first appearance, with turns in seconds from the audio's start.
 */
export function diarize(windows: WindowSeg[], embeddings: (Float32Array | undefined)[][], o: DiarizeOptions = DIARIZE_DEFAULTS): Speaker[] {
  const train: { w: number; s: number }[] = [];
  for (let w = 0; w < windows.length; w++) for (let s = 0; s < SEG.speakers; s++) if (embeddings[w][s]) train.push({ w, s });
  if (!train.length) return [];
  // Long videos: an even sample is clustered (every window is still assigned to the clusters).
  const sample = train.length > MAX_CLUSTERED ? Array.from({ length: MAX_CLUSTERED }, (_, i) => train[Math.floor((i * train.length) / MAX_CLUSTERED)]) : train;
  const vecs = sample.map(({ w, s }) => embeddings[w][s]!);
  const labels = o.maxSpeakers === 1 ? new Int32Array(vecs.length) : cluster(vecs, o);
  const k = Math.max(...labels) + 1;
  const dim = vecs[0].length;
  const centroids = Array.from({ length: k }, () => new Float32Array(dim));
  const counts = new Int32Array(k);
  labels.forEach((l, i) => {
    counts[l]++;
    for (let d = 0; d < dim; d++) centroids[l][d] += vecs[i][d];
  });
  centroids.forEach((c, l) => c.forEach((_, d) => (c[d] /= counts[l])));

  // Each window's local speakers to distinct clusters (similarity 2 - cosine distance; speakers
  // without an embedding score the lowest similarity seen, so they take what is left).
  const sims = embeddings.map((e) => e.map((v) => (v ? centroids.map((c) => 2 - cosine(v, c)) : undefined)));
  let low = Infinity;
  for (const w of sims) for (const s of w) if (s) for (const v of s) low = Math.min(low, v);
  const hard = windows.map((win, w) => {
    const pick = assignMax(Array.from({ length: SEG.speakers }, (_, s) => sims[w][s] ?? new Array(k).fill(low)));
    // Speakers silent in the window are not assigned.
    return pick.map((c, s) => (speakerMask(win.act, s).active ? c : -1));
  });

  // Per global frame: each cluster's activity summed over the windows, and how many speak.
  const count = speakerCount(windows);
  const total = count.length;
  const activity = new Float32Array(total * k);
  windows.forEach((win, w) => {
    const off = offsetOf(win);
    const frames = win.act.length / SEG.speakers;
    for (let c = 0; c < k; c++) {
      const locals = hard[w].flatMap((h, s) => (h === c ? [s] : []));
      if (!locals.length) continue;
      for (let f = 0; f < frames; f++) if (locals.some((s) => win.act[f * SEG.speakers + s])) activity[(off + f) * k + c] += 1;
    }
  });
  const on = new Uint8Array(total * k);
  const order = Array.from({ length: k }, (_, c) => c);
  for (let t = 0; t < total; t++) {
    if (!count[t]) continue;
    const ranked = order.filter((c) => activity[t * k + c] > 0).sort((a, b) => activity[t * k + b] - activity[t * k + a]);
    for (const c of ranked.slice(0, count[t])) on[t * k + c] = 1;
  }

  // Runs of frames to turns: from the middle of the first frame to the middle of the frame after.
  const time = (i: number) => Math.round((i * FRAME_STEP + FRAME_MID) * 1000) / 1000;
  const turns: [number, number][][] = Array.from({ length: k }, () => []);
  for (let c = 0; c < k; c++) {
    let from = -1;
    for (let t = 0; t <= total; t++) {
      const v = t < total && on[t * k + c];
      if (v && from < 0) from = t;
      else if (!v && from >= 0) {
        turns[c].push([time(from), time(t)]);
        from = -1;
      }
    }
  }
  return order
    .filter((c) => turns[c].length)
    .sort((a, b) => turns[a][0][0] - turns[b][0][0])
    .map((c, i) => ({ id: i + 1, turns: turns[c], voice: unit(centroids[c]) }));
}
