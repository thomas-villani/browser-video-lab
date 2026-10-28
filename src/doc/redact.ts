// Redaction, the pure half: which regions to cover on each frame (from the faces layer and faces
// added by hand), the review plan a person saves (<video>.redact.json) and the redaction log
// written next to the output. The browser half (src/redact.ts) decodes, draws over these
// regions and encodes. Node computes the log without decoding anything.
import type { Box, Track, VideoDoc } from './schema.ts';
import { trackBox } from './tracks.ts';

export type RedactStyle = 'blur' | 'pixelate' | 'fill';
export const REDACT_STYLES: RedactStyle[] = ['blur', 'pixelate', 'fill'];

export interface RedactOptions {
  /** blur: strong Gaussian blur; pixelate: coarse blocks; fill: solid color (nothing left to undo). */
  style: RedactStyle;
  /** Grow each box by this fraction of its size on every side (detector boxes stop at hair and ears). */
  margin: number;
  /** Keep covering this many seconds before a track starts and after it ends: faces turning,
   *  entering or leaving are where the detector loses them for a few frames. */
  hold: number;
  /** Track ids to leave visible (e.g. a public official, after review). */
  keep: number[];
}

export const REDACT_DEFAULTS: RedactOptions = { style: 'blur', margin: 0.15, hold: 0.3, keep: [] };

export interface CursorOptions extends Omit<RedactOptions, 'style' | 'hold'> {
  /** `hold` in frames. */
  holdFrames: number;
  /** First frames of shots (after the first): a hold never crosses a cut. */
  cuts?: number[];
  width: number;
  height: number;
}

/** A region to cover on one frame: the track it comes from (negative ids: added by hand), its
 *  box grown by the margin, and whether the track is only held there (before or after it). */
export interface Region { id: number; box: Box; held: boolean }

/**
 * The regions to cover, frame by frame. Built for frames in increasing order (decode order):
 * tracks are swept in by start, so each frame costs the tracks on screen, not all of them (a
 * crowded street has 25k tracks). Going back (a seek in the review UI) restarts the sweep.
 */
export class RedactionCursor {
  private order: Track[];
  private next = 0;
  private active: Track[] = [];
  private last = -1;
  private o: CursorOptions;

  constructor(tracks: Track[], o: CursorOptions) {
    this.o = o;
    const keep = new Set(o.keep);
    this.order = tracks.filter((t) => !keep.has(t.id)).sort((a, b) => a.f0 - b.f0);
  }

  at(frame: number): Box[] {
    return this.regions(frame).map((r) => r.box);
  }

  regions(frame: number): Region[] {
    if (frame < this.last) {
      this.next = 0;
      this.active = [];
    }
    this.last = frame;
    const h = this.o.holdFrames;
    while (this.next < this.order.length && this.order[this.next].f0 - h <= frame) this.active.push(this.order[this.next++]);
    this.active = this.active.filter((t) => t.f1 + h >= frame);
    const live: Region[] = [];
    const held: Region[] = [];
    for (const t of this.active) {
      if (t.f0 - h > frame) continue;
      const isLive = frame >= t.f0 && frame <= t.f1;
      if (!isLive && this.shot(frame) !== this.shot(t.f0)) continue;
      // Held frames reuse the track's first or last box.
      (isLive ? live : held).push({ id: t.id, box: trackBox(t, Math.min(Math.max(frame, t.f0), t.f1))!, held: !isLive });
    }
    // A held box over a live one is the same face picked up by a new track (tracks fragment
    // under motion): it is covered already.
    const extra = held.filter((b) => !live.some((l) => overlap(b.box, l.box) > 0.3));
    const out: Region[] = [];
    for (const r of [...live, ...extra]) {
      const g = this.grow(r.box);
      if (g) out.push({ ...r, box: g });
    }
    return out;
  }

  /** Index of the shot holding `frame`. */
  private shot(frame: number): number {
    const c = this.o.cuts ?? [];
    let lo = 0;
    let hi = c.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (c[mid] <= frame) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  grow([x, y, w, h]: Box): Box | undefined {
    const m = this.o.margin;
    const x0 = Math.max(0, x - m * w);
    const y0 = Math.max(0, y - m * h);
    const x1 = Math.min(this.o.width, x + w + m * w);
    const y1 = Math.min(this.o.height, y + h + m * h);
    return x1 > x0 && y1 > y0 ? [x0, y0, x1 - x0, y1 - y0] : undefined;
  }
}

/** Intersection over the smaller box's area: 1 when one box holds the other. */
function overlap(a: Box, b: Box): number {
  const w = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  return w > 0 && h > 0 ? (w * h) / Math.min(a[2] * a[3], b[2] * b[3]) : 0;
}

/** The cursor for `doc`'s faces plus faces `added` by hand (a plan's). */
export function cursorFor(doc: VideoDoc, o: RedactOptions, added: Track[] = []): RedactionCursor {
  const faces = doc.layers.faces;
  if (!faces) throw new Error('the doc has no faces layer');
  return new RedactionCursor([...faces.items, ...added], {
    margin: o.margin,
    keep: o.keep,
    holdFrames: Math.round(o.hold * doc.source.fps),
    cuts: (doc.layers.scenes?.items ?? []).slice(1).map((s) => s.f0),
    width: doc.source.width,
    height: doc.source.height,
  });
}

export interface RedactionLog {
  schema: 'videodoc-redactions/0';
  source: { name: string; fingerprint: string; frames: number; fps: number; width: number; height: number };
  output?: { name: string; frames: number; ms: number };
  /** Where the regions came from: the faces layer's producer. */
  detector: { model?: string; configHash: string; device: string; createdAt: string };
  options: RedactOptions;
  totals: { tracks: number; redacted: number; kept: number; added: number; framesWithRegions: number; regions: number; maxPerFrame: number };
  caveat: string;
  /** Every face track: kept ones were left visible; added ones were drawn by hand (negative ids). Times in seconds. */
  tracks: { id: number; t0: number; t1: number; frames: number; kept: boolean; added?: true }[];
}

export const REDACTION_CAVEAT =
  'Regions come from automatic face detection, which misses some faces (on a crowded street, ' +
  '8.1% of hand-labeled faces, mostly small ones in the background) and does not cover other ' +
  'identifying details (plates, screens, tattoos, voices). Review the output before release.';

/** The redaction log for `doc` with `o` (and faces `added` by hand): totals are computed here
 *  by sweeping every frame. */
export function redactionLog(doc: VideoDoc, o: RedactOptions, output?: RedactionLog['output'], added: Track[] = []): RedactionLog {
  const faces = doc.layers.faces;
  if (!faces) throw new Error('the doc has no faces layer');
  const cursor = cursorFor(doc, o, added);
  let framesWithRegions = 0;
  let regions = 0;
  let maxPerFrame = 0;
  for (let f = 0; f < doc.source.frames; f++) {
    const n = cursor.at(f).length;
    if (n) framesWithRegions++;
    regions += n;
    maxPerFrame = Math.max(maxPerFrame, n);
  }
  const keep = new Set(o.keep);
  const fps = doc.source.fps;
  const r3 = (v: number) => Math.round(v * 1000) / 1000;
  const tracks: RedactionLog['tracks'] = faces.items.map((t) => ({ id: t.id, t0: r3(t.f0 / fps), t1: r3((t.f1 + 1) / fps), frames: t.f1 - t.f0 + 1, kept: keep.has(t.id) }));
  for (const t of added) tracks.push({ id: t.id, t0: r3(t.f0 / fps), t1: r3((t.f1 + 1) / fps), frames: t.f1 - t.f0 + 1, kept: false, added: true });
  const kept = tracks.filter((t) => t.kept).length;
  const s = doc.source;
  return {
    schema: 'videodoc-redactions/0',
    source: { name: s.name, fingerprint: s.fingerprint, frames: s.frames, fps: s.fps, width: s.width, height: s.height },
    ...(output && { output }),
    detector: { model: faces.producer.model, configHash: faces.producer.configHash, device: faces.device, createdAt: faces.createdAt },
    options: o,
    totals: { tracks: tracks.length, redacted: tracks.length - kept, kept, added: added.length, framesWithRegions, regions, maxPerFrame },
    caveat: REDACTION_CAVEAT,
    tracks,
  };
}

/** JSON with one track per line (a crowded video has tens of thousands). */
export function formatLog(log: RedactionLog): string {
  const { tracks, ...head } = log;
  const top = JSON.stringify(head, null, 1).replace(/\n}$/, '');
  return `${top},\n "tracks": [\n${tracks.map((t) => `  ${JSON.stringify(t)}`).join(',\n')}\n ]\n}\n`;
}

/** Parse "1,4,7" or [1, 4, 7] into track ids. */
export function parseKeep(v: string | number[] | undefined): number[] {
  if (v === undefined || v === '') return [];
  const ids = (Array.isArray(v) ? v : v.split(',')).map((x) => Number(String(x).trim()));
  const bad = ids.find((x) => !Number.isInteger(x) || x < 1);
  if (bad !== undefined) throw new Error(`keep takes track ids (positive integers), got '${v}'`);
  return ids;
}

// ---- review: the plan a person makes, saved as <video>.redact.json ----

export const planPathFor = (video: string) => `${video}.redact.json`;

/**
 * A person's review of the regions: the options, the tracks to leave visible and the faces the
 * detector missed, drawn by hand. The app writes it next to the video; `videodoc redact` uses
 * it when it is there.
 */
export interface RedactPlan {
  schema: 'videodoc-redact/0';
  source: { name: string; fingerprint: string };
  /** The faces layer's configHash: keep ids name its tracks, and other settings number them differently. */
  faces: string;
  options: RedactOptions;
  /** Faces added by hand: tracks with negative ids, boxes in source pixels, interpolated between samples. */
  added: Track[];
}

export function newPlan(doc: VideoDoc, options: RedactOptions = REDACT_DEFAULTS): RedactPlan {
  const faces = doc.layers.faces;
  if (!faces) throw new Error('the doc has no faces layer');
  return { schema: 'videodoc-redact/0', source: { name: doc.source.name, fingerprint: doc.source.fingerprint }, faces: faces.producer.configHash, options: { ...options, keep: [...options.keep] }, added: [] };
}

/** Check a plan read from disk against `doc`. Keep ids from another faces layer no longer name
 *  the same faces, so they are dropped (with a warning); faces added by hand stay. */
export function checkPlan(v: unknown, doc: VideoDoc): { plan: RedactPlan; warnings: string[] } {
  const p = v as RedactPlan;
  if (p?.schema !== 'videodoc-redact/0') throw new Error(`not a videodoc review plan (schema ${JSON.stringify(p?.schema)})`);
  if (p.source?.fingerprint !== doc.source.fingerprint) throw new Error(`the review plan is for another file (${p.source?.name})`);
  const plan: RedactPlan = { ...newPlan(doc, { ...REDACT_DEFAULTS, ...p.options }), added: Array.isArray(p.added) ? p.added : [] };
  const warnings: string[] = [];
  if (!REDACT_STYLES.includes(plan.options.style)) {
    warnings.push(`unknown style '${plan.options.style}', using ${REDACT_DEFAULTS.style}`);
    plan.options.style = REDACT_DEFAULTS.style;
  }
  if (p.faces !== plan.faces && plan.options.keep.length) {
    warnings.push(`faces were found again with other settings since the review, so its ${plan.options.keep.length} kept face(s) are blurred again; review them again`);
    plan.options.keep = [];
  }
  return { plan, warnings };
}

export const parsePlan = (text: string, doc: VideoDoc) => checkPlan(JSON.parse(text), doc);

/** JSON with one added track per line. */
export function formatPlan(p: RedactPlan): string {
  const { added, ...head } = p;
  const top = JSON.stringify(head, null, 1).replace(/\n}$/, '');
  return `${top},\n "added": [${added.length ? `\n${added.map((t) => `  ${JSON.stringify(t)}`).join(',\n')}\n ` : ''}]\n}\n`;
}

const sample = (frame: number, [x, y, w, h]: Box) => [frame, ...[x, y, w, h].map((v) => Math.round(v * 10) / 10)];

/** A face added by hand: `box` at `frame`, held still until `f1`. */
export function handTrack(id: number, frame: number, box: Box, f1 = frame): Track {
  return { id, f0: frame, f1: Math.max(frame, f1), samples: f1 > frame ? [sample(frame, box), sample(f1, box)] : [sample(frame, box)] };
}

/**
 * Move a hand track's box at `frame` (beyond its ends: extend it). Later samples still holding
 * the box from before this frame were carried forward, so they move too: drawing a face again
 * further along a still box makes it follow the face.
 */
export function setBox(t: Track, frame: number, box: Box): Track {
  const before = trackBox(t, Math.min(Math.max(frame, t.f0), t.f1))!.map((v) => Math.round(v * 10) / 10);
  const same = (x: number[]) => x.slice(1).every((v, i) => v === before[i]);
  const next = sample(frame, box);
  const samples = t.samples.filter((x) => x[0] !== frame).map((x) => (x[0] > frame && same(x) ? [x[0], ...next.slice(1)] : x));
  samples.push(next);
  samples.sort((a, b) => a[0] - b[0]);
  return { id: t.id, f0: samples[0][0], f1: samples[samples.length - 1][0], samples };
}

/** Make a hand track start or end at `frame`, keeping the box it has there. */
export function setEnd(t: Track, frame: number, end: 'start' | 'end'): Track {
  const box = trackBox(t, Math.min(Math.max(frame, t.f0), t.f1))!;
  if (end === 'start' ? frame > t.f1 : frame < t.f0) return handTrack(t.id, frame, box);
  const keep = t.samples.filter((x) => (end === 'start' ? x[0] > frame : x[0] < frame));
  const samples = end === 'start' ? [sample(frame, box), ...keep] : [...keep, sample(frame, box)];
  return { id: t.id, f0: samples[0][0], f1: samples[samples.length - 1][0], samples };
}
