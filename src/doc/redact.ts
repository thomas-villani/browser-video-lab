// Redaction, the pure half: which regions to cover on each frame (from the faces layer) and the
// redaction log written next to the output. The browser half (src/redact.ts) decodes, draws
// over these regions and encodes. Node computes the log without decoding anything.
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

/**
 * The regions to cover, frame by frame. Frames must be visited in increasing order (decode
 * order): tracks are swept in by start, so each frame costs the tracks on screen, not all of
 * them (a crowded street has 25k tracks).
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
    if (frame < this.last) throw new Error(`RedactionCursor: frame ${frame} after ${this.last}; frames must not go back`);
    this.last = frame;
    const h = this.o.holdFrames;
    while (this.next < this.order.length && this.order[this.next].f0 - h <= frame) this.active.push(this.order[this.next++]);
    this.active = this.active.filter((t) => t.f1 + h >= frame);
    const live: Box[] = [];
    const held: Box[] = [];
    for (const t of this.active) {
      if (t.f0 - h > frame) continue;
      if ((frame < t.f0 || frame > t.f1) && this.shot(frame) !== this.shot(t.f0)) continue;
      // Held frames reuse the track's first or last box.
      (frame >= t.f0 && frame <= t.f1 ? live : held).push(trackBox(t, Math.min(Math.max(frame, t.f0), t.f1))!);
    }
    // A held box over a live one is the same face picked up by a new track (tracks fragment
    // under motion): it is covered already.
    const extra = held.filter((b) => !live.some((l) => overlap(b, l) > 0.3));
    const out: Box[] = [];
    for (const b of [...live, ...extra]) {
      const g = this.grow(b);
      if (g) out.push(g);
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

  private grow([x, y, w, h]: Box): Box | undefined {
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

export function cursorFor(doc: VideoDoc, o: RedactOptions): RedactionCursor {
  const faces = doc.layers.faces;
  if (!faces) throw new Error('the doc has no faces layer');
  return new RedactionCursor(faces.items, {
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
  totals: { tracks: number; redacted: number; kept: number; framesWithRegions: number; regions: number; maxPerFrame: number };
  caveat: string;
  /** Every face track: kept ones were left visible. Times in seconds. */
  tracks: { id: number; t0: number; t1: number; frames: number; kept: boolean }[];
}

export const REDACTION_CAVEAT =
  'Regions come from automatic face detection, which misses some faces (on a crowded street, ' +
  '8.1% of hand-labeled faces, mostly small ones in the background) and does not cover other ' +
  'identifying details (plates, screens, tattoos, voices). Review the output before release.';

/** The redaction log for `doc` with `o`: totals are computed here by sweeping every frame. */
export function redactionLog(doc: VideoDoc, o: RedactOptions, output?: RedactionLog['output']): RedactionLog {
  const faces = doc.layers.faces;
  if (!faces) throw new Error('the doc has no faces layer');
  const cursor = cursorFor(doc, o);
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
  const tracks = faces.items.map((t) => ({ id: t.id, t0: r3(t.f0 / fps), t1: r3((t.f1 + 1) / fps), frames: t.f1 - t.f0 + 1, kept: keep.has(t.id) }));
  const kept = tracks.filter((t) => t.kept).length;
  const s = doc.source;
  return {
    schema: 'videodoc-redactions/0',
    source: { name: s.name, fingerprint: s.fingerprint, frames: s.frames, fps: s.fps, width: s.width, height: s.height },
    ...(output && { output }),
    detector: { model: faces.producer.model, configHash: faces.producer.configHash, device: faces.device, createdAt: faces.createdAt },
    options: o,
    totals: { tracks: tracks.length, redacted: tracks.length - kept, kept, framesWithRegions, regions, maxPerFrame },
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
