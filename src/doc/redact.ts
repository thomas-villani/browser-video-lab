// Redaction, the pure half: which regions to cover on each frame (from the faces and plates layers
// and regions added by hand), which stretches of audio to bleep (words.ts), the review plan a
// person saves (<video>.redact.json) and the redaction log written next to the output. The
// browser half (src/redact.ts) decodes, draws over these regions, rewrites the audio and
// encodes. Node computes the log without decoding anything.
import type { Box, Track, VideoDoc } from './schema.ts';
import { trackBox } from './tracks.ts';
import { findWords, mergeTimes, PROFANITY, type WordMatch } from './words.ts';

export type RedactStyle = 'blur' | 'pixelate' | 'fill';
export const REDACT_STYLES: RedactStyle[] = ['blur', 'pixelate', 'fill'];
/** Layers of tracks that redaction covers. */
export type RedactKind = 'faces' | 'plates';
export const REDACT_KINDS: RedactKind[] = ['faces', 'plates'];
/** bleep: a 1 kHz tone over the word; mute: silence. */
export type RedactSound = 'bleep' | 'mute';
export const REDACT_SOUNDS: RedactSound[] = ['bleep', 'mute'];

export interface RedactOptions {
  /** blur: strong Gaussian blur; pixelate: coarse blocks; fill: solid color (nothing left to undo). */
  style: RedactStyle;
  /** Grow each box by this fraction of its size on every side (detector boxes stop at hair and ears). */
  margin: number;
  /** Keep covering this many seconds before a track starts and after it ends: faces turning,
   *  entering or leaving are where the detector loses them for a few frames. */
  hold: number;
  /** Face track ids to leave visible (e.g. a public official, after review). */
  keep: number[];
  /** Plate track ids to leave visible (a sign the detector took for a plate, a plate meant to be seen). */
  keepPlates: number[];
  /** Which layers to cover, each when the doc has it (faces is required when listed). None:
   *  the video is copied as it is (audio redaction only). */
  cover: RedactKind[];
  /** Words and phrases to bleep wherever they are spoken (names, places, numbers): see
   *  words.ts for inflections, `*` wildcards and "exact" quotes. Needs word timestamps. */
  words: string[];
  /** Also bleep the built-in strong-language list (PROFANITY in words.ts). */
  profanity: boolean;
  sound: RedactSound;
  /** Seconds covered before each word and after it. Whisper's word starts run 0.1-0.4 s late
   *  (NOTES.md), so the lead is the longer. */
  lead: number;
  tail: number;
  /** Matches to leave audible, by the key of their first word ("<transcript item>.<word>"). */
  keepWords: string[];
}

export const REDACT_DEFAULTS: RedactOptions = {
  style: 'blur', margin: 0.15, hold: 0.3, keep: [], keepPlates: [], cover: ['faces', 'plates'],
  words: [], profanity: false, sound: 'bleep', lead: 0.4, tail: 0.2, keepWords: [],
};

/** A track to cover and the layer it comes from (hand-drawn tracks count as faces). */
export interface CoverTrack { track: Track; kind: RedactKind }

export interface CursorOptions {
  margin: number;
  /** `hold` in frames. */
  holdFrames: number;
  /** First frames of shots (after the first): a hold never crosses a cut. */
  cuts?: number[];
  width: number;
  height: number;
}

/** A region to cover on one frame: the track it comes from (negative ids: added by hand) and its
 *  layer, its box grown by the margin, and whether the track is only held there (before or after it). */
export interface Region { id: number; kind: RedactKind; box: Box; held: boolean }

/**
 * The regions to cover, frame by frame. Built for frames in increasing order (decode order):
 * tracks are swept in by start, so each frame costs the tracks on screen, not all of them (a
 * crowded street has 25k tracks). Going back (a seek in the review UI) restarts the sweep.
 */
export class RedactionCursor {
  private order: CoverTrack[];
  private next = 0;
  private active: CoverTrack[] = [];
  private last = -1;
  private o: CursorOptions;

  /** Every track given is covered (cursorFor leaves kept ones out); plain tracks are faces. */
  constructor(tracks: (Track | CoverTrack)[], o: CursorOptions) {
    this.o = o;
    this.order = tracks.map((t) => ('track' in t ? t : { track: t, kind: 'faces' as const })).sort((a, b) => a.track.f0 - b.track.f0);
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
    while (this.next < this.order.length && this.order[this.next].track.f0 - h <= frame) this.active.push(this.order[this.next++]);
    this.active = this.active.filter((c) => c.track.f1 + h >= frame);
    const live: Region[] = [];
    const held: Region[] = [];
    for (const { track: t, kind } of this.active) {
      if (t.f0 - h > frame) continue;
      const isLive = frame >= t.f0 && frame <= t.f1;
      if (!isLive && this.shot(frame) !== this.shot(t.f0)) continue;
      // Held frames reuse the track's first or last box.
      (isLive ? live : held).push({ id: t.id, kind, box: trackBox(t, Math.min(Math.max(frame, t.f0), t.f1))!, held: !isLive });
    }
    // A held box over a live one of its kind is the same face (or plate) picked up by a new
    // track (tracks fragment under motion): it is covered already.
    const extra = held.filter((b) => !live.some((l) => l.kind === b.kind && overlap(b.box, l.box) > 0.3));
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

/** The layers `o` covers that `doc` has (faces is required when covered). */
export function covered(doc: VideoDoc, o: Pick<RedactOptions, 'cover'>): RedactKind[] {
  if (o.cover.includes('faces') && !doc.layers.faces) throw new Error('the doc has no faces layer');
  return REDACT_KINDS.filter((k) => o.cover.includes(k) && doc.layers[k]);
}

/** Whether `o` (and spans `added` by hand) redact any audio. */
export const coversAudio = (o: Pick<RedactOptions, 'words' | 'profanity'>, added: [number, number][] = []) =>
  o.words.some((w) => w.trim()) || o.profanity || added.length > 0;

/** A stretch of audio to bleep: a list match (with its words) or a span added by hand. */
export interface AudioSpan {
  /** Seconds, with lead and tail. */
  t0: number;
  t1: number;
  /** The match's first word ("<item>.<word>"); absent for spans added by hand. */
  key?: string;
  text?: string;
  term?: string;
  kept: boolean;
  added?: true;
}

/**
 * Every match of the word lists in `doc`'s transcript, and the spans `added` by hand, padded
 * by lead and tail and clipped to the video. Kept matches are listed (kept: true) but not
 * covered; `audioCover` gives the merged stretches to rewrite.
 */
export function audioSpans(doc: VideoDoc, o: RedactOptions, added: [number, number][] = []): AudioSpan[] {
  const r3 = (v: number) => Math.round(v * 1000) / 1000;
  const clip = (a: number, b: number) => ({ t0: r3(Math.max(0, a)), t1: r3(Math.min(doc.source.duration, b)) });
  const out: AudioSpan[] = [];
  const terms = [...o.words.filter((w) => w.trim()).map((term) => ({ term })), ...(o.profanity ? PROFANITY.map((term) => ({ term, label: 'profanity' })) : [])];
  if (terms.length) {
    const t = doc.layers.transcript;
    if (!t) throw new Error('the doc has no transcript layer (audio redaction needs one, with words)');
    if (t.items.length && !t.items.some((i) => i.words)) throw new Error('the transcript has no word timestamps; analyze again with transcript.words=true');
    const keep = new Set(o.keepWords);
    const pad = (m: WordMatch) => clip(m.t0 - o.lead, m.t1 + o.tail);
    for (const m of findWords(doc, terms)) out.push({ ...pad(m), key: m.key, text: m.text, term: m.term, kept: keep.has(m.key) });
  }
  for (const [a, b] of added) out.push({ ...clip(a, b), kept: false, added: true });
  return out.sort((a, b) => a.t0 - b.t0);
}

/** The stretches of audio to rewrite: covered spans, merged. */
export const audioCover = (spans: AudioSpan[]): [number, number][] => mergeTimes(spans.filter((s) => !s.kept && s.t1 > s.t0).map((s) => [s.t0, s.t1]));

/** The tracks to cover: the covered layers' tracks less the kept ones, plus tracks `added` by hand. */
export function coverTracks(doc: VideoDoc, o: RedactOptions, added: Track[] = []): CoverTrack[] {
  const out: CoverTrack[] = [];
  for (const kind of covered(doc, o)) {
    const keep = new Set(kind === 'faces' ? o.keep : o.keepPlates);
    for (const track of doc.layers[kind]!.items) if (!keep.has(track.id)) out.push({ track, kind });
  }
  for (const track of added) out.push({ track, kind: 'faces' });
  return out;
}

/** The cursor for `doc`'s covered layers plus regions `added` by hand (a plan's). */
export function cursorFor(doc: VideoDoc, o: RedactOptions, added: Track[] = []): RedactionCursor {
  return new RedactionCursor(coverTracks(doc, o, added), {
    margin: o.margin,
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
  /** Where the regions came from: each covered layer's producer (and the transcript's, for audio). */
  detectors: Partial<Record<RedactKind | 'transcript', { model?: string; configHash: string; device: string; createdAt: string }>>;
  options: RedactOptions;
  /** `plates`: how many of `tracks` are plate tracks. */
  totals: { tracks: number; redacted: number; kept: number; added: number; plates: number; framesWithRegions: number; regions: number; maxPerFrame: number };
  caveat: string;
  /** Every track of the covered layers (`layer` says which; ids are per layer): kept ones were
   *  left visible; added ones were drawn by hand (negative ids). Times in seconds. */
  tracks: { id: number; layer: RedactKind; t0: number; t1: number; frames: number; kept: boolean; added?: true }[];
  /** Audio redaction, when any: every match (kept ones left audible) and span added by hand,
   *  with lead and tail; `seconds` is the audio rewritten (overlaps counted once). */
  audio?: { sound: RedactSound; totals: { matches: number; kept: number; added: number; seconds: number }; spans: AudioSpan[] };
}

export const REDACTION_CAVEAT =
  'Regions come from automatic detection, which misses some faces (on a crowded street, 8.1% of ' +
  'hand-labeled faces, mostly small ones in the background) and some plates (angled, blurred or ' +
  'distant ones), and does not cover other identifying details (screens, tattoos, voices). ' +
  'Review the output before release.';

export const AUDIO_CAVEAT =
  'Bleeped words come from automatic transcription, which misspells some words (names most of ' +
  'all) and misses some, and from word timestamps that are off by up to about half a second. ' +
  'Listen to the output before release.';

/** The redaction log for `doc` with `o` (and faces `added` by hand): totals are computed here
 *  by sweeping every frame. */
export function redactionLog(doc: VideoDoc, o: RedactOptions, output?: RedactionLog['output'], added: Track[] = [], addedAudio: [number, number][] = []): RedactionLog {
  const kinds = covered(doc, o);
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
  const fps = doc.source.fps;
  const r3 = (v: number) => Math.round(v * 1000) / 1000;
  const entry = (t: Track, layer: RedactKind, kept: boolean) => ({ id: t.id, layer, t0: r3(t.f0 / fps), t1: r3((t.f1 + 1) / fps), frames: t.f1 - t.f0 + 1, kept });
  const tracks: RedactionLog['tracks'] = [];
  const detectors: RedactionLog['detectors'] = {};
  for (const kind of kinds) {
    const l = doc.layers[kind]!;
    const keep = new Set(kind === 'faces' ? o.keep : o.keepPlates);
    for (const t of l.items) tracks.push(entry(t, kind, keep.has(t.id)));
    detectors[kind] = { model: l.producer.model, configHash: l.producer.configHash, device: l.device, createdAt: l.createdAt };
  }
  for (const t of added) tracks.push({ ...entry(t, 'faces', false), added: true });
  const kept = tracks.filter((t) => t.kept).length;
  const plates = tracks.filter((t) => t.layer === 'plates').length;
  let audio: RedactionLog['audio'];
  if (coversAudio(o, addedAudio)) {
    const spans = audioSpans(doc, o, addedAudio);
    const t = doc.layers.transcript;
    if (t && spans.some((x) => !x.added)) detectors.transcript = { model: t.producer.model, configHash: t.producer.configHash, device: t.device, createdAt: t.createdAt };
    const seconds = audioCover(spans).reduce((n, [a, b]) => n + b - a, 0);
    const matches = spans.filter((x) => !x.added);
    audio = { sound: o.sound, totals: { matches: matches.length, kept: matches.filter((x) => x.kept).length, added: addedAudio.length, seconds: Math.round(seconds * 100) / 100 }, spans };
  }
  const s = doc.source;
  return {
    schema: 'videodoc-redactions/0',
    source: { name: s.name, fingerprint: s.fingerprint, frames: s.frames, fps: s.fps, width: s.width, height: s.height },
    ...(output && { output }),
    detectors,
    options: o,
    totals: { tracks: tracks.length, redacted: tracks.length - kept, kept, added: added.length, plates, framesWithRegions, regions, maxPerFrame },
    caveat: [...(kinds.length || added.length || !audio ? [REDACTION_CAVEAT] : []), ...(audio ? [AUDIO_CAVEAT] : [])].join(' '),
    tracks,
    ...(audio && { audio }),
  };
}

/** JSON with one track (and one audio span) per line (a crowded video has tens of thousands). */
export function formatLog(log: RedactionLog): string {
  const { tracks, audio, ...head } = log;
  const top = JSON.stringify(head, null, 1).replace(/\n}$/, '');
  const lines = (xs: unknown[]) => (xs.length ? `\n${xs.map((t) => `  ${JSON.stringify(t)}`).join(',\n')}\n ` : '');
  const a = audio ? `,\n "audio": {"sound": ${JSON.stringify(audio.sound)}, "totals": ${JSON.stringify(audio.totals)},\n  "spans": [${lines(audio.spans)}]}` : '';
  return `${top},\n "tracks": [${lines(tracks)}]${a}\n}\n`;
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
  /** The faces layer's configHash: keep ids name its tracks, and other settings number them
   *  differently. Absent when the doc had no faces layer (a review of the audio only). */
  faces?: string;
  /** The plates layer's configHash, when the doc had one (for keepPlates, likewise). */
  plates?: string;
  /** The transcript's configHash, when the doc had one (keepWords name its words). */
  transcript?: string;
  options: RedactOptions;
  /** Faces added by hand: tracks with negative ids, boxes in source pixels, interpolated between samples. */
  added: Track[];
  /** Audio spans to bleep added by hand: [t0, t1] seconds (words the transcript missed). */
  addedAudio?: [number, number][];
}

export function newPlan(doc: VideoDoc, options: RedactOptions = REDACT_DEFAULTS): RedactPlan {
  const hash = (k: 'faces' | 'plates' | 'transcript') => doc.layers[k]?.producer.configHash;
  const [faces, plates, transcript] = [hash('faces'), hash('plates'), hash('transcript')];
  return {
    schema: 'videodoc-redact/0', source: { name: doc.source.name, fingerprint: doc.source.fingerprint }, ...(faces && { faces }), ...(plates && { plates }), ...(transcript && { transcript }),
    options: { ...options, keep: [...options.keep], keepPlates: [...options.keepPlates], cover: [...options.cover], words: [...options.words], keepWords: [...options.keepWords] }, added: [], addedAudio: [],
  };
}

/** Check a plan read from disk against `doc`. Keep ids from another faces layer no longer name
 *  the same faces, so they are dropped (with a warning); faces added by hand stay. */
export function checkPlan(v: unknown, doc: VideoDoc): { plan: RedactPlan; warnings: string[] } {
  const p = v as RedactPlan;
  if (p?.schema !== 'videodoc-redact/0') throw new Error(`not a videodoc review plan (schema ${JSON.stringify(p?.schema)})`);
  if (p.source?.fingerprint !== doc.source.fingerprint) throw new Error(`the review plan is for another file (${p.source?.name})`);
  const plan: RedactPlan = {
    ...newPlan(doc, { ...REDACT_DEFAULTS, ...p.options }),
    added: Array.isArray(p.added) ? p.added : [],
    addedAudio: Array.isArray(p.addedAudio) ? p.addedAudio.filter((s) => Array.isArray(s) && s.length === 2 && s.every(Number.isFinite) && s[1] > s[0]) : [],
  };
  const warnings: string[] = [];
  if (!REDACT_STYLES.includes(plan.options.style)) {
    warnings.push(`unknown style '${plan.options.style}', using ${REDACT_DEFAULTS.style}`);
    plan.options.style = REDACT_DEFAULTS.style;
  }
  if (p.faces !== plan.faces && plan.options.keep.length) {
    warnings.push(`faces were found again with other settings since the review, so its ${plan.options.keep.length} kept face(s) are blurred again; review them again`);
    plan.options.keep = [];
  }
  if (p.plates !== plan.plates && plan.options.keepPlates.length) {
    warnings.push(`plates were found again since the review, so its ${plan.options.keepPlates.length} kept plate(s) are blurred again; review them again`);
    plan.options.keepPlates = [];
  }
  if (p.transcript !== plan.transcript && plan.options.keepWords.length) {
    warnings.push(`the transcript was made again since the review, so its ${plan.options.keepWords.length} word(s) left audible are bleeped again; review them again`);
    plan.options.keepWords = [];
  }
  if (!REDACT_SOUNDS.includes(plan.options.sound)) {
    warnings.push(`unknown sound '${plan.options.sound}', using ${REDACT_DEFAULTS.sound}`);
    plan.options.sound = REDACT_DEFAULTS.sound;
  }
  plan.options.cover = plan.options.cover.filter((k) => REDACT_KINDS.includes(k));
  plan.options.words = plan.options.words.filter((w) => typeof w === 'string' && w.trim());
  return { plan, warnings };
}

export const parsePlan = (text: string, doc: VideoDoc) => checkPlan(JSON.parse(text), doc);

/** JSON with one added track per line. */
export function formatPlan(p: RedactPlan): string {
  const { added, ...head } = p;
  if (!head.addedAudio?.length) delete head.addedAudio;
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
