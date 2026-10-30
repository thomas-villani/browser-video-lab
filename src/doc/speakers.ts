// Who said what: the speakers layer (diarization) applied to the transcript, and the edits a
// person makes to it: names, merges (two speakers who are one person) and moves (speech given to
// the wrong speaker). Edits live next to the video (<video>.speakers.json), not in the doc, so
// they survive re-analysis: each edit remembers the voice of the speaker it names, and after the
// speakers are found again it follows that voice. withSpeakers gives the doc as edited, which
// the Markdown, search, summaries and redaction read. Pure.
import { hungarianMax } from '../audio/der.ts';
import type { SpeakerItem, TranscriptItem, VideoDoc, Word } from './schema.ts';

/** A stretch of the transcript said by one speaker (undefined: nobody was found speaking). */
export interface Said {
  t0: number;
  t1: number;
  text: string;
  speaker?: number;
  /** The transcript item it comes from. */
  item: number;
  /** Its words, when the transcript has word times. */
  words?: Word[];
}

/** Seconds of [t0, t1] each speaker talks over; the one talking most, or undefined. */
export function speakerAt(speakers: SpeakerItem[], t0: number, t1: number): number | undefined {
  let best: number | undefined;
  let most = 0;
  for (const s of speakers) {
    let n = 0;
    for (const [a, b] of s.turns) if (b > t0 && a < t1) n += Math.min(b, t1) - Math.max(a, t0);
    if (n > most) {
      most = n;
      best = s.id;
    }
  }
  return best;
}

/** Split transcript segments where the speaker changes (by word, when the words have times).
 *  Words in no one's turn go with the word before (or after) them. */
export function whoSaid(items: TranscriptItem[], speakers: SpeakerItem[]): Said[] {
  const out: Said[] = [];
  items.forEach((it, item) => {
    if (!it.words?.length) {
      out.push({ t0: it.t0, t1: it.t1, text: it.text.trim(), speaker: speakerAt(speakers, it.t0, it.t1), item });
      return;
    }
    const who = it.words.map(([a, b]) => speakerAt(speakers, a, Math.max(b, a + 0.01)));
    for (let i = 1; i < who.length; i++) who[i] ??= who[i - 1];
    for (let i = who.length - 2; i >= 0; i--) who[i] ??= who[i + 1];
    let from = 0;
    for (let i = 1; i <= it.words.length; i++) {
      if (i < it.words.length && who[i] === who[from]) continue;
      const words = it.words.slice(from, i);
      out.push({
        t0: words[0][0],
        t1: words[words.length - 1][1],
        text: words
          .map((w) => w[2].trim())
          .join(' ')
          .replace(/\s+([,.?!;:…])/g, '$1'),
        speaker: who[from],
        item,
        words,
      });
      from = i;
    }
  });
  return out;
}

/** "Dana", or "Speaker 2" for a speaker nobody named. */
export const speakerName = (s: Pick<SpeakerItem, 'id' | 'name'>) => s.name ?? `Speaker ${s.id}`;

/** Seconds a speaker talks. */
export const talkTime = (s: Pick<SpeakerItem, 'turns'>) => s.turns.reduce((n, [a, b]) => n + b - a, 0);

/** The speaker `ref` names: a name (any case), "Speaker 2" or "2" (also a speaker merged into it). */
export function findSpeaker(speakers: SpeakerItem[], ref: string | number): SpeakerItem | undefined {
  const r = String(ref).trim().toLowerCase();
  const id = /^(speaker\s*)?\d+$/.test(r) ? Number(r.replace(/\D/g, '')) : undefined;
  return speakers.find((s) => s.name?.toLowerCase() === r) ?? speakers.find((s) => s.id === id) ?? speakers.find((s) => id !== undefined && s.merged?.includes(id));
}

// ---- turns as sets of times ----

type Span = [number, number];
const r3 = (v: number) => Math.round(v * 1000) / 1000;

/** Spans sorted and joined where they overlap or touch. */
export function unionTurns(spans: Span[]): Span[] {
  const out: Span[] = [];
  for (const [a, b] of [...spans].sort((x, y) => x[0] - y[0])) {
    const last = out[out.length - 1];
    if (last && a <= last[1]) last[1] = Math.max(last[1], b);
    else if (b > a) out.push([a, b]);
  }
  return out;
}

/** `spans` with [t0, t1] cut out, and the pieces cut. */
function cut(spans: Span[], t0: number, t1: number): { kept: Span[]; cut: Span[] } {
  const kept: Span[] = [];
  const out: Span[] = [];
  for (const [a, b] of spans) {
    if (b <= t0 || a >= t1) {
      kept.push([a, b]);
      continue;
    }
    if (a < t0) kept.push([a, t0]);
    if (b > t1) kept.push([t1, b]);
    out.push([Math.max(a, t0), Math.min(b, t1)]);
  }
  return { kept, cut: out };
}

// ---- edits: <video>.speakers.json ----

export const SPEAKERS_SCHEMA = 'videodoc-speakers/0';
export const speakersPathFor = (video: string) => `${video}.speakers.json`;

/**
 * Two voices closer than this (cosine distance) are taken for one person: an edit follows its
 * voice to the closest speaker found again within it, and a voice named in another video is
 * suggested for one within it. Measured on AMI meetings cut into parts that stand in for
 * separate videos (scripts/eval-voices.ts, NOTES.md): one person's voices in parts of 4-12
 * minutes were at most 0.39 apart (median 0.06); two people's, in the same room, at least 0.16
 * and 97% over 0.4, in different rooms at least 0.54.
 */
export const SAME_VOICE = 0.4;

/** One speaker as a person edited it. `id` is the speakers layer's (or, for one `added`, its own). */
export interface SpeakerEdit {
  id: number;
  name?: string;
  /** This speaker is that one (same person): its turns go to it. */
  same?: number;
  /** A speaker the analysis missed, given speech by a move (it has no voice). */
  added?: true;
  /** The layer's voice for `id` when the edit was made: after re-analysis, the edit goes to the
   *  speaker with the closest voice (within SAME_VOICE), and names are suggested from it in
   *  other videos. */
  voice?: number[];
}

/** Speech in [t0, t1] is `to`'s: `from`'s turns there move to `to` (without `from`, everyone's,
 *  and all of [t0, t1] becomes `to`'s). Applied in order, after merges. */
export interface SpeakerMove {
  t0: number;
  t1: number;
  from?: number;
  to: number;
}

export interface SpeakerEdits {
  schema: typeof SPEAKERS_SCHEMA;
  source: { name: string; fingerprint: string };
  speakers: SpeakerEdit[];
  moves: SpeakerMove[];
}

export function newEdits(doc: VideoDoc): SpeakerEdits {
  return { schema: SPEAKERS_SCHEMA, source: { name: doc.source.name, fingerprint: doc.source.fingerprint }, speakers: [], moves: [] };
}

const cosine = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += a[i] * b[i];
  return 1 - d;
};

/** The speaker a merge chain ends at (cycles stop where they close). */
function root(edits: SpeakerEdits, id: number): number {
  const seen = new Set<number>();
  for (let e = edits.speakers.find((x) => x.id === id); e?.same !== undefined && !seen.has(e.id); e = edits.speakers.find((x) => x.id === id)) {
    seen.add(e.id);
    id = e.same;
  }
  return id;
}

/**
 * Check edits read from disk against `doc`'s speakers layer, and move them to its speakers
 * when they were found again since (by voice; ids number speakers by first appearance, so they
 * change). Edits whose voice is not among the speakers now are dropped, with a warning.
 */
export function checkEdits(v: unknown, doc: VideoDoc): { edits: SpeakerEdits; warnings: string[] } {
  const e = v as SpeakerEdits;
  if (e?.schema !== SPEAKERS_SCHEMA) throw new Error(`not a videodoc speakers file (schema ${JSON.stringify(e?.schema)})`);
  if (e.source?.fingerprint !== doc.source.fingerprint) throw new Error(`the speakers file is for another file (${e.source?.name})`);
  const items = doc.layers.speakers?.items ?? [];
  const warnings: string[] = [];
  const ok = (x: unknown): x is number => Number.isInteger(x);
  let list = (Array.isArray(e.speakers) ? e.speakers : [])
    .filter((s) => s && ok(s.id))
    .map(
      (s): SpeakerEdit => ({
        id: s.id,
        ...(typeof s.name === 'string' && s.name.trim() && { name: s.name.trim() }),
        ...(ok(s.same) && s.same !== s.id && { same: s.same }),
        ...(s.added === true && { added: true as const }),
        ...(Array.isArray(s.voice) && s.voice.length && s.voice.every(Number.isFinite) && { voice: s.voice }),
      }),
    );
  const label = (s: SpeakerEdit) => s.name ?? `Speaker ${s.id}`;

  // Which layer speaker each voiced edit is now: the same id when its voice is unchanged (the
  // speakers were not found again), else the closest voice, one to one.
  const map = new Map<number, number>();
  const voiced = list.filter((s) => !s.added && s.voice);
  const unchanged = voiced.every((s) => {
    const it = items.find((x) => x.id === s.id);
    return it && it.voice.length === s.voice!.length && cosine(it.voice, s.voice!) < 0.02;
  });
  if (unchanged) for (const s of voiced) map.set(s.id, s.id);
  else if (voiced.length && items.length) {
    const score = voiced.map((s) => items.map((it) => (it.voice.length === s.voice!.length ? Math.max(0, 1 - cosine(it.voice, s.voice!)) : 0)));
    hungarianMax(score).forEach((j, i) => {
      if (j >= 0 && score[i][j] > 1 - SAME_VOICE) map.set(voiced[i].id, items[j].id);
    });
  }
  // Edits without a voice name a layer speaker only while it has that id.
  for (const s of list) if (!s.added && !s.voice && unchanged && items.some((x) => x.id === s.id)) map.set(s.id, s.id);
  const lost = list.filter((s) => !s.added && !map.has(s.id));
  for (const s of lost) warnings.push(`${label(s)}: no speaker found now has that voice (the speakers were found again), so this edit is dropped`);
  // Added speakers keep their id unless a layer speaker has it now.
  let next = Math.max(0, ...items.map((x) => x.id), ...map.values()) + 1;
  const used = new Set([...items.map((x) => x.id)]);
  for (const s of list.filter((x) => x.added)) {
    const id = used.has(s.id) ? next++ : s.id;
    used.add(id);
    map.set(s.id, id);
  }
  list = list
    .filter((s) => map.has(s.id))
    .map((s) => ({ ...s, id: map.get(s.id)!, ...(s.voice && !s.added && { voice: items.find((x) => x.id === map.get(s.id))?.voice ?? s.voice }) }));
  // References to layer speakers nobody edited (a merge into Speaker 3) need it unchanged.
  const ref = (id: number) => map.get(id) ?? (unchanged && items.some((x) => x.id === id) ? id : undefined);
  list = list.flatMap((s) => {
    if (s.same === undefined) return [s];
    const to = ref(s.same);
    if (to === undefined) {
      warnings.push(`${label(s)}: the speaker it was merged into is gone, so it stands alone again`);
      const { same: _, ...rest } = s;
      return [rest];
    }
    return [{ ...s, same: to }];
  });
  const moves: SpeakerMove[] = [];
  for (const m of Array.isArray(e.moves) ? e.moves : []) {
    if (!m || !Number.isFinite(m.t0) || !Number.isFinite(m.t1) || !(m.t1 > m.t0) || !ok(m.to)) continue;
    const to = ref(m.to);
    const from = m.from === undefined ? undefined : ref(m.from);
    if (to === undefined || (m.from !== undefined && from === undefined)) {
      warnings.push(`the speech moved at ${r3(m.t0)}-${r3(m.t1)} s names a speaker that is gone, so the move is dropped`);
      continue;
    }
    moves.push({ t0: m.t0, t1: m.t1, ...(from !== undefined && { from }), to });
  }
  return { edits: { schema: SPEAKERS_SCHEMA, source: { name: doc.source.name, fingerprint: doc.source.fingerprint }, speakers: list, moves }, warnings };
}

export const parseEdits = (text: string, doc: VideoDoc) => checkEdits(JSON.parse(text), doc);

/** JSON with one speaker and one move per line. */
export function formatEdits(e: SpeakerEdits): string {
  const lines = (xs: unknown[]) => (xs.length ? `\n${xs.map((x) => `  ${JSON.stringify(x)}`).join(',\n')}\n ` : '');
  // Keys in one order, the voice last (so a person can edit names by hand).
  const speakers = e.speakers.map(({ id, name, same, added, voice }) => ({ id, name, same, added, voice }));
  return `{\n "schema": ${JSON.stringify(e.schema)},\n "source": ${JSON.stringify(e.source)},\n "speakers": [${lines(speakers)}],\n "moves": [${lines(e.moves)}]\n}\n`;
}

/**
 * The speakers as edited: merged speakers' turns go to the speaker they are, moves are applied
 * in order, names are set. Speakers keep their ids (a merged one is gone; its id is listed in
 * `merged`); those left with no speech are dropped.
 */
export function applyEdits(items: SpeakerItem[], edits?: SpeakerEdits): SpeakerItem[] {
  if (!edits || (!edits.speakers.length && !edits.moves.length)) return items;
  const out = new Map<number, SpeakerItem>(items.map((s) => [s.id, { ...s, turns: s.turns.map((t): Span => [t[0], t[1]]) }]));
  for (const s of edits.speakers) if (s.added && !out.has(s.id)) out.set(s.id, { id: s.id, turns: [], voice: [] });
  for (const s of edits.speakers) {
    const to = root(edits, s.id);
    const from = out.get(s.id);
    const into = out.get(to);
    if (to === s.id || !from || !into) continue;
    into.turns = unionTurns([...into.turns, ...from.turns]);
    into.merged = [...(into.merged ?? []), s.id, ...(from.merged ?? [])];
    out.delete(s.id);
  }
  for (const m of edits.moves) {
    const to = out.get(root(edits, m.to));
    if (!to) continue;
    const from = m.from === undefined ? undefined : root(edits, m.from);
    const got: Span[] = m.from === undefined ? [[m.t0, m.t1]] : [];
    for (const s of out.values()) {
      if (s === to || (from !== undefined && s.id !== from)) continue;
      const c = cut(s.turns, m.t0, m.t1);
      s.turns = c.kept;
      if (from !== undefined) got.push(...c.cut);
    }
    to.turns = unionTurns([...to.turns, ...got]);
  }
  for (const s of edits.speakers) {
    const it = out.get(s.id);
    if (it && s.name) it.name = s.name;
  }
  return [...out.values()]
    .filter((s) => s.turns.length)
    .map((s) => ({ ...s, turns: s.turns.map((t): Span => [r3(t[0]), r3(t[1])]) }))
    .sort((a, b) => a.id - b.id);
}

/** `doc` with its speakers as edited (the doc itself is not changed). */
export function withSpeakers(doc: VideoDoc, edits?: SpeakerEdits): VideoDoc {
  const l = doc.layers.speakers;
  if (!l || !edits) return doc;
  return { ...doc, layers: { ...doc.layers, speakers: { ...l, items: applyEdits(l.items, edits) } } };
}

// ---- making edits (each returns new edits; `items` is the layer as analyzed) ----

/** The edit for `id`, created (with the layer's voice) when there is none. */
function entry(edits: SpeakerEdits, items: SpeakerItem[], id: number): { edits: SpeakerEdits; e: SpeakerEdit } {
  const have = edits.speakers.find((s) => s.id === id);
  if (have) return { edits, e: have };
  const it = items.find((s) => s.id === id);
  if (!it) throw new Error(`no speaker ${id}`);
  const e: SpeakerEdit = { id, voice: it.voice };
  return { edits: { ...edits, speakers: [...edits.speakers, e].sort((a, b) => a.id - b.id) }, e };
}

const replace = (edits: SpeakerEdits, e: SpeakerEdit, next: SpeakerEdit): SpeakerEdits => ({ ...edits, speakers: edits.speakers.map((s) => (s === e ? next : s)) });

/** Name speaker `id` (an empty name removes it). A name already given to another speaker is an
 *  error: they are one person, so merge them. */
export function nameSpeaker(edits: SpeakerEdits, items: SpeakerItem[], id: number, name: string): SpeakerEdits {
  const n = name.trim();
  const taken = edits.speakers.find((s) => s.id !== id && s.name?.toLowerCase() === n.toLowerCase() && root(edits, s.id) !== root(edits, id));
  if (n && taken) throw new Error(`${n} is Speaker ${taken.id} already: merge them instead`);
  const r = entry(edits, items, root(edits, id));
  const { name: _, ...rest } = r.e;
  return replace(r.edits, r.e, n ? { ...rest, name: n } : rest);
}

/** Speaker `id` is speaker `into` (one person): its turns go to `into`, which keeps its name. */
export function mergeSpeakers(edits: SpeakerEdits, items: SpeakerItem[], id: number, into: number): SpeakerEdits {
  if (root(edits, into) === root(edits, id)) throw new Error(`Speaker ${id} and Speaker ${into} are one speaker already`);
  let r = entry(edits, items, into);
  r = entry(r.edits, items, id);
  const { name, ...rest } = r.e;
  let out = replace(r.edits, r.e, { ...rest, same: root(r.edits, into) });
  // A name on the merged speaker goes to the one it joins, when that one has none.
  if (name) {
    const target = out.speakers.find((s) => s.id === root(out, into))!;
    if (!target.name) out = replace(out, target, { ...target, name });
  }
  return out;
}

/** Undo a merge: speaker `id` stands alone again. */
export function unmergeSpeaker(edits: SpeakerEdits, id: number): SpeakerEdits {
  const e = edits.speakers.find((s) => s.id === id && s.same !== undefined);
  if (!e) throw new Error(`Speaker ${id} is not merged into another`);
  const { same: _, ...rest } = e;
  return replace(edits, e, rest);
}

/** A new speaker (one the analysis missed), for moves; returns its id. */
export function addSpeaker(edits: SpeakerEdits, items: SpeakerItem[], name?: string): { edits: SpeakerEdits; id: number } {
  const id = Math.max(0, ...items.map((s) => s.id), ...edits.speakers.map((s) => s.id)) + 1;
  const n = name?.trim();
  if (n && edits.speakers.some((s) => s.name?.toLowerCase() === n.toLowerCase())) throw new Error(`${n} is a speaker already`);
  return { edits: { ...edits, speakers: [...edits.speakers, { id, ...(n && { name: n }), added: true }] }, id };
}

/** Speech in [t0, t1] is `to`'s (only `from`'s, when given). */
export function moveSpeech(edits: SpeakerEdits, items: SpeakerItem[], t0: number, t1: number, to: number, from?: number): SpeakerEdits {
  if (!(t1 > t0)) throw new Error(`a move needs t1 > t0 (got ${t0}-${t1})`);
  let out = edits;
  for (const id of from === undefined ? [to] : [to, from]) if (!edits.speakers.some((s) => s.id === id)) out = entry(out, items, id).edits;
  return { ...out, moves: [...out.moves, { t0: r3(t0), t1: r3(t1), ...(from !== undefined && { from }), to }] };
}

/** Remove move `index`. */
export function unmove(edits: SpeakerEdits, index: number): SpeakerEdits {
  if (!edits.moves[index]) throw new Error(`no move ${index} (there are ${edits.moves.length})`);
  return { ...edits, moves: edits.moves.filter((_, i) => i !== index) };
}

// ---- names from other videos ----

/** A voice a person named: in which video, under which name. */
export interface NamedVoice {
  name: string;
  voice: number[];
  video: string;
}

/** The named voices of an edits file (`video` says where it comes from). */
export const namedVoices = (edits: SpeakerEdits, video: string): NamedVoice[] =>
  edits.speakers.flatMap((s) => (s.name && s.voice && !s.added ? [{ name: s.name, voice: s.voice, video }] : []));

export interface NameSuggestion {
  id: number;
  name: string;
  distance: number;
  video: string;
}

/**
 * Names for unnamed speakers from voices named in other videos: for each, the closest named
 * voice within SAME_VOICE, one name per speaker and one speaker per name (the closest pair
 * first), names already used in this video left out. Suggestions only: a person applies them.
 */
export function suggestSpeakerNames(speakers: SpeakerItem[], known: NamedVoice[], threshold = SAME_VOICE): NameSuggestion[] {
  const used = new Set(speakers.flatMap((s) => (s.name ? [s.name.toLowerCase()] : [])));
  const pairs: NameSuggestion[] = [];
  for (const s of speakers) {
    if (s.name || !s.voice.length) continue;
    for (const k of known) {
      if (used.has(k.name.toLowerCase()) || k.voice.length !== s.voice.length) continue;
      const distance = Math.round(cosine(s.voice, k.voice) * 1000) / 1000;
      if (distance < threshold) pairs.push({ id: s.id, name: k.name, distance, video: k.video });
    }
  }
  pairs.sort((a, b) => a.distance - b.distance);
  const out: NameSuggestion[] = [];
  for (const p of pairs) if (!out.some((o) => o.id === p.id || o.name.toLowerCase() === p.name.toLowerCase())) out.push(p);
  return out.sort((a, b) => a.id - b.id);
}
