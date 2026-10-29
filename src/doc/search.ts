// Keyword search over transcripts (BM25), transcript-first: the transcript alone carried ~80% of
// video-understanding performance in the bodycam work. Pure TS, so the CLI and MCP server (Node)
// and the web app (browser) search alike. Docs are small, so the index is built per query.
//
// Passages are overlapping windows of consecutive transcript segments (one starting at every
// segment, up to `window` seconds), so a phrase never falls across a boundary; overlapping or
// touching hits in the same video are collapsed to the best one.
import type { KeyframeItem, TranscriptItem, VideoDoc } from './schema.ts';
import { speakerAt } from './speakers.ts';

const STOP = new Set(
  ('a an and are as at be been but by can did do does for from had has have he her his i if in into is it its me my no not of on or our ' +
    'she so than that the their them then there these they this to too up us was we were what when where which who will with would you your')
    .split(' '),
);

/** Crude English stemming, applied alike to queries and passages ("cuts" ~ "cut", "taxes" ~ "tax"). */
export function stem(w: string): string {
  if (w.length <= 3) return w;
  if (w.endsWith("'s")) w = w.slice(0, -2);
  if (w.endsWith('ies') && w.length > 4) return `${w.slice(0, -3)}y`;
  if (/(ss|x|ch|sh)es$/.test(w)) return w.slice(0, -2);
  if (w.endsWith('s') && !/(ss|us|is)$/.test(w)) w = w.slice(0, -1);
  if (w.endsWith('ing') && w.length > 5) return w.slice(0, -3);
  if (w.endsWith('ed') && w.length > 4) return w.slice(0, -2);
  return w;
}

const words = (s: string) => s.toLowerCase().match(/[\p{L}\p{N}]+(?:['’][\p{L}]+)*/gu) ?? [];

/** Index terms of a text: lowercased words, stopwords dropped, stemmed. */
export function terms(s: string): string[] {
  return words(s).map((w) => w.replace('’', "'")).filter((w) => !STOP.has(w)).map(stem);
}

export interface Passage {
  /** Which doc in the collection (index into the docs given to search). */
  doc: number;
  t0: number;
  t1: number;
  /** Transcript items [i0, i1] (inclusive) the passage spans. */
  i0: number;
  i1: number;
  text: string;
}

export function passages(items: TranscriptItem[], doc: number, window: number): Passage[] {
  const out: Passage[] = [];
  for (let i = 0; i < items.length; i++) {
    let j = i;
    while (j + 1 < items.length && items[j + 1].t1 - items[i].t0 <= window) j++;
    const text = items.slice(i, j + 1).map((x) => x.text.trim()).join(' ');
    if (text) out.push({ doc, t0: items[i].t0, t1: items[j].t1, i0: i, i1: j, text });
  }
  return out;
}

/** Seconds before a matched word's time that a hit's `at` points to. */
const AT_LEAD = 0.5;

export interface SearchOptions {
  top?: number;
  /** Passage length, seconds. */
  window?: number;
  /** Hits in one video closer than this (seconds) count as overlapping; only the best is kept. */
  gap?: number;
  k1?: number;
  b?: number;
}

export interface SearchHit extends Passage {
  score: number;
  /** Where to seek to hear the first matched word: half a second before it is said (word times
   *  run late), or the start of its transcript item when the transcript has no word times. */
  at: number;
  /** Query terms found in the passage (stemmed). */
  matched: string[];
  /** The passage with matched words in **bold**. */
  snippet: string;
  /** Keyframe nearest the hit (index into the doc's keyframes layer), if the doc has keyframes. */
  keyframe?: number;
  /** Who says the first matched word (speakers layer id), if the doc has speakers. */
  speaker?: number;
}

/** Wrap words whose stem is in `stems` in **bold**. */
export function highlight(text: string, stems: Set<string>): string {
  return text.replace(/[\p{L}\p{N}]+(?:['’][\p{L}]+)*/gu, (w) => (stems.has(stem(w.toLowerCase().replace('’', "'"))) ? `**${w}**` : w));
}

function nearestKeyframe(kfs: KeyframeItem[] | undefined, t0: number, t1: number): number | undefined {
  if (!kfs?.length) return undefined;
  let best = 0;
  let bestD = Infinity;
  kfs.forEach((k, i) => {
    const d = k.t < t0 ? t0 - k.t : k.t > t1 ? k.t - t1 : 0;
    if (d < bestD) [best, bestD] = [i, d];
  });
  return best;
}

/**
 * Rank transcript passages of `docs` for `query` (BM25, plus a bonus when the passage contains
 * the query's words in order). Returns at most `top` hits, best first, none overlapping in time
 * within a video.
 */
export function search(docs: VideoDoc[], query: string, o: SearchOptions = {}): SearchHit[] {
  const { top = 10, window = 20, gap = 2, k1 = 1.2, b = 0.75 } = o;
  const q = [...new Set(terms(query))];
  if (!q.length) return [];
  const all = docs.flatMap((d, i) => passages(d.layers.transcript?.items ?? [], i, window));
  const items = docs.map((d) => d.layers.transcript?.items ?? []);
  const toks = all.map((p) => terms(p.text));
  const avg = toks.reduce((n, t) => n + t.length, 0) / Math.max(1, toks.length);
  const df = new Map<string, number>(q.map((t) => [t, 0]));
  for (const t of toks) for (const w of new Set(t)) if (df.has(w)) df.set(w, df.get(w)! + 1);
  const idf = (t: string) => Math.log(1 + (all.length - df.get(t)! + 0.5) / (df.get(t)! + 0.5));
  const phrase = q.length > 1 ? words(query).filter((w) => !STOP.has(w)).join(' ') : '';

  const scored: { p: Passage; score: number; matched: string[] }[] = [];
  all.forEach((p, i) => {
    const tf = new Map<string, number>();
    for (const w of toks[i]) if (df.has(w)) tf.set(w, (tf.get(w) ?? 0) + 1);
    if (!tf.size) return;
    let score = 0;
    for (const [t, f] of tf) score += (idf(t) * f * (k1 + 1)) / (f + k1 * (1 - b + (b * toks[i].length) / avg));
    if (phrase && words(p.text).filter((w) => !STOP.has(w)).join(' ').includes(phrase)) score *= 1.5;
    scored.push({ p, score, matched: [...tf.keys()] });
  });
  scored.sort((x, y) => y.score - x.score || x.p.doc - y.p.doc || x.p.t0 - y.p.t0);

  const hits: SearchHit[] = [];
  for (const s of scored) {
    if (hits.length >= top) break;
    if (hits.some((h) => h.doc === s.p.doc && h.t0 < s.p.t1 + gap && s.p.t0 < h.t1 + gap)) continue;
    const keyframe = nearestKeyframe(docs[s.p.doc].layers.keyframes?.items, s.p.t0, s.p.t1);
    const matched = new Set(s.matched);
    const first = items[s.p.doc].slice(s.p.i0, s.p.i1 + 1).find((x) => terms(x.text).some((w) => matched.has(w)));
    const word = first?.words?.find((w) => terms(w[2]).some((x) => matched.has(x)));
    const at = word ? Math.max(first!.t0, Math.round((word[0] - AT_LEAD) * 100) / 100) : (first?.t0 ?? s.p.t0);
    const spk = docs[s.p.doc].layers.speakers?.items;
    const speaker = spk && (word ? speakerAt(spk, word[0], Math.max(word[1], word[0] + 0.01)) : first && speakerAt(spk, first.t0, first.t1));
    hits.push({ ...s.p, score: Math.round(s.score * 1000) / 1000, at, matched: s.matched, snippet: highlight(s.p.text, new Set(s.matched)), ...(keyframe !== undefined && { keyframe }), ...(speaker !== undefined && { speaker }) });
  }
  return hits;
}
