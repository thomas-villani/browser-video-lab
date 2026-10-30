// Audio redaction, the pure half: which spans of the audio to bleep or mute, from the words
// and phrases a person lists (and the built-in profanity list) matched against the
// transcript's word timestamps. The browser half (src/redact.ts) rewrites the audio samples.
import type { VideoDoc, Word } from './schema.ts';

/** One spoken token: a transcript word, or part of a hyphenated one ("well-known" is two
 *  tokens sharing the word's times). `key` names the word: "<item>.<word>". */
interface Token {
  norm: string;
  key: string;
  t0: number;
  t1: number;
  text: string;
}

/** A list entry that matched: the words it covers, by the key of the first one. */
export interface WordMatch {
  key: string;
  /** The words as transcribed. */
  text: string;
  /** The list entry that matched it ("profanity" for the built-in list). */
  term: string;
  /** The words' own times, seconds (before lead and tail). */
  t0: number;
  t1: number;
}

/** Lowercase, accents and punctuation off ("Okafor's" -> "okafor's", "-213" -> "213"). */
export function normWord(s: string): string {
  return s
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[^a-z0-9'*]/g, '')
    .replace(/^'+|'+$/g, '');
}

/** A word's tokens: hyphens and slashes split it, each part keeps the word's times. */
function tokens(doc: VideoDoc): Token[] {
  const out: Token[] = [];
  for (const [i, item] of (doc.layers.transcript?.items ?? []).entries()) {
    for (const [j, w] of (item.words ?? []).entries()) {
      const [t0, t1, text] = w as Word;
      for (const part of text.split(/[-\u2013\u2014/]+/)) {
        const norm = normWord(part);
        if (norm) out.push({ norm, key: `${i}.${j}`, t0, t1, text });
      }
    }
  }
  return out;
}

/**
 * One word of a term as a test. Plain words match themselves and their simple inflections
 * (plural, possessive, -ed, -ing, -er): "Okafor" matches "Okafor's", "screw" matches
 * "screwed". `*` matches any run of letters: "fuck*", "*shit*". With `exact`, only the word.
 */
function wordTest(p: string, exact: boolean): (w: string) => boolean {
  if (exact) return (w) => w === p;
  if (p.includes('*')) {
    const re = new RegExp(
      `^${p
        .split('*')
        .map((s) => s.replace(/[.+?^${}()|[\]\\]/g, '\\$&'))
        .join("[a-z0-9']*")}$`,
    );
    return (w) => re.test(w) || re.test(w.replace(/'s$/, ''));
  }
  const forms = new Set([p, `${p}s`, `${p}es`, `${p}'s`, `${p}s'`, `${p}ed`, `${p}d`, `${p}ing`, `${p}er`, `${p}ers`]);
  if (p.endsWith('e')) forms.add(`${p.slice(0, -1)}ing`);
  if (p.endsWith('y')) for (const s of ['ies', 'ied']) forms.add(`${p.slice(0, -1)}${s}`);
  // Doubled final consonant: "drug" -> "drugged", "slam" -> "slamming".
  if (/[^aeiou][aeiou][bdgklmnprt]$/.test(p)) for (const s of ['ed', 'ing', 'er']) forms.add(`${p}${p.at(-1)}${s}`);
  return (w) => forms.has(w);
}

/** A list entry as a phrase: one test per word ("Daniel Reyes", "555-213-4987"). In double
 *  quotes, words match exactly (`"spic"` matches "spic" only, not "spiced"). */
function compile(term: string): ((w: string) => boolean)[] {
  const exact = /^\s*".*"\s*$/.test(term);
  return term
    .split(/[\s\-\u2013\u2014/]+/)
    .map(normWord)
    .filter(Boolean)
    .map((p) => wordTest(p, exact));
}

/**
 * Every place a term is spoken, by its words' timestamps. Each term is a word or a phrase
 * (consecutive words). A match inside a longer one ("bitch" in "son of a bitch") is left out,
 * and of two matches on the same words the first term listed wins; matches that only
 * overlap are both kept (`audioCover` merges them).
 */
export function findWords(doc: VideoDoc, terms: { term: string; label?: string }[]): WordMatch[] {
  const toks = tokens(doc);
  const found: (WordMatch & { keys: string[]; order: number })[] = [];
  for (const [order, { term, label }] of terms.entries()) {
    const tests = compile(term);
    if (!tests.length) continue;
    for (let i = 0; i + tests.length <= toks.length; i++) {
      if (!tests.every((t, k) => t(toks[i + k].norm))) continue;
      const run = toks.slice(i, i + tests.length);
      const words = run.filter((t, k) => k === 0 || t.key !== run[k - 1].key);
      found.push({
        key: run[0].key,
        keys: words.map((t) => t.key),
        text: words.map((t) => t.text).join(' '),
        term: label ?? term,
        t0: run[0].t0,
        t1: run[run.length - 1].t1,
        order,
      });
    }
  }
  // Longest first (then the first term listed), so a match inside one already taken is dropped.
  found.sort((a, b) => b.keys.length - a.keys.length || a.order - b.order);
  const taken = new Set<string>();
  const out: WordMatch[] = [];
  for (const { keys, order, ...m } of found) {
    if (keys.every((k) => taken.has(k))) continue;
    keys.forEach((k) => taken.add(k));
    out.push(m);
  }
  return out.sort((a, b) => a.t0 - b.t0);
}

/** Merge spans that overlap or touch (the audio is rewritten once per stretch). */
export function mergeTimes(spans: [number, number][]): [number, number][] {
  const out: [number, number][] = [];
  for (const [a, b] of [...spans].sort((x, y) => x[0] - y[0])) {
    const last = out[out.length - 1];
    if (last && a <= last[1]) last[1] = Math.max(last[1], b);
    else out.push([a, b]);
  }
  return out;
}

/**
 * The strong-language list for platforms that restrict it (opt-in: `profanity`). Words, not
 * mild ones (damn, hell, crap: add them to your own list) and not ones with common clean
 * senses (cock, dick, ass alone). Wildcards follow wordTest.
 */
export const PROFANITY: string[] = [
  '*fuck*',
  '*shit*',
  'asshole*',
  'arsehole*',
  'jackass',
  'dumbass*',
  'son of a bitch',
  'bitch*',
  'bastard*',
  'cunt*',
  'twat*',
  'wank*',
  'piss',
  'dickhead*',
  'cocksucker*',
  'goddamn*',
  'goddam',
  'nigger*',
  'nigga*',
  'faggot*',
  'fag',
  'retard',
  'kike*',
  '"spic"',
  '"spics"',
  'tranny',
  'trannies',
];

/** Capitalized words that are not names. */
const NOT_NAMES = new Set([
  'i',
  "i'm",
  "i'll",
  "i've",
  "i'd",
  'ok',
  'okay',
  'mr',
  'mrs',
  'ms',
  'dr',
  'god',
  'oh',
  'yes',
  'no',
  'hey',
  'monday',
  'tuesday',
  'wednesday',
  'thursday',
  'friday',
  'saturday',
  'sunday',
  'january',
  'february',
  'march',
  'april',
  'may',
  'june',
  'july',
  'august',
  'september',
  'october',
  'november',
  'december',
  'english',
  'american',
]);

/**
 * Words the transcript writes capitalized in mid-sentence and never in lowercase, by how often:
 * names of people and places, mostly (suggestions for the bleep list; a person picks). The
 * spelling is the transcript's, which is what matching needs.
 */
export function suggestNames(doc: VideoDoc, min = 2): { word: string; count: number }[] {
  const caps = new Map<string, { word: string; count: number }>();
  const lower = new Set<string>();
  for (const item of doc.layers.transcript?.items ?? []) {
    const words = item.text.split(/\s+/).filter(Boolean);
    for (const [i, raw] of words.entries()) {
      const w = raw.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '').replace(/['’]s$/, '');
      if (!w) continue;
      const norm = normWord(w);
      if (!/^\p{Lu}/u.test(w)) {
        lower.add(norm);
        continue;
      }
      // The first word of a sentence is capitalized anyway.
      if (i === 0 || /[.?!:]["')]?$/.test(words[i - 1]) || NOT_NAMES.has(norm) || w.length < 2) continue;
      const c = caps.get(norm) ?? { word: w, count: 0 };
      c.count++;
      caps.set(norm, c);
    }
  }
  return [...caps]
    .filter(([n, c]) => c.count >= min && !lower.has(n))
    .map(([, c]) => c)
    .sort((a, b) => b.count - a.count || a.word.localeCompare(b.word));
}
