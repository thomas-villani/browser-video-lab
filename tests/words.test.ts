// Audio redaction: word matching (src/doc/words.ts) and the spans, log and plan (src/doc/redact.ts). Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  audioCover, audioSpans, checkPlan, findWords, formatLog, formatPlan, mergeTimes, newDoc, newPlan, normWord, redactionLog, REDACT_DEFAULTS, suggestNames, withLayers,
  type Layer, type TranscriptItem, type Word,
} from '../src/doc/index.ts';

const layer = <T>(items: T[], configHash = 't') => ({ producer: { module: 'm', version: '0', model: 'whisper', configHash, config: {} }, items, coverage: [], createdAt: 'now', ms: 1, device: 'webgpu' }) as unknown as Layer<T>;

// Words one per 0.5 s from `t`, each 0.4 s long.
const item = (t: number, text: string): TranscriptItem => {
  const words = text.split(' ').map((w, i): Word => [t + i * 0.5, t + i * 0.5 + 0.4, w]);
  return { t0: t, t1: words[words.length - 1][1], text, words };
};
const transcript = [
  item(1, 'My name is Margaret Okafor.'),
  item(5, "Okafor's report was screwed up, the whole bullshit thing."),
  item(12, 'Call 555-213-4987 about the shipment, Daniel Reyes said.'),
  item(18, 'Spiced tea for the spics? Fucking hell.'),
];
const doc = withLayers(
  newDoc({ name: 'v.mp4', fingerprint: 'v0:1:x', size: 1, duration: 21, width: 320, height: 240, fps: 25, frames: 525, video: { codec: 'avc1' } }),
  { transcript: layer(transcript) },
);
const find = (...terms: string[]) => findWords(doc, terms.map((term) => ({ term }))).map((m) => [m.key, m.text]);

test('normWord: case, accents, quotes and punctuation', () => {
  assert.equal(normWord('Okafor’s,'), "okafor's");
  assert.equal(normWord('José'), 'jose');
  assert.equal(normWord('-213'), '213');
  assert.equal(normWord('"Hello!"'), 'hello');
});

test('findWords: inflections, phrases, hyphenated numbers, wildcards, exact quotes', () => {
  // A name matches its possessive; matches come in time order.
  assert.deepEqual(find('Okafor'), [['0.4', 'Okafor.'], ['1.0', "Okafor's"]]);
  assert.deepEqual(find('screw'), [['1.3', 'screwed']]);
  // A phrase matches consecutive words; a hyphenated word is several tokens.
  assert.deepEqual(find('Daniel Reyes'), [['2.5', 'Daniel Reyes']]);
  assert.deepEqual(find('555 213 4987'), [['2.1', '555-213-4987']]);
  assert.deepEqual(find('555-213-4987'), [['2.1', '555-213-4987']]);
  assert.deepEqual(find('213'), [['2.1', '555-213-4987']]);
  // Wildcards do not inflect; they cover compounds.
  assert.deepEqual(find('*shit*'), [['1.7', 'bullshit']]);
  assert.deepEqual(find('fuck*'), [['3.5', 'Fucking']]);
  // A plain word inflects (spic -> spiced), a quoted one matches only itself.
  assert.deepEqual(find('spic'), [['3.0', 'Spiced'], ['3.4', 'spics?']]);
  assert.deepEqual(find('"spic"'), []);
  assert.deepEqual(find('"spics"'), [['3.4', 'spics?']]);
  // A word in two lists is one match (the first term listed wins).
  assert.deepEqual(findWords(doc, [{ term: 'okafor' }, { term: 'Okafor', label: 'x' }]).map((m) => m.term), ['okafor', 'okafor']);
  assert.deepEqual(find('nothing here'), []);
});

test('audioSpans: lead and tail, clipped; kept matches listed but not covered; spans added by hand', () => {
  const o = { ...REDACT_DEFAULTS, words: ['Okafor'], profanity: true, lead: 0.4, tail: 0.2, keepWords: ['1.0'] };
  const spans = audioSpans(doc, o, [[20.5, 30]]);
  assert.deepEqual(
    spans.map((s) => [s.t0, s.t1, s.term ?? 'hand', s.kept]),
    [
      [2.6, 3.6, 'Okafor', false], // word 3.0-3.4
      [4.6, 5.6, 'Okafor', true],
      [8.1, 9.1, 'profanity', false], // bullshit 8.5-8.9
      [19.6, 20.6, 'profanity', false], // spics 20.0-20.4
      [20.1, 21, 'profanity', false], // Fucking 20.5-20.9, clipped to the duration
      [20.5, 21, 'hand', false],
    ],
  );
  assert.deepEqual(audioCover(spans), [[2.6, 3.6], [8.1, 9.1], [19.6, 21]]);
  assert.deepEqual(mergeTimes([[3, 4], [1, 2], [2, 2.5]]), [[1, 2.5], [3, 4]]);
});

test('audioSpans: needs word timestamps', () => {
  const noWords = withLayers(doc, { transcript: layer(transcript.map(({ words, ...s }) => s)) });
  assert.throws(() => audioSpans(noWords, { ...REDACT_DEFAULTS, words: ['x'] }), /no word timestamps/);
  const none = withLayers(newDoc(doc.source), {});
  assert.throws(() => audioSpans(none, { ...REDACT_DEFAULTS, words: ['x'] }), /no transcript/);
  // Spans added by hand need no transcript.
  assert.equal(audioSpans(none, REDACT_DEFAULTS, [[1, 2]]).length, 1);
});

test('log: audio section with totals and one span per line; video-only logs have none', () => {
  const o = { ...REDACT_DEFAULTS, cover: [], words: ['Okafor', 'Daniel Reyes'], keepWords: ['1.0'] };
  const log = redactionLog(doc, o, undefined, [], [[15, 15.5]]);
  assert.deepEqual(log.audio!.totals, { matches: 3, kept: 1, added: 1, seconds: 2.5 });
  assert.equal(log.detectors.transcript?.configHash, 't');
  assert.match(log.caveat, /Listen to the output/);
  const text = formatLog(log);
  assert.deepEqual(JSON.parse(text), log);
  assert.equal(text.split('\n').filter((l) => l.includes('"term"')).length, 3);
  assert.equal(redactionLog(doc, { ...REDACT_DEFAULTS, cover: [] }).audio, undefined);
});

test('plan: audio settings round-trip; kept words are dropped when the transcript changes', () => {
  const plan = newPlan(doc, { ...REDACT_DEFAULTS, cover: [], words: ['Okafor'], keepWords: ['1.0'] });
  assert.equal(plan.faces, undefined);
  assert.equal(plan.transcript, 't');
  plan.addedAudio = [[3, 4]];
  const back = checkPlan(JSON.parse(formatPlan(plan)), doc);
  assert.deepEqual(back.plan, plan);
  assert.deepEqual(back.warnings, []);
  const again = withLayers(doc, { transcript: layer(transcript, 't2') });
  const stale = checkPlan(JSON.parse(formatPlan(plan)), again);
  assert.deepEqual(stale.plan.options.keepWords, []);
  assert.deepEqual(stale.plan.addedAudio, [[3, 4]]);
  assert.match(stale.warnings[0], /transcript was made again/);
});

test('findWords: a match inside a longer one is dropped; overlapping ones are both kept', () => {
  const d = withLayers(newDoc(doc.source), { transcript: layer([item(0, 'you son of a bitch, Daniel Reyes Smith')]) });
  const f = (...terms: string[]) => findWords(d, terms.map((term) => ({ term }))).map((m) => m.text);
  assert.deepEqual(f('bitch*', 'son of a bitch'), ['son of a bitch,']);
  assert.deepEqual(f('Daniel Reyes', 'Reyes Smith'), ['Daniel Reyes', 'Reyes Smith']);
});

test('suggestNames: mid-sentence capitals never written in lowercase', () => {
  const d = withLayers(newDoc(doc.source), {
    transcript: layer([
      { t0: 0, t1: 1, text: 'Officer Patel met Dixie at the Police station.' },
      { t0: 1, t1: 2, text: "The police called Patel's office. Dixie said I was late, Monday." },
      { t0: 2, t1: 3, text: 'Then Patel left for Fairfield.' },
    ]),
  });
  // Dixie starts a sentence the second time (once is not enough); Police is also written in
  // lowercase; Monday is never a name; Fairfield comes up once.
  assert.deepEqual(suggestNames(d), [{ word: 'Patel', count: 3 }]);
  assert.deepEqual(suggestNames(d, 1).map((s) => s.word), ['Patel', 'Dixie', 'Fairfield']);
});
