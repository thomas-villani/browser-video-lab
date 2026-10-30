// Speaker diarization, the pure part (src/audio): powerset decoding, counting, clustering,
// assignment, reconstruction, and the DER used to measure it. Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assignMax, cluster, decodePowerset, diarize, DIARIZE_DEFAULTS, FRAME_STEP, SEG, speakerCount, windowStarts, type WindowSeg } from '../src/audio/diarize.ts';
import { der, hungarianMax, parseRttm } from '../src/audio/der.ts';
import { markdownToHtml } from '../src/app/render.ts';
import {
  addSpeaker, applyEdits, checkEdits, findSpeaker, formatEdits, mergeSpeakers, moveSpeech, nameSpeaker, namedVoices, newDoc, newEdits, REDACT_DEFAULTS, redactionLog, SAME_VOICE, search,
  speakerAt, suggestSpeakerNames, summarize, toMarkdown, unmergeSpeaker, unmove, whoSaid, withLayers, withSpeakers, type Layer, type SpeakerItem, type Word,
} from '../src/doc/index.ts';

/** A voice: a unit vector near axis `k` (dimension 8), jittered by `j`. */
const voice = (k: number, j = 0) => {
  const v = new Float32Array(8);
  v[k] = 1;
  v[(k + 1) % 8] = j;
  return v;
};

test('windows slide like pyannote: whole windows, then one padded window for the rest', () => {
  assert.deepEqual(windowStarts(25, 5), [0, 5, 10, 15]);
  assert.deepEqual(windowStarts(26, 5), [0, 5, 10, 15, 20]);
  assert.deepEqual(windowStarts(4, 5), [0]);
});

test('powerset logits decode to three local speakers', () => {
  const logits = new Float32Array(3 * 7);
  logits[0 * 7 + 0] = 5; // nobody
  logits[1 * 7 + 2] = 5; // speaker 2 alone
  logits[2 * 7 + 5] = 5; // speakers 1 and 3
  assert.deepEqual([...decodePowerset(logits, 3)], [0, 0, 0, 0, 1, 0, 1, 0, 1]);
});

test('clustering: centroid linkage stops at the threshold; small clusters join large ones; maxSpeakers merges on', () => {
  const vecs = [...Array.from({ length: 20 }, (_, i) => voice(0, i * 0.01)), ...Array.from({ length: 20 }, (_, i) => voice(3, i * 0.01)), voice(0, 0.6)];
  const labels = cluster(vecs, DIARIZE_DEFAULTS);
  assert.equal(new Set(labels.slice(0, 20)).size, 1);
  assert.equal(new Set(labels.slice(20, 40)).size, 1);
  assert.notEqual(labels[0], labels[20]);
  assert.equal(labels[40], labels[0], 'the outlier joins the nearest large cluster');
  assert.equal(new Set(cluster(vecs, { ...DIARIZE_DEFAULTS, maxSpeakers: 1 })).size, 1);
  assert.deepEqual(assignMax([[1, 5], [4, 2], [3, 3]]), [1, 0, -1]);
  assert.deepEqual(hungarianMax([[1, 5, 0], [4, 2, 0]]), [1, 0]);
});

test('diarize: two speakers taking turns across overlapping windows', () => {
  // Speaker A talks 0-8 s, B 8-20 s; windows of 10 s every 5 s. In each window, local speaker
  // 0 is whoever talks first.
  const frames = SEG.frames;
  const windows: WindowSeg[] = [];
  const embeddings: (Float32Array | undefined)[][] = [];
  for (const start of windowStarts(20, 5)) {
    const act = new Uint8Array(frames * 3);
    const who: number[] = [];
    for (let f = 0; f < frames; f++) {
      const t = start + f * FRAME_STEP + 0.03;
      const g = t < 8 ? 0 : 1;
      if (!who.includes(g)) who.push(g);
      act[f * 3 + who.indexOf(g)] = 1;
    }
    windows.push({ start, act });
    embeddings.push([0, 1, 2].map((s) => (s < who.length ? voice(who[s] * 3, Math.random() * 0.02) : undefined)));
  }
  const speakers = diarize(windows, embeddings, { ...DIARIZE_DEFAULTS, minClusterSize: 1 });
  assert.equal(speakers.length, 2);
  assert.equal(speakers[0].turns.length, 1);
  assert.ok(Math.abs(speakers[0].turns[0][1] - 8) < 0.05, `A stops at 8 s (${speakers[0].turns[0][1]})`);
  assert.ok(Math.abs(speakers[1].turns[0][0] - 8) < 0.05);
  assert.ok(speakers[1].turns[0][1] > 19.9);
  assert.equal(speakerCount(windows)[Math.round(4 / FRAME_STEP)], 1);
  assert.ok(Math.abs(Math.hypot(...speakers[0].voice) - 1) < 1e-6, 'voices are unit length');

  // Scored against the truth: near zero; a swapped hypothesis is all confusion.
  const ref = parseRttm('SPEAKER x 1 0 8 <NA> <NA> A <NA> <NA>\nSPEAKER x 1 8 12 <NA> <NA> B <NA> <NA>');
  const hyp = speakers.flatMap((s) => s.turns.map(([t0, t1]) => ({ t0, t1, speaker: String(s.id) })));
  assert.ok(der(ref, hyp, [0, 20]).der < 0.01);
  const one = der(ref, [{ t0: 0, t1: 20, speaker: 'x' }], [0, 20]);
  assert.ok(Math.abs(one.der - 0.4) < 1e-9, 'one speaker throughout: B is confused');
  assert.ok(Math.abs(der(ref, [], [0, 20]).der - 1) < 1e-9, 'nothing found: all missed');
});

test('who said what: words go to the speaker talking over them; paragraphs break where the speaker changes', () => {
  const speakers = [
    { id: 1, turns: [[0, 2.1]] as [number, number][], voice: [] },
    { id: 2, turns: [[2.1, 5]] as [number, number][], voice: [] },
  ];
  const items = [{ t0: 0, t1: 4, text: ' Did you see it? Yes, twice.', words: [[0, 0.4, ' Did'], [0.4, 0.6, ' you'], [0.6, 0.9, ' see'], [0.9, 1.6, ' it?'], [2.3, 2.8, ' Yes,'], [3, 3.9, ' twice.']] as Word[] }];
  assert.deepEqual(whoSaid(items, speakers).map(({ words, ...s }) => ({ ...s, words: words!.length })), [
    { t0: 0, t1: 1.6, text: 'Did you see it?', speaker: 1, item: 0, words: 4 },
    { t0: 2.3, t1: 3.9, text: 'Yes, twice.', speaker: 2, item: 0, words: 2 },
  ]);
  assert.equal(speakerAt(speakers, 1.9, 2.5), 2);
  assert.equal(speakerAt(speakers, 6, 7), undefined);
  const doc = withLayers(newDoc({ name: 'a.mp4', fingerprint: 'v0:1:x', size: 1, duration: 5, width: 640, height: 360, fps: 25, frames: 125, video: { codec: 'avc1' } }), {
    transcript: { producer: { configHash: 'h', config: {} }, items } as unknown as Layer<never>,
    speakers: { producer: { configHash: 'h', config: {} }, items: speakers } as unknown as Layer<never>,
  });
  const md = toMarkdown(withSpeakers(doc, nameSpeaker(newEdits(doc), speakers, 2, 'Dana')));
  assert.match(md, /2 speakers/);
  assert.match(md, /\*\*\[00:00\] Speaker 1:\*\* Did you see it\?\n\n\*\*\[00:02\] Dana:\*\* Yes, twice\./);
  assert.match(markdownToHtml(md, () => undefined), /<button class="ts" data-t="2">00:02<\/button> <b class="who">Dana:<\/b> Yes, twice\./);
});

// ---- edits: names, merges, moves (<video>.speakers.json) ----

const unit = (k: number, j = 0) => {
  const v = Array(8).fill(0);
  v[k] = 1;
  v[(k + 1) % 8] = j;
  const n = Math.hypot(...v);
  return v.map((x) => x / n);
};
const SRC = { name: 'a.mp4', fingerprint: 'v0:1:x', size: 1, duration: 30, width: 640, height: 360, fps: 25, frames: 750, video: { codec: 'avc1' } };
/** A doc with three speakers taking turns over 30 s, and a transcript of one word per second. */
function talk(speakers: SpeakerItem[]) {
  const words: Word[] = Array.from({ length: 30 }, (_, i) => [i + 0.1, i + 0.6, ` w${i}`]);
  const items = [0, 10, 20].map((t) => ({ t0: t, t1: t + 10, text: words.slice(t, t + 10).map((w) => w[2]).join(''), words: words.slice(t, t + 10) }));
  return withLayers(newDoc(SRC), {
    transcript: { producer: { configHash: 'h', config: {} }, items } as unknown as Layer<never>,
    speakers: { producer: { configHash: 's', config: {} }, items: speakers } as unknown as Layer<never>,
  });
}
const three = (): SpeakerItem[] => [
  { id: 1, turns: [[0, 10]], voice: unit(0) },
  { id: 2, turns: [[10, 20]], voice: unit(2) },
  { id: 3, turns: [[20, 25]], voice: unit(4) },
];

test('speaker edits: names, a merge and moves give the doc as edited; the doc itself is unchanged', () => {
  const doc = talk([...three(), { id: 4, turns: [[25, 30]], voice: unit(2, 0.2) }]);
  const items = doc.layers.speakers!.items;
  let e = newEdits(doc);
  e = nameSpeaker(e, items, 2, 'Dana');
  e = mergeSpeakers(e, items, 4, 2);
  assert.throws(() => nameSpeaker(e, items, 1, 'dana'), /merge them instead/);
  e = moveSpeech(e, items, 12, 14, 1, 2);
  const guest = addSpeaker(e, items, 'Guest');
  e = moveSpeech(guest.edits, items, 22, 23, guest.id);
  const s = withSpeakers(doc, e).layers.speakers!.items;
  assert.deepEqual(s.map((x) => [x.id, x.name, x.turns, x.merged]), [
    [1, undefined, [[0, 10], [12, 14]], undefined],
    [2, 'Dana', [[10, 12], [14, 20], [25, 30]], [4]],
    [3, undefined, [[20, 22], [23, 25]], undefined],
    [5, 'Guest', [[22, 23]], undefined],
  ]);
  assert.equal(doc.layers.speakers!.items.length, 4, 'the doc keeps its speakers');
  assert.equal(findSpeaker(s, 'dana')?.id, 2);
  assert.equal(findSpeaker(s, 'Speaker 4')?.id, 2, 'a merged id finds the speaker it joined');
  assert.equal(findSpeaker(s, '3')?.id, 3);
  // The Markdown, search and the summary read the names.
  const md = toMarkdown(withSpeakers(doc, e));
  assert.match(md, /4 speakers/);
  assert.match(md, /\*\*\[00:10\] Dana:\*\* w10 w11\n\n\*\*\[00:12\] Speaker 1:\*\* w12 w13\n\n\*\*\[00:14\] Dana:\*\*/);
  assert.match(md, /\*\*\[00:22\] Guest:\*\* w22/);
  const hit = search([withSpeakers(doc, e)], 'w15')[0];
  assert.equal(hit.speaker, 'Dana');
  assert.deepEqual(search([withSpeakers(doc, e)], 'w12 w15 w27', { speaker: 'Dana' }).map((h) => [h.t0, h.matched.sort()]), [[14.1, ['w15', 'w27']]], 'only what Dana says');
  assert.equal(summarize(withSpeakers(doc, e)).speakers.named, 2);
  // Undo.
  assert.deepEqual(applyEdits(items, unmove(unmergeSpeaker(e, 4), 0)).map((x) => x.id), [1, 2, 3, 4, 5]);
  // Saved and read back.
  const back = checkEdits(JSON.parse(formatEdits(e)), doc);
  assert.deepEqual(back.warnings, []);
  assert.deepEqual(back.edits, e);
  assert.throws(() => checkEdits({ ...e, source: { name: 'b.mp4', fingerprint: 'v0:2:y' } }, doc), /another file/);
});

test('speaker edits follow the voices when the speakers are found again, and drop what has no voice now', () => {
  const doc = talk(three());
  let e = newEdits(doc);
  e = nameSpeaker(e, three(), 1, 'Ann');
  e = nameSpeaker(e, three(), 3, 'Cal');
  e = mergeSpeakers(e, three(), 2, 1);
  const guest = addSpeaker(e, three(), 'Guest');
  e = moveSpeech(guest.edits, three(), 5, 6, guest.id, 1);
  // Found again: ids renumbered, voices a little different, Cal's voice gone, a fourth speaker.
  const again = talk([
    { id: 1, turns: [[0, 10]], voice: unit(2, 0.1) },
    { id: 2, turns: [[10, 20]], voice: unit(0, 0.1) },
    { id: 3, turns: [[20, 25]], voice: unit(6) },
    { id: 4, turns: [[25, 30]], voice: unit(7) },
  ]);
  const r = checkEdits(JSON.parse(formatEdits(e)), again);
  assert.equal(r.warnings.length, 1);
  assert.match(r.warnings[0], /^Cal: no speaker found now has that voice/);
  const s = applyEdits(again.layers.speakers!.items, r.edits);
  assert.deepEqual(s.map((x) => [x.id, x.name, x.turns, x.merged]), [
    [2, 'Ann', [[0, 5], [6, 20]], [1]],
    [3, undefined, [[20, 25]], undefined],
    [4, undefined, [[25, 30]], undefined],
    [5, 'Guest', [[5, 6]], undefined],
  ]);
});

test('names suggested from voices named elsewhere: the closest within SAME_VOICE, one each, never a name in use', () => {
  const s: SpeakerItem[] = [
    { id: 1, turns: [[0, 1]], voice: unit(0, 0.1) },
    { id: 2, turns: [[1, 2]], voice: unit(0, 0.3) },
    { id: 3, turns: [[2, 3]], voice: unit(4), name: 'Bo' },
    { id: 4, turns: [[3, 4]], voice: unit(6) },
  ];
  const known = [
    { name: 'Ann', voice: unit(0), video: 'a.mp4' },
    { name: 'Bo', voice: unit(6), video: 'b.mp4' },
    { name: 'Ann', voice: unit(0, 0.05), video: 'c.mp4' },
  ];
  assert.deepEqual(suggestSpeakerNames(s, known).map((x) => [x.id, x.name, x.video]), [[1, 'Ann', 'c.mp4']]);
  assert.ok(SAME_VOICE > 0.3 && SAME_VOICE < 0.6);
  const e = nameSpeaker(newEdits(talk(three())), three(), 2, 'Dana');
  assert.deepEqual(namedVoices(e, 'x.mp4'), [{ name: 'Dana', voice: unit(2), video: 'x.mp4' }]);
});

test('redaction bleeps every turn of a speaker, by name', () => {
  const doc = talk(three());
  const e = nameSpeaker(newEdits(doc), three(), 2, 'Dana');
  const log = redactionLog(withSpeakers(doc, e), { ...REDACT_DEFAULTS, cover: [], speakers: ['Dana'] });
  assert.deepEqual(log.audio?.spans.map((x) => [x.t0, x.t1, x.speaker]), [[9.75, 20.25, 'Dana']]);
  assert.equal(log.audio?.totals.turns, 1);
  assert.match(log.caveat, /diarization/);
  assert.throws(() => redactionLog(doc, { ...REDACT_DEFAULTS, cover: [], speakers: ['Dana'] }), /no speaker 'Dana'/);
});
