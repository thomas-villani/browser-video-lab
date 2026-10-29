// Speaker diarization, the pure part (src/audio): powerset decoding, counting, clustering,
// assignment, reconstruction, and the DER used to measure it. Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assignMax, cluster, decodePowerset, diarize, DIARIZE_DEFAULTS, FRAME_STEP, SEG, speakerCount, windowStarts, type WindowSeg } from '../src/audio/diarize.ts';
import { der, hungarianMax, parseRttm } from '../src/audio/der.ts';
import { markdownToHtml } from '../src/app/render.ts';
import { newDoc, speakerAt, toMarkdown, whoSaid, withLayers, type Layer, type Word } from '../src/doc/index.ts';

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
  assert.deepEqual(whoSaid(items, speakers), [
    { t0: 0, t1: 1.6, text: 'Did you see it?', speaker: 1 },
    { t0: 2.3, t1: 3.9, text: 'Yes, twice.', speaker: 2 },
  ]);
  assert.equal(speakerAt(speakers, 1.9, 2.5), 2);
  assert.equal(speakerAt(speakers, 6, 7), undefined);
  const doc = withLayers(newDoc({ name: 'a.mp4', fingerprint: 'v0:1:x', size: 1, duration: 5, width: 640, height: 360, fps: 25, frames: 125, video: { codec: 'avc1' } }), {
    transcript: { producer: { configHash: 'h', config: {} }, items } as unknown as Layer<never>,
    speakers: { producer: { configHash: 'h', config: {} }, items: speakers } as unknown as Layer<never>,
  });
  const md = toMarkdown(doc, { names: { 2: 'Dana' } });
  assert.match(md, /2 speakers/);
  assert.match(md, /\*\*\[00:00\] Speaker 1:\*\* Did you see it\?\n\n\*\*\[00:02\] Dana:\*\* Yes, twice\./);
  assert.match(markdownToHtml(md, () => undefined), /<button class="ts" data-t="2">00:02<\/button> <b class="who">Dana:<\/b> Yes, twice\./);
});
