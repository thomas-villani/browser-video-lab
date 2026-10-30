// Every layer computed in headless Edge on the fixtures, with figures pinned in baseline.json.
// Run: npm run test:browser (cut the local fixtures first: node scripts/fixtures.mjs).
import assert from 'node:assert/strict';
import { copyFileSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { after, test } from 'node:test';
import { analyze, ROOT } from '../../cli/lib.ts';
import { der, parseRttm } from '../../src/audio/der.ts';
import { type Layer, normWord, REDACT_DEFAULTS, redactionLog, summarize, type Track, toMarkdown } from '../../src/doc/index.ts';
import { context, fixture, fresh, identical, LOCAL_FIXTURES, loadDoc, near, notReady, saveBaseline, speed } from './lib.ts';

const ctx = context();
after(async () => {
  saveBaseline();
  await ctx.browser.close();
});

/** Longest common subsequence of two token lists: matched index pairs. */
function align(a: string[], b: string[]): [number, number][] {
  const L = Array.from({ length: a.length + 1 }, () => new Int32Array(b.length + 1));
  for (let i = a.length - 1; i >= 0; i--) for (let j = b.length - 1; j >= 0; j--) L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out: [number, number][] = [];
  for (let i = 0, j = 0; i < a.length && j < b.length; ) {
    if (a[i] === b[j]) out.push([i++, j++]);
    else if (L[i + 1][j] >= L[i][j + 1]) i++;
    else j++;
  }
  return out;
}

const median = (xs: number[]) => [...xs].sort((x, y) => x - y)[Math.floor(xs.length / 2)];
const r3 = (x: number) => Math.round(x * 1000) / 1000;

test('tts-words: speech, and a transcript whose words and word times match the TTS ground truth', async (t) => {
  const why = await notReady(ctx);
  if (why) return t.skip(why);
  const video = fresh(fixture('tts-words')!);
  await analyze(ctx, video, { layers: ['speech', 'transcript'] });
  const doc = loadDoc(video);
  // SAPI positions count at 16 kHz while writing 22.05 kHz (tests/fixtures/tts.ps1).
  const truth = readFileSync(join(ROOT, 'tests/fixtures/tts-words.tsv'), 'utf8')
    .trim()
    .split(/\r?\n/)
    .map((l) => {
      const [s, w] = l.split('\t');
      return { t: (Number(s) * 16000) / 22050, w: normWord(w) };
    });
  const words = doc.layers.transcript!.items.flatMap((i) => i.words ?? []);
  assert.ok(words.length > 100, `word times: ${words.length} words`);
  const pairs = align(
    truth.map((x) => x.w),
    words.map((w) => normWord(w[2])),
  );
  const err = pairs.map(([i, j]) => words[j][0] - truth[i].t);
  // Covered whole by the redaction defaults (lead 0.4, tail 0.2): from before the true start to
  // the next word's start (0.5 s at most, after pauses).
  const covered = pairs.filter(
    ([i, j]) => i + 1 < truth.length && words[j][0] - 0.4 <= truth[i].t + 0.03 && words[j][1] + 0.2 >= Math.min(truth[i + 1].t - 0.08, truth[i].t + 0.5),
  ).length;
  t.diagnostic(`matched ${pairs.length}/${truth.length} words, start error median ${r3(median(err))} s, covered ${covered}`);
  assert.ok(pairs.length / truth.length > 0.8, 'most words transcribed as spoken');
  assert.ok(
    err.every((e) => e > -0.3 && e < 1),
    'word starts near the truth',
  );
  near(t, 'tts-words.speech.seconds', summarize(doc).speech.speechSeconds as number, { abs: 1 });
  near(t, 'tts-words.transcript.matched', pairs.length, { abs: 3 });
  near(t, 'tts-words.transcript.startErrorMedian', r3(median(err)), { abs: 0.03 });
  near(t, 'tts-words.transcript.coveredWhole', covered, { abs: 3 });
  identical(t, 'tts-words.transcript', doc.layers.transcript!.items);
  speed(t, 'tts-words.transcript.realtime', +(doc.source.duration / (doc.layers.transcript!.ms / 1000)).toFixed(2));
});

test('tts-dialog: three voices taking turns are three speakers, and the Markdown says who speaks', async (t) => {
  const why = await notReady(ctx);
  if (why) return t.skip(why);
  const video = fresh(fixture('tts-dialog')!);
  await analyze(ctx, video, { layers: ['transcript', 'speakers'] });
  const doc = loadDoc(video);
  const speakers = doc.layers.speakers!.items;
  const ref = parseRttm(readFileSync(join(ROOT, 'tests/fixtures/tts-dialog.rttm'), 'utf8'));
  const hyp = speakers.flatMap((s) => s.turns.map(([t0, t1]) => ({ t0, t1, speaker: String(s.id) })));
  const r = der(ref, hyp, [0, doc.source.duration]);
  t.diagnostic(
    `${speakers.length} speakers, DER ${(100 * r.der).toFixed(1)}% (missed ${r.missed.toFixed(1)} s, false ${r.falseAlarm.toFixed(1)} s, confused ${r.confusion.toFixed(1)} s), mapping ${JSON.stringify(r.mapping)}`,
  );
  assert.equal(speakers.length, 3);
  assert.equal(new Set(Object.values(r.mapping)).size, 3, 'each voice is its own speaker');
  assert.ok(r.der < 0.15, 'turns match the script');
  assert.equal(speakers[0].voice.length, 256);
  const md = toMarkdown(doc);
  assert.match(md, /\*\*\[00:0\d\] Speaker 1:\*\* Welcome back/);
  assert.match(md, /Speaker 3:\*\* I agree completely/);
  assert.match(md, /Speaker 2:\*\* It really is\./);
  near(t, 'tts-dialog.speakers.der', +(100 * r.der).toFixed(1), { abs: 2 });
  identical(
    t,
    'tts-dialog.speakers',
    speakers.map((s) => s.turns),
  );
  speed(t, 'tts-dialog.speakers.realtime', +(doc.source.duration / (doc.layers.speakers!.ms / 1000)).toFixed(2));
});

for (const name of LOCAL_FIXTURES) {
  test(`${name}: scenes, faces, plates and keyframes`, async (t) => {
    const why = await notReady(ctx);
    if (why) return t.skip(why);
    const path = fixture(name);
    if (!path) return t.skip(`fixture not cut: node scripts/fixtures.mjs (needs public/media footage)`);
    const video = fresh(path);
    await analyze(ctx, video, { layers: ['scenes', 'faces', 'plates', 'keyframes'] });
    const doc = loadDoc(video);
    const sum = summarize(doc);
    const { scenes, faces, plates, keyframes } = doc.layers;
    assert.ok(scenes && faces && plates && keyframes, 'all layers present');
    // Scenes tile the video.
    assert.equal(scenes.items[0].f0, 0);
    assert.equal(scenes.items.at(-1)!.f1, doc.source.frames - 1);
    for (let i = 1; i < scenes.items.length; i++) assert.equal(scenes.items[i].f0, scenes.items[i - 1].f1 + 1);
    // Tracks stay inside the frame and the video.
    for (const tr of [...faces.items, ...plates.items]) {
      assert.ok(tr.f0 >= 0 && tr.f1 < doc.source.frames && tr.f0 <= tr.f1);
      for (const [, x, y, w, h] of tr.samples) assert.ok(w > 0 && h > 0 && x > -w && y > -h && x < doc.source.width && y < doc.source.height);
    }
    near(t, `${name}.scenes.cuts`, scenes.items.length - 1, { abs: 1 });
    near(t, `${name}.keyframes`, keyframes.items.length, { abs: 2 });
    near(t, `${name}.faces.tracks`, faces.items.length, { rel: 0.05, abs: 2 });
    near(t, `${name}.faces.faceFrames`, sum.faces.faceFrames as number, { rel: 0.03, abs: 10 });
    near(t, `${name}.plates.tracks`, plates.items.length, { rel: 0.1, abs: 2 });
    near(t, `${name}.plates.plateFrames`, sum.plates.plateFrames as number, { rel: 0.05, abs: 10 });
    for (const l of ['scenes', 'faces', 'plates', 'keyframes'] as const) identical(t, `${name}.${l}`, doc.layers[l]!.items);
    speed(t, `${name}.faces.fps`, Math.round(doc.source.frames / (faces.ms / 1000)));

    // A second run reads everything from the doc.
    const again = (await analyze(ctx, video, { layers: ['scenes', 'faces', 'plates', 'keyframes'] })) as { layers: Record<string, { status: string }> };
    for (const [l, s] of Object.entries(again.layers)) assert.equal(s.status, 'cached', l);
  });
}

test("rickroll-cuts: a pipeline's track layer, on a model file next to the preset, is computed, cached and redactable", async (t) => {
  const why = await notReady(ctx);
  if (why) return t.skip(why);
  const path = fixture('rickroll-cuts');
  if (!path) return t.skip('fixture not cut: node scripts/fixtures.mjs');
  const video = fresh(path);
  // The model is a copy of YuNet beside the preset, named by a relative path: the CLI serves it to the page.
  const dir = dirname(video);
  copyFileSync(join(ROOT, 'public/models/yunet_face_dyn.onnx'), join(dir, 'heads.onnx'));
  const preset = join(dir, 'heads.json');
  writeFileSync(
    preset,
    JSON.stringify({
      schema: 'videodoc-pipeline/0',
      layers: ['heads'],
      models: { heads: { url: './heads.onnx', decoder: 'yunet', labels: ['face'], name: 'yunet copy' } },
      tracks: { heads: { model: 'heads', maxStride: 8, minScore: 0.6 } },
    }),
  );
  await analyze(ctx, video, { pipeline: preset });
  const doc = loadDoc(video);
  const heads = doc.layers.heads as Layer<Track>;
  assert.equal(heads.producer.kind, 'track');
  assert.equal(heads.producer.model, 'yunet copy');
  assert.ok(heads.items.length > 0, 'heads found');
  near(t, 'rickroll-cuts.heads.tracks', heads.items.length, { rel: 0.05, abs: 2 });
  identical(t, 'rickroll-cuts.heads', heads.items);
  const again = (await analyze(ctx, video, { pipeline: preset })) as { layers: Record<string, { status: string }> };
  assert.deepEqual(
    Object.values(again.layers).map((s) => s.status),
    ['cached', 'cached'],
  );
  // Redaction covers it when listed.
  const log = redactionLog(doc, { ...REDACT_DEFAULTS, cover: ['heads'] });
  assert.equal(log.totals.tracks, heads.items.length);
  assert.ok(log.totals.framesWithRegions > 0);
});
