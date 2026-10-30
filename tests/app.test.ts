// The app's renderer: the Markdown agents read, as HTML with seek targets. Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { decoderOf, readPreset } from '../src/app/presets.ts';
import { analysisSteps, estimate, markdownToHtml, secs, stepOf, suggestions, trackCost } from '../src/app/render.ts';
import { highlight, type Layer, newDoc, toMarkdown, withLayers } from '../src/doc/index.ts';

const layer = <T>(items: T[]) => ({ producer: { configHash: 'h', config: {} }, items }) as unknown as Layer<T>;
const doc = withLayers(newDoc({ name: 'a <b>.mp4', fingerprint: 'v0:1:x', size: 1, duration: 70, width: 640, height: 360, fps: 25, frames: 1750, video: { codec: 'avc1' } }), {
  speech: layer([{ t0: 0, t1: 70 }]),
  transcript: layer([{ t0: 61, t1: 65, text: 'The zoning board approved it.' }]),
  scenes: layer([
    { t0: 0, t1: 1, f0: 0, f1: 24, score: 0 },
    { t0: 1, t1: 70, f0: 25, f1: 1749, score: 0.5 },
  ]),
  keyframes: layer([{ t: 1.5, frame: 37, scene: 1, why: 'scene' }]),
});

test('render: header, compact empty shots, timed paragraphs, keyframes, escaping', () => {
  const html = markdownToHtml(toMarkdown(doc, { image: (k) => `kf:${k.frame}` }), (f) => (f === 37 ? 'blob:x' : undefined));
  assert.match(html, /<h1>a &lt;b&gt;\.mp4<\/h1>\n<p class="facts">01:10 · 640×360/);
  assert.match(html, /<p class="made"[^>]*>videodoc\/0 · speech · transcript · scenes · keyframes<\/p>/);
  assert.match(html, /<h2 class="bare" data-t="0">Shot 1 · 00:00–00:01<\/h2>/);
  assert.match(html, /<h2 data-t="1">Shot 2 · 00:01–01:10<\/h2>/);
  assert.match(html, /<figure class="kf" data-t="1" data-frame="37"><img src="blob:x"/);
  assert.match(html, /<p class="said" data-t="61"><button class="ts" data-t="61">01:01<\/button> The zoning board approved it\.<\/p>/);
  // Search terms show as <mark>.
  const marked = markdownToHtml(
    toMarkdown(doc),
    () => undefined,
    (t) => highlight(t, new Set(['zon', 'board'])),
  );
  assert.match(marked, /The <mark>zoning<\/mark> <mark>board<\/mark> approved/);
  assert.equal(secs('1:01:23'), 3683);
});

test('guidance: estimates, analysis steps, search suggestions', () => {
  assert.equal(estimate(30, { transcript: true, faces: false }), 'under a minute');
  assert.equal(estimate(3600, { transcript: true, faces: false }), 'about 16 min');
  assert.equal(estimate(3 * 3600, { transcript: true, faces: true }), 'about 76 min');
  assert.equal(estimate(6 * 3600, { transcript: true, faces: true }), 'about 2.5 h');
  assert.deepEqual(analysisSteps(['speech', 'scenes', 'keyframes', 'transcript'], new Set()), ['Listening', 'Transcribing', 'Finding shots']);
  // Adding faces to an analyzed video: only the frames pass runs.
  const steps = analysisSteps(['speech', 'scenes', 'keyframes', 'transcript', 'faces'], new Set(['speech', 'transcript', 'scenes', 'keyframes']));
  assert.deepEqual(steps, ['Finding faces']);
  assert.deepEqual(analysisSteps(['scenes', 'faces', 'plates'], new Set(['scenes', 'faces'])), ['Finding plates']);
  // A preset's track layers are found in the same pass.
  assert.deepEqual(analysisSteps(['scenes', 'faces', 'plates', 'vehicles'], new Set(['scenes']), ['faces', 'plates', 'vehicles']), ['Finding faces, plates and vehicles']);
  assert.equal(estimate(3600, { transcript: true, faces: false, extra: trackCost('detr') }), 'about 9.3 h');
  assert.equal(trackCost('yolo'), trackCost('yolo-end2end'));
  assert.equal(stepOf({ pass: 'frames', stage: 'scenes + plates' }, ['Finding plates', 'Blurring']), 0);
  const full = ['Listening', 'Transcribing', 'Finding shots'];
  assert.equal(stepOf({ pass: 'probe', stage: 'reading container' }, full), -1);
  assert.equal(stepOf({ pass: 'audio', stage: 'voice activity' }, full), 0);
  assert.equal(stepOf({ pass: 'audio', stage: 'loading whisper-base' }, full), 1);
  assert.equal(stepOf({ pass: 'frames', stage: 'scenes' }, full), 2);
  // Names said mid-sentence, at least twice, taking turns between videos; sentence starts don't count.
  const s = suggestions([
    ['Then we went to Michigan and back to Michigan.', 'Detroit is in it, said the Governor of Michigan.', 'Tariffs hurt Detroit and Detroit.'],
    ['The Senate in Alaska met.', 'Alaska again, and Alaska.'],
  ]);
  assert.deepEqual(s, ['Michigan', 'Alaska', 'Detroit']);
});

test('presets: local model files become object URLs; missing ones say how to supply them', async () => {
  const text = JSON.stringify({
    schema: 'videodoc-pipeline/0',
    name: 'dashcam',
    layers: ['faces', 'vehicles', 'logos'],
    models: { logos: { url: './models/logos.onnx', decoder: 'yolo', labels: ['logo'] }, web: { url: 'https://example.com/m.onnx', decoder: 'detr' } },
    tracks: { vehicles: { model: 'rfdetr', labels: 'car,truck' }, logos: { model: 'logos' } },
  });
  const asked: string[] = [];
  const p = await readPreset(text, 'dashcam.json', async (path) => (asked.push(path), new Blob(['onnx'])), 'pick them');
  assert.deepEqual(asked, ['./models/logos.onnx'], 'only local paths are looked up');
  assert.match(p.run.models!.logos.url, /^blob:/);
  assert.equal(p.run.models!.web.url, 'https://example.com/m.onnx');
  assert.equal(p.pipeline.models!.logos.url, './models/logos.onnx', 'the preset keeps its paths, for saving');
  assert.equal(decoderOf(p.pipeline, 'vehicles'), 'detr', 'built-in models are known');
  assert.equal(decoderOf(p.pipeline, 'logos'), 'yolo');
  await assert.rejects(
    readPreset(text, 'dashcam.json', async () => undefined, 'pick them'),
    /model file \.\/models\/logos\.onnx not found; pick them/,
  );
  await assert.rejects(
    readPreset('{"schema": "x"', 'bad.json', async () => undefined, ''),
    /bad\.json: not JSON/,
  );
});
