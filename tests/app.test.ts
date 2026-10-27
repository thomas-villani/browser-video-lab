// The app's renderer: the Markdown agents read, as HTML with seek targets. Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { highlight, newDoc, toMarkdown, withLayers, type Layer } from '../src/doc/index.ts';
import { markdownToHtml, secs } from '../src/app/render.ts';

const layer = <T>(items: T[]) => ({ producer: { configHash: 'h', config: {} }, items }) as unknown as Layer<T>;
const doc = withLayers(
  newDoc({ name: 'a <b>.mp4', fingerprint: 'v0:1:x', size: 1, duration: 70, width: 640, height: 360, fps: 25, frames: 1750, video: { codec: 'avc1' } }),
  {
    speech: layer([{ t0: 0, t1: 70 }]),
    transcript: layer([{ t0: 61, t1: 65, text: 'The zoning board approved it.' }]),
    scenes: layer([
      { t0: 0, t1: 1, f0: 0, f1: 24, score: 0 },
      { t0: 1, t1: 70, f0: 25, f1: 1749, score: 0.5 },
    ]),
    keyframes: layer([{ t: 1.5, frame: 37, scene: 1, why: 'scene' }]),
  },
);

test('render: header, compact empty shots, timed paragraphs, keyframes, escaping', () => {
  const html = markdownToHtml(toMarkdown(doc, { image: (k) => `kf:${k.frame}` }), (f) => (f === 37 ? 'blob:x' : undefined));
  assert.match(html, /<h1>a &lt;b&gt;\.mp4<\/h1>\n<p class="facts">01:10 · 640×360/);
  assert.match(html, /<p class="made"[^>]*>videodoc\/0 · speech · transcript · scenes · keyframes<\/p>/);
  assert.match(html, /<h2 class="bare" data-t="0">Shot 1 · 00:00–00:01<\/h2>/);
  assert.match(html, /<h2 data-t="1">Shot 2 · 00:01–01:10<\/h2>/);
  assert.match(html, /<figure class="kf" data-t="1" data-frame="37"><img src="blob:x"/);
  assert.match(html, /<p class="said" data-t="61"><button class="ts" data-t="61">01:01<\/button> The zoning board approved it\.<\/p>/);
  // Search terms show as <mark>.
  const marked = markdownToHtml(toMarkdown(doc), () => undefined, (t) => highlight(t, new Set(['zon', 'board'])));
  assert.match(marked, /The <mark>zoning<\/mark> <mark>board<\/mark> approved/);
  assert.equal(secs('1:01:23'), 3683);
});
