// Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { geometryFor, yunetFace } from '../src/models.ts';
import { SceneScorer } from '../src/scene.ts';
import { bisect, intervals, type AdaptiveOptions, type Sampled } from '../src/track/adaptive.ts';
import type { Box, Detection } from '../src/track/bytetrack.ts';

test('intervals: stride grid, both sides of each cut, nothing spans a cut', () => {
  assert.deepEqual(intervals(40, { maxStride: 16 }), [[0, 16], [16, 32], [32, 39]]);
  assert.deepEqual(intervals(40, { maxStride: 16, cuts: [20] }), [[0, 16], [16, 19], [20, 32], [32, 39]]);
  // A one-frame scene (cuts at 10 and 11) is still solved, as the interval [10, 10].
  assert.deepEqual(intervals(20, { maxStride: 16, cuts: [10, 11] }), [[0, 9], [10, 10], [11, 16], [16, 19]]);
});

test('intervals: a chunk solved locally matches the global grid (streaming == offline)', () => {
  const S = 16;
  const cuts = [5, 16, 17, 40];
  const global = intervals(60, { maxStride: S, cuts });
  const local: [number, number][] = [];
  for (let a = 0; a < 59; a += S) {
    const g = Math.min(a + S, 59);
    const lc = cuts.filter((c) => c > a && c <= g).map((c) => c - a);
    for (const [x, y] of intervals(g - a + 1, { maxStride: S, cuts: lc })) local.push([a + x, a + y]);
  }
  // A cut on a chunk boundary is solved as [f, f] by both chunks; the cache makes that free.
  const dedupe = (iv: [number, number][]) => [...new Map(iv.map((x) => [x.join(), x])).values()];
  assert.deepEqual(dedupe(local), global);
});

test('bisect: yields one batch per level, only new frames', () => {
  const opts: AdaptiveOptions = { maxStride: 16, minIou: 0.3, minScore: 0.5 };
  // A face that jumps at frame 11: every level until the jump is isolated disagrees there.
  const det = (f: number): Detection[] => [{ box: (f < 11 ? [0, 0, 50, 50] : [300, 0, 50, 50]) as Box, score: 0.9 }];
  const gen = bisect(0, 16, opts, new Map<number, Sampled>());
  const batches: number[][] = [];
  let step = gen.next();
  while (!step.done) {
    batches.push(step.value);
    step = gen.next(step.value.map(det));
  }
  assert.deepEqual(batches, [[0, 16], [8], [12], [10], [11]]);
  assert.deepEqual(step.value[10], [[0, 0, 50, 50]]);
  assert.deepEqual(step.value[11], [[300, 0, 50, 50]]);
});

test('scene scorer: a cut spikes, a steady pan does not', () => {
  const s = new SceneScorer();
  const frame = (v: number) => new Float32Array(48).fill(v);
  const scores = [0.1, 0.12, 0.14, 0.16, 0.9, 0.9].map((v) => s.push(frame(v)));
  // Steady change of 0.02/frame: mafd 2%, but |mafd - prev| ~ 0 after the first step.
  assert.ok(scores[3] < 0.01, `pan ${scores[3]}`);
  assert.ok(scores[4] > 0.5, `cut ${scores[4]}`);
  assert.equal(scores[5], 0);
});

test('geometry: dynamic input fits 16:9 tightly, crops map back to the frame', () => {
  const g = geometryFor(yunetFace, 1920, 1080, { longSide: 640 });
  assert.deepEqual([g.inW, g.inH, g.contentW, g.contentH], [640, 384, 640, 360]);
  const roi = geometryFor(yunetFace, 1920, 1080, { crop: { x: 1000, y: 200, w: 300, h: 300 }, longSide: 160 });
  assert.deepEqual([roi.inW, roi.inH], [160, 160]);
  // A box decoded at the input's top-left corner lands at the crop's origin in frame pixels.
  const out: Record<string, Float32Array> = {};
  for (const s of [8, 16, 32]) {
    const cells = (160 / s) ** 2;
    out[`cls_${s}`] = new Float32Array(cells);
    out[`obj_${s}`] = new Float32Array(cells);
    out[`bbox_${s}`] = new Float32Array(4 * cells);
  }
  out.cls_8[0] = out.obj_8[0] = 1; // cell (0,0): center (0,0) + size exp(0) * 8 = 8 px
  const [d] = yunetFace.decode(out, roi);
  const k = 300 / 160;
  assert.deepEqual([d.x, d.y, d.w, d.h].map((v) => Math.round(v * 100) / 100), [1000 - 4 * k, 200 - 4 * k, 8 * k, 8 * k].map((v) => Math.round(v * 100) / 100));
});
