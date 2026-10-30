// Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { geometryFor, yunetFace } from '../src/models.ts';
import { activeTiles, mergeBoxes, ownBoxes, refinedFor, tileGrid } from '../src/runtime/refine.ts';
import { SceneScorer } from '../src/scene.ts';
import { type AdaptiveOptions, bisect, intervals, type Sampled } from '../src/track/adaptive.ts';
import type { Box, Detection } from '../src/track/bytetrack.ts';
import { DEFAULT_ROI, roiBoxes } from '../src/track/roi.ts';

test('intervals: stride grid, both sides of each cut, nothing spans a cut', () => {
  assert.deepEqual(intervals(40, { maxStride: 16 }), [
    [0, 16],
    [16, 32],
    [32, 39],
  ]);
  assert.deepEqual(intervals(40, { maxStride: 16, cuts: [20] }), [
    [0, 16],
    [16, 19],
    [20, 32],
    [32, 39],
  ]);
  // A one-frame scene (cuts at 10 and 11) is still solved, as the interval [10, 10].
  assert.deepEqual(intervals(20, { maxStride: 16, cuts: [10, 11] }), [
    [0, 9],
    [10, 10],
    [11, 16],
    [16, 19],
  ]);
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
  assert.deepEqual(
    [d.x, d.y, d.w, d.h].map((v) => Math.round(v * 100) / 100),
    [1000 - 4 * k, 200 - 4 * k, 8 * k, 8 * k].map((v) => Math.round(v * 100) / 100),
  );
});

test('roi: an entering face is tracked backward with crops, without full-frame calls', () => {
  // A static face throughout; a second face enters at frame 5 and walks right 10 px/frame.
  const dets = (f: number): Detection[] => {
    const ds: Detection[] = [{ box: [100, 100, 50, 50], score: 0.9 }];
    if (f >= 5) ds.push({ box: [500 + 10 * f, 300, 60, 60], score: 0.9 });
    return ds;
  };
  const inCrop = (f: number, c?: Box) => dets(f).filter(({ box: [x, y, w, h] }) => !c || (x >= c[0] && y >= c[1] && x + w <= c[0] + c[2] && y + h <= c[1] + c[3]));
  const r = roiBoxes(9, (q) => inCrop(q.frame, q.crop), { ...DEFAULT_ROI, window: 8, frameW: 1920, frameH: 1080 });
  assert.equal(r.fullCalls, 2);
  assert.ok(r.roiCrops >= 3 && r.roiCrops <= 4, `crops ${r.roiCrops}`);
  for (let f = 0; f < 9; f++) assert.equal(r.boxes[f].length, f >= 5 ? 2 : 1, `frame ${f}`);
});

test('refine: tile grid covers the frame with the requested overlap', () => {
  const tiles = tileGrid(1920, 1080, 640, 384, 32);
  assert.equal(tiles.length, 12); // 4 columns x 3 rows
  const xs = [...new Set(tiles.map((t) => t[0]))];
  const ys = [...new Set(tiles.map((t) => t[1]))];
  assert.deepEqual([xs[0], xs.at(-1)! + 640, ys[0], ys.at(-1)! + 384], [0, 1920, 0, 1080]);
  for (let i = 1; i < xs.length; i++) assert.ok(xs[i - 1] + 640 - xs[i] >= 32);
  for (let i = 1; i < ys.length; i++) assert.ok(ys[i - 1] + 384 - ys[i] >= 32);
  assert.deepEqual(tileGrid(640, 360, 640, 384, 32), [[0, 0, 640, 360]]);
});

test('refine: active tiles, owned boxes, merging, and reuse across frames', () => {
  const tiles = tileGrid(1920, 1080, 640, 384, 32);
  const face: Box = [100, 100, 20, 20];
  assert.equal(activeTiles(tiles, [face], 0).length, 1);
  assert.ok(activeTiles(tiles, [face], 600).length > 1);
  // A box centered in the tile is kept; one centered outside (cut by the edge) is not.
  assert.deepEqual(ownBoxes([0, 0, 640, 384], [face, [630, 100, 30, 30]]), [face]);
  assert.deepEqual(
    mergeBoxes(
      [face],
      [
        [101, 101, 20, 20],
        [300, 300, 20, 20],
      ],
    ),
    [face, [300, 300, 20, 20]],
  );
  // A tile that cuts a close-up sees part of it as a smaller face: not a new face.
  assert.deepEqual(mergeBoxes([[800, 100, 560, 740]], [[900, 200, 300, 280]]), [[800, 100, 560, 740]]);
  const noCut = () => false;
  assert.deepEqual(refinedFor(8, 2, 16, noCut), [8]);
  assert.deepEqual(refinedFor(9, 2, 16, noCut), [8, 10]);
  assert.deepEqual(refinedFor(15, 2, 15, noCut), [14]);
  // A cut at frame 10 separates frame 9 from refined frame 10.
  assert.deepEqual(
    refinedFor(9, 2, 16, (a, b) => a < 10 && 10 <= b),
    [8],
  );
});
