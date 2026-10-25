// Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { adaptiveBoxes } from '../src/track/adaptive.ts';
import { linearAssignment } from '../src/track/assignment.ts';
import { ByteTracker, type Box } from '../src/track/bytetrack.ts';
import { coverage } from '../src/track/coverage.ts';
import { fromXyah, initiate, predict, toXyah, update } from '../src/track/kalman.ts';
import { sparseBoxes } from '../src/track/sparse.ts';

test('assignment: picks the minimum-cost matching', () => {
  const cost = [
    [4, 1, 3],
    [2, 0, 5],
    [3, 2, 2],
  ];
  const r = linearAssignment(cost, 10, 3);
  const total = r.matches.reduce((s, [i, j]) => s + cost[i][j], 0);
  assert.equal(total, 5); // 1 + 2 + 2
});

test('assignment: zero rows still reports every column as unmatched', () => {
  assert.deepEqual(linearAssignment([], 0.8, 3).unmatchedCols, [0, 1, 2]);
});

test('assignment: respects the cost limit and handles rectangular input', () => {
  const r = linearAssignment([[0.1, 0.9], [0.95, 0.99], [0.2, 0.3]], 0.5, 2);
  assert.deepEqual(new Set(r.matches.map((m) => m.join())), new Set(['0,0', '2,1']));
  assert.deepEqual(r.unmatchedRows, [1]);
  assert.deepEqual(r.unmatchedCols, []);
});

test('kalman: xyah round-trips and learns constant velocity', () => {
  const box = [10, 20, 30, 60];
  assert.deepEqual(fromXyah(toXyah(box)).map(Math.round), box);
  let s = initiate(toXyah(box));
  for (let i = 1; i <= 20; i++) s = update(predict(s), toXyah([10 + 5 * i, 20, 30, 60]));
  const next = fromXyah(predict(s).mean.slice(0, 4));
  assert.ok(Math.abs(next[0] - (10 + 5 * 21)) < 1, `predicted x ${next[0]}`);
});

test('bytetrack: keeps identities for two crossing-free movers', () => {
  const tracker = new ByteTracker();
  let ids: number[] = [];
  for (let f = 0; f < 30; f++) {
    const out = tracker.update([
      { box: [100 + 3 * f, 100, 50, 50], score: 0.9 },
      { box: [400 - 3 * f, 300, 50, 50], score: 0.9 },
    ]);
    if (f === 0) ids = out.map((t) => t.id).sort();
    else assert.deepEqual(out.map((t) => t.id).sort(), ids);
  }
  assert.equal(ids.length, 2);
});

test('bytetrack: low-score detections keep a track alive (the BYTE step)', () => {
  const tracker = new ByteTracker();
  tracker.update([{ box: [100, 100, 50, 50], score: 0.9 }]);
  tracker.update([{ box: [102, 100, 50, 50], score: 0.9 }]);
  // Occlusion: the detector is unsure, but the box is where the track predicts.
  const out = tracker.update([{ box: [104, 100, 50, 50], score: 0.3 }]);
  assert.equal(out.length, 1);
  assert.equal(out[0].observed, true);
});

function mover(pxPerFrame: number, frames = 60) {
  const dets = Array.from({ length: frames }, (_, f) => [{ box: [pxPerFrame * f, 0, 80, 80] as Box, score: 0.9 }]);
  return { dets, gt: dets.map((ds) => ds.map((d) => d.box)) };
}

test('sparse offline: interpolation beats holding for a steady mover', () => {
  // 3 px/frame at stride 10 = 30 px between keyframes on an 80 px box (IoU ~0.45).
  const { dets, gt } = mover(3);
  const hold = coverage(gt, sparseBoxes(dets, { stride: 10, mode: 'hold', fps: 25 }));
  const offline = coverage(gt, sparseBoxes(dets, { stride: 10, mode: 'offline', fps: 25 }));
  assert.ok(offline.meanCoverage > 0.98, `offline ${offline.meanCoverage}`);
  assert.ok(hold.meanCoverage < offline.meanCoverage, `hold ${hold.meanCoverage}`);
});

test('adaptive: stable scenes stay sparse, changes get subdivided', () => {
  // A steady mover for 64 frames, then a second face appears at frame 40.
  const dets = Array.from({ length: 64 }, (_, f) => {
    const ds = [{ box: [f, 0, 80, 80] as Box, score: 0.9 }];
    if (f >= 40) ds.push({ box: [500, 500, 60, 60], score: 0.9 });
    return ds;
  });
  const gt = dets.map((ds) => ds.map((d) => d.box));
  const r = adaptiveBoxes(64, (f) => dets[f], { maxStride: 16, minIou: 0.3, minScore: 0.5 });
  assert.equal(coverage(gt, r.boxes).missRate, 0);
  assert.ok(r.sampled.length < 16, `sampled ${r.sampled.length} frames`);
  assert.ok(r.sampled.includes(40) || r.sampled.includes(39), 'bisection should find the entrance');
});

test('adaptive extendOneSided: entrances are held instead of bisected, still fully covered', () => {
  // A stable face plus a new face entering at frame 40 (same setup as above).
  const dets = Array.from({ length: 64 }, (_, f) => {
    const ds = [{ box: [f, 0, 80, 80] as Box, score: 0.9 }];
    if (f >= 40) ds.push({ box: [500, 500, 60, 60], score: 0.9 });
    return ds;
  });
  const gt = dets.map((ds) => ds.map((d) => d.box));
  const opts = { maxStride: 16, minIou: 0.3, minScore: 0.5 };
  const bisect = adaptiveBoxes(64, (f) => dets[f], opts);
  const extend = adaptiveBoxes(64, (f) => dets[f], { ...opts, extendOneSided: true });
  assert.equal(coverage(gt, extend.boxes).missRate, 0);
  assert.ok(extend.sampled.length < bisect.sampled.length, `${extend.sampled.length} vs ${bisect.sampled.length}`);
});

test('sparse offline: never drops detections even when association breaks', () => {
  // 12 px/frame at stride 10 = 120 px jumps: no IoU overlap, tracks never confirm.
  const { dets, gt } = mover(12);
  const hold = coverage(gt, sparseBoxes(dets, { stride: 10, mode: 'hold', fps: 25 }));
  const offline = coverage(gt, sparseBoxes(dets, { stride: 10, mode: 'offline', fps: 25 }));
  assert.ok(offline.meanCoverage >= hold.meanCoverage, `offline ${offline.meanCoverage} < hold ${hold.meanCoverage}`);
});
