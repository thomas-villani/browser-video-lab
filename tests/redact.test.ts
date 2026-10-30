// Redaction regions and log (src/doc/redact.ts). Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  checkPlan,
  cursorFor,
  formatLog,
  formatPlan,
  handTrack,
  type Layer,
  newDoc,
  newPlan,
  parseKeep,
  parsePlan,
  REDACT_DEFAULTS,
  RedactionCursor,
  redactionLog,
  setBox,
  setEnd,
  type Track,
  trackBox,
  withLayers,
} from '../src/doc/index.ts';

// Track 1: frames 10-19, still. Track 2: frames 20-29, same place (the same face, re-acquired).
// Track 3: frames 40-44 at the right edge. Shots: 0-34, 35-59 (cut at 35).
const tracks: Track[] = [
  {
    id: 1,
    f0: 10,
    f1: 19,
    samples: [
      [10, 100, 100, 40, 50],
      [19, 100, 100, 40, 50],
    ],
  },
  {
    id: 2,
    f0: 20,
    f1: 29,
    samples: [
      [20, 102, 100, 40, 50],
      [29, 102, 100, 40, 50],
    ],
  },
  {
    id: 3,
    f0: 40,
    f1: 44,
    samples: [
      [40, 300, 10, 20, 20],
      [44, 300, 10, 20, 20],
    ],
  },
];
const cursor = (o: Partial<ConstructorParameters<typeof RedactionCursor>[1]> = {}) => new RedactionCursor(tracks, { margin: 0, holdFrames: 0, width: 310, height: 200, ...o });

test('regions: margin, clamped to the frame', () => {
  const c = cursor({ margin: 0.25 });
  assert.deepEqual(c.at(10), [[90, 87.5, 60, 75]]);
  // Track 3 grows past the right edge: clipped at x = 310.
  assert.deepEqual(c.at(40), [[295, 5, 15, 30]]);
});

test('regions: hold before and after a track, not across a cut, not over a live face', () => {
  const at = (c: RedactionCursor, f: number) => c.at(f).length;
  const c = cursor({ holdFrames: 3, cuts: [35] });
  const counts = [6, 7, 10, 19, 20, 29, 32, 33, 34, 35, 37, 38].map((f) => [f, at(c, f)]);
  assert.deepEqual(counts, [
    [6, 0],
    [7, 1], // track 1 held 3 frames before its start
    [10, 1],
    [19, 1],
    [20, 1], // track 1 held after its end, but track 2 covers the same face: one region
    [29, 1],
    [32, 1],
    [33, 0],
    [34, 0],
    [35, 0], // track 3 is not held back into frames before the cut (it starts at 40, hold 3 reaches 37)
    [37, 1],
    [38, 1],
  ]);
  // Hold 8: before the cut only track 2 (held after its end), after it only track 3 (held before its start).
  const across = cursor({ holdFrames: 8, cuts: [35] });
  assert.deepEqual(
    across.at(34).map((r) => r[0]),
    [102],
  );
  assert.deepEqual(
    across.at(35).map((r) => r[0]),
    [300],
  );
});

const layer = <T>(items: T[], extra = {}) =>
  ({ producer: { module: 'm', version: '0', configHash: 'h', config: {}, ...extra }, items, coverage: [], createdAt: 'now', ms: 1, device: 'webgpu' }) as unknown as Layer<T>;
const doc = withLayers(newDoc({ name: 'v.mp4', fingerprint: 'v0:1:x', size: 1, duration: 2.4, width: 310, height: 200, fps: 25, frames: 60, video: { codec: 'avc1' } }), {
  faces: layer(tracks, { model: 'yunet' }),
  scenes: layer([
    { t0: 0, t1: 1.4, f0: 0, f1: 34, score: 0 },
    { t0: 1.4, t1: 2.4, f0: 35, f1: 59, score: 0.9 },
  ]),
});

test('regions: kept tracks stay visible; going back restarts the sweep', () => {
  const c = cursorFor(doc, { ...REDACT_DEFAULTS, margin: 0, hold: 0, keep: [1] });
  assert.equal(c.at(15).length, 0);
  assert.equal(c.at(25).length, 1);
  assert.deepEqual(c.regions(42), [{ id: 3, kind: 'faces', box: [300, 10, 10, 20], held: false }]);
  assert.deepEqual(
    c.regions(24).map((r) => r.id),
    [2],
  );
  assert.equal(c.at(15).length, 0);
  assert.deepEqual(parseKeep('3, 7'), [3, 7]);
  assert.deepEqual(parseKeep(undefined), []);
  assert.throws(() => parseKeep('3,x'), /track ids/);
});

// Plate 1 at frames 12-16 (overlapping face track 1's place: kinds never merge), plate 2 at 50-51.
const plates: Track[] = [
  {
    id: 1,
    f0: 12,
    f1: 16,
    samples: [
      [12, 100, 100, 40, 12],
      [16, 100, 100, 40, 12],
    ],
  },
  {
    id: 2,
    f0: 50,
    f1: 51,
    samples: [
      [50, 200, 150, 30, 10],
      [51, 200, 150, 30, 10],
    ],
  },
];
const withPlates = withLayers(doc, { plates: layer(plates, { model: 'yolo-plate', configHash: 'p' }) });

test('plates: covered with faces, kept by their own ids, left out when not covered', () => {
  const o = { ...REDACT_DEFAULTS, margin: 0, hold: 0 };
  assert.deepEqual(
    cursorFor(withPlates, o)
      .regions(14)
      .map((r) => `${r.kind}/${r.id}`),
    ['faces/1', 'plates/1'],
  );
  // keepPlates: [1] leaves plate 1 visible, not face 1.
  assert.deepEqual(
    cursorFor(withPlates, { ...o, keepPlates: [1] })
      .regions(14)
      .map((r) => `${r.kind}/${r.id}`),
    ['faces/1'],
  );
  assert.deepEqual(cursorFor(withPlates, { ...o, cover: ['faces'] }).regions(50), []);
  const log = redactionLog(withPlates, { ...o, keepPlates: [2] });
  assert.deepEqual(log.totals, { tracks: 5, redacted: 4, kept: 1, added: 0, plates: 2, framesWithRegions: 25, regions: 30, maxPerFrame: 2 });
  assert.deepEqual(log.tracks.at(-1), { id: 2, layer: 'plates', t0: 2, t1: 2.08, frames: 2, kept: true });
  assert.deepEqual(Object.keys(log.detectors), ['faces', 'plates']);
  assert.equal(log.detectors.plates?.model, 'yolo-plate');
  // A doc without plates covers faces only.
  assert.deepEqual(Object.keys(redactionLog(doc, o).detectors), ['faces']);
  // Kept plates from another plates layer are dropped from a plan, like faces.
  const plan = newPlan(withPlates, { ...o, keepPlates: [2] });
  assert.equal(plan.plates, 'p');
  const stale = checkPlan({ ...plan, plates: 'q' }, withPlates);
  assert.deepEqual(stale.plan.options.keepPlates, []);
  assert.match(stale.warnings[0], /plates were found again/);
});

test('log: totals from a sweep over every frame, one track per line, valid JSON', () => {
  const log = redactionLog(doc, { ...REDACT_DEFAULTS, hold: 0, keep: [3] }, { name: 'v.redacted.mp4', frames: 60, ms: 5 });
  assert.deepEqual(log.totals, { tracks: 3, redacted: 2, kept: 1, added: 0, plates: 0, framesWithRegions: 20, regions: 20, maxPerFrame: 1 });
  assert.deepEqual(log.tracks[2], { id: 3, layer: 'faces', t0: 1.6, t1: 1.8, frames: 5, kept: true });
  assert.equal(log.detectors.faces?.model, 'yunet');
  const text = formatLog(log);
  assert.deepEqual(JSON.parse(text), log);
  assert.equal(text.split('\n').filter((l) => l.startsWith('  {"id"')).length, 3);
  // The default hold (0.3 s = 8 frames at 25 fps), within each shot: frames 2-34 (tracks 1, 2), 35-52 (track 3).
  assert.equal(redactionLog(doc, REDACT_DEFAULTS).totals.framesWithRegions, 33 + 18);
});

test('hand tracks: a still box that follows the face as it is drawn again further on', () => {
  const t = handTrack(-1, 10, [0, 0, 10, 10], 30);
  assert.deepEqual(trackBox(t, 20), [0, 0, 10, 10]);
  // Drawn again at 20: the carried box at 30 moves with it; 10-20 interpolates.
  const u = setBox(t, 20, [20, 0, 10, 10]);
  assert.deepEqual(u.samples, [
    [10, 0, 0, 10, 10],
    [20, 20, 0, 10, 10],
    [30, 20, 0, 10, 10],
  ]);
  assert.deepEqual(trackBox(u, 15), [10, 0, 10, 10]);
  // A fix earlier on leaves the later drawings alone.
  assert.deepEqual(
    setBox(u, 12, [5, 5, 10, 10]).samples.map((x) => x[1]),
    [0, 5, 20, 20],
  );
  // Beyond the end: extends it.
  assert.deepEqual([setBox(u, 40, [30, 0, 10, 10]).f0, setBox(u, 40, [30, 0, 10, 10]).f1], [10, 40]);
  assert.deepEqual(setEnd(u, 15, 'end').samples, [
    [10, 0, 0, 10, 10],
    [15, 10, 0, 10, 10],
  ]);
  assert.deepEqual(setEnd(u, 25, 'start').samples, [
    [25, 20, 0, 10, 10],
    [30, 20, 0, 10, 10],
  ]);
  assert.deepEqual(setEnd(u, 5, 'start').f0, 5);
});

test('plan: hand tracks are covered and logged; keep ids from other faces settings are dropped', () => {
  const plan = newPlan(doc, { ...REDACT_DEFAULTS, hold: 0, keep: [1] });
  plan.added.push(handTrack(-1, 50, [10, 10, 30, 30], 54));
  const back = parsePlan(formatPlan(plan), doc);
  assert.deepEqual(back, { plan, warnings: [] });
  const log = redactionLog(doc, plan.options, undefined, plan.added);
  assert.deepEqual(log.tracks.at(-1), { id: -1, layer: 'faces', t0: 2, t1: 2.2, frames: 5, kept: false, added: true });
  assert.equal(log.totals.added, 1);
  assert.equal(log.totals.framesWithRegions, 10 + 5 + 5);
  const stale = checkPlan({ ...plan, faces: 'other' }, doc);
  assert.deepEqual(stale.plan.options.keep, []);
  assert.equal(stale.plan.added.length, 1);
  assert.match(stale.warnings[0], /found again with other settings/);
  assert.throws(() => checkPlan({ ...plan, source: { name: 'w.mp4', fingerprint: 'v0:2:y' } }, doc), /another file/);
});
