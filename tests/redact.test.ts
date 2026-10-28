// Redaction regions and log (src/doc/redact.ts). Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { formatLog, newDoc, parseKeep, RedactionCursor, redactionLog, REDACT_DEFAULTS, withLayers, type Layer, type Track } from '../src/doc/index.ts';

// Track 1: frames 10-19, still. Track 2: frames 20-29, same place (the same face, re-acquired).
// Track 3: frames 40-44 at the right edge. Shots: 0-34, 35-59 (cut at 35).
const tracks: Track[] = [
  { id: 1, f0: 10, f1: 19, samples: [[10, 100, 100, 40, 50], [19, 100, 100, 40, 50]] },
  { id: 2, f0: 20, f1: 29, samples: [[20, 102, 100, 40, 50], [29, 102, 100, 40, 50]] },
  { id: 3, f0: 40, f1: 44, samples: [[40, 300, 10, 20, 20], [44, 300, 10, 20, 20]] },
];
const cursor = (o: Partial<ConstructorParameters<typeof RedactionCursor>[1]> = {}) =>
  new RedactionCursor(tracks, { margin: 0, keep: [], holdFrames: 0, width: 310, height: 200, ...o });

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
    [6, 0], [7, 1], // track 1 held 3 frames before its start
    [10, 1], [19, 1],
    [20, 1], // track 1 held after its end, but track 2 covers the same face: one region
    [29, 1], [32, 1], [33, 0],
    [34, 0], [35, 0], // track 3 is not held back into frames before the cut (it starts at 40, hold 3 reaches 37)
    [37, 1], [38, 1],
  ]);
  // Hold 8: before the cut only track 2 (held after its end), after it only track 3 (held before its start).
  const across = cursor({ holdFrames: 8, cuts: [35] });
  assert.deepEqual(across.at(34).map((r) => r[0]), [102]);
  assert.deepEqual(across.at(35).map((r) => r[0]), [300]);
});

test('regions: kept tracks stay visible; frames only go forward', () => {
  const c = cursor({ keep: [1] });
  assert.equal(c.at(15).length, 0);
  assert.equal(c.at(25).length, 1);
  assert.throws(() => c.at(24), /must not go back/);
  assert.deepEqual(parseKeep('3, 7'), [3, 7]);
  assert.deepEqual(parseKeep(undefined), []);
  assert.throws(() => parseKeep('3,x'), /track ids/);
});

test('log: totals from a sweep over every frame, one track per line, valid JSON', () => {
  const layer = <T>(items: T[], extra = {}) => ({ producer: { module: 'm', version: '0', configHash: 'h', config: {}, ...extra }, items, coverage: [], createdAt: 'now', ms: 1, device: 'webgpu' }) as unknown as Layer<T>;
  const doc = withLayers(
    newDoc({ name: 'v.mp4', fingerprint: 'v0:1:x', size: 1, duration: 2.4, width: 310, height: 200, fps: 25, frames: 60, video: { codec: 'avc1' } }),
    {
      faces: layer(tracks, { model: 'yunet' }),
      scenes: layer([{ t0: 0, t1: 1.4, f0: 0, f1: 34, score: 0 }, { t0: 1.4, t1: 2.4, f0: 35, f1: 59, score: 0.9 }]),
    },
  );
  const log = redactionLog(doc, { ...REDACT_DEFAULTS, hold: 0, keep: [3] }, { name: 'v.redacted.mp4', frames: 60, ms: 5 });
  assert.deepEqual(log.totals, { tracks: 3, redacted: 2, kept: 1, framesWithRegions: 20, regions: 20, maxPerFrame: 1 });
  assert.deepEqual(log.tracks[2], { id: 3, t0: 1.6, t1: 1.8, frames: 5, kept: true });
  assert.equal(log.detector.model, 'yunet');
  const text = formatLog(log);
  assert.deepEqual(JSON.parse(text), log);
  assert.equal(text.split('\n').filter((l) => l.startsWith('  {"id"')).length, 3);
  // The default hold (0.3 s = 8 frames at 25 fps), within each shot: frames 2-34 (tracks 1, 2), 35-52 (track 3).
  assert.equal(redactionLog(doc, REDACT_DEFAULTS).totals.framesWithRegions, 33 + 18);
});
