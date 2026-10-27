// Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  boxesAt, fingerprint, formatDoc, linkTracks, newDoc, parseDoc, parseSettings, plan, PlanError, resolve, stableStringify, withLayers,
  type Box, type Layer, type SourceInfo, type SpeechItem,
} from '../src/doc/index.ts';

const source: SourceInfo = {
  name: 'clip.mp4', fingerprint: 'v0:1:x', size: 1, duration: 10, width: 1920, height: 1080, fps: 25, frames: 250, video: { codec: 'avc1' },
};

test('resolve: dependencies are added, in dependency order', () => {
  assert.deepEqual(resolve(['faces', 'transcript']), ['speech', 'transcript', 'scenes', 'faces']);
  assert.throws(() => resolve(['subtitles']), PlanError);
});

test('plan: matching layers are cached; a changed dependency invalidates dependents', async () => {
  const first = await plan(undefined, ['faces']);
  assert.deepEqual(first.passes, { frames: ['scenes', 'faces'] });
  const layer = (configHash: string) => ({ producer: { configHash } }) as unknown as Layer<never>;
  const hashes = Object.fromEntries(first.layers.map((l) => [l.name, l.configHash]));
  const doc = withLayers(newDoc(source), { scenes: layer(hashes.scenes), faces: layer(hashes.faces) });

  const again = await plan(doc, ['faces']);
  assert.deepEqual(again.passes, {});
  assert.deepEqual(again.layers.map((l) => l.status), ['cached', 'cached']);
  // A different scene threshold recomputes scenes, and faces too (its cuts came from scenes).
  const cuts = await plan(doc, ['faces'], { scenes: { threshold: 0.3 } });
  assert.deepEqual(cuts.passes, { frames: ['scenes', 'faces'] });
  // A faces-only setting leaves scenes cached.
  const every = await plan(doc, ['faces'], parseSettings(['faces.every=4']));
  assert.deepEqual(every.layers.map((l) => l.status), ['cached', 'compute']);
  assert.deepEqual((await plan(doc, ['faces'], {}, true)).passes, { frames: ['scenes', 'faces'] });
});

test('settings: typed by their defaults, unknown keys rejected with the valid ones', () => {
  assert.deepEqual(parseSettings(['faces.every=4', 'faces.refine=false', 'transcript.model=x/y']), {
    faces: { every: 4, refine: false },
    transcript: { model: 'x/y' },
  });
  assert.throws(() => parseSettings(['faces.evry=4']), /faces\.every/);
  assert.throws(() => parseSettings(['faces.every=lots']), /takes a number/);
  assert.throws(() => parseSettings(['every=4']), /layer\.key=value/);
});

test('stableStringify: key order does not matter', () => {
  assert.equal(stableStringify({ b: 1, a: { d: [1, 2], c: 'x' } }), stableStringify({ a: { c: 'x', d: [1, 2] }, b: 1 }));
});

test('fingerprint: size plus both ends of the file', async () => {
  const bytes = new Uint8Array(3 << 20).map((_, i) => i % 251);
  const read = async (a: number, b: number) => bytes.slice(a, b);
  const fp = await fingerprint(bytes.length, read);
  assert.match(fp, /^v0:3145728:[0-9a-f]{32}$/);
  // A change in the middle is not seen (by design); a change at the end is.
  bytes[bytes.length >> 1] ^= 1;
  assert.equal(await fingerprint(bytes.length, read), fp);
  bytes[bytes.length - 1] ^= 1;
  assert.notEqual(await fingerprint(bytes.length, read), fp);
});

test('tracks: linked across frames, split at cuts, compressed, reproduced by interpolation', () => {
  // Face A moves right 4 px/frame for 20 frames; face B appears at frame 5 and stays still.
  const frames: Box[][] = Array.from({ length: 20 }, (_, f) => {
    const boxes: Box[] = [[100 + 4 * f, 100, 50, 60]];
    if (f >= 5) boxes.push([800, 400, 40, 40]);
    return boxes;
  });
  const tracks = linkTracks(frames);
  assert.equal(tracks.length, 2);
  assert.deepEqual(tracks.map((t) => [t.f0, t.f1, t.samples.length]), [[0, 19, 2], [5, 19, 2]]);
  for (let f = 0; f < 20; f++) assert.deepEqual(boxesAt(tracks, f).map((x) => x.box), frames[f]);

  // A cut at frame 10 ends both tracks; the same faces start new ones.
  const cut = linkTracks(frames, [10]);
  assert.deepEqual(cut.map((t) => [t.id, t.f0, t.f1]), [[1, 0, 9], [2, 5, 9], [3, 10, 19], [4, 10, 19]]);

  // A turn mid-track needs a sample at the turn.
  const turn = linkTracks(Array.from({ length: 11 }, (_, f) => [[100 + 10 * Math.min(f, 5), 100 + 10 * Math.max(0, f - 5), 50, 50] as Box]));
  assert.deepEqual(turn[0].samples.map((s) => s[0]), [0, 5, 10]);
});

test('doc: one item per line round-trips; bad docs say how to fix them', () => {
  const speech: Layer<SpeechItem> = {
    producer: { module: 'speech', version: '0', config: {}, configHash: 'h' }, coverage: [[0, 10]], createdAt: '', ms: 1, device: 'wasm',
    items: [{ t0: 1, t1: 2 }, { t0: 3, t1: 4 }],
  };
  const doc = withLayers(newDoc(source), { speech, scenes: { ...speech, items: [] } as never });
  const text = formatDoc(doc);
  assert.deepEqual(parseDoc(text), doc);
  assert.equal(text.split('\n').filter((l) => l.includes('"t0"')).length, 2);
  assert.throws(() => parseDoc('{"schema":"videodoc/9"}'), /re-run analyze with --force/);
  assert.throws(() => parseDoc('not json'), /not valid JSON/);
});
