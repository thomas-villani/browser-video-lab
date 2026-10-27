// Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  boxesAt, deriveLayers, fingerprint, fmtTime, formatDoc, linkTracks, maxFaces, newDoc, parseDoc, parseSettings, plan, PlanError, PRODUCERS, resolve,
  selectKeyframes, stableStringify, toMarkdown, withLayers,
  type Box, type KeyframeConfig, type Layer, type SceneItem, type SourceInfo, type SpeechItem,
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

const KF = PRODUCERS.keyframes.defaults as unknown as KeyframeConfig;
const scene = (t0: number, t1: number, fps = 25): SceneItem => ({ t0, t1, f0: Math.round(t0 * fps), f1: Math.round(t1 * fps) - 1, score: t0 ? 0.5 : 0 });

test('keyframes: shot starts, silent gaps, fill-ins for long shots', () => {
  // One 100 s shot, speech everywhere except 40-50 s.
  const one = selectKeyframes([scene(0, 100)], [{ t0: 0, t1: 40 }, { t0: 50, t1: 100 }], 100, 25, KF);
  assert.deepEqual(one.map((k) => [k.t, k.why]), [[0.5, 'scene'], [22.75, 'coverage'], [45, 'silence'], [72.5, 'coverage']]);
  for (let i = 1; i < one.length; i++) assert.ok(one[i].t - one[i - 1].t <= KF.maxGap);
  assert.equal(one[2].frame, 45 * 25);

  // Quick cuts: shots shorter than minGap are skipped near a kept frame (a longer shot's always
  // is); a short shot's frame sits in its middle; frames point at their shot.
  const quick = selectKeyframes([scene(0, 0.6), scene(0.6, 1.5), scene(1.5, 12), scene(12, 12.6)], [{ t0: 0, t1: 12.6 }], 12.6, 25, KF);
  assert.deepEqual(quick.map((k) => [k.t, k.scene]), [[2, 2], [12.3, 3]]);
  assert.deepEqual(selectKeyframes([], [], 10, 25, KF), []);
});

test('keyframes: planned as a pure layer and derived without a browser', async () => {
  const p = await plan(undefined, ['keyframes']);
  assert.deepEqual(p.passes, { audio: ['speech'], frames: ['scenes'], pure: ['keyframes'] });
  const layer = <T>(items: T[]) => ({ producer: { configHash: 'h' }, items }) as unknown as Layer<T>;
  const doc = withLayers(newDoc(source), { speech: layer([{ t0: 0, t1: 10 }]), scenes: layer([scene(0, 4), scene(4, 10)]) });
  const kf = deriveLayers(doc, p.layers).keyframes!;
  assert.deepEqual(kf.items.map((k) => k.t), [0.5, 4.5]);
  assert.equal(kf.device, 'cpu');
  assert.equal(kf.producer.configHash, p.layers.find((l) => l.name === 'keyframes')!.configHash);
});

test('markdown: shots, timed paragraphs, silent gaps, keyframe images, faces', () => {
  const layer = <T>(items: T[], extra = {}) => ({ producer: { configHash: 'h', config: {}, ...extra }, items }) as unknown as Layer<T>;
  const doc = withLayers(newDoc(source), {
    speech: layer([{ t0: 0, t1: 3 }]),
    transcript: layer([{ t0: 0, t1: 1.5, text: ' Hello' }, { t0: 1.5, t1: 3, text: 'world.' }], { model: 'whisper' }),
    scenes: layer([scene(0, 4), scene(4, 10)]),
    keyframes: layer([{ t: 0.5, frame: 12, scene: 0, why: 'scene' }, { t: 4.5, frame: 112, scene: 1, why: 'scene' }]),
    faces: layer([{ id: 1, f0: 0, f1: 50, samples: [] }, { id: 2, f0: 20, f1: 120, samples: [] }]),
  });
  // Only shot 2's keyframe has an image; shot 1's still ends the paragraph.
  const md = toMarkdown(doc, { image: (k) => (k.scene ? `clip keyframes/${k.frame}.jpg` : undefined) });
  assert.equal(md, [
    '# clip.mp4', '',
    '00:10 · 1920×1080 · 25 fps · 2 shots · 30% speech · 2 words · 2 face tracks',
    '<!-- videodoc/0 · speech · transcript (whisper) · scenes · keyframes · faces -->', '',
    '## Shot 1 · 00:00–00:04 · up to 2 faces', '',
    '**[00:00]** Hello world.', '',
    '*(no speech 00:03–00:10)*', '',
    '## Shot 2 · 00:04–00:10 · up to 1 face', '',
    '![00:04](clip%20keyframes/112.jpg)', '',
  ].join('\n'));
  assert.equal(fmtTime(3725, true), '1:02:05');
  assert.equal(maxFaces([{ id: 1, f0: 0, f1: 9, samples: [] }, { id: 2, f0: 10, f1: 19, samples: [] }], 0, 19), 1);
});
