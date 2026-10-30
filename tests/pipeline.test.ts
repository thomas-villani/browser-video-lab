// Models as data, decoder presets, pipeline presets, and track layers a pipeline adds
// (src/doc/models.ts, pipeline.ts, plan.ts, doc.ts, redact.ts). Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  checkPipeline,
  checkPlan,
  coverTracks,
  DocError,
  decode,
  formatDoc,
  type InputGeometry,
  type Layer,
  MODELS,
  newDoc,
  newPlan,
  PIPELINE_SCHEMA,
  type Pipeline,
  PlanError,
  parseDoc,
  pipelineContext,
  plan,
  REDACT_DEFAULTS,
  redactionLog,
  resolveModel,
  summarize,
  type Track,
  trackLayers,
  withLayers,
} from '../src/doc/index.ts';

const r2 = (v: number) => Math.round(v * 100) / 100;
const round = (d: { x: number; y: number; w: number; h: number; score: number; label: string }) => ({ ...d, x: r2(d.x), y: r2(d.y), w: r2(d.w), h: r2(d.h), score: r2(d.score) });

// A 1280x640 frame letterboxed into 640x640: content 640x320, input pixels are 2 source pixels.
const geo: InputGeometry = { frameW: 1280, frameH: 640, crop: { x: 0, y: 0, w: 1280, h: 640 }, inW: 640, inH: 640, contentW: 640, contentH: 320 };

test('decode yolo (Ultralytics, no NMS): channels-first rows, best class, NMS per label, back to source pixels', () => {
  const m = resolveModel({ url: 'x.onnx', decoder: 'yolo', labels: ['car', 'sign'] });
  // [4 + 2 classes][3 anchors]: two overlapping cars, one sign.
  const rows = [
    [100, 102, 300], // cx
    [100, 100, 50], // cy
    [20, 20, 10], // w
    [20, 20, 10], // h
    [0.9, 0.8, 0], // car
    [0.1, 0, 0.7], // sign
  ];
  const dets = decode(m, { output0: new Float32Array(rows.flat()) }, geo).map(round);
  assert.deepEqual(dets, [
    { x: 180, y: 180, w: 40, h: 40, score: 0.9, label: 'car' },
    { x: 590, y: 90, w: 20, h: 20, score: 0.7, label: 'sign' },
  ]);
  assert.throws(() => resolveModel({ url: 'x.onnx', decoder: 'yolo' }), /needs labels/);
  assert.throws(() => decode(m, { output0: new Float32Array(7) }, geo), /multiple of 4 \+ 2/);
});

test('decode yolo-end2end: 7-column rows (open-image-models) and 6-column rows (Ultralytics nms=True) by output shape', () => {
  const plate = resolveModel(MODELS['yolo-plate']);
  const seven = new Float32Array([0, 10, 10, 30, 20, 0, 0.8, 0, 0, 0, 5, 5, 0, 0.1]);
  assert.deepEqual(decode(plate, { output0: seven }, geo).map(round), [{ x: 20, y: 20, w: 40, h: 20, score: 0.8, label: 'plate' }]);
  const m = resolveModel({ url: 'x.onnx', decoder: 'yolo-end2end', labels: ['a', 'b'] });
  const six = new Float32Array([10, 10, 30, 20, 0.9, 1, 0, 0, 5, 5, 0.1, 0]);
  assert.deepEqual(decode(m, { output0: six }, geo, undefined, { output0: [1, 2, 6] }).map(round), [{ x: 20, y: 20, w: 40, h: 20, score: 0.9, label: 'b' }]);
  assert.throws(() => decode(m, { output0: six }, geo, undefined, { output0: [1, 3, 4] }), /rows have 4 values/);
});

test('decode detr: sigmoid logits, relative boxes on the crop, COCO ids as labels', () => {
  const m = resolveModel(MODELS.rfdetr);
  const logits = new Float32Array(2 * 91).fill(-10);
  logits[3] = 2; // query 0: car
  logits[91 + 1] = -1; // query 1: person, below 0.5
  const boxes = new Float32Array([0.5, 0.5, 0.25, 0.5, 0.1, 0.1, 0.1, 0.1]);
  const stretched = { ...geo, inW: 384, inH: 384, contentW: 384, contentH: 384 };
  assert.deepEqual(decode(m, { logits, pred_boxes: boxes }, stretched).map(round), [{ x: 480, y: 160, w: 320, h: 320, score: 0.88, label: 'car' }]);
});

const vehicles: Pipeline = {
  schema: PIPELINE_SCHEMA,
  layers: ['vehicles', 'faces'],
  settings: { 'vehicles.minScore': 0.5, 'faces.every': 4 },
  tracks: { vehicles: { model: 'rfdetr', labels: 'car,truck,bus' } },
};

test('pipeline: track layers plan after the built-in ones, with their model; settings typed and checked', async () => {
  const pc = pipelineContext(checkPipeline(vehicles));
  const p = await plan(undefined, vehicles.layers, pc.settings, false, pc);
  assert.deepEqual(
    p.layers.map((l) => l.name),
    ['scenes', 'faces', 'vehicles'],
  );
  assert.deepEqual(p.passes, { frames: ['scenes', 'faces', 'vehicles'] });
  const v = p.layers.find((l) => l.name === 'vehicles')!;
  assert.equal(v.kind, 'track');
  assert.equal(v.model!.id, 'rfdetr');
  assert.equal(v.model!.decoder, 'detr');
  assert.deepEqual([v.config.labels, v.config.minScore, v.config.maxStride], ['car,truck,bus', 0.5, 8]);
  assert.equal(p.layers.find((l) => l.name === 'faces')!.config.every, 4);
  assert.equal(p.layers.find((l) => l.name === 'faces')!.model!.id, 'yunet');

  const bad = (over: Partial<Pipeline>, re: RegExp) =>
    assert.throws(
      () => checkPipeline({ ...vehicles, ...over }),
      (e) => e instanceof PlanError && re.test(e.message),
    );
  bad({ tracks: { faces: { model: 'rfdetr' } } }, /built-in layer/);
  bad({ tracks: { 'Bad-Name': { model: 'rfdetr' } } }, /lowercase/);
  bad({ tracks: { vehicles: { model: 'nope' } } }, /unknown model 'nope'/);
  bad({ tracks: { vehicles: { model: 'rfdetr', every: 2 } } }, /unknown setting 'every'/);
  bad({ tracks: { vehicles: { model: 'rfdetr', minScore: 'high' } } }, /'minScore' takes a number/);
  bad({ settings: { 'faces.refine': 'maybe' } }, /takes true or false/);
  bad({ settings: { 'vehicles.nope': 1 } }, /unknown setting 'vehicles.nope'/);
  bad({ models: { mine: { url: 'm.onnx', decoder: 'ssd' as never } } }, /decoder 'ssd'/);
  bad({ schema: 'videodoc-pipeline/9' as never }, /schema/);
  // A pipeline's own model, named by a track layer.
  const own = checkPipeline({
    ...vehicles,
    settings: {},
    models: { logos: { url: './logos.onnx', decoder: 'yolo', labels: ['logo'] } },
    tracks: { logos: { model: 'logos' } },
    layers: ['logos'],
  });
  const q = await plan(undefined, own.layers, {}, false, pipelineContext(own));
  assert.equal(q.layers.at(-1)!.model!.url, './logos.onnx');
});

test("planner: a track layer's hash follows its model file (fingerprint) and description, not where the file is", async () => {
  const hash = async (fp: string | undefined, url = '/models/rfdetr_nano.onnx', labels = 'car,truck,bus') => {
    const pc = pipelineContext({ ...vehicles, tracks: { vehicles: { model: 'rfdetr', labels } } });
    const p = await plan(undefined, ['vehicles'], {}, false, { ...pc, models: { rfdetr: { ...MODELS.rfdetr, url } }, fingerprint: async () => fp });
    return Object.fromEntries(p.layers.map((l) => [l.name, l.configHash]));
  };
  const a = await hash('v0:1:a');
  assert.equal((await hash('v0:1:a', '/elsewhere/rfdetr_nano.onnx')).vehicles, a.vehicles, 'moved file: same hash');
  assert.notEqual((await hash('v0:1:b')).vehicles, a.vehicles, 'replaced file: new hash');
  assert.equal((await hash('v0:1:b')).scenes, a.scenes);
  assert.notEqual((await hash('v0:1:a', undefined, 'car')).vehicles, a.vehicles, 'other labels: new hash');
  // Without a fingerprint the URL stands in for the file.
  assert.notEqual((await hash(undefined)).vehicles, (await hash(undefined, '/elsewhere/rfdetr_nano.onnx')).vehicles);
});

const layer = <T>(items: T[], extra = {}) =>
  ({ producer: { module: 'm', version: '0', configHash: 'h', config: {}, ...extra }, items, coverage: [], createdAt: 'now', ms: 1, device: 'webgpu' }) as unknown as Layer<T>;
const cars: Track[] = [
  {
    id: 1,
    f0: 0,
    f1: 9,
    samples: [
      [0, 10, 10, 50, 30],
      [9, 20, 10, 50, 30],
    ],
  },
  { id: 2, f0: 5, f1: 14, samples: [[5, 200, 100, 40, 30]] },
];
const face: Track[] = [{ id: 1, f0: 0, f1: 4, samples: [[0, 100, 20, 20, 20]] }];
const base = newDoc({ name: 'v.mp4', fingerprint: 'v0:1:x', size: 1, duration: 1, width: 310, height: 200, fps: 25, frames: 25, video: { codec: 'avc1' } });
const doc = withLayers(base, {
  vehicles: layer(cars, { module: 'vehicles', kind: 'track', configHash: 'veh1' }),
  faces: layer(face, { configHash: 'fac1' }),
  scenes: layer([{ t0: 0, t1: 1, f0: 0, f1: 24, score: 0 }]),
});

test("doc: a pipeline's track layers are kept after the built-in ones, saved, read back and summarized", () => {
  assert.deepEqual(Object.keys(doc.layers), ['scenes', 'faces', 'vehicles']);
  assert.deepEqual(trackLayers(doc), ['faces', 'vehicles']);
  const back = parseDoc(formatDoc(doc));
  assert.deepEqual(back.layers.vehicles!.items, cars);
  assert.equal(summarize(back).vehicles.trackFrames, 20);
  // A layer the code knows nothing about is still refused.
  const odd = formatDoc(withLayers(base, { captions2: layer([]) }));
  assert.throws(
    () => parseDoc(odd),
    (e) => e instanceof DocError && /unknown layer 'captions2'/.test(e.message),
  );
});

test("redaction covers a pipeline's track layer, keeps its kept ids, and logs it by layer", () => {
  const o = { ...REDACT_DEFAULTS, cover: ['faces', 'vehicles'], keepTracks: { vehicles: [2] } };
  assert.deepEqual(
    coverTracks(doc, o).map((c) => `${c.kind}:${c.track.id}`),
    ['faces:1', 'vehicles:1'],
  );
  const log = redactionLog(doc, o);
  assert.deepEqual(Object.keys(log.detectors), ['faces', 'vehicles']);
  assert.deepEqual(
    log.tracks.map((t) => [t.layer, t.id, t.kept]),
    [
      ['faces', 1, false],
      ['vehicles', 1, false],
      ['vehicles', 2, true],
    ],
  );
  // Not covered unless listed.
  assert.deepEqual(
    coverTracks(doc, REDACT_DEFAULTS).map((c) => c.kind),
    ['faces'],
  );

  // A review's kept ids name the tracks of the layer it saw: found again, they are dropped.
  const saved = newPlan(doc, o);
  assert.deepEqual(saved.tracks, { vehicles: 'veh1' });
  assert.deepEqual(checkPlan(JSON.parse(JSON.stringify(saved)), doc).plan.options.keepTracks, { vehicles: [2] });
  const again = withLayers(doc, { vehicles: layer(cars, { module: 'vehicles', kind: 'track', configHash: 'veh2' }) });
  const r = checkPlan(JSON.parse(JSON.stringify(saved)), again);
  assert.deepEqual(r.plan.options.keepTracks, {});
  assert.match(r.warnings.join(' '), /vehicles were found again/);
});
