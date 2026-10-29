// The planner: which layers a request needs, which are already in the doc with the same
// config, and which decode pass computes the rest. Pure (no browser APIs), so the CLI can plan
// in Node and skip launching a browser when nothing is missing.
import { hashOf } from './hash.ts';
import { selectKeyframes, type KeyframeConfig } from './keyframes.ts';
import { MODELS, type ModelConfig } from './models.ts';
import { LAYER_NAMES, VERSION, type Layer, type LayerName, type Layers, type VideoDoc } from './schema.ts';

export type ConfigValue = number | string | boolean;
/** 'pure' layers are derived from other layers (deriveLayers): no browser, no decode. */
export type Pass = 'audio' | 'frames' | 'pure';

export interface ProducerSpec {
  name: string;
  needs: string[];
  /** Audio is decoded on its own (cheap); every frame consumer shares one video decode; pure
   *  layers are computed from the doc. */
  pass: Pass;
  description: string;
  defaults: Record<string, ConfigValue>;
  help: Record<string, string>;
  /** Track layers: a detector (the `model` setting) over the shared decode, linked into tracks
   *  that redaction and the review can cover. */
  kind?: 'track';
}

/** Producers by layer name: the built-in ones plus any a pipeline defines (pipeline.ts). */
export type Producers = Record<string, ProducerSpec>;

/** Defaults are the best measured settings (NOTES.md). */
export const PRODUCERS: Record<LayerName, ProducerSpec> = {
  speech: {
    name: 'speech',
    needs: [],
    pass: 'audio',
    description: 'Speech spans from voice activity detection (Silero VAD v5, wasm).',
    defaults: { threshold: 0.5, minSilence: 0.3, minSpeech: 0.25, pad: 0.2 },
    help: {
      threshold: 'speech probability that starts a span',
      minSilence: 'seconds below threshold - 0.15 that end a span',
      minSpeech: 'drop spans shorter than this, seconds',
      pad: 'grow each span by this much on both sides, seconds',
    },
  },
  transcript: {
    name: 'transcript',
    needs: ['speech'],
    pass: 'audio',
    description: 'Timed transcript (Whisper via transformers.js), speech spans only.',
    defaults: { model: 'onnx-community/whisper-base_timestamped', encoder: 'fp32', decoder: 'q4', language: 'en', device: 'webgpu', words: true },
    help: {
      model: 'Whisper repo id under public/models (whisper-base ~4x real time, whisper-small ~1.5x, fewer errors; words need a _timestamped export)',
      words: 'word timestamps (audio redaction needs them; about a third slower)',
      encoder: 'encoder ONNX dtype',
      decoder: 'decoder ONNX dtype',
      language: 'spoken language code',
      device: 'webgpu | wasm',
    },
  },
  speakers: {
    name: 'speakers',
    needs: ['speech'],
    pass: 'audio',
    description: 'Who speaks when (pyannote segmentation-3.0 + WeSpeaker ResNet34 voices, clustered as pyannote 3.1 does).',
    defaults: {
      segmentation: 'onnx-community/pyannote-segmentation-3.0',
      embedding: 'onnx-community/wespeaker-voxceleb-resnet34-LM',
      step: 2,
      threshold: 0.7046,
      minClusterSize: 12,
      maxSpeakers: 0,
      device: 'webgpu',
    },
    help: {
      segmentation: 'segmentation model repo under public/models',
      embedding: 'voice embedding model repo under public/models',
      step: 'seconds between the starts of the 10 s windows (1 is pyannote\'s; 5 is twice as fast, a little less accurate)',
      threshold: 'voices closer than this are one speaker (lower splits more)',
      minClusterSize: 'voices found in fewer windows join the most similar speaker',
      maxSpeakers: 'at most this many speakers (0 = as many as found)',
      device: 'webgpu | wasm (voice embeddings)',
    },
  },
  scenes: {
    name: 'scenes',
    needs: [],
    pass: 'frames',
    description: 'Shots between scene cuts (GPU thumbnails, ffmpeg-style scene score).',
    defaults: { threshold: 0.15 },
    help: { threshold: 'scene score at or above which a frame starts a new shot' },
  },
  keyframes: {
    name: 'keyframes',
    needs: ['scenes', 'speech'],
    pass: 'pure',
    description: 'Frames worth looking at: each shot start, silent gaps, fill-ins in long shots (no decode).',
    defaults: { offset: 0.5, silence: 4, minGap: 2, maxGap: 30 },
    help: {
      offset: 'seconds into a shot for its frame (skips fades at the cut)',
      silence: 'silent gaps at least this long get a frame at their middle, seconds',
      minGap: 'drop candidates closer than this to a kept frame, seconds (shot starts win)',
      maxGap: 'longest stretch of a shot without a frame, seconds',
    },
  },
  faces: {
    name: 'faces',
    needs: ['scenes'],
    pass: 'frames',
    description: 'Face tracks (YuNet, adaptive sampling, native-resolution tiles in crowds).',
    kind: 'track',
    defaults: { model: 'yunet', longSide: 640, maxStride: 16, minIou: 0.3, minScore: 0.5, refine: true, tile: '640x384', every: 2, trigger: 0.3, small: 120 },
    help: {
      model: 'detector model',
      longSide: 'detector input long side for the full-frame pass, px',
      maxStride: 'coarsest spacing between detected frames',
      minIou: 'overlap for two detections to count as the same face',
      minScore: 'detection score threshold',
      refine: 'also detect native-resolution tiles where faces were seen (small faces in crowds)',
      tile: 'refinement tile size, WxH px',
      every: 'refine every Nth frame (should divide maxStride)',
      trigger: 'reduced-pass score that marks a tile for refinement',
      small: 'only faces shorter than this (px) mark tiles; 0 = any',
    },
  },
  plates: {
    name: 'plates',
    needs: ['scenes'],
    pass: 'frames',
    description: 'License plate tracks (YOLOv9-t, open-image-models; adaptive sampling, shares the decode with faces).',
    kind: 'track',
    defaults: { model: 'yolo-plate', maxStride: 8, minIou: 0.2, minScore: 0.4, lowScore: 0.2, maxShift: 1, maxGap: 6 },
    help: {
      model: 'detector model',
      maxStride: 'coarsest spacing between detected frames (must divide faces.maxStride when both run)',
      minIou: 'overlap for two detections to count as the same plate (plates are small and move fast)',
      minScore: 'detection score threshold',
      lowScore: 'a plate seen at minScore on one side of an interval still matches this weak a detection on the other (scores flicker)',
      maxShift: 'link a plate to a track whose motion puts it at most this many plate sizes away (passing cars move fast)',
      maxGap: 'frames a plate track may go undetected and continue (interpolated); 0 = none',
    },
  },
};

export type Overrides = Record<string, Record<string, ConfigValue>>;

/** A track layer's detector, as the page needs it to run (and as its identity is hashed). */
export type PlannedModel = ModelConfig & { id: string; fingerprint?: string };

export interface PlannedLayer {
  name: string;
  status: 'cached' | 'compute';
  config: Record<string, ConfigValue>;
  configHash: string;
  kind?: 'track';
  /** Track layers: the model named by config.model. */
  model?: PlannedModel;
}

export interface Plan {
  /** Requested layers and their dependencies, in dependency order. */
  layers: PlannedLayer[];
  /** Decode passes needed and the layers each produces. */
  passes: Partial<Record<Pass, string[]>>;
}

export interface PlanContext {
  /** Default: PRODUCERS. A pipeline adds its track layers (pipeline.ts). */
  producers?: Producers;
  /** Models by name, on top of the built-in MODELS. */
  models?: Record<string, ModelConfig>;
  /** Identity of a model file (fingerprint in hash.ts), so replacing the file recomputes the
   *  layers it made. Unset (or undefined for a URL): the URL stands in for the file. */
  fingerprint?: (url: string) => Promise<string | undefined>;
}

export class PlanError extends Error {}

/** Requested layers plus everything they need, in dependency order (built-in layers first,
 *  then a pipeline's, by name). */
export function resolve(requested: string[], producers: Producers = PRODUCERS): string[] {
  const want = new Set<string>();
  const add = (name: string) => {
    if (!(name in producers)) throw new PlanError(`unknown layer '${name}' (have: ${Object.keys(producers).join(', ')})`);
    if (want.has(name)) return;
    want.add(name);
    producers[name].needs.forEach(add);
  };
  requested.forEach(add);
  const builtIn = LAYER_NAMES.filter((n) => want.has(n));
  return [...builtIn, ...[...want].filter((n) => !(LAYER_NAMES as string[]).includes(n)).sort()];
}

export async function plan(doc: VideoDoc | undefined, requested: string[], overrides: Overrides = {}, force = false, ctx: PlanContext = {}): Promise<Plan> {
  const producers: Producers = ctx.producers ?? PRODUCERS;
  const models = { ...MODELS, ...ctx.models };
  const prints = new Map<string, Promise<string | undefined>>();
  const hashes = new Map<string, string>();
  const layers: PlannedLayer[] = [];
  for (const name of resolve(requested, producers)) {
    const p = producers[name];
    for (const k of Object.keys(overrides[name] ?? {})) {
      if (!(k in p.defaults)) throw new PlanError(`unknown setting '${name}.${k}' (have: ${Object.keys(p.defaults).map((x) => `${name}.${x}`).join(', ')})`);
    }
    const config = { ...p.defaults, ...overrides[name] };
    let model: PlannedModel | undefined;
    if (p.kind === 'track') {
      const id = String(config.model);
      if (!models[id]) throw new PlanError(`${name}.model: unknown model '${id}' (have: ${Object.keys(models).join(', ')}; a pipeline can add models)`);
      const url = models[id].url;
      if (ctx.fingerprint && !prints.has(url)) prints.set(url, ctx.fingerprint(url));
      const fingerprint = await prints.get(url);
      model = { ...models[id], id, ...(fingerprint && { fingerprint }) };
    }
    // A layer's identity includes the identities of the layers it was computed from, and its
    // model's: the description (without the display name) and the file (or its URL).
    const identity = model && (({ name: _, id: __, url, fingerprint, ...rest }) => (fingerprint ? { ...rest, fingerprint } : { ...rest, url }))(model);
    const configHash = await hashOf({ module: name, config, needs: p.needs.map((n) => hashes.get(n)), ...(identity && { model: identity }) });
    hashes.set(name, configHash);
    const cached = !force && doc?.layers[name]?.producer.configHash === configHash;
    layers.push({ name, status: cached ? 'cached' : 'compute', config, configHash, ...(p.kind && { kind: p.kind }), ...(model && { model }) });
  }
  const passes: Plan['passes'] = {};
  for (const l of layers) if (l.status === 'compute') (passes[producers[l.name].pass] ??= []).push(l.name);
  return { layers, passes };
}

/** Parse `layer.key=value` settings, typed by each key's default. */
export function parseSettings(settings: string[], producers: Producers = PRODUCERS): Overrides {
  const out: Overrides = {};
  for (const s of settings) {
    const m = /^(\w+)\.(\w+)=(.*)$/.exec(s);
    if (!m) throw new PlanError(`bad setting '${s}' (expected layer.key=value, e.g. faces.every=4)`);
    const [, layer, key, raw] = m;
    const p = producers[layer];
    if (!p) throw new PlanError(`unknown layer '${layer}' in '${s}' (have: ${Object.keys(producers).join(', ')})`);
    const d = p.defaults[key];
    if (d === undefined) throw new PlanError(`unknown setting '${layer}.${key}' (have: ${Object.keys(p.defaults).map((x) => `${layer}.${x}`).join(', ')})`);
    let v: ConfigValue = raw;
    if (typeof d === 'number') {
      v = Number(raw);
      if (raw === '' || !Number.isFinite(v)) throw new PlanError(`'${layer}.${key}' takes a number, got '${raw}'`);
    } else if (typeof d === 'boolean') {
      if (!/^(true|false|1|0)$/.test(raw)) throw new PlanError(`'${layer}.${key}' takes true or false, got '${raw}'`);
      v = raw === 'true' || raw === '1';
    }
    (out[layer] ??= {})[key] = v;
  }
  return out;
}

/** Compute the planned 'pure' layers of `doc` (their inputs must already be in it). */
export function deriveLayers(doc: VideoDoc, planned: PlannedLayer[]): Partial<Layers> {
  const out: Partial<Layers> = {};
  for (const l of planned) {
    if (l.status !== 'compute' || PRODUCERS[l.name as LayerName]?.pass !== 'pure') continue;
    const t0 = performance.now();
    const layer = <T>(items: T[]): Layer<T> => ({
      producer: { module: l.name, version: VERSION, config: l.config, configHash: l.configHash },
      coverage: [[0, doc.source.duration]],
      createdAt: new Date().toISOString(),
      ms: Math.round(performance.now() - t0),
      device: 'cpu',
      items,
    });
    if (l.name === 'keyframes') {
      const { scenes, speech } = doc.layers;
      if (!scenes || !speech) throw new PlanError('keyframes needs the scenes and speech layers');
      const c = l.config as unknown as KeyframeConfig;
      out.keyframes = layer(selectKeyframes(scenes.items, speech.items, doc.source.duration, doc.source.fps, c));
    }
  }
  return out;
}
