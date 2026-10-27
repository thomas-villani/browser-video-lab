// The planner: which layers a request needs, which are already in the doc with the same
// config, and which decode pass computes the rest. Pure (no browser APIs), so the CLI can plan
// in Node and skip launching a browser when nothing is missing.
import { hashOf } from './hash.ts';
import { selectKeyframes, type KeyframeConfig } from './keyframes.ts';
import { LAYER_NAMES, VERSION, type Layer, type LayerName, type Layers, type VideoDoc } from './schema.ts';

export type ConfigValue = number | string | boolean;
/** 'pure' layers are derived from other layers (deriveLayers): no browser, no decode. */
export type Pass = 'audio' | 'frames' | 'pure';

export interface ProducerSpec {
  name: LayerName;
  needs: LayerName[];
  /** Audio is decoded on its own (cheap); every frame consumer shares one video decode; pure
   *  layers are computed from the doc. */
  pass: Pass;
  description: string;
  defaults: Record<string, ConfigValue>;
  help: Record<string, string>;
}

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
    defaults: { model: 'onnx-community/whisper-base', encoder: 'fp32', decoder: 'q4', language: 'en', device: 'webgpu' },
    help: {
      model: 'Whisper repo id under public/models (whisper-base ~5x real time, whisper-small ~2x, fewer errors)',
      encoder: 'encoder ONNX dtype',
      decoder: 'decoder ONNX dtype',
      language: 'spoken language code',
      device: 'webgpu | wasm',
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
};

export type Overrides = Partial<Record<LayerName, Record<string, ConfigValue>>>;

export interface PlannedLayer {
  name: LayerName;
  status: 'cached' | 'compute';
  config: Record<string, ConfigValue>;
  configHash: string;
}

export interface Plan {
  /** Requested layers and their dependencies, in dependency order. */
  layers: PlannedLayer[];
  /** Decode passes needed and the layers each produces. */
  passes: Partial<Record<Pass, LayerName[]>>;
}

export class PlanError extends Error {}

/** Requested layers plus everything they need, in dependency order. */
export function resolve(requested: string[]): LayerName[] {
  const want = new Set<LayerName>();
  const add = (name: string) => {
    if (!(name in PRODUCERS)) throw new PlanError(`unknown layer '${name}' (have: ${LAYER_NAMES.join(', ')})`);
    const n = name as LayerName;
    if (want.has(n)) return;
    want.add(n);
    PRODUCERS[n].needs.forEach(add);
  };
  requested.forEach(add);
  return LAYER_NAMES.filter((n) => want.has(n));
}

export async function plan(doc: VideoDoc | undefined, requested: string[], overrides: Overrides = {}, force = false): Promise<Plan> {
  const hashes = new Map<LayerName, string>();
  const layers: PlannedLayer[] = [];
  for (const name of resolve(requested)) {
    const p = PRODUCERS[name];
    for (const k of Object.keys(overrides[name] ?? {})) {
      if (!(k in p.defaults)) throw new PlanError(`unknown setting '${name}.${k}' (have: ${Object.keys(p.defaults).map((x) => `${name}.${x}`).join(', ')})`);
    }
    const config = { ...p.defaults, ...overrides[name] };
    // A layer's identity includes the identities of the layers it was computed from.
    const configHash = await hashOf({ module: name, config, needs: p.needs.map((n) => hashes.get(n)) });
    hashes.set(name, configHash);
    const cached = !force && doc?.layers[name]?.producer.configHash === configHash;
    layers.push({ name, status: cached ? 'cached' : 'compute', config, configHash });
  }
  const passes: Plan['passes'] = {};
  for (const l of layers) if (l.status === 'compute') (passes[PRODUCERS[l.name].pass] ??= []).push(l.name);
  return { layers, passes };
}

/** Parse `layer.key=value` settings, typed by each key's default. */
export function parseSettings(settings: string[]): Overrides {
  const out: Overrides = {};
  for (const s of settings) {
    const m = /^(\w+)\.(\w+)=(.*)$/.exec(s);
    if (!m) throw new PlanError(`bad setting '${s}' (expected layer.key=value, e.g. faces.every=4)`);
    const [, layer, key, raw] = m;
    const p = PRODUCERS[layer as LayerName];
    if (!p) throw new PlanError(`unknown layer '${layer}' in '${s}' (have: ${LAYER_NAMES.join(', ')})`);
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
    (out[layer as LayerName] ??= {})[key] = v;
  }
  return out;
}

/** Compute the planned 'pure' layers of `doc` (their inputs must already be in it). */
export function deriveLayers(doc: VideoDoc, planned: PlannedLayer[]): Partial<Layers> {
  const out: Partial<Layers> = {};
  for (const l of planned) {
    if (l.status !== 'compute' || PRODUCERS[l.name].pass !== 'pure') continue;
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
