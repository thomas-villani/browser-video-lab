// Pipeline presets: which layers to compute, their settings, models to add and track layers to
// define on them, as one JSON file (`videodoc analyze --pipeline preset.json`, the MCP analyze
// tool's `pipeline`, the app's pipeline builder). Pure.
//
//   {
//     "schema": "videodoc-pipeline/0",
//     "name": "dashcam",
//     "layers": ["transcript", "keyframes", "faces", "plates", "vehicles"],
//     "settings": { "faces.every": 4 },
//     "models": { "coco": { "url": "/models/rfdetr_nano.onnx", "decoder": "detr", "labels": [...] } },
//     "tracks": { "vehicles": { "model": "rfdetr", "labels": "car,truck,bus,motorcycle" } }
//   }
import { type ModelError, MODELS, resolveModel, type ModelConfig } from './models.ts';
import { parseSettings, PlanError, PRODUCERS, type ConfigValue, type Overrides, type PlanContext, type ProducerSpec, type Producers } from './plan.ts';
import { LAYER_NAME_RE, LAYER_NAMES } from './schema.ts';

export const PIPELINE_SCHEMA = 'videodoc-pipeline/0';

/** A track layer defined by a pipeline: a detector over the shared decode, linked into tracks. */
export interface TrackDef {
  model: string;
  description?: string;
  /** Keep only these labels, comma-separated (a COCO model used for "car,truck,bus"); '' = all. */
  labels?: string;
  /** Any of TRACK_DEFAULTS' settings. */
  [setting: string]: ConfigValue | undefined;
}

export interface Pipeline {
  schema: typeof PIPELINE_SCHEMA;
  name?: string;
  description?: string;
  /** Layers to compute (their dependencies are added). */
  layers: string[];
  /** `layer.key` -> value, typed like the setting's default (`"faces.every": 4`). */
  settings?: Record<string, ConfigValue>;
  /** Models by name, for track layers (and for faces.model / plates.model). */
  models?: Record<string, ModelConfig>;
  tracks?: Record<string, TrackDef>;
}

/** Settings of a pipeline's track layer, as for plates (small objects, fast motion). */
export const TRACK_DEFAULTS: Record<string, ConfigValue> = { model: '', labels: '', maxStride: 8, minIou: 0.3, minScore: 0.4, lowScore: 0.2, maxShift: 1, maxGap: 6 };

export const TRACK_HELP: Record<string, string> = {
  model: 'detector model (built-in or from the pipeline\'s models)',
  labels: 'keep only these labels, comma-separated; empty = all the model finds',
  maxStride: 'coarsest spacing between detected frames (must divide the other track layers\')',
  minIou: 'overlap for two detections to count as the same object',
  minScore: 'detection score threshold',
  lowScore: 'an object seen at minScore on one side of an interval still matches this weak a detection on the other',
  maxShift: 'link a detection to a track whose motion puts it at most this many object sizes away',
  maxGap: 'frames a track may go undetected and continue (interpolated); 0 = none',
};

/** The producer for a pipeline's track layer `name`. */
export function trackProducer(name: string, def: TrackDef): ProducerSpec {
  const { description, ...settings } = def;
  for (const k of Object.keys(settings)) {
    if (!(k in TRACK_DEFAULTS)) throw new PlanError(`track layer '${name}': unknown setting '${k}' (have: ${Object.keys(TRACK_DEFAULTS).join(', ')})`);
    if (typeof settings[k] !== typeof TRACK_DEFAULTS[k]) throw new PlanError(`track layer '${name}': '${k}' takes a ${typeof TRACK_DEFAULTS[k]}`);
  }
  if (!def.model) throw new PlanError(`track layer '${name}' needs a model`);
  return {
    name,
    needs: ['scenes'],
    pass: 'frames',
    kind: 'track',
    description: description ?? `${name} tracks (${def.model}${def.labels ? `: ${def.labels}` : ''})`,
    defaults: { ...TRACK_DEFAULTS, ...(settings as Record<string, ConfigValue>) },
    help: TRACK_HELP,
  };
}

/** Check a pipeline read from JSON; errors say what is wrong and how to fix it. */
export function checkPipeline(v: unknown, where = 'pipeline'): Pipeline {
  const p = v as Pipeline;
  if (!p || typeof p !== 'object') throw new PlanError(`${where}: not a pipeline (expected an object)`);
  if (p.schema !== PIPELINE_SCHEMA) throw new PlanError(`${where}: schema ${JSON.stringify(p.schema)} is not '${PIPELINE_SCHEMA}'`);
  if (!Array.isArray(p.layers) || !p.layers.every((l) => typeof l === 'string')) throw new PlanError(`${where}: layers takes a list of layer names`);
  for (const [name, m] of Object.entries(p.models ?? {})) {
    try {
      resolveModel(m, `model '${name}'`);
    } catch (e) {
      throw new PlanError(`${where}: ${(e as ModelError).message}`);
    }
  }
  const models = { ...MODELS, ...p.models };
  for (const [name, t] of Object.entries(p.tracks ?? {})) {
    if (!LAYER_NAME_RE.test(name)) throw new PlanError(`${where}: track layer name '${name}' must be lowercase letters, digits and _ (starting with a letter)`);
    if ((LAYER_NAMES as string[]).includes(name)) throw new PlanError(`${where}: '${name}' is a built-in layer; give the track layer another name`);
    if (!t || typeof t.model !== 'string') throw new PlanError(`${where}: track layer '${name}' needs a model`);
    if (!models[t.model]) throw new PlanError(`${where}: track layer '${name}': unknown model '${t.model}' (have: ${Object.keys(models).join(', ')})`);
  }
  if (p.settings && (typeof p.settings !== 'object' || Array.isArray(p.settings))) throw new PlanError(`${where}: settings takes an object of "layer.key": value`);
  // Checks every setting against its layer and type.
  pipelineContext(p);
  return p;
}

export const parsePipeline = (text: string, where = 'pipeline') => {
  let v: unknown;
  try {
    v = JSON.parse(text);
  } catch (e) {
    throw new PlanError(`${where}: not valid JSON (${(e as Error).message})`);
  }
  return checkPipeline(v, where);
};

/** The planner's producers and models for `p`, and its settings as overrides. */
export function pipelineContext(p?: Pipeline): Required<Pick<PlanContext, 'producers' | 'models'>> & { settings: Overrides } {
  const producers: Producers = { ...PRODUCERS };
  for (const [name, t] of Object.entries(p?.tracks ?? {})) producers[name] = trackProducer(name, t);
  const settings = parseSettings(Object.entries(p?.settings ?? {}).map(([k, v]) => `${k}=${v}`), producers);
  return { producers, models: { ...p?.models }, settings };
}

/** Overrides `b` on top of `a`, per layer. */
export function mergeOverrides(a: Overrides, b: Overrides): Overrides {
  const out: Overrides = { ...a };
  for (const [l, s] of Object.entries(b)) out[l] = { ...out[l], ...s };
  return out;
}
