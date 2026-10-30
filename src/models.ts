// Runnable model specs: input geometry/normalization plus an output decoder. Models are
// described as data (src/doc/models.ts: MODELS, decoder presets); a spec with a hand-written
// `decode` can be registered too (registerModel). Decoders return boxes in source-frame pixels,
// so callers never see crops or padding.
import { MODELS as CONFIGS, type Detection, decode, type InputGeometry, type ModelConfig, ModelError, type Preprocess, type Rect, resolveModel } from './doc/models.ts';

export { type Detection, type InputGeometry, type ModelConfig, nms, type Rect } from './doc/models.ts';

export interface ModelSpec extends Preprocess {
  id: string;
  url: string;
  /** Shown as the layer's model. */
  name?: string;
  /** Returns boxes in source-frame pixels. `minScore` overrides the model's default threshold;
   *  `dims` are the output shapes. */
  decode(outputs: Record<string, Float32Array>, geo: InputGeometry, minScore?: number, dims?: Record<string, readonly number[]>): Detection[];
}

/** A runnable spec from a model description. */
export function specFor(id: string, config: ModelConfig): ModelSpec {
  const m = resolveModel(config, `model '${id}'`);
  return { ...m, id, decode: (out, geo, minScore, dims) => decode(m, out, geo, minScore, dims) };
}

const registry = new Map<string, ModelSpec>();

/**
 * Make a model available to layers by name (`faces.model=<name>`, or a pipeline's track layer):
 * a description (an ONNX URL plus a decoder preset) or a spec with its own `decode`.
 */
export function registerModel(name: string, model: ModelConfig | ModelSpec): ModelSpec {
  const spec = 'decode' in model && typeof model.decode === 'function' ? { ...model, id: name } : specFor(name, model as ModelConfig);
  registry.set(name, spec);
  return spec;
}

/** A model by name: registered ones first, then the built-in table. */
export function modelSpec(name: string): ModelSpec {
  const s = registry.get(name) ?? (CONFIGS[name] && specFor(name, CONFIGS[name]));
  if (!s) throw new ModelError(`unknown model '${name}' (have: ${[...new Set([...registry.keys(), ...Object.keys(CONFIGS)])].join(', ')})`);
  return s;
}

export const MODELS: Record<string, ModelSpec> = Object.fromEntries(Object.entries(CONFIGS).map(([k, c]) => [k, specFor(k, c)]));
export const yunetFace = MODELS.yunet;
export const yoloPlate = MODELS['yolo-plate'];

export interface GeometryOptions {
  /** Region to detect in (default: the whole frame). */
  crop?: Rect;
  /**
   * Dynamic-size models only: scale the crop so its longer side is this many pixels, with the
   * input rounded up to the size multiple. A 1080p frame at 640 gets a 640x384 input instead
   * of 640x640; a 300 px ROI at 160 gets a 160x160 input.
   */
  longSide?: number;
}

export function geometryFor(spec: Preprocess & { id: string }, frameW: number, frameH: number, opts: GeometryOptions = {}): InputGeometry {
  const crop = opts.crop ?? { x: 0, y: 0, w: frameW, h: frameH };
  let [inW, inH] = [spec.width, spec.height];
  if (opts.longSide) {
    if (!spec.sizeMultiple) throw new Error(`model '${spec.id}' has a fixed input size`);
    const m = spec.sizeMultiple;
    const s = opts.longSide / Math.max(crop.w, crop.h);
    [inW, inH] = [Math.max(m, Math.ceil((crop.w * s) / m) * m), Math.max(m, Math.ceil((crop.h * s) / m) * m)];
  }
  if (!spec.letterbox) return { frameW, frameH, crop, inW, inH, contentW: inW, contentH: inH };
  const s = Math.min(inW / crop.w, inH / crop.h);
  return { frameW, frameH, crop, inW, inH, contentW: Math.round(crop.w * s), contentH: Math.round(crop.h * s) };
}
