// Detector models as data: where the ONNX file is, how to feed it (input geometry and
// normalization) and how to read its outputs (a decoder preset). Pure, so the planner can hash a
// model's description in Node, a pipeline preset can name models in JSON, and the decoders are
// tested without a browser. src/models.ts turns a description into a runnable ModelSpec.

export interface Detection {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  label: string;
}

/** A rectangle in source-frame pixels. */
export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** How a region of a source frame maps into the model input tensor. */
export interface InputGeometry {
  frameW: number;
  frameH: number;
  /** Region of the frame fed to the model: the whole frame unless cropping to an ROI. */
  crop: Rect;
  /** Input tensor width/height (the spec's size unless the model accepts dynamic sizes). */
  inW: number;
  inH: number;
  /** Width/height of the resized crop inside the input tensor (top-left aligned; the rest is padding). */
  contentW: number;
  contentH: number;
}

/**
 * Output layouts:
 * - `yunet`: OpenCV YuNet 2023mar (cls/obj/bbox per stride 8, 16, 32).
 * - `yolo`: Ultralytics YOLOv8-v11 export without NMS: one output [1, 4 + classes, anchors],
 *   cx, cy, w, h in input pixels then a score per class (needs `labels`).
 * - `yolo-end2end`: NMS inside the model. Rows of [batch, x1, y1, x2, y2, class, score]
 *   (7 columns, open-image-models) or [x1, y1, x2, y2, score, class] (6, Ultralytics `nms=True`
 *   and YOLOv10), in input pixels.
 * - `detr`: DETR / RT-DETR / RF-DETR: `logits` [1, queries, classes] (sigmoid) and `pred_boxes`
 *   [1, queries, 4] as cx, cy, w, h relative to the input.
 */
export type Decoder = 'yunet' | 'yolo' | 'yolo-end2end' | 'detr';
export const DECODERS: Decoder[] = ['yunet', 'yolo', 'yolo-end2end', 'detr'];

/** How pixels become the input tensor. */
export interface Preprocess {
  inputName: string;
  /** Default input size. */
  width: number;
  height: number;
  /** When set, the model accepts any input size that is a multiple of this. */
  sizeMultiple?: number;
  /** Multiplier applied to [0,1] pixel values before mean/std (255 for OpenCV-style models). */
  scale: number;
  bgr: boolean;
  mean: [number, number, number];
  std: [number, number, number];
  /** Preserve aspect ratio with padding (true) or stretch to fill (false). */
  letterbox: boolean;
  /** Padding value in [0,1] pixel units (gray 114/255 for YOLO); default 0. */
  pad?: number;
}

/** A detector model, as JSON (pipeline presets hold these). Unset fields come from the decoder's preset. */
export interface ModelConfig extends Partial<Preprocess> {
  /** The ONNX file: `/models/...` (served from public/), an http(s) URL, or (CLI and presets) a
   *  local path, relative to the preset. */
  url: string;
  decoder: Decoder;
  /** Shown as the layer's model, e.g. "yunet_face_dyn (2023mar)". */
  name?: string;
  /** Class index -> label. Required for `yolo` (it sets the class count); `class_<i>` when missing. */
  labels?: string[];
  /** Default score threshold. */
  score?: number;
  /** NMS overlap for decoders without NMS inside (yunet, yolo). */
  nms?: number;
}

export type ResolvedModel = Required<Omit<ModelConfig, 'sizeMultiple' | 'pad' | 'labels' | 'name'>> & Pick<ModelConfig, 'sizeMultiple' | 'pad' | 'labels' | 'name'>;

const YOLO_PRE: Preprocess & { score: number; nms: number } = {
  inputName: 'images',
  width: 640,
  height: 640,
  scale: 1,
  bgr: false,
  mean: [0, 0, 0],
  std: [1, 1, 1],
  letterbox: true,
  pad: 114 / 255,
  score: 0.4,
  nms: 0.45,
};

/** Each decoder's usual preprocessing and threshold (what its exporters produce). */
export const DECODER_DEFAULTS: Record<Decoder, Preprocess & { score: number; nms: number }> = {
  yunet: { inputName: 'input', width: 640, height: 640, sizeMultiple: 32, scale: 255, bgr: true, mean: [0, 0, 0], std: [1, 1, 1], letterbox: true, score: 0.6, nms: 0.3 },
  yolo: YOLO_PRE,
  'yolo-end2end': YOLO_PRE,
  detr: { inputName: 'pixel_values', width: 384, height: 384, scale: 1, bgr: false, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], letterbox: false, score: 0.5, nms: 0 },
};

/** COCO category ids (DETR family: index = id, '' where the id is unused). */
export const COCO91 = [
  '',
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  '',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  '',
  'backpack',
  'umbrella',
  '',
  '',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  '',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  '',
  'dining table',
  '',
  '',
  'toilet',
  '',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  '',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush',
];
/** The 80 COCO classes in training order (Ultralytics YOLO models trained on COCO). */
export const COCO80 = COCO91.filter(Boolean);

/** The models the built-in layers use, by name (`faces.model=yunet`). */
export const MODELS: Record<string, ModelConfig> = {
  yunet: { url: '/models/yunet_face_dyn.onnx', decoder: 'yunet', name: 'yunet_face_dyn (2023mar)', labels: ['face'] },
  'yolo-plate': { url: '/models/yolo_plate_t640.onnx', decoder: 'yolo-end2end', name: 'yolo-v9-t-640-license-plates (open-image-models)', labels: ['plate'] },
  rfdetr: { url: '/models/rfdetr_nano.onnx', decoder: 'detr', name: 'rf-detr nano (COCO)', labels: COCO91 },
  rfdetr16: { url: '/models/rfdetr_nano_fp16.onnx', decoder: 'detr', name: 'rf-detr nano fp16 (COCO)', labels: COCO91 },
};

export class ModelError extends Error {}

/** A model config with the decoder's defaults filled in, checked. */
export function resolveModel(c: ModelConfig, where = 'model'): ResolvedModel {
  if (!c || typeof c !== 'object') throw new ModelError(`${where}: expected an object with url and decoder`);
  if (typeof c.url !== 'string' || !c.url) throw new ModelError(`${where}: missing url (an .onnx file)`);
  if (!DECODERS.includes(c.decoder)) throw new ModelError(`${where}: decoder '${c.decoder}' is not one of ${DECODERS.join(', ')}`);
  if (c.labels !== undefined && !(Array.isArray(c.labels) && c.labels.every((l) => typeof l === 'string'))) throw new ModelError(`${where}: labels takes a list of strings`);
  if (c.decoder === 'yolo' && !c.labels?.length) throw new ModelError(`${where}: the yolo decoder needs labels (one per class, in training order; COCO models: the 80 COCO names)`);
  const out = { ...DECODER_DEFAULTS[c.decoder], ...c } as ResolvedModel;
  for (const k of ['width', 'height', 'scale', 'score', 'nms'] as const) {
    if (typeof out[k] !== 'number' || !Number.isFinite(out[k])) throw new ModelError(`${where}: ${k} takes a number`);
  }
  for (const k of ['mean', 'std'] as const) {
    if (!Array.isArray(out[k]) || out[k].length !== 3 || !out[k].every(Number.isFinite)) throw new ModelError(`${where}: ${k} takes three numbers (R, G, B)`);
  }
  return out;
}

const labelOf = (labels: string[] | undefined, i: number) => labels?.[i] || `class_${i}`;

function iou(a: Detection, b: Detection): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter || 1);
}

/** Greedy non-maximum suppression, per label. */
export function nms(dets: Detection[], threshold: number): Detection[] {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const keep: Detection[] = [];
  for (const d of sorted) {
    if (keep.every((k) => k.label !== d.label || iou(k, d) < threshold)) keep.push(d);
  }
  return keep;
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

type Outputs = Record<string, Float32Array>;
type Dims = Record<string, readonly number[]>;

/** The only output, or the named one; a clear error when neither is there. */
function output(out: Outputs, name: string): Float32Array {
  if (out[name]) return out[name];
  const keys = Object.keys(out);
  if (keys.length === 1) return out[keys[0]];
  throw new ModelError(`model output '${name}' not found (outputs: ${keys.join(', ')})`);
}

/**
 * Decode a model's outputs into detections in source-frame pixels. `dims` (output shapes) is
 * needed by the layouts whose row width varies (yolo-end2end); `minScore` overrides the model's.
 */
export function decode(m: ResolvedModel, out: Outputs, geo: InputGeometry, minScore?: number, dims: Dims = {}): Detection[] {
  const threshold = minScore ?? m.score;
  const toSrc = geo.crop.w / geo.contentW;
  const { x: ox, y: oy } = geo.crop;
  const dets: Detection[] = [];
  const push = (x1: number, y1: number, x2: number, y2: number, score: number, cls: number) =>
    dets.push({ x: ox + x1 * toSrc, y: oy + y1 * toSrc, w: (x2 - x1) * toSrc, h: (y2 - y1) * toSrc, score, label: labelOf(m.labels, cls) });

  if (m.decoder === 'yunet') {
    for (const s of [8, 16, 32]) {
      const cols = geo.inW / s;
      const [cls, obj, bbox] = [out[`cls_${s}`], out[`obj_${s}`], out[`bbox_${s}`]];
      for (let i = 0; i < cls.length; i++) {
        const score = Math.sqrt(clamp01(cls[i]) * clamp01(obj[i]));
        if (score < threshold) continue;
        const r = Math.floor(i / cols);
        const c = i % cols;
        const cx = (c + bbox[4 * i]) * s;
        const cy = (r + bbox[4 * i + 1]) * s;
        const w = Math.exp(bbox[4 * i + 2]) * s;
        const h = Math.exp(bbox[4 * i + 3]) * s;
        push(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, 0);
      }
    }
    return nms(dets, m.nms);
  }

  if (m.decoder === 'yolo') {
    const t = output(out, 'output0');
    const nc = m.labels!.length;
    const n = t.length / (4 + nc);
    if (!Number.isInteger(n)) throw new ModelError(`yolo output has ${t.length} values, not a multiple of 4 + ${nc} classes (do the labels match the model?)`);
    for (let a = 0; a < n; a++) {
      let best = 0;
      let cls = 0;
      for (let k = 0; k < nc; k++) {
        const v = t[(4 + k) * n + a];
        if (v > best) [best, cls] = [v, k];
      }
      if (best < threshold) continue;
      const [cx, cy, w, h] = [t[a], t[n + a], t[2 * n + a], t[3 * n + a]];
      push(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, cls);
    }
    return nms(dets, m.nms);
  }

  if (m.decoder === 'yolo-end2end') {
    const name = out.output0 ? 'output0' : Object.keys(out)[0];
    const rows = out[name] ?? new Float32Array(0);
    const shape = dims[name];
    const width = shape?.length ? shape[shape.length - 1] : 7;
    if (width !== 6 && width !== 7)
      throw new ModelError(`yolo-end2end rows have ${width} values (expected 7: batch, x1, y1, x2, y2, class, score; or 6: x1, y1, x2, y2, score, class)`);
    for (let i = 0; i + width <= rows.length; i += width) {
      const [x1, y1, x2, y2, score, cls] =
        width === 7 ? [rows[i + 1], rows[i + 2], rows[i + 3], rows[i + 4], rows[i + 6], rows[i + 5]] : [rows[i], rows[i + 1], rows[i + 2], rows[i + 3], rows[i + 4], rows[i + 5]];
      if (score < threshold) continue;
      push(x1, y1, x2, y2, score, Math.round(cls));
    }
    return dets;
  }

  // detr: boxes are relative to the (stretched) input, so they map straight onto the crop.
  const [logits, boxes] = [output(out, 'logits'), out.pred_boxes];
  if (!boxes) throw new ModelError(`detr model output 'pred_boxes' not found (outputs: ${Object.keys(out).join(', ')})`);
  const queries = boxes.length / 4;
  const classes = logits.length / queries;
  for (let q = 0; q < queries; q++) {
    let best = -Infinity;
    let cls = 0;
    for (let k = 0; k < classes; k++) {
      const v = logits[q * classes + k];
      if (v > best) [best, cls] = [v, k];
    }
    const score = 1 / (1 + Math.exp(-best));
    if (score < threshold) continue;
    const [cx, cy, w, h] = boxes.subarray(q * 4, q * 4 + 4);
    dets.push({
      x: ox + (cx - w / 2) * geo.crop.w,
      y: oy + (cy - h / 2) * geo.crop.h,
      w: w * geo.crop.w,
      h: h * geo.crop.h,
      score,
      label: labelOf(m.labels, cls),
    });
  }
  return dets;
}
