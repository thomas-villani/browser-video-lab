// Model specs: input geometry/normalization plus an output decoder.
// Decoders return boxes in source-frame pixels, so callers never see crops or padding.

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

export interface ModelSpec {
  id: string;
  url: string;
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
  /** Returns boxes in source-frame pixels. `minScore` overrides the model's default threshold. */
  decode(outputs: Record<string, Float32Array>, geo: InputGeometry, minScore?: number): Detection[];
}

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

export function geometryFor(spec: ModelSpec, frameW: number, frameH: number, opts: GeometryOptions = {}): InputGeometry {
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

function iou(a: Detection, b: Detection): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter || 1);
}

export function nms(dets: Detection[], threshold: number): Detection[] {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const keep: Detection[] = [];
  for (const d of sorted) {
    if (keep.every((k) => k.label !== d.label || iou(k, d) < threshold)) keep.push(d);
  }
  return keep;
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** OpenCV YuNet (2023mar): anchor-free, strides 8/16/32, score = sqrt(cls * obj). */
export const yunetFace: ModelSpec = {
  id: 'yunet',
  url: '/models/yunet_face_dyn.onnx',
  inputName: 'input',
  width: 640,
  height: 640,
  sizeMultiple: 32,
  scale: 255,
  bgr: true,
  mean: [0, 0, 0],
  std: [1, 1, 1],
  letterbox: true,
  decode(out, geo, minScore) {
    const threshold = minScore ?? 0.6;
    const toSrc = geo.crop.w / geo.contentW;
    const { x: ox, y: oy } = geo.crop;
    const dets: Detection[] = [];
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
        dets.push({ x: ox + (cx - w / 2) * toSrc, y: oy + (cy - h / 2) * toSrc, w: w * toSrc, h: h * toSrc, score, label: 'face' });
      }
    }
    return nms(dets, 0.3);
  },
};

const COCO: Record<number, string> = { 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 6: 'bus', 8: 'truck' };

function rfdetr(id: string, file: string): ModelSpec {
  return {
    id,
    url: `/models/${file}`,
    inputName: 'pixel_values',
    width: 384,
    height: 384,
    scale: 1,
    bgr: false,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    letterbox: false,
    decode(out, geo, minScore) {
      const threshold = minScore ?? 0.5;
      const [logits, boxes] = [out.logits, out.pred_boxes];
      const classes = logits.length / 300;
      const dets: Detection[] = [];
      for (let q = 0; q < 300; q++) {
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
          x: geo.crop.x + (cx - w / 2) * geo.crop.w,
          y: geo.crop.y + (cy - h / 2) * geo.crop.h,
          w: w * geo.crop.w,
          h: h * geo.crop.h,
          score,
          label: COCO[cls] ?? `coco_${cls}`,
        });
      }
      return dets;
    },
  };
}

export const MODELS: Record<string, ModelSpec> = {
  yunet: yunetFace,
  rfdetr: rfdetr('rfdetr', 'rfdetr_nano.onnx'),
  rfdetr16: rfdetr('rfdetr16', 'rfdetr_nano_fp16.onnx'),
};
