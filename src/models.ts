// Model specs: input geometry/normalization plus an output decoder.
// Decoders return boxes in *source frame* pixel coordinates.

export interface Detection {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  label: string;
}

/** How a source frame maps into the model input tensor. */
export interface InputGeometry {
  srcW: number;
  srcH: number;
  /** Width/height of the resized image inside the input tensor (top-left aligned; the rest is padding). */
  contentW: number;
  contentH: number;
}

export interface ModelSpec {
  id: string;
  url: string;
  inputName: string;
  width: number;
  height: number;
  /** Multiplier applied to [0,1] pixel values before mean/std (255 for OpenCV-style models). */
  scale: number;
  bgr: boolean;
  mean: [number, number, number];
  std: [number, number, number];
  /** Preserve aspect ratio with padding (true) or stretch to fill (false). */
  letterbox: boolean;
  /** `minScore` overrides the model's default detection threshold. */
  decode(outputs: Record<string, Float32Array>, geo: InputGeometry, minScore?: number): Detection[];
}

export function geometryFor(spec: ModelSpec, srcW: number, srcH: number): InputGeometry {
  if (!spec.letterbox) return { srcW, srcH, contentW: spec.width, contentH: spec.height };
  const s = Math.min(spec.width / srcW, spec.height / srcH);
  return { srcW, srcH, contentW: Math.round(srcW * s), contentH: Math.round(srcH * s) };
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
  url: '/models/yunet_face.onnx',
  inputName: 'input',
  width: 640,
  height: 640,
  scale: 255,
  bgr: true,
  mean: [0, 0, 0],
  std: [1, 1, 1],
  letterbox: true,
  decode(out, geo, minScore) {
    const threshold = minScore ?? 0.6;
    const toSrc = geo.srcW / geo.contentW;
    const dets: Detection[] = [];
    for (const s of [8, 16, 32]) {
      const cols = this.width / s;
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
        dets.push({ x: (cx - w / 2) * toSrc, y: (cy - h / 2) * toSrc, w: w * toSrc, h: h * toSrc, score, label: 'face' });
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
          x: (cx - w / 2) * geo.srcW,
          y: (cy - h / 2) * geo.srcH,
          w: w * geo.srcW,
          h: h * geo.srcH,
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
