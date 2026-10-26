// Crowd refinement: a reduced-resolution pass finds where faces are; native-resolution tiles
// are then detected only there, recovering the small faces the reduced pass cannot see.
// Pure helpers, shared by the runtime (stream.ts) and the offline simulation (eval-refine).
import type { Box } from '../track/bytetrack';

export interface RefineOptions {
  /** Tile size in source pixels; each tile is detected at native resolution. */
  tileW: number;
  tileH: number;
  /** Minimum overlap between neighboring tiles, so faces on a seam are whole in one tile. */
  overlap: number;
  /** Reduced-pass score at which a face activates its tile (below the scheduler's minScore). */
  trigger: number;
  /** A face activates every tile within this many pixels of its center. */
  grow: number;
  /** Refine every Nth frame; frames in between reuse the nearest refined frames' tile boxes. */
  every: number;
}

export const DEFAULT_REFINE: RefineOptions = { tileW: 640, tileH: 384, overlap: 32, trigger: 0.3, grow: 0, every: 2 };

/** Evenly spaced tiles of size tw x th covering a w x h frame, overlapping by at least `overlap`. */
export function tileGrid(w: number, h: number, tw: number, th: number, overlap: number): Box[] {
  const axis = (len: number, t: number) => {
    if (t >= len) return [0];
    const k = Math.ceil((len - overlap) / (t - overlap));
    return Array.from({ length: k }, (_, i) => Math.round((i * (len - t)) / (k - 1)));
  };
  return axis(h, th).flatMap((y) => axis(w, tw).map((x) => [x, y, Math.min(tw, w), Math.min(th, h)] as Box));
}

const center = (b: Box) => [b[0] + b[2] / 2, b[1] + b[3] / 2];
const contains = (t: Box, [cx, cy]: number[], m = 0) => cx >= t[0] - m && cx < t[0] + t[2] + m && cy >= t[1] - m && cy < t[1] + t[3] + m;

/** Tiles holding (or within `grow` px of) the center of any trigger box. */
export function activeTiles(tiles: Box[], triggers: Box[], grow: number): Box[] {
  const c = triggers.map(center);
  return tiles.filter((t) => c.some((p) => contains(t, p, grow)));
}

/** Keep a tile's boxes only if centered in it (a face cut by the tile edge is whole in a neighbor). */
export function ownBoxes(tile: Box, boxes: Box[]): Box[] {
  return boxes.filter((b) => contains(tile, center(b)));
}

function iou(a: Box, b: Box): number {
  const w = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  const inter = w > 0 && h > 0 ? w * h : 0;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter || 1);
}

/** `base` plus each extra box that does not duplicate one already kept (IoU >= minIou). */
export function mergeBoxes(base: Box[], extra: Box[], minIou = 0.5): Box[] {
  const out = [...base];
  for (const b of extra) if (!out.some((o) => iou(o, b) >= minIou)) out.push(b);
  return out;
}

/** Refined frames whose tile boxes frame `f` uses: itself, or its neighbors without a cut between. */
export function refinedFor(f: number, every: number, last: number, cutBetween: (a: number, b: number) => boolean): number[] {
  const f0 = f - (f % every);
  if (f0 === f) return [f];
  return [f0, f0 + every].filter((g) => g <= last && !cutBetween(Math.min(f, g), Math.max(f, g)));
}
