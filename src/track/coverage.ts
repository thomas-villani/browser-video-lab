// Redaction-oriented metrics: how much of each ground-truth face is covered by the
// predicted boxes? (IoU is the wrong metric for redaction — a bigger box that fully covers
// a face is a success, not an error. Over-coverage is tracked separately as cost.)

import type { Box } from './bytetrack.ts';

export interface CoverageReport {
  gtBoxes: number;
  /** Mean fraction of each GT box's area covered by the best single predicted box. */
  meanCoverage: number;
  /** GT boxes less than 50% covered — effectively leaked faces. */
  missRate: number;
  /** GT boxes at least 90% covered. */
  fullRate: number;
  /** Total predicted area / total GT area: how much extra gets blurred. */
  areaRatio: number;
}

function intersection(a: Box, b: Box): number {
  const w = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  return w > 0 && h > 0 ? w * h : 0;
}

export function coverage(gt: Box[][], pred: Box[][]): CoverageReport {
  let count = 0;
  let sum = 0;
  let misses = 0;
  let full = 0;
  let gtArea = 0;
  let predArea = 0;
  gt.forEach((boxes, f) => {
    const p = pred[f] ?? [];
    for (const b of p) predArea += b[2] * b[3];
    for (const g of boxes) {
      const area = g[2] * g[3];
      if (area <= 0) continue;
      const c = p.reduce((best, b) => Math.max(best, intersection(g, b) / area), 0);
      count++;
      sum += c;
      gtArea += area;
      if (c < 0.5) misses++;
      if (c >= 0.9) full++;
    }
  });
  const r = (v: number) => Math.round(v * 10000) / 10000;
  return {
    gtBoxes: count,
    meanCoverage: r(count ? sum / count : 1),
    missRate: r(count ? misses / count : 0),
    fullRate: r(count ? full / count : 1),
    areaRatio: r(gtArea ? predArea / gtArea : 0),
  };
}
