// Step 3 of the face-labeling kit: merge candidate faces from RetinaFace (retina.json) and
// YuNet (the dense dumps) into review tasks, most uncertain frames first.
//
//   node scripts/label-tasks.mjs
//
// Reads public/label/frames.json, public/label/retina.json, results/<clip>.dets.json.
// Writes public/label/tasks.json. stdout: JSON summary.
import { readFileSync, writeFileSync } from 'node:fs';

const frames = JSON.parse(readFileSync('public/label/frames.json', 'utf8'));
const retina = JSON.parse(readFileSync('public/label/retina.json', 'utf8'));
const dumps = {};
const yunetAt = (clip, frame) => {
  dumps[clip] ??= new Map(JSON.parse(readFileSync(`results/${clip}.dets.json`, 'utf8')).frameDetections.map((fd) => [fd.frame, fd.boxes]));
  return dumps[clip].get(frame) ?? [];
};

const iou = (a, b) => {
  const w = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
  const inter = w > 0 && h > 0 ? w * h : 0;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter || 1);
};

const tasks = [];
let agreeing = 0;
for (const { clip, frame, kind } of frames) {
  const r = retina[`${clip}/${frame}`] ?? [];
  const y = yunetAt(clip, frame);
  // Greedy match by IoU (highest first); unmatched boxes stay single-source candidates.
  const pairs = [];
  r.forEach((a, i) => y.forEach((b, j) => pairs.push([iou(a, b), i, j])));
  pairs.sort((p, q) => q[0] - p[0]);
  const usedR = new Set();
  const usedY = new Set();
  const candidates = [];
  for (const [o, i, j] of pairs) {
    if (o < 0.4 || usedR.has(i) || usedY.has(j)) continue;
    usedR.add(i);
    usedY.add(j);
    // RetinaFace boxes are tighter and include the forehead consistently; keep its geometry.
    candidates.push({ box: r[i].slice(0, 4), retina: r[i][4], yunet: y[j][4] });
  }
  r.forEach((a, i) => usedR.has(i) || candidates.push({ box: a.slice(0, 4), retina: a[4] }));
  // Weak YuNet-only boxes (<0.5) that RetinaFace at native resolution did not see are almost all
  // noise; dropping them keeps review fast (a missed face can still be added by hand).
  y.forEach((b, j) => usedY.has(j) || b[4] < 0.5 || candidates.push({ box: b.slice(0, 4), yunet: b[4] }));
  // Uncertainty: candidates the two detectors disagree on, or that neither is confident about.
  const confident = (c) => (c.retina >= 0.5 && c.yunet >= 0.5) || c.retina >= 0.8;
  const disagreement = candidates.filter((c) => !confident(c)).length;
  agreeing += candidates.length - disagreement;
  tasks.push({ clip, frame, kind, image: `/label/${clip}/${frame}.jpg`, candidates, disagreement });
}

// Random frames first, most uncertain first (so stopping early still captures the hard ones);
// then the dense stretches in time order (they are labeled as sequences).
const random = tasks.filter((t) => t.kind === 'random').sort((a, b) => b.disagreement - a.disagreement);
const stretches = tasks.filter((t) => t.kind !== 'random').sort((a, b) => a.kind.localeCompare(b.kind) || a.frame - b.frame);
const ordered = [...random, ...stretches];
writeFileSync('public/label/tasks.json', JSON.stringify(ordered));
const candidates = ordered.reduce((n, t) => n + t.candidates.length, 0);
console.log(JSON.stringify({ tasks: ordered.length, candidates, agreeing, uncertain: candidates - agreeing, out: 'public/label/tasks.json' }));
