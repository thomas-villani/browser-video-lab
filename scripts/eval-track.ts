// Evaluate sparse detection + tracking against a stride-1 detection dump.
//
// Ground truth = every frame's detections at or above --gt-score (the detector run on every
// frame). Each strategy sees only every Nth frame and must produce boxes for all frames;
// we measure how much of each ground-truth face its boxes cover (redaction view).
//
//   node scripts/eval-track.ts --dump results/<name>.dets.json
//   node scripts/eval-track.ts --dump ... --strides 1,5,10 --modes hold,offline --pads 0,0.15
//   node scripts/eval-track.ts --dump ... --cuts results/<video>.cuts.txt --scene 0,1
//   node scripts/eval-track.ts --dump ... --pred results/<clip>-pipeline-<name>.json   (score a live run)
//
// --cuts adds near-cut vs elsewhere miss rates; --scene 1 runs scene-aware sparse detection
// (tracker reset per scene, detector always on each scene's first and last frame).
// --modes roi: windowed bidirectional ROI tracking (src/track/roi.ts) with an oracle ROI
// detector (dense detections inside the crop); stride = window, --expand = ROI size factor,
// --group = windows packed per mosaic, --tile = ROI tile px. mpxPerFrame = model-input cost.
// --modes adaptive: divide-and-conquer scheduler; there, stride = the coarsest keyframe
// spacing, --min-iou sets when endpoint detections "agree", and --low-score sets the
// hysteresis threshold for pairing a confident box with a weak one (dump must include it).
// --extend holds one-sided (entering/leaving) faces across an interval instead of
// subdividing. --min-iou and --extend are single values, not matrix lists.
//
// stdout: JSONL (one row per stride x mode x pad). stderr: summary table.
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { adaptiveBoxes, DEFAULT_ADAPTIVE } from '../src/track/adaptive.ts';
import type { Box, Detection } from '../src/track/bytetrack.ts';
import { coverage } from '../src/track/coverage.ts';
import { DEFAULT_ROI, roiBoxes } from '../src/track/roi.ts';
import { padBox, type SparseMode, sparse } from '../src/track/sparse.ts';

const { values: a } = parseArgs({
  options: {
    dump: { type: 'string' },
    strides: { type: 'string', default: '1,2,3,5,8,10,15,25' },
    modes: { type: 'string', default: 'hold,online,offline' },
    pads: { type: 'string', default: '0,0.15' },
    'gt-score': { type: 'string', default: '0.6' },
    cuts: { type: 'string' },
    scene: { type: 'string', default: '0,1' },
    'min-iou': { type: 'string', default: '0.3' },
    'low-score': { type: 'string', default: '0.5' },
    extend: { type: 'boolean', default: false },
    pred: { type: 'string' },
    expand: { type: 'string', default: '2.5' },
    group: { type: 'string', default: '4' },
    tile: { type: 'string', default: '96' },
  },
});
if (!a.dump) {
  console.error('usage: node scripts/eval-track.ts --dump results/<name>.dets.json [--strides ...] [--modes ...] [--pads ...]');
  process.exit(2);
}

interface Dump {
  video?: { width: number; height: number };
  fps_video?: number;
  decodedFrames: number;
  frameDetections: { frame: number; t: number; boxes: number[][] }[];
}
const dump: Dump = JSON.parse(readFileSync(a.dump, 'utf8'));
const n = dump.decodedFrames;
const fps = dump.fps_video ?? 30;
const frames: Detection[][] = Array.from({ length: n }, () => []);
for (const fd of dump.frameDetections) {
  frames[fd.frame] = fd.boxes.map(([x, y, w, h, score]) => ({ box: [x, y, w, h] as Box, score }));
}
const gtScore = Number(a['gt-score']);
const gt: Box[][] = frames.map((ds) => ds.filter((d) => d.score >= gtScore).map((d) => d.box));
console.error(`dump: ${n} frames @ ${fps} fps, ${gt.flat().length} GT boxes (score >= ${gtScore}) in ${gt.filter((g) => g.length).length} frames`);

const nums = (s: string) => s.split(',').map(Number);

// --pred: score the boxes a live pipeline run produced (pipeline.mjs --dump) instead of
// replaying strategies over the dump.
if (a.pred) {
  const cols = ['pad', 'callsPct', 'meanCoverage', 'missRate', 'fullRate', 'areaRatio'] as const;
  console.error(`${'pred'.padEnd(44)}${cols.map((c) => c.padStart(13)).join('')}`);
  for (const file of a.pred.split(',')) {
    const run: { boxes: Box[][]; frames: number; callsPct: number } = JSON.parse(readFileSync(file, 'utf8'));
    if (run.frames !== n) console.error(`warning: ${file} has ${run.frames} frames, dump has ${n}`);
    for (const pad of nums(a.pads)) {
      const pred = run.boxes.map((bs) => bs.map((b) => padBox(b, pad)));
      const row = { pred: file, pad, callsPct: run.callsPct, ...coverage(gt, pred) };
      process.stdout.write(JSON.stringify(row) + '\n');
      console.error(`${file.slice(-44).padEnd(44)}${cols.map((c) => String(row[c]).padStart(13)).join('')}`);
    }
  }
  process.exit(0);
}

// Optional scene cuts: one frame index per line (first frame of each new scene).
const cuts = a.cuts
  ? readFileSync(a.cuts, 'utf8')
      .split(/\s+/)
      .filter(Boolean)
      .map(Number)
      .sort((x, y) => x - y)
  : [];
const minIou = Number(a['min-iou']);
if (!(minIou > 0 && minIou <= 1)) {
  console.error(`--min-iou must be a single number in (0, 1], got '${a['min-iou']}'`);
  process.exit(2);
}
if (cuts.length) console.error(`cuts: ${cuts.length} (mean scene ${Math.round(n / (cuts.length + 1))} frames)`);

/** GT restricted to frames within `radius` of a cut (or everything else). */
function nearCuts(radius: number, near: boolean): Box[][] {
  const mask = new Array<boolean>(n).fill(false);
  for (const c of cuts) for (let f = Math.max(0, c - radius); f < Math.min(n, c + radius); f++) mask[f] = true;
  return gt.map((boxes, f) => (mask[f] === near ? boxes : []));
}

const scenes = cuts.length ? nums(a.scene) : [0];

// Cost model in model-input pixels: a full-frame call at 640 long side (rounded up to 32), or
// one ROI tile of --tile px per crop (what a mosaic packs).
const [W, H] = [dump.video?.width ?? 1920, dump.video?.height ?? 1080];
const fullPx = 640 * Math.ceil((640 * Math.min(W, H)) / Math.max(W, H) / 32) * 32;
const tilePx = Number(a.tile) ** 2;

/** Oracle ROI detector for replay: the dense pass's detections lying mostly inside the crop. */
function oracle(f: number, crop?: Box): Detection[] {
  if (!crop) return frames[f];
  const [cx, cy, cw, ch] = crop;
  return frames[f].filter(({ box: [x, y, w, h] }) => {
    const iw = Math.min(x + w, cx + cw) - Math.max(x, cx);
    const ih = Math.min(y + h, cy + ch) - Math.max(y, cy);
    return iw > 0 && ih > 0 && (iw * ih) / (w * h) >= 0.6;
  });
}

type Mode = SparseMode | 'adaptive' | 'roi';

function run(stride: number, mode: Mode, scene: number, pad: number, lowScore: number, expand: number) {
  const sceneCuts = scene ? cuts : undefined;
  const padded = (boxes: Box[][]) => (pad ? boxes.map((bs) => bs.map((b) => padBox(b, pad))) : boxes);
  if (mode === 'roi') {
    const r = roiBoxes(
      n,
      (q) => oracle(q.frame, q.crop),
      {
        ...DEFAULT_ROI,
        window: stride,
        expand,
        minIou,
        lowScore,
        cuts: sceneCuts,
        frameW: W,
        frameH: H,
      },
      Number(a.group),
    );
    // Model invocations: full-frame calls plus one mosaic per batch of crops.
    return { boxes: padded(r.boxes), detectorCalls: r.fullCalls + r.roiBatches, px: r.fullCalls * fullPx + r.roiCrops * tilePx, crops: r.roiCrops };
  }
  if (mode !== 'adaptive') {
    const r = sparse(frames, { stride, mode, fps, pad, cuts: sceneCuts });
    return { ...r, px: r.detectorCalls * fullPx, crops: 0 };
  }
  const { boxes, sampled } = adaptiveBoxes(n, (f) => frames[f], {
    ...DEFAULT_ADAPTIVE,
    maxStride: stride,
    minIou,
    lowScore: lowScore,
    extendOneSided: a.extend,
    cuts: sceneCuts,
  });
  return { boxes: padded(boxes), detectorCalls: sampled.length, px: sampled.length * fullPx, crops: 0 };
}

const rows = [];
for (const stride of nums(a.strides))
  for (const mode of a.modes.split(',') as Mode[])
    for (const scene of scenes)
      for (const pad of nums(a.pads))
        for (const low of mode === 'adaptive' || mode === 'roi' ? nums(a['low-score']) : [0.5])
          for (const expand of mode === 'roi' ? nums(a.expand) : [0]) {
            const t0 = performance.now();
            const { boxes: pred, detectorCalls, px, crops } = run(stride, mode, scene, pad, low, expand);
            const ms = Math.round(performance.now() - t0);
            const row = {
              stride,
              mode,
              scene,
              pad,
              low,
              expand,
              detectorCalls,
              callsPct: Math.round((1000 * detectorCalls) / n) / 10,
              cropsPerFrame: Math.round((100 * crops) / n) / 100,
              mpxPerFrame: Math.round(px / n / 1000) / 1000,
              trackMs: ms,
              ...coverage(gt, pred),
              // Where do misses happen? Near cuts = within one stride of a scene change.
              ...(cuts.length && {
                missNearCut: coverage(nearCuts(stride, true), pred).missRate,
                missElsewhere: coverage(nearCuts(stride, false), pred).missRate,
              }),
            };
            process.stdout.write(JSON.stringify(row) + '\n');
            rows.push(row);
          }

const cols = [
  'stride',
  'mode',
  'scene',
  'pad',
  'low',
  'expand',
  'callsPct',
  'mpxPerFrame',
  'meanCoverage',
  'missRate',
  'fullRate',
  'areaRatio',
  ...(cuts.length ? (['missNearCut', 'missElsewhere'] as const) : []),
] as const;
console.error(cols.map((c) => c.padStart(12)).join(''));
for (const r of rows) console.error(cols.map((c) => String(r[c]).padStart(12)).join(''));
