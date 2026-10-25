// Evaluate sparse detection + tracking against a stride-1 detection dump.
//
// Ground truth = every frame's detections at or above --gt-score (the detector run on every
// frame). Each strategy sees only every Nth frame and must produce boxes for all frames;
// we measure how much of each ground-truth face its boxes cover (redaction view).
//
//   node scripts/eval-track.ts --dump results/<name>.dets.json
//   node scripts/eval-track.ts --dump ... --strides 1,5,10 --modes hold,offline --pads 0,0.15
//   node scripts/eval-track.ts --dump ... --cuts results/<video>.cuts.txt --scene 0,1
//
// --cuts adds near-cut vs elsewhere miss rates; --scene 1 runs scene-aware sparse detection
// (tracker reset per scene, detector always on each scene's first and last frame).
// --modes adaptive: divide-and-conquer scheduler; there, stride = the coarsest keyframe
// spacing, --min-iou sets when endpoint detections "agree", and --low-score sets the
// hysteresis threshold for pairing a confident box with a weak one (dump must include it).
// --extend holds one-sided (entering/leaving) faces across an interval instead of
// subdividing. --min-iou and --extend are single values, not matrix lists.
//
// stdout: JSONL (one row per stride x mode x pad). stderr: summary table.
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import type { Box, Detection } from '../src/track/bytetrack.ts';
import { coverage } from '../src/track/coverage.ts';
import { adaptiveBoxes, DEFAULT_ADAPTIVE } from '../src/track/adaptive.ts';
import { padBox, sparse, type SparseMode } from '../src/track/sparse.ts';

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
  },
});
if (!a.dump) {
  console.error('usage: node scripts/eval-track.ts --dump results/<name>.dets.json [--strides ...] [--modes ...] [--pads ...]');
  process.exit(2);
}

interface Dump {
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
console.error(
  `dump: ${n} frames @ ${fps} fps, ${gt.flat().length} GT boxes (score >= ${gtScore}) in ${gt.filter((g) => g.length).length} frames`,
);

// Optional scene cuts: one frame index per line (first frame of each new scene).
const cuts = a.cuts
  ? readFileSync(a.cuts, 'utf8').split(/\s+/).filter(Boolean).map(Number).sort((x, y) => x - y)
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

const nums = (s: string) => s.split(',').map(Number);
const scenes = cuts.length ? nums(a.scene) : [0];

function run(stride: number, mode: SparseMode | 'adaptive', scene: number, pad: number, lowScore: number) {
  const sceneCuts = scene ? cuts : undefined;
  if (mode !== 'adaptive') return sparse(frames, { stride, mode, fps, pad, cuts: sceneCuts });
  const { boxes, sampled } = adaptiveBoxes(n, (f) => frames[f], {
    ...DEFAULT_ADAPTIVE,
    maxStride: stride,
    minIou,
    lowScore: lowScore,
    extendOneSided: a.extend,
    cuts: sceneCuts,
  });
  return { boxes: pad ? boxes.map((bs) => bs.map((b) => padBox(b, pad))) : boxes, detectorCalls: sampled.length };
}

const rows = [];
for (const stride of nums(a.strides))
  for (const mode of a.modes.split(',') as (SparseMode | 'adaptive')[])
    for (const scene of scenes)
      for (const pad of nums(a.pads))
      for (const low of mode === 'adaptive' ? nums(a['low-score']) : [0.5]) {
        const t0 = performance.now();
        const { boxes: pred, detectorCalls } = run(stride, mode, scene, pad, low);
        const ms = Math.round(performance.now() - t0);
        const row = {
          stride,
          mode,
          scene,
          pad,
          low,
          detectorCalls,
          callsPct: Math.round((1000 * detectorCalls) / n) / 10,
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
  'stride', 'mode', 'scene', 'pad', 'low', 'callsPct', 'meanCoverage', 'missRate', 'fullRate', 'areaRatio',
  ...(cuts.length ? (['missNearCut', 'missElsewhere'] as const) : []),
] as const;
console.error(cols.map((c) => c.padStart(12)).join(''));
for (const r of rows) console.error(cols.map((c) => String(r[c]).padStart(12)).join(''));
