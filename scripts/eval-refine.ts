// Simulate crowd refinement against human labels: a cheap low-resolution pass on every frame,
// plus native-resolution tiles only where the cheap pass sees faces. Oracle: a native tile is
// assumed to find what the full native-resolution run found inside it (boxes whose center lies
// in the tile), so policies can be swept in seconds before building them into the runtime.
//
//   node scripts/eval-refine.ts --clip walk
//   node scripts/eval-refine.ts --clip walk --tile 640x384,960x544 --grow 0,64,192 --every 1,2,4
//
// Inputs: results/<clip>.dets.json (dense low-res pass, scores >= 0.3), the native-resolution
// pipeline dump (--high), and results/labels/faces.jsonl.
//
// Policy knobs (comma-separated values form a cartesian product):
//   --tile    native tile size WxH; tiles overlap by --overlap px       (default 640x384)
//   --trig    low-res score that marks a face as a trigger              (default 0.3,0.5)
//   --small   only faces shorter than this (px) trigger; 0 = any        (default 0)
//   --grow    a trigger activates tiles within this many px of it       (default 0,128)
//   --every   refine every Nth frame; others reuse the nearest refined
//             frames' tile boxes (before and after)                     (default 1,2,4)
// Scoring as in eval-labels: miss = <50% of the face covered, boxes padded by --pad.
//
// stdout: JSONL (one row per policy). stderr: summary table.
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { activeTiles, ownBoxes, tileGrid } from '../src/runtime/refine.ts';

const { values: a } = parseArgs({
  options: {
    clip: { type: 'string', default: 'walk' },
    high: { type: 'string' },
    labels: { type: 'string', default: 'results/labels/faces.jsonl' },
    tile: { type: 'string', default: '640x384' },
    overlap: { type: 'string', default: '32' },
    trig: { type: 'string', default: '0.3,0.5' },
    small: { type: 'string', default: '0' },
    grow: { type: 'string', default: '0,128' },
    every: { type: 'string', default: '1,2,4' },
    pad: { type: 'string', default: '0.15' },
  },
});

type Box = [number, number, number, number];
const clip = a.clip!;
const dense = JSON.parse(readFileSync(`results/${clip}.dets.json`, 'utf8'));
const high = JSON.parse(readFileSync(a.high ?? `results/${clip}-pipeline-yunet-m16-l1920-c0.15.json`, 'utf8'));
const W: number = high.video.width;
const H: number = high.video.height;
const low = new Map<number, number[][]>(dense.frameDetections.map((fd: { frame: number; boxes: number[][] }) => [fd.frame, fd.boxes]));
const highBoxes: Box[][] = high.boxes;
const n = highBoxes.length;
// The cheap pass: 640 wide, at the source aspect (rounded up to YuNet's multiple of 32).
const LOW_MPX = (640 * Math.ceil((640 * H) / W / 32) * 32) / 1e6;

const latest = new Map<number, { faces: { box: Box; status: string }[] }>();
for (const line of readFileSync(a.labels!, 'utf8').split('\n').filter(Boolean)) {
  const l = JSON.parse(line);
  if (l.clip === clip) latest.set(l.frame, l);
}

const inter = (g: Box, b: Box) => {
  const w = Math.min(g[0] + g[2], b[0] + b[2]) - Math.max(g[0], b[0]);
  const h = Math.min(g[1] + g[3], b[1] + b[3]) - Math.max(g[1], b[1]);
  return w > 0 && h > 0 ? w * h : 0;
};
const covers = (b: Box, g: Box) => inter(g, b) / (g[2] * g[3]) >= 0.5;
const iou = (x: Box, y: Box) => inter(x, y) / (x[2] * x[3] + y[2] * y[3] - inter(x, y) || 1);
const pad = Number(a.pad);
const padBox = ([x, y, w, h]: Box): Box => [x - w * pad, y - h * pad, w * (1 + 2 * pad), h * (1 + 2 * pad)];
const grid = (tw: number, th: number, ov: number) => tileGrid(W, H, tw, th, ov);

interface Policy { tile: string; trig: number; small: number; grow: number; every: number }
const product = (o: Record<string, string>) =>
  Object.entries(o).reduce<Record<string, string>[]>((acc, [k, v]) => acc.flatMap((p) => v.split(',').map((x) => ({ ...p, [k]: x }))), [{}]);

function score(boxesAt: (f: number) => Box[]) {
  let faces = 0, missed = 0, small = 0, smallMissed = 0, preds = 0, correct = 0;
  for (const [f, l] of latest) {
    const pred = boxesAt(f).map(padBox);
    const known = l.faces.filter((x) => x.status !== 'rejected').map((x) => x.box);
    for (const g of l.faces.filter((x) => x.status === 'face').map((x) => x.box)) {
      const miss = !pred.some((b) => covers(b, g));
      faces++;
      missed += +miss;
      if (g[3] < 40) (small++, (smallMissed += +miss));
    }
    preds += pred.length;
    correct += pred.filter((b) => known.some((g) => iou(b, g) >= 0.3 || covers(b, g))).length;
  }
  const r = (v: number) => Math.round(v * 1000) / 10;
  return { missRate: r(missed / faces), missUnder40: r(smallMissed / small), precision: r(correct / preds) };
}

const lowAt = (f: number) => (low.get(f) ?? []).filter((b) => b[4] >= 0.5).map((b) => b.slice(0, 4) as Box);
const rows: Record<string, unknown>[] = [
  { policy: 'low only (640)', mpxPerFrame: LOW_MPX, tilesPerFrame: 0, ...score(lowAt) },
  { policy: `full native (${W}x${H})`, mpxPerFrame: (W * H) / 1e6, tilesPerFrame: 0, ...score((f) => highBoxes[f] ?? []) },
];

for (const o of product({ tile: a.tile!, trig: a.trig!, small: a.small!, grow: a.grow!, every: a.every! })) {
  const p: Policy = { tile: o.tile, trig: Number(o.trig), small: Number(o.small), grow: Number(o.grow), every: Number(o.every) };
  const [tw, th] = p.tile.split('x').map(Number);
  const tiles = grid(tw, th, Number(a.overlap));
  // Active tiles per refined frame, and the native boxes they yield.
  const refined = new Map<number, Box[]>();
  let tileCount = 0;
  for (let f = 0; f < n; f += p.every) {
    const trig = (low.get(f) ?? []).filter((b) => b[4] >= p.trig && (!p.small || b[3] < p.small)).map((b) => b.slice(0, 4) as Box);
    const active = activeTiles(tiles, trig, p.grow);
    tileCount += active.length;
    refined.set(f, active.flatMap((t) => ownBoxes(t, highBoxes[f] ?? [])));
  }
  const at = (f: number) => {
    const f0 = f - (f % p.every);
    const near = f === f0 ? [f0] : [f0, f0 + p.every];
    return [...lowAt(f), ...near.flatMap((g) => refined.get(g) ?? [])];
  };
  const refinedFrames = Math.ceil(n / p.every);
  rows.push({
    policy: `tile ${p.tile} trig>=${p.trig}${p.small ? ` h<${p.small}` : ''} grow ${p.grow} every ${p.every}`,
    ...p,
    mpxPerFrame: LOW_MPX + (tileCount * tw * th) / 1e6 / n,
    tilesPerFrame: tileCount / refinedFrames,
    ...score(at),
  });
}

console.error(`${clip}: ${latest.size} labeled frames, ${n} frames, ${grid(...(a.tile!.split(',')[0].split('x').map(Number) as [number, number]), Number(a.overlap)).length} tiles in the first grid`);
console.error(`${'policy'.padEnd(50)}${['Mpx/fr', 'tiles', 'miss %', '<40px', 'prec %'].map((c) => c.padStart(8)).join('')}`);
for (const r of rows) {
  process.stdout.write(JSON.stringify(r) + '\n');
  const num = (v: unknown, d = 1) => (v as number).toFixed(d);
  console.error(`${String(r.policy).padEnd(50)}${[num(r.mpxPerFrame, 3), num(r.tilesPerFrame), num(r.missRate), num(r.missUnder40), num(r.precision)].map((c) => c.padStart(8)).join('')}`);
}
