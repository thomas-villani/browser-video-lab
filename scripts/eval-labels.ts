// Score detectors and pipeline runs against human face labels (label.html →
// results/labels/faces.jsonl; the last saved line per frame wins).
//
//   node scripts/eval-labels.ts                                  (YuNet dense + RetinaFace)
//   node scripts/eval-labels.ts --pred results/walk-pipeline-yunet-m16-l640-c0.15.json --pad 0.15
//
// A labeled face is *missed* when no predicted box covers at least half of it (the redaction
// criterion used throughout). Faces marked uncertain are ignored, as are predictions on them.
// Precision: share of predicted boxes that overlap a face or uncertain label (IoU >= 0.3 or
// covering half of it). Results are broken down by clip and by face height in pixels.
//
// stdout: JSONL (one row per source x clip x size bucket). stderr: summary table.
import { existsSync, readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';

const { values: a } = parseArgs({
  options: {
    labels: { type: 'string', default: 'results/labels/faces.jsonl' },
    yunet: { type: 'string', default: '0.3,0.5,0.6' },
    pred: { type: 'string' },
    pad: { type: 'string', default: '0' },
  },
});

type Box = [number, number, number, number];
interface Label {
  clip: string;
  frame: number;
  kind: string;
  faces: { box: Box; status: 'face' | 'rejected' | 'uncertain'; source: string }[];
}

const latest = new Map<string, Label>();
for (const line of readFileSync(a.labels, 'utf8').split('\n').filter(Boolean)) {
  const l: Label = JSON.parse(line);
  latest.set(`${l.clip}/${l.frame}`, l);
}
const labels = [...latest.values()];
console.error(`${labels.length} labeled frames, ${labels.reduce((n, l) => n + l.faces.filter((f) => f.status === 'face').length, 0)} faces`);

const inter = (g: Box, b: Box) => {
  const w = Math.min(g[0] + g[2], b[0] + b[2]) - Math.max(g[0], b[0]);
  const h = Math.min(g[1] + g[3], b[1] + b[3]) - Math.max(g[1], b[1]);
  return w > 0 && h > 0 ? w * h : 0;
};
const covers = (b: Box, g: Box) => inter(g, b) / (g[2] * g[3]) >= 0.5;
const iou = (x: Box, y: Box) => inter(x, y) / (x[2] * x[3] + y[2] * y[3] - inter(x, y) || 1);
const padBox = ([x, y, w, h]: Box, p: number): Box => [x - w * p, y - h * p, w * (1 + 2 * p), h * (1 + 2 * p)];

const BUCKETS: [string, number, number][] = [
  ['all', 0, Infinity],
  ['<20px', 0, 20],
  ['20-40px', 20, 40],
  ['40-80px', 40, 80],
  ['>=80px', 80, Infinity],
];

/** Predictions per labeled frame from one source. */
type Source = { name: string; boxes: (clip: string, frame: number) => Box[] | undefined };
const sources: Source[] = [];

const dumps = new Map<string, Map<number, number[][]>>();
const dense = (clip: string) => {
  if (!dumps.has(clip)) {
    const file = `results/${clip}.dets.json`;
    dumps.set(
      clip,
      existsSync(file) ? new Map(JSON.parse(readFileSync(file, 'utf8')).frameDetections.map((fd: { frame: number; boxes: number[][] }) => [fd.frame, fd.boxes])) : new Map(),
    );
  }
  return dumps.get(clip)!;
};
for (const t of a.yunet.split(',').filter(Boolean).map(Number)) {
  sources.push({ name: `yunet-dense>=${t}`, boxes: (c, f) => (dense(c).get(f) ?? []).filter((b) => b[4] >= t).map((b) => b.slice(0, 4) as Box) });
}
if (existsSync('public/label/retina.json')) {
  const retina: Record<string, number[][]> = JSON.parse(readFileSync('public/label/retina.json', 'utf8'));
  for (const t of [0.5, 0.8]) sources.push({ name: `retina>=${t}`, boxes: (c, f) => retina[`${c}/${f}`]?.filter((b) => b[4] >= t).map((b) => b.slice(0, 4) as Box) });
}
for (const file of (a.pred ?? '').split(',').filter(Boolean)) {
  const run = JSON.parse(readFileSync(file, 'utf8'));
  const clip = run.video?.url
    ?.split('/')
    .pop()
    ?.replace(/\.[^.]+$/, '');
  sources.push({
    name: file
      .split('/')
      .pop()!
      .replace(/\.json$/, ''),
    boxes: (c, f) => (c === clip ? run.boxes[f] : undefined),
  });
}

const pad = Number(a.pad);
const rows = [];
for (const src of sources) {
  const clips = ['all', ...new Set(labels.map((l) => l.clip))];
  for (const clip of clips) {
    for (const [bucket, lo, hi] of BUCKETS) {
      let faces = 0;
      let missed = 0;
      let preds = 0;
      let correct = 0;
      let frames = 0;
      for (const l of labels) {
        if (clip !== 'all' && l.clip !== clip) continue;
        const raw = src.boxes(l.clip, l.frame);
        if (!raw) continue;
        frames++;
        const pred = raw.map((b) => (pad ? padBox(b, pad) : b));
        const gt = l.faces.filter((x) => x.status === 'face').map((x) => x.box);
        const known = l.faces.filter((x) => x.status !== 'rejected').map((x) => x.box);
        for (const g of gt) {
          if (g[3] < lo || g[3] >= hi) continue;
          faces++;
          if (!pred.some((b) => covers(b, g))) missed++;
        }
        if (bucket === 'all') {
          preds += pred.length;
          correct += pred.filter((b) => known.some((g) => iou(b, g) >= 0.3 || covers(b, g))).length;
        }
      }
      if (!frames || (!faces && bucket !== 'all')) continue;
      const r = (v: number) => Math.round(v * 1000) / 10;
      const row = {
        source: src.name,
        clip,
        bucket,
        frames,
        faces,
        missed,
        missRate: r(faces ? missed / faces : 0),
        ...(bucket === 'all' && { preds, precision: r(preds ? correct / preds : 1) }),
      };
      process.stdout.write(JSON.stringify(row) + '\n');
      rows.push(row);
    }
  }
}
console.error(`${'source'.padEnd(40)}${'clip'.padEnd(15)}${'bucket'.padEnd(9)}${['faces', 'miss %', 'prec %'].map((c) => c.padStart(8)).join('')}`);
for (const r of rows) {
  console.error(
    `${r.source.slice(0, 39).padEnd(40)}${r.clip.padEnd(15)}${r.bucket.padEnd(9)}${[r.faces, r.missRate, 'precision' in r ? r.precision : ''].map((c) => String(c).padStart(8)).join('')}`,
  );
}
