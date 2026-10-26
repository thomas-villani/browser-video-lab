// Scene-cut accuracy of the pipeline's thumbnail scores against reference cuts (e.g. ffmpeg's,
// from scripts/scene-cuts.mjs), swept over thresholds. A predicted cut within --tolerance frames
// of an unmatched reference cut is a hit.
//
//   node scripts/eval-cuts.ts --pred results/<clip>-pipeline-<name>.json --ref results/<clip>.cuts.txt
//   node scripts/eval-cuts.ts --pred ... --ref ... --thresholds 0.05,0.1,0.2 --tolerance 1
//
// stdout: JSONL (one row per threshold). stderr: summary table.
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';

const { values: a } = parseArgs({
  options: {
    pred: { type: 'string' },
    ref: { type: 'string' },
    thresholds: { type: 'string', default: '0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3' },
    tolerance: { type: 'string', default: '1' },
  },
});
if (!a.pred || !a.ref) {
  console.error('usage: node scripts/eval-cuts.ts --pred <pipeline dump> --ref <cuts.txt> [--thresholds ...] [--tolerance 1]');
  process.exit(2);
}
const scores: number[] = JSON.parse(readFileSync(a.pred, 'utf8')).sceneScores;
if (!scores) {
  console.error(`${a.pred} has no sceneScores (run the pipeline with --dump)`);
  process.exit(2);
}
const ref = readFileSync(a.ref, 'utf8').split(/\s+/).filter(Boolean).map(Number).filter((c) => c > 0);
const tol = Number(a.tolerance);

const rows = [];
for (const threshold of a.thresholds.split(',').map(Number)) {
  const pred = scores.flatMap((s, f) => (f > 0 && s >= threshold ? [f] : []));
  const used = new Set<number>();
  let hits = 0;
  for (const p of pred) {
    const j = ref.findIndex((r, i) => !used.has(i) && Math.abs(r - p) <= tol);
    if (j >= 0) {
      used.add(j);
      hits++;
    }
  }
  const precision = pred.length ? hits / pred.length : 1;
  const recall = ref.length ? hits / ref.length : 1;
  const r3 = (v: number) => Math.round(v * 1000) / 1000;
  const row = {
    threshold,
    predicted: pred.length,
    reference: ref.length,
    hits,
    precision: r3(precision),
    recall: r3(recall),
    f1: r3(precision + recall ? (2 * precision * recall) / (precision + recall) : 0),
    missed: ref.filter((_, i) => !used.has(i)),
  };
  process.stdout.write(JSON.stringify(row) + '\n');
  rows.push(row);
}
const cols = ['threshold', 'predicted', 'reference', 'hits', 'precision', 'recall', 'f1'] as const;
console.error(cols.map((c) => c.padStart(11)).join(''));
for (const r of rows) console.error(cols.map((c) => String(r[c]).padStart(11)).join(''));
