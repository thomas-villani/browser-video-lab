// Headless runner for the single-pass adaptive pipeline (pipeline.html).
//
// stdout: one JSON summary per run (JSONL). stderr: progress + a summary table.
//
//   npm run pipeline -- --video /media/rickroll.mp4
//   npm run pipeline -- --video /media/walk.mp4 --maxstride 4,8,16 --longside 640 --dump
//
// Options (comma-separated values of matrix knobs form a cartesian product):
//   --video      URL path under public/ (required)
//   --maxstride  coarsest keyframe spacing            (matrix; default 16)
//   --longside   detector input long side, e.g. 640; 0 = model size, 640x640 (matrix; default 0)
//   --cut        scene-cut threshold, 0 = off         (matrix; default 0.15)
//   --miniou --minscore --lowscore --extend   scheduler options (see src/track/adaptive.ts)
//   --model      (default yunet)   --frames  0 = all (default 0)   --depth (default 2)
//   --dump       write per-frame boxes, sampled frames and scene scores to
//                results/<video>-pipeline-<name>.json (the JSONL line points at it)
//   --refine     native-resolution tiles where the reduced pass sees faces (runtime/refine.ts);
//                with it, matrix knobs --tile 640x384 --every 2 --grow 0 --trigger 0.3 --small 120
//   --browser msedge|chrome   --headed   --no-build
import { mkdirSync, writeFileSync } from 'node:fs';
import { basename, extname } from 'node:path';
import { parseArgs } from 'node:util';
import { withBrowser } from './lib/harness.mjs';

const { values: a } = parseArgs({
  options: {
    video: { type: 'string' },
    model: { type: 'string', default: 'yunet' },
    maxstride: { type: 'string', default: '16' },
    longside: { type: 'string', default: '0' },
    cut: { type: 'string', default: '0.15' },
    miniou: { type: 'string' },
    minscore: { type: 'string' },
    lowscore: { type: 'string' },
    extend: { type: 'boolean', default: false },
    frames: { type: 'string', default: '0' },
    depth: { type: 'string', default: '2' },
    dump: { type: 'boolean', default: false },
    refine: { type: 'boolean', default: false },
    tile: { type: 'string', default: '640x384' },
    every: { type: 'string', default: '2' },
    grow: { type: 'string', default: '0' },
    trigger: { type: 'string', default: '0.3' },
    small: { type: 'string', default: '120' },
    browser: { type: 'string', default: 'msedge' },
    headed: { type: 'boolean', default: false },
    'no-build': { type: 'boolean', default: false },
  },
});
if (!a.video) {
  console.error('usage: npm run pipeline -- --video /media/<clip>.mp4 [--maxstride 8,16] [--longside 640] [--dump]');
  process.exit(2);
}

const list = (s) => (s ?? '').split(',').map((x) => x.trim()).filter(Boolean);
let combos = [{}];
const knobs = ['maxstride', 'longside', 'cut', ...(a.refine ? ['tile', 'every', 'grow', 'trigger', 'small'] : [])];
for (const k of knobs) {
  if (list(a[k]).length) combos = combos.flatMap((c) => list(a[k]).map((v) => ({ ...c, [k]: v })));
}
const clip = basename(a.video, extname(a.video));
const fixed = { video: a.video, model: a.model, frames: a.frames, depth: a.depth };
for (const k of ['miniou', 'minscore', 'lowscore']) if (a[k]) fixed[k] = a[k];
if (a.extend) fixed.extend = '1';
if (a.dump) fixed.dump = '1';
if (a.refine) fixed.refine = '1';

const rows = [];
await withBrowser({ build: !a['no-build'], browser: a.browser, headed: a.headed }, async (open) => {
  for (const c of combos) {
    const refine = a.refine ? `-r${c.tile}e${c.every}g${c.grow}t${c.trigger}s${c.small}` : '';
    const name = `${a.model}-m${c.maxstride}-l${c.longside}-c${c.cut}${refine}`;
    console.error(`running ${clip} ${name}...`);
    const { result, page } = await open('/pipeline.html', { ...fixed, ...c });
    await page.close();
    result.name = name;
    if (result.boxes) {
      mkdirSync('results', { recursive: true });
      const file = `results/${clip}-pipeline-${name}.json`;
      writeFileSync(file, JSON.stringify(result));
      for (const k of ['boxes', 'sampled', 'sceneScores']) delete result[k];
      result.dump = file;
    }
    process.stdout.write(JSON.stringify(result) + '\n');
    rows.push(result);
  }
});

const cols = ['frames', 'callsPct', 'tilesPerFrame', 'inputMpxPerFrame', 'fps', 'ms'];
const w = Math.max(28, ...rows.map((r) => r.name.length + 2));
console.error(`\n${'name'.padEnd(w)}${cols.map((c) => c.padStart(17)).join('')}  cuts`);
for (const r of rows) {
  if (r.error) console.error(`${r.name.padEnd(w)} ERROR ${r.error}`);
  else console.error(`${r.name.padEnd(w)}${cols.map((c) => String(r[c]).padStart(17)).join('')}  ${r.cuts.length}`);
}
