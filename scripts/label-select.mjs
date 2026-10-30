// Step 1 of the face-labeling kit: choose frames to hand-label and extract them as JPEGs.
//
//   node scripts/label-select.mjs                 (default plan below; seeded, reproducible)
//   node scripts/label-select.mjs --seed 2 --random 150
//
// Per clip: `--random` stratified-random frames (one per equal time slice), plus for the walk
// `--stretches` dense stretches of `--stretch-seconds` at every `--stretch-step`th frame, in the
// busiest parts of the clip (most YuNet faces per frame), for measuring temporal gap-filling.
//
// Writes public/label/<clip>/<frame>.jpg and public/label/frames.json [{clip, frame, kind}].
// stdout: JSON summary.
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';

const { values: a } = parseArgs({
  options: {
    seed: { type: 'string', default: '1' },
    random: { type: 'string', default: '120' },
    stretches: { type: 'string', default: '2' },
    'stretch-seconds': { type: 'string', default: '10' },
    'stretch-step': { type: 'string', default: '5' },
  },
});

// Clip -> number of stratified-random frames (crowds get the most labels).
const PLAN = { walk: Number(a.random), rickroll: 40, 'briefing-full': 30 };

// mulberry32: small seeded PRNG so the selection is reproducible.
let s = Number(a.seed) >>> 0;
const rand = () => {
  s = (s + 0x6d2b79f5) >>> 0;
  let t = s;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const frames = [];
for (const [clip, count] of Object.entries(PLAN)) {
  const dump = JSON.parse(readFileSync(`results/${clip}.dets.json`, 'utf8'));
  const n = dump.decodedFrames;
  const fps = dump.fps_video ?? 30;
  const chosen = new Map();
  for (let i = 0; i < count; i++) {
    const f = Math.min(n - 1, Math.floor(((i + rand()) * n) / count));
    chosen.set(f, 'random');
  }
  if (clip === 'walk') {
    // Busiest windows by YuNet face count (score >= 0.5), non-overlapping.
    const len = Math.round(Number(a['stretch-seconds']) * fps);
    const counts = new Array(n).fill(0);
    for (const fd of dump.frameDetections) counts[fd.frame] = fd.boxes.filter((b) => b[4] >= 0.5).length;
    const prefix = [0];
    for (const c of counts) prefix.push(prefix[prefix.length - 1] + c);
    const taken = [];
    for (let k = 0; k < Number(a.stretches); k++) {
      let best = -1;
      let bestSum = -1;
      for (let st = 0; st + len <= n; st += Math.round(fps)) {
        if (taken.some((t) => st < t + len && t < st + len)) continue;
        const sum = prefix[st + len] - prefix[st];
        if (sum > bestSum) [best, bestSum] = [st, sum];
      }
      taken.push(best);
      for (let f = best; f < best + len; f += Number(a['stretch-step'])) chosen.set(f, `stretch${k}`);
    }
  }
  const list = [...chosen.entries()].sort((x, y) => x[0] - y[0]);
  mkdirSync(`public/label/${clip}`, { recursive: true });
  // ffmpeg's frame counter n is in presentation order, like the pipeline's frame index. Its
  // expression parser chokes on hundreds of terms, so select in batches (one decode pass each).
  for (let i = 0; i < list.length; i += 50) {
    const batch = list.slice(i, i + 50);
    execFileSync('ffmpeg', [
      '-v', 'error', '-y', '-i', `public/media/${clip}.mp4`,
      '-vf', `select='${batch.map(([f]) => `eq(n,${f})`).join('+')}'`, '-fps_mode', 'passthrough', '-q:v', '2',
      `public/label/${clip}/_%05d.jpg`,
    ]);
    // ffmpeg numbers outputs sequentially; rename to frame indices.
    for (const [j, [f, kind]] of batch.entries()) {
      renameSync(`public/label/${clip}/_${String(j + 1).padStart(5, '0')}.jpg`, `public/label/${clip}/${f}.jpg`);
      frames.push({ clip, frame: f, kind });
    }
  }
  console.error(`${clip}: ${list.length} frames`);
}
writeFileSync('public/label/frames.json', JSON.stringify(frames));
const byKind = {};
for (const f of frames) byKind[f.kind] = (byKind[f.kind] ?? 0) + 1;
console.log(JSON.stringify({ frames: frames.length, byKind, out: 'public/label/frames.json' }));
