// Detect scene cuts with ffmpeg's scene-change score (a stand-in for the planned GPU
// frame-difference shader). Writes one frame index per line: the first frame of each new scene.
//
//   node scripts/scene-cuts.mjs public/media/<clip>.mp4 [--threshold 0.3] [--out results/<clip>.cuts.txt]
//
// stdout: JSON summary { video, fps, threshold, cuts, out }.
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { basename, extname } from 'node:path';
import { parseArgs } from 'node:util';

const { values: a, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    threshold: { type: 'string', default: '0.3' },
    out: { type: 'string' },
  },
});
const video = positionals[0];
if (!video) {
  console.error('usage: node scripts/scene-cuts.mjs <video> [--threshold 0.3] [--out file]');
  process.exit(2);
}

const probe = execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'json', video]).toString();
const [num, den] = JSON.parse(probe).streams[0].r_frame_rate.split('/').map(Number);
const fps = num / den;

// showinfo logs each selected frame's pts_time to stderr; downscaling keeps this fast.
const ff = spawnSync('ffmpeg', ['-hide_banner', '-nostats', '-i', video, '-an', '-vf', `scale=320:-2,select='gt(scene,${a.threshold})',showinfo`, '-f', 'null', '-'], {
  encoding: 'utf8',
  maxBuffer: 256 * 1024 * 1024,
});
if (ff.status !== 0) {
  console.error(ff.stderr.slice(-2000));
  process.exit(1);
}
const cuts = [...ff.stderr.matchAll(/pts_time:([\d.]+)/g)].map((m) => Math.round(Number(m[1]) * fps));

const out = a.out ?? `results/${basename(video, extname(video))}.cuts.txt`;
mkdirSync('results', { recursive: true });
writeFileSync(out, cuts.join('\n') + '\n');
process.stdout.write(JSON.stringify({ video, fps, threshold: Number(a.threshold), cuts: cuts.length, out }) + '\n');
