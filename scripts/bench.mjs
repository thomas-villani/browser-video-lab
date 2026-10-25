// Headless benchmark runner. Builds the page, serves it with `vite preview`, and drives
// Edge/Chrome via playwright-core over a matrix of configs.
//
// stdout: one JSON result per line (JSONL) -- machine-readable.
// stderr: progress + a summary table -- human-readable.
//
//   npm run bench -- --model yunet,rfdetr16 --path gpu,cpu --frames 300
//   npm run bench -- --path decode
//   npm run bench -- --model yunet --path gpu --stride 1,5 --screenshot
//
// Options (comma-separated values form a cartesian matrix):
//   --model    yunet | rfdetr | rfdetr16            (default yunet)
//   --path     gpu | cpu | decode                   (default gpu,cpu)
//   --ep       webgpu | wasm                        (default webgpu)
//   --frames   frames to decode per run, 0 = all    (default 300)
//   --stride   run the model on every Nth frame     (default 1)
//   --video    URL path under public/               (default /media/pan1080.mp4)
//   --outloc   cpu | gpu | batch: output readback mode (default cpu; gpu path only)
//   --capture  0 | 1: ORT WebGPU graph capture       (default 0; gpu path only)
//   --depth    frames in flight, 1 = sequential      (default 1)
//   --browser  msedge | chrome                      (default msedge)
//   --headed   show the browser window
//   --screenshot  save results/<config>.png of the first-frame detections
//   --dump     write every processed frame's detections to results/<video>-<name>.dets.json
//   --score    override the model's detection threshold (e.g. 0.3 to keep weak boxes for ByteTrack)
//   --no-build reuse the existing dist/
import { execSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { basename, extname } from 'node:path';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright-core';
import { preview } from 'vite';

const { values: a } = parseArgs({
  options: {
    model: { type: 'string', default: 'yunet' },
    path: { type: 'string', default: 'gpu,cpu' },
    ep: { type: 'string', default: 'webgpu' },
    frames: { type: 'string', default: '300' },
    stride: { type: 'string', default: '1' },
    video: { type: 'string', default: '/media/pan1080.mp4' },
    browser: { type: 'string', default: 'msedge' },
    headed: { type: 'boolean', default: false },
    screenshot: { type: 'boolean', default: false },
    outloc: { type: 'string', default: 'cpu' },
    capture: { type: 'string', default: '0' },
    depth: { type: 'string', default: '1' },
    dump: { type: 'boolean', default: false },
    score: { type: 'string' },
    'no-build': { type: 'boolean', default: false },
  },
});

// Matrix knobs: each accepts a comma-separated list; runs are the cartesian product.
const KNOBS = ['model', 'path', 'ep', 'stride', 'outloc', 'capture', 'depth'];
const list = (s) => s.split(',').map((x) => x.trim()).filter(Boolean);
let combos = [{}];
for (const k of KNOBS) combos = combos.flatMap((c) => list(a[k]).map((v) => ({ ...c, [k]: v })));

const matrix = [];
const seen = new Set();
for (const c of combos) {
  if (c.path === 'gpu' && c.ep !== 'webgpu') continue;
  // GPU-only knobs don't apply to the CPU-input paths.
  if (c.path !== 'gpu' && (c.outloc !== 'cpu' || c.capture !== '0')) continue;
  const name =
    c.path === 'decode'
      ? 'decode'
      : `${c.model}-${c.ep}-${c.path}-s${c.stride}-o${c.outloc}-d${c.depth}${c.capture === '1' ? '-cap' : ''}`;
  // Decode-only ignores the other knobs: run it once.
  if (seen.has(name)) continue;
  seen.add(name);
  const cfg = { ...c, frames: a.frames, video: a.video };
  if (a.dump) cfg.dump = '1';
  if (a.score) cfg.score = a.score;
  matrix.push({ name, cfg });
}

if (!a['no-build']) {
  console.error('building...');
  execSync('npx vite build --logLevel warn', { stdio: ['ignore', 'ignore', 'inherit'] });
}
const server = await preview({ preview: { port: 4317, strictPort: false }, logLevel: 'warn' });
const base = server.resolvedUrls.local[0];

const browser = await chromium.launch({
  channel: a.browser,
  headless: !a.headed,
  args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
});

const rows = [];
try {
  for (const { name, cfg } of matrix) {
    console.error(`running ${name}...`);
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    // Relay only the first few browser errors (first line each) so a failing shader can't flood the output.
    let relayed = 0;
    const relay = (kind, text) => relayed++ < 5 && console.error(`  [${kind}] ${text.split('\n')[0].slice(0, 200)}`);
    page.on('pageerror', (e) => relay('pageerror', e.message));
    page.on('console', (m) => m.type() === 'error' && relay('console', m.text()));
    await page.goto(`${base}?${new URLSearchParams(cfg)}`);
    await page.waitForFunction(() => window.__BENCH_RESULT__, null, { timeout: 15 * 60_000, polling: 250 });
    const result = await page.evaluate(() => window.__BENCH_RESULT__);
    result.name = name;
    mkdirSync('results', { recursive: true });
    if (a.screenshot && !result.error) {
      await page.locator('#preview').screenshot({ path: `results/${name}.png` });
      result.screenshot = `results/${name}.png`;
    }
    if (result.frameDetections) {
      // Per-frame detections go to their own file (named by video, so dumps of different
      // clips never overwrite each other); the JSONL line points at it.
      const file = `results/${basename(a.video, extname(a.video))}-${name}.dets.json`;
      const { frameDetections, ...meta } = result;
      writeFileSync(file, JSON.stringify({ ...meta, frameDetections }));
      delete result.frameDetections;
      result.dump = file;
    }
    process.stdout.write(JSON.stringify(result) + '\n');
    rows.push(result);
    await page.close();
  }
} finally {
  await browser.close();
  await server.close();
}

// Summary table: mean ms per stage.
const STAGES = ['pre', 'infer', 'readback', 'post', 'decodeWait'];
const fmt = (s) => (s ? `${s.mean}` : '-');
console.error(`\n${'name'.padEnd(34)} ${'fps'.padStart(6)} ${STAGES.map((s) => s.padStart(10)).join(' ')}  dets@f0`);
for (const r of rows) {
  if (r.error) {
    console.error(`${r.name.padEnd(34)} ERROR ${r.error}`);
    continue;
  }
  const cells = STAGES.map((s) => fmt(r.stageMs[s]).padStart(10)).join(' ');
  console.error(`${r.name.padEnd(34)} ${String(r.fps).padStart(6)} ${cells}  ${r.firstFrameDetections.length}`);
}
if (rows[0]?.env?.adapter) console.error(`\nadapter: ${JSON.stringify(rows[0].env.adapter)}`);
