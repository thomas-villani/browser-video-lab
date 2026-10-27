#!/usr/bin/env node
// videodoc CLI: analyze a video once into <video>.videodoc.json, then query the doc.
// Compute runs in a headless Edge/Chrome (WebGPU) driving analyze.html; planning, caching and
// reading docs run in Node, so commands with nothing to compute never start a browser.
//
// stdout: JSON results (one object per command; `markdown` without --out prints Markdown).
// stderr: progress and human summaries. Exit codes: 0 ok, 1 failure, 2 usage error.
import { existsSync, mkdirSync, openSync, readFileSync, readSync, closeSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import {
  deriveLayers, DocError, docPathFor, fingerprint, formatDoc, LAYER_NAMES, newDoc, parseDoc, parseSettings, plan, PlanError, PRODUCERS, summarize,
  toMarkdown, withLayers,
  type KeyframeItem, type Layers, type Plan, type VideoDoc,
} from '../src/doc/index.ts';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const DEFAULT_LAYERS = ['speech', 'transcript', 'scenes', 'keyframes'];
const MARKDOWN_LAYERS = ['transcript', 'keyframes'];
// User paths are relative to where the user ran the command (npm run moves cwd to the repo root).
const userPath = (p: string) => resolve(process.env.INIT_CWD ?? process.cwd(), p);

const HELP = `videodoc: analyze a video once, query it many times. Runs in a local headless browser (WebGPU).

Usage: videodoc <command> [options]

Commands:
  check                  What this machine can run: browser, WebGPU adapter, decoders, models.
  analyze <video>        Compute layers into <video>.videodoc.json. Layers already in the doc with
                         the same settings are reused; with nothing to compute, no browser starts.
  markdown <video>       Read the video: shots, timed transcript paragraphs, silent gaps and
                         (with --images) keyframe JPEGs, as Markdown. Analyzes missing layers first
                         (${MARKDOWN_LAYERS.join(', ')} and their dependencies, plus any --layers).
  info <video|doc>       Summarize a saved doc (no browser).
  layers                 Layers, their dependencies and settings with defaults (JSON).

analyze options:
  --layers <list>        Comma-separated (default: ${DEFAULT_LAYERS.join(',')}); dependencies are added.
                         Available: ${LAYER_NAMES.join(', ')}.
  --set <layer.key=v>    Override a setting, repeatable (e.g. --set faces.every=4). See 'videodoc layers'.
  --out <path>           Doc path (default: <video>.videodoc.json).
  --force                Recompute the requested layers even if cached.
  --dry-run              Print the plan and exit.

markdown options:
  --out <path>           Write the Markdown here (image links relative to it) and print a JSON
                         summary; default: Markdown on stdout with absolute image paths.
  --images               Link keyframe JPEGs, extracting missing ones into <video>.keyframes/
                         (named by frame; delete the folder to re-extract at another size).
  --image-size <px>      Long side of extracted keyframes (default 768).
  --layers, --set, --force   As for analyze (e.g. --layers faces adds faces per shot).

Common options:
  --browser msedge|chrome   (default msedge)   --headed   --help

stdout is JSON; progress and summaries go to stderr.`;

class UsageError extends Error {}

const { values: a, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    layers: { type: 'string' },
    set: { type: 'string', multiple: true, default: [] },
    out: { type: 'string' },
    force: { type: 'boolean', default: false },
    'dry-run': { type: 'boolean', default: false },
    images: { type: 'boolean', default: false },
    'image-size': { type: 'string', default: '768' },
    browser: { type: 'string', default: 'msedge' },
    headed: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
});

const out = (v: unknown) => process.stdout.write(JSON.stringify(v) + '\n');
const log = (s: string) => process.stderr.write(s + '\n');

/** Model files each layer needs (dev layout: public/models), with how to get them. */
function missingModels(p: Plan): string[] {
  const need: [string, string][] = [];
  for (const l of p.layers) {
    if (l.status !== 'compute') continue;
    if (l.name === 'speech') need.push(['public/models/silero_vad.onnx', 'see README (Silero VAD v5)']);
    if (l.name === 'faces') need.push([`public/models/yunet_face_dyn.onnx`, 'see README (YuNet, dynamic shape)']);
    if (l.name === 'transcript') {
      const { model, encoder, decoder } = l.config as Record<string, string>;
      const suffix = (d: string) => (d === 'fp32' ? '' : d === 'q8' ? '_quantized' : `_${d}`);
      const files = [`encoder_model${suffix(encoder)}`, `decoder_model_merged${suffix(decoder)}`];
      for (const f of files) need.push([`public/models/${model}/onnx/${f}.onnx`, `node scripts/fetch-model.mjs ${model} ${files.join(' ')}`]);
    }
  }
  return need.filter(([f]) => !existsSync(join(ROOT, f))).map(([f, fix]) => `missing ${f} (fix: ${fix})`);
}

async function fingerprintFile(path: string) {
  const size = statSync(path).size;
  const fd = openSync(path, 'r');
  try {
    const read = async (start: number, end: number) => {
      const buf = Buffer.alloc(end - start);
      readSync(fd, buf, 0, buf.length, start);
      return new Uint8Array(buf);
    };
    return { size, fingerprint: await fingerprint(size, read) };
  } finally {
    closeSync(fd);
  }
}

function loadDoc(path: string): VideoDoc | undefined {
  return existsSync(path) ? parseDoc(readFileSync(path, 'utf8'), path) : undefined;
}

function saveDoc(path: string, doc: VideoDoc) {
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, formatDoc(doc));
  renameSync(tmp, path);
}

async function harness() {
  // The harness builds and serves the pages from the repo root.
  process.chdir(ROOT);
  return import(new URL('../scripts/lib/harness.mjs', import.meta.url).href);
}

/** Progress on stderr: rewritten in place on a terminal, every ~10% otherwise. */
function progressPrinter() {
  const tty = process.stderr.isTTY;
  let last = '';
  let lastStage = '';
  let lastPct = 0;
  const t0 = Date.now();
  return {
    update(p: { pass: string; stage: string; done: number; total: number }) {
      const pct = p.total ? Math.floor((100 * p.done) / p.total) : 0;
      const secs = (Date.now() - t0) / 1000;
      const line = `${p.pass}: ${p.stage}${p.total > 1 ? ` ${p.done}/${p.total} (${pct}%)` : ''}  ${secs.toFixed(0)}s`;
      const stage = `${p.pass}:${p.stage}`;
      if (tty) process.stderr.write(`\r${line.padEnd(last.length)}`);
      else if (stage !== lastStage || pct >= lastPct + 10) {
        log(line);
        lastPct = stage !== lastStage ? 0 : pct - (pct % 10);
        lastStage = stage;
      }
      last = line;
    },
    end() {
      if (tty && last) process.stderr.write('\n');
    },
  };
}

/** Run `fn` on the compute page (analyze.html) with `files` served and progress on stderr. */
// `page` is a Playwright Page (the harness is untyped JS).
async function onComputePage<T>(files: Record<string, string>, fn: (page: any, base: string) => Promise<T>): Promise<T> {
  const { withBrowser, newPage } = await harness();
  return withBrowser({ build: 'auto', browser: a.browser, headed: a.headed, files }, async (_open: unknown, { browser, base }: { browser: unknown; base: string }) => {
    const page = await newPage(browser);
    const progress = progressPrinter();
    await page.exposeFunction('__videodocProgress', (x: Parameters<typeof progress.update>[0]) => progress.update(x));
    await page.goto(`${base}analyze.html`);
    await page.waitForFunction(() => window.videodoc, null, { timeout: 60_000 });
    try {
      return await fn(page, base);
    } finally {
      progress.end();
    }
  });
}

interface Ensured {
  doc: VideoDoc;
  docPath: string;
  plan: Plan;
  wallMs: number;
  adapter?: unknown;
  replaced?: string;
}

/** Bring the doc at `docPath` up to date with the `requested` layers: plan in Node, run decode
 *  passes in the browser (only if any), derive pure layers, save. */
async function ensure(video: string, docPath: string, requested: string[]): Promise<Ensured | { dryRun: object }> {
  const settings = parseSettings(a.set ?? []);
  const id = await fingerprintFile(video);
  let existing = loadDoc(docPath);
  let replaced: string | undefined;
  if (existing && existing.source.fingerprint !== id.fingerprint) {
    replaced = `${docPath} was for a different file (fingerprint changed); starting a new doc`;
    log(`note: ${replaced}`);
    existing = undefined;
  }
  const p = await plan(existing, requested, settings, a.force);
  log(`plan: ${p.layers.map((l) => `${l.name} (${l.status})`).join(', ')}`);
  const missing = missingModels(p);
  if (a['dry-run']) return { dryRun: { doc: docPath, plan: p.layers.map((l) => ({ name: l.name, status: l.status })), passes: p.passes, missing } };
  if (missing.length) throw new Error(missing.join('; '));

  let doc = existing;
  let adapter: unknown;
  const wall0 = Date.now();
  if (p.passes.audio || p.passes.frames) {
    // Cached layers that browser-computed ones depend on (e.g. speech for the transcript).
    const inputs: Partial<Layers> = {};
    for (const l of p.layers) {
      if (l.status !== 'compute' || PRODUCERS[l.name].pass === 'pure') continue;
      for (const dep of PRODUCERS[l.name].needs) {
        if (existing?.layers[dep] && p.layers.find((x) => x.name === dep)?.status === 'cached') {
          (inputs as Record<string, unknown>)[dep] = existing.layers[dep];
        }
      }
    }
    const result = await onComputePage<Awaited<ReturnType<Window['videodoc']['compute']>>>({ video }, (page, base) =>
      page.evaluate(
        ([url, layers, inp]: [string, typeof p.layers, Partial<Layers>]) => window.videodoc.compute(url, layers, inp),
        [`${base}@file/video`, p.layers, inputs],
      ),
    );
    adapter = result.adapter;
    doc ??= newDoc({ name: basename(video), ...id, ...result.probe });
    doc = withLayers({ ...doc, source: { ...doc.source, ...result.probe } }, result.layers);
  }
  if (!doc) throw new Error('nothing to compute and no doc');
  if (p.passes.pure) doc = withLayers(doc, deriveLayers(doc, p.layers));
  if (Object.keys(p.passes).length) saveDoc(docPath, doc);
  return { doc, docPath, plan: p, wallMs: Date.now() - wall0, adapter, replaced };
}

function videoArg(arg: string) {
  const video = userPath(arg);
  if (!existsSync(video) || !statSync(video).isFile()) throw new UsageError(`no such file: ${arg}`);
  return video;
}

const listArg = (s: string | undefined) => (s ?? '').split(',').map((x) => x.trim()).filter(Boolean);

/** One stderr line per planned layer: status, items, time, device, model, layer figures. */
function logLayers(doc: VideoDoc, p: Plan) {
  const summary = summarize(doc);
  const status = Object.fromEntries(p.layers.map((l) => [l.name, l.status]));
  log(`\n${doc.source.name}: ${fmtTime(doc.source.duration)}, ${doc.source.width}x${doc.source.height}, ${doc.source.fps} fps`);
  for (const [name, s] of Object.entries(summary)) {
    if (!(name in status)) continue;
    const extra = Object.entries(s).filter(([k]) => !['items', 'ms', 'device', 'configHash', 'model'].includes(k)).map(([k, v]) => `${k} ${v}`);
    log(`  ${name.padEnd(11)} ${status[name].padEnd(8)} ${String(s.items).padStart(6)} items  ${(s.ms / 1000).toFixed(1).padStart(7)} s  ${s.device}${s.model ? `  ${s.model}` : ''}${extra.length ? `  (${extra.join(', ')})` : ''}`);
  }
  return summary;
}

async function analyzeCmd(arg: string) {
  const video = videoArg(arg);
  const docPath = a.out ? userPath(a.out) : docPathFor(video);
  const r = await ensure(video, docPath, a.layers ? listArg(a.layers) : DEFAULT_LAYERS);
  if ('dryRun' in r) return out(r.dryRun);
  const summary = logLayers(r.doc, r.plan);
  log(`  -> ${docPath}`);
  out({
    doc: docPath,
    source: r.doc.source,
    layers: Object.fromEntries(r.plan.layers.map((l) => [l.name, { status: l.status, ...summary[l.name] }])),
    wallMs: r.wallMs,
    ...(r.adapter ? { adapter: r.adapter } : {}),
    ...(r.replaced ? { note: r.replaced } : {}),
  });
}

/** Keyframe JPEG path: in <video>.keyframes/, named by frame so re-selection reuses files. */
const imagePath = (video: string, k: KeyframeItem) => join(`${video}.keyframes`, `${String(k.frame).padStart(6, '0')}.jpg`);

/** Extract the keyframe JPEGs not on disk yet (one seek pass); returns how many were written. */
async function extractImages(video: string, keyframes: KeyframeItem[]): Promise<number> {
  const todo = keyframes.filter((k) => !existsSync(imagePath(video, k)));
  if (!todo.length) return 0;
  const size = Number(a['image-size']);
  if (!(size >= 16)) throw new UsageError(`--image-size takes a number of pixels, got '${a['image-size']}'`);
  log(`keyframe images: extracting ${todo.length} of ${keyframes.length}`);
  const jpegs: string[] = await onComputePage({ video }, (page, base) =>
    page.evaluate(
      ([url, times, s]: [string, number[], number]) => window.videodoc.images(url, times, s),
      [`${base}@file/video`, todo.map((k) => k.t), size],
    ),
  );
  mkdirSync(`${video}.keyframes`, { recursive: true });
  todo.forEach((k, i) => writeFileSync(imagePath(video, k), Buffer.from(jpegs[i], 'base64')));
  return todo.length;
}

async function markdownCmd(arg: string) {
  const video = videoArg(arg);
  const r = await ensure(video, docPathFor(video), [...MARKDOWN_LAYERS, ...listArg(a.layers)]);
  if ('dryRun' in r) return out(r.dryRun);
  const mdPath = a.out ? userPath(a.out) : undefined;
  const keyframes = r.doc.layers.keyframes?.items ?? [];
  const extracted = a.images ? await extractImages(video, keyframes) : 0;
  // Links are relative to the Markdown file, or absolute when it goes to stdout.
  const link = (p: string) => (mdPath ? relative(dirname(mdPath), p) : p).replace(/\\/g, '/');
  const md = toMarkdown(r.doc, { image: a.images ? (k) => link(imagePath(video, k)) : undefined });
  logLayers(r.doc, r.plan);
  if (a.images) log(`  images      ${keyframes.length} (${extracted} extracted) in ${video}.keyframes`);
  if (!mdPath) return void process.stdout.write(md);
  writeFileSync(mdPath, md);
  log(`  -> ${mdPath}`);
  out({ markdown: mdPath, doc: r.docPath, chars: md.length, keyframes: keyframes.length, ...(a.images && { images: `${video}.keyframes`, extracted }) });
}

function infoCmd(arg: string) {
  const path = userPath(arg.endsWith('.videodoc.json') ? arg : docPathFor(arg));
  const doc = loadDoc(path);
  if (!doc) throw new UsageError(`no doc at ${path} (run: videodoc analyze ${arg})`);
  const summary = summarize(doc);
  log(`${doc.source.name}: ${fmtTime(doc.source.duration)}, ${doc.source.width}x${doc.source.height}, ${doc.source.fps} fps`);
  for (const [name, s] of Object.entries(summary)) log(`  ${name.padEnd(11)} ${String(s.items).padStart(6)} items  ${s.model ?? ''}`);
  const missing = LAYER_NAMES.filter((n) => !doc.layers[n]);
  if (missing.length) log(`  not analyzed: ${missing.join(', ')}`);
  out({ doc: path, schema: doc.schema, source: doc.source, layers: summary, missing });
}

function layersCmd() {
  out(
    Object.fromEntries(
      LAYER_NAMES.map((n) => {
        const p = PRODUCERS[n];
        const settings = Object.fromEntries(Object.entries(p.defaults).map(([k, v]) => [k, { default: v, help: p.help[k] }]));
        return [n, { description: p.description, needs: p.needs, pass: p.pass, default: DEFAULT_LAYERS.includes(n), settings }];
      }),
    ),
  );
}

async function checkCmd() {
  const { withBrowser, newPage } = await harness();
  const problems: string[] = [];
  const models = Object.fromEntries(
    ['yunet_face_dyn.onnx', 'silero_vad.onnx', ...['whisper-base', 'whisper-small'].map((m) => `onnx-community/${m}`)].map((m) => [m, existsSync(join(ROOT, 'public/models', m))]),
  );
  const env = await withBrowser({ build: 'auto', browser: a.browser, headed: a.headed }, async (_o: unknown, { browser, base }: { browser: unknown; base: string }) => {
    const page = await newPage(browser);
    await page.goto(`${base}analyze.html`);
    await page.waitForFunction(() => window.videodoc, null, { timeout: 60_000 });
    return page.evaluate(() => window.videodoc.check());
  });
  if (!env.webgpu) problems.push('no WebGPU adapter: faces and scenes need it (try --browser chrome, or update GPU drivers)');
  if (!env.crossOriginIsolated) problems.push('page is not cross-origin isolated: wasm runs single-threaded');
  if (!env.decode.h264) problems.push('no H.264 decoder in this browser');
  for (const [m, ok] of Object.entries(models)) if (!ok) problems.push(`model not found: public/models/${m}`);
  const gpu = env.webgpu ? `${env.webgpu.vendor} ${env.webgpu.architecture}${env.webgpu.description ? ` (${env.webgpu.description})` : ''}` : 'none';
  log(`browser   ${a.browser}: ${env.userAgent.match(/(Edg|Chrome)\/[\d.]+/g)?.join(' ') ?? env.userAgent}`);
  log(`webgpu    ${gpu}${env.webgpu?.shaderF16 ? ', shader-f16' : ''}`);
  log(`decode    ${Object.entries(env.decode).map(([k, v]) => `${k} ${v ? 'yes' : 'no'}`).join(', ')}`);
  log(`models    ${Object.entries(models).map(([k, v]) => `${k.split('/').pop()} ${v ? 'yes' : 'no'}`).join(', ')}`);
  log(problems.length ? problems.map((p) => `problem   ${p}`).join('\n') : 'ok');
  out({ ok: !problems.length, browser: a.browser, ...env, models, problems });
  if (problems.length) process.exitCode = 1;
}

const fmtTime = (s: number) => {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
};

async function main() {
  const [cmd, ...rest] = positionals;
  if (a.help || !cmd) {
    console.log(HELP);
    if (!cmd && !a.help) process.exitCode = 2;
    return;
  }
  if (cmd === 'check') return checkCmd();
  if (cmd === 'layers') return layersCmd();
  const cmds: Record<string, (arg: string) => unknown> = { analyze: analyzeCmd, markdown: markdownCmd, info: infoCmd };
  if (cmds[cmd]) {
    if (rest.length !== 1) throw new UsageError(`usage: videodoc ${cmd} <video>`);
    return cmds[cmd](rest[0]);
  }
  throw new UsageError(`unknown command '${cmd}' (have: check, analyze, markdown, info, layers; see --help)`);
}

main().catch((e: unknown) => {
  const err = e as Error;
  const usage = e instanceof UsageError || e instanceof PlanError;
  log(`error: ${err.message}`);
  if (!usage && !(e instanceof DocError) && process.env.VIDEODOC_DEBUG) log(err.stack ?? '');
  out({ error: err.message, kind: usage ? 'usage' : 'failure' });
  process.exitCode = usage ? 2 : 1;
});
