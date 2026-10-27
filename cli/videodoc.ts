#!/usr/bin/env node
// videodoc CLI: analyze a video once into <video>.videodoc.json, then query the doc.
// Compute runs in a headless Edge/Chrome (WebGPU) driving analyze.html; planning, caching and
// reading docs run in Node, so commands with nothing to compute never start a browser.
//
// stdout: JSON results (one object per command). stderr: progress and human summaries.
// Exit codes: 0 ok, 1 failure, 2 usage error.
import { existsSync, openSync, readFileSync, readSync, closeSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import {
  DocError, docPathFor, fingerprint, formatDoc, LAYER_NAMES, newDoc, parseDoc, parseSettings, plan, PlanError, PRODUCERS, summarize, withLayers,
  type Layers, type Plan, type VideoDoc,
} from '../src/doc/index.ts';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const DEFAULT_LAYERS = ['speech', 'transcript', 'scenes'];
// User paths are relative to where the user ran the command (npm run moves cwd to the repo root).
const userPath = (p: string) => resolve(process.env.INIT_CWD ?? process.cwd(), p);

const HELP = `videodoc: analyze a video once, query it many times. Runs in a local headless browser (WebGPU).

Usage: videodoc <command> [options]

Commands:
  check                  What this machine can run: browser, WebGPU adapter, decoders, models.
  analyze <video>        Compute layers into <video>.videodoc.json. Layers already in the doc with
                         the same settings are reused; with nothing to compute, no browser starts.
  info <video|doc>       Summarize a saved doc (no browser).
  layers                 Layers, their dependencies and settings with defaults (JSON).

analyze options:
  --layers <list>        Comma-separated (default: ${DEFAULT_LAYERS.join(',')}); dependencies are added.
                         Available: ${LAYER_NAMES.join(', ')}.
  --set <layer.key=v>    Override a setting, repeatable (e.g. --set faces.every=4). See 'videodoc layers'.
  --out <path>           Doc path (default: <video>.videodoc.json).
  --force                Recompute the requested layers even if cached.
  --dry-run              Print the plan and exit.

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

async function analyzeCmd(videoArg: string) {
  const video = userPath(videoArg);
  if (!existsSync(video) || !statSync(video).isFile()) throw new UsageError(`no such file: ${videoArg}`);
  const docPath = a.out ? userPath(a.out) : docPathFor(video);
  const requested = (a.layers ?? DEFAULT_LAYERS.join(',')).split(',').map((s) => s.trim()).filter(Boolean);
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
  const planned = p.layers.map((l) => ({ name: l.name, status: l.status }));
  log(`plan: ${p.layers.map((l) => `${l.name} (${l.status})`).join(', ')}`);

  if (a['dry-run']) return out({ doc: docPath, plan: planned, passes: p.passes, missing: missingModels(p) });
  const missing = missingModels(p);
  if (missing.length) throw new Error(missing.join('; '));

  let doc = existing;
  let adapter: unknown;
  const wall0 = Date.now();
  if (Object.keys(p.passes).length) {
    const { withBrowser, newPage } = await harness();
    // Cached layers that computed ones depend on (e.g. speech for the transcript).
    const inputs: Partial<Layers> = {};
    for (const l of p.layers) {
      if (l.status !== 'compute') continue;
      for (const dep of PRODUCERS[l.name].needs) {
        if (existing?.layers[dep] && p.layers.find((x) => x.name === dep)?.status === 'cached') {
          (inputs as Record<string, unknown>)[dep] = existing.layers[dep];
        }
      }
    }
    const result = await withBrowser(
      { build: 'auto', browser: a.browser, headed: a.headed, files: { video } },
      async (_open: unknown, { browser, base }: { browser: unknown; base: string }) => {
        const page = await newPage(browser);
        const progress = progressPrinter();
        await page.exposeFunction('__videodocProgress', (x: Parameters<typeof progress.update>[0]) => progress.update(x));
        await page.goto(`${base}analyze.html`);
        await page.waitForFunction(() => window.videodoc, null, { timeout: 60_000 });
        try {
          return await page.evaluate(
            ([url, layers, inp]: [string, typeof p.layers, Partial<Layers>]) => window.videodoc.compute(url, layers, inp),
            [`${base}@file/video`, p.layers, inputs],
          );
        } finally {
          progress.end();
        }
      },
    );
    adapter = result.adapter;
    doc ??= newDoc({ name: basename(video), ...id, ...result.probe });
    doc = withLayers({ ...doc, source: { ...doc.source, ...result.probe } }, result.layers);
    saveDoc(docPath, doc);
  }
  if (!doc) throw new Error('nothing to compute and no doc');

  const summary = summarize(doc);
  const status = Object.fromEntries(p.layers.map((l) => [l.name, l.status]));
  log(`\n${doc.source.name}: ${fmtTime(doc.source.duration)}, ${doc.source.width}x${doc.source.height}, ${doc.source.fps} fps`);
  for (const [name, s] of Object.entries(summary)) {
    if (!(name in status)) continue;
    const extra = Object.entries(s).filter(([k]) => !['items', 'ms', 'device', 'configHash', 'model'].includes(k)).map(([k, v]) => `${k} ${v}`);
    log(`  ${name.padEnd(11)} ${status[name].padEnd(8)} ${String(s.items).padStart(6)} items  ${(s.ms / 1000).toFixed(1).padStart(7)} s  ${s.device}${s.model ? `  ${s.model}` : ''}${extra.length ? `  (${extra.join(', ')})` : ''}`);
  }
  log(`  -> ${docPath}`);
  out({
    doc: docPath,
    source: doc.source,
    layers: Object.fromEntries(p.layers.map((l) => [l.name, { status: l.status, ...summary[l.name] }])),
    wallMs: Date.now() - wall0,
    ...(adapter ? { adapter } : {}),
    ...(replaced ? { note: replaced } : {}),
  });
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
  if (cmd === 'analyze' || cmd === 'info') {
    if (rest.length !== 1) throw new UsageError(`usage: videodoc ${cmd} <video>`);
    return cmd === 'analyze' ? analyzeCmd(rest[0]) : infoCmd(rest[0]);
  }
  throw new UsageError(`unknown command '${cmd}' (have: check, analyze, info, layers; see --help)`);
}

main().catch((e: unknown) => {
  const err = e as Error;
  const usage = e instanceof UsageError || e instanceof PlanError;
  log(`error: ${err.message}`);
  if (!usage && !(e instanceof DocError) && process.env.VIDEODOC_DEBUG) log(err.stack ?? '');
  out({ error: err.message, kind: usage ? 'usage' : 'failure' });
  process.exitCode = usage ? 2 : 1;
});
