// The videodoc commands, shared by the CLI (cli/videodoc.ts) and the MCP server (cli/mcp.ts).
// Commands take explicit options, return JSON-able results and report through a Reporter;
// nothing here writes to stdout (under MCP, stdout is the protocol channel).
//
// Compute runs in a headless Edge/Chrome (WebGPU) driving analyze.html; planning, caching and
// reading docs run in Node, so commands with nothing to compute never touch a browser.
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, openSync, readdirSync, readFileSync, readSync, closeSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  deriveLayers, docPathFor, fingerprint, fmtTime, formatDoc, LAYER_NAMES, newDoc, parseDoc, parseSettings, plan, PRODUCERS, search as searchDocs,
  summarize, toMarkdown, withLayers,
  type KeyframeItem, type Layers, type Plan, type VideoDoc,
} from '../src/doc/index.ts';

export const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
export const DEFAULT_LAYERS = ['speech', 'transcript', 'scenes', 'keyframes'];
export const MARKDOWN_LAYERS = ['transcript', 'keyframes'];
export const DEFAULT_IMAGE_SIZE = 768;

/** A mistake in the request (exit code 2 in the CLI); the message says how to fix it. */
export class UsageError extends Error {}

export interface Progress { pass: string; stage: string; done: number; total: number }

export interface Reporter {
  /** A line for people (stderr in the CLI, a log message under MCP). */
  log(line: string): void;
  progress(p: Progress): void;
}

/** A Playwright Page (the harness is untyped JS). */
type Page = any;

/** Where compute runs: a fresh browser per command (CLI) or one kept warm (MCP). */
export interface Browser {
  /** Run `fn` on the compute page with `video` served at `${base}@file/video`. Progress from
   *  the page goes to `onProgress`; aborting `signal` stops the run (the page is discarded). */
  run<T>(video: string | undefined, fn: (page: Page, base: string) => Promise<T>, onProgress?: (p: Progress) => void, signal?: AbortSignal): Promise<T>;
  close(): Promise<void>;
}

export interface BrowserOptions { browser: string; headed: boolean }

async function harness() {
  // The harness builds and serves the pages from the repo root.
  process.chdir(ROOT);
  return import(new URL('../scripts/lib/harness.mjs', import.meta.url).href);
}

/** Build if needed and serve the app (app.html) until the process ends; resolves with its URL. */
export async function serveApp(port = 4317): Promise<string> {
  const { serve } = await harness();
  const { base } = await serve({ build: 'auto', port });
  return `${base}app.html`;
}

/** Open `url` in the system's default browser. */
export function openUrl(url: string) {
  const [cmd, args] =
    process.platform === 'win32' ? ['cmd', ['/c', 'start', '""', url]] : process.platform === 'darwin' ? ['open', [url]] : ['xdg-open', [url]];
  spawn(cmd as string, args as string[], { stdio: 'ignore', detached: true, windowsHide: true }).on('error', () => {}).unref();
}

/** A loaded compute page whose progress callback can be pointed at the current run. */
async function computePage(browser: unknown, base: string) {
  const { newPage } = await harness();
  const page = await newPage(browser);
  const state: { onProgress?: (p: Progress) => void } = {};
  await page.exposeFunction('__videodocProgress', (p: Progress) => state.onProgress?.(p));
  await page.goto(`${base}analyze.html`);
  await page.waitForFunction(() => window.videodoc, null, { timeout: 60_000 });
  return { page, state };
}

/** Run `fn` on `page`; an abort closes the page, which ends any work in flight. */
async function runOn<T>(page: Page, fn: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  signal?.throwIfAborted();
  const onAbort = () => void page.close().catch(() => {});
  signal?.addEventListener('abort', onAbort, { once: true });
  try {
    return await fn();
  } catch (e) {
    if (signal?.aborted) throw new Error('cancelled');
    throw e;
  } finally {
    signal?.removeEventListener('abort', onAbort);
  }
}

/** A new browser for every run, closed afterwards (the CLI: one command per process). */
export function oneShotBrowser(o: BrowserOptions): Browser {
  return {
    async run(video, fn, onProgress, signal) {
      const { openBrowser } = await harness();
      const b = await openBrowser({ build: 'auto', browser: o.browser, headed: o.headed, files: video ? { video } : {} });
      try {
        const { page, state } = await computePage(b.browser, b.base);
        state.onProgress = onProgress;
        return await runOn(page, () => fn(page, b.base), signal);
      } finally {
        await b.close();
      }
    },
    async close() {},
  };
}

/**
 * One browser and compute page kept between runs (the MCP server), so later calls skip the
 * build check, browser start and page load. Runs are serialized (one GPU). The browser closes
 * after `idleMs` without runs and restarts on demand; a cancelled run discards the page.
 */
export function warmBrowser(o: BrowserOptions, idleMs = 10 * 60_000): Browser {
  type Warm = { b: { browser: unknown; base: string; files: Record<string, string>; close(): Promise<void> }; page?: Awaited<ReturnType<typeof computePage>> };
  let warm: Promise<Warm> | undefined;
  let queue: Promise<unknown> = Promise.resolve();
  let idle: ReturnType<typeof setTimeout> | undefined;
  const shutdown = async () => {
    const w = warm;
    warm = undefined;
    if (w) await (await w.catch(() => undefined))?.b.close().catch(() => {});
  };
  const run = async <T>(video: string | undefined, fn: (page: Page, base: string) => Promise<T>, onProgress?: (p: Progress) => void, signal?: AbortSignal) => {
    clearTimeout(idle);
    warm ??= harness().then(async ({ openBrowser }) => ({ b: await openBrowser({ build: 'auto', browser: o.browser, headed: o.headed, files: {} }) }));
    let w: Warm;
    try {
      w = await warm;
    } catch (e) {
      warm = undefined;
      throw e;
    }
    if (!w.page || w.page.page.isClosed()) w.page = await computePage(w.b.browser, w.b.base);
    const { page, state } = w.page;
    if (video) w.b.files.video = video;
    else delete w.b.files.video;
    state.onProgress = onProgress;
    try {
      return await runOn(page, () => fn(page, w.b.base), signal);
    } finally {
      state.onProgress = undefined;
      idle = setTimeout(() => void shutdown(), idleMs);
      idle.unref();
    }
  };
  return {
    run(video, fn, onProgress, signal) {
      const next = queue.then(() => run(video, fn, onProgress, signal));
      queue = next.catch(() => {});
      return next;
    },
    async close() {
      clearTimeout(idle);
      await shutdown();
    },
  };
}

export interface Context {
  browser: Browser;
  report: Reporter;
  /** Relative paths resolve against this. */
  cwd: string;
  signal?: AbortSignal;
}

export interface AnalyzeOptions {
  /** Layers to bring up to date (dependencies are added). */
  layers?: string[];
  /** `layer.key=value` overrides. */
  set?: string[];
  force?: boolean;
  dryRun?: boolean;
  /** Doc path (default: <video>.videodoc.json). */
  doc?: string;
}

export interface MarkdownOptions extends Omit<AnalyzeOptions, 'doc'> {
  /** Write the Markdown here; image links are then relative to it, otherwise absolute. */
  out?: string;
  images?: boolean;
  imageSize?: number;
  /** Only [from, to): seconds, "mm:ss" or "h:mm:ss". */
  from?: string | number;
  to?: string | number;
}

/** Seconds from 83, "83.5", "1:23" or "1:01:23". */
export function parseTime(v: string | number, what = 'time'): number {
  if (typeof v === 'number') return v;
  const parts = v.trim().split(':');
  if (parts.length > 3 || parts.some((p) => !/^\d+(\.\d+)?$/.test(p))) throw new UsageError(`${what} takes seconds, mm:ss or h:mm:ss, got '${v}'`);
  return parts.reduce((s, p) => s * 60 + Number(p), 0);
}

const VIDEO_EXT = /\.(mp4|m4v|mov|mkv|webm)$/i;

/** Videos named by `paths`: video files as given, a doc file stands for its video, folders are
 *  searched recursively for video files (sorted, so results are stable). */
export function collect(ctx: Context, paths: string[]): string[] {
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const e of readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      if (e.isDirectory() && !e.name.endsWith('.keyframes') && !e.name.startsWith('.')) walk(join(dir, e.name));
      else if (e.isFile() && VIDEO_EXT.test(e.name)) out.push(join(dir, e.name));
    }
  };
  for (const arg of paths) {
    const p = userPath(ctx, arg);
    if (!existsSync(p)) throw new UsageError(`no such file or folder: ${p}`);
    if (statSync(p).isDirectory()) walk(p);
    else out.push(p.endsWith('.videodoc.json') ? p.slice(0, -'.videodoc.json'.length) : p);
  }
  return [...new Set(out)];
}

const userPath = (ctx: Context, p: string) => (isAbsolute(p) ? p : resolve(ctx.cwd, p));

function videoPath(ctx: Context, arg: string) {
  const video = userPath(ctx, arg);
  if (!existsSync(video) || !statSync(video).isFile()) throw new UsageError(`no such file: ${video}`);
  return video;
}

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
async function ensure(ctx: Context, video: string, docPath: string, requested: string[], o: AnalyzeOptions): Promise<Ensured | { dryRun: object }> {
  const settings = parseSettings(o.set ?? []);
  const id = await fingerprintFile(video);
  let existing = loadDoc(docPath);
  let replaced: string | undefined;
  if (existing && existing.source.fingerprint !== id.fingerprint) {
    replaced = `${docPath} was for a different file (fingerprint changed); starting a new doc`;
    ctx.report.log(`note: ${replaced}`);
    existing = undefined;
  }
  const p = await plan(existing, requested, settings, o.force);
  ctx.report.log(`plan: ${p.layers.map((l) => `${l.name} (${l.status})`).join(', ')}`);
  const missing = missingModels(p);
  if (o.dryRun) return { dryRun: { doc: docPath, plan: p.layers.map((l) => ({ name: l.name, status: l.status })), passes: p.passes, missing } };
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
    const result: Awaited<ReturnType<Window['videodoc']['compute']>> = await ctx.browser.run(
      video,
      (page, base) =>
        page.evaluate(
          ([url, layers, inp]: [string, typeof p.layers, Partial<Layers>]) => window.videodoc.compute(url, layers, inp),
          [`${base}@file/video`, p.layers, inputs],
        ),
      ctx.report.progress,
      ctx.signal,
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

export const fmtDuration = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

/** One line per planned layer: status, items, time, device, model, layer figures. */
function logLayers(ctx: Context, doc: VideoDoc, p: Plan) {
  const summary = summarize(doc);
  const status = Object.fromEntries(p.layers.map((l) => [l.name, l.status]));
  ctx.report.log(`\n${doc.source.name}: ${fmtDuration(doc.source.duration)}, ${doc.source.width}x${doc.source.height}, ${doc.source.fps} fps`);
  for (const [name, s] of Object.entries(summary)) {
    if (!(name in status)) continue;
    const extra = Object.entries(s).filter(([k]) => !['items', 'ms', 'device', 'configHash', 'model'].includes(k)).map(([k, v]) => `${k} ${v}`);
    ctx.report.log(`  ${name.padEnd(11)} ${status[name].padEnd(8)} ${String(s.items).padStart(6)} items  ${(s.ms / 1000).toFixed(1).padStart(7)} s  ${s.device}${s.model ? `  ${s.model}` : ''}${extra.length ? `  (${extra.join(', ')})` : ''}`);
  }
  return summary;
}

/** Analyze a video, or every video in a folder (one after another; a failure is reported and
 *  the rest continue). */
export async function analyze(ctx: Context, videoArg: string, o: AnalyzeOptions = {}) {
  const p = userPath(ctx, videoArg);
  if (existsSync(p) && statSync(p).isDirectory()) {
    if (o.doc) throw new UsageError('a doc path cannot be given when analyzing a folder');
    const videos = collect(ctx, [p]);
    if (!videos.length) throw new UsageError(`no video files (${VIDEO_EXT.source}) in ${p}`);
    const results: unknown[] = [];
    for (const [i, v] of videos.entries()) {
      ctx.signal?.throwIfAborted();
      ctx.report.log(`\n[${i + 1}/${videos.length}] ${v}`);
      try {
        results.push(await analyzeOne(ctx, v, o));
      } catch (e) {
        if (ctx.signal?.aborted) throw e;
        ctx.report.log(`error: ${(e as Error).message}`);
        results.push({ video: v, error: (e as Error).message });
      }
    }
    return { folder: p, videos: results };
  }
  return analyzeOne(ctx, videoArg, o);
}

async function analyzeOne(ctx: Context, videoArg: string, o: AnalyzeOptions) {
  const video = videoPath(ctx, videoArg);
  const docPath = o.doc ? userPath(ctx, o.doc) : docPathFor(video);
  const r = await ensure(ctx, video, docPath, o.layers?.length ? o.layers : DEFAULT_LAYERS, o);
  if ('dryRun' in r) return r.dryRun;
  const summary = logLayers(ctx, r.doc, r.plan);
  ctx.report.log(`  -> ${docPath}`);
  return {
    doc: docPath,
    source: r.doc.source,
    layers: Object.fromEntries(r.plan.layers.map((l) => [l.name, { status: l.status, ...summary[l.name] }])),
    wallMs: r.wallMs,
    ...(r.adapter ? { adapter: r.adapter } : {}),
    ...(r.replaced ? { note: r.replaced } : {}),
  };
}

/** Keyframe JPEG path: in <video>.keyframes/, named by frame so re-selection reuses files. */
export const imagePath = (video: string, k: KeyframeItem) => join(`${video}.keyframes`, `${String(k.frame).padStart(6, '0')}.jpg`);

/** Extract the keyframe JPEGs not on disk yet (one seek pass): how many were written, and the
 *  time spent seeking and encoding in the page (excludes browser start and build). */
async function extractImages(ctx: Context, video: string, keyframes: KeyframeItem[], size: number): Promise<{ extracted: number; ms: number }> {
  const todo = keyframes.filter((k) => !existsSync(imagePath(video, k)));
  if (!todo.length) return { extracted: 0, ms: 0 };
  if (!(size >= 16)) throw new UsageError(`image size takes a number of pixels >= 16, got '${size}'`);
  ctx.report.log(`keyframe images: extracting ${todo.length} of ${keyframes.length}`);
  let ms = 0;
  const jpegs: string[] = await ctx.browser.run(
    video,
    async (page, base) => {
      const t0 = Date.now();
      const r = await page.evaluate(
        ([url, times, s]: [string, number[], number]) => window.videodoc.images(url, times, s),
        [`${base}@file/video`, todo.map((k) => k.t), size],
      );
      ms = Date.now() - t0;
      return r;
    },
    ctx.report.progress,
    ctx.signal,
  );
  mkdirSync(`${video}.keyframes`, { recursive: true });
  todo.forEach((k, i) => writeFileSync(imagePath(video, k), Buffer.from(jpegs[i], 'base64')));
  return { extracted: todo.length, ms };
}

/** The Markdown view; analyzes missing layers first. Returns the text and a JSON summary. */
export async function markdown(ctx: Context, videoArg: string, o: MarkdownOptions = {}) {
  const video = videoPath(ctx, videoArg);
  const r = await ensure(ctx, video, docPathFor(video), [...MARKDOWN_LAYERS, ...(o.layers ?? [])], o);
  if ('dryRun' in r) return { summary: r.dryRun };
  const mdPath = o.out ? userPath(ctx, o.out) : undefined;
  const keyframes = r.doc.layers.keyframes?.items ?? [];
  const wall0 = Date.now();
  const { extracted, ms: imageMs } = o.images ? await extractImages(ctx, video, keyframes, o.imageSize ?? DEFAULT_IMAGE_SIZE) : { extracted: 0, ms: 0 };
  // Links are relative to the Markdown file, or absolute when there is none.
  const link = (p: string) => (mdPath ? relative(dirname(mdPath), p) : p).replace(/\\/g, '/');
  const from = o.from !== undefined ? parseTime(o.from, 'from') : undefined;
  const to = o.to !== undefined ? parseTime(o.to, 'to') : undefined;
  const text = toMarkdown(r.doc, { image: o.images ? (k) => link(imagePath(video, k)) : undefined, from, to });
  logLayers(ctx, r.doc, r.plan);
  if (o.images) ctx.report.log(`  images      ${keyframes.length} (${extracted} extracted${extracted ? ` in ${(imageMs / 1000).toFixed(1)} s` : ''}) in ${video}.keyframes`);
  if (mdPath) {
    writeFileSync(mdPath, text);
    ctx.report.log(`  -> ${mdPath}`);
  }
  return {
    text,
    summary: {
      ...(mdPath && { markdown: mdPath }),
      doc: r.docPath,
      chars: text.length,
      keyframes: keyframes.length,
      ...(o.images && { images: `${video}.keyframes`, extracted, imageMs }),
      wallMs: r.wallMs + Date.now() - wall0,
    },
    /** Keyframe image files in Markdown order (with images). */
    imageFiles: o.images ? keyframes.map((k) => imagePath(video, k)) : [],
  };
}

export interface SearchOptions {
  top?: number;
  /** Passage length, seconds. */
  window?: number;
}

/**
 * Keyword search over the transcripts of saved docs (no browser, no compute). Videos without a
 * doc or a transcript are listed so the caller can analyze them.
 */
export function search(ctx: Context, query: string, paths: string[], o: SearchOptions = {}) {
  if (!query.trim()) throw new UsageError('search needs a query');
  const videos = collect(ctx, paths);
  if (!videos.length) throw new UsageError(`no video files found in ${paths.join(', ')}`);
  const docs: VideoDoc[] = [];
  const found: string[] = [];
  const notAnalyzed: string[] = [];
  const noTranscript: string[] = [];
  for (const v of videos) {
    const doc = loadDoc(docPathFor(v));
    if (!doc) notAnalyzed.push(v);
    else if (!doc.layers.transcript) noTranscript.push(v);
    else {
      docs.push(doc);
      found.push(v);
    }
  }
  const t0 = performance.now();
  const hits = searchDocs(docs, query, o);
  const ms = Math.round(performance.now() - t0);
  const results = hits.map((h) => {
    const doc = docs[h.doc];
    const video = found[h.doc];
    const long = doc.source.duration >= 3600;
    const k = h.keyframe !== undefined ? doc.layers.keyframes!.items[h.keyframe] : undefined;
    const image = k && imagePath(video, k);
    return {
      video,
      time: `${fmtTime(h.t0, long)}–${fmtTime(h.t1, long)}`,
      t0: h.t0,
      t1: h.t1,
      at: h.at,
      score: h.score,
      snippet: h.snippet,
      ref: `transcript/${h.i0}`,
      ...(k && { keyframe: { ref: `keyframes/${h.keyframe}`, t: k.t, ...(image && existsSync(image) && { image }) } }),
    };
  });
  ctx.report.log(`"${query}": ${hits.length} hits in ${docs.length} transcripts (${ms} ms)`);
  for (const r of results) ctx.report.log(`  ${r.score.toFixed(2).padStart(6)}  ${basename(r.video)}  ${r.time}  ${r.snippet.length > 160 ? `${r.snippet.slice(0, 157)}...` : r.snippet}`);
  if (notAnalyzed.length) ctx.report.log(`  not analyzed (run videodoc analyze): ${notAnalyzed.join(', ')}`);
  if (noTranscript.length) ctx.report.log(`  no transcript layer: ${noTranscript.join(', ')}`);
  return { query, hits: results, searched: docs.length, ms, ...(notAnalyzed.length && { notAnalyzed }), ...(noTranscript.length && { noTranscript }) };
}

export function info(ctx: Context, arg: string) {
  const path = userPath(ctx, arg.endsWith('.videodoc.json') ? arg : docPathFor(arg));
  const doc = loadDoc(path);
  if (!doc) throw new UsageError(`no doc at ${path} (run: videodoc analyze ${arg})`);
  const summary = summarize(doc);
  ctx.report.log(`${doc.source.name}: ${fmtDuration(doc.source.duration)}, ${doc.source.width}x${doc.source.height}, ${doc.source.fps} fps`);
  for (const [name, s] of Object.entries(summary)) ctx.report.log(`  ${name.padEnd(11)} ${String(s.items).padStart(6)} items  ${s.model ?? ''}`);
  const missing = LAYER_NAMES.filter((n) => !doc.layers[n]);
  if (missing.length) ctx.report.log(`  not analyzed: ${missing.join(', ')}`);
  return { doc: path, schema: doc.schema, source: doc.source, layers: summary, missing };
}

/** Layers, their dependencies and settings with defaults (generated from PRODUCERS). */
export function layers() {
  return Object.fromEntries(
    LAYER_NAMES.map((n) => {
      const p = PRODUCERS[n];
      const settings = Object.fromEntries(Object.entries(p.defaults).map(([k, v]) => [k, { default: v, help: p.help[k] }]));
      return [n, { description: p.description, needs: p.needs, pass: p.pass, default: DEFAULT_LAYERS.includes(n), settings }];
    }),
  );
}

/** What this machine can run. `ok` is false when something a default layer needs is missing. */
export async function check(ctx: Context, browserName: string) {
  const problems: string[] = [];
  const models = Object.fromEntries(
    ['yunet_face_dyn.onnx', 'silero_vad.onnx', ...['whisper-base', 'whisper-small'].map((m) => `onnx-community/${m}`)].map((m) => [m, existsSync(join(ROOT, 'public/models', m))]),
  );
  const env: Awaited<ReturnType<Window['videodoc']['check']>> = await ctx.browser.run(undefined, (page) => page.evaluate(() => window.videodoc.check()), undefined, ctx.signal);
  if (!env.webgpu) problems.push('no WebGPU adapter: faces and scenes need it (try --browser chrome, or update GPU drivers)');
  if (!env.crossOriginIsolated) problems.push('page is not cross-origin isolated: wasm runs single-threaded');
  if (!env.decode.h264) problems.push('no H.264 decoder in this browser');
  for (const [m, ok] of Object.entries(models)) if (!ok) problems.push(`model not found: public/models/${m}`);
  const gpu = env.webgpu ? `${env.webgpu.vendor} ${env.webgpu.architecture}${env.webgpu.description ? ` (${env.webgpu.description})` : ''}` : 'none';
  ctx.report.log(`browser   ${browserName}: ${env.userAgent.match(/(Edg|Chrome)\/[\d.]+/g)?.join(' ') ?? env.userAgent}`);
  ctx.report.log(`webgpu    ${gpu}${env.webgpu?.shaderF16 ? ', shader-f16' : ''}`);
  ctx.report.log(`decode    ${Object.entries(env.decode).map(([k, v]) => `${k} ${v ? 'yes' : 'no'}`).join(', ')}`);
  ctx.report.log(`models    ${Object.entries(models).map(([k, v]) => `${k.split('/').pop()} ${v ? 'yes' : 'no'}`).join(', ')}`);
  ctx.report.log(problems.length ? problems.map((p) => `problem   ${p}`).join('\n') : 'ok');
  return { ok: !problems.length, browser: browserName, ...env, models, problems };
}
