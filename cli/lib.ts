// The videodoc commands, shared by the CLI (cli/videodoc.ts) and the MCP server (cli/mcp.ts).
// Commands take explicit options, return JSON-able results and report through a Reporter;
// nothing here writes to stdout (under MCP, stdout is the protocol channel).
//
// Compute runs in a headless Edge/Chrome (WebGPU) driving analyze.html; planning, caching and
// reading docs run in Node, so commands with nothing to compute never touch a browser.
import { spawn } from 'node:child_process';
import { closeSync, existsSync, mkdirSync, openSync, readdirSync, readFileSync, readSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  addSpeaker,
  checkEdits,
  type DocLayers,
  deriveLayers,
  docPathFor,
  findSpeaker,
  fingerprint,
  fmtTime,
  formatDoc,
  formatEdits,
  formatLog,
  isTrackLayer,
  type KeyframeItem,
  LAYER_NAME_RE,
  LAYER_NAMES,
  mergeOverrides,
  mergeSpeakers,
  moveSpeech,
  type NamedVoice,
  namedVoices,
  nameSpeaker,
  newDoc,
  newEdits,
  type Pipeline,
  type Plan,
  parseDoc,
  parseKeep,
  parsePipeline,
  parsePlan,
  parseSettings,
  pipelineContext,
  plan,
  planPathFor,
  REDACT_DEFAULTS,
  REDACT_KINDS,
  REDACT_SOUNDS,
  REDACT_STYLES,
  type RedactKind,
  type RedactOptions as RedactSettings,
  type RedactSound,
  type RedactStyle,
  redactionLog,
  type SpeakerEdits,
  type SpeakerItem,
  search as searchDocs,
  speakerName,
  speakersPathFor,
  suggestSpeakerNames,
  summarize,
  type Track,
  talkTime,
  toMarkdown,
  unmergeSpeaker,
  unmove,
  type VideoDoc,
  withLayers,
  withSpeakers,
} from '../src/doc/index.ts';

export const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
export const DEFAULT_LAYERS = ['speech', 'transcript', 'scenes', 'keyframes'];
export const MARKDOWN_LAYERS = ['transcript', 'keyframes'];
export const DEFAULT_IMAGE_SIZE = 768;

/** A mistake in the request (exit code 2 in the CLI); the message says how to fix it. */
export class UsageError extends Error {}

export interface Progress {
  pass: string;
  stage: string;
  done: number;
  total: number;
}

export interface Reporter {
  /** A line for people (stderr in the CLI, a log message under MCP). */
  log(line: string): void;
  progress(p: Progress): void;
}

/** A Playwright Page (the harness is untyped JS). */
type Page = any;

/** Where compute runs: a fresh browser per command (CLI) or one kept warm (MCP). */
export interface Browser {
  /** Run `fn` on the compute page with `files` (key -> path) served at `${base}@file/<key>`
   *  (keys starting with `out` are writable, see the harness). Progress from the page goes to
   *  `onProgress`; aborting `signal` stops the run (the page is discarded). */
  run<T>(files: Record<string, string>, fn: (page: Page, base: string) => Promise<T>, onProgress?: (p: Progress) => void, signal?: AbortSignal): Promise<T>;
  close(): Promise<void>;
}

export interface BrowserOptions {
  browser: string;
  headed: boolean;
}

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
  const [cmd, args] = process.platform === 'win32' ? ['cmd', ['/c', 'start', '""', url]] : process.platform === 'darwin' ? ['open', [url]] : ['xdg-open', [url]];
  spawn(cmd as string, args as string[], { stdio: 'ignore', detached: true, windowsHide: true })
    .on('error', () => {})
    .unref();
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
    async run(files, fn, onProgress, signal) {
      const { openBrowser } = await harness();
      const b = await openBrowser({ build: 'auto', browser: o.browser, headed: o.headed, files: { ...files }, handleSIGINT: false });
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
  const run = async <T>(files: Record<string, string>, fn: (page: Page, base: string) => Promise<T>, onProgress?: (p: Progress) => void, signal?: AbortSignal) => {
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
    for (const k of Object.keys(w.b.files)) delete w.b.files[k];
    Object.assign(w.b.files, files);
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
    run(files, fn, onProgress, signal) {
      const next = queue.then(() => run(files, fn, onProgress, signal));
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
  /** A pipeline preset (JSON, videodoc-pipeline/0): its layers (when `layers` is unset), settings
   *  (`set` wins), models and track layers. */
  pipeline?: string;
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

const PLATE_MODEL_URL = 'https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx';

const MODEL_FIX: Record<string, string> = {
  yunet: 'see README (YuNet, dynamic shape)',
  'yolo-plate': `curl -L -o public/models/yolo_plate_t640.onnx ${PLATE_MODEL_URL}`,
  rfdetr: 'the fp32 ONNX from huggingface.co/onnx-community/rfdetr_nano-ONNX, saved as public/models/rfdetr_nano.onnx',
};

const isRemote = (url: string) => /^https?:\/\//i.test(url);

/** A model URL's file on this machine: `/models/...` is under public/; anything else that is
 *  not http(s) is a path (a pipeline's are made absolute when it is loaded). */
function modelFile(url: string): string | undefined {
  if (isRemote(url)) return undefined;
  return url.startsWith('/models/') ? join(ROOT, 'public', url) : url;
}

/** A model file's identity for the planner (local files, and servers with range requests). */
async function modelFingerprint(url: string): Promise<string | undefined> {
  const file = modelFile(url);
  if (file) return existsSync(file) ? (await fingerprintFile(file)).fingerprint : undefined;
  try {
    const read = async (a: number, b: number) => {
      const r = await fetch(url, { headers: { Range: `bytes=${a}-${b - 1}` } });
      if (r.status !== 206) throw new Error('no range requests');
      return new Uint8Array(await r.arrayBuffer());
    };
    const head = await fetch(url, { headers: { Range: 'bytes=0-1' } });
    const size = Number(head.headers.get('Content-Range')?.split('/')[1]);
    await head.body?.cancel();
    return Number.isFinite(size) ? await fingerprint(size, read) : undefined;
  } catch {
    // Unreachable or no range requests: the URL stands in for the file.
    return undefined;
  }
}

/** Read a pipeline preset; its models' relative paths resolve against the preset's folder. */
export function loadPipeline(ctx: Context, arg: string): Pipeline {
  const path = userPath(ctx, arg);
  if (!existsSync(path)) throw new UsageError(`no pipeline at ${path}`);
  let p: Pipeline;
  try {
    p = parsePipeline(readFileSync(path, 'utf8'), path);
  } catch (e) {
    throw new UsageError((e as Error).message);
  }
  const local = (url: string) => !isRemote(url) && !url.startsWith('/models/');
  const models = Object.fromEntries(Object.entries(p.models ?? {}).map(([k, m]) => [k, local(m.url) ? { ...m, url: resolve(dirname(path), m.url) } : m]));
  return { ...p, models };
}

/** Model files each layer needs (dev layout: public/models), with how to get them. */
function missingModels(p: Plan): string[] {
  const need: [string, string][] = [];
  for (const l of p.layers) {
    if (l.status !== 'compute') continue;
    if (l.name === 'speech') need.push(['public/models/silero_vad.onnx', 'see README (Silero VAD v5)']);
    const file = l.model && modelFile(l.model.url);
    if (file) need.push([file, MODEL_FIX[l.model!.id] ?? `${l.name}.model is '${l.model!.id}'; check its url`]);
    if (l.name === 'speakers') {
      const { segmentation, embedding } = l.config as Record<string, string>;
      for (const m of [segmentation, embedding]) need.push([`public/models/${m}/onnx/model.onnx`, `node scripts/fetch-model.mjs ${m} model`]);
    }
    if (l.name === 'transcript') {
      const { model, encoder, decoder } = l.config as Record<string, string>;
      const suffix = (d: string) => (d === 'fp32' ? '' : d === 'q8' ? '_quantized' : `_${d}`);
      const files = [`encoder_model${suffix(encoder)}`, `decoder_model_merged${suffix(decoder)}`];
      for (const f of files) need.push([`public/models/${model}/onnx/${f}.onnx`, `node scripts/fetch-model.mjs ${model} ${files.join(' ')}`]);
    }
  }
  return need.filter(([f]) => !existsSync(isAbsolute(f) ? f : join(ROOT, f))).map(([f, fix]) => `missing ${f} (fix: ${fix})`);
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

/** A person's speaker edits for `video` (<video>.speakers.json), checked against `doc`; none
 *  when there is no file (or no speakers layer). A broken file is an error only when `strict`. */
function loadEdits(ctx: Context, video: string, doc: VideoDoc, strict = false): { edits?: SpeakerEdits; warnings: string[] } {
  const path = speakersPathFor(video);
  if (!doc.layers.speakers || !existsSync(path)) return { warnings: [] };
  try {
    const r = checkEdits(JSON.parse(readFileSync(path, 'utf8')), doc);
    for (const w of r.warnings) ctx.report.log(`  warning: ${path}: ${w}`);
    return r;
  } catch (e) {
    if (strict) throw new UsageError(`${path}: ${(e as Error).message} (fix or delete it, or pass --reset)`);
    ctx.report.log(`  warning: ${path}: ${(e as Error).message}; speaker names are not used`);
    return { warnings: [(e as Error).message] };
  }
}

/** `doc` with the speakers as a person edited them (names, merges, moves), for reading. Never saved. */
const edited = (ctx: Context, video: string, doc: VideoDoc) => withSpeakers(doc, loadEdits(ctx, video, doc).edits);

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
  const pc = pipelineContext(o.pipeline ? loadPipeline(ctx, o.pipeline) : undefined);
  const settings = mergeOverrides(pc.settings, parseSettings(o.set ?? [], pc.producers));
  const id = await fingerprintFile(video);
  let existing = loadDoc(docPath);
  let replaced: string | undefined;
  if (existing && existing.source.fingerprint !== id.fingerprint) {
    replaced = `${docPath} was for a different file (fingerprint changed); starting a new doc`;
    ctx.report.log(`note: ${replaced}`);
    existing = undefined;
  }
  // A pipeline's track layer already in the doc is used as saved when its pipeline is not given
  // (redact --cover vehicles after analyzing with the preset).
  const asSaved = requested.filter((n) => !(n in pc.producers) && existing && isTrackLayer(existing, n));
  if (asSaved.length) ctx.report.log(`note: ${asSaved.join(', ')} used as saved in the doc (pass its --pipeline to recompute)`);
  const p = await plan(
    existing,
    requested.filter((n) => !asSaved.includes(n)),
    settings,
    o.force,
    { ...pc, fingerprint: modelFingerprint },
  );
  ctx.report.log(`plan: ${p.layers.map((l) => `${l.name} (${l.status})`).join(', ')}`);
  const missing = missingModels(p);
  if (o.dryRun) return { dryRun: { doc: docPath, plan: p.layers.map((l) => ({ name: l.name, status: l.status })), passes: p.passes, missing } };
  if (missing.length) throw new Error(missing.join('; '));

  let doc = existing;
  let adapter: unknown;
  const wall0 = Date.now();
  if (p.passes.audio || p.passes.frames) {
    // Cached layers that browser-computed ones depend on (e.g. speech for the transcript).
    const inputs: DocLayers = {};
    for (const l of p.layers) {
      if (l.status !== 'compute' || pc.producers[l.name].pass === 'pure') continue;
      for (const dep of pc.producers[l.name].needs) {
        if (existing?.layers[dep] && p.layers.find((x) => x.name === dep)?.status === 'cached') {
          (inputs as Record<string, unknown>)[dep] = existing.layers[dep];
        }
      }
    }
    // Model files outside public/ are served to the page like the video.
    const files: Record<string, string> = { video };
    const served = new Map<string, string>();
    for (const l of p.layers) {
      const file = l.model && modelFile(l.model.url);
      if (!file || l.model!.url.startsWith('/models/') || served.has(l.model!.url)) continue;
      const key = `model-${served.size}`;
      served.set(l.model!.url, key);
      files[key] = file;
    }
    const result: Awaited<ReturnType<Window['videodoc']['compute']>> = await ctx.browser.run(
      files,
      (page, base) => {
        const layers = p.layers.map((l) => (l.model && served.has(l.model.url) ? { ...l, model: { ...l.model, url: `${base}@file/${served.get(l.model.url)}` } } : l));
        return page.evaluate(([url, layers, inp]: [string, typeof p.layers, DocLayers]) => window.videodoc.compute(url, layers, inp), [`${base}@file/video`, layers, inputs]);
      },
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
    const extra = Object.entries(s)
      .filter(([k]) => !['items', 'ms', 'device', 'configHash', 'model'].includes(k))
      .map(([k, v]) => `${k} ${v}`);
    ctx.report.log(
      `  ${name.padEnd(11)} ${status[name].padEnd(8)} ${String(s.items).padStart(6)} items  ${(s.ms / 1000).toFixed(1).padStart(7)} s  ${s.device}${s.model ? `  ${s.model}` : ''}${extra.length ? `  (${extra.join(', ')})` : ''}`,
    );
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
  const requested = o.layers?.length ? o.layers : o.pipeline ? loadPipeline(ctx, o.pipeline).layers : DEFAULT_LAYERS;
  const r = await ensure(ctx, video, docPath, requested, o);
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
    { video },
    async (page, base) => {
      const t0 = Date.now();
      const r = await page.evaluate(([url, times, s]: [string, number[], number]) => window.videodoc.images(url, times, s), [`${base}@file/video`, todo.map((k) => k.t), size]);
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
  const text = toMarkdown(edited(ctx, video, r.doc), { image: o.images ? (k) => link(imagePath(video, k)) : undefined, from, to });
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

export interface RedactOptions extends Omit<AnalyzeOptions, 'layers' | 'doc'> {
  /** Output MP4 (default: <video name>.redacted.mp4 next to the video). */
  out?: string;
  style?: string;
  margin?: number;
  hold?: number;
  /** Face track ids to leave visible: "3,7" or [3, 7]. Replaces the review plan's. */
  keep?: string | number[];
  /** Plate track ids to leave visible, likewise. */
  keepPlates?: string | number[];
  /** A pipeline's track layers: ids to leave visible, by layer ({ vehicles: [3] }, or
   *  "vehicles=3,7" strings). Replaces the review plan's for the layers given. */
  keepTracks?: Record<string, string | number[]> | string[];
  /** Layers to cover: "faces,plates" (default), "faces", or "none" (audio only). */
  cover?: string | string[];
  /** Words and phrases to bleep: "Okafor,Daniel Reyes" or a list. Replaces the review plan's. */
  words?: string | string[];
  /** A file of words and phrases to bleep, one per line (# starts a comment); added to `words`. */
  wordsFile?: string;
  /** Also bleep the built-in strong-language list. */
  profanity?: boolean;
  sound?: string;
  lead?: number;
  tail?: number;
  /** Matches to leave audible, by first-word key ("12.3"): "12.3,40.0" or a list. */
  keepWords?: string | string[];
  /** Speakers to bleep every turn of: names (given with `speakers`) or "Speaker 2", as
   *  "Dana,2" or a list. Replaces the review plan's. */
  speakers?: string | string[];
  /** false: ignore <video>.redact.json (the app's review). */
  plan?: boolean;
}

const list = (v: string | string[] | undefined) => (v === undefined ? undefined : (Array.isArray(v) ? v : v.split(',')).map((x) => x.trim()).filter(Boolean));

/** Check and complete redaction settings (the unset ones from `base`: a review plan's, or the defaults). */
export function redactSettings(o: RedactOptions, base: RedactSettings = REDACT_DEFAULTS): RedactSettings {
  const style = (o.style ?? base.style) as RedactStyle;
  if (!REDACT_STYLES.includes(style)) throw new UsageError(`style takes ${REDACT_STYLES.join(', ')}, got '${o.style}'`);
  const num = (v: number | undefined, d: number, what: string, max: number) => {
    const n = v ?? d;
    if (!(Number.isFinite(n) && n >= 0 && n <= max)) throw new UsageError(`${what} takes a number from 0 to ${max}, got '${v}'`);
    return n;
  };
  let keep: number[];
  let keepPlates: number[];
  const keepTracks = { ...base.keepTracks };
  try {
    keep = o.keep === undefined ? base.keep : parseKeep(o.keep);
    keepPlates = o.keepPlates === undefined ? base.keepPlates : parseKeep(o.keepPlates);
    const pairs = Array.isArray(o.keepTracks)
      ? o.keepTracks.map((x) => {
          const m = /^(\w+)=(.*)$/.exec(x);
          if (!m) throw new Error(`keep-tracks takes layer=ids, e.g. vehicles=3,7, got '${x}'`);
          return [m[1], m[2]] as const;
        })
      : Object.entries(o.keepTracks ?? {});
    for (const [k, ids] of pairs) keepTracks[k] = parseKeep(ids);
  } catch (e) {
    throw new UsageError((e as Error).message);
  }
  let cover = list(o.cover) ?? base.cover;
  if (cover.length === 1 && cover[0] === 'none') cover = [];
  const bad = cover.find((k) => !LAYER_NAME_RE.test(k) || ((LAYER_NAMES as string[]).includes(k) && !REDACT_KINDS.includes(k)));
  if (bad !== undefined) throw new UsageError(`cover takes track layers (faces, plates or a pipeline's), comma-separated, or none, got '${cover.join(',')}'`);
  const sound = (o.sound ?? base.sound) as RedactSound;
  if (!REDACT_SOUNDS.includes(sound)) throw new UsageError(`sound takes ${REDACT_SOUNDS.join(', ')}, got '${o.sound}'`);
  let words = list(o.words) ?? base.words;
  if (o.wordsFile) {
    if (!existsSync(o.wordsFile)) throw new UsageError(`no words file at ${o.wordsFile}`);
    const lines = readFileSync(o.wordsFile, 'utf8')
      .split(/\r?\n/)
      .map((l) => l.replace(/#.*/, '').trim())
      .filter(Boolean);
    words = [...(o.words === undefined ? [] : words), ...lines];
  }
  const keepWords = list(o.keepWords) ?? base.keepWords;
  const badKey = keepWords.find((k) => !/^\d+\.\d+$/.test(k));
  if (badKey !== undefined) throw new UsageError(`keepWords takes word keys like 12.3 (from the redaction log), got '${badKey}'`);
  return {
    style,
    margin: num(o.margin, base.margin, 'margin', 2),
    hold: num(o.hold, base.hold, 'hold', 10),
    keep,
    keepPlates,
    keepTracks,
    cover: cover as RedactKind[],
    words,
    profanity: o.profanity ?? base.profanity,
    sound,
    lead: num(o.lead, base.lead, 'lead', 5),
    tail: num(o.tail, base.tail, 'tail', 5),
    keepWords,
    speakers: list(o.speakers) ?? base.speakers ?? [],
  };
}

/**
 * Blur (or pixelate, or fill) every face and plate, and bleep (or mute) listed words: analyzes
 * what is needed first, writes the redacted MP4 and a redaction log, <out>.redactions.json,
 * listing every track and bleeped word and which were kept. A review saved by the app
 * (<video>.redact.json: kept faces, faces added by hand, options) is used when present; options
 * given here override it.
 */
export async function redact(ctx: Context, videoArg: string, o: RedactOptions = {}) {
  const video = videoPath(ctx, videoArg);
  if (o.wordsFile) o = { ...o, wordsFile: userPath(ctx, o.wordsFile) };
  let settings = redactSettings(o);
  const out = o.out ? userPath(ctx, o.out) : join(dirname(video), `${basename(video).replace(/\.[^.]+$/, '')}.redacted.mp4`);
  if (resolve(out) === resolve(video)) throw new UsageError('the output would overwrite the video; pass another --out');
  if (!/\.mp4$/i.test(out)) throw new UsageError(`the output is MP4; name it .mp4, got ${out}`);
  // What the review covers applies where no option overrides it, so read it before deciding layers.
  const planPath = planPathFor(video);
  const usePlan = o.plan !== false && existsSync(planPath);
  let need = settings;
  let addedAudio: [number, number][] = [];
  if (usePlan) {
    try {
      const p = JSON.parse(readFileSync(planPath, 'utf8')) as { options?: Partial<RedactSettings>; addedAudio?: [number, number][] };
      need = redactSettings(o, { ...REDACT_DEFAULTS, ...p.options, cover: (p.options?.cover ?? REDACT_DEFAULTS.cover).filter((k) => LAYER_NAME_RE.test(k)) });
      addedAudio = Array.isArray(p.addedAudio) ? p.addedAudio : [];
    } catch {
      // parsePlan below reports it.
    }
  }
  const layers = [...need.cover, ...(need.words.length || need.profanity ? ['transcript'] : []), ...(need.speakers.length ? ['speakers'] : [])];
  if (!layers.length && !addedAudio.length)
    throw new UsageError('nothing to redact: cover is none and there are no words or speakers to bleep (pass --words, --words-file, --profanity or --speakers)');
  const r = await ensure(ctx, video, docPathFor(video), layers, o);
  if ('dryRun' in r) return r.dryRun;
  // Speakers by the names a person gave them.
  const doc = edited(ctx, video, r.doc);
  let added: Track[] = [];
  const warnings: string[] = [];
  if (usePlan) {
    let p: ReturnType<typeof parsePlan>;
    try {
      p = parsePlan(readFileSync(planPath, 'utf8'), doc);
    } catch (e) {
      throw new UsageError(`${planPath}: ${(e as Error).message} (delete it, or pass --no-plan)`);
    }
    settings = redactSettings(o, p.plan.options);
    added = p.plan.added;
    addedAudio = p.plan.addedAudio ?? [];
    warnings.push(...p.warnings);
    ctx.report.log(`review: ${planPath}`);
    for (const w of p.warnings) ctx.report.log(`  warning: ${w}`);
  }
  let pre: ReturnType<typeof redactionLog>;
  try {
    pre = redactionLog(doc, settings, undefined, added, addedAudio);
  } catch (e) {
    throw new UsageError((e as Error).message);
  }
  const t = pre.totals;
  const n = (k: RedactKind) => pre.tracks.filter((x) => x.layer === k && !x.kept).length;
  const coversVideo = settings.cover.length > 0 || added.length > 0;
  if (coversVideo) {
    const kinds = [...new Set(pre.tracks.map((x) => x.layer))];
    const what = `${(kinds.length ? kinds : ['faces']).map((k) => `${n(k)} ${k === 'faces' ? 'face' : k === 'plates' ? 'plate' : k}`).join(' and ')} tracks`;
    ctx.report.log(
      `redacting ${what}${t.added ? ` (${t.added} added by hand)` : ''}${t.kept ? `, keeping ${t.kept} visible,` : ''} on ${t.framesWithRegions} of ${doc.source.frames} frames (${settings.style})`,
    );
  }
  if (pre.audio) {
    const a = pre.audio.totals;
    const what = [
      `${a.matches - a.kept} spoken matches`,
      ...(a.turns ? [`${a.turns} turns of ${settings.speakers.join(', ')}`] : []),
      ...(a.added ? [`${a.added} spans added by hand`] : []),
    ];
    ctx.report.log(`${settings.sound === 'bleep' ? 'bleeping' : 'muting'} ${what.join(', ')}${a.kept ? `, leaving ${a.kept} audible` : ''}: ${a.seconds} s of audio`);
    const long = doc.source.duration >= 3600;
    for (const s of pre.audio.spans.filter((x) => !x.kept).slice(0, 20))
      ctx.report.log(
        `  ${fmtTime(s.t0, long)}  ${s.added ? '(added by hand)' : s.speaker ? `(${s.speaker} speaking, to ${fmtTime(s.t1, long)})` : `"${s.text}"${s.term === s.text ? '' : ` (${s.term})`}`}`,
      );
    if (pre.audio.spans.length > 20) ctx.report.log(`  ... (all ${pre.audio.spans.length} in the log)`);
    if (a.matches === 0 && !a.added && !a.turns && settings.words.length)
      ctx.report.log('  note: no listed word is in the transcript (check its spelling there: names are often transcribed differently)');
  }
  const part = `${out}.part`;
  writeFileSync(part, '');
  let result: { frames: number; ms: number };
  try {
    // The page gets the doc without the layers it does not need (a transcript can be large).
    const sent = {
      ...doc,
      layers: {
        faces: doc.layers.faces,
        plates: doc.layers.plates,
        scenes: doc.layers.scenes,
        ...(pre.audio && { transcript: doc.layers.transcript, speakers: doc.layers.speakers }),
      },
    };
    result = await ctx.browser.run(
      { video, out: part },
      (page, base) =>
        page.evaluate(
          ([url, outUrl, d, s]: [string, string, VideoDoc, RedactSettings & { added: Track[]; addedAudio: [number, number][] }]) => window.videodoc.redact(url, outUrl, d, s),
          [`${base}@file/video`, `${base}@file/out`, sent, { ...settings, added, addedAudio }],
        ),
      ctx.report.progress,
      ctx.signal,
    );
  } catch (e) {
    rmSync(part, { force: true });
    throw e;
  }
  if (coversVideo && result.frames !== doc.source.frames) {
    rmSync(part, { force: true });
    throw new Error(`decoded ${result.frames} frames but the doc has ${doc.source.frames}; the regions would not line up (re-analyze with --force)`);
  }
  renameSync(part, out);
  const log = redactionLog(doc, settings, { name: basename(out), frames: result.frames, ms: result.ms }, added, addedAudio);
  const logPath = `${out}.redactions.json`;
  writeFileSync(logPath, formatLog(log));
  if (coversVideo) ctx.report.log(`  ${result.frames} frames in ${(result.ms / 1000).toFixed(1)} s (${((result.frames * 1000) / result.ms).toFixed(0)} fps)`);
  else ctx.report.log(`  audio in ${(result.ms / 1000).toFixed(1)} s (video copied)`);
  ctx.report.log(`  -> ${out}\n  -> ${logPath}`);
  ctx.report.log(`  note: ${log.caveat}`);
  return {
    out,
    log: logPath,
    doc: r.docPath,
    ...(usePlan && { review: planPath, warnings }),
    settings,
    totals: log.totals,
    ...(log.audio && { audio: log.audio.totals }),
    frames: result.frames,
    ms: result.ms,
    wallMs: r.wallMs + result.ms,
    caveat: log.caveat,
  };
}

export interface SearchOptions {
  top?: number;
  /** Passage length, seconds. */
  window?: number;
  /** Only what this speaker says: a name given with `speakers`, or "Speaker 2". */
  speaker?: string;
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
      docs.push(edited(ctx, v, doc));
      found.push(v);
    }
  }
  const noSpeaker = o.speaker === undefined ? [] : found.filter((_, i) => !docs[i].layers.speakers || !findSpeaker(docs[i].layers.speakers!.items, o.speaker!));
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
      ...(h.speaker !== undefined && { speaker: h.speaker }),
      ref: `transcript/${h.i0}`,
      ...(k && { keyframe: { ref: `keyframes/${h.keyframe}`, t: k.t, ...(image && existsSync(image) && { image }) } }),
    };
  });
  ctx.report.log(`"${query}"${o.speaker !== undefined ? ` said by ${o.speaker}` : ''}: ${hits.length} hits in ${docs.length - noSpeaker.length} transcripts (${ms} ms)`);
  for (const r of results)
    ctx.report.log(`  ${r.score.toFixed(2).padStart(6)}  ${basename(r.video)}  ${r.time}  ${r.snippet.length > 160 ? `${r.snippet.slice(0, 157)}...` : r.snippet}`);
  if (notAnalyzed.length) ctx.report.log(`  not analyzed (run videodoc analyze): ${notAnalyzed.join(', ')}`);
  if (noTranscript.length) ctx.report.log(`  no transcript layer: ${noTranscript.join(', ')}`);
  if (noSpeaker.length) ctx.report.log(`  no speaker ${o.speaker} (or no speakers layer): ${noSpeaker.join(', ')}`);
  return {
    query,
    ...(o.speaker !== undefined && { speaker: o.speaker }),
    hits: results,
    searched: docs.length - noSpeaker.length,
    ms,
    ...(notAnalyzed.length && { notAnalyzed }),
    ...(noTranscript.length && { noTranscript }),
    ...(noSpeaker.length && { noSpeaker }),
  };
}

export function info(ctx: Context, arg: string) {
  const path = userPath(ctx, arg.endsWith('.videodoc.json') ? arg : docPathFor(arg));
  const saved = loadDoc(path);
  if (!saved) throw new UsageError(`no doc at ${path} (run: videodoc analyze ${arg})`);
  const doc = edited(ctx, path.slice(0, -'.videodoc.json'.length), saved);
  const summary = summarize(doc);
  ctx.report.log(`${doc.source.name}: ${fmtDuration(doc.source.duration)}, ${doc.source.width}x${doc.source.height}, ${doc.source.fps} fps`);
  for (const [name, s] of Object.entries(summary)) ctx.report.log(`  ${name.padEnd(11)} ${String(s.items).padStart(6)} items  ${s.model ?? ''}`);
  const missing = LAYER_NAMES.filter((n) => !doc.layers[n]);
  if (missing.length) ctx.report.log(`  not analyzed: ${missing.join(', ')}`);
  return { doc: path, schema: doc.schema, source: doc.source, layers: summary, missing };
}

export interface SpeakersOptions {
  /** Name speakers: "2=Dana" (an empty name removes one), or { "2": "Dana" }. */
  name?: string[] | Record<string, string>;
  /** One person found as two: "3=Dana" (speaker 3 is Dana), or { "3": "Dana" }. After names. */
  merge?: string[] | Record<string, string>;
  /** Speakers to stand alone again (undo a merge), by id. */
  unmerge?: (string | number)[];
  /** Speech given to the wrong speaker: "1:02-1:05=Dana" (all speech then is Dana's) or
   *  "Speaker 2@1:02-1:05=Dana" (only Speaker 2's), or { t0, t1, to, from? }. A `to` that is no
   *  speaker's name adds a speaker (one the analysis missed). */
  move?: (string | { t0: number | string; t1: number | string; to: string; from?: string })[];
  /** Moves to undo, by index (as listed). */
  unmove?: (string | number)[];
  /** Folders (searched recursively) whose speakers files name voices, for suggestions; the
   *  video's own folder is always searched. */
  voices?: string[];
  /** Delete <video>.speakers.json first (all names, merges and moves). */
  reset?: boolean;
}

const pairs = (v: string[] | Record<string, string> | undefined, what: string): [string, string][] =>
  Array.isArray(v)
    ? v.map((x) => {
        const i = x.indexOf('=');
        if (i < 1) throw new UsageError(`${what} takes speaker=name pairs, e.g. 2=Dana, got '${x}'`);
        return [x.slice(0, i).trim(), x.slice(i + 1).trim()];
      })
    : Object.entries(v ?? {});

/** Named voices in the speakers files under `dirs` (recursively) and next to `video`, other than its own. */
function knownVoices(ctx: Context, video: string, dirs: string[]): NamedVoice[] {
  const files = new Set<string>();
  const walk = (dir: string, deep: boolean) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      if (deep && e.isDirectory() && !e.name.endsWith('.keyframes') && !e.name.startsWith('.')) walk(join(dir, e.name), true);
      else if (e.isFile() && e.name.endsWith('.speakers.json')) files.add(join(dir, e.name));
    }
  };
  walk(dirname(video), false);
  for (const d of dirs) {
    const p = userPath(ctx, d);
    if (!existsSync(p) || !statSync(p).isDirectory()) throw new UsageError(`voices takes folders, got ${p}`);
    walk(p, true);
  }
  files.delete(speakersPathFor(video));
  return [...files].flatMap((f) => {
    try {
      return namedVoices(JSON.parse(readFileSync(f, 'utf8')) as SpeakerEdits, f.slice(0, -'.speakers.json'.length));
    } catch {
      ctx.report.log(`  warning: ${f} is not a speakers file; skipped`);
      return [];
    }
  });
}

/**
 * The speakers of a video, and a person's edits to them, saved as <video>.speakers.json: names,
 * merges (one person found as two) and moves (speech given to the wrong speaker). Names follow
 * the voices when the speakers are found again. Suggests names for unnamed speakers from voices
 * named in other videos (never applies them). Needs the speakers layer; no browser.
 */
export function speakers(ctx: Context, videoArg: string, o: SpeakersOptions = {}) {
  const video = videoPath(ctx, videoArg);
  const docPath = docPathFor(video);
  const doc = loadDoc(docPath);
  if (!doc?.layers.speakers) throw new UsageError(`no speakers layer for ${video} (run: videodoc analyze ${videoArg} --layers speakers)`);
  const path = speakersPathFor(video);
  const items = doc.layers.speakers.items;
  if (o.reset && existsSync(path)) {
    rmSync(path);
    ctx.report.log(`deleted ${path}`);
  }
  const loaded = loadEdits(ctx, video, doc, true);
  let edits = loaded.edits ?? newEdits(doc);
  const before = formatEdits(edits);
  const now = () => withSpeakers(doc, edits).layers.speakers!.items;
  const find = (ref: string, list = now()): SpeakerItem => {
    const s = findSpeaker(list, ref);
    if (!s) throw new UsageError(`no speaker '${ref}' (the speakers are ${list.map((x) => speakerName(x)).join(', ')})`);
    return s;
  };
  try {
    for (const i of (o.unmove ?? []).map(Number).sort((a, b) => b - a)) edits = unmove(edits, i);
    for (const id of o.unmerge ?? []) edits = unmergeSpeaker(edits, Number(String(id).replace(/\D/g, '')));
    for (const [a, b] of pairs(o.name, 'name')) edits = nameSpeaker(edits, items, find(a).id, b);
    for (const [a, b] of pairs(o.merge, 'merge')) edits = mergeSpeakers(edits, items, find(a).id, find(b).id);
    for (const m of o.move ?? []) {
      let t0: number, t1: number, to: string, from: string | undefined;
      if (typeof m === 'string') {
        const x = /^(?:(.+)@)?\s*([\d:.]+)\s*-\s*([\d:.]+)\s*=(.+)$/.exec(m);
        if (!x) throw new UsageError(`move takes t0-t1=speaker or speaker@t0-t1=speaker (times in seconds, mm:ss or h:mm:ss), e.g. 1:02-1:05=Dana, got '${m}'`);
        [from, t0, t1, to] = [x[1]?.trim(), parseTime(x[2], 'move'), parseTime(x[3], 'move'), x[4].trim()];
      } else [t0, t1, to, from] = [parseTime(m.t0, 'move t0'), parseTime(m.t1, 'move t1'), m.to, m.from];
      let target = findSpeaker(now(), to)?.id;
      if (target === undefined) {
        if (/^(speaker\s*)?\d+$/i.test(to)) throw new UsageError(`no speaker '${to}'`);
        ({ edits, id: target } = addSpeaker(edits, items, to));
        ctx.report.log(`  added a speaker: ${to}`);
      }
      edits = moveSpeech(edits, items, t0, t1, target, from === undefined ? undefined : find(from).id);
    }
  } catch (e) {
    throw e instanceof UsageError ? e : new UsageError((e as Error).message);
  }
  const changed = formatEdits(edits) !== before;
  if (changed) {
    if (edits.speakers.length || edits.moves.length) writeFileSync(path, formatEdits(edits));
    else rmSync(path, { force: true });
  }
  const list = now();
  const suggestions = suggestSpeakerNames(list, knownVoices(ctx, video, o.voices ?? []));
  const total = list.reduce((n, s) => n + talkTime(s), 0);
  const long = doc.source.duration >= 3600;
  const r1 = (v: number) => Math.round(v * 10) / 10;
  const rows = list.map((s) => {
    const sug = suggestions.find((x) => x.id === s.id);
    return {
      id: s.id,
      name: speakerName(s),
      named: !!s.name,
      seconds: r1(talkTime(s)),
      share: r1((100 * talkTime(s)) / Math.max(1e-9, total)),
      turns: s.turns.length,
      first: fmtTime(s.turns[0][0], long),
      ...(s.merged && { merged: s.merged }),
      ...(sug && { suggestion: { name: sug.name, distance: sug.distance, from: sug.video } }),
    };
  });
  ctx.report.log(`${doc.source.name}: ${list.length} speakers${changed ? ` (saved ${existsSync(path) ? path : 'no edits'})` : ''}`);
  for (const r of rows)
    ctx.report.log(
      `  ${String(r.id).padStart(3)}  ${r.name.padEnd(20)} ${String(r.seconds).padStart(7)} s  ${String(r.share).padStart(5)}%  from ${r.first}${r.merged ? `  (merged: ${r.merged.join(', ')})` : ''}${r.suggestion ? `  maybe ${r.suggestion.name}? (voice ${r.suggestion.distance} from ${basename(r.suggestion.from)})` : ''}`,
    );
  // A speaker merged since a move names the one it joined.
  const id2name = (id: number) => speakerName(list.find((s) => s.id === id || s.merged?.includes(id)) ?? { id });
  edits.moves.forEach((m, i) =>
    ctx.report.log(`  move ${i}: ${fmtTime(m.t0, long)}-${fmtTime(m.t1, long)} ${m.from !== undefined ? `of ${id2name(m.from)} ` : ''}to ${id2name(m.to)}`),
  );
  return {
    video,
    doc: docPath,
    ...(existsSync(path) && { edits: path }),
    changed,
    speakers: rows,
    moves: edits.moves.map((m, index) => ({ index, t0: m.t0, t1: m.t1, ...(m.from !== undefined && { from: id2name(m.from) }), to: id2name(m.to) })),
    ...(loaded.warnings.length && { warnings: loaded.warnings }),
  };
}

/** Layers, their dependencies and settings with defaults (generated from PRODUCERS; with a
 *  pipeline, its track layers too). */
export function layers(ctx?: Context, pipeline?: string) {
  const { producers } = pipelineContext(ctx && pipeline ? loadPipeline(ctx, pipeline) : undefined);
  return Object.fromEntries(
    Object.keys(producers).map((n) => {
      const p = producers[n];
      const settings = Object.fromEntries(Object.entries(p.defaults).map(([k, v]) => [k, { default: v, help: p.help[k] }]));
      return [n, { description: p.description, needs: p.needs, pass: p.pass, default: DEFAULT_LAYERS.includes(n), settings }];
    }),
  );
}

/** What this machine can run. `ok` is false when something a default layer needs is missing. */
export async function check(ctx: Context, browserName: string) {
  const problems: string[] = [];
  const models = Object.fromEntries(
    ['yunet_face_dyn.onnx', 'silero_vad.onnx', 'yolo_plate_t640.onnx', ...['whisper-base_timestamped', 'whisper-small_timestamped'].map((m) => `onnx-community/${m}`)].map((m) => [
      m,
      existsSync(join(ROOT, 'public/models', m)),
    ]),
  );
  const env: Awaited<ReturnType<Window['videodoc']['check']>> = await ctx.browser.run({}, (page) => page.evaluate(() => window.videodoc.check()), undefined, ctx.signal);
  if (!env.webgpu) problems.push('no WebGPU adapter: faces and scenes need it (try --browser chrome, or update GPU drivers)');
  if (!env.crossOriginIsolated) problems.push('page is not cross-origin isolated: wasm runs single-threaded');
  if (!env.decode.h264) problems.push('no H.264 decoder in this browser');
  // whisper-small is optional (--set transcript.model=...): listed, never a problem. The plate
  // model is required: redaction covers plates by default.
  for (const [m, ok] of Object.entries(models)) if (!ok && !m.endsWith('whisper-small_timestamped')) problems.push(`model not found: public/models/${m}`);
  const gpu = env.webgpu ? `${env.webgpu.vendor} ${env.webgpu.architecture}${env.webgpu.description ? ` (${env.webgpu.description})` : ''}` : 'none';
  ctx.report.log(`browser   ${browserName}: ${env.userAgent.match(/(Edg|Chrome)\/[\d.]+/g)?.join(' ') ?? env.userAgent}`);
  ctx.report.log(`webgpu    ${gpu}${env.webgpu?.shaderF16 ? ', shader-f16' : ''}`);
  ctx.report.log(
    `decode    ${Object.entries(env.decode)
      .map(([k, v]) => `${k} ${v ? 'yes' : 'no'}`)
      .join(', ')}`,
  );
  ctx.report.log(
    `models    ${Object.entries(models)
      .map(([k, v]) => `${k.split('/').pop()} ${v ? 'yes' : 'no'}`)
      .join(', ')}`,
  );
  ctx.report.log(problems.length ? problems.map((p) => `problem   ${p}`).join('\n') : 'ok');
  return { ok: !problems.length, browser: browserName, ...env, models, problems };
}
