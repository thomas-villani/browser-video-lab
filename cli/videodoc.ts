#!/usr/bin/env node
// videodoc CLI: analyze a video once into <video>.videodoc.json, then query the doc. The
// commands live in cli/lib.ts (shared with the MCP server, cli/mcp.ts); this file parses
// arguments and prints.
//
// stdout: JSON results (one object per command; `markdown` without --out prints Markdown).
// stderr: progress and human summaries. Exit codes: 0 ok, 1 failure, 2 usage error.
import { parseArgs } from 'node:util';
import { DocError, LAYER_NAMES, PlanError } from '../src/doc/index.ts';
import {
  analyze, check, DEFAULT_IMAGE_SIZE, DEFAULT_LAYERS, info, layers, MARKDOWN_LAYERS, markdown, oneShotBrowser, UsageError,
  type Context, type Progress,
} from './lib.ts';

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
  --image-size <px>      Long side of extracted keyframes (default ${DEFAULT_IMAGE_SIZE}).
  --layers, --set, --force   As for analyze (e.g. --layers faces adds faces per shot).

Common options:
  --browser msedge|chrome   (default msedge)   --headed   --help

stdout is JSON; progress and summaries go to stderr. For agents: 'npm run mcp' serves the same
commands over MCP (stdio) with a warm browser.`;

const { values: a, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    layers: { type: 'string' },
    set: { type: 'string', multiple: true, default: [] },
    out: { type: 'string' },
    force: { type: 'boolean', default: false },
    'dry-run': { type: 'boolean', default: false },
    images: { type: 'boolean', default: false },
    'image-size': { type: 'string', default: String(DEFAULT_IMAGE_SIZE) },
    browser: { type: 'string', default: 'msedge' },
    headed: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
});

const out = (v: unknown) => process.stdout.write(JSON.stringify(v) + '\n');
const log = (s: string) => process.stderr.write(s + '\n');

/** Progress on stderr: rewritten in place on a terminal, every ~10% otherwise. */
function progressPrinter() {
  const tty = process.stderr.isTTY;
  let last = '';
  let lastStage = '';
  let lastPct = 0;
  const t0 = Date.now();
  return {
    update(p: Progress) {
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
    /** Finish an in-place line before other output. */
    end() {
      if (tty && last) process.stderr.write('\n');
      last = '';
    },
  };
}

const progress = progressPrinter();
const ctx: Context = {
  browser: oneShotBrowser({ browser: a.browser!, headed: a.headed! }),
  report: {
    log: (s) => {
      progress.end();
      log(s);
    },
    progress: (p) => progress.update(p),
  },
  // User paths are relative to where the user ran the command (npm run moves cwd to the repo root).
  cwd: process.env.INIT_CWD ?? process.cwd(),
};

const list = (s: string | undefined) => (s ?? '').split(',').map((x) => x.trim()).filter(Boolean);

async function main() {
  const [cmd, ...rest] = positionals;
  if (a.help || !cmd) {
    console.log(HELP);
    if (!cmd && !a.help) process.exitCode = 2;
    return;
  }
  if (cmd === 'check') {
    const r = await check(ctx, a.browser!);
    out(r);
    if (!r.ok) process.exitCode = 1;
    return;
  }
  if (cmd === 'layers') return out(layers());
  if (!['analyze', 'markdown', 'info'].includes(cmd)) throw new UsageError(`unknown command '${cmd}' (have: check, analyze, markdown, info, layers; see --help)`);
  if (rest.length !== 1) throw new UsageError(`usage: videodoc ${cmd} <video>`);
  const common = { layers: list(a.layers), set: a.set, force: a.force, dryRun: a['dry-run'] };
  if (cmd === 'info') return out(info(ctx, rest[0]));
  if (cmd === 'analyze') return out(await analyze(ctx, rest[0], { ...common, doc: a.out }));
  const size = Number(a['image-size']);
  if (!Number.isFinite(size)) throw new UsageError(`--image-size takes a number of pixels, got '${a['image-size']}'`);
  const r = await markdown(ctx, rest[0], { ...common, out: a.out, images: a.images, imageSize: size });
  if (r.text !== undefined && !a.out) process.stdout.write(r.text);
  else out(r.summary);
}

main().catch((e: unknown) => {
  const err = e as Error;
  const usage = e instanceof UsageError || e instanceof PlanError;
  log(`error: ${err.message}`);
  if (!usage && !(e instanceof DocError) && process.env.VIDEODOC_DEBUG) log(err.stack ?? '');
  out({ error: err.message, kind: usage ? 'usage' : 'failure' });
  process.exitCode = usage ? 2 : 1;
});
