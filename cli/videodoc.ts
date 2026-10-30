#!/usr/bin/env node
// videodoc CLI: analyze a video once into <video>.videodoc.json, then query the doc. The
// commands live in cli/lib.ts (shared with the MCP server, cli/mcp.ts); this file parses
// arguments and prints.
//
// stdout: JSON results (one object per command; `markdown` without --out prints Markdown).
// stderr: progress and human summaries. Exit codes: 0 ok, 1 failure, 2 usage error, 130 cancelled.
import { parseArgs } from 'node:util';
import { DocError, LAYER_NAMES, PlanError, REDACT_DEFAULTS, REDACT_SOUNDS, REDACT_STYLES } from '../src/doc/index.ts';
import {
  analyze,
  type Context,
  check,
  DEFAULT_IMAGE_SIZE,
  DEFAULT_LAYERS,
  info,
  layers,
  MARKDOWN_LAYERS,
  markdown,
  oneShotBrowser,
  openUrl,
  type Progress,
  redact,
  search,
  serveApp,
  speakers,
  UsageError,
} from './lib.ts';

const HELP = `videodoc: analyze a video once, query it many times. Runs in a local headless browser (WebGPU).

Usage: videodoc <command> [options]

Commands:
  ui                     The app in your browser: open a folder or drop videos, analyze, read and
                         search them. Uses the same docs as the CLI. Needs Chrome or Edge (WebGPU).
  check                  What this machine can run: browser, WebGPU adapter, decoders, models.
  analyze <video|folder> Compute layers into <video>.videodoc.json (a folder: every video in it).
                         Layers already in the doc with the same settings are reused; with nothing
                         to compute, no browser starts.
  markdown <video>       Read the video: shots, timed transcript paragraphs, silent gaps and
                         (with --images) keyframe JPEGs, as Markdown. Analyzes missing layers first
                         (${MARKDOWN_LAYERS.join(', ')} and their dependencies, plus any --layers).
  search <query> <paths...>
                         Find where something is said: ranked transcript passages across videos
                         and folders, with times and the nearest keyframe (no browser; analyze first).
  speakers <video>       Who speaks, and your edits: name speakers, merge one person found as
                         two, move speech given to the wrong speaker. Saved as
                         <video>.speakers.json; the Markdown, search, info and redact use the
                         names. Suggests names from voices named in other videos (no browser;
                         analyze with --layers speakers first).
  redact <video>         Blur every face and license plate, and bleep listed words (--words,
                         --profanity) or speakers (--speakers): writes <name>.redacted.mp4 and a redaction log
                         (<out>.redactions.json: every track and bleeped word, which were kept).
                         Analyzes what it needs first. Uses <video>.redact.json when there (a
                         review made in the app: kept tracks, regions added by hand).
                         Watch and listen before release: detection and transcription miss some.
  info <video|doc>       Summarize a saved doc (no browser).
  layers                 Layers, their dependencies and settings with defaults (JSON).

analyze options:
  --layers <list>        Comma-separated (default: ${DEFAULT_LAYERS.join(',')}); dependencies are added.
                         Available: ${LAYER_NAMES.join(', ')}.
  --set <layer.key=v>    Override a setting, repeatable (e.g. --set faces.every=4). See 'videodoc layers'.
  --pipeline <file>      A pipeline preset (JSON, docs/pipelines.md): its layers (unless --layers),
                         settings (--set wins), models and track layers (e.g. vehicles from a COCO
                         detector). Also for markdown, redact and layers.
  --out <path>           Doc path (default: <video>.videodoc.json).
  --force                Recompute the requested layers even if cached.
  --dry-run              Print the plan and exit.

markdown options:
  --out <path>           Write the Markdown here (image links relative to it) and print a JSON
                         summary; default: Markdown on stdout with absolute image paths.
  --images               Link keyframe JPEGs, extracting missing ones into <video>.keyframes/
                         (named by frame; delete the folder to re-extract at another size).
  --image-size <px>      Long side of extracted keyframes (default ${DEFAULT_IMAGE_SIZE}).
  --from, --to <time>    Only this stretch (seconds, mm:ss or h:mm:ss), e.g. around a search hit.
  --layers, --set, --force   As for analyze (e.g. --layers faces adds faces per shot).

redact options:
  --out <path>           Output MP4 (default: <name>.redacted.mp4 next to the video).
  --style <s>            ${REDACT_STYLES.join(' | ')} (default ${REDACT_DEFAULTS.style}; fill leaves nothing to undo).
  --margin <f>           Grow each face box by this fraction on every side (default ${REDACT_DEFAULTS.margin}).
  --hold <s>             Keep covering this long before and after each face track (default ${REDACT_DEFAULTS.hold}).
  --keep <ids>           Face track ids to leave visible, e.g. 3,7 (ids are in the log and the doc).
  --keep-plates <ids>    Plate track ids to leave visible (a sign taken for a plate; ids are per layer).
  --cover <list>         What to cover: faces,plates (default), faces, plates, a pipeline's track
                         layers (faces,vehicles with --pipeline), or none (audio only: the video
                         is copied).
  --keep-tracks <l=ids>  A pipeline track layer's ids to leave visible, repeatable (vehicles=3,7).
  --words <list>         Words and phrases to bleep, comma-separated, e.g. "Okafor,Daniel Reyes".
                         Plain words also match plurals, possessives, -ed and -ing; * is a
                         wildcard ("fuck*"); "quoted" matches exactly. Spell names as the
                         transcript does (videodoc markdown shows it).
  --words-file <path>    Words and phrases to bleep, one per line (# comments); adds to --words.
  --profanity            Also bleep the built-in strong-language list (src/doc/words.ts).
  --sound <s>            ${REDACT_SOUNDS.join(' | ')} (default ${REDACT_DEFAULTS.sound}).
  --lead, --tail <s>     Seconds covered before and after each word (default ${REDACT_DEFAULTS.lead}, ${REDACT_DEFAULTS.tail}:
                         word times from Whisper run late).
  --keep-words <keys>    Matches to leave audible, by word key from the log (e.g. 12.3).
  --speakers <list>      Bleep everything these speakers say: names or ids, e.g. "Dana,3".
                         Options given here override the review's.
  --no-plan              Ignore <video>.redact.json.
  --set, --force         As for analyze (detection settings, e.g. --set faces.every=1).

search options:
  --top <n>              Hits to return (default 10).
  --speaker <s>          Only what this speaker says (a name given with speakers, or 2).

speakers options:
  --name <id=name>       Name a speaker, repeatable: --name 2=Dana (2= removes the name).
  --merge <a=b>          a is b (one person found as two), repeatable: --merge 3=Dana.
  --unmerge <id>         Undo a merge: the speaker stands alone again.
  --move <span=to>       Speech given to the wrong speaker, repeatable: 1:02-1:05=Dana (all
                         speech then) or "Speaker 2@1:02-1:05=Dana" (only theirs). A name that is
                         no speaker's adds one.
  --unmove <i>           Undo move i (as listed).
  --voices <folder>      Also suggest names from speakers files under this folder (recursive;
                         the video's own folder is always searched), repeatable.
  --reset                Delete <video>.speakers.json first.

ui options:
  --port <n>             Preferred port (default 4317; the next free one is used).
  --no-open              Print the URL instead of opening the browser.

Common options:
  --browser msedge|chrome   (default msedge)   --headed   --help

stdout is JSON; progress and summaries go to stderr. For agents: 'npm run mcp' serves the same
commands over MCP (stdio) with a warm browser.`;

const { values: a, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    layers: { type: 'string' },
    set: { type: 'string', multiple: true, default: [] },
    pipeline: { type: 'string' },
    'keep-tracks': { type: 'string', multiple: true },
    out: { type: 'string' },
    force: { type: 'boolean', default: false },
    'dry-run': { type: 'boolean', default: false },
    images: { type: 'boolean', default: false },
    'image-size': { type: 'string', default: String(DEFAULT_IMAGE_SIZE) },
    from: { type: 'string' },
    to: { type: 'string' },
    top: { type: 'string', default: '10' },
    style: { type: 'string' },
    margin: { type: 'string' },
    hold: { type: 'string' },
    keep: { type: 'string' },
    'keep-plates': { type: 'string' },
    cover: { type: 'string' },
    words: { type: 'string' },
    'words-file': { type: 'string' },
    profanity: { type: 'boolean' },
    sound: { type: 'string' },
    lead: { type: 'string' },
    tail: { type: 'string' },
    'keep-words': { type: 'string' },
    speakers: { type: 'string' },
    speaker: { type: 'string' },
    name: { type: 'string', multiple: true },
    merge: { type: 'string', multiple: true },
    unmerge: { type: 'string', multiple: true },
    move: { type: 'string', multiple: true },
    unmove: { type: 'string', multiple: true },
    voices: { type: 'string', multiple: true },
    reset: { type: 'boolean', default: false },
    port: { type: 'string', default: '4317' },
    'no-open': { type: 'boolean', default: false },
    'no-plan': { type: 'boolean', default: false },
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
// Ctrl+C cancels: the compute page is discarded, a partly written output is deleted, nothing
// half-done is saved. A second Ctrl+C exits at once.
const cancel = new AbortController();
process.on('SIGINT', () => {
  if (cancel.signal.aborted) process.exit(130);
  progress.end();
  log('cancelling… (Ctrl+C again to quit at once)');
  cancel.abort();
});
const ctx: Context = {
  signal: cancel.signal,
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

const list = (s: string | undefined) =>
  (s ?? '')
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean);

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
  if (cmd === 'layers') return out(layers(ctx, a.pipeline));
  if (cmd === 'ui') {
    const port = Number(a.port);
    if (!Number.isInteger(port) || port < 1 || port > 65535) throw new UsageError(`--port takes a port number, got '${a.port}'`);
    const url = await serveApp(port);
    log(`videodoc app: ${url}  (Chrome or Edge; Ctrl+C to stop)`);
    out({ url });
    if (!a['no-open']) openUrl(url);
    return new Promise(() => {}); // serve until interrupted
  }
  if (cmd === 'search') {
    if (rest.length < 2) throw new UsageError('usage: videodoc search <query> <video|folder>...');
    const top = Number(a.top);
    if (!(top >= 1)) throw new UsageError(`--top takes a positive number, got '${a.top}'`);
    return out(search(ctx, rest[0], rest.slice(1), { top, speaker: a.speaker }));
  }
  if (!['analyze', 'markdown', 'redact', 'info', 'speakers'].includes(cmd))
    throw new UsageError(`unknown command '${cmd}' (have: ui, check, analyze, markdown, search, speakers, redact, info, layers; see --help)`);
  if (rest.length !== 1) throw new UsageError(`usage: videodoc ${cmd} <video>`);
  const common = { layers: list(a.layers), set: a.set, force: a.force, dryRun: a['dry-run'], pipeline: a.pipeline };
  if (cmd === 'info') return out(info(ctx, rest[0]));
  if (cmd === 'speakers')
    return out(speakers(ctx, rest[0], { name: a.name, merge: a.merge, unmerge: a.unmerge, move: a.move, unmove: a.unmove, voices: a.voices, reset: a.reset }));
  if (cmd === 'analyze') return out(await analyze(ctx, rest[0], { ...common, doc: a.out }));
  if (cmd === 'redact') {
    const num = (v: string | undefined, what: string) => {
      if (v === undefined) return undefined;
      const n = Number(v);
      if (!Number.isFinite(n)) throw new UsageError(`--${what} takes a number, got '${v}'`);
      return n;
    };
    return out(
      await redact(ctx, rest[0], {
        set: a.set,
        force: a.force,
        dryRun: a['dry-run'],
        pipeline: a.pipeline,
        out: a.out,
        style: a.style,
        margin: num(a.margin, 'margin'),
        hold: num(a.hold, 'hold'),
        keep: a.keep,
        keepPlates: a['keep-plates'],
        keepTracks: a['keep-tracks'],
        cover: a.cover,
        words: a.words,
        wordsFile: a['words-file'],
        profanity: a.profanity,
        sound: a.sound,
        lead: num(a.lead, 'lead'),
        tail: num(a.tail, 'tail'),
        keepWords: a['keep-words'],
        speakers: a.speakers,
        plan: !a['no-plan'],
      }),
    );
  }
  const size = Number(a['image-size']);
  if (!Number.isFinite(size)) throw new UsageError(`--image-size takes a number of pixels, got '${a['image-size']}'`);
  const r = await markdown(ctx, rest[0], { ...common, out: a.out, images: a.images, imageSize: size, from: a.from, to: a.to });
  if (r.text !== undefined && !a.out) process.stdout.write(r.text);
  else out(r.summary);
}

main().catch((e: unknown) => {
  if (cancel.signal.aborted) {
    log('cancelled');
    out({ error: 'cancelled', kind: 'cancelled' });
    process.exitCode = 130;
    return;
  }
  const err = e as Error;
  const usage = e instanceof UsageError || e instanceof PlanError;
  log(`error: ${err.message}`);
  if (!usage && !(e instanceof DocError) && process.env.VIDEODOC_DEBUG) log(err.stack ?? '');
  out({ error: err.message, kind: usage ? 'usage' : 'failure' });
  process.exitCode = usage ? 2 : 1;
});
