#!/usr/bin/env node
// videodoc MCP server (stdio): the CLI's commands as tools for agents, sharing cli/lib.ts. The
// browser stays warm between calls (no build check, browser start or page load after the first
// call; closed after VIDEODOC_IDLE_MS idle, default 10 min). Calls run one at a time, report MCP
// progress when the client sends a progress token, and stop when cancelled.
//
// Usage: node cli/mcp.ts [--browser msedge|chrome] [--headed]
//   e.g. claude mcp add videodoc -- node /path/to/browser-video-lab/cli/mcp.ts
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema, type CallToolResult } from '@modelcontextprotocol/sdk/types.js';
import { LAYER_NAMES, PlanError, PRODUCERS, REDACT_DEFAULTS, REDACT_STYLES, VERSION } from '../src/doc/index.ts';
import {
  analyze, check, DEFAULT_IMAGE_SIZE, DEFAULT_LAYERS, info, layers, MARKDOWN_LAYERS, markdown, redact, search, UsageError, warmBrowser,
  type Context, type Progress,
} from './lib.ts';

// stdout carries the protocol: anything else that prints (vite, libraries) goes to stderr.
console.log = console.error;
console.info = console.error;

const { values: a } = parseArgs({
  options: {
    browser: { type: 'string', default: 'msedge' },
    headed: { type: 'boolean', default: false },
  },
});
// Captured before the harness changes directory: relative paths resolve against the client's cwd.
const cwd = process.cwd();
const browser = warmBrowser({ browser: a.browser!, headed: a.headed! }, Number(process.env.VIDEODOC_IDLE_MS) || undefined);

const settingsList = LAYER_NAMES.flatMap((n) => Object.entries(PRODUCERS[n].defaults).map(([k, v]) => `${n}.${k}=${v}`)).join(', ');
const layerList = LAYER_NAMES.map((n) => `${n}: ${PRODUCERS[n].description}${PRODUCERS[n].needs.length ? ` Needs ${PRODUCERS[n].needs.join(', ')}.` : ''}`).join('\n');

const VIDEO = { type: 'string', description: 'Path to a local video file (absolute, or relative to the server\'s working directory).' };
const LAYERS = {
  type: 'array',
  items: { type: 'string', enum: LAYER_NAMES },
  description: `Layers to compute; dependencies are added, layers already in the doc with the same settings are reused.\n${layerList}`,
};
const SET = {
  type: 'array',
  items: { type: 'string', pattern: '^\\w+\\.\\w+=.*$' },
  description: `Setting overrides as layer.key=value. Defaults: ${settingsList}. Help for each: the layers tool.`,
};
const FORCE = { type: 'boolean', description: 'Recompute the requested layers even if cached.' };
const DRY_RUN = { type: 'boolean', description: 'Only return the plan: which layers are cached, which would be computed, missing models.' };

const TOOLS = [
  {
    name: 'markdown',
    title: 'Read a video',
    description:
      'Read a video as Markdown: shots, timed transcript paragraphs, silent gaps, faces per shot and optionally keyframe images. ' +
      `Analyzes missing layers first (${MARKDOWN_LAYERS.join(', ')} and dependencies; first run of a long video can take minutes, later calls are instant). ` +
      'Start here to understand what a video contains.',
    inputSchema: {
      type: 'object',
      properties: {
        video: VIDEO,
        layers: { ...LAYERS, description: `Extra layers to include, e.g. ["faces"] for face counts per shot. ${LAYERS.description}` },
        images: { type: 'boolean', description: `Extract keyframe JPEGs (${DEFAULT_IMAGE_SIZE} px, into <video>.keyframes/) and link them by absolute path.` },
        embedImages: { type: 'integer', minimum: 0, description: 'Also return up to this many keyframes as image content, evenly spread over the video (implies images). Default 0.' },
        imageSize: { type: 'integer', minimum: 16, description: `Long side of extracted keyframes, px (default ${DEFAULT_IMAGE_SIZE}; existing files are reused).` },
        out: { type: 'string', description: 'Also write the Markdown to this path (image links relative to it).' },
        from: { type: ['string', 'number'], description: 'Only render from this time: seconds, "mm:ss" or "h:mm:ss" (e.g. a search hit\'s start minus a minute).' },
        to: { type: ['string', 'number'], description: 'Only render up to this time (same formats).' },
        set: SET,
        force: FORCE,
        dryRun: DRY_RUN,
      },
      required: ['video'],
    },
  },
  {
    name: 'search',
    title: 'Search videos',
    description:
      'Find where something is said across videos: ranked transcript passages (keyword/BM25) with times (`at`: where the first match is said), a highlighted snippet and the nearest keyframe. ' +
      'Reads saved docs only (instant); videos not yet analyzed are listed, analyze them first. Follow up with markdown from/to around a hit.',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Words to find; passages containing them in order rank higher.' },
        paths: { type: 'array', items: { type: 'string' }, minItems: 1, description: 'Video files, .videodoc.json files or folders (searched recursively).' },
        top: { type: 'integer', minimum: 1, description: 'Hits to return (default 10).' },
      },
      required: ['query', 'paths'],
    },
  },
  {
    name: 'analyze',
    title: 'Analyze videos',
    description:
      'Compute layers into <video>.videodoc.json (saved next to the video) and return a summary per layer; given a folder, analyzes every video in it. ' +
      `Default layers: ${DEFAULT_LAYERS.join(', ')}. Cached layers are reused, so repeating a call is cheap.`,
    inputSchema: {
      type: 'object',
      properties: { video: { ...VIDEO, description: `${VIDEO.description} A folder analyzes every video in it.` }, layers: LAYERS, set: SET, force: FORCE, dryRun: DRY_RUN },
      required: ['video'],
    },
  },
  {
    name: 'redact',
    title: 'Blur faces in a video',
    description:
      'Write a copy of the video with every detected face blurred (or pixelated, or filled), audio copied, plus a redaction log (<out>.redactions.json) listing every face track and which were kept visible. ' +
      'Analyzes faces first if needed (slow: roughly 2-12x real time on an integrated GPU, crowds slowest). Detection misses some faces, so a person must review the output before release; ' +
      'say so when reporting the result. Track ids for keep come from a previous log or the doc.',
    inputSchema: {
      type: 'object',
      properties: {
        video: VIDEO,
        out: { type: 'string', description: 'Output MP4 path (default: <name>.redacted.mp4 next to the video).' },
        style: { type: 'string', enum: REDACT_STYLES, description: `How faces are covered (default ${REDACT_DEFAULTS.style}; fill leaves nothing to undo).` },
        margin: { type: 'number', minimum: 0, maximum: 2, description: `Grow each face box by this fraction on every side (default ${REDACT_DEFAULTS.margin}).` },
        hold: { type: 'number', minimum: 0, maximum: 10, description: `Seconds to keep covering before and after each face track, within its shot (default ${REDACT_DEFAULTS.hold}).` },
        keep: { type: 'array', items: { type: 'integer', minimum: 1 }, description: 'Face track ids to leave visible.' },
        set: SET,
        force: FORCE,
        dryRun: DRY_RUN,
      },
      required: ['video'],
    },
  },
  {
    name: 'info',
    title: 'Summarize a saved doc',
    description: 'Summarize the saved videodoc of a video (or a .videodoc.json path): source facts, layers present and missing. No compute.',
    inputSchema: { type: 'object', properties: { video: { ...VIDEO, description: `${VIDEO.description} A .videodoc.json path also works.` } }, required: ['video'] },
  },
  {
    name: 'layers',
    title: 'List layers',
    description: 'Layers videodoc can compute, their dependencies, and every setting with its default and meaning.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'check',
    title: 'Check this machine',
    description: 'What this machine can run: browser, WebGPU adapter, video/audio decoders, model files. Run when analysis fails.',
    inputSchema: { type: 'object', properties: {} },
  },
] as const;

const server = new Server(
  { name: 'videodoc', title: 'videodoc', version: VERSION },
  {
    capabilities: { tools: {}, logging: {} },
    instructions:
      'videodoc analyzes local video files on this machine (in a headless browser with WebGPU) into a saved document of time-indexed layers, then answers from it. ' +
      'To understand a video, call markdown (add images: true or embedImages to see keyframes). To find where something is said across videos, ' +
      'call search over files or folders, then markdown with from/to around a hit. To blur faces before sharing a video, call redact. ' +
      'Analysis is cached per video and settings; analyze takes a folder.',
  },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS as unknown as never[] }));

const json = (v: unknown) => JSON.stringify(v, null, 1);

/** n indices spread evenly over [0, len). */
const spread = (len: number, n: number) => (n >= len ? [...Array(len).keys()] : [...Array(n).keys()].map((i) => Math.round(((i + 0.5) * len) / n - 0.5)));

server.setRequestHandler(CallToolRequestSchema, async (req, extra): Promise<CallToolResult> => {
  const args = (req.params.arguments ?? {}) as Record<string, unknown>;
  const token = extra._meta?.progressToken;
  let step = 0;
  let lastSent = 0;
  let lastStage = '';
  const progress = (p: Progress) => {
    if (token === undefined) return;
    // MCP progress must increase; stages restart their counts, so send a running step number
    // and put the stage in the message. Throttled to ~2 per second, plus every stage change.
    const stage = `${p.pass}: ${p.stage}`;
    const now = Date.now();
    if (stage === lastStage && now - lastSent < 500) return;
    lastSent = now;
    lastStage = stage;
    const message = `${stage}${p.total > 1 ? ` ${p.done}/${p.total}` : ''}`;
    void extra.sendNotification({ method: 'notifications/progress', params: { progressToken: token, progress: ++step, message } }).catch(() => {});
  };
  const ctx: Context = {
    browser,
    cwd,
    signal: extra.signal,
    report: {
      log: (line) => {
        process.stderr.write(line + '\n');
        if (line.trim()) void server.sendLoggingMessage({ level: 'info', logger: 'videodoc', data: line.trim() }).catch(() => {});
      },
      progress,
    },
  };
  const common = {
    layers: args.layers as string[] | undefined,
    set: args.set as string[] | undefined,
    force: args.force as boolean | undefined,
    dryRun: args.dryRun as boolean | undefined,
  };
  const video = () => {
    if (typeof args.video !== 'string' || !args.video) throw new UsageError('video is required: a path to a local video file');
    return args.video;
  };
  try {
    switch (req.params.name) {
      case 'markdown': {
        const embed = Number(args.embedImages ?? 0);
        const r = await markdown(ctx, video(), {
          ...common,
          out: args.out as string | undefined,
          images: Boolean(args.images) || embed > 0,
          imageSize: args.imageSize as number | undefined,
          from: args.from as string | number | undefined,
          to: args.to as string | number | undefined,
        });
        if (r.text === undefined) return { content: [{ type: 'text', text: json(r.summary) }] };
        const content: CallToolResult['content'] = [{ type: 'text', text: r.text }];
        for (const i of spread(r.imageFiles.length, embed)) {
          const f = r.imageFiles[i];
          content.push({ type: 'text', text: `Keyframe ${i + 1}/${r.imageFiles.length}: ${f.replace(/\\/g, '/')}` });
          content.push({ type: 'image', data: readFileSync(f).toString('base64'), mimeType: 'image/jpeg' });
        }
        content.push({ type: 'text', text: json(r.summary) });
        return { content };
      }
      case 'search': {
        if (typeof args.query !== 'string') throw new UsageError('query is required');
        if (!Array.isArray(args.paths) || !args.paths.length) throw new UsageError('paths is required: video files or folders to search');
        return { content: [{ type: 'text', text: json(search(ctx, args.query, args.paths as string[], { top: args.top as number | undefined })) }] };
      }
      case 'analyze':
        return { content: [{ type: 'text', text: json(await analyze(ctx, video(), common)) }] };
      case 'redact':
        return {
          content: [
            {
              type: 'text',
              text: json(
                await redact(ctx, video(), {
                  set: common.set, force: common.force, dryRun: common.dryRun,
                  out: args.out as string | undefined, style: args.style as string | undefined,
                  margin: args.margin as number | undefined, hold: args.hold as number | undefined, keep: args.keep as number[] | undefined,
                }),
              ),
            },
          ],
        };
      case 'info':
        return { content: [{ type: 'text', text: json(info(ctx, video())) }] };
      case 'layers':
        return { content: [{ type: 'text', text: json(layers()) }] };
      case 'check':
        return { content: [{ type: 'text', text: json(await check(ctx, a.browser!)) }] };
      default:
        throw new UsageError(`unknown tool '${req.params.name}' (have: ${TOOLS.map((t) => t.name).join(', ')})`);
    }
  } catch (e) {
    const err = e as Error;
    const kind = e instanceof UsageError || e instanceof PlanError ? 'usage' : 'failure';
    process.stderr.write(`error: ${err.message}\n`);
    return { isError: true, content: [{ type: 'text', text: json({ error: err.message, kind }) }] };
  }
});

const shutdown = async () => {
  await browser.close();
  process.exit(0);
};
process.stdin.on('close', () => void shutdown());
process.on('SIGINT', () => void shutdown());
process.on('SIGTERM', () => void shutdown());

await server.connect(new StdioServerTransport());
process.stderr.write(`videodoc MCP server ${VERSION} ready (browser: ${a.browser})\n`);
