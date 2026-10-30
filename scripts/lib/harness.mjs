// Shared headless harness: build the pages, serve them with `vite preview`, and drive
// Edge/Chrome via playwright-core. A page signals completion by setting window.__BENCH_RESULT__.
import { execSync } from 'node:child_process';
import { createReadStream, existsSync, readdirSync, statSync } from 'node:fs';
import { open } from 'node:fs/promises';
import { join } from 'node:path';
import { chromium } from 'playwright-core';
import { preview } from 'vite';

/** Build sources: anything newer than the last build means the build is stale. */
const SOURCES = ['src', 'index.html', 'pipeline.html', 'transcribe.html', 'label.html', 'analyze.html', 'app.html', 'vite.config.ts'];

function newest(path) {
  const s = statSync(path);
  if (!s.isDirectory()) return s.mtimeMs;
  return Math.max(0, ...readdirSync(path).map((f) => newest(join(path, f))));
}

export function buildIsStale() {
  if (!existsSync('dist/analyze.html')) return true;
  const built = statSync('dist/analyze.html').mtimeMs;
  return SOURCES.some((p) => existsSync(p) && newest(p) > built);
}

/**
 * Serve local files to the page at `/@file/<key>`, with range requests (mediabunny reads
 * videos in ranges, so large files are never loaded whole). Keys starting with `out` are
 * outputs: the page writes them with `PUT /@file/<key>?pos=<byte offset>` (the file must exist).
 */
function serveFiles(files) {
  return {
    name: 'serve-local-files',
    configurePreviewServer(server) {
      server.middlewares.use('/@file/', (req, res) => {
        const key = decodeURIComponent((req.url ?? '').replace(/^\//, '').split('?')[0]);
        const path = files[key];
        if (!path) {
          res.statusCode = 404;
          return res.end();
        }
        if (req.method === 'PUT') return writeAt(key, path, req, res);
        const size = statSync(path).size;
        res.setHeader('Accept-Ranges', 'bytes');
        res.setHeader('Content-Type', 'application/octet-stream');
        const m = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? '');
        let [start, end] = [0, size - 1];
        if (m) {
          start = m[1] ? Number(m[1]) : Math.max(0, size - Number(m[2]));
          end = m[1] && m[2] ? Math.min(Number(m[2]), size - 1) : size - 1;
          if (start >= size || start > end) {
            res.statusCode = 416;
            res.setHeader('Content-Range', `bytes */${size}`);
            return res.end();
          }
          res.statusCode = 206;
          res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
        }
        res.setHeader('Content-Length', end - start + 1);
        if (req.method === 'HEAD') return res.end();
        createReadStream(path, { start, end }).pipe(res);
      });
    },
  };
}

async function writeAt(key, path, req, res) {
  const pos = Number(new URLSearchParams((req.url ?? '').split('?')[1] ?? '').get('pos'));
  if (!key.startsWith('out') || !Number.isSafeInteger(pos) || pos < 0) {
    res.statusCode = key.startsWith('out') ? 400 : 403;
    return res.end();
  }
  const parts = [];
  for await (const c of req) parts.push(c);
  const data = Buffer.concat(parts);
  const fh = await open(path, 'r+');
  try {
    await fh.write(data, 0, data.length, pos);
  } finally {
    await fh.close();
  }
  res.statusCode = 204;
  res.end();
}

/** A page whose first few browser errors (first line each) are relayed to stderr. */
export async function newPage(browser) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  let relayed = 0;
  // ORT's notice that some nodes run on the CPU is expected (shape ops), not an error.
  const relay = (kind, text) => !/VerifyEachNodeIsAssignedToAnEp/.test(text) && relayed++ < 5 && console.error(`  [${kind}] ${text.split('\n')[0].slice(0, 200)}`);
  page.on('pageerror', (e) => relay('pageerror', e.message));
  page.on('console', (m) => m.type() === 'error' && relay('console', m.text()));
  return page;
}

/**
 * Build if needed and serve the pages (cross-origin isolated, see vite.config.ts); resolves with
 * the vite preview server and its base URL. `files` maps keys to local paths served at
 * `${base}@file/<key>`; it is read on every request, so long-lived callers can change it.
 * `build`: true, false, or 'auto' (when stale). `port` is a preference (the next free one is used).
 */
export async function serve({ build = true, files = {}, port = 4317 } = {}) {
  if (build === true || (build === 'auto' && buildIsStale())) {
    console.error('building...');
    execSync('npx vite build --logLevel warn', { stdio: ['ignore', 'ignore', 'inherit'] });
  }
  const server = await preview({ preview: { port, strictPort: false }, logLevel: 'warn', plugins: [serveFiles(files)] });
  return { server, base: server.resolvedUrls.local[0] };
}

/**
 * Build if needed, serve the pages and launch a browser; resolves with { browser, base, files,
 * close }. See serve() for `build` and `files`.
 */
export async function openBrowser({ build = true, browser = 'msedge', headed = false, files = {}, handleSIGINT = true }) {
  const { server, base } = await serve({ build, files });
  let b;
  try {
    b = await chromium.launch({
      channel: browser,
      headless: !headed,
      args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
      // false: the caller handles Ctrl+C (the videodoc CLI cancels, cleans up, then closes).
      handleSIGINT,
    });
  } catch (e) {
    await server.close();
    const other = browser === 'msedge' ? 'chrome' : 'msedge';
    throw new Error(`could not launch ${browser} (${e.message.split('\n')[0]}); install it or pass --browser ${other}`);
  }
  const close = async () => {
    await b.close();
    await server.close();
  };
  return { browser: b, base, files, close };
}

/**
 * Run `fn(open, ctx)` with a served build and a browser (see openBrowser). `open(page, params)`
 * loads e.g. `pipeline.html?<params>` and resolves with { result, page } once the page reports
 * a result. `ctx` = { browser, base } for drivers that talk to a page directly.
 */
export async function withBrowser(opts, fn) {
  const { browser: b, base, close } = await openBrowser(opts);
  const open = async (path, params, { timeoutMs = 30 * 60_000 } = {}) => {
    const page = await newPage(b);
    await page.goto(`${base}${path}?${new URLSearchParams(params)}`);
    await page.waitForFunction(() => window.__BENCH_RESULT__, null, { timeout: timeoutMs, polling: 250 });
    return { result: await page.evaluate(() => window.__BENCH_RESULT__), page };
  };
  try {
    return await fn(open, { browser: b, base });
  } finally {
    await close();
  }
}
