// Shared headless harness: build the pages, serve them with `vite preview`, and drive
// Edge/Chrome via playwright-core. A page signals completion by setting window.__BENCH_RESULT__.
import { execSync } from 'node:child_process';
import { chromium } from 'playwright-core';
import { preview } from 'vite';

/**
 * Run `fn(open)` with a served build and a browser. `open(page, params)` loads e.g.
 * `pipeline.html?<params>` and resolves with { result, page } once the page reports a result.
 */
export async function withBrowser({ build = true, browser = 'msedge', headed = false }, fn) {
  if (build) {
    console.error('building...');
    execSync('npx vite build --logLevel warn', { stdio: ['ignore', 'ignore', 'inherit'] });
  }
  const server = await preview({ preview: { port: 4317, strictPort: false }, logLevel: 'warn' });
  const base = server.resolvedUrls.local[0];
  const b = await chromium.launch({
    channel: browser,
    headless: !headed,
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
  });
  const open = async (path, params, { timeoutMs = 30 * 60_000 } = {}) => {
    const page = await b.newPage({ viewport: { width: 1280, height: 900 } });
    // Relay only the first few browser errors (first line each) so a failing shader can't flood the output.
    let relayed = 0;
    const relay = (kind, text) => relayed++ < 5 && console.error(`  [${kind}] ${text.split('\n')[0].slice(0, 200)}`);
    page.on('pageerror', (e) => relay('pageerror', e.message));
    page.on('console', (m) => m.type() === 'error' && relay('console', m.text()));
    await page.goto(`${base}${path}?${new URLSearchParams(params)}`);
    await page.waitForFunction(() => window.__BENCH_RESULT__, null, { timeout: timeoutMs, polling: 250 });
    return { result: await page.evaluate(() => window.__BENCH_RESULT__), page };
  };
  try {
    return await fn(open);
  } finally {
    await b.close();
    await server.close();
  }
}
