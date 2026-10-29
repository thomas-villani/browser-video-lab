// The app (app.html) in headless Edge, as a person uses it: add a video, analyze it, read and
// search it, seek from a hit; review a crowd's faces and download the blurred copy and its log.
// Run: npm run test:browser
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, test } from 'node:test';
import { ROOT } from '../../cli/lib.ts';
import { context, fixture, fresh, notReady } from './lib.ts';

type Page = any; // playwright-core is untyped JS here, as in cli/lib.ts

let app: { browser: any; base: string; close(): Promise<void> } | undefined;
let skip: string | undefined;

before(async () => {
  const ctx = context();
  skip = await notReady(ctx);
  await ctx.browser.close();
  if (skip) return;
  process.chdir(ROOT);
  const { openBrowser } = await import(new URL('../../scripts/lib/harness.mjs', import.meta.url).href);
  app = await openBrowser({ build: 'auto', browser: process.env.VIDEODOC_BROWSER ?? 'msedge', headed: false });
});
after(() => app?.close());

/** A fresh app page; errors other than ORT's CPU-node notice are collected. */
async function open(): Promise<{ page: Page; errors: string[] }> {
  const page = await app!.browser.newPage({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
  const errors: string[] = [];
  page.on('pageerror', (e: Error) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m: any) => m.type() === 'error' && !/VerifyEachNodeIsAssignedToAnEp/.test(m.text()) && errors.push(`console: ${m.text()}`));
  // No File System Access in tests: outputs come back as downloads.
  await page.addInitScript(() => delete (window as { showSaveFilePicker?: unknown }).showSaveFilePicker);
  await page.goto(`${app!.base}app.html`);
  await page.waitForFunction(() => !document.querySelector('#gpu')!.textContent!.includes('checking'));
  return { page, errors };
}

const clean = (s: string | null) => (s ?? '').replace(/\s+/g, ' ').trim();

test('app: add a video, analyze it, read it, search it and seek from a hit', async (t) => {
  if (skip) return t.skip(skip);
  const { page, errors } = await open();
  await page.setInputFiles('#add-files', [fresh(fixture('tts-words')!)]);
  await page.waitForSelector('#list li');
  await page.click('#list li');
  await page.click('#read .analyze-this');
  await page.waitForFunction(() => document.querySelector('#list li')?.classList.contains('ready'), null, { timeout: 300_000 });
  const read = clean(await page.textContent('#read'));
  assert.match(read, /Daniel Reyes/, 'the transcript is in the reading view');

  await page.fill('#q', 'Daniel Reyes');
  await page.waitForSelector('#results li.hit');
  const hits = await page.$$eval('#results li.hit', (e: Element[]) => e.length);
  assert.ok(hits >= 1, 'search finds the name');
  const want = Number(await page.getAttribute('#results li.hit', 'data-t'));
  assert.ok(want > 3 && want < 6, `the hit points at the name, said at 4.5 s (${want})`);
  await page.click('#results li.hit');
  await page.waitForFunction((t: number) => Math.abs((document.querySelector('#player') as HTMLVideoElement).currentTime - t) < 0.1, want);
  assert.deepEqual(errors, []);
  await page.close();
});

test('app: review a crowd, then download the blurred copy and its log', async (t) => {
  if (skip) return t.skip(skip);
  const path = fixture('walk-crowd');
  if (!path) return t.skip('fixture not cut: node scripts/fixtures.mjs');
  const { page, errors } = await open();
  await page.setInputFiles('#add-files', [fresh(path)]);
  await page.waitForSelector('#list li');
  await page.click('#list li');
  await page.click('#read .analyze-this');
  await page.waitForSelector('#doc-actions #review-faces', { timeout: 300_000 });
  await page.click('#doc-actions #review-faces');
  // Faces are found first, then the review opens.
  await page.waitForSelector('#review .rv-head', { timeout: 300_000 });
  const head = clean(await page.textContent('#review .rv-head h1'));
  assert.match(head, /faces/i);
  const regions = await page.$$eval('#review .rv-region, #review [data-track]', (e: Element[]) => e.length);
  t.diagnostic(`review: "${head}", ${regions} regions on screen`);

  const downloads: any[] = [];
  page.on('download', (d: any) => downloads.push(d));
  await page.click('#review [data-act=redact]');
  await page.waitForFunction(() => document.querySelector('#list li .status')?.textContent?.includes('redacted in'), null, { timeout: 300_000 });
  await page.waitForTimeout(1500);
  const names = downloads.map((d) => d.suggestedFilename() as string);
  assert.ok(names.some((n) => n.endsWith('.mp4')), `the copy downloaded (${names.join(', ')})`);
  const logDl = downloads.find((d) => d.suggestedFilename().endsWith('.redactions.json'));
  assert.ok(logDl, 'the log downloaded');
  const logPath = join(mkdtempSync(join(tmpdir(), 'videodoc-app-')), 'log.json');
  await logDl.saveAs(logPath);
  const log = JSON.parse(readFileSync(logPath, 'utf8'));
  assert.ok(log.totals.tracks > 100, `log lists the crowd's tracks (${log.totals.tracks})`);
  assert.equal(log.output.frames, 300);
  assert.deepEqual(errors, []);
  await page.close();
});
