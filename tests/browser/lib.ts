// Shared setup for the browser tests (npm run test:browser): one warm Edge per test file, the
// fixtures copied to a fresh temp folder (so docs are computed, not read from cache), and
// figures pinned in baseline.json with tolerances, so a dependency upgrade or a change that
// shifts results fails loudly. UPDATE_BASELINE=1 rewrites the figures a run measures.
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import type { TestContext } from 'node:test';
import { check, ROOT, warmBrowser, type Context } from '../../cli/lib.ts';
import { parseDoc, type VideoDoc } from '../../src/doc/index.ts';
// @ts-expect-error untyped JS
import { fixturePath, FIXTURES } from '../../scripts/fixtures.mjs';

export const VERBOSE = Boolean(process.env.VIDEODOC_TEST_VERBOSE);
const UPDATE = Boolean(process.env.UPDATE_BASELINE);
const BASELINE = join(ROOT, 'tests/browser/baseline.json');

/** A context on one warm browser; logs go to stderr with VIDEODOC_TEST_VERBOSE=1. */
export function context(): Context {
  return {
    browser: warmBrowser({ browser: process.env.VIDEODOC_BROWSER ?? 'msedge', headed: false }),
    report: { log: (l) => VERBOSE && console.error(l), progress() {} },
    cwd: ROOT,
  };
}

let ready: Promise<string | undefined> | undefined;

/** Why the browser tests cannot run here (no WebGPU, a model missing), or undefined. */
export function notReady(ctx: Context): Promise<string | undefined> {
  ready ??= check({ ...ctx, report: { log() {}, progress() {} } }, process.env.VIDEODOC_BROWSER ?? 'msedge').then(
    (r) => (r.ok ? undefined : r.problems.join('; ')),
    (e: Error) => `browser check failed: ${e.message}`,
  );
  return ready;
}

/** The committed synthetic fixture, or a local cut (scripts/fixtures.mjs); undefined if missing. */
export function fixture(name: string): string | undefined {
  const p = name.startsWith('tts-') ? join(ROOT, `tests/fixtures/${name}.mp4`) : join(ROOT, fixturePath(name));
  return existsSync(p) ? p : undefined;
}

export const LOCAL_FIXTURES = Object.keys(FIXTURES as Record<string, unknown>);

/** A copy of the fixture in a new temp folder, with no doc next to it. */
export function fresh(path: string): string {
  const dir = mkdtempSync(join(tmpdir(), 'videodoc-browser-'));
  const p = join(dir, basename(path));
  copyFileSync(path, p);
  return p;
}

export const loadDoc = (video: string): VideoDoc => parseDoc(readFileSync(`${video}.videodoc.json`, 'utf8'));

type Tolerance = { abs?: number; rel?: number };
type Pinned = Record<string, number | string>;

const pinned: Pinned = existsSync(BASELINE) ? JSON.parse(readFileSync(BASELINE, 'utf8')) : {};
const measured: Pinned = {};

/**
 * Check `value` against the pinned figure `key` within `abs` or `rel` (whichever is wider).
 * Missing from the baseline, or with UPDATE_BASELINE=1: recorded (and the test passes).
 */
export function near(t: TestContext, key: string, value: number, tol: Tolerance) {
  measured[key] = value;
  const want = pinned[key];
  if (UPDATE || want === undefined) {
    t.diagnostic(`${key} = ${value}${want === undefined ? ' (new)' : ` (was ${want})`}`);
    return;
  }
  if (typeof want !== 'number') throw new Error(`${key}: baseline.json holds ${JSON.stringify(want)}, not a number`);
  const slack = Math.max(tol.abs ?? 0, (tol.rel ?? 0) * Math.abs(want));
  if (Math.abs(value - want) > slack + 1e-9) {
    throw new Error(`${key}: ${value}, baseline ${want} (allowed +/-${+slack.toFixed(4)}). If the change is intended, re-run with UPDATE_BASELINE=1 and commit baseline.json.`);
  }
  if (VERBOSE) t.diagnostic(`${key} = ${value} (baseline ${want})`);
}

/**
 * A layer's items, hashed: on one machine and set of dependencies results repeat exactly, so a
 * different hash means something changed the output, even within near()'s tolerances. Reported,
 * not a failure (another GPU or driver differs in the last digits).
 */
export function identical(t: TestContext, key: string, items: unknown) {
  const hash = createHash('sha256').update(JSON.stringify(items)).digest('hex').slice(0, 16);
  measured[`${key}.hash`] = hash;
  const want = pinned[`${key}.hash`];
  if (want !== undefined && want !== hash && !UPDATE) t.diagnostic(`${key}: output differs from the baseline run (within tolerances; UPDATE_BASELINE=1 to accept)`);
}

/** Speed is noisy (power mode, other load): recorded and reported, never a failure. */
export function speed(t: TestContext, key: string, value: number) {
  measured[key] = value;
  const want = pinned[key];
  const pct = typeof want === 'number' && want ? Math.round((100 * value) / want) : undefined;
  t.diagnostic(`${key} = ${value}${pct ? ` (${pct}% of baseline ${want})` : ''}${pct && pct < 70 ? ' SLOWER: on AC power with Best performance?' : ''}`);
}

/** Write the measured figures into baseline.json (merged: other files' keys are kept). */
export function saveBaseline() {
  const missing = Object.keys(measured).some((k) => !(k in pinned));
  if (!UPDATE && !missing) return;
  const now = existsSync(BASELINE) ? JSON.parse(readFileSync(BASELINE, 'utf8')) : {};
  const merged: Pinned = { ...now };
  for (const [k, v] of Object.entries(measured)) if (UPDATE || !(k in now)) merged[k] = v;
  writeFileSync(BASELINE, `${JSON.stringify(Object.fromEntries(Object.entries(merged).sort(([a], [b]) => a.localeCompare(b))), null, 1)}\n`);
}
