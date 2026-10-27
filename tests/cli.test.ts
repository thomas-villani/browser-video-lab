// The shared command library (cli/lib.ts), parts that need no browser. Run: npm test
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { collect, oneShotBrowser, parseTime, UsageError, type Context } from '../cli/lib.ts';

test('parseTime: seconds, mm:ss, h:mm:ss; anything else says what it takes', () => {
  assert.equal(parseTime(83), 83);
  assert.equal(parseTime('83.5'), 83.5);
  assert.equal(parseTime('1:23'), 83);
  assert.equal(parseTime('1:01:23'), 3683);
  assert.throws(() => parseTime('1m23s', 'from'), /from takes seconds, mm:ss or h:mm:ss/);
});

test('collect: files as given, docs stand for their video, folders searched recursively', () => {
  const dir = mkdtempSync(join(tmpdir(), 'videodoc-'));
  mkdirSync(join(dir, 'sub'));
  mkdirSync(join(dir, 'b.mp4.keyframes'));
  for (const f of ['b.mp4', 'b.mp4.videodoc.json', 'a.MOV', 'notes.txt', 'sub/c.webm', 'b.mp4.keyframes/000001.jpg']) writeFileSync(join(dir, f), '');
  const ctx: Context = { browser: oneShotBrowser({ browser: 'msedge', headed: false }), report: { log() {}, progress() {} }, cwd: dir };
  assert.deepEqual(collect(ctx, ['.']), ['a.MOV', 'b.mp4', join('sub', 'c.webm')].map((f) => join(dir, f)));
  assert.deepEqual(collect(ctx, ['b.mp4.videodoc.json', 'b.mp4']), [join(dir, 'b.mp4')]);
  assert.throws(() => collect(ctx, ['missing']), UsageError);
});
