// The MCP server over real stdio, for the tools that need no browser. Run: npm test
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, test } from 'node:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { formatDoc, LAYER_NAMES, newDoc, withLayers, type Layer } from '../src/doc/index.ts';

const client = new Client({ name: 'videodoc-test', version: '0' });
const text = (r: Awaited<ReturnType<Client['callTool']>>) => (r.content as { type: string; text: string }[])[0].text;

before(async () => {
  await client.connect(new StdioClientTransport({ command: process.execPath, args: ['cli/mcp.ts'], stderr: 'ignore' }));
});
after(() => client.close());

test('mcp: tools are listed with schemas generated from the layer registry', async () => {
  const { tools } = await client.listTools();
  assert.deepEqual(tools.map((t) => t.name), ['markdown', 'search', 'speakers', 'analyze', 'redact', 'info', 'layers', 'check']);
  const analyze = tools.find((t) => t.name === 'analyze')!;
  const layersDoc = (analyze.inputSchema.properties!.layers as { description: string }).description;
  for (const n of LAYER_NAMES) assert.ok(layersDoc.includes(`\n${n}: `), `${n} described`);
  assert.ok(analyze.inputSchema.properties!.pipeline, 'analyze takes a pipeline');
  assert.match((analyze.inputSchema.properties!.set as { description: string }).description, /faces\.every=2/);
  assert.match(client.getInstructions() ?? '', /call markdown/);
});

test('mcp: layers and info answer without a browser', async () => {
  const layers = JSON.parse(text(await client.callTool({ name: 'layers', arguments: {} })));
  assert.deepEqual(layers.keyframes.needs, ['scenes', 'speech']);

  const dir = mkdtempSync(join(tmpdir(), 'videodoc-'));
  const path = join(dir, 'clip.mp4.videodoc.json');
  writeFileSync(path, formatDoc(newDoc({
    name: 'clip.mp4', fingerprint: 'v0:1:x', size: 1, duration: 10, width: 640, height: 360, fps: 25, frames: 250, video: { codec: 'avc1' },
  })));
  const info = JSON.parse(text(await client.callTool({ name: 'info', arguments: { video: path } })));
  assert.equal(info.source.name, 'clip.mp4');
  assert.deepEqual(info.missing, LAYER_NAMES);
});

test('mcp: mistakes come back as tool errors that say how to fix them', async () => {
  const missing = await client.callTool({ name: 'markdown', arguments: { video: join(tmpdir(), 'no-such-video.mp4') } });
  assert.equal(missing.isError, true);
  assert.match(text(missing), /no such file/);
  const setting = await client.callTool({ name: 'analyze', arguments: { video: 'package.json', set: ['faces.evry=4'], dryRun: true } });
  assert.equal(setting.isError, true);
  assert.match(text(setting), /faces\.every/);
  const style = await client.callTool({ name: 'redact', arguments: { video: 'package.json', style: 'smudge' } });
  assert.equal(style.isError, true);
  assert.match(text(style), /style takes blur, pixelate, fill/);
  const over = await client.callTool({ name: 'redact', arguments: { video: 'package.json', out: 'package.json' } });
  assert.match(text(over), /overwrite the video/);
});

test('mcp: search finds passages across a folder and lists videos not analyzed yet', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'videodoc-'));
  for (const name of ['a.mp4', 'b.mp4']) writeFileSync(join(dir, name), '');
  const doc = withLayers(newDoc({ name: 'a.mp4', fingerprint: 'v0:1:x', size: 1, duration: 60, width: 640, height: 360, fps: 25, frames: 1500, video: { codec: 'avc1' } }), {
    transcript: { producer: { configHash: 'h', config: {} }, items: [{ t0: 61, t1: 65, text: 'The zoning board approved it.' }] } as unknown as Layer<never>,
  });
  writeFileSync(join(dir, 'a.mp4.videodoc.json'), formatDoc(doc));
  const r = JSON.parse(text(await client.callTool({ name: 'search', arguments: { query: 'zoning board', paths: [dir] } })));
  assert.equal(r.hits.length, 1);
  assert.equal(r.hits[0].time, '01:01–01:05');
  assert.match(r.hits[0].snippet, /\*\*zoning\*\* \*\*board\*\*/);
  assert.deepEqual(r.notAnalyzed, [join(dir, 'b.mp4')]);
});

test('mcp: speakers are named and merged into <video>.speakers.json; other videos get suggestions; search and info use the names', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'videodoc-'));
  const voice = (k: number) => Array.from({ length: 8 }, (_, i) => (i === k ? 1 : 0));
  for (const [name, fp, ids] of [['a.mp4', 'v0:1:a', [0, 2, 0.5]], ['b.mp4', 'v0:1:b', [2, 4]]] as const) {
    writeFileSync(join(dir, name), '');
    const doc = withLayers(newDoc({ name, fingerprint: fp, size: 1, duration: 30, width: 640, height: 360, fps: 25, frames: 750, video: { codec: 'avc1' } }), {
      transcript: { producer: { configHash: 'h', config: {} }, items: [{ t0: 0, t1: 9, text: 'The zoning board met.' }, { t0: 10, t1: 19, text: 'The zoning board voted.' }] } as unknown as Layer<never>,
      speakers: {
        producer: { configHash: 's', config: {} },
        items: ids.map((k, i) => ({ id: i + 1, turns: [[i * 10, i * 10 + 9.5]], voice: Number.isInteger(k) ? voice(k) : [0.8, 0.6, 0, 0, 0, 0, 0, 0] })),
      } as unknown as Layer<never>,
    });
    writeFileSync(join(dir, `${name}.videodoc.json`), formatDoc(doc));
  }
  const call = async (args: Record<string, unknown>) => JSON.parse(text(await client.callTool({ name: 'speakers', arguments: { video: join(dir, 'a.mp4'), ...args } })));
  let r = await call({ name: { 1: 'Ann', 2: 'Dana' }, merge: { 3: 'Ann' } });
  assert.equal(r.changed, true);
  assert.equal(r.edits, join(dir, 'a.mp4.speakers.json'));
  assert.deepEqual(r.speakers.map((s: { name: string; merged?: number[] }) => [s.name, s.merged]), [['Ann', [3]], ['Dana', undefined]]);
  r = await call({ move: [{ t0: 12, t1: 14, to: 'Ann', from: 'Dana' }] });
  assert.deepEqual(r.moves, [{ index: 0, t0: 12, t1: 14, from: 'Dana', to: 'Ann' }]);
  r = await call({ name: { 1: 'Dana' } });
  assert.equal(r.kind, 'usage');
  assert.match(r.error, /merge them instead/);
  // b.mp4's first voice is Dana's: suggested, not applied.
  const b = JSON.parse(text(await client.callTool({ name: 'speakers', arguments: { video: join(dir, 'b.mp4') } })));
  assert.deepEqual(b.speakers.map((s: { name: string; suggestion?: { name: string } }) => [s.name, s.suggestion?.name]), [['Speaker 1', 'Dana'], ['Speaker 2', undefined]]);
  assert.equal(b.changed, false);
  const hits = JSON.parse(text(await client.callTool({ name: 'search', arguments: { query: 'zoning board', paths: [dir], speaker: 'Dana' } })));
  assert.deepEqual(hits.hits.map((h: { time: string; speaker: string }) => [h.time, h.speaker]), [['00:10–00:19', 'Dana']]);
  assert.deepEqual(hits.noSpeaker, [join(dir, 'b.mp4')]);
  const info = JSON.parse(text(await client.callTool({ name: 'info', arguments: { video: join(dir, 'a.mp4') } })));
  assert.equal(info.layers.speakers.named, 2);
  r = await call({ reset: true });
  assert.equal(r.edits, undefined);
});
