// The MCP server over real stdio, for the tools that need no browser. Run: npm test
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, test } from 'node:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { formatDoc, LAYER_NAMES, newDoc } from '../src/doc/index.ts';

const client = new Client({ name: 'videodoc-test', version: '0' });
const text = (r: Awaited<ReturnType<Client['callTool']>>) => (r.content as { type: string; text: string }[])[0].text;

before(async () => {
  await client.connect(new StdioClientTransport({ command: process.execPath, args: ['cli/mcp.ts'], stderr: 'ignore' }));
});
after(() => client.close());

test('mcp: tools are listed with schemas generated from the layer registry', async () => {
  const { tools } = await client.listTools();
  assert.deepEqual(tools.map((t) => t.name), ['markdown', 'analyze', 'info', 'layers', 'check']);
  const analyze = tools.find((t) => t.name === 'analyze')!;
  assert.deepEqual((analyze.inputSchema.properties!.layers as { items: { enum: string[] } }).items.enum, LAYER_NAMES);
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
});
