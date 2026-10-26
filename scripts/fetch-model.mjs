// Download a Hugging Face model repo's config/tokenizer files plus selected ONNX weights into
// public/models/<repo>/, so pages load models locally (transformers.js env.localModelPath)
// instead of re-fetching them in every fresh headless browser profile.
//
//   node scripts/fetch-model.mjs onnx-community/whisper-base encoder_model decoder_model_merged_q4
//
// The trailing names are ONNX files under onnx/ (without .onnx; external .onnx_data is fetched
// when listed in the repo). stdout: JSON summary { repo, files, bytes }.
import { createWriteStream, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const [repo, ...onnx] = process.argv.slice(2);
if (!repo) {
  console.error('usage: node scripts/fetch-model.mjs <org/repo> [onnx file stems...]');
  process.exit(2);
}
const tree = await (await fetch(`https://huggingface.co/api/models/${repo}/tree/main?recursive=1`)).json();
const wanted = tree
  .filter((f) => f.type === 'file')
  .map((f) => f.path)
  .filter((p) => (!p.includes('/') && p.endsWith('.json')) || onnx.some((s) => p === `onnx/${s}.onnx` || p.startsWith(`onnx/${s}.onnx_data`)));

let bytes = 0;
for (const path of wanted) {
  const dest = `public/models/${repo}/${path}`;
  if (existsSync(dest)) {
    bytes += statSync(dest).size;
    continue;
  }
  mkdirSync(dirname(dest), { recursive: true });
  console.error(`fetching ${path}...`);
  const res = await fetch(`https://huggingface.co/${repo}/resolve/main/${path}`);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  await pipeline(Readable.fromWeb(res.body), createWriteStream(dest));
  bytes += statSync(dest).size;
}
const missing = onnx.filter((s) => !wanted.includes(`onnx/${s}.onnx`));
if (missing.length) console.error(`warning: not in repo: ${missing.join(', ')}`);
console.log(JSON.stringify({ repo, files: wanted, bytes }));
