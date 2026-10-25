// Copy ONNX Runtime Web's wasm runtime files into public/ort so they are served
// as static assets (ort.env.wasm.wasmPaths = '/ort/').
import { cpSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const src = join('node_modules', 'onnxruntime-web', 'dist');
const dst = join('public', 'ort');
mkdirSync(dst, { recursive: true });
const files = readdirSync(src).filter((f) => /^ort-wasm.*\.(wasm|mjs)$/.test(f));
for (const f of files) cpSync(join(src, f), join(dst, f));
console.error(`copy-ort: ${files.length} files -> ${dst}`);
