// Copy ONNX Runtime Web's wasm runtime files into public/ so they are served as static assets:
//  - public/ort/:    our onnxruntime-web (ort.env.wasm.wasmPaths = '/ort/')
//  - public/ort-hf/: the onnxruntime-web build bundled with transformers.js (a different
//                    version; its asyncify build serves both WebGPU and wasm)
import { cpSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const copies = [
  [join('node_modules', 'onnxruntime-web', 'dist'), join('public', 'ort'), /^ort-wasm.*\.(wasm|mjs)$/],
  [join('node_modules', '@huggingface', 'transformers', 'node_modules', 'onnxruntime-web', 'dist'), join('public', 'ort-hf'), /^ort-wasm-simd-threaded\.asyncify\.(wasm|mjs)$/],
];
for (const [src, dst, re] of copies) {
  if (!existsSync(src)) continue;
  mkdirSync(dst, { recursive: true });
  const files = readdirSync(src).filter((f) => re.test(f));
  for (const f of files) cpSync(join(src, f), join(dst, f));
  console.error(`copy-ort: ${files.length} files -> ${dst}`);
}
