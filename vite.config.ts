import { defineConfig } from 'vite';

// Cross-origin isolation enables SharedArrayBuffer, which ORT's threaded wasm backend needs.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  server: { headers: isolation },
  preview: { headers: isolation },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: {
    target: 'esnext',
    rollupOptions: { input: { bench: 'index.html', pipeline: 'pipeline.html', transcribe: 'transcribe.html' } },
  },
});
