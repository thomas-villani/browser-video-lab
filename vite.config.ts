import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { defineConfig, type Plugin } from 'vite';

// Cross-origin isolation enables SharedArrayBuffer, which ORT's threaded wasm backend needs.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

// Dev-server-only label store for label.html: POST appends one JSON line per saved frame to
// results/labels/faces.jsonl (append-only, so nothing is ever lost; the last line per
// clip/frame wins); GET returns the current label for every frame.
const LABELS = 'results/labels/faces.jsonl';
function labelStore(): Plugin {
  return {
    name: 'label-store',
    configureServer(server) {
      server.middlewares.use('/api/labels', (req, res) => {
        if (req.method === 'POST') {
          let body = '';
          req.on('data', (c) => (body += c));
          req.on('end', () => {
            const label = JSON.parse(body);
            mkdirSync('results/labels', { recursive: true });
            appendFileSync(LABELS, JSON.stringify({ ...label, savedAt: new Date().toISOString() }) + '\n');
            res.end('{"ok":true}');
          });
          return;
        }
        const latest: Record<string, unknown> = {};
        if (existsSync(LABELS)) {
          for (const line of readFileSync(LABELS, 'utf8').split('\n').filter(Boolean)) {
            const l = JSON.parse(line);
            latest[`${l.clip}/${l.frame}`] = l;
          }
        }
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(latest));
      });
    },
  };
}

export default defineConfig({
  plugins: [labelStore()],
  server: { headers: isolation },
  preview: { headers: isolation },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: {
    target: 'esnext',
    // ORT and transformers.js are large by nature; the warning is noise here.
    chunkSizeWarningLimit: 4096,
    rollupOptions: {
      input: { bench: 'index.html', pipeline: 'pipeline.html', transcribe: 'transcribe.html', label: 'label.html', analyze: 'analyze.html' },
    },
  },
});
