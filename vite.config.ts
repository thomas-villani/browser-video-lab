import { appendFileSync, cpSync, existsSync, mkdirSync, readFileSync, renameSync } from 'node:fs';
import { join } from 'node:path';
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

// ONNX Runtime loads its wasm glue with a dynamic import() of public/ort*/…mjs. The dev server
// rewrites that import to add ?import and then refuses it for public files ("should not be
// imported from source code"), so serve those files as-is before its transform middleware.
// The build and `vite preview` (videodoc ui, the CLI) are unaffected.
function ortRuntime(): Plugin {
  return {
    name: 'ort-runtime',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = req.url?.split('?')[0] ?? '';
        if (!/^\/ort(-hf)?\/[\w.-]+\.mjs$/.test(path) || !existsSync(`public${path}`)) return next();
        res.setHeader('Content-Type', 'text/javascript');
        for (const [k, v] of Object.entries(isolation)) res.setHeader(k, v);
        res.end(readFileSync(`public${path}`));
      });
    },
  };
}

// The GitHub Pages build (`npm run build:pages`, i.e. --mode pages; .github/workflows/pages.yml):
// the app alone, as the site's index.html under PAGES_BASE (default '/browser-video-lab/'). Pages
// cannot send the isolation headers, so coi-serviceworker adds them (the page reloads once, the
// first time). Of public/ it ships the runtimes and the default models (scripts/fetch-models.mjs).
function pagesSite(pages: string): Plugin {
  const sw = 'coi-serviceworker.min.js';
  let outDir = 'dist-pages';
  return {
    name: 'pages-site',
    apply: 'build',
    configResolved: (c) => void (outDir = c.build.outDir),
    transformIndexHtml: () => [{ tag: 'script', attrs: { src: `${pages}${sw}` }, injectTo: 'head-prepend' }],
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: sw, source: readFileSync(`node_modules/coi-serviceworker/${sw}`) });
    },
    async writeBundle() {
      renameSync(join(outDir, 'app.html'), join(outDir, 'index.html'));
      const { modelFiles } = await import('./scripts/fetch-models.mjs');
      // Models not fetched (CI builds without them) are left out; pages.yml runs `npm run models` first.
      const missing: string[] = [];
      for (const p of ['favicon.ico', 'ort', 'ort-hf', 'models/LICENSE-yunet', ...modelFiles().map((f: string) => `models/${f}`)]) {
        if (existsSync(join('public', p))) cpSync(join('public', p), join(outDir, p), { recursive: true });
        else missing.push(p);
      }
      if (missing.length) this.warn(`not in public/ (npm run models), left out: ${missing.join(', ')}`);
    },
  };
}

export default defineConfig(({ mode }) => {
  const pages = mode === 'pages' ? (process.env.PAGES_BASE ?? '/browser-video-lab/') : undefined;
  const input: Record<string, string> = pages
    ? { app: 'app.html' }
    : { bench: 'index.html', pipeline: 'pipeline.html', transcribe: 'transcribe.html', label: 'label.html', analyze: 'analyze.html', app: 'app.html' };
  return {
    base: pages ?? '/',
    publicDir: pages ? false : 'public',
    plugins: [ortRuntime(), labelStore(), ...(pages ? [pagesSite(pages)] : [])],
    server: { headers: isolation },
    preview: { headers: isolation },
    optimizeDeps: { exclude: ['onnxruntime-web'] },
    build: {
      outDir: pages ? 'dist-pages' : 'dist',
      target: 'esnext',
      // ORT and transformers.js are large by nature; the warning is noise here.
      chunkSizeWarningLimit: 4096,
      rollupOptions: { input },
    },
  };
});
