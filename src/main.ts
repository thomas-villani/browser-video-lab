// Benchmark page. Configured via URL query params so humans (click a link) and
// agents (scripts/bench.mjs) drive it the same way. The result is a JSON object
// rendered into <pre id="result"> and exposed as window.__BENCH_RESULT__.
//
//   ?model=yunet|rfdetr|rfdetr16  &ep=webgpu|wasm  &path=gpu|cpu|decode
//   &frames=300  &stride=1  &video=/media/pan1080.mp4  &outloc=cpu|gpu|batch  &capture=0|1  &depth=1
//   &longside=640 (dynamic-size models: input scaled to this long side, e.g. 640x384 for 16:9)

import { ALL_FORMATS, Input, UrlSource, VideoSampleSink } from 'mediabunny';
import * as ort from 'onnxruntime-web/webgpu';
import { type Detection, geometryFor, MODELS } from './models';
import { CpuPreprocessor, GpuPreprocessor, type Preprocessor } from './preprocess';
import { BatchedReadback, readEach } from './readback';

declare global {
  interface Window {
    __BENCH_RESULT__?: unknown;
  }
}

ort.env.wasm.wasmPaths = '/ort/';
ort.env.logLevel = 'error';

interface Config {
  model: string;
  ep: 'webgpu' | 'wasm';
  path: 'gpu' | 'cpu' | 'decode';
  frames: number;
  stride: number;
  video: string;
  /**
   * Where session outputs land and how they get to the CPU:
   * 'cpu' (ORT reads back inside run), 'gpu' (per-tensor getData), 'batch' (one staging buffer, one map).
   */
  outloc: 'cpu' | 'gpu' | 'batch';
  /** ORT WebGPU graph capture: replay recorded GPU commands (needs static shapes + GPU IO). */
  capture: boolean;
  /** Frames in flight (1 = fully sequential). */
  depth: number;
  /** Include every processed frame's detections in the result (for offline tracker eval). */
  dump: boolean;
  /** Override the model's detection score threshold (lower keeps weak boxes for ByteTrack). */
  score?: number;
  /** Dynamic-size models: scale the frame so its longer side is this many pixels. */
  longside?: number;
}

/** Per-frame detections in the compact dump format: box = [x, y, w, h, score]. */
export interface FrameDetections {
  frame: number;
  t: number;
  boxes: number[][];
}

function readConfig(): Config {
  const q = new URLSearchParams(location.search);
  return {
    model: q.get('model') ?? 'yunet',
    ep: (q.get('ep') as Config['ep']) ?? 'webgpu',
    path: (q.get('path') as Config['path']) ?? 'gpu',
    // frames=0 means the whole video.
    frames: Number(q.get('frames') ?? 300) || Infinity,
    stride: Number(q.get('stride') ?? 1),
    video: q.get('video') ?? '/media/pan1080.mp4',
    outloc: (q.get('outloc') as Config['outloc']) ?? 'cpu',
    capture: q.get('capture') === '1',
    depth: Math.max(1, Number(q.get('depth') ?? 1)),
    dump: q.get('dump') === '1',
    score: q.has('score') ? Number(q.get('score')) : undefined,
    longside: q.has('longside') ? Number(q.get('longside')) : undefined,
  };
}

function stats(xs: number[]) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const pct = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const r = (v: number) => Math.round(v * 100) / 100;
  return { mean: r(mean), p50: r(pct(0.5)), p95: r(pct(0.95)), n: xs.length };
}

function drawDetections(frame: VideoFrame, dets: Detection[]) {
  const canvas = document.getElementById('preview') as HTMLCanvasElement;
  canvas.width = frame.displayWidth;
  canvas.height = frame.displayHeight;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(frame, 0, 0);
  ctx.lineWidth = 4;
  ctx.font = '28px sans-serif';
  for (const d of dets) {
    ctx.strokeStyle = ctx.fillStyle = '#00e676';
    ctx.strokeRect(d.x, d.y, d.w, d.h);
    ctx.fillText(`${d.label} ${d.score.toFixed(2)}`, d.x, Math.max(28, d.y - 6));
  }
}

async function run(cfg: Config) {
  const status = document.getElementById('status')!;
  const log = (m: string) => (status.textContent = m);
  const spec = MODELS[cfg.model];
  if (!spec) throw new Error(`unknown model '${cfg.model}' (have: ${Object.keys(MODELS).join(', ')})`);
  if (cfg.path === 'gpu' && cfg.ep !== 'webgpu') throw new Error('path=gpu requires ep=webgpu');

  let session: ort.InferenceSession | undefined;
  let pre: Preprocessor | undefined;
  let sessionMs = 0;
  let adapter: Record<string, string> | undefined;
  let readers: BatchedReadback[] = [];

  if (cfg.path !== 'decode') {
    log(`loading ${spec.url} on ${cfg.ep}...`);
    const t0 = performance.now();
    session = await ort.InferenceSession.create(spec.url, {
      executionProviders: [cfg.ep],
      graphOptimizationLevel: 'all',
      preferredOutputLocation: cfg.outloc === 'cpu' ? 'cpu' : 'gpu-buffer',
      enableGraphCapture: cfg.capture,
    });
    sessionMs = performance.now() - t0;
    if (cfg.ep === 'webgpu') {
      const device = await ort.env.webgpu.device;
      const info = device.adapterInfo;
      adapter = { vendor: info.vendor, architecture: info.architecture, description: info.description };
      pre = cfg.path === 'gpu' ? new GpuPreprocessor(device, spec) : new CpuPreprocessor(spec);
      if (cfg.outloc === 'batch') readers = Array.from({ length: cfg.depth }, () => new BatchedReadback(device));
    } else {
      pre = new CpuPreprocessor(spec);
    }
  }

  const input = new Input({ source: new UrlSource(cfg.video), formats: ALL_FORMATS });
  const track = await input.getPrimaryVideoTrack();
  if (!track) throw new Error('no video track');
  const sink = new VideoSampleSink(track);

  const WARMUP = 5;
  const t = {
    decodeWait: [] as number[],
    pre: [] as number[],
    infer: [] as number[],
    readback: [] as number[],
    post: [] as number[],
  };
  let processed = 0;
  let decoded = 0;
  let totalDetections = 0;
  let firstFrameDetections: Detection[] = [];
  let steadyStart = 0;
  const wallStart = performance.now();

  // Up to `depth` frames are in flight: frame N+1 is preprocessed and submitted while
  // frame N's outputs are still being computed / read back. Each in-flight frame owns a
  // ring slot (input buffer + staging buffer), so slots are never reused while busy.
  const inflight: Promise<void>[] = [];
  const dumped: FrameDetections[] = [];

  let tWait = performance.now();
  for await (const sample of sink.samples()) {
    const waited = performance.now() - tWait;
    const frame = sample.toVideoFrame();
    try {
      if (decoded === WARMUP) steadyStart = performance.now();
      const steady = decoded >= WARMUP;
      if (steady) t.decodeWait.push(waited);
      if (session && pre && decoded % cfg.stride === 0) {
        const slot = processed % cfg.depth;
        const frameIndex = decoded;
        const timestamp = sample.timestamp;
        const geo = geometryFor(spec, frame.displayWidth, frame.displayHeight, { longSide: cfg.longside });
        const t1 = performance.now();
        const tensor = pre.run(frame, geo, slot);
        const t2 = performance.now();
        const outputs = await session.run({ [spec.inputName]: tensor });
        const t3 = performance.now();
        const job = (async () => {
          const arrays =
            cfg.outloc === 'cpu'
              ? Object.fromEntries(Object.entries(outputs).map(([k, v]) => [k, v.data as Float32Array]))
              : readers.length
                ? await readers[slot].read(outputs)
                : await readEach(outputs);
          // GPU-resident outputs are fresh buffers each run; release them.
          if (cfg.outloc !== 'cpu') for (const v of Object.values(outputs)) v.dispose();
          const t4 = performance.now();
          const dets = spec.decode(arrays, geo, cfg.score);
          const t5 = performance.now();
          if (cfg.dump) {
            dumped.push({
              frame: frameIndex,
              t: Math.round(timestamp * 1000) / 1000,
              boxes: dets.map((d) => [d.x, d.y, d.w, d.h].map((v) => Math.round(v * 10) / 10).concat(Math.round(d.score * 1000) / 1000)),
            });
          }
          if (steady) {
            t.pre.push(t2 - t1);
            t.infer.push(t3 - t2);
            // With depth > 1 this is latency (submit -> data on CPU), not per-frame cost.
            t.readback.push(t4 - t3);
            t.post.push(t5 - t4);
          }
          if (!firstFrameDetections.length) firstFrameDetections = dets;
          totalDetections += dets.length;
        })();
        if (decoded === 0) {
          await job;
          drawDetections(frame, firstFrameDetections);
        } else {
          inflight.push(job);
          if (inflight.length >= cfg.depth) await inflight.shift();
        }
        processed++;
      }
    } finally {
      frame.close();
      sample.close();
    }
    decoded++;
    if (decoded % 30 === 0) log(`frame ${decoded}/${cfg.frames}`);
    if (decoded >= cfg.frames) break;
    tWait = performance.now();
  }
  await Promise.all(inflight);

  const end = performance.now();
  const steadyFrames = decoded - WARMUP;
  const round = (v: number) => Math.round(v * 10) / 10;
  return {
    config: cfg,
    env: { userAgent: navigator.userAgent, adapter, crossOriginIsolated },
    video: { codec: await track.getCodecParameterString(), width: track.displayWidth, height: track.displayHeight },
    sessionMs: round(sessionMs),
    decodedFrames: decoded,
    processedFrames: processed,
    // Throughput in *video* frames/sec (with stride>1, skipped frames still count: they were decoded).
    fps: round((steadyFrames * 1000) / (end - steadyStart)),
    wallMs: round(end - wallStart),
    stageMs: Object.fromEntries(Object.entries(t).map(([k, v]) => [k, stats(v)])),
    totalDetections,
    firstFrameDetections: firstFrameDetections.map((d) => ({
      label: d.label,
      score: Math.round(d.score * 1000) / 1000,
      box: [d.x, d.y, d.w, d.h].map(Math.round),
    })),
    ...(cfg.dump && {
      fps_video: await track.computePacketStats(100).then((s) => s.averagePacketRate),
      frameDetections: dumped.sort((x, y) => x.frame - y.frame),
    }),
  };
}

const cfg = readConfig();
const out = document.getElementById('result')!;
run(cfg)
  .then((r) => {
    window.__BENCH_RESULT__ = r;
    // Dumps can be megabytes; show the summary only.
    out.textContent = JSON.stringify({ ...r, frameDetections: undefined }, null, 2);
    document.getElementById('status')!.textContent = `done: ${r.fps} fps`;
  })
  .catch((e: unknown) => {
    const error = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
    window.__BENCH_RESULT__ = { config: cfg, error };
    out.textContent = error;
    document.getElementById('status')!.textContent = 'failed';
  });
