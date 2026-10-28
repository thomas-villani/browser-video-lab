// Compute planned layers in the browser: one audio pass (speech, transcript) and one video
// decode shared by every frame consumer (scenes, faces); keyframe images seek to selected frames
// only. Planning, merging and pure layers live in src/doc, so the same code serves the JS API
// (analyze below) and the CLI (which plans in Node).
import { CanvasSink } from 'mediabunny';
import {
  deriveLayers, fingerprint, newDoc, plan, VERSION, withLayers, linkTracks,
  type Box, type Layer, type LayerName, type Layers, type Overrides, type PlannedLayer, type SpeechItem, type VideoDoc,
} from './doc/index.ts';
import { decodeAudio } from './runtime/audio';
import { Detector } from './runtime/detector';
import { sceneStream } from './runtime/scenes';
import { openInput, probe, VideoSource, type MediaInput, type Probe } from './runtime/source';
import { adaptiveStream } from './runtime/stream';
import { Transcriber } from './runtime/transcribe';
import { Vad } from './runtime/vad';
import { groupSpans } from './transcript';

export interface Progress {
  pass: 'probe' | 'audio' | 'frames';
  stage: string;
  done: number;
  total: number;
}

export interface ComputeOptions {
  onProgress?: (p: Progress) => void;
  signal?: AbortSignal;
  /** Layers the computed ones depend on that are already in the doc (e.g. cached speech). */
  inputs?: Partial<Layers>;
}

export interface Computed {
  probe: Probe;
  layers: Partial<Layers>;
  /** Adapter used for the frame pass, for the run summary. */
  adapter?: { vendor: string; architecture: string; description: string };
}

const MODEL_NAMES: Record<string, string> = { yunet: 'yunet_face_dyn (2023mar)' };

/** Compute every layer of `planned` with status 'compute'. Cached ones only supply configs. */
export async function computeLayers(input: MediaInput, planned: PlannedLayer[], opts: ComputeOptions = {}): Promise<Computed> {
  const { onProgress, signal } = opts;
  const want = new Map(planned.filter((l) => l.status === 'compute').map((l) => [l.name, l]));
  const cfg = new Map(planned.map((l) => [l.name, l]));
  onProgress?.({ pass: 'probe', stage: 'reading container', done: 0, total: 1 });
  const info = await probe(input);
  const layers: Partial<Layers> = {};
  let adapter: Computed['adapter'];

  const layer = <T>(name: LayerName, items: T[], ms: number, device: 'webgpu' | 'wasm', coverage: [number, number][], model?: string): Layer<T> => {
    const p = want.get(name)!;
    return {
      producer: { module: name, version: VERSION, ...(model && { model }), config: p.config, configHash: p.configHash },
      coverage,
      createdAt: new Date().toISOString(),
      ms: Math.round(ms),
      device,
      items,
    };
  };

  if ((want.has('speech') || want.has('transcript')) && !info.audio) {
    // No audio track (screen recordings, muted exports): nothing is said, so both layers are empty.
    const all: [number, number][] = [[0, info.duration]];
    if (want.has('speech')) layers.speech = layer('speech', [], 0, 'wasm', all, 'silero_vad v5');
    if (want.has('transcript')) layers.transcript = layer('transcript', [], 0, want.get('transcript')!.config.device as 'webgpu' | 'wasm', all, want.get('transcript')!.config.model as string);
  } else if (want.has('speech') || want.has('transcript')) {
    const t0 = performance.now();
    const step = (stage: string, done = 0, total = 1) => onProgress?.({ pass: 'audio', stage, done, total });
    step('decoding audio');
    const pcm = await decodeAudio(input);
    const span: [number, number] = [pcm.t0, round2(pcm.t0 + pcm.samples.length / pcm.sampleRate)];
    let speech: SpeechItem[] | undefined = opts.inputs?.speech?.items;
    if (want.has('speech')) {
      step('voice activity');
      const c = want.get('speech')!.config as { threshold: number; minSilence: number; minSpeech: number; pad: number };
      const spans = (await (await Vad.create()).speech(pcm, c)).spans;
      speech = spans.map((s) => ({ t0: s.start, t1: s.end }));
      layers.speech = layer('speech', speech, performance.now() - t0, 'wasm', [span], 'silero_vad v5');
    }
    signal?.throwIfAborted();
    if (want.has('transcript')) {
      if (!speech) throw new Error('transcript needs the speech layer');
      const ta = performance.now();
      const c = want.get('transcript')!.config as { model: string; encoder: string; decoder: string; language: string; device: 'webgpu' | 'wasm' };
      const chunks = groupSpans(speech.map((s) => ({ start: s.t0, end: s.t1 })));
      // No speech, no Whisper (it takes seconds to load).
      let segs: { start: number; end: number; text: string }[] = [];
      if (chunks.length) {
        step(`loading ${c.model}`);
        const asr = await Transcriber.create({ model: c.model, device: c.device, dtype: { encoder_model: c.encoder, decoder_model_merged: c.decoder } });
        segs = await asr.transcribeSpans(pcm, chunks, {
          language: c.language,
          onSpan: (i) => {
            signal?.throwIfAborted();
            step('transcribing', i, chunks.length);
          },
        });
      }
      const items = segs.map((s) => ({ t0: s.start, t1: s.end, text: s.text }));
      layers.transcript = layer('transcript', items, performance.now() - ta, c.device, [span], c.model);
    }
  }

  if (want.has('scenes') || want.has('faces')) {
    const t0 = performance.now();
    const source = await VideoSource.open(input);
    const threshold = cfg.get('scenes')!.config.threshold as number;
    const times: number[] = [];
    const scores: number[] = [];
    const boxes: Box[][] = [];
    const tick = (f: number) => {
      if (f % 30 === 0) {
        signal?.throwIfAborted();
        onProgress?.({ pass: 'frames', stage: want.has('faces') ? 'scenes + faces' : 'scenes', done: f, total: info.frames });
      }
    };
    if (want.has('faces')) {
      const c = want.get('faces')!.config as {
        model: string; longSide: number; maxStride: number; minIou: number; minScore: number; refine: boolean; tile: string; every: number; trigger: number; small: number;
      };
      const [tileW, tileH] = c.tile.split('x').map(Number);
      const detector = await Detector.create({ model: c.model, depth: 2, score: Math.min(c.minScore, c.refine ? c.trigger : 1) });
      adapter = detector.adapter;
      for await (const r of adaptiveStream(source, detector, {
        maxStride: c.maxStride, minIou: c.minIou, minScore: c.minScore, cutThreshold: threshold,
        detect: c.longSide ? { longSide: c.longSide } : undefined,
        ingest: { maxSide: 1920 },
        refine: c.refine ? { tileW, tileH, overlap: 32, trigger: c.trigger, grow: 0, every: c.every, small: c.small } : undefined,
      })) {
        times.push(r.t);
        scores.push(r.sceneScore);
        boxes.push(r.boxes);
        tick(r.frame);
      }
    } else {
      const gpu = await navigator.gpu?.requestAdapter();
      if (!gpu) throw new Error('WebGPU unavailable in this browser; use Chrome or Edge 113+');
      const device = await gpu.requestDevice();
      adapter = { vendor: gpu.info.vendor, architecture: gpu.info.architecture, description: gpu.info.description };
      try {
        for await (const r of sceneStream(source, device)) {
          times.push(r.t);
          scores.push(r.score);
          tick(r.frame);
        }
      } finally {
        // Also when cancelled: an app that cancels and starts again must not pile up devices.
        device.destroy();
      }
    }
    const ms = performance.now() - t0;
    const n = times.length;
    const end = n ? round2(Math.max(info.duration, times[n - 1])) : info.duration;
    const cuts = scores.flatMap((s, f) => (f > 0 && threshold > 0 && s >= threshold ? [f] : []));
    if (want.has('scenes')) {
      const starts = [0, ...cuts];
      const items = starts.map((f0, i) => {
        const f1 = (starts[i + 1] ?? n) - 1;
        return { t0: round3(times[f0]), t1: round3(starts[i + 1] !== undefined ? times[starts[i + 1]] : end), f0, f1, score: round3(f0 ? scores[f0] : 0) };
      });
      layers.scenes = layer('scenes', n ? items : [], ms, 'webgpu', [[0, end]]);
    }
    if (want.has('faces')) {
      const model = want.get('faces')!.config.model as string;
      layers.faces = layer('faces', linkTracks(boxes, cuts), ms, 'webgpu', [[0, end]], MODEL_NAMES[model] ?? model);
    }
    // The decoded frame count is the truth when the container's packet count disagrees.
    info.frames = n;
  }
  return { probe: info, layers, adapter };
}

const round2 = (v: number) => Math.round(v * 100) / 100;
const round3 = (v: number) => Math.round(v * 1000) / 1000;

/** Fingerprint a File/Blob, or a URL whose server supports range requests. */
export async function fingerprintOf(input: MediaInput): Promise<{ size: number; fingerprint: string }> {
  if (typeof input !== 'string') {
    const read = async (a: number, b: number) => new Uint8Array(await input.slice(a, b).arrayBuffer());
    return { size: input.size, fingerprint: await fingerprint(input.size, read) };
  }
  const read = async (a: number, b: number) => {
    const r = await fetch(input, { headers: { Range: `bytes=${a}-${b - 1}` } });
    if (r.status !== 206) throw new Error(`${input}: the server ignores range requests (status ${r.status}); pass the file as a Blob instead`);
    return new Uint8Array(await r.arrayBuffer());
  };
  const head = await fetch(input, { headers: { Range: 'bytes=0-0' } });
  const size = Number(head.headers.get('Content-Range')?.split('/')[1]);
  await head.body?.cancel();
  if (!Number.isFinite(size)) throw new Error(`${input}: no Content-Range in the response; the server must support range requests`);
  return { size, fingerprint: await fingerprint(size, read) };
}

export interface AnalyzeOptions extends Omit<ComputeOptions, 'inputs'> {
  /** A previously saved doc for this video: layers with a matching config are reused. */
  doc?: VideoDoc;
  settings?: Overrides;
  force?: boolean;
}

/**
 * The happy path: analyze a video in the browser and return its doc.
 *
 *   const doc = await analyze(file, ['transcript', 'scenes', 'faces'], { onProgress });
 */
export async function analyze(input: MediaInput, layers: LayerName[], opts: AnalyzeOptions = {}): Promise<VideoDoc> {
  const id = await fingerprintOf(input);
  let doc = opts.doc?.source.fingerprint === id.fingerprint ? opts.doc : undefined;
  const p = await plan(doc, layers, opts.settings, opts.force);
  if (doc && !Object.keys(p.passes).length) return doc;
  if (!doc || p.passes.audio || p.passes.frames) {
    const r = await computeLayers(input, p.layers, { ...opts, inputs: doc?.layers });
    const name = typeof input === 'string' ? decodeURIComponent(input.split('/').pop() ?? input) : 'name' in input ? (input as File).name : 'video';
    doc ??= newDoc({ name, ...id, ...r.probe });
    doc = withLayers({ ...doc, source: { ...doc.source, ...r.probe } }, r.layers);
  }
  return withLayers(doc, deriveLayers(doc, p.layers));
}

export interface ImageOptions {
  /** Long side of the images, px (never upscaled). */
  maxSide?: number;
  /** JPEG quality, 0-1. */
  quality?: number;
  signal?: AbortSignal;
  onProgress?: (p: Progress) => void;
}

/**
 * JPEGs of the frames shown at `times` (seconds, ascending), e.g. a doc's keyframes. Seeks to
 * each frame and decodes from its preceding key frame only, never the whole video.
 */
export async function frameImages(input: MediaInput, times: number[], opts: ImageOptions = {}): Promise<Blob[]> {
  const { maxSide = 768, quality = 0.85, signal, onProgress } = opts;
  const track = await openInput(input).getPrimaryVideoTrack();
  if (!track) throw new Error('no video track');
  const { displayWidth: w, displayHeight: h } = track;
  const scale = Math.min(1, maxSide / Math.max(w, h));
  const sink = new CanvasSink(track, { width: Math.round(w * scale), height: Math.round(h * scale), fit: 'fill', poolSize: 2 });
  const out: Blob[] = [];
  for await (const c of sink.canvasesAtTimestamps(times)) {
    signal?.throwIfAborted();
    const t = times[out.length];
    if (!c) throw new Error(`no frame at ${t} s (before the first frame?)`);
    const cv = c.canvas;
    out.push(
      cv instanceof HTMLCanvasElement
        ? await new Promise<Blob>((ok, fail) => cv.toBlob((b) => (b ? ok(b) : fail(new Error('JPEG encode failed'))), 'image/jpeg', quality))
        : await cv.convertToBlob({ type: 'image/jpeg', quality }),
    );
    onProgress?.({ pass: 'frames', stage: 'keyframe images', done: out.length, total: times.length });
  }
  return out;
}
