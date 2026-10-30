// Speaker diarization in the browser: pyannote segmentation-3.0 over 10 s windows (wasm, it is
// small and recurrent) and a WeSpeaker ResNet34 voice embedding per window and local speaker
// (WebGPU), from Kaldi-style fbank features of the frames where that speaker talks alone. The
// clustering that turns these into speakers is pure: src/audio/diarize.ts. Models load from
// public/models/<repo>/onnx/ (scripts/fetch-model.mjs).
import * as ort from 'onnxruntime-web/webgpu';
import { WeSpeakerFeatureExtractor } from '@huggingface/transformers';
import { decodePowerset, embeddingMask, SEG, windowStarts, type WindowSeg } from '../audio/diarize.ts';
import type { Pcm } from './audio';
import { asset } from './asset';

ort.env.wasm.wasmPaths = asset('/ort/');

export interface DiarizerOptions {
  /** Hugging Face repo ids under public/models. */
  segmentation: string;
  embedding: string;
  device: 'webgpu' | 'wasm';
}

/** The embedding model's fbank front end (its preprocessor_config.json). */
const FBANK = { feature_extractor_type: 'WeSpeakerFeatureExtractor', sampling_rate: 16000, num_mel_bins: 80, min_num_frames: 9, fbank_centering_span: null };
/** Fewer fbank frames than this make no embedding (the model's minimum input). */
const MIN_FBANK = 9;
/** Segmentation frames a speaker must have alone for only those to be used (pyannote: the
 *  embedding model's minimum input, in segmentation frames). */
const MIN_CLEAN = Math.ceil((SEG.frames * MIN_FBANK * 160) / SEG.window);
const BATCH = 16;

/** A local speaker's frames to embed: window w, speaker s, frames x 80 fbank values. */
interface Request {
  w: number;
  s: number;
  x: Float32Array;
}

export interface DiarizerRun {
  windows: WindowSeg[];
  /** Per window, per local speaker: the voice, or undefined (silent, or too little audio). */
  embeddings: (Float32Array | undefined)[][];
  stats: { windows: number; skipped: number; embeddings: number; segMs: number; embMs: number; fbankMs: number };
}

export class Diarizer {
  private constructor(
    private seg: ort.InferenceSession,
    private emb: ort.InferenceSession,
    private fx: WeSpeakerFeatureExtractor,
  ) {}

  static async create(o: DiarizerOptions): Promise<Diarizer> {
    const seg = await ort.InferenceSession.create(asset(`/models/${o.segmentation}/onnx/model.onnx`), { executionProviders: ['wasm'] });
    const emb = await ort.InferenceSession.create(asset(`/models/${o.embedding}/onnx/model.onnx`), { executionProviders: [o.device], graphOptimizationLevel: 'all' });
    return new Diarizer(seg, emb, new WeSpeakerFeatureExtractor(FBANK as never));
  }

  /**
   * Segment and embed `pcm` in windows every `step` seconds. Windows without any `speech` (VAD
   * spans, media time) are skipped: nobody speaks in them.
   */
  async run(pcm: Pcm, speech: { t0: number; t1: number }[], step: number, onProgress?: (done: number, total: number) => void, signal?: AbortSignal): Promise<DiarizerRun> {
    if (pcm.sampleRate !== SEG.rate) throw new Error('diarization needs 16 kHz audio');
    const all = windowStarts(pcm.samples.length / SEG.rate, step);
    const starts = all.filter((s) => speech.some((p) => p.t1 > pcm.t0 + s && p.t0 < pcm.t0 + s + SEG.window / SEG.rate));
    const windows: WindowSeg[] = [];
    const embeddings: (Float32Array | undefined)[][] = [];
    const stats = { windows: starts.length, skipped: all.length - starts.length, embeddings: 0, segMs: 0, embMs: 0, fbankMs: 0 };
    const audio = (start: number) => {
      const a = Math.round(start * SEG.rate);
      return pcm.samples.subarray(a, Math.min(pcm.samples.length, a + SEG.window));
    };
    for (let b = 0; b < starts.length; b += BATCH) {
      signal?.throwIfAborted();
      const batch = starts.slice(b, b + BATCH);
      const t0 = performance.now();
      const input = new Float32Array(batch.length * SEG.window);
      batch.forEach((s, i) => input.set(audio(s), i * SEG.window));
      const out = await this.seg.run({ input_values: new ort.Tensor('float32', input, [batch.length, 1, SEG.window]) });
      const logits = out.logits.data as Float32Array;
      const frames = out.logits.dims[1];
      stats.segMs += performance.now() - t0;
      const todo: Request[] = [];
      for (const [i, start] of batch.entries()) {
        const act = decodePowerset(logits.subarray(i * frames * 7, (i + 1) * frames * 7), frames);
        windows.push({ start, act });
        embeddings.push([undefined, undefined, undefined]);
        todo.push(...(await this.features(audio(start), act, embeddings.length - 1, stats)));
      }
      await this.embed(todo, embeddings, stats);
      onProgress?.(Math.min(starts.length, b + BATCH), starts.length);
    }
    return { windows, embeddings, stats };
  }

  /** Each local speaker's frames in one window, to embed. */
  private async features(samples: Float32Array, act: Uint8Array, w: number, stats: DiarizerRun['stats']): Promise<Request[]> {
    const masks = [0, 1, 2].map((s) => embeddingMask(act, s, MIN_CLEAN));
    if (!masks.some(Boolean) || samples.length < 400) return [];
    const t0 = performance.now();
    // Features of the whole window (mean-normalized over it, as pyannote does), then each
    // speaker's frames: fbank frame f is centered on sample 160 f + 200.
    const { input_features: feats } = await this.fx(samples);
    stats.fbankMs += performance.now() - t0;
    const data = feats.data as Float32Array;
    const n = feats.dims[1];
    const frames = act.length / SEG.speakers;
    const out: Request[] = [];
    masks.forEach((mask, s) => {
      if (!mask) return;
      const keep: number[] = [];
      for (let f = 0; f < n; f++) if (mask[Math.min(frames - 1, Math.max(0, Math.round((160 * f + 200 - SEG.size / 2) / SEG.step)))]) keep.push(f);
      if (keep.length < MIN_FBANK) return;
      const x = new Float32Array(keep.length * 80);
      keep.forEach((f, i) => x.set(data.subarray(f * 80, f * 80 + 80), i * 80));
      out.push({ w, s, x });
    });
    return out;
  }

  /** Run the voice model on each request. (Batching equal-length inputs was no faster on an
   *  integrated GPU, and fp16 weights do not load: NOTES.md.) */
  private async embed(todo: Request[], into: (Float32Array | undefined)[][], stats: DiarizerRun['stats']) {
    const t0 = performance.now();
    for (const r of todo) {
      const out = await this.emb.run({ input_features: new ort.Tensor('float32', r.x, [1, r.x.length / 80, 80]) });
      into[r.w][r.s] = new Float32Array(out.last_hidden_state.data as Float32Array);
      stats.embeddings++;
    }
    stats.embMs += performance.now() - t0;
  }
}
