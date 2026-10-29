// Speech-to-text with Whisper via transformers.js (its own ONNX Runtime build, WebGPU).
// Models load from public/models/<repo>/ (fetch with scripts/fetch-model.mjs), never remotely.
import { env, pipeline, type AutomaticSpeechRecognitionPipeline } from '@huggingface/transformers';
import { dropRepeats, type Span } from '../transcript';
import type { Pcm } from './audio';

env.allowRemoteModels = false;
env.allowLocalModels = true; // off by default in browsers
env.localModelPath = '/models/';
env.backends.onnx.wasm!.wasmPaths = {
  mjs: '/ort-hf/ort-wasm-simd-threaded.asyncify.mjs',
  wasm: '/ort-hf/ort-wasm-simd-threaded.asyncify.wasm',
};

export interface Segment {
  /** Seconds on the media timeline. */
  start: number;
  end: number;
  text: string;
  /** With word timestamps: the segment's words. */
  words?: Segment[];
}

export interface TranscriberOptions {
  /** Hugging Face repo id under public/models, e.g. onnx-community/whisper-base. */
  model: string;
  device?: 'webgpu' | 'wasm';
  /** Per-file ONNX dtypes, e.g. { encoder_model: 'fp32', decoder_model_merged: 'q4' }. */
  dtype?: Record<string, string>;
}

export class Transcriber {
  private constructor(private asr: AutomaticSpeechRecognitionPipeline) {}

  static async create(opts: TranscriberOptions): Promise<Transcriber> {
    const asr = await pipeline('automatic-speech-recognition', opts.model, {
      device: opts.device ?? 'webgpu',
      dtype: (opts.dtype ?? { encoder_model: 'fp32', decoder_model_merged: 'q4' }) as never,
    });
    return new Transcriber(asr as AutomaticSpeechRecognitionPipeline);
  }

  /** Whole-file transcription in overlapping 30 s windows (Whisper sees the silences too). */
  async transcribe(pcm: Pcm, opts: { language?: string; chunkSeconds?: number; strideSeconds?: number } = {}): Promise<Segment[]> {
    const out = await this.asr(pcm.samples, {
      language: opts.language ?? 'en',
      task: 'transcribe',
      chunk_length_s: opts.chunkSeconds ?? 30,
      stride_length_s: opts.strideSeconds ?? 5,
      return_timestamps: true,
    });
    return dropRepeats(segments(out, pcm.t0, pcm.t0 + pcm.samples.length / pcm.sampleRate));
  }

  /**
   * Transcribe only the given spans (each at most 30 s; see groupSpans), one model call each.
   * Silence never reaches Whisper, which removes its repetition loops on silent stretches.
   */
  async transcribeSpans(pcm: Pcm, spans: Span[], opts: { language?: string; words?: boolean; onSpan?: (i: number) => void } = {}): Promise<Segment[]> {
    const all: Segment[] = [];
    for (const [i, span] of spans.entries()) {
      opts.onSpan?.(i);
      const a = Math.max(0, Math.round((span.start - pcm.t0) * pcm.sampleRate));
      const b = Math.min(pcm.samples.length, Math.round((span.end - pcm.t0) * pcm.sampleRate));
      if (b <= a) continue;
      const out = await this.asr(pcm.samples.subarray(a, b), { language: opts.language ?? 'en', task: 'transcribe', return_timestamps: opts.words ? 'word' : true });
      all.push(...(opts.words ? wordSegments(segments(out, span.start, span.end)) : segments(out, span.start, span.end)));
    }
    return dropRepeats(all);
  }
}

/**
 * Word timestamps (a model exported with cross-attentions, e.g. whisper-base_timestamped) give
 * words only; segments are rebuilt from them: a segment ends at a sentence end or a pause.
 */
function wordSegments(words: Segment[]): Segment[] {
  const out: Segment[] = [];
  let cur: Segment[] = [];
  const flush = () => {
    if (cur.length) out.push({ start: cur[0].start, end: cur[cur.length - 1].end, text: cur.map((w) => w.text).join(' '), words: cur });
    cur = [];
  };
  for (const w of words) {
    if (!w.text) continue;
    const prev = cur[cur.length - 1];
    if (prev && (w.start - prev.end >= 1 || cur.length >= 40)) flush();
    cur.push(w);
    if (/[.?!]["')\]]?$/.test(w.text) && cur.length >= 4) flush();
  }
  flush();
  return out;
}

type AsrOutput = Awaited<ReturnType<AutomaticSpeechRecognitionPipeline>>;

/** Pipeline output → segments shifted to the media timeline. */
function segments(out: AsrOutput, offset: number, end: number): Segment[] {
  const chunks = (Array.isArray(out) ? out[0] : out).chunks ?? [];
  return chunks.map((c: { timestamp: [number | null, number | null]; text: string }) => ({
    start: round(Math.min(end, offset + (c.timestamp[0] ?? 0))),
    // Whisper leaves the last segment open-ended when audio stops mid-sentence; a last word's
    // end can also run past the audio it was given.
    end: round(c.timestamp[1] === null ? end : Math.min(end, offset + c.timestamp[1])),
    text: c.text.trim(),
  }));
}

const round = (v: number) => Math.round(v * 100) / 100;
