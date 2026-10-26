// Voice activity detection with Silero VAD v5 (MIT, ~2 MB) on onnxruntime-web's wasm backend:
// 32 ms windows at 16 kHz, recurrent state carried across windows.
import * as ort from 'onnxruntime-web/webgpu';
import { DEFAULT_VAD, speechSpans, type Span, type VadOptions } from '../transcript';
import type { Pcm } from './audio';

ort.env.wasm.wasmPaths = '/ort/';

const WINDOW = 512;
const CONTEXT = 64;

export class Vad {
  private constructor(private session: ort.InferenceSession) {}

  static async create(url = '/models/silero_vad.onnx'): Promise<Vad> {
    return new Vad(await ort.InferenceSession.create(url, { executionProviders: ['wasm'] }));
  }

  /** Speech probability per 32 ms window. */
  async probabilities(pcm: Pcm): Promise<Float32Array> {
    if (pcm.sampleRate !== 16000) throw new Error('Silero VAD needs 16 kHz audio');
    const n = Math.floor(pcm.samples.length / WINDOW);
    const probs = new Float32Array(n);
    const sr = new ort.Tensor('int64', BigInt64Array.from([16000n]), []);
    let state: ort.Tensor = new ort.Tensor('float32', new Float32Array(2 * 128), [2, 1, 128]);
    const input = new Float32Array(CONTEXT + WINDOW);
    for (let i = 0; i < n; i++) {
      // Each window is prefixed with the previous window's last 64 samples.
      input.copyWithin(0, WINDOW, WINDOW + CONTEXT);
      input.set(pcm.samples.subarray(i * WINDOW, (i + 1) * WINDOW), CONTEXT);
      const out = await this.session.run({ input: new ort.Tensor('float32', input, [1, CONTEXT + WINDOW]), state, sr });
      probs[i] = (out.output.data as Float32Array)[0];
      state = out.stateN;
    }
    return probs;
  }

  async speech(pcm: Pcm, opts: VadOptions = DEFAULT_VAD): Promise<{ spans: Span[]; probs: Float32Array }> {
    const probs = await this.probabilities(pcm);
    return { spans: speechSpans(probs, WINDOW / pcm.sampleRate, pcm.t0, opts), probs };
  }
}
