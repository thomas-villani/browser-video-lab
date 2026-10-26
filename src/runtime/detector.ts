// A reusable object detector: preprocess (GPU) -> ORT WebGPU -> batched readback -> decode.
//
//   const det = await Detector.create({ model: 'yunet' });
//   const boxes = await det.detect(frameOrTexture, { crop, longSide: 320 });
//
// Calls may overlap: session.run is serialized (ORT runs one at a time per session), while
// readback and decoding of up to `depth` calls proceed concurrently.
import * as ort from 'onnxruntime-web/webgpu';
import { geometryFor, MODELS, type Detection, type GeometryOptions, type ModelSpec } from '../models';
import { GpuPreprocessor, type FrameSource } from '../preprocess';
import { BatchedReadback } from '../readback';

ort.env.wasm.wasmPaths = '/ort/';
ort.env.logLevel = 'error';

export interface DetectorOptions {
  model: string | ModelSpec;
  /** Calls in flight (ring slots). */
  depth?: number;
  /** Override the model's score threshold. */
  score?: number;
}

/** Cumulative cost counters, so callers can report what a strategy spent. */
export interface DetectorStats {
  calls: number;
  /** Total model-input pixels: a cost proxy that accounts for ROI and reduced-size calls. */
  inputPixels: number;
  /** Wall-clock ms spent submitting (preprocess + session.run), and from submit to decoded result. */
  submitMs: number;
  resultMs: number;
}

export class Detector {
  readonly stats: DetectorStats = { calls: 0, inputPixels: 0, submitMs: 0, resultMs: 0 };
  private next = 0;
  private busy: Promise<void>[];
  private runChain: Promise<unknown> = Promise.resolve();

  private constructor(
    readonly spec: ModelSpec,
    readonly device: GPUDevice,
    private session: ort.InferenceSession,
    private pre: GpuPreprocessor,
    private readers: BatchedReadback[],
    private score?: number,
  ) {
    this.busy = readers.map(() => Promise.resolve());
  }

  static async create(opts: DetectorOptions): Promise<Detector> {
    const spec = typeof opts.model === 'string' ? MODELS[opts.model] : opts.model;
    if (!spec) throw new Error(`unknown model '${opts.model}' (have: ${Object.keys(MODELS).join(', ')})`);
    const session = await ort.InferenceSession.create(spec.url, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
      preferredOutputLocation: 'gpu-buffer',
    });
    const device = await ort.env.webgpu.device;
    const depth = Math.max(1, opts.depth ?? 2);
    const readers = Array.from({ length: depth }, () => new BatchedReadback(device));
    return new Detector(spec, device, session, new GpuPreprocessor(device, spec), readers, opts.score);
  }

  /**
   * Queue a detection. Resolves once the GPU work is submitted — a VideoFrame may be closed
   * then — with `result` resolving to the detections in source-frame pixels.
   */
  async submit(source: FrameSource, opts: GeometryOptions = {}): Promise<{ result: Promise<Detection[]> }> {
    const slot = this.next++ % this.readers.length;
    // Claim the slot synchronously so concurrent callers queue behind each other in order.
    const previous = this.busy[slot];
    let release!: () => void;
    this.busy[slot] = new Promise((r) => (release = r));
    await previous;

    const [w, h] = 'displayWidth' in source ? [source.displayWidth, source.displayHeight] : [source.width, source.height];
    const geo = geometryFor(this.spec, w, h, opts);
    const t0 = performance.now();
    const run = this.runChain.then(() =>
      this.session.run({ [this.spec.inputName]: this.pre.run(source, geo, slot) }),
    );
    this.runChain = run.catch(() => undefined);
    let outputs: ort.InferenceSession.OnnxValueMapType;
    try {
      outputs = await run;
    } catch (e) {
      release();
      throw e;
    }
    const t1 = performance.now();
    this.stats.calls++;
    this.stats.inputPixels += geo.inW * geo.inH;
    this.stats.submitMs += t1 - t0;

    const result = (async () => {
      try {
        const arrays = await this.readers[slot].read(outputs);
        return this.spec.decode(arrays, geo, this.score);
      } finally {
        // GPU-resident outputs are fresh buffers each run; release them.
        for (const v of Object.values(outputs)) v.dispose();
        this.stats.resultMs += performance.now() - t1;
        release();
      }
    })();
    return { result };
  }

  async detect(source: FrameSource, opts?: GeometryOptions): Promise<Detection[]> {
    return (await this.submit(source, opts)).result;
  }

  /** Adapter info for result metadata. */
  get adapter() {
    const i = this.device.adapterInfo;
    return { vendor: i.vendor, architecture: i.architecture, description: i.description };
  }
}
