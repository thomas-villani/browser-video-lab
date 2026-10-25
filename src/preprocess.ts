// Two ways to turn a VideoFrame into a planar float32 NCHW model input:
//  - GpuPreprocessor: importExternalTexture (zero-copy) -> WGSL compute -> GPUBuffer -> ort.Tensor.fromGpuBuffer
//  - CpuPreprocessor: drawImage -> getImageData -> JS loop -> Float32Array (the naive baseline)
import * as ort from 'onnxruntime-web/webgpu';
import type { InputGeometry, ModelSpec } from './models';

const SHADER = /* wgsl */ `
struct Params {
  dstW: u32, dstH: u32,
  contentW: u32, contentH: u32,
  scale: f32, bgr: u32, _p0: u32, _p1: u32,
  mean: vec4f, stdev: vec4f,
};
@group(0) @binding(0) var src: texture_external;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.dstW || id.y >= p.dstH) { return; }
  var rgb = vec3f(0.0);
  if (id.x < p.contentW && id.y < p.contentH) {
    let uv = (vec2f(id.xy) + 0.5) / vec2f(f32(p.contentW), f32(p.contentH));
    rgb = textureSampleBaseClampToEdge(src, samp, uv).rgb;
  }
  var c = (rgb * p.scale - p.mean.xyz) / p.stdev.xyz;
  if (p.bgr == 1u) { c = c.zyx; }
  let plane = p.dstW * p.dstH;
  let i = id.y * p.dstW + id.x;
  dst[i] = c.x;
  dst[plane + i] = c.y;
  dst[2u * plane + i] = c.z;
}`;

export interface Preprocessor {
  readonly kind: 'gpu' | 'cpu';
  /** `slot` selects a ring buffer so several frames can be in flight at once. */
  run(frame: VideoFrame, geo: InputGeometry, slot: number): ort.Tensor;
}

function params(spec: ModelSpec, geo: InputGeometry): ArrayBuffer {
  const buf = new ArrayBuffer(64);
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  u.set([spec.width, spec.height, geo.contentW, geo.contentH]);
  f[4] = spec.scale;
  u[5] = spec.bgr ? 1 : 0;
  // Mean/std are in RGB order; the shader swaps to BGR after normalizing.
  f.set([...spec.mean, 0], 8);
  f.set([...spec.std, 1], 12);
  return buf;
}

export class GpuPreprocessor implements Preprocessor {
  readonly kind = 'gpu';
  private pipeline: GPUComputePipeline;
  private sampler: GPUSampler;
  private outputs: GPUBuffer[];
  private tensors: ort.Tensor[];
  private uniforms: GPUBuffer;
  private lastGeo?: InputGeometry;

  constructor(private device: GPUDevice, private spec: ModelSpec, slots = 1) {
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: 'main' },
    });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const bytes = 3 * spec.width * spec.height * 4;
    this.outputs = Array.from({ length: slots }, () =>
      device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
    );
    this.uniforms = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.tensors = this.outputs.map((buf) =>
      ort.Tensor.fromGpuBuffer(buf, { dataType: 'float32', dims: [1, 3, spec.height, spec.width] }),
    );
  }

  run(frame: VideoFrame, geo: InputGeometry, slot: number): ort.Tensor {
    if (this.lastGeo?.contentW !== geo.contentW || this.lastGeo?.contentH !== geo.contentH) {
      this.device.queue.writeBuffer(this.uniforms, 0, params(this.spec, geo));
      this.lastGeo = geo;
    }
    // External textures are only valid for the current task, so the bind group is rebuilt per frame.
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.device.importExternalTexture({ source: frame }) },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.outputs[slot] } },
        { binding: 3, resource: { buffer: this.uniforms } },
      ],
    });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.spec.width / 16), Math.ceil(this.spec.height / 16));
    pass.end();
    this.device.queue.submit([enc.finish()]);
    return this.tensors[slot];
  }
}

export class CpuPreprocessor implements Preprocessor {
  readonly kind = 'cpu';
  private canvas: OffscreenCanvas;
  private ctx: OffscreenCanvasRenderingContext2D;
  private data: Float32Array;

  constructor(private spec: ModelSpec) {
    this.canvas = new OffscreenCanvas(spec.width, spec.height);
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })!;
    this.data = new Float32Array(3 * spec.width * spec.height);
  }

  // ORT uploads CPU tensors during run(), so one scratch array serves every slot.
  run(frame: VideoFrame, geo: InputGeometry, _slot: number): ort.Tensor {
    const { width: W, height: H, scale, mean, std, bgr } = this.spec;
    this.ctx.fillStyle = 'black';
    this.ctx.fillRect(0, 0, W, H);
    this.ctx.drawImage(frame, 0, 0, geo.contentW, geo.contentH);
    const px = this.ctx.getImageData(0, 0, W, H).data;
    const plane = W * H;
    const order = bgr ? [2, 1, 0] : [0, 1, 2];
    for (let i = 0; i < plane; i++) {
      for (let c = 0; c < 3; c++) {
        const ch = order[c];
        this.data[c * plane + i] = ((px[4 * i + ch] / 255) * scale - mean[ch]) / std[ch];
      }
    }
    return new ort.Tensor('float32', this.data, [1, 3, H, W]);
  }
}
