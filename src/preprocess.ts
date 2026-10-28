// Two ways to turn a frame (or a crop of one) into a planar float32 NCHW model input:
//  - GpuPreprocessor: VideoFrame via importExternalTexture (zero-copy), or a retained GPUTexture
//    -> WGSL compute -> GPUBuffer -> ort.Tensor.fromGpuBuffer
//  - CpuPreprocessor: drawImage -> getImageData -> JS loop -> Float32Array (the naive baseline)
import * as ort from 'onnxruntime-web/webgpu';
import type { InputGeometry, ModelSpec } from './models';

/** Either a decoded frame or a frame retained on the GPU (see runtime/ingest.ts). */
export type FrameSource = VideoFrame | GPUTexture;

const shader = (external: boolean) => /* wgsl */ `
struct Params {
  dstW: u32, dstH: u32,
  contentW: u32, contentH: u32,
  scale: f32, bgr: u32, pad: f32, _p1: u32,
  mean: vec4f, stdev: vec4f,
  crop: vec4f, // uv origin (xy) and size (zw) of the region to sample
};
@group(0) @binding(0) var src: ${external ? 'texture_external' : 'texture_2d<f32>'};
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.dstW || id.y >= p.dstH) { return; }
  var rgb = vec3f(p.pad);
  if (id.x < p.contentW && id.y < p.contentH) {
    let uv = p.crop.xy + (vec2f(id.xy) + 0.5) / vec2f(f32(p.contentW), f32(p.contentH)) * p.crop.zw;
    rgb = ${external ? 'textureSampleBaseClampToEdge(src, samp, uv)' : 'textureSampleLevel(src, samp, uv, 0.0)'}.rgb;
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
  run(source: FrameSource, geo: InputGeometry, slot: number): ort.Tensor;
}

function params(spec: ModelSpec, geo: InputGeometry): ArrayBuffer {
  const buf = new ArrayBuffer(80);
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  u.set([geo.inW, geo.inH, geo.contentW, geo.contentH]);
  f[4] = spec.scale;
  u[5] = spec.bgr ? 1 : 0;
  f[6] = spec.pad ?? 0;
  // Mean/std are in RGB order; the shader swaps to BGR after normalizing.
  f.set([...spec.mean, 0], 8);
  f.set([...spec.std, 1], 12);
  const { x, y, w, h } = geo.crop;
  f.set([x / geo.frameW, y / geo.frameH, w / geo.frameW, h / geo.frameH], 16);
  return buf;
}

const isTexture = (s: FrameSource): s is GPUTexture => 'createView' in s;

export class GpuPreprocessor implements Preprocessor {
  readonly kind = 'gpu';
  private external: GPUComputePipeline;
  private texture: GPUComputePipeline;
  private sampler: GPUSampler;
  /** Output buffers keyed by slot and input size (dynamic-size models use several sizes). */
  private outputs = new Map<string, ort.Tensor>();
  private uniforms: GPUBuffer;

  constructor(
    private device: GPUDevice,
    private spec: ModelSpec,
  ) {
    const pipeline = (external: boolean) =>
      device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: shader(external) }), entryPoint: 'main' },
      });
    this.external = pipeline(true);
    this.texture = pipeline(false);
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.uniforms = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  private output(slot: number, w: number, h: number): ort.Tensor {
    const key = `${slot}:${w}x${h}`;
    let t = this.outputs.get(key);
    if (!t) {
      const buf = this.device.createBuffer({
        size: 3 * w * h * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      t = ort.Tensor.fromGpuBuffer(buf, { dataType: 'float32', dims: [1, 3, h, w] });
      this.outputs.set(key, t);
    }
    return t;
  }

  run(source: FrameSource, geo: InputGeometry, slot: number): ort.Tensor {
    // Queue writes are ordered with submits, so one uniform buffer serves every in-flight frame.
    this.device.queue.writeBuffer(this.uniforms, 0, params(this.spec, geo));
    const tensor = this.output(slot, geo.inW, geo.inH);
    const pipeline = isTexture(source) ? this.texture : this.external;
    // External textures are only valid for the current task, so the bind group is rebuilt per frame.
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: isTexture(source) ? source.createView() : this.device.importExternalTexture({ source }),
        },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: tensor.gpuBuffer } },
        { binding: 3, resource: { buffer: this.uniforms } },
      ],
    });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(geo.inW / 16), Math.ceil(geo.inH / 16));
    pass.end();
    this.device.queue.submit([enc.finish()]);
    return tensor;
  }
}

export class CpuPreprocessor implements Preprocessor {
  readonly kind = 'cpu';
  private canvas = new OffscreenCanvas(1, 1);
  private ctx = this.canvas.getContext('2d', { willReadFrequently: true })!;

  constructor(private spec: ModelSpec) {}

  // ORT uploads CPU tensors during run(), so a fresh array per call is safe across slots.
  run(source: FrameSource, geo: InputGeometry, _slot: number): ort.Tensor {
    if (isTexture(source)) throw new Error('CpuPreprocessor needs a VideoFrame');
    const { scale, mean, std, bgr } = this.spec;
    const { inW: W, inH: H, crop } = geo;
    if (this.canvas.width !== W || this.canvas.height !== H) [this.canvas.width, this.canvas.height] = [W, H];
    const g = Math.round((this.spec.pad ?? 0) * 255);
    this.ctx.fillStyle = `rgb(${g},${g},${g})`;
    this.ctx.fillRect(0, 0, W, H);
    this.ctx.drawImage(source, crop.x, crop.y, crop.w, crop.h, 0, 0, geo.contentW, geo.contentH);
    const px = this.ctx.getImageData(0, 0, W, H).data;
    const plane = W * H;
    const data = new Float32Array(3 * plane);
    const order = bgr ? [2, 1, 0] : [0, 1, 2];
    for (let i = 0; i < plane; i++) {
      for (let c = 0; c < 3; c++) {
        const ch = order[c];
        data[c * plane + i] = ((px[4 * i + ch] / 255) * scale - mean[ch]) / std[ch];
      }
    }
    return new ort.Tensor('float32', data, [1, 3, H, W]);
  }
}
