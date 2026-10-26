// Per-frame GPU ingest, run once on every decoded frame:
//  - retain: copy the frame into a ring of textures, so a scheduler can detect on any frame of
//    the current window after decoding it (decode once, detect anywhere in the window);
//  - thumb: an area-averaged RGB thumbnail (default 64 px wide) for scene-change scoring.
// Thumbnails stay on the GPU until a window's worth is read back with a single map.

const SHADER = /* wgsl */ `
struct Params { outW: u32, outH: u32, thumbW: u32, thumbH: u32, thumbOffset: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var src: texture_external;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read_write> thumbs: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn retain(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.outW || id.y >= p.outH) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(f32(p.outW), f32(p.outH));
  textureStore(dst, id.xy, vec4f(textureSampleBaseClampToEdge(src, samp, uv).rgb, 1.0));
}

// 4x4 bilinear taps per thumbnail pixel approximate an area average, which keeps noise and
// fine texture from dominating the frame difference.
@compute @workgroup_size(8, 8)
fn thumb(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.thumbW || id.y >= p.thumbH) { return; }
  var acc = vec3f(0.0);
  for (var j = 0u; j < 4u; j++) {
    for (var i = 0u; i < 4u; i++) {
      let uv = (vec2f(id.xy) + (vec2f(f32(i), f32(j)) + 0.5) / 4.0) / vec2f(f32(p.thumbW), f32(p.thumbH));
      acc += textureSampleBaseClampToEdge(src, samp, uv).rgb;
    }
  }
  let k = p.thumbOffset + (id.y * p.thumbW + id.x) * 3u;
  acc = acc / 16.0;
  thumbs[k] = acc.x;
  thumbs[k + 1u] = acc.y;
  thumbs[k + 2u] = acc.z;
}`;

export interface IngestOptions {
  /** Cap on the retained frames' longer side (memory: capacity x w x h x 4 bytes). */
  maxSide?: number;
  thumbWidth?: number;
}

export class FrameIngest {
  readonly width: number;
  readonly height: number;
  readonly thumbW: number;
  readonly thumbH: number;
  private textures: GPUTexture[];
  private retainPipe: GPUComputePipeline;
  private thumbPipe: GPUComputePipeline;
  private sampler: GPUSampler;
  private thumbBuf: GPUBuffer;
  private staging: GPUBuffer;
  private uniforms: GPUBuffer[];

  constructor(
    private device: GPUDevice,
    frameW: number,
    frameH: number,
    readonly capacity: number,
    opts: IngestOptions = {},
  ) {
    const s = Math.min(1, (opts.maxSide ?? 1920) / Math.max(frameW, frameH));
    [this.width, this.height] = [Math.round(frameW * s), Math.round(frameH * s)];
    this.thumbW = opts.thumbWidth ?? 64;
    this.thumbH = Math.max(1, Math.round((this.thumbW * frameH) / frameW));
    this.textures = Array.from({ length: capacity }, () =>
      device.createTexture({
        size: [this.width, this.height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      }),
    );
    const module = device.createShaderModule({ code: SHADER });
    this.retainPipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'retain' } });
    this.thumbPipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'thumb' } });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const thumbBytes = capacity * this.thumbFloats * 4;
    this.thumbBuf = device.createBuffer({ size: thumbBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.staging = device.createBuffer({ size: thumbBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    // One uniform buffer per slot: only the thumbnail offset differs, so they are written once.
    this.uniforms = Array.from({ length: capacity }, (_, slot) => {
      const buf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buf, 0, new Uint32Array([this.width, this.height, this.thumbW, this.thumbH, slot * this.thumbFloats, 0, 0, 0]));
      return buf;
    });
  }

  private get thumbFloats() {
    return this.thumbW * this.thumbH * 3;
  }

  /** Retain `frame` in `slot` and compute its thumbnail. The frame may be closed afterwards. */
  put(frame: VideoFrame, slot: number): void {
    const src = this.device.importExternalTexture({ source: frame });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.retainPipe);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: this.retainPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: src },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: this.textures[slot].createView() },
          { binding: 4, resource: { buffer: this.uniforms[slot] } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(this.width / 16), Math.ceil(this.height / 16));
    pass.setPipeline(this.thumbPipe);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: this.thumbPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: src },
          { binding: 1, resource: this.sampler },
          { binding: 3, resource: { buffer: this.thumbBuf } },
          { binding: 4, resource: { buffer: this.uniforms[slot] } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(this.thumbW / 8), Math.ceil(this.thumbH / 8));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  texture(slot: number): GPUTexture {
    return this.textures[slot];
  }

  /** Read back the thumbnails of `slots` (RGB floats in [0, 1], row-major) with one map. */
  async thumbs(slots: number[]): Promise<Float32Array[]> {
    if (!slots.length) return [];
    const bytes = this.thumbFloats * 4;
    const enc = this.device.createCommandEncoder();
    slots.forEach((slot, i) => enc.copyBufferToBuffer(this.thumbBuf, slot * bytes, this.staging, i * bytes, bytes));
    this.device.queue.submit([enc.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ, 0, slots.length * bytes);
    const all = new Float32Array(this.staging.getMappedRange(0, slots.length * bytes).slice(0));
    this.staging.unmap();
    return slots.map((_, i) => all.subarray(i * this.thumbFloats, (i + 1) * this.thumbFloats));
  }

  destroy(): void {
    for (const t of this.textures) t.destroy();
    this.thumbBuf.destroy();
    this.staging.destroy();
  }
}
