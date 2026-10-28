// Read model outputs back to the CPU.
//
// ORT's tensor.getData() maps each GPU output separately; YuNet has 12 outputs, so that
// is 12 serialized GPU->CPU round trips (~4 ms each on Iris Xe). Copying every output into
// one staging buffer and mapping it once costs a single round trip.
import type * as ort from 'onnxruntime-web/webgpu';

type Outputs = ort.InferenceSession.OnnxValueMapType;

/** Per-tensor readback via ORT (one map per output). */
export async function readEach(outputs: Outputs): Promise<Record<string, Float32Array>> {
  const entries = await Promise.all(
    Object.entries(outputs).map(async ([k, v]) => [k, (await v.getData()) as Float32Array] as const),
  );
  return Object.fromEntries(entries);
}

/** Batched readback: one staging buffer, one mapAsync. Outputs must be GPU-resident float32. */
export class BatchedReadback {
  private staging?: GPUBuffer;

  constructor(private device: GPUDevice) {}

  async read(outputs: Outputs): Promise<Record<string, Float32Array>> {
    // Empty outputs (a detector with NMS inside that found nothing) have nothing to copy.
    const empty = Object.entries(outputs).filter(([, t]) => !t.size).map(([name]) => [name, new Float32Array(0)] as const);
    const items = Object.entries(outputs).filter(([, t]) => t.size).map(([name, t]) => ({ name, buf: t.gpuBuffer, bytes: t.size * 4 }));
    if (!items.length) return Object.fromEntries(empty);
    // copyBufferToBuffer offsets must be 4-byte aligned; float32 sizes always are.
    let total = 0;
    const offsets = items.map((it) => {
      const o = total;
      total += it.bytes;
      return o;
    });
    if (!this.staging || this.staging.size < total) {
      this.staging?.destroy();
      this.staging = this.device.createBuffer({ size: total, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    const enc = this.device.createCommandEncoder();
    items.forEach((it, i) => enc.copyBufferToBuffer(it.buf, 0, this.staging!, offsets[i], it.bytes));
    this.device.queue.submit([enc.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ, 0, total);
    // Copy out before unmapping: the mapped range is detached on unmap.
    const all = new Float32Array(this.staging.getMappedRange(0, total).slice(0));
    this.staging.unmap();
    return Object.fromEntries([...empty, ...items.map((it, i) => [it.name, all.subarray(offsets[i] / 4, (offsets[i] + it.bytes) / 4)] as const)]);
  }
}
