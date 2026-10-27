// Scene scores for every frame without a detector: GPU thumbnails read back once per chunk.
// (With a detector, adaptiveStream computes the same scores in its own pass.)
import { SceneScorer } from '../scene';
import { FrameIngest } from './ingest';
import type { VideoSource } from './source';

export interface SceneFrame {
  frame: number;
  t: number;
  score: number;
}

export async function* sceneStream(source: VideoSource, device: GPUDevice, chunk = 32): AsyncGenerator<SceneFrame> {
  // Only thumbnails are needed; keep the retained textures tiny.
  const ingest = new FrameIngest(device, source.width, source.height, chunk, { maxSide: 64 });
  const scorer = new SceneScorer();
  let pending: { frame: number; t: number }[] = [];
  async function* flush() {
    const thumbs = await ingest.thumbs(pending.map((_, i) => i));
    for (const [i, p] of pending.entries()) yield { ...p, score: scorer.push(thumbs[i]) };
    pending = [];
  }
  try {
    for await (const { index, t, frame } of source.frames()) {
      ingest.put(frame, pending.length);
      pending.push({ frame: index, t });
      if (pending.length === chunk) yield* flush();
    }
    if (pending.length) yield* flush();
  } finally {
    ingest.destroy();
  }
}
