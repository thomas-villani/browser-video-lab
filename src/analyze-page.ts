// The compute page the CLI drives (analyze.html). It exposes window.videodoc and does nothing
// on its own; the driver calls compute()/check() with page.evaluate and receives progress
// through window.__videodocProgress (if the driver exposed it).

import { computeLayers, frameImages, type Progress } from './analyze';
import type { Layers, PlannedLayer, RedactOptions, Track, VideoDoc } from './doc/index.ts';
import { redact } from './redact';
import { decodeAudio } from './runtime/audio';
import { Diarizer } from './runtime/diarize';

declare global {
  interface Window {
    videodoc: typeof api;
    __videodocProgress?: (p: Progress) => void;
  }
}

const VIDEO_CODECS: Record<string, string> = {
  h264: 'avc1.640028',
  hevc: 'hvc1.1.6.L120.90',
  vp9: 'vp09.00.40.08',
  av1: 'av01.0.08M.08',
};
const AUDIO_CODECS: Record<string, AudioDecoderConfig> = {
  aac: { codec: 'mp4a.40.2', sampleRate: 48000, numberOfChannels: 2 },
  opus: { codec: 'opus', sampleRate: 48000, numberOfChannels: 2 },
};

const api = {
  async compute(url: string, planned: PlannedLayer[], inputs?: Partial<Layers>) {
    const status = document.getElementById('status')!;
    return computeLayers(url, planned, {
      inputs,
      onProgress: (p) => {
        status.textContent = `${p.pass}: ${p.stage} ${p.done}/${p.total}`;
        window.__videodocProgress?.(p);
      },
    });
  },

  /** What the speaker models say (windows, voices), before clustering: scripts/eval-speakers.ts
   *  clusters and scores them in Node, so the settings that only affect clustering are tuned
   *  without running the models again. Activity is base64 (0/1 bytes). */
  async speakerInputs(url: string, speech: { t0: number; t1: number }[], c: { segmentation: string; embedding: string; step: number; device: 'webgpu' | 'wasm' }) {
    const pcm = await decodeAudio(url);
    const t0 = performance.now();
    const d = await Diarizer.create(c);
    const r = await d.run(pcm, speech, c.step, (done, total) => window.__videodocProgress?.({ pass: 'audio', stage: 'finding speakers', done, total }));
    const b64 = (a: Uint8Array) => {
      let s = '';
      for (let i = 0; i < a.length; i += 0x8000) s += String.fromCharCode(...a.subarray(i, i + 0x8000));
      return btoa(s);
    };
    return {
      t0: pcm.t0,
      windows: r.windows.map((w) => ({ start: w.start, act: b64(w.act) })),
      embeddings: r.embeddings.map((e) => e.map((v) => (v ? [...v] : null))),
      stats: { ...r.stats, ms: performance.now() - t0 },
    };
  },

  /** JPEGs of the frames at `times`, base64 (page.evaluate results must be JSON). */
  async images(url: string, times: number[], maxSide?: number, quality?: number) {
    const blobs = await frameImages(url, times, { maxSide, quality, onProgress: (p) => window.__videodocProgress?.(p) });
    return Promise.all(
      blobs.map(async (b) => {
        let s = '';
        const bytes = new Uint8Array(await b.arrayBuffer());
        for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
        return btoa(s);
      }),
    );
  },

  /** Redact the video at `url` with the regions and words from `doc`, writing the MP4 to `outUrl` (PUT, by byte offset). */
  async redact(url: string, outUrl: string, doc: VideoDoc, opts: Partial<RedactOptions> & { added?: Track[]; addedAudio?: [number, number][] }) {
    const sink = new WritableStream<{ type: 'write'; data: Uint8Array<ArrayBuffer>; position: number }>({
      async write(chunk) {
        const r = await fetch(`${outUrl}?pos=${chunk.position}`, { method: 'PUT', body: chunk.data });
        if (!r.ok) throw new Error(`writing the output failed (HTTP ${r.status})`);
      },
    });
    const r = await redact(url, doc, sink, { ...opts, onProgress: (p) => window.__videodocProgress?.(p) });
    return { frames: r.frames, audioSeconds: r.audioSeconds, ms: r.ms };
  },

  /** What this browser can do: WebGPU adapter, decoders, isolation (threaded wasm). */
  async check() {
    const adapter = await navigator.gpu?.requestAdapter();
    const decode: Record<string, boolean> = {};
    for (const [k, codec] of Object.entries(VIDEO_CODECS)) {
      decode[k] = !!(await VideoDecoder.isConfigSupported({ codec }).catch(() => ({ supported: false }))).supported;
    }
    for (const [k, c] of Object.entries(AUDIO_CODECS)) {
      decode[k] = !!(await AudioDecoder.isConfigSupported(c).catch(() => ({ supported: false }))).supported;
    }
    return {
      userAgent: navigator.userAgent,
      webgpu: adapter
        ? {
            vendor: adapter.info.vendor,
            architecture: adapter.info.architecture,
            description: adapter.info.description,
            shaderF16: adapter.features.has('shader-f16'),
            maxBufferSize: adapter.limits.maxBufferSize,
          }
        : null,
      decode,
      crossOriginIsolated,
      hardwareConcurrency: navigator.hardwareConcurrency,
    };
  },
};

window.videodoc = api;
document.getElementById('status')!.textContent = 'ready';
