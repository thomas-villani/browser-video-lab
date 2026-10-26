// Transcript page: decode a file's audio, run Whisper (WebGPU), report timed segments and
// speech coverage. Configured by URL params; result in window.__BENCH_RESULT__.
//
//   ?video=/media/briefing-full.mp4  &model=onnx-community/whisper-base
//   &encoder=fp32  &decoder=q4  &device=webgpu|wasm  &lang=en  &seconds=0 (0 = all)  &mingap=2
//   &vad=1 (Silero VAD: transcribe speech spans only, and measure coverage from VAD)
import { decodeAudio } from './runtime/audio';
import { Transcriber } from './runtime/transcribe';
import { groupSpans, speechCoverage } from './transcript';
import { Vad } from './runtime/vad';

declare global {
  interface Window {
    __BENCH_RESULT__?: unknown;
  }
}

const q = new URLSearchParams(location.search);
const cfg = {
  video: q.get('video') ?? '/media/briefing-full.mp4',
  model: q.get('model') ?? 'onnx-community/whisper-base',
  encoder: q.get('encoder') ?? 'fp32',
  decoder: q.get('decoder') ?? 'q4',
  device: (q.get('device') ?? 'webgpu') as 'webgpu' | 'wasm',
  lang: q.get('lang') ?? 'en',
  seconds: Number(q.get('seconds') ?? 0),
  minGap: Number(q.get('mingap') ?? 2),
  vad: q.get('vad') !== '0',
};
const r1 = (v: number) => Math.round(v * 10) / 10;

async function run() {
  const status = document.getElementById('status')!;
  status.textContent = 'decoding audio...';
  const t0 = performance.now();
  const pcm = await decodeAudio(cfg.video, { end: cfg.seconds || undefined });
  const t1 = performance.now();
  status.textContent = `loading ${cfg.model}...`;
  const asr = await Transcriber.create({
    model: cfg.model,
    device: cfg.device,
    dtype: { encoder_model: cfg.encoder, decoder_model_merged: cfg.decoder },
  });
  const t2 = performance.now();
  let speech = null;
  let chunks = null;
  let vadMs = 0;
  if (cfg.vad) {
    status.textContent = 'voice activity...';
    const tv = performance.now();
    speech = (await (await Vad.create()).speech(pcm)).spans;
    chunks = groupSpans(speech);
    vadMs = performance.now() - tv;
  }
  const t3a = performance.now();
  status.textContent = 'transcribing...';
  const segments = chunks
    ? await asr.transcribeSpans(pcm, chunks, { language: cfg.lang, onSpan: (i) => (status.textContent = `transcribing chunk ${i + 1}/${chunks!.length}`) })
    : await asr.transcribe(pcm, { language: cfg.lang });
  const t3 = performance.now();
  const audioSeconds = pcm.samples.length / pcm.sampleRate;
  return {
    config: cfg,
    env: { userAgent: navigator.userAgent, crossOriginIsolated },
    audio: { seconds: r1(audioSeconds), t0: pcm.t0, source: pcm.source },
    decodeMs: r1(t1 - t0),
    loadMs: r1(t2 - t1),
    vadMs: r1(vadMs),
    asrMs: r1(t3 - t3a),
    // Audio seconds per wall-clock second of VAD + ASR (>1 = faster than real time).
    speed: r1(audioSeconds / ((t3 - t2) / 1000)),
    chunks: chunks?.length,
    words: segments.reduce((n, s) => n + s.text.split(/\s+/).filter(Boolean).length, 0),
    // From VAD when enabled (Whisper's segment times stretch across silences).
    coverage: speechCoverage(speech ?? segments, pcm.t0 + audioSeconds, cfg.minGap),
    segments,
    speech,
  };
}

const out = document.getElementById('result')!;
run()
  .then((r) => {
    window.__BENCH_RESULT__ = r;
    out.textContent = JSON.stringify(r, null, 2);
    document.getElementById('status')!.textContent = `done: ${r.speed}x real time`;
  })
  .catch((e: unknown) => {
    const error = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
    window.__BENCH_RESULT__ = { config: cfg, error };
    out.textContent = error;
    document.getElementById('status')!.textContent = 'failed';
  });
