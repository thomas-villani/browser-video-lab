// Decode a file's audio track to mono PCM at a target rate (16 kHz for speech models).
import { ALL_FORMATS, AudioSampleSink, Input, UrlSource } from 'mediabunny';

export interface Pcm {
  samples: Float32Array;
  sampleRate: number;
  /** Media time of samples[0], seconds (audio may not start at 0). */
  t0: number;
  /** Source track's sample rate and channel count, for metadata. */
  source: { sampleRate: number; channels: number; codec: string | null };
}

/** Decode [start, end) seconds of the primary audio track, downmixed and resampled. */
export async function decodeAudio(url: string, opts: { rate?: number; start?: number; end?: number } = {}): Promise<Pcm> {
  const rate = opts.rate ?? 16000;
  const input = new Input({ source: new UrlSource(url), formats: ALL_FORMATS });
  const track = await input.getPrimaryAudioTrack();
  if (!track) throw new Error(`no audio track in ${url}`);
  const sink = new AudioSampleSink(track);

  const chunks: Float32Array[] = [];
  let srcRate = track.sampleRate;
  let t0: number | undefined;
  for await (const s of sink.samples(opts.start, opts.end)) {
    t0 ??= s.timestamp;
    srcRate = s.sampleRate;
    const mono = new Float32Array(s.numberOfFrames);
    const plane = new Float32Array(s.numberOfFrames);
    for (let c = 0; c < s.numberOfChannels; c++) {
      s.copyTo(plane, { planeIndex: c, format: 'f32-planar' });
      for (let i = 0; i < plane.length; i++) mono[i] += plane[i] / s.numberOfChannels;
    }
    chunks.push(mono);
    s.close();
  }
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const all = new Float32Array(total);
  let o = 0;
  for (const c of chunks) {
    all.set(c, o);
    o += c.length;
  }

  // The browser's resampler (band-limited) via an offline audio graph.
  let samples = all;
  if (srcRate !== rate && total) {
    const ctx = new OfflineAudioContext(1, Math.ceil((total * rate) / srcRate), rate);
    const buf = ctx.createBuffer(1, total, srcRate);
    buf.copyToChannel(all, 0);
    const node = ctx.createBufferSource();
    node.buffer = buf;
    node.connect(ctx.destination);
    node.start();
    samples = (await ctx.startRendering()).getChannelData(0);
  }
  return {
    samples,
    sampleRate: rate,
    t0: t0 ?? 0,
    source: { sampleRate: srcRate, channels: track.numberOfChannels, codec: track.codec },
  };
}
