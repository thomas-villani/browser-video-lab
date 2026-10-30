// Redaction, the browser half: decode the video, cover the regions the faces layer gives for
// each frame (RedactionCursor, src/doc/redact.ts), encode H.264 into MP4. Frames without regions
// pass through untouched. Audio is copied as-is when MP4 can hold it, or, with words to bleep
// (audioSpans), decoded, rewritten over those spans and encoded again. With no layer to
// cover, the video is copied and only the audio changes.
import { AudioSample, BufferTarget, Conversion, Mp4OutputFormat, Output, QUALITY_HIGH, Quality, StreamTarget, type StreamTargetChunk, type VideoSample } from 'mediabunny';
import { fingerprintOf, type Progress } from './analyze';
import {
  audioCover,
  audioSpans,
  type Box,
  covered,
  coversAudio,
  cursorFor,
  REDACT_DEFAULTS,
  type RedactOptions,
  type RedactSound,
  type RedactStyle,
  type Track,
  type VideoDoc,
} from './doc/index.ts';
import { type MediaInput, openInput } from './runtime/source';

export interface RedactRun extends Partial<RedactOptions> {
  /** Faces added by hand in a review (RedactPlan.added). */
  added?: Track[];
  /** Audio spans to bleep added by hand (RedactPlan.addedAudio). */
  addedAudio?: [number, number][];
  onProgress?: (p: Progress) => void;
  signal?: AbortSignal;
}

export interface Redacted {
  /** Frames encoded; must equal doc.source.frames for the regions to line up. 0 when the video
   *  was copied (no layer covered). */
  frames: number;
  /** Seconds of audio bleeped or muted. */
  audioSeconds: number;
  ms: number;
  /** The MP4, when no stream was given. */
  buffer?: ArrayBuffer;
}

/**
 * Redact `input` with the regions from `doc` (which must be this video's doc, with a faces
 * layer). Writes to `stream` (e.g. a FileSystemWritableFileStream) or returns the MP4 bytes.
 */
export async function redact(input: MediaInput, doc: VideoDoc, stream?: WritableStream<StreamTargetChunk>, o: RedactRun = {}): Promise<Redacted> {
  const { added, addedAudio, onProgress, signal, ...given } = o;
  const opts: RedactOptions = { ...REDACT_DEFAULTS, ...Object.fromEntries(Object.entries(given).filter(([, v]) => v !== undefined)) } as RedactOptions;
  const id = await fingerprintOf(input);
  if (id.fingerprint !== doc.source.fingerprint) throw new Error(`the doc is for a different file (${doc.source.name}); analyze this video first`);
  const cursor = cursorFor(doc, opts, added);
  const video = covered(doc, opts).length > 0 || (added?.length ?? 0) > 0;
  const spans = coversAudio(opts, addedAudio) ? audioCover(audioSpans(doc, opts, addedAudio)) : [];
  if (!video && !spans.length) throw new Error('nothing to redact: no layer to cover and no words to bleep');
  const { width, height } = doc.source;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d')!;
  const cover = coverer(canvas, opts.style);
  let frame = 0;
  const t0 = performance.now();

  // Re-encode at the source's bitrate plus a quarter (one more generation of compression);
  // covered regions are flat and cost little.
  const media = openInput(input);
  const bitrate = (await (await media.getPrimaryVideoTrack())?.computePacketStats(300))?.averageBitrate;
  const quality = bitrate ? new Quality({ bitrate: Math.round(Math.min(Math.max(bitrate * 1.25, 1e6), 50e6)) }) : QUALITY_HIGH;

  const target = stream ? new StreamTarget(stream, { chunked: true }) : new BufferTarget();
  const output = new Output({ format: new Mp4OutputFormat({ fastStart: stream ? false : 'in-memory' }), target });
  const conversion = await Conversion.init({
    input: media,
    output,
    tracks: 'primary',
    video: video
      ? {
          codec: 'avc',
          quality,
          forceTranscode: true,
          process: (sample: VideoSample) => {
            const regions = cursor.at(frame++);
            if (!regions.length) return sample;
            sample.draw(ctx, 0, 0, width, height);
            cover(regions);
            return canvas;
          },
          processedWidth: width,
          processedHeight: height,
        }
      : undefined,
    audio: spans.length ? { bitrate: QUALITY_HIGH, forceTranscode: true, process: soundOver(spans, opts.sound) } : undefined,
  });
  if (!conversion.isValid) {
    const why = conversion.discardedTracks.map((d) => `${d.track.type}: ${d.reason}`).join('; ');
    throw new Error(`cannot write this video as MP4 here (${why})`);
  }
  conversion.onProgress = (p) =>
    onProgress?.(
      video
        ? { pass: 'frames', stage: 'redacting', done: frame, total: Math.max(frame, doc.source.frames) }
        : { pass: 'audio', stage: 'redacting audio', done: Math.round(p * 100), total: 100 },
    );
  const onAbort = () => void conversion.cancel();
  signal?.addEventListener('abort', onAbort, { once: true });
  try {
    await conversion.execute();
  } finally {
    signal?.removeEventListener('abort', onAbort);
  }
  const audioSeconds = Math.round(spans.reduce((n, [a, b]) => n + b - a, 0) * 100) / 100;
  return { frames: frame, audioSeconds, ms: Math.round(performance.now() - t0), ...(target instanceof BufferTarget && { buffer: target.buffer! }) };
}

/** Fade in and out over this long at each span's edges (a hard cut clicks). */
const FADE = 0.008;
/** The bleep: a 1 kHz sine at -10 dBFS, the broadcast convention. */
const TONE_HZ = 1000;
const TONE_GAIN = 0.32;

/**
 * Returns the audio `process` hook: samples overlapping `spans` (sorted, merged; seconds) are
 * crossfaded to the tone or to silence; others pass through. Samples arrive in time order,
 * so a cursor walks the spans.
 */
export function soundOver(spans: [number, number][], sound: RedactSound) {
  let i = 0;
  let last = -Infinity;
  return (s: AudioSample): AudioSample => {
    const t0 = s.timestamp;
    const t1 = t0 + s.duration;
    if (t0 < last) i = 0;
    last = t0;
    while (i < spans.length && spans[i][1] + FADE <= t0) i++;
    if (i >= spans.length || spans[i][0] - FADE >= t1) return s;
    const n = s.numberOfFrames;
    const ch = s.numberOfChannels;
    const rate = s.sampleRate;
    const data = new Float32Array(n * ch);
    for (let c = 0; c < ch; c++) s.copyTo(data.subarray(c * n, (c + 1) * n), { planeIndex: c, format: 'f32-planar' });
    for (let k = 0; k < n; k++) {
      const t = t0 + k / rate;
      // Gain of the cover at t: 1 inside a span, ramping to 0 over FADE outside its edges.
      let g = 0;
      for (let j = i; j < spans.length && spans[j][0] - FADE < t1; j++) {
        g = Math.max(g, Math.min(1, (t - spans[j][0] + FADE) / FADE, (spans[j][1] + FADE - t) / FADE));
      }
      if (g <= 0) continue;
      const tone = sound === 'bleep' ? TONE_GAIN * Math.sin(2 * Math.PI * TONE_HZ * t) : 0;
      for (let c = 0; c < ch; c++) data[c * n + k] = data[c * n + k] * (1 - g) + tone * g;
    }
    s.close();
    return new AudioSample({ data, format: 'f32-planar', numberOfChannels: ch, sampleRate: rate, timestamp: t0 });
  };
}

/** Strength levels: level k blurs with radius 6 * 2^k px, or pixelates with blocks 4 * 2^k px. */
const MAX_LEVEL = 6;

/**
 * Returns cover(regions), which draws over the regions on `canvas` (holding the frame; the
 * review UI previews with the same drawing on a page canvas). Regions
 * are grouped by strength level (by size: blur radius about a fifth of the face, about 6 blocks
 * across it); each level costs one small copy of the frame and one clipped draw, however many
 * faces it covers (a crowd has hundreds per frame).
 *
 * The shape is a rounded rectangle whose corner radius (30% of the shorter side) stays inside a
 * margin of 10% or more, so the detector's box is always fully covered.
 */
export function coverer(canvas: OffscreenCanvas | HTMLCanvasElement, style: RedactStyle) {
  const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
  const { width, height } = canvas;
  const small = Array.from({ length: MAX_LEVEL + 1 }, () => new OffscreenCanvas(1, 1));
  const shape = (boxes: Box[]) => {
    ctx.beginPath();
    for (const [x, y, w, h] of boxes) ctx.roundRect(x, y, w, h, 0.3 * Math.min(w, h));
  };
  const level = ([, , w, h]: Box) => {
    const size = Math.max(w, h);
    const want = style === 'blur' ? (0.2 * size) / 6 : size / 6 / 4;
    return Math.min(MAX_LEVEL, Math.max(0, Math.ceil(Math.log2(Math.max(1, want)))));
  };
  return (regions: Box[]) => {
    if (style === 'fill') {
      ctx.fillStyle = '#202020';
      shape(regions);
      ctx.fill();
      return;
    }
    const byLevel = new Map<number, Box[]>();
    for (const r of regions) {
      const k = level(r);
      byLevel.set(k, [...(byLevel.get(k) ?? []), r]);
    }
    // Copies first, from the untouched frame, then the clipped draws.
    for (const k of byLevel.keys()) {
      const f = 2 ** k * (style === 'blur' ? 1 : 4);
      const c = small[k];
      const [sw, sh] = [Math.ceil(width / f), Math.ceil(height / f)];
      if (c.width !== sw || c.height !== sh) [c.width, c.height] = [sw, sh];
      const s = c.getContext('2d')!;
      s.imageSmoothingQuality = 'high';
      // Blur: downscale by 2^k and blur 6 px there (radius 6 * 2^k at full size).
      s.filter = style === 'blur' ? 'blur(6px)' : 'none';
      s.drawImage(canvas, 0, 0, sw, sh);
    }
    for (const [k, boxes] of byLevel) {
      const f = 2 ** k * (style === 'blur' ? 1 : 4);
      const c = small[k];
      ctx.save();
      shape(boxes);
      ctx.clip();
      ctx.imageSmoothingEnabled = style === 'blur';
      ctx.drawImage(c, 0, 0, c.width, c.height, 0, 0, c.width * f, c.height * f);
      ctx.restore();
    }
  };
}
