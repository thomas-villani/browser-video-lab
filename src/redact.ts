// Redaction, the browser half: decode the video, cover the regions the faces layer gives for
// each frame (RedactionCursor, src/doc/redact.ts), encode H.264 into MP4. Audio is copied
// as-is when MP4 can hold it. Frames without regions pass through untouched.
import { BufferTarget, Conversion, Mp4OutputFormat, Output, Quality, QUALITY_HIGH, StreamTarget, type StreamTargetChunk, type VideoSample } from 'mediabunny';
import { cursorFor, REDACT_DEFAULTS, type Box, type RedactOptions, type RedactStyle, type Track, type VideoDoc } from './doc/index.ts';
import { fingerprintOf, type Progress } from './analyze';
import { openInput, type MediaInput } from './runtime/source';

export interface RedactRun extends Partial<RedactOptions> {
  /** Faces added by hand in a review (RedactPlan.added). */
  added?: Track[];
  onProgress?: (p: Progress) => void;
  signal?: AbortSignal;
}

export interface Redacted {
  /** Frames encoded; must equal doc.source.frames for the regions to line up. */
  frames: number;
  ms: number;
  /** The MP4, when no stream was given. */
  buffer?: ArrayBuffer;
}

/**
 * Redact `input` with the regions from `doc` (which must be this video's doc, with a faces
 * layer). Writes to `stream` (e.g. a FileSystemWritableFileStream) or returns the MP4 bytes.
 */
export async function redact(input: MediaInput, doc: VideoDoc, stream?: WritableStream<StreamTargetChunk>, o: RedactRun = {}): Promise<Redacted> {
  const { added, onProgress, signal, ...given } = o;
  const opts: RedactOptions = { ...REDACT_DEFAULTS, ...Object.fromEntries(Object.entries(given).filter(([, v]) => v !== undefined)) } as RedactOptions;
  const id = await fingerprintOf(input);
  if (id.fingerprint !== doc.source.fingerprint) throw new Error(`the doc is for a different file (${doc.source.name}); analyze this video first`);
  const cursor = cursorFor(doc, opts, added);
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
    video: {
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
    },
  });
  if (!conversion.isValid) {
    const why = conversion.discardedTracks.map((d) => `${d.track.type}: ${d.reason}`).join('; ');
    throw new Error(`cannot write this video as MP4 here (${why})`);
  }
  conversion.onProgress = () => onProgress?.({ pass: 'frames', stage: 'redacting', done: frame, total: Math.max(frame, doc.source.frames) });
  const onAbort = () => void conversion.cancel();
  signal?.addEventListener('abort', onAbort, { once: true });
  try {
    await conversion.execute();
  } finally {
    signal?.removeEventListener('abort', onAbort);
  }
  return { frames: frame, ms: Math.round(performance.now() - t0), ...(target instanceof BufferTarget && { buffer: target.buffer! }) };
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
