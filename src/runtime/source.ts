// Sequential frame access with frame indices, over mediabunny.
import { ALL_FORMATS, BlobSource, Input, UrlSource, VideoSampleSink, type InputVideoTrack } from 'mediabunny';

/** A video by URL (range requests) or as a File/Blob. */
export type MediaInput = string | Blob;

export function openInput(input: MediaInput): Input {
  return new Input({ source: typeof input === 'string' ? new UrlSource(input) : new BlobSource(input), formats: ALL_FORMATS });
}

export interface Probe {
  duration: number;
  width: number;
  height: number;
  fps: number;
  frames: number;
  video: { codec: string };
  audio?: { codec: string; sampleRate: number; channels: number };
}

/** Container-level facts about a video, without decoding it. */
export async function probe(input: MediaInput): Promise<Probe> {
  const media = openInput(input);
  const video = await media.getPrimaryVideoTrack();
  if (!video) throw new Error('no video track');
  const audio = await media.getPrimaryAudioTrack();
  const stats = await video.computePacketStats();
  return {
    duration: Math.round((await media.computeDuration()) * 1000) / 1000,
    width: video.displayWidth,
    height: video.displayHeight,
    fps: Math.round(stats.averagePacketRate * 1000) / 1000,
    frames: stats.packetCount,
    video: { codec: await video.getCodecParameterString() ?? video.codec ?? 'unknown' },
    ...(audio && { audio: { codec: (await audio.getCodecParameterString()) ?? audio.codec ?? 'unknown', sampleRate: audio.sampleRate, channels: audio.numberOfChannels } }),
  };
}

export interface SourceFrame {
  index: number;
  /** Presentation timestamp, seconds. */
  t: number;
  /** Valid until the consumer asks for the next frame; closed by the source. */
  frame: VideoFrame;
}

export class VideoSource {
  private constructor(
    readonly url: string,
    private track: InputVideoTrack,
    readonly fps: number,
    readonly duration: number,
  ) {}

  static async open(input: MediaInput): Promise<VideoSource> {
    const url = typeof input === 'string' ? input : 'name' in input ? (input as File).name : 'blob';
    const track = await openInput(input).getPrimaryVideoTrack();
    if (!track) throw new Error(`no video track in ${url}`);
    const stats = await track.computePacketStats(100);
    return new VideoSource(url, track, stats.averagePacketRate, await track.computeDuration());
  }

  get width() {
    return this.track.displayWidth;
  }

  get height() {
    return this.track.displayHeight;
  }

  async meta() {
    return {
      url: this.url,
      codec: await this.track.getCodecParameterString(),
      width: this.width,
      height: this.height,
      fps: Math.round(this.fps * 1000) / 1000,
      duration: Math.round(this.duration * 1000) / 1000,
    };
  }

  /** Frames in presentation order. Each frame is closed when the consumer moves on. */
  async *frames(limit = Infinity): AsyncGenerator<SourceFrame> {
    const sink = new VideoSampleSink(this.track);
    let index = 0;
    for await (const sample of sink.samples()) {
      const frame = sample.toVideoFrame();
      try {
        yield { index, t: sample.timestamp, frame };
      } finally {
        frame.close();
        sample.close();
      }
      if (++index >= limit) break;
    }
  }
}
