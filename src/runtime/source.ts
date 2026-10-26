// Sequential frame access with frame indices, over mediabunny.
import { ALL_FORMATS, Input, UrlSource, VideoSampleSink, type InputVideoTrack } from 'mediabunny';

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

  static async open(url: string): Promise<VideoSource> {
    const input = new Input({ source: new UrlSource(url), formats: ALL_FORMATS });
    const track = await input.getPrimaryVideoTrack();
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
