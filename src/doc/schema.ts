// The VideoDoc: one JSON document per video, holding time-indexed layers on a common timeline
// (seconds, plus frame indices where frames matter). Analysis writes it once; every feature
// reads from it. Pure types and constants: loads in Node, a worker, or the browser.

export const SCHEMA = 'videodoc/0';
export const VERSION = '0.0.1';

export type Box = [x: number, y: number, w: number, h: number];

export interface SourceInfo {
  /** File name (no directory), for display. */
  name: string;
  /** Cheap identity (see fingerprint in hash.ts): matches a doc to its video without decoding
   *  or hashing the whole file. */
  fingerprint: string;
  size: number;
  duration: number;
  width: number;
  height: number;
  /** Average frame rate; frame f is at roughly f / fps seconds (exact times live in scenes). */
  fps: number;
  frames: number;
  video: { codec: string };
  audio?: { codec: string; sampleRate: number; channels: number };
}

export interface Producer {
  /** Producer name, e.g. "faces". */
  module: string;
  version: string;
  model?: string;
  /** 'track': the items are Tracks (built-in faces and plates, and a pipeline's track layers). */
  kind?: 'track';
  config: Record<string, unknown>;
  /** Hash of config + model + the hashes of the layers this one was computed from. Re-running
   *  with the same hash is a no-op. */
  configHash: string;
}

export interface Layer<T> {
  producer: Producer;
  /** What was analyzed, as [t0, t1] second ranges. */
  coverage: [number, number][];
  createdAt: string;
  /** Wall time of the pass that produced the layer. Layers produced by one shared pass (scenes
   *  and faces share a decode) report the same time. */
  ms: number;
  /** 'cpu' for layers derived from other layers (no decode, no model). */
  device: 'webgpu' | 'wasm' | 'cpu';
  items: T[];
}

/** VAD speech span. */
export interface SpeechItem { t0: number; t1: number }
/** A spoken word: start and end in seconds, and the text as transcribed (with punctuation). */
export type Word = [t0: number, t1: number, text: string];
/** A transcript segment; `words` when the model gave word timestamps (transcript.words). */
export interface TranscriptItem { t0: number; t1: number; text: string; words?: Word[] }
/**
 * A speaker (diarization: who speaks when). Ids are per video, numbered in order of first
 * appearance; names are not stored here (a person names speakers, and the name survives
 * re-analysis). `turns` are when the speaker talks, in seconds (they may overlap other
 * speakers'); `voice` is the speaker's voice embedding (WeSpeaker ResNet34, unit length), for
 * recognizing the same voice in other videos.
 */
export interface SpeakerItem { id: number; turns: [t0: number, t1: number][]; voice: number[] }
/** A shot: frames [f0, f1] inclusive; `score` is the cut score that started it (0 for the first). */
export interface SceneItem { t0: number; t1: number; f0: number; f1: number; score: number }
/** A frame worth looking at: the start of a shot, the middle of a silent gap, or a fill-in
 *  so long shots are covered. `scene` indexes the scenes layer. */
export interface KeyframeItem { t: number; frame: number; scene: number; why: 'scene' | 'silence' | 'coverage' }

/**
 * A tracked object over frames [f0, f1]. `samples` are [frame, x, y, w, h] in source pixels;
 * boxes between samples are linearly interpolated (see boxesAt in tracks.ts).
 */
export interface Track { id: number; f0: number; f1: number; samples: number[][] }

export interface Layers {
  speech: Layer<SpeechItem>;
  transcript: Layer<TranscriptItem>;
  speakers: Layer<SpeakerItem>;
  scenes: Layer<SceneItem>;
  keyframes: Layer<KeyframeItem>;
  faces: Layer<Track>;
  /** License plates, tracked like faces (ids are per layer). */
  plates: Layer<Track>;
}

/** The built-in layers. */
export type LayerName = keyof Layers;
/** In dependency order. */
export const LAYER_NAMES: LayerName[] = ['speech', 'transcript', 'speakers', 'scenes', 'keyframes', 'faces', 'plates'];
/** The built-in track layers (a pipeline can add more: their producer.kind is 'track'). */
export const TRACK_LAYERS: LayerName[] = ['faces', 'plates'];
/** Names a pipeline may give its layers. */
export const LAYER_NAME_RE = /^[a-z][a-z0-9_]*$/;

/** A doc's layers: the built-in ones, plus track layers a pipeline added (e.g. `vehicles`). */
export type DocLayers = Partial<Layers> & { [custom: string]: Layer<unknown> | undefined };

export interface VideoDoc {
  schema: typeof SCHEMA;
  source: SourceInfo;
  layers: DocLayers;
}

/** Pointer to an item in a layer, e.g. "transcript/12". */
export type Ref = `${string}/${number}`;
