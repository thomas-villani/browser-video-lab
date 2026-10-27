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
  device: 'webgpu' | 'wasm';
  items: T[];
}

/** VAD speech span. */
export interface SpeechItem { t0: number; t1: number }
export interface TranscriptItem { t0: number; t1: number; text: string }
/** A shot: frames [f0, f1] inclusive; `score` is the cut score that started it (0 for the first). */
export interface SceneItem { t0: number; t1: number; f0: number; f1: number; score: number }

/**
 * A tracked object over frames [f0, f1]. `samples` are [frame, x, y, w, h] in source pixels;
 * boxes between samples are linearly interpolated (see boxesAt in tracks.ts).
 */
export interface Track { id: number; f0: number; f1: number; samples: number[][] }

export interface Layers {
  speech: Layer<SpeechItem>;
  transcript: Layer<TranscriptItem>;
  scenes: Layer<SceneItem>;
  faces: Layer<Track>;
}

export type LayerName = keyof Layers;
export const LAYER_NAMES: LayerName[] = ['speech', 'transcript', 'scenes', 'faces'];

export interface VideoDoc {
  schema: typeof SCHEMA;
  source: SourceInfo;
  layers: Partial<Layers>;
}

/** Pointer to an item in a layer, e.g. "transcript/12". */
export type Ref = `${LayerName}/${number}`;
