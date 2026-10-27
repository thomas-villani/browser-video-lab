// Creating, merging, validating and summarizing VideoDocs. Pure TS.
import { LAYER_NAMES, SCHEMA, type Layers, type SourceInfo, type VideoDoc } from './schema.ts';

export class DocError extends Error {}

/** Where a video's doc lives by default: next to it. */
export const docPathFor = (video: string) => `${video}.videodoc.json`;

export function newDoc(source: SourceInfo): VideoDoc {
  return { schema: SCHEMA, source, layers: {} };
}

/** A new doc with `layers` added or replaced (layers are never edited in place). */
export function withLayers(doc: VideoDoc, layers: Partial<Layers>): VideoDoc {
  const merged = { ...doc.layers, ...layers };
  // Keep a stable key order so diffs of saved docs stay readable.
  const ordered: Partial<Layers> = {};
  for (const n of LAYER_NAMES) if (merged[n]) (ordered as Record<string, unknown>)[n] = merged[n];
  return { ...doc, layers: ordered };
}

/** Serialize with one layer item per line: compact for long videos, readable, greppable, and
 *  diffs stay small when a layer is replaced. */
export function formatDoc(doc: VideoDoc): string {
  const layers = Object.entries(doc.layers).map(([name, l]) => {
    const { items, ...meta } = l as { items: unknown[] };
    const body = items.length ? `[\n${items.map((i) => `      ${JSON.stringify(i)}`).join(',\n')}\n    ]` : '[]';
    return `  ${JSON.stringify(name)}: {${JSON.stringify(meta).slice(1, -1)},\n    "items": ${body}}`;
  });
  return `{"schema": ${JSON.stringify(doc.schema)},\n "source": ${JSON.stringify(doc.source)},\n "layers": {\n${layers.join(',\n')}\n }}\n`;
}

/** Parse and check a saved doc; errors say what is wrong and how to fix it. */
export function parseDoc(text: string, where = 'doc'): VideoDoc {
  let v: unknown;
  try {
    v = JSON.parse(text);
  } catch (e) {
    throw new DocError(`${where}: not valid JSON (${(e as Error).message}); delete it or re-run analyze with --force`);
  }
  const d = v as Partial<VideoDoc>;
  if (!d || typeof d !== 'object' || typeof d.schema !== 'string') throw new DocError(`${where}: not a videodoc (no "schema" field)`);
  if (d.schema !== SCHEMA) throw new DocError(`${where}: schema '${d.schema}' is not supported by this version (expects '${SCHEMA}'); re-run analyze with --force`);
  if (!d.source || typeof d.source.fingerprint !== 'string') throw new DocError(`${where}: missing source.fingerprint`);
  if (!d.layers || typeof d.layers !== 'object') throw new DocError(`${where}: missing layers`);
  for (const [name, layer] of Object.entries(d.layers)) {
    if (!(LAYER_NAMES as string[]).includes(name)) throw new DocError(`${where}: unknown layer '${name}'`);
    if (!layer?.producer?.configHash || !Array.isArray(layer.items)) throw new DocError(`${where}: layer '${name}' is missing producer or items`);
  }
  return d as VideoDoc;
}

export interface LayerSummary {
  items: number;
  ms: number;
  device: string;
  model?: string;
  configHash: string;
  /** Layer-specific figures, e.g. words, speech seconds, tracks. */
  [k: string]: unknown;
}

const r1 = (v: number) => Math.round(v * 10) / 10;

export function summarize(doc: VideoDoc): Record<string, LayerSummary> {
  const out: Record<string, LayerSummary> = {};
  const { speech, transcript, scenes, keyframes, faces } = doc.layers;
  const base = (l: NonNullable<VideoDoc['layers'][keyof Layers]>) => ({
    items: l.items.length,
    ms: Math.round(l.ms),
    device: l.device,
    ...(l.producer.model && { model: l.producer.model }),
    configHash: l.producer.configHash,
  });
  if (speech) {
    const s = speech.items.reduce((n, x) => n + x.t1 - x.t0, 0);
    out.speech = { ...base(speech), speechSeconds: r1(s), speechPct: r1((100 * s) / doc.source.duration) };
  }
  if (transcript) out.transcript = { ...base(transcript), words: transcript.items.reduce((n, x) => n + x.text.split(/\s+/).filter(Boolean).length, 0) };
  if (scenes) out.scenes = base(scenes);
  if (keyframes) {
    const why: Record<string, number> = {};
    for (const k of keyframes.items) why[k.why] = (why[k.why] ?? 0) + 1;
    out.keyframes = { ...base(keyframes), ...why };
  }
  if (faces) {
    const frames = faces.items.reduce((n, t) => n + t.f1 - t.f0 + 1, 0);
    out.faces = { ...base(faces), samples: faces.items.reduce((n, t) => n + t.samples.length, 0), faceFrames: frames };
  }
  return out;
}
