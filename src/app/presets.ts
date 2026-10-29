// Pipeline presets in the app (docs/pipelines.md): the same videodoc-pipeline/0 files the CLI
// takes. Read from the top of an opened folder (videodoc.pipeline.json, used for every video
// in it) or picked together with their model files; saved back from "What to analyze". Local
// model paths (relative to the preset) become object URLs of the files they name.
import { checkPipeline, MODELS, type ModelConfig, type Pipeline } from '../doc/index.ts';

/** The preset a folder uses, at its top. */
export const PRESET_FILE = 'videodoc.pipeline.json';

export interface Preset {
  /** As written (model paths as in the file), for saving. */
  pipeline: Pipeline;
  /** For analyze(): local model files as object URLs. */
  run: Pipeline;
  /** The file it came from. */
  source: string;
}

/** A path relative to the preset, not a URL the page can fetch as it is. */
const isLocal = (url: string) => !/^(\/|https?:|blob:|data:)/i.test(url);

/** Parse a preset, finding its local model files with `find` (a path relative to the preset ->
 *  the file, or undefined). `missing` says how to supply files that were not found. */
export async function readPreset(text: string, source: string, find: (path: string) => Promise<Blob | undefined>, missing: string): Promise<Preset> {
  let pipeline: Pipeline;
  try {
    pipeline = checkPipeline(JSON.parse(text), source);
  } catch (err) {
    throw new Error(err instanceof SyntaxError ? `${source}: not JSON (${err.message})` : (err as Error).message);
  }
  const models: Record<string, ModelConfig> = {};
  const lost: string[] = [];
  for (const [k, m] of Object.entries(pipeline.models ?? {})) {
    if (!isLocal(m.url)) models[k] = m;
    else {
      const f = await find(m.url);
      if (f) models[k] = { ...m, url: URL.createObjectURL(f) };
      else lost.push(m.url);
    }
  }
  if (lost.length) {
    for (const m of Object.values(models)) if (m.url.startsWith('blob:')) URL.revokeObjectURL(m.url);
    throw new Error(`${source}: model file${lost.length > 1 ? 's' : ''} ${lost.join(', ')} not found; ${missing}`);
  }
  return { pipeline, run: { ...pipeline, models }, source };
}

/** Free a preset's object URLs. */
export function dropPreset(p: Preset) {
  for (const m of Object.values(p.run.models ?? {})) if (m.url.startsWith('blob:')) URL.revokeObjectURL(m.url);
}

/** A file under `dir` by a relative path (`./models/x.onnx`); paths above it are not reachable. */
export async function fileIn(dir: FileSystemDirectoryHandle, path: string): Promise<File | undefined> {
  const parts = path.replace(/\\/g, '/').split('/').filter((p) => p && p !== '.');
  if (!parts.length || parts.includes('..')) return undefined;
  try {
    let d = dir;
    for (const p of parts.slice(0, -1)) d = await d.getDirectoryHandle(p);
    return await (await d.getFileHandle(parts[parts.length - 1])).getFile();
  } catch {
    return undefined;
  }
}

/** The preset at the top of an opened folder, if there is one. */
export async function folderPreset(root: FileSystemDirectoryHandle): Promise<Preset | undefined> {
  const f = await fileIn(root, PRESET_FILE);
  if (!f) return undefined;
  return readPreset(await f.text(), PRESET_FILE, (p) => fileIn(root, p), 'put them in the folder where the preset says');
}

/** A preset picked (or dropped) with its model files, matched by file name. */
export async function pickedPreset(json: File, files: File[]): Promise<Preset> {
  const byName = (p: string) => files.find((f) => f.name === p.replace(/\\/g, '/').split('/').pop());
  return readPreset(await json.text(), json.name, async (p) => byName(p), 'choose them together with the preset');
}

/** The decoder a track layer's model uses (for time estimates), or undefined when unknown. */
export function decoderOf(p: Pipeline, track: string): string | undefined {
  const id = p.tracks?.[track]?.model;
  return id ? (p.models?.[id] ?? MODELS[id])?.decoder : undefined;
}
