// The app's video library: videos from a folder (File System Access, Chrome/Edge) or dropped
// files. A folder is the same collection the CLI uses: docs are read from and saved to
// <video>.videodoc.json next to each video, keyframe JPEGs to <video>.keyframes/<frame>.jpg.
// Dropped files keep their docs in memory (downloadable); a dropped .videodoc.json is matched
// to its video by name. A face review is <video>.redact.json, next to the video likewise.
import { checkPlan, formatDoc, formatPlan, parseDoc, type KeyframeItem, type RedactPlan, type VideoDoc } from '../doc/index.ts';

type DirHandle = FileSystemDirectoryHandle & { values(): AsyncIterable<FileSystemDirectoryHandle | FileSystemFileHandle> };

declare global {
  interface Window {
    showDirectoryPicker?: (o?: { mode?: 'read' | 'readwrite'; id?: string }) => Promise<FileSystemDirectoryHandle>;
    showSaveFilePicker?: (o?: { suggestedName?: string; id?: string; types?: { description: string; accept: Record<string, string[]> }[] }) => Promise<FileSystemFileHandle>;
  }
}

const VIDEO_EXT = /\.(mp4|m4v|mov|mkv|webm)$/i;

export interface Entry {
  id: number;
  /** Path within the opened folder, or the dropped file's name. */
  path: string;
  name: string;
  file: () => Promise<File>;
  /** The folder holding the video, when opened from a folder (docs and images are saved there). */
  dir?: DirHandle;
  doc?: VideoDoc;
  status: 'new' | 'queued' | 'analyzing' | 'ready' | 'error';
  progress?: string;
  error?: string;
  /** Analysis steps (analysisSteps) and the current one, while analyzing. */
  steps?: string[];
  step?: number;
  /** Video length in seconds before there is a doc (probeDuration), for time estimates. */
  duration?: number;
  /** Set when a redacted copy was asked for: it is made after the analysis in the queue. */
  redactTo?: RedactTarget;
  /** The face review (kept faces, faces added by hand), once loaded or started. */
  plan?: RedactPlan;
  /** A dropped .redact.json, checked against the doc when the review opens. */
  planFile?: File;
  /** Open the review once the faces are found (Review faces on a video without them). */
  reviewAfter?: boolean;
  /** Transcribe again with word times (the review asked for them to bleep words). */
  wantWords?: boolean;
  /** Keyframe images by frame: object URLs. */
  images: Map<number, string>;
}

let nextId = 1;

async function readDoc(dir: DirHandle, name: string): Promise<VideoDoc | undefined> {
  try {
    const f = await (await dir.getFileHandle(`${name}.videodoc.json`)).getFile();
    return parseDoc(await f.text(), `${name}.videodoc.json`);
  } catch (e) {
    if ((e as DOMException).name === 'NotFoundError') return undefined;
    console.warn(e);
    return undefined;
  }
}

export const canOpenFolders = () => typeof window.showDirectoryPicker === 'function';

/** Ask for a folder (read-write, so docs can be saved) and list its videos, recursively. */
export async function openFolder(): Promise<{ folder: string; entries: Entry[] }> {
  const root = (await window.showDirectoryPicker!({ mode: 'readwrite', id: 'videodoc' })) as DirHandle;
  const entries: Entry[] = [];
  const walk = async (dir: DirHandle, prefix: string) => {
    const items: (FileSystemDirectoryHandle | FileSystemFileHandle)[] = [];
    for await (const h of dir.values()) items.push(h);
    items.sort((a, b) => a.name.localeCompare(b.name));
    for (const h of items) {
      if (h.kind === 'directory' && !h.name.endsWith('.keyframes') && !h.name.startsWith('.')) await walk(h as DirHandle, `${prefix}${h.name}/`);
      else if (h.kind === 'file' && VIDEO_EXT.test(h.name)) {
        const fh = h as FileSystemFileHandle;
        const doc = await readDoc(dir, h.name);
        entries.push({ id: nextId++, path: prefix + h.name, name: h.name, file: () => fh.getFile(), dir, doc, status: doc ? 'ready' : 'new', images: new Map() });
      }
    }
  };
  await walk(root, '');
  return { folder: root.name, entries };
}

/** Entries for dropped or picked files; .videodoc.json files among them attach to their video. */
export async function fromFiles(files: File[]): Promise<Entry[]> {
  const plans = new Map(files.filter((f) => f.name.endsWith('.redact.json')).map((f) => [f.name.slice(0, -'.redact.json'.length), f]));
  const docs = new Map<string, VideoDoc>();
  for (const f of files.filter((f) => f.name.endsWith('.videodoc.json'))) {
    try {
      docs.set(f.name.slice(0, -'.videodoc.json'.length), parseDoc(await f.text(), f.name));
    } catch (e) {
      console.warn(e);
    }
  }
  return files
    .filter((f) => VIDEO_EXT.test(f.name))
    .map((f) => {
      const doc = docs.get(f.name);
      return { id: nextId++, path: f.name, name: f.name, file: async () => f, doc, planFile: plans.get(f.name), status: doc ? 'ready' : 'new', images: new Map() } as Entry;
    });
}

/** Save the doc next to the video (folder entries); returns false when there is nowhere to save. */
export async function saveDoc(e: Entry): Promise<boolean> {
  if (!e.dir || !e.doc) return false;
  const w = await (await e.dir.getFileHandle(`${e.name}.videodoc.json`, { create: true })).createWritable();
  await w.write(formatDoc(e.doc));
  await w.close();
  return true;
}

/** Load the saved face review into e.plan (from the folder, or a dropped file); returns warnings
 *  to show. Leaves e.plan unset when there is none. */
export async function loadPlan(e: Entry): Promise<string[]> {
  if (e.plan || !e.doc?.layers.faces) return [];
  let file = e.planFile;
  if (!file && e.dir) {
    try {
      file = await (await e.dir.getFileHandle(`${e.name}.redact.json`)).getFile();
    } catch {
      return [];
    }
  }
  if (!file) return [];
  try {
    const { plan, warnings } = checkPlan(JSON.parse(await file.text()), e.doc);
    e.plan = plan;
    return warnings;
  } catch (err) {
    return [`the saved review (${file.name}) was not used: ${(err as Error).message}`];
  }
}

/** Save the face review next to the video (folder entries); false when there is nowhere to save. */
export async function savePlan(e: Entry): Promise<boolean> {
  if (!e.dir || !e.plan) return false;
  const w = await (await e.dir.getFileHandle(`${e.name}.redact.json`, { create: true })).createWritable();
  await w.write(formatPlan(e.plan));
  await w.close();
  return true;
}

export function downloadDoc(e: Entry) {
  if (!e.doc) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([formatDoc(e.doc)], { type: 'application/json' }));
  a.download = `${e.name}.videodoc.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const imageName = (k: KeyframeItem) => `${String(k.frame).padStart(6, '0')}.jpg`;

/** Keyframe JPEGs already saved in <video>.keyframes/ (folder entries), loaded into `images`. */
export async function loadSavedImages(e: Entry, keyframes: KeyframeItem[]): Promise<void> {
  if (!e.dir) return;
  let dir: FileSystemDirectoryHandle;
  try {
    dir = await e.dir.getDirectoryHandle(`${e.name}.keyframes`);
  } catch {
    return;
  }
  for (const k of keyframes) {
    if (e.images.has(k.frame)) continue;
    try {
      const f = await (await dir.getFileHandle(imageName(k))).getFile();
      e.images.set(k.frame, URL.createObjectURL(f));
    } catch {
      // not extracted yet
    }
  }
}

/** Keep extracted keyframe JPEGs, in memory and (folder entries) on disk. */
export async function storeImages(e: Entry, keyframes: KeyframeItem[], blobs: Blob[]): Promise<void> {
  keyframes.forEach((k, i) => e.images.set(k.frame, URL.createObjectURL(blobs[i])));
  if (!e.dir) return;
  const dir = await e.dir.getDirectoryHandle(`${e.name}.keyframes`, { create: true });
  for (const [i, k] of keyframes.entries()) {
    const w = await (await dir.getFileHandle(imageName(k), { create: true })).createWritable();
    await w.write(blobs[i]);
    await w.close();
  }
}

/** A video's length from its metadata (fast; undefined when the browser cannot play it, e.g. some mkv). */
export function probeDuration(file: File): Promise<number | undefined> {
  return new Promise((resolve) => {
    const v = document.createElement('video');
    const url = URL.createObjectURL(file);
    const done = (d?: number) => {
      URL.revokeObjectURL(url);
      resolve(d);
    };
    v.preload = 'metadata';
    v.onloadedmetadata = () => done(Number.isFinite(v.duration) ? v.duration : undefined);
    v.onerror = () => done(undefined);
    v.src = url;
  });
}

/** Where a redacted copy goes: a file (next to the video for folder entries, else one the
 *  person picks) or, without the File System Access API, a download of the whole MP4. */
export interface RedactTarget {
  name: string;
  file?: FileSystemFileHandle;
  /** The folder the files were created in, when they did not exist before (removed on cancel). */
  created?: FileSystemDirectoryHandle;
  /** The redaction log's file (folder entries); otherwise it is downloaded. */
  log?: FileSystemFileHandle;
}

/** Ask where to save the redacted copy. Call from a click (the save dialog needs one);
 *  undefined when the person cancels. */
export async function redactTarget(e: Entry): Promise<RedactTarget | undefined> {
  const name = `${e.name.replace(/\.[^.]+$/, '')}.redacted.mp4`;
  if (e.dir) {
    const existed = await e.dir.getFileHandle(name).then(() => true, () => false);
    return {
      name,
      file: await e.dir.getFileHandle(name, { create: true }),
      log: await e.dir.getFileHandle(`${name}.redactions.json`, { create: true }),
      ...(!existed && { created: e.dir }),
    };
  }
  if (!window.showSaveFilePicker) return { name };
  try {
    const file = await window.showSaveFilePicker({ suggestedName: name, id: 'videodoc-redacted', types: [{ description: 'MP4 video', accept: { 'video/mp4': ['.mp4'] } }] });
    return { name: file.name, file };
  } catch (err) {
    if ((err as DOMException).name === 'AbortError') return undefined;
    throw err;
  }
}

/** Save a file the browser way: a download. */
export function download(name: string, data: BlobPart, type: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([data], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}
