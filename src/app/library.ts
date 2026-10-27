// The app's video library: videos from a folder (File System Access, Chrome/Edge) or dropped
// files. A folder is the same collection the CLI uses: docs are read from and saved to
// <video>.videodoc.json next to each video, keyframe JPEGs to <video>.keyframes/<frame>.jpg.
// Dropped files keep their docs in memory (downloadable); a dropped .videodoc.json is matched
// to its video by name.
import { formatDoc, parseDoc, type KeyframeItem, type VideoDoc } from '../doc/index.ts';

type DirHandle = FileSystemDirectoryHandle & { values(): AsyncIterable<FileSystemDirectoryHandle | FileSystemFileHandle> };

declare global {
  interface Window {
    showDirectoryPicker?: (o?: { mode?: 'read' | 'readwrite'; id?: string }) => Promise<FileSystemDirectoryHandle>;
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
      return { id: nextId++, path: f.name, name: f.name, file: async () => f, doc, status: doc ? 'ready' : 'new', images: new Map() } as Entry;
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
