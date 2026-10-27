// videodoc app (app.html): open a folder or drop videos, analyze them in this browser, read them
// (the same Markdown view agents get, with a player beside it) and search across them. Nothing
// is uploaded: analysis runs here on WebGPU, docs are saved next to the videos or kept in memory.
import { highlight, search, toMarkdown, type LayerName } from '../doc/index.ts';
import { analyze, frameImages } from '../analyze';
import { canOpenFolders, downloadDoc, fromFiles, loadSavedImages, openFolder, saveDoc, storeImages, type Entry } from './library';
import { esc, markdownToHtml, marks } from './render';
import './style.css';

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;

const state = {
  entries: [] as Entry[],
  selected: undefined as Entry | undefined,
  query: '',
  folder: '',
  gpu: undefined as boolean | undefined,
  /** Object URL of the video in the player. */
  videoUrl: '',
  /** Search terms (stemmed) to mark in the reading view, when it was opened from a hit. */
  marked: new Set<string>(),
};

// ---- analysis queue: one video at a time (one GPU) ----

const queue: Entry[] = [];
let running = false;

function layersWanted(): LayerName[] {
  const layers: LayerName[] = ['speech', 'scenes', 'keyframes'];
  if ($<HTMLInputElement>('#opt-transcript').checked) layers.push('transcript');
  if ($<HTMLInputElement>('#opt-faces').checked) layers.push('faces');
  return layers;
}

function enqueue(e: Entry) {
  if (e.status === 'queued' || e.status === 'analyzing') return;
  e.status = 'queued';
  e.error = undefined;
  queue.push(e);
  renderList();
  void runQueue();
}

async function runQueue() {
  if (running) return;
  running = true;
  while (queue.length) {
    const e = queue.shift()!;
    e.status = 'analyzing';
    e.progress = 'starting';
    renderList();
    try {
      const t0 = performance.now();
      e.doc = await analyze(await e.file(), layersWanted(), {
        doc: e.doc,
        onProgress: (p) => {
          e.progress = `${p.stage}${p.total > 1 ? ` ${Math.floor((100 * p.done) / p.total)}%` : ''}`;
          renderEntry(e);
        },
      });
      e.status = 'ready';
      e.progress = `done in ${((performance.now() - t0) / 1000).toFixed(0)} s`;
      if (!(await saveDoc(e))) e.progress += ' (not saved: download the doc to keep it)';
    } catch (err) {
      console.error(err);
      e.status = 'error';
      e.error = (err as Error).message;
    }
    renderList();
    if (state.selected === e) await showEntry(e);
    if (state.query) renderResults();
  }
  running = false;
}

// ---- video list ----

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

function entryHtml(e: Entry): string {
  const d = e.doc;
  const chips = d ? Object.keys(d.layers).map((n) => `<span class="chip">${n}</span>`).join('') : '';
  const meta = d ? `${fmt(d.source.duration)} · ${d.source.height}p` : '';
  const status =
    e.status === 'analyzing' ? `<span class="busy">${esc(e.progress ?? '')}</span>`
    : e.status === 'queued' ? '<span class="busy">queued</span>'
    : e.status === 'error' ? `<span class="err" title="${esc(e.error ?? '')}">failed: ${esc(e.error ?? '')}</span>`
    : e.progress ? `<span class="muted">${esc(e.progress)}</span>` : '';
  const action = e.status === 'new' || e.status === 'error' ? '<button class="analyze">Analyze</button>' : '';
  return `<li data-id="${e.id}" class="${state.selected === e ? 'sel' : ''} ${e.status}">
    <div class="row"><span class="name" title="${esc(e.path)}">${esc(e.path)}</span>${action}</div>
    <div class="meta">${meta} ${chips}</div>${status ? `<div class="status">${status}</div>` : ''}</li>`;
}

function renderEntry(e: Entry) {
  const li = document.querySelector(`#list li[data-id="${e.id}"]`);
  if (li) li.outerHTML = entryHtml(e);
}

function renderList() {
  const n = state.entries.length;
  const todo = state.entries.filter((e) => e.status === 'new' || e.status === 'error').length;
  $('#list-head').innerHTML = n
    ? `<span>${esc(state.folder || 'Videos')} · ${n}</span>${todo ? `<button id="analyze-all">Analyze ${todo}</button>` : ''}`
    : '';
  $('#list').innerHTML = state.entries.map(entryHtml).join('');
}

// ---- reading view ----

const video = () => $<HTMLVideoElement>('#player');

async function showEntry(e: Entry, at?: number, marked: string[] = []) {
  state.selected = e;
  state.marked = new Set(marked);
  state.query = '';
  $<HTMLInputElement>('#q').value = '';
  renderList();
  $('#results').hidden = true;
  $('#empty').hidden = true;
  $('#viewer').hidden = false;
  const file = await e.file();
  if (state.videoUrl && video().dataset.id !== String(e.id)) {
    URL.revokeObjectURL(state.videoUrl);
    state.videoUrl = '';
  }
  if (!state.videoUrl) {
    state.videoUrl = URL.createObjectURL(file);
    video().src = state.videoUrl;
    video().dataset.id = String(e.id);
  }
  $('#doc-actions').innerHTML = e.doc
    ? `<button id="copy-md">Copy Markdown</button><button id="dl-doc">Download doc</button>${e.status === 'ready' ? '<button id="reanalyze" title="Compute the layers ticked above that are missing">Add layers</button>' : ''}`
    : '';
  const read = $('#read');
  if (!e.doc) {
    read.innerHTML = `<h1>${esc(e.name)}</h1><p class="note">Not analyzed yet.</p>${e.status === 'new' ? '<p><button class="analyze-this primary">Analyze this video</button></p>' : ''}`;
  } else {
    renderRead(e);
    const kfs = e.doc.layers.keyframes?.items ?? [];
    await loadSavedImages(e, kfs);
    renderRead(e);
    const missing = kfs.filter((k) => !e.images.has(k.frame));
    if (missing.length && state.gpu !== false) {
      $('#read-status').textContent = `extracting ${missing.length} keyframes…`;
      try {
        await storeImages(e, missing, await frameImages(file, missing.map((k) => k.t)));
      } catch (err) {
        console.warn(err);
      }
      $('#read-status').textContent = '';
      if (state.selected === e) fillImages(e);
    }
  }
  if (at !== undefined) seek(at, true);
}

function renderRead(e: Entry) {
  const md = toMarkdown(e.doc!, { image: (k) => `kf:${k.frame}` });
  const mark = state.marked.size ? (t: string) => highlight(t, state.marked) : undefined;
  $('#read').innerHTML = markdownToHtml(md, (f) => e.images.get(f), mark);
}

/** Put newly available keyframe images into the reading view without re-rendering it. */
function fillImages(e: Entry) {
  for (const fig of document.querySelectorAll<HTMLElement>('#read figure.kf')) {
    const src = e.images.get(Number(fig.dataset.frame));
    const pending = fig.querySelector('.pending');
    if (src && pending) pending.outerHTML = `<img src="${src}" alt="frame at ${fig.dataset.t} s">`;
  }
}

function seek(t: number, scroll = false) {
  const v = video();
  const go = () => {
    v.currentTime = t;
  };
  if (v.readyState >= 1) go();
  else v.addEventListener('loadedmetadata', go, { once: true });
  follow(t, scroll);
}

/** Mark the paragraph being spoken at time t (and optionally scroll to it). */
function follow(t: number, scroll: boolean) {
  let current: HTMLElement | undefined;
  for (const p of document.querySelectorAll<HTMLElement>('#read p.said')) {
    if (Number(p.dataset.t) <= t + 0.25) current = p;
    else break;
  }
  const prev = document.querySelector('#read .now');
  if (prev !== current) {
    prev?.classList.remove('now');
    current?.classList.add('now');
  }
  if (scroll && current) current.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

// ---- search ----

function renderResults() {
  const searchable = state.entries.filter((e) => e.doc?.layers.transcript);
  const hits = search(searchable.map((e) => e.doc!), state.query, { top: 30 });
  const box = $('#results');
  box.hidden = false;
  $('#viewer').hidden = true;
  $('#empty').hidden = true;
  const skipped = state.entries.length - searchable.length;
  const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? '' : 's'}`;
  const head = `<p class="muted">${plural(hits.length, 'passage')} in ${plural(searchable.length, 'transcript')}${skipped ? ` · ${plural(skipped, 'video')} not searchable yet (analyze with the transcript layer)` : ''}</p>`;
  box.innerHTML =
    head +
    hits
      .map((h) => {
        const e = searchable[h.doc];
        const k = h.keyframe !== undefined ? e.doc!.layers.keyframes!.items[h.keyframe] : undefined;
        const img = k && e.images.get(k.frame);
        const long = e.doc!.source.duration >= 3600;
        const ts = (t: number) => (long ? `${Math.floor(t / 3600)}:${fmt(t % 3600).padStart(5, '0')}` : fmt(t));
        return `<li class="hit" data-id="${e.id}" data-t="${h.at}" data-terms="${esc(h.matched.join(' '))}">
          ${img ? `<img src="${img}" alt="">` : '<div class="thumb"></div>'}
          <div><div class="where"><b>${esc(e.path)}</b> · ${ts(h.t0)}–${ts(h.t1)}</div><div class="snip">${marks(h.snippet)}</div></div></li>`;
      })
      .join('');
}

let searchTimer: ReturnType<typeof setTimeout> | undefined;
function onSearch() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.query = $<HTMLInputElement>('#q').value.trim();
    if (state.query) renderResults();
    else if (state.selected) void showEntry(state.selected);
    else {
      $('#results').hidden = true;
      $('#empty').hidden = false;
    }
  }, 120);
}

// ---- adding videos ----

function addEntries(entries: Entry[], folder = '') {
  if (folder) {
    state.entries = entries;
    state.folder = folder;
    state.selected = undefined;
  } else {
    const known = new Set(state.entries.map((e) => e.path));
    state.entries.push(...entries.filter((e) => !known.has(e.path)));
  }
  renderList();
  const first = entries.find((e) => e.doc) ?? entries[0];
  if (first && !state.query) void showEntry(first);
}

async function pickFolder() {
  try {
    const { folder, entries } = await openFolder();
    if (!entries.length) return alert(`No videos (mp4, mov, mkv, webm) in ${folder}.`);
    addEntries(entries, folder);
  } catch (err) {
    if ((err as DOMException).name !== 'AbortError') alert((err as Error).message);
  }
}

// ---- wiring ----

function wire() {
  $('#open-folder').hidden = !canOpenFolders();
  $('#open-folder').addEventListener('click', () => void pickFolder());
  $('#add-files').addEventListener('change', async (ev) => {
    const input = ev.target as HTMLInputElement;
    addEntries(await fromFiles([...(input.files ?? [])]));
    input.value = '';
  });
  $('#q').addEventListener('input', onSearch);
  document.addEventListener('keydown', (ev) => {
    if (ev.key === '/' && document.activeElement?.tagName !== 'INPUT') {
      ev.preventDefault();
      $<HTMLInputElement>('#q').focus();
    }
  });

  $('#list-head').addEventListener('click', (ev) => {
    if ((ev.target as HTMLElement).id === 'analyze-all') state.entries.filter((e) => e.status === 'new' || e.status === 'error').forEach(enqueue);
  });
  $('#list').addEventListener('click', (ev) => {
    const li = (ev.target as HTMLElement).closest('li');
    const e = li && state.entries.find((x) => x.id === Number(li.dataset.id));
    if (!e) return;
    if ((ev.target as HTMLElement).classList.contains('analyze')) enqueue(e);
    else void showEntry(e);
  });
  $('#results').addEventListener('click', (ev) => {
    const li = (ev.target as HTMLElement).closest<HTMLElement>('li.hit');
    const e = li && state.entries.find((x) => x.id === Number(li.dataset.id));
    if (e) void showEntry(e, Number(li.dataset.t), li.dataset.terms?.split(' ').filter(Boolean));
  });
  $('#read').addEventListener('click', (ev) => {
    const el = (ev.target as HTMLElement).closest<HTMLElement>('[data-t]');
    if ((ev.target as HTMLElement).classList.contains('analyze-this') && state.selected) return enqueue(state.selected);
    if (el) seek(Number(el.dataset.t));
  });
  $('#doc-actions').addEventListener('click', async (ev) => {
    const e = state.selected;
    const id = (ev.target as HTMLElement).id;
    if (!e?.doc) return;
    if (id === 'copy-md') {
      await navigator.clipboard.writeText(toMarkdown(e.doc));
      (ev.target as HTMLElement).textContent = 'Copied';
      setTimeout(() => ((ev.target as HTMLElement).textContent = 'Copy Markdown'), 1500);
    } else if (id === 'dl-doc') downloadDoc(e);
    else if (id === 'reanalyze') {
      e.status = 'new';
      enqueue(e);
    }
  });
  video().addEventListener('timeupdate', () => follow(video().currentTime, false));

  // Drop videos (and their .videodoc.json files) anywhere.
  const body = document.body;
  body.addEventListener('dragover', (ev) => {
    ev.preventDefault();
    body.classList.add('dragging');
  });
  body.addEventListener('dragleave', (ev) => {
    if (ev.relatedTarget === null) body.classList.remove('dragging');
  });
  body.addEventListener('drop', async (ev) => {
    ev.preventDefault();
    body.classList.remove('dragging');
    const entries = await fromFiles([...(ev.dataTransfer?.files ?? [])]);
    if (entries.length) addEntries(entries);
  });
}

async function checkGpu() {
  const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
  state.gpu = !!adapter;
  $('#gpu').textContent = adapter ? `WebGPU · ${adapter.info.vendor} ${adapter.info.architecture}`.trim() : 'No WebGPU';
  $('#gpu').className = adapter ? 'ok' : 'warn';
  if (!adapter) {
    $('#banner').hidden = false;
    $('#banner').textContent = 'WebGPU is unavailable in this browser, so videos cannot be analyzed here (use Chrome or Edge 113+). Reading and searching existing docs still works.';
  }
}

wire();
renderList();
void checkGpu();
