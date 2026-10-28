// videodoc app (app.html): open a folder or drop videos, analyze them in this browser, read them
// (the same Markdown view agents get, with a player beside it) and search across them. Nothing
// is uploaded: analysis runs here on WebGPU, docs are saved next to the videos or kept in memory.
import { formatLog, highlight, REDACT_DEFAULTS, redactionLog, search, toMarkdown, type LayerName } from '../doc/index.ts';
import { analyze, frameImages } from '../analyze';
import { redact } from '../redact';
import {
  canOpenFolders, download, downloadDoc, fromFiles, loadSavedImages, openFolder, probeDuration, redactTarget, saveDoc, storeImages, type Entry,
} from './library';
import { analysisSteps, esc, estimate, markdownToHtml, marks, stepOf, suggestions } from './render';
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
  /** Example queries from the loaded transcripts. */
  suggest: [] as string[],
};

const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? '' : 's'}`;

// ---- toasts: what just happened and what to do next ----

/** Show a message at the bottom of the page; ms = 0 keeps it until dismissed. Returns close(). */
function toast(html: string, o: { action?: string; onAction?: () => void; ms?: number } = {}): () => void {
  const el = document.createElement('div');
  el.className = 'toast';
  el.innerHTML = `<span>${html}</span>${o.action ? `<button class="primary">${esc(o.action)}</button>` : ''}<button class="x" title="Dismiss">×</button>`;
  const close = () => el.remove();
  el.querySelector('.primary')?.addEventListener('click', () => {
    close();
    o.onAction?.();
  });
  el.querySelector('.x')!.addEventListener('click', close);
  $('#toasts').append(el);
  if (o.ms !== 0) setTimeout(close, o.ms ?? 8000);
  return close;
}

/** The "N videos not analyzed yet" toast, closed once they are queued. */
let closeTodo: (() => void) | undefined;

// ---- analysis queue: one video at a time (one GPU) ----

const queue: Entry[] = [];
let running = false;

function layersWanted(e?: Entry): LayerName[] {
  const layers: LayerName[] = ['speech', 'scenes', 'keyframes'];
  if ($<HTMLInputElement>('#opt-transcript').checked) layers.push('transcript');
  // Redaction needs faces; a video analyzed before keeps the layers it has.
  if ($<HTMLInputElement>('#opt-faces').checked || e?.redactTo || e?.doc?.layers.faces) layers.push('faces');
  return layers;
}

const timeFor = (seconds: number) =>
  estimate(seconds, { transcript: $<HTMLInputElement>('#opt-transcript').checked, faces: $<HTMLInputElement>('#opt-faces').checked });

/** Queue every video not analyzed yet (or failed), saying what happens next. */
function analyzeAll() {
  const todo = state.entries.filter((e) => e.status === 'new' || e.status === 'error');
  todo.forEach(enqueue);
  closeTodo?.();
  if (todo.length > 1)
    toast(`Analyzing ${plural(todo.length, 'video')}, one at a time: follow along in the list. You can read and search the others meanwhile.`);
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
    e.progress = '';
    e.steps = [...analysisSteps(layersWanted(e), new Set(Object.keys(e.doc?.layers ?? {}))), ...(e.redactTo ? ['Blurring faces'] : [])];
    e.step = -1;
    renderList();
    if (state.selected === e && !e.doc) renderPending(e);
    try {
      const t0 = performance.now();
      const onProgress = (p: { pass: string; stage: string; done: number; total: number }) => {
        e.step = p.stage === 'redacting' ? e.steps!.length - 1 : stepOf(p, e.steps!);
        e.progress = p.total > 1 ? `${Math.floor((100 * p.done) / p.total)}%` : '';
        renderEntry(e);
        if (state.selected === e && !e.doc) renderPending(e);
      };
      e.doc = await analyze(await e.file(), layersWanted(e), { doc: e.doc, onProgress });
      const saved = await saveDoc(e);
      if (e.redactTo) {
        await redactEntry(e, onProgress);
        e.status = 'ready';
        e.progress = `redacted in ${((performance.now() - t0) / 1000).toFixed(0)} s`;
      } else {
        e.status = 'ready';
        e.progress = `done in ${((performance.now() - t0) / 1000).toFixed(0)} s`;
        if (!saved) e.progress += ' (not saved: download the doc to keep it)';
        const more = queue.length ? ` ${queue.length} more to go.` : '';
        const said = e.doc.layers.transcript?.items.length;
        toast(`<b>${esc(e.name)}</b> is ready to read${said ? ' and search' : said === 0 ? ' (no speech found, so nothing to search)' : ''}.${more}`, state.selected === e ? {} : { action: 'Open', onAction: () => void showEntry(e) });
      }
    } catch (err) {
      console.error(err);
      e.status = e.doc ? 'ready' : 'error';
      e.error = (err as Error).message;
      e.progress = e.doc ? `failed: ${e.error}` : undefined;
      toast(`Could not ${e.redactTo ? 'redact' : 'analyze'} <b>${esc(e.name)}</b>: ${esc(e.error)}`, { ms: 0 });
    }
    e.redactTo = undefined;
    renderList();
    if (state.selected === e) await showEntry(e);
    if (state.query) renderResults();
  }
  running = false;
}

// ---- redaction ----

/** Ask where to save a redacted copy of e (from the click), then queue it: faces first if needed. */
async function startRedact(e: Entry) {
  const to = await redactTarget(e);
  if (!to) return;
  e.redactTo = to;
  if (e.status === 'ready') e.status = 'new';
  enqueue(e);
  const faces = e.doc?.layers.faces ? '' : ' Faces are found first, which takes a while.';
  toast(`Making <b>${esc(to.name)}</b>: every face blurred, sound kept.${faces} Progress shows in the list.`);
}

async function redactEntry(e: Entry, onProgress: (p: { pass: string; stage: string; done: number; total: number }) => void) {
  const to = e.redactTo!;
  const doc = e.doc!;
  const writable = to.file ? ((await to.file.createWritable()) as unknown as WritableStream) : undefined;
  const r = await redact(await e.file(), doc, writable, { ...REDACT_DEFAULTS, onProgress });
  if (r.buffer) download(to.name, r.buffer, 'video/mp4');
  const log = formatLog(redactionLog(doc, REDACT_DEFAULTS, { name: to.name, frames: r.frames, ms: r.ms }));
  if (to.log) {
    const w = await to.log.createWritable();
    await w.write(log);
    await w.close();
  } else download(`${to.name}.redactions.json`, log, 'application/json');
  const where = e.dir ? ' next to the video' : '';
  const watch = to.file ? { action: 'Watch it', onAction: async () => playFile(await to.file!.getFile()) } : {};
  toast(`Saved <b>${esc(to.name)}</b>${where}, with its redaction log. Detection misses some faces: watch it before you share it.`, { ms: 0, ...watch });
}

/** Play a file (e.g. a redacted copy) in the player; selecting a video goes back to it. */
function playFile(f: File) {
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.videoUrl = URL.createObjectURL(f);
  video().src = state.videoUrl;
  video().dataset.id = 'file';
  $('#read-status').textContent = `playing ${f.name}`;
  void video().play().catch(() => {});
}

// ---- video list ----

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

/** "✓ Listening → Transcribing 42% → Finding shots", the current step lit. */
function stepsHtml(e: Entry): string {
  const steps = e.steps ?? [];
  const at = e.step ?? -1;
  if (!steps.length) return '<span class="busy">analyzing…</span>';
  return `<span class="steps">${steps
    .map((s, i) => (i < at ? `<span class="done">✓ ${s}</span>` : i === at ? `<span class="cur">${s}${e.progress ? ` ${e.progress}` : '…'}</span>` : `<span>${s}</span>`))
    .join(' → ')}</span>`;
}

/** Videos ahead of e in the queue (including the one being analyzed). */
const ahead = (e: Entry) => queue.indexOf(e) + 1;

function entryHtml(e: Entry): string {
  const d = e.doc;
  const chips = d ? Object.keys(d.layers).map((n) => `<span class="chip">${n}</span>`).join('') : '';
  const meta = d ? `${fmt(d.source.duration)} · ${d.source.height}p` : e.duration ? fmt(e.duration) : '';
  const status =
    e.status === 'analyzing' ? stepsHtml(e)
    : e.status === 'queued' ? `<span class="busy">queued · ${plural(ahead(e), 'video')} ahead</span>`
    : e.status === 'error' ? `<span class="err" title="${esc(e.error ?? '')}">failed: ${esc(e.error ?? '')}</span>`
    : e.status === 'new' ? `<span class="muted">not analyzed yet${e.duration ? ` · takes ${timeFor(e.duration)}` : ''}</span>`
    : e.progress ? `<span class="muted">${esc(e.progress)}</span>` : '';
  const action = e.status === 'new' || e.status === 'error' ? `<button class="analyze primary">${e.status === 'error' ? 'Retry' : 'Analyze'}</button>` : '';
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
    ? `<span>${esc(state.folder || 'Videos')} · ${n}</span>${todo ? `<button id="analyze-all" class="primary">Analyze ${todo === n ? 'all' : todo}</button>` : ''}`
    : '';
  $('#list').innerHTML = state.entries.map(entryHtml).join('');
  updateSearchHints();
}

/** Search box placeholder: what can be searched, with an example query from the transcripts. */
function updateSearchHints() {
  const searchable = state.entries.filter((e) => e.doc?.layers.transcript?.items.length);
  state.suggest = suggestions(searchable.map((e) => e.doc!.layers.transcript!.items.map((i) => i.text)));
  $<HTMLInputElement>('#q').placeholder = searchable.length
    ? `Search what's said in ${plural(searchable.length, 'video')}${state.suggest.length ? `, e.g. ${state.suggest[0]}` : ''}  ( / )`
    : state.entries.length ? "Analyze a video to search what's said in it" : "Add videos, then search what's said in them";
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
    ? `<button id="copy-md">Copy Markdown</button><button id="dl-doc">Download doc</button>${e.status === 'ready' ? `<button id="reanalyze" title="Compute the layers ticked above that are missing">Add layers</button>${state.gpu === false ? '' : '<button id="redact" title="Save a copy with every face blurred (sound kept) and a redaction log">Blur faces…</button>'}` : ''}`
    : '';
  $('#tip').hidden = !e.doc || tipSeen();
  $('#read-status').textContent = '';
  if (!e.doc) {
    renderPending(e);
  } else {
    renderRead(e);
    const kfs = e.doc.layers.keyframes?.items ?? [];
    await loadSavedImages(e, kfs);
    renderRead(e);
    const missing = kfs.filter((k) => !e.images.has(k.frame));
    if (missing.length && state.gpu !== false) {
      const status = (s: string) => {
        if (state.selected === e) $('#read-status').textContent = s;
      };
      status(`extracting ${missing.length} keyframes…`);
      try {
        await storeImages(e, missing, await frameImages(file, missing.map((k) => k.t)));
      } catch (err) {
        console.warn(err);
      }
      status('');
      if (state.selected === e) fillImages(e);
    }
  }
  if (at !== undefined) seek(at, true);
}

/** The reading view of a video without a doc: what analysis will do, or how far along it is. */
function renderPending(e: Entry) {
  const time = e.duration ? ` It takes ${timeFor(e.duration)} here.` : '';
  const body =
    e.status === 'analyzing' ? `<p>Analyzing in this browser:</p><p class="big-steps">${stepsHtml(e)}</p><p class="muted">You can read and search other videos meanwhile.</p>`
    : e.status === 'queued' ? `<p>Queued: ${plural(ahead(e), 'video')} ahead of it. Videos are analyzed one at a time.</p>`
    : state.gpu === false ? '<p>Not analyzed yet, and this browser has no WebGPU to analyze it. Use Chrome or Edge, or the <code>videodoc</code> CLI.</p>'
    : `${e.status === 'error' ? `<p class="err">Analysis failed: ${esc(e.error ?? '')}</p>` : ''}
       <p>Not analyzed yet. Analysis transcribes what's said, finds the shots and picks a picture for each, in this browser.${time}</p>
       <p><button class="analyze-this primary">${e.status === 'error' ? 'Try again' : 'Analyze this video'}</button></p>`;
  $('#read').innerHTML = `<h1>${esc(e.name)}</h1>${body}`;
}

const TIP = 'videodoc.tip.seek';
const tipSeen = () => {
  try {
    return localStorage.getItem(TIP) === '1';
  } catch {
    return false;
  }
};

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
  const head = hits.length
    ? `<p class="muted">${plural(hits.length, 'passage')} in ${plural(searchable.length, 'video')}${skipped ? ` · ${plural(skipped, 'video')} not searchable yet (analyze with “What's said” on)` : ''}</p>`
    : searchable.length ? ''
    : `<p>Nothing to search yet: ${state.entries.length ? 'analyze a video first (with “What’s said” on).' : 'add videos and analyze them first.'}</p>`;
  const tryThese = state.suggest.filter((s) => s.toLowerCase() !== state.query.toLowerCase());
  const none = searchable.length && !hits.length
    ? `<p>No matches in ${plural(searchable.length, 'video')}.${tryThese.length ? ` Try: ${tryThese.map((s) => `<button class="suggest">${esc(s)}</button>`).join(' ')}` : ''}</p>`
    : '';
  box.innerHTML =
    head +
    none +
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

async function addEntries(entries: Entry[], folder = '') {
  if (folder) {
    state.entries = entries;
    state.folder = folder;
    state.selected = undefined;
  } else {
    const known = new Set(state.entries.map((e) => e.path));
    entries = entries.filter((e) => !known.has(e.path));
    state.entries.push(...entries);
  }
  renderList();
  const first = entries.find((e) => e.doc) ?? entries[0];
  if (first && !state.query) void showEntry(first);

  // Say what is ready and what still needs analyzing (with how long it takes).
  const todo = entries.filter((e) => !e.doc);
  const ready = entries.length - todo.length;
  for (const e of todo) e.duration = await probeDuration(await e.file());
  renderList();
  if (state.selected && !state.selected.doc) renderPending(state.selected);
  const readyText = ready ? `${plural(ready, 'video')} ready to read and search.` : '';
  if (!todo.length) {
    if (ready) toast(readyText, { ms: 5000 });
    return;
  }
  const total = todo.reduce((s, e) => s + (e.duration ?? 0), 0);
  const time = total && todo.every((e) => e.duration) ? ` Analyzing takes ${timeFor(total)} in this browser.` : '';
  const what = `${readyText} ${plural(todo.length, 'video')} not analyzed yet.`.trim();
  closeTodo?.();
  closeTodo =
    state.gpu === false
      ? toast(`${what} This browser can't analyze them (no WebGPU): use Chrome or Edge, or the videodoc CLI.`, { ms: 0 })
      : toast(`${what}${time}`, { action: todo.length === 1 ? 'Analyze it' : `Analyze ${todo.length}`, onAction: analyzeAll, ms: 0 });
}

async function pickFolder() {
  try {
    const { folder, entries } = await openFolder();
    if (!entries.length) return void toast(`No videos (mp4, mov, mkv, webm) in ${esc(folder)}.`);
    await addEntries(entries, folder);
  } catch (err) {
    if ((err as DOMException).name !== 'AbortError') alert((err as Error).message);
  }
}

// ---- wiring ----

function wire() {
  $('#open-folder').hidden = !canOpenFolders();
  $('#open-folder').addEventListener('click', () => void pickFolder());
  $('#start-folder').hidden = !canOpenFolders();
  $('#start-folder').addEventListener('click', () => void pickFolder());
  $('#start-files').addEventListener('click', () => $('#add-files').click());
  $('#add-files').addEventListener('change', async (ev) => {
    const input = ev.target as HTMLInputElement;
    const files = [...(input.files ?? [])];
    input.value = '';
    await addEntries(await fromFiles(files));
  });
  // Time estimates depend on what is analyzed.
  for (const id of ['#opt-transcript', '#opt-faces'])
    $(id).addEventListener('change', () => {
      renderList();
      if (state.selected && !state.selected.doc && !state.query) renderPending(state.selected);
    });
  $('#tip-ok').addEventListener('click', () => {
    $('#tip').hidden = true;
    try {
      localStorage.setItem(TIP, '1');
    } catch {
      // Private window: the tip comes back next time.
    }
  });
  $('#q').addEventListener('input', onSearch);
  document.addEventListener('keydown', (ev) => {
    if (ev.key === '/' && document.activeElement?.tagName !== 'INPUT') {
      ev.preventDefault();
      $<HTMLInputElement>('#q').focus();
    }
  });

  $('#list-head').addEventListener('click', (ev) => {
    if ((ev.target as HTMLElement).id === 'analyze-all') analyzeAll();
  });
  $('#list').addEventListener('click', (ev) => {
    const li = (ev.target as HTMLElement).closest('li');
    const e = li && state.entries.find((x) => x.id === Number(li.dataset.id));
    if (!e) return;
    if ((ev.target as HTMLElement).classList.contains('analyze')) enqueue(e);
    else void showEntry(e);
  });
  $('#results').addEventListener('click', (ev) => {
    const target = ev.target as HTMLElement;
    if (target.classList.contains('suggest')) {
      $<HTMLInputElement>('#q').value = target.textContent ?? '';
      return onSearch();
    }
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
    else if (id === 'redact') void startRedact(e);
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
    if (entries.length) await addEntries(entries);
    else toast('Only video files (mp4, mov, mkv, webm) and their .videodoc.json files can be added.');
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
