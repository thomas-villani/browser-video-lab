// videodoc app (app.html): open a folder or drop videos, analyze them in this browser, read them
// (the same Markdown view agents get, with a player beside it) and search across them. Nothing
// is uploaded: analysis runs here on WebGPU, docs are saved next to the videos or kept in memory.
// Faces and license plates can be reviewed over the player (review.ts) and blurred into a copy.
// A pipeline preset (presets.ts) adds its track layers and settings to what is analyzed.
import { coversAudio, formatLog, highlight, PIPELINE_SCHEMA, pipelineContext, REDACT_DEFAULTS, redactionLog, resolve, search, toMarkdown, type Pipeline } from '../doc/index.ts';
import { analyze, frameImages } from '../analyze';
import { redact } from '../redact';
import {
  canOpenFolders, download, downloadDoc, fromFiles, loadPlan, loadSavedImages, openFolder, probeDuration, redactTarget, saveDoc, savePlan, storeImages, type Entry,
} from './library';
import { decoderOf, dropPreset, folderPreset, pickedPreset, PRESET_FILE, type Preset } from './presets';
import { analysisSteps, esc, estimate, list, markdownToHtml, marks, stepOf, suggestions, trackCost } from './render';
import { Review } from './review';
import './style.css';

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;

const state = {
  entries: [] as Entry[],
  selected: undefined as Entry | undefined,
  query: '',
  folder: '',
  /** The opened folder (its preset is read from and saved to its top). */
  root: undefined as FileSystemDirectoryHandle | undefined,
  /** The pipeline preset in use, and which of its track layers are ticked. */
  preset: undefined as Preset | undefined,
  tracksOn: new Set<string>(),
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
/** The job running now, to cancel it. */
let current: { e: Entry; ctl: AbortController } | undefined;

const checked = (id: string) => $<HTMLInputElement>(`#opt-${id}`).checked;

/** The preset's track layers. */
const presetTracks = () => Object.keys(state.preset?.pipeline.tracks ?? {});

/** Every track layer this session can compute: faces, plates and the preset's. */
const trackNames = () => ['faces', 'plates', ...presetTracks()];

/** The layers a review or redaction of e covers that can be computed here (faces always). */
function coverWanted(e: Entry): string[] {
  const cover = e.plan?.options.cover ?? REDACT_DEFAULTS.cover;
  return ['faces', ...cover.filter((k) => k !== 'faces' && trackNames().includes(k))];
}

function layersWanted(e?: Entry): string[] {
  const layers = ['speech', 'scenes', 'keyframes'];
  // Redaction needs faces and plates, and the transcript (with word times) for words to bleep;
  // a video analyzed before keeps the layers it has.
  const redacting = e?.redactTo || e?.reviewAfter;
  if (checked('transcript') || e?.wantWords || (e?.redactTo && e.plan && coversAudio(e.plan.options))) layers.push('transcript');
  for (const k of trackNames())
    if (state.tracksOn.has(k) || ((k === 'faces' || k === 'plates') && checked(k)) || (redacting && coverWanted(e).includes(k)) || e?.doc?.layers[k]) layers.push(k);
  return layers;
}

/** Seconds per second of video the ticked preset layers add. */
const presetCost = () => [...state.tracksOn].reduce((s, k) => s + trackCost(decoderOf(state.preset!.pipeline, k)), 0);

const timeFor = (seconds: number) => estimate(seconds, { transcript: checked('transcript'), faces: checked('faces'), plates: checked('plates'), extra: presetCost() });

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
    // A transcript without word times is made again when words are wanted.
    const have = new Set(Object.keys(e.doc?.layers ?? {}));
    if (e.wantWords) have.delete('transcript');
    e.steps = [...analysisSteps(layersWanted(e), have, trackNames()), ...(e.redactTo ? ['Making the copy'] : [])];
    e.step = -1;
    renderList();
    if (state.selected === e && !e.doc) renderPending(e);
    const ctl = new AbortController();
    current = { e, ctl };
    try {
      const t0 = performance.now();
      const onProgress = (p: { pass: string; stage: string; done: number; total: number }) => {
        e.step = p.stage.startsWith('redacting') ? e.steps!.length - 1 : stepOf(p, e.steps!);
        e.progress = p.total > 1 ? `${Math.floor((100 * p.done) / p.total)}%` : '';
        renderEntry(e);
        if (state.selected === e && !e.doc) renderPending(e);
      };
      e.doc = await analyze(await e.file(), layersWanted(e), { doc: e.doc, pipeline: state.preset?.run, onProgress, signal: ctl.signal });
      const saved = await saveDoc(e);
      if (e.redactTo) {
        await redactEntry(e, onProgress, ctl.signal);
        e.status = 'ready';
        e.progress = `redacted in ${((performance.now() - t0) / 1000).toFixed(0)} s`;
      } else {
        e.status = 'ready';
        e.progress = `done in ${((performance.now() - t0) / 1000).toFixed(0)} s`;
        if (!saved) e.progress += ' (not saved: download the doc to keep it)';
        const more = queue.length ? ` ${queue.length} more to go.` : '';
        const said = e.doc.layers.transcript?.items.length;
        if (e.reviewAfter) {
          const found = e.wantWords ? 'Word times found' : 'Faces and plates found';
          if (state.selected !== e) toast(`${found} in <b>${esc(e.name)}</b>.${more}`, { action: 'Review them', onAction: () => void openReview(e), ms: 0 });
        } else toast(`<b>${esc(e.name)}</b> is ready to read${said ? ' and search' : said === 0 ? ' (no speech found, so nothing to search)' : ''}.${more}`, state.selected === e ? {} : { action: 'Open', onAction: () => void showEntry(e) });
      }
    } catch (err) {
      if (ctl.signal.aborted) {
        // Back to how it was: nothing half-done is kept.
        e.status = e.doc ? 'ready' : 'new';
        e.progress = undefined;
        e.reviewAfter = undefined;
        toast(`Cancelled ${e.redactTo ? 'the blurred copy of' : 'analyzing'} <b>${esc(e.name)}</b>${e.redactTo ? '; nothing was saved' : ''}.`);
      } else {
        console.error(err);
        e.status = e.doc ? 'ready' : 'error';
        e.error = (err as Error).message;
        e.progress = e.doc ? `failed: ${e.error}` : undefined;
        toast(`Could not ${e.redactTo ? 'redact' : 'analyze'} <b>${esc(e.name)}</b>: ${esc(e.error)}`, { ms: 0 });
      }
    }
    current = undefined;
    e.redactTo = undefined;
    e.wantWords = undefined;
    renderList();
    if (state.selected === e && review.entry !== e) {
      await showEntry(e);
      if (e.reviewAfter && reviewable(e)) await openReview(e);
    }
    if (reviewable(e)) e.reviewAfter = undefined;
    if (state.query) renderResults();
  }
  running = false;
}

/** Cancel e's job: drop it from the queue, or stop it if it is running (within seconds). */
function cancelJob(e: Entry) {
  if (current?.e === e) return current.ctl.abort();
  const i = queue.indexOf(e);
  if (i < 0) return;
  queue.splice(i, 1);
  e.status = e.doc ? 'ready' : 'new';
  e.redactTo = e.reviewAfter = e.wantWords = undefined;
  renderList();
  if (state.selected === e && !e.doc) renderPending(e);
}

// ---- redaction ----

/** Whether e's doc has every layer its redaction covers (faces, plates unless left out, and the
 *  preset's layers it lists). */
function reviewable(e: Entry) {
  return coverWanted(e).every((k) => !!e.doc?.layers[k]);
}

/** Ask where to save a redacted copy of e (from the click), then queue it: faces and plates first if needed. */
async function startRedact(e: Entry) {
  const to = await redactTarget(e);
  if (!to) return;
  e.redactTo = to;
  if (e.status === 'ready') e.status = 'new';
  enqueue(e);
  const faces = reviewable(e) ? '' : ' Faces and plates are found first, which takes a while.';
  const kept = (e.plan?.options.keep.length ?? 0) + (e.plan?.options.keepPlates.length ?? 0);
  const what = kept ? `faces and plates covered except the ${plural(kept, 'one')} you left visible` : 'every face and plate covered';
  const o = e.plan?.options ?? REDACT_DEFAULTS;
  const sound = coversAudio(o, e.plan?.addedAudio) ? `the words you listed ${o.sound === 'bleep' ? 'bleeped' : 'silenced'}` : 'sound kept';
  toast(`Making <b>${esc(to.name)}</b>: ${what}, ${sound}.${faces} Progress shows in the list.`);
}

async function redactEntry(e: Entry, onProgress: (p: { pass: string; stage: string; done: number; total: number }) => void, signal: AbortSignal) {
  const to = e.redactTo!;
  const doc = e.doc!;
  await loadPlan(e);
  const opts = e.plan?.options ?? REDACT_DEFAULTS;
  const added = e.plan?.added ?? [];
  const addedAudio = e.plan?.addedAudio ?? [];
  const writable = to.file ? ((await to.file.createWritable()) as unknown as WritableStream) : undefined;
  let r: Awaited<ReturnType<typeof redact>>;
  try {
    r = await redact(await e.file(), doc, writable, { ...opts, added, addedAudio, onProgress, signal });
  } catch (err) {
    // A file stream only replaces the file when closed: aborting it keeps whatever was there.
    await writable?.abort().catch(() => {});
    if (to.created) for (const n of [to.name, `${to.name}.redactions.json`]) await to.created.removeEntry(n).catch(() => {});
    throw err;
  }
  if (r.buffer) download(to.name, r.buffer, 'video/mp4');
  const log = formatLog(redactionLog(doc, opts, { name: to.name, frames: r.frames, ms: r.ms }, added, addedAudio));
  if (to.log) {
    const w = await to.log.createWritable();
    await w.write(log);
    await w.close();
  } else download(`${to.name}.redactions.json`, log, 'application/json');
  const where = e.dir ? ' next to the video' : '';
  const watch = to.file ? { action: 'Watch it', onAction: async () => playFile(await to.file!.getFile()) } : {};
  const check = r.audioSeconds ? 'Detection misses some faces and plates, and transcription some words: watch and listen to it before you share it.' : 'Detection misses some faces and plates: watch it before you share it.';
  toast(`Saved <b>${esc(to.name)}</b>${where}, with its redaction log. ${check}`, { ms: 0, ...watch });
}

// ---- face review ----

const review = new Review({
  video: $<HTMLVideoElement>('#player'),
  overlay: $<HTMLCanvasElement>('#overlay'),
  bar: $('#review-bar'),
  panel: $('#review'),
  changed: (e) => {
    clearTimeout(planTimer);
    planTimer = setTimeout(() => void savePlan(e).catch((err) => toast(`Could not save the review: ${esc((err as Error).message)}`)), 400);
  },
  keptWhere: (e) => (e.dir ? `saved next to the video as ${e.name}.redact.json` : 'kept in this tab until you close it'),
  redact: (e) => void startRedact(e),
  findWords: (e) => {
    if (state.gpu === false) return void toast('Transcribing needs WebGPU (Chrome or Edge).');
    closeReview();
    e.wantWords = e.reviewAfter = true;
    e.status = 'new';
    enqueue(e);
    toast(`Transcribing <b>${esc(e.name)}</b> again with word times (about a third of its length). The review opens again when it is done.`);
  },
  close: () => closeReview(),
});
let planTimer: ReturnType<typeof setTimeout> | undefined;

/** Review the faces and plates of e over the player; finds them first when the doc lacks them. */
async function openReview(e: Entry) {
  if (!reviewable(e)) {
    if (state.gpu === false) return void toast('Finding faces and plates needs WebGPU (Chrome or Edge).');
    e.reviewAfter = true;
    e.status = 'new';
    enqueue(e);
    toast(`Finding the ${e.doc?.layers.faces ? 'plates' : 'faces and plates'} in <b>${esc(e.name)}</b> first (about as long as the video, or longer for crowds). The review opens when they are found.`);
    return;
  }
  for (const w of await loadPlan(e)) toast(esc(w), { ms: 0 });
  if (state.selected !== e) await showEntry(e);
  await loadVideo(e);
  $('#read').hidden = true;
  $('#tip').hidden = true;
  $('#review').hidden = false;
  $('#doc-actions').hidden = true;
  review.open(e);
}

function closeReview() {
  if (!review.entry) return;
  review.close();
  $('#review').hidden = true;
  $('#read').hidden = false;
  $('#doc-actions').hidden = false;
  $('#tip').hidden = !state.selected?.doc || tipSeen();
}

/** Play a file (e.g. a redacted copy) in the player; selecting a video goes back to it. */
function playFile(f: File) {
  closeReview();
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
  const action =
    e.status === 'new' || e.status === 'error' ? `<button class="analyze primary">${e.status === 'error' ? 'Retry' : 'Analyze'}</button>`
    : e.status === 'queued' || e.status === 'analyzing' ? `<button class="cancel" title="${e.status === 'queued' ? 'Take it off the queue' : 'Stop now; nothing half-done is kept'}">Cancel</button>`
    : '';
  return `<li data-id="${e.id}" class="${state.selected === e ? 'sel' : ''} ${e.status}">
    <div class="row"><span class="name" title="${esc(e.path)}">${esc(e.path)}</span>${action}</div>
    <div class="meta">${meta} ${chips}</div>${status ? `<div class="status">${status}</div>` : ''}</li>`;
}

/** Update e's row in place, replacing only the parts that changed: progress ticks many times a
 *  second, and a button rebuilt between mousedown and mouseup never gets the click (Cancel). */
function renderEntry(e: Entry) {
  const li = document.querySelector(`#list li[data-id="${e.id}"]`);
  if (!li) return;
  const tpl = document.createElement('template');
  tpl.innerHTML = entryHtml(e).trim();
  const next = tpl.content.firstElementChild!;
  if (li.className !== next.className || li.children.length !== next.children.length) return li.replaceWith(next);
  [...next.children].forEach((c, i) => {
    if (li.children[i].outerHTML !== c.outerHTML) li.children[i].replaceWith(c);
  });
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

/** Put e's video in the player, unless it is there already. */
async function loadVideo(e: Entry, file?: File) {
  if (state.videoUrl && video().dataset.id === String(e.id)) return;
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.videoUrl = URL.createObjectURL(file ?? (await e.file()));
  video().src = state.videoUrl;
  video().dataset.id = String(e.id);
}

async function showEntry(e: Entry, at?: number, marked: string[] = []) {
  closeReview();
  state.selected = e;
  state.marked = new Set(marked);
  state.query = '';
  $<HTMLInputElement>('#q').value = '';
  renderList();
  $('#results').hidden = true;
  $('#empty').hidden = true;
  $('#viewer').hidden = false;
  const file = await e.file();
  await loadVideo(e, file);
  const ready = e.status === 'ready';
  const faces = !!e.doc?.layers.faces;
  $('#doc-actions').innerHTML = e.doc
    ? `<button id="copy-md">Copy Markdown</button><button id="dl-doc">Download doc</button>${ready ? '<button id="reanalyze" title="Compute the layers ticked above that are missing">Add layers</button>' : ''}${
        ready && (faces || state.gpu !== false) ? `<button id="review-faces" title="See what gets blurred over the video; leave faces or plates visible, cover missed ones">Review faces and plates</button>` : ''}${
        ready && state.gpu !== false ? `<button id="redact" title="Save a copy with faces and plates blurred (sound kept) and a redaction log${e.plan ? ', as reviewed' : ''}">Blur faces and plates…</button>` : ''}`
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
    e.status === 'analyzing' ? `<p>Analyzing in this browser:</p><p class="big-steps">${stepsHtml(e)}</p><p class="muted">You can read and search other videos meanwhile.</p><p><button class="cancel-this">Cancel</button></p>`
    : e.status === 'queued' ? `<p>Queued: ${plural(ahead(e), 'video')} ahead of it. Videos are analyzed one at a time.</p><p><button class="cancel-this">Cancel</button></p>`
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
  closeReview();
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
    const { folder, root, entries } = await openFolder();
    if (!entries.length) return void toast(`No videos (mp4, mov, mkv, webm) in ${esc(folder)}.`);
    state.root = root;
    try {
      const p = await folderPreset(root);
      if (p) {
        usePreset(p);
        toast(`Using this folder's <b>${esc(p.pipeline.name ?? PRESET_FILE)}</b> preset (${PRESET_FILE}): see What to analyze.`);
      }
    } catch (err) {
      toast(`This folder's ${PRESET_FILE} was not used: ${esc((err as Error).message)}`, { ms: 0 });
    }
    await addEntries(entries, folder);
  } catch (err) {
    if ((err as DOMException).name !== 'AbortError') alert((err as Error).message);
  }
}

// ---- what to analyze: the layers ticked, the preset, the plan ----

/** What a person calls each layer in the plan line. */
const SAYS: Record<string, string> = { speech: 'when people speak', transcript: "what's said", scenes: 'shots', faces: 'faces', plates: 'plates' };

/** The preset part of "What to analyze": its track layers to tick, and the plan line. */
function renderPreset() {
  const p = state.preset?.pipeline;
  const tracks = Object.keys(p?.tracks ?? {});
  const settings = Object.keys(p?.settings ?? {});
  const about = (k: string) => {
    const t = p!.tracks![k];
    const slow = trackCost(decoderOf(p!, k)) > 1 ? '; slow' : '';
    return `${esc(t.description ?? `${t.model}${t.labels ? `: ${t.labels}` : ''}`)}${slow}`;
  };
  $('#preset').innerHTML = p
    ? `<p class="preset-head"><b>${esc(p.name ?? state.preset!.source)}</b> preset${p.description ? ` <span class="muted">${esc(p.description)}</span>` : ''} <button id="preset-drop" title="Stop using this preset">×</button></p>
       ${tracks.map((k) => `<label><input type="checkbox" data-track="${esc(k)}"${state.tracksOn.has(k) ? ' checked' : ''}> ${esc(k)} <span class="muted">(${about(k)})</span></label>`).join('')}
       ${settings.length ? `<p class="muted">${settings.length} setting${settings.length === 1 ? '' : 's'} from the preset: ${esc(settings.join(', '))}</p>` : ''}`
    : '';
  const wanted = resolve(layersWanted(), pipelineContext(state.preset?.run).producers);
  const frames = wanted.filter((n) => n === 'scenes' || trackNames().includes(n)).map((n) => SAYS[n] ?? n);
  const sound = wanted.filter((n) => n === 'speech' || n === 'transcript').map((n) => SAYS[n]);
  $('#plan-line').textContent = `One pass over the sound (${list(sound)}), one over the frames (${list(frames)}), then key pictures: ${timeFor(3600)} per hour of video.`;
}

/** Use a preset (or none): tick what it computes. */
function usePreset(p: Preset | undefined) {
  if (state.preset) dropPreset(state.preset);
  state.preset = p;
  state.tracksOn = new Set(p ? presetTracks().filter((k) => p.pipeline.layers.includes(k)) : []);
  if (p) for (const id of ['transcript', 'faces', 'plates']) $<HTMLInputElement>(`#opt-${id}`).checked = p.pipeline.layers.includes(id);
  optionsChanged();
}

/** Picked or dropped files that hold a preset (a .json other than docs and reviews) with its
 *  model files: use it. Returns the other files. */
async function takePreset(files: File[]): Promise<File[]> {
  const json = files.find((f) => f.name.endsWith('.json') && !/\.(videodoc|redact|redactions)\.json$/.test(f.name));
  if (!json) return files;
  try {
    const p = await pickedPreset(json, files);
    usePreset(p);
    toast(`Using the <b>${esc(p.pipeline.name ?? json.name)}</b> preset for the videos analyzed from now on: see What to analyze.`);
  } catch (err) {
    toast(`Could not use ${esc(json.name)}: ${esc((err as Error).message)}`, { ms: 0 });
  }
  return files.filter((f) => f !== json && !f.name.endsWith('.onnx'));
}

/** A preset of what is ticked now, with the loaded one's models, track layers and settings. */
function currentPipeline(): Pipeline {
  const p = state.preset?.pipeline;
  const layers = ['keyframes', ...['transcript', 'faces', 'plates'].filter(checked), ...presetTracks().filter((k) => state.tracksOn.has(k))];
  return {
    schema: PIPELINE_SCHEMA,
    name: p?.name ?? (state.folder || 'videodoc'),
    ...(p?.description && { description: p.description }),
    layers,
    ...(p?.settings && { settings: p.settings }),
    ...(p?.models && { models: p.models }),
    ...(p?.tracks && { tracks: p.tracks }),
  };
}

/** Save what is ticked as a preset: at the top of the opened folder (used whenever it is
 *  opened), or as a download. */
async function savePreset() {
  const text = `${JSON.stringify(currentPipeline(), null, 2)}\n`;
  if (!state.root) {
    download(PRESET_FILE, text, 'application/json');
    return void toast(`Downloaded <b>${PRESET_FILE}</b>. Put it at the top of a folder to use it for every video there, or pass it to the CLI (<code>videodoc analyze --pipeline</code>).`);
  }
  try {
    const w = await (await state.root.getFileHandle(PRESET_FILE, { create: true })).createWritable();
    await w.write(text);
    await w.close();
    toast(`Saved as <b>${PRESET_FILE}</b> in ${esc(state.folder)}: used whenever this folder is opened, and by <code>videodoc analyze --pipeline</code>.`);
  } catch (err) {
    toast(`Could not save the preset: ${esc((err as Error).message)}`, { ms: 0 });
  }
}

/** Time estimates and the plan line depend on what is analyzed. */
function optionsChanged() {
  renderPreset();
  renderList();
  if (state.selected && !state.selected.doc && !state.query) renderPending(state.selected);
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
    const files = await takePreset([...(input.files ?? [])]);
    input.value = '';
    if (files.length) await addEntries(await fromFiles(files));
  });
  for (const id of ['#opt-transcript', '#opt-faces', '#opt-plates']) $(id).addEventListener('change', optionsChanged);
  $('#preset').addEventListener('change', (ev) => {
    const k = (ev.target as HTMLInputElement).dataset.track;
    if (!k) return;
    if ((ev.target as HTMLInputElement).checked) state.tracksOn.add(k);
    else state.tracksOn.delete(k);
    optionsChanged();
  });
  $('#preset').addEventListener('click', (ev) => {
    if ((ev.target as HTMLElement).id === 'preset-drop') usePreset(undefined);
  });
  $('#preset-file').addEventListener('change', async (ev) => {
    const input = ev.target as HTMLInputElement;
    const files = [...(input.files ?? [])];
    input.value = '';
    if ((await takePreset(files)).length === files.length) toast('Choose a pipeline preset (.json), with the model files it names.');
  });
  $('#preset-save').addEventListener('click', () => void savePreset());
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
    if (ev.key === '/' && document.activeElement?.tagName !== 'INPUT' && document.activeElement?.tagName !== 'SELECT') {
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
    else if ((ev.target as HTMLElement).classList.contains('cancel')) cancelJob(e);
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
    if ((ev.target as HTMLElement).classList.contains('cancel-this') && state.selected) return cancelJob(state.selected);
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
    else if (id === 'review-faces') void openReview(e);
    else if (id === 'reanalyze') {
      e.status = 'new';
      enqueue(e);
    }
  });
  video().addEventListener('timeupdate', () => {
    if (!review.entry) follow(video().currentTime, false);
  });

  // Drop videos (and their .videodoc.json and .redact.json files) anywhere.
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
    const files = [...(ev.dataTransfer?.files ?? [])];
    const rest = await takePreset(files);
    const entries = await fromFiles(rest);
    if (entries.length) await addEntries(entries);
    else if (rest.length === files.length) toast('Only video files (mp4, mov, mkv, webm) and their .videodoc.json and .redact.json files can be added.');
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
renderPreset();
renderList();
void checkGpu();
