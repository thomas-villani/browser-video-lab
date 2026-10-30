// Face and plate review: the regions drawn over the player frame by frame, with the cover previewed
// live (the same drawing the redacted copy gets). Click a face or plate to leave it visible, drag
// to cover one the detector missed. Tracks are named by key, `<layer>:<id>` (`faces:3`, hand
// tracks are faces with negative ids, `plates:7`, a pipeline's `vehicles:2`): ids are per layer.
// Which layers are covered besides faces is ticked in the panel. The choices are the entry's
// RedactPlan (saved as <video>.redact.json for folder entries), which Blur faces and `videodoc
// redact` use. Nothing is decoded here: the
// boxes come from the doc, the picture from the <video> element. Spoken words to bleep are
// listed above the shots; while the preview plays, the player is muted over them and the tone
// plays instead.
import {
  type AudioSpan,
  audioCover,
  audioSpans,
  type Box,
  covered,
  cursorFor,
  handTrack,
  keptIds,
  newPlan,
  REDACT_SOUNDS,
  REDACT_STYLES,
  type RedactionCursor,
  type RedactKind,
  type RedactPlan,
  type RedactSound,
  type RedactStyle,
  type Region,
  type SceneItem,
  setBox,
  setEnd,
  speakerName,
  suggestNames,
  type Track,
  trackBox,
  trackLayers,
  tracksOf,
  type VideoDoc,
  withSpeakers,
} from '../doc/index.ts';
import { coverer } from '../redact';
import { type Entry, speakerEdits } from './library';
import { esc } from './render';

export interface ReviewHost {
  video: HTMLVideoElement;
  overlay: HTMLCanvasElement;
  /** The transport bar under the player. */
  bar: HTMLElement;
  /** Where the face list goes (the reading column). */
  panel: HTMLElement;
  /** The plan changed: save it. Returns where it is kept, for the summary line. */
  changed: (e: Entry) => void;
  /** Where the plan is kept ("saved next to the video" or "kept in this tab"). */
  keptWhere: (e: Entry) => string;
  redact: (e: Entry) => void;
  /** Transcribe e again with word times (its transcript has none), then reopen the review. */
  findWords: (e: Entry) => void;
  close: () => void;
}

const STYLE_NAMES: Record<RedactStyle, string> = { blur: 'Blur', pixelate: 'Pixelate', fill: 'Solid box' };
const SOUND_NAMES: Record<RedactSound, string> = { bleep: 'Bleep', mute: 'Silence' };
/** Spoken matches listed until more are asked for. */
const WORDS_LISTED = 100;
/** Faces listed per shot (longest first) until more are asked for, a page at a time (a crowded
 *  shot has thousands). */
const PER_SHOT = 24;
const MORE = 120;
/** Default length of a face added by hand, seconds (cut at the end of its shot). */
const HAND_SECONDS = 2;

/** A track's key in the review: layer and id. */
const keyOf = (kind: RedactKind, id: number) => `${kind}:${id}`;
const parseKey = (k: string) => ({ kind: k.slice(0, k.lastIndexOf(':')) as RedactKind, id: Number(k.slice(k.lastIndexOf(':') + 1)) });
const isHand = (k: string) => k.startsWith('faces:-');
/** "face", "plate", or a pipeline layer's name made singular ("vehicles" -> "vehicle"). */
const noun = (kind: RedactKind, n = 1) => {
  const one = kind === 'faces' ? 'face' : kind === 'plates' ? 'plate' : kind.replace(/s$/, '');
  return n === 1 ? one : `${one}s`;
};
/** Box colors by layer: faces white, plates cyan, a pipeline's layers after them (hand boxes are amber). */
const KIND_COLORS: [string, string][] = [
  ['#ffffff', 'white'],
  ['#67e8f9', 'cyan'],
  ['#c084fc', 'violet'],
  ['#a3e635', 'lime'],
  ['#f472b6', 'pink'],
  ['#fb923c', 'orange'],
];

const fmtT = (s: number, frac = false) => {
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${frac ? r.toFixed(2).padStart(5, '0') : String(Math.floor(r)).padStart(2, '0')}`;
};

export class Review {
  private h: ReviewHost;
  private e?: Entry;
  private doc!: VideoDoc;
  private plan!: RedactPlan;
  private cursor!: RedactionCursor;
  private cover!: ReturnType<typeof coverer>;
  private byId = new Map<string, Track>();
  private shots: SceneItem[] = [];
  /** Track keys by shot: hand tracks first, then faces, plates and other layers, longest first. */
  private shotTracks: string[][] = [];
  private fps = 25;
  /** Presentation time of frame 0 (first keyframe's t minus its frame time). */
  private off = 0;
  private frame = 0;
  private shot = -1;
  private regions: Region[] = [];
  private preview = true;
  private selected?: string;
  /** Tracks listed per shot and layer ("3:faces" faces and hand boxes, "3:plates"), when more
   *  than PER_SHOT were asked for. */
  private listed = new Map<string, number>();
  private hover?: string;
  private press?: { x: number; y: number; cx: number; cy: number; moved: boolean };
  private drag?: Box;
  private thumbs = new WeakMap<Entry, Map<string, string>>();
  /** Spoken matches and spans added by hand (audioSpans), and the merged stretches covered. */
  private spans: AudioSpan[] = [];
  private bleeps: [number, number][] = [];
  /** Whether the transcript has word times (matching needs them). */
  private hasWords = false;
  private wordsListed = WORDS_LISTED;
  private wordsTimer?: ReturnType<typeof setTimeout>;
  /** The preview's tone, made on first use (an AudioContext needs a click first). */
  private tone?: { ctx: AudioContext; gain: GainNode };
  /** Whether the preview has the player muted over a covered word now. */
  private muting = false;
  private rvfc = 0;
  private unlisten: (() => void)[] = [];
  private resize = new ResizeObserver(() => this.layout());
  private thumbCanvas = document.createElement('canvas');
  /** A second, hidden player that seeks to faces in the list scrolled into view, for their
   *  thumbnails (the main player stays where the person is). */
  private filler = document.createElement('video');
  private wanted: string[] = [];
  private filling = false;
  private seen = new IntersectionObserver((items) => {
    for (const it of items) {
      const id = (it.target as HTMLElement).dataset.track!;
      if (it.isIntersecting && !this.wanted.includes(id)) this.wanted.push(id);
    }
    void this.fill();
  });

  constructor(h: ReviewHost) {
    this.h = h;
    this.thumbCanvas.width = this.thumbCanvas.height = 72;
    this.filler.muted = true;
    this.filler.preload = 'auto';
  }

  get entry() {
    return this.e;
  }

  open(e: Entry) {
    this.close();
    // Speakers by the names given them (to bleep one).
    const doc = withSpeakers(e.doc!, speakerEdits(e));
    this.e = e;
    this.doc = doc;
    e.plan ??= newPlan(doc);
    this.plan = e.plan;
    if (!this.thumbs.has(e)) this.thumbs.set(e, new Map());
    this.fps = doc.source.fps;
    const k = doc.layers.keyframes?.items[0];
    this.off = k ? k.t - k.frame / this.fps : 0;
    this.shots = doc.layers.scenes?.items.length ? doc.layers.scenes.items : [{ t0: 0, t1: doc.source.duration, f0: 0, f1: doc.source.frames - 1, score: 0 }];
    this.selected = this.hover = undefined;
    this.listed.clear();
    this.wordsListed = WORDS_LISTED;
    this.preview = true;
    this.rebuild();

    const { video, overlay, bar } = this.h;
    overlay.width = doc.source.width;
    overlay.height = doc.source.height;
    overlay.hidden = false;
    this.cover = coverer(overlay, this.plan.options.style);
    this.filler.src = video.currentSrc || video.src;
    video.controls = false;
    bar.hidden = false;
    bar.innerHTML = `<button data-act="play" class="play" title="Play or pause (Space)">▶</button>
      <button data-act="prev-shot" title="Previous shot ( [ )">⏮</button><button data-act="prev" title="Back one frame (←; Shift: 10)">‹</button>
      <button data-act="next" title="Forward one frame (→; Shift: 10)">›</button><button data-act="next-shot" title="Next shot ( ] )">⏭</button>
      <input type="range" min="0" max="${doc.source.frames - 1}" step="1" value="0" aria-label="Frame">
      <span class="time"></span>
      <label title="Show the video as the copy will look; off: boxes only"><input type="checkbox" data-act="preview" checked> Preview</label>`;
    this.renderPanel();

    const on = <K extends keyof HTMLElementEventMap>(el: HTMLElement | Document, type: K, fn: (ev: HTMLElementEventMap[K]) => void) => {
      el.addEventListener(type, fn as EventListener);
      this.unlisten.push(() => el.removeEventListener(type, fn as EventListener));
    };
    on(bar, 'click', (ev) => {
      const b = (ev.target as HTMLElement).closest<HTMLElement>('[data-act]');
      // Unfocused, so Space plays and pauses rather than pressing the button again.
      if (b?.tagName === 'BUTTON') b.blur();
      this.onBar(b?.dataset.act);
    });
    on(bar, 'input', (ev) => {
      const t = ev.target as HTMLInputElement;
      if (t.type === 'range') this.seekFrame(Number(t.value));
      else if (t.dataset.act === 'preview') {
        this.preview = t.checked;
        this.draw();
      }
    });
    on(this.h.panel, 'click', (ev) => this.onPanel(ev.target as HTMLElement));
    on(this.h.panel, 'change', (ev) => {
      const t = ev.target as HTMLSelectElement & HTMLInputElement;
      const o = this.plan.options;
      if (t.id === 'rv-style') {
        o.style = t.value as RedactStyle;
        this.cover = coverer(overlay, o.style);
        this.changed();
      } else if (t.dataset.cover) {
        const k = t.dataset.cover;
        o.cover = [...o.cover.filter((c) => c !== k), ...(t.checked ? [k] : [])];
        if (this.selected && parseKey(this.selected).kind === k) this.selected = undefined;
        this.rebuild();
        this.renderPanel();
        this.draw();
        this.changed();
      } else if (t.id === 'rv-profanity') {
        o.profanity = t.checked;
        this.wordsChanged();
      } else if (t.dataset.bleepSpeaker !== undefined) {
        const who = t.dataset.bleepSpeaker;
        o.speakers = [...o.speakers.filter((s) => s !== who), ...(t.checked ? [who] : [])];
        this.wordsChanged();
      } else if (t.id === 'rv-sound') {
        o.sound = t.value as RedactSound;
        this.wordsChanged();
      }
    });
    on(this.h.panel, 'input', (ev) => {
      const t = ev.target as HTMLTextAreaElement;
      if (t.id !== 'rv-words') return;
      clearTimeout(this.wordsTimer);
      this.wordsTimer = setTimeout(() => {
        this.plan.options.words = t.value
          .split('\n')
          .map((w) => w.trim())
          .filter(Boolean);
        this.wordsChanged();
      }, 400);
    });
    on(overlay, 'pointerdown', (ev) => this.onDown(ev));
    on(overlay, 'pointermove', (ev) => this.onMove(ev));
    on(overlay, 'pointerup', (ev) => this.onUp(ev));
    on(overlay, 'pointerleave', () => {
      if (!this.press && this.hover !== undefined) {
        this.hover = undefined;
        this.draw();
      }
    });
    on(document, 'keydown', (ev) => this.onKey(ev));
    on(video, 'seeked', () => this.atTime(video.currentTime));
    on(video, 'loadeddata', () => this.atTime(video.currentTime));
    on(video, 'play', () => this.playing(true));
    on(video, 'pause', () => this.playing(false));
    this.resize.observe(video);
    on(video, 'loadedmetadata', () => this.layout());
    const loop = (_: number, meta: VideoFrameCallbackMetadata) => {
      this.at(this.frameAt(meta.mediaTime));
      this.rvfc = video.requestVideoFrameCallback(loop);
    };
    this.rvfc = video.requestVideoFrameCallback(loop);
    this.layout();
    this.playing(!video.paused);
    this.atTime(video.currentTime);
  }

  close() {
    if (!this.e) return;
    const { video, overlay, bar } = this.h;
    clearTimeout(this.wordsTimer);
    this.sound(false);
    video.cancelVideoFrameCallback(this.rvfc);
    this.unlisten.forEach((f) => f());
    this.unlisten = [];
    this.resize.disconnect();
    this.seen.disconnect();
    this.wanted = [];
    this.filler.removeAttribute('src');
    this.filler.load();
    overlay.hidden = true;
    bar.hidden = true;
    bar.innerHTML = '';
    video.controls = true;
    this.e = undefined;
  }

  // ---- frames and time ----

  private frameAt(t: number) {
    return Math.min(this.doc.source.frames - 1, Math.max(0, Math.floor((t - this.off) * this.fps + 0.01)));
  }

  private timeOf(frame: number) {
    return this.off + (frame + 0.5) / this.fps;
  }

  private atTime(t: number) {
    this.at(this.frameAt(t));
  }

  private at(frame: number) {
    this.frame = frame;
    this.regions = this.cursor.regions(frame);
    this.draw();
    const v = this.h.video;
    const t = v.currentTime;
    this.sound(this.preview && !v.paused && this.bleeps.some(([a, b]) => t >= a && t < b));
    const range = this.h.bar.querySelector<HTMLInputElement>('input[type=range]');
    if (range && document.activeElement !== range) range.value = String(frame);
    const time = this.h.bar.querySelector('.time');
    if (time) time.textContent = `${fmtT(frame / this.fps, true)} · frame ${frame} · ${this.regions.length} covered`;
    const shot = this.shotOf(frame);
    if (shot !== this.shot) {
      this.shot = shot;
      for (const li of this.h.panel.querySelectorAll('.rv-shot.cur')) li.classList.remove('cur');
      const li = this.h.panel.querySelector(`.rv-shot[data-shot="${shot}"]`);
      li?.classList.add('cur');
      li?.scrollIntoView({ block: 'nearest' });
    }
  }

  private seekFrame(frame: number) {
    const f = Math.min(this.doc.source.frames - 1, Math.max(0, frame));
    this.h.video.currentTime = this.timeOf(f);
    this.at(f);
  }

  private shotOf(frame: number) {
    let lo = 0;
    let hi = this.shots.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.shots[mid].f0 <= frame) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  private playing(on: boolean) {
    const b = this.h.bar.querySelector('.play');
    if (b) b.textContent = on ? '❚❚' : '▶';
    if (!on) this.sound(false);
  }

  /** The preview of the audio cover: the player muted over a covered word, the tone on for bleeps.
   *  Checked once per video frame, so edges are a frame late (the copy is exact). */
  private sound(on: boolean) {
    if (on === this.muting) return;
    this.muting = on;
    const v = this.h.video;
    v.muted = on;
    const bleep = on && this.plan.options.sound === 'bleep';
    if (bleep && !this.tone) {
      const ctx = new AudioContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.frequency.value = 1000;
      gain.gain.value = 0;
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      this.tone = { ctx, gain };
    }
    if (!this.tone) return;
    void this.tone.ctx.resume();
    this.tone.gain.gain.setTargetAtTime(bleep ? 0.25 * v.volume : 0, this.tone.ctx.currentTime, 0.004);
  }

  // ---- drawing ----

  /** Fit the overlay over the picture inside the <video> box (letterboxed). */
  private layout() {
    const { video, overlay } = this.h;
    const vw = video.videoWidth || this.doc.source.width;
    const vh = video.videoHeight || this.doc.source.height;
    const s = Math.min(video.clientWidth / vw, video.clientHeight / vh);
    if (!(s > 0)) return;
    const [w, h] = [vw * s, vh * s];
    Object.assign(overlay.style, {
      left: `${video.offsetLeft + (video.clientWidth - w) / 2}px`,
      top: `${video.offsetTop + (video.clientHeight - h) / 2}px`,
      width: `${w}px`,
      height: `${h}px`,
    });
    this.draw();
  }

  private keptBoxes(): Region[] {
    const out: Region[] = [];
    const o = this.plan.options;
    for (const kind of this.kinds()) {
      for (const id of keptIds(o, kind)) {
        const t = this.byId.get(keyOf(kind, id));
        const b = t && trackBox(t, this.frame);
        const g = b && this.cursor.grow(b);
        if (g) out.push({ id, kind, box: g, held: false });
      }
    }
    return out;
  }

  /** The layers this review covers: faces, and the covered track layers the doc has. */
  private kinds(): RedactKind[] {
    return ['faces', ...covered(this.doc, this.plan.options).filter((k) => k !== 'faces')];
  }

  private keep(kind: RedactKind) {
    const o = this.plan.options;
    if (kind === 'faces') return o.keep;
    if (kind === 'plates') return o.keepPlates;
    return (o.keepTracks[kind] ??= []);
  }

  /** [hex, name] of the layer's box color, by its place in the doc (the same whichever are covered). */
  private color(kind: RedactKind) {
    const all = ['faces', ...trackLayers(this.doc).filter((k) => k !== 'faces')];
    return KIND_COLORS[Math.max(0, all.indexOf(kind)) % KIND_COLORS.length];
  }

  private isKept(k: string) {
    const { kind, id } = parseKey(k);
    return this.keep(kind).includes(id);
  }

  private draw() {
    if (!this.e) return;
    const { overlay, video } = this.h;
    const ctx = overlay.getContext('2d')!;
    const { width: W, height: H } = overlay;
    ctx.clearRect(0, 0, W, H);
    const kept = this.keptBoxes();
    if (video.readyState >= 2) this.capture([...this.regions, ...kept]);
    if (this.preview && video.readyState >= 2 && this.regions.length) {
      ctx.drawImage(video, 0, 0, W, H);
      this.cover(this.regions.map((r) => r.box));
    }
    // Source pixels per CSS pixel, so lines and labels keep their size on screen.
    const px = W / (overlay.clientWidth || W);
    const stroke = (b: Box, color: string, width: number, dash: number[] = []) => {
      ctx.setLineDash(dash.map((d) => d * px));
      ctx.lineWidth = (width + 2) * px;
      ctx.strokeStyle = 'rgb(0 0 0 / 0.55)';
      ctx.strokeRect(...b);
      ctx.lineWidth = width * px;
      ctx.strokeStyle = color;
      ctx.strokeRect(...b);
    };
    const label = (b: Box, text: string, color: string) => {
      ctx.setLineDash([]);
      ctx.font = `600 ${12 * px}px system-ui, sans-serif`;
      const w = ctx.measureText(text).width + 8 * px;
      const y = b[1] > 18 * px ? b[1] - 17 * px : b[1] + b[3] + 1 * px;
      ctx.fillStyle = color;
      ctx.fillRect(b[0], y, w, 16 * px);
      ctx.fillStyle = '#fff';
      ctx.fillText(text, b[0] + 4 * px, y + 12 * px);
    };
    const name = (r: Region) => `${r.kind === 'faces' ? '' : `${noun(r.kind)} `}#${r.id}`;
    for (const r of this.regions) {
      const k = keyOf(r.kind, r.id);
      const on = k === this.selected || k === this.hover;
      const color = r.id < 0 ? '#f59e0b' : this.color(r.kind)[0];
      stroke(r.box, on ? '#7ea2ff' : color, on ? 3 : 1.5, r.held ? [4, 3] : []);
      if (on) label(r.box, r.id < 0 ? 'added by hand' : `${name(r)} blurred · click: leave visible`, '#2b5fd9');
    }
    for (const r of kept) {
      const k = keyOf(r.kind, r.id);
      const on = k === this.selected || k === this.hover;
      stroke(r.box, '#22c55e', on ? 3 : 2, [6, 4]);
      label(r.box, on ? `${name(r)} visible · click: blur` : 'visible', '#15803d');
    }
    if (this.drag) stroke(this.drag, '#f59e0b', 2, [6, 4]);
  }

  /** Thumbnails for the face list, cropped from the frame on screen as faces come by. */
  private capture(regions: Region[]) {
    const thumbs = this.thumbs.get(this.e!)!;
    let n = 0;
    for (const r of regions) {
      const k = keyOf(r.kind, r.id);
      if (r.held || thumbs.has(k) || n >= 4) continue;
      n++;
      this.thumb(k, this.h.video, r.box);
    }
  }

  /** Crop `box` (source pixels, square around its center) from `v`'s frame as track `id`'s thumbnail. */
  private thumb(id: string, v: HTMLVideoElement, [x, y, w, h]: Box) {
    const sx = v.videoWidth / this.doc.source.width;
    const sy = v.videoHeight / this.doc.source.height;
    const side = Math.max(w, h);
    const c = this.thumbCanvas.getContext('2d')!;
    c.fillStyle = '#000';
    c.fillRect(0, 0, 72, 72);
    c.drawImage(v, (x + w / 2 - side / 2) * sx, (y + h / 2 - side / 2) * sy, side * sx, side * sy, 0, 0, 72, 72);
    const url = this.thumbCanvas.toDataURL('image/jpeg', 0.8);
    this.thumbs.get(this.e!)!.set(id, url);
    const ph = this.h.panel.querySelector(`.face[data-track="${id}"] .ph`);
    if (ph) ph.outerHTML = `<img src="${url}" alt="">`;
  }

  /** Thumbnails for the faces scrolled into view, one seek of the hidden player each. */
  private async fill() {
    if (this.filling) return;
    this.filling = true;
    const e = this.e;
    try {
      while (this.wanted.length && this.e === e) {
        const id = this.wanted.shift()!;
        const t = this.byId.get(id);
        if (!t || this.thumbs.get(e!)!.has(id)) continue;
        const frame = Math.round((t.f0 + t.f1) / 2);
        const v = this.filler;
        await new Promise<void>((resolve) => {
          const done = () => {
            clearTimeout(timer);
            v.removeEventListener('seeked', done);
            resolve();
          };
          const timer = setTimeout(done, 3000);
          v.addEventListener('seeked', done);
          v.currentTime = this.timeOf(frame);
        });
        const box = trackBox(t, frame);
        if (this.e === e && box && v.readyState >= 2) this.thumb(id, v, this.cursor.grow(box) ?? box);
      }
    } finally {
      this.filling = false;
    }
  }

  // ---- plan ----

  private rebuild() {
    const all: [string, Track][] = this.plan.added.map((t) => [keyOf('faces', t.id), t]);
    for (const kind of this.kinds()) for (const t of tracksOf(this.doc, kind)!.items) all.push([keyOf(kind, t.id), t]);
    this.byId = new Map(all);
    this.cursor = cursorFor(this.doc, this.plan.options, this.plan.added);
    const kinds = this.kinds();
    const rank = (k: string) => (isHand(k) ? 0 : 1 + kinds.indexOf(parseKey(k).kind));
    const len = (k: string) => this.byId.get(k)!.f1 - this.byId.get(k)!.f0;
    this.shotTracks = this.shots.map(() => []);
    for (const [k, t] of all) this.shotTracks[this.shotOf(t.f0)].push(k);
    for (const ids of this.shotTracks) ids.sort((a, b) => rank(a) - rank(b) || len(b) - len(a));
    this.regions = this.cursor.regions(this.frame);
    this.rebuildWords();
  }

  private rebuildWords() {
    const t = this.doc.layers.transcript;
    this.hasWords = !!t?.items.some((i) => i.words);
    // Without word times only the spans added by hand (and speakers) can be covered; a speaker
    // no longer in the doc (renamed, found again) is dropped.
    const sp = this.doc.layers.speakers?.items ?? [];
    this.plan.options.speakers = this.plan.options.speakers.filter((n) => sp.some((s) => speakerName(s) === n));
    const o = this.hasWords ? this.plan.options : { ...this.plan.options, words: [], profanity: false };
    this.spans = audioSpans(this.doc, o, this.plan.addedAudio ?? []);
    this.bleeps = audioCover(this.spans);
  }

  /** The word list or its settings changed: save, and redo the matches (not the whole panel,
   *  so the word box keeps its focus). */
  private wordsChanged() {
    this.rebuildWords();
    this.h.changed(this.e!);
    this.renderMatches();
    this.renderTitle();
  }

  private toggleWord(key: string) {
    const keep = this.plan.options.keepWords;
    const i = keep.indexOf(key);
    if (i >= 0) keep.splice(i, 1);
    else keep.push(key);
    this.wordsChanged();
  }

  private changed() {
    this.rebuild();
    this.h.changed(this.e!);
    this.renderPanel();
    this.draw();
  }

  private toggleKeep(k: string) {
    const { kind, id } = parseKey(k);
    const keep = this.keep(kind);
    const i = keep.indexOf(id);
    if (i >= 0) keep.splice(i, 1);
    else keep.push(id);
    keep.sort((a, b) => a - b);
    this.changed();
  }

  private replaceHand(t: Track) {
    this.plan.added = this.plan.added.map((x) => (x.id === t.id ? t : x));
    this.changed();
  }

  private deleteHand(k: string) {
    const id = parseKey(k).id;
    this.plan.added = this.plan.added.filter((x) => x.id !== id);
    if (this.selected === k) this.selected = undefined;
    this.thumbs.get(this.e!)!.delete(k);
    this.changed();
  }

  /** A box drawn on the frame: moves the selected hand track there, or adds a new one. */
  private drawn(box: Box) {
    const sel = this.selected !== undefined && isHand(this.selected) ? this.byId.get(this.selected) : undefined;
    if (sel) {
      this.thumbs.get(this.e!)!.delete(this.selected!);
      return this.replaceHand(setBox(sel, this.frame, box));
    }
    const id = Math.min(0, ...this.plan.added.map((t) => t.id)) - 1;
    const end = Math.min(this.shots[this.shotOf(this.frame)].f1, this.frame + Math.round(HAND_SECONDS * this.fps));
    this.plan.added = [...this.plan.added, handTrack(id, this.frame, box, end)];
    this.selected = keyOf('faces', id);
    this.changed();
  }

  // ---- input ----

  private point(ev: PointerEvent) {
    const r = this.h.overlay.getBoundingClientRect();
    return { x: ((ev.clientX - r.left) / r.width) * this.doc.source.width, y: ((ev.clientY - r.top) / r.height) * this.doc.source.height };
  }

  /** The smallest box under (x, y): covered ones and kept ones. */
  private hit(x: number, y: number): string | undefined {
    let best: Region | undefined;
    for (const r of [...this.regions, ...this.keptBoxes()]) {
      const [bx, by, bw, bh] = r.box;
      if (x >= bx && x <= bx + bw && y >= by && y <= by + bh && (!best || bw * bh < best.box[2] * best.box[3])) best = r;
    }
    return best && keyOf(best.kind, best.id);
  }

  private onDown(ev: PointerEvent) {
    if (ev.button !== 0) return;
    this.h.video.pause();
    const p = this.point(ev);
    this.press = { ...p, cx: ev.clientX, cy: ev.clientY, moved: false };
    this.h.overlay.setPointerCapture(ev.pointerId);
  }

  private onMove(ev: PointerEvent) {
    const p = this.point(ev);
    if (this.press) {
      if (Math.hypot(ev.clientX - this.press.cx, ev.clientY - this.press.cy) > 5) this.press.moved = true;
      if (this.press.moved) {
        const { width: W, height: H } = this.doc.source;
        const x0 = Math.max(0, Math.min(p.x, this.press.x));
        const y0 = Math.max(0, Math.min(p.y, this.press.y));
        this.drag = [x0, y0, Math.min(W, Math.max(p.x, this.press.x)) - x0, Math.min(H, Math.max(p.y, this.press.y)) - y0];
        this.draw();
      }
      return;
    }
    const id = this.hit(p.x, p.y);
    this.h.overlay.style.cursor = id === undefined ? 'crosshair' : 'pointer';
    if (id !== this.hover) {
      this.hover = id;
      this.draw();
    }
  }

  private onUp(ev: PointerEvent) {
    const press = this.press;
    this.press = undefined;
    if (!press) return;
    const drag = this.drag;
    this.drag = undefined;
    if (press.moved) {
      if (drag && drag[2] >= 6 && drag[3] >= 6) this.drawn(drag);
      else this.draw();
      return;
    }
    const p = this.point(ev);
    const id = this.hit(p.x, p.y);
    this.selected = id;
    if (id !== undefined && !isHand(id)) return this.toggleKeep(id);
    this.renderPanel();
    this.draw();
  }

  private onKey(ev: KeyboardEvent) {
    const tag = (document.activeElement as HTMLElement | null)?.tagName;
    if (tag === 'INPUT' && (document.activeElement as HTMLInputElement).type !== 'range' && (document.activeElement as HTMLInputElement).type !== 'checkbox') return;
    if (tag === 'SELECT' || tag === 'TEXTAREA') return;
    const step = ev.shiftKey ? 10 : 1;
    const act: Record<string, () => void> = {
      ' ': () => this.onBar('play'),
      ArrowLeft: () => this.seekFrame(this.frame - step),
      ArrowRight: () => this.seekFrame(this.frame + step),
      '[': () => this.onBar('prev-shot'),
      ']': () => this.onBar('next-shot'),
      Escape: () => {
        this.selected = undefined;
        this.renderPanel();
        this.draw();
      },
      Delete: () => this.selected !== undefined && isHand(this.selected) && this.deleteHand(this.selected),
    };
    const fn = act[ev.key];
    if (!fn) return;
    ev.preventDefault();
    if (ev.key !== ' ') this.h.video.pause();
    fn();
  }

  private onBar(act?: string) {
    const v = this.h.video;
    if (act === 'play') {
      if (v.paused) void v.play();
      else v.pause();
      return;
    }
    if (!act || act === 'preview') return;
    v.pause();
    const s = this.shotOf(this.frame);
    if (act === 'prev') this.seekFrame(this.frame - 1);
    else if (act === 'next') this.seekFrame(this.frame + 1);
    // Back to this shot's start, or the one before when already there.
    else if (act === 'prev-shot') this.seekFrame(this.frame > this.shots[s].f0 ? this.shots[s].f0 : this.shots[Math.max(0, s - 1)].f0);
    else if (act === 'next-shot' && s + 1 < this.shots.length) this.seekFrame(this.shots[s + 1].f0);
  }

  private onPanel(el: HTMLElement) {
    const act = el.closest<HTMLElement>('[data-act]')?.dataset.act;
    const sel = this.selected !== undefined ? this.byId.get(this.selected) : undefined;
    if (act === 'done') return this.h.close();
    if (act === 'redact') return this.h.redact(this.e!);
    if (act === 'find-words') return this.h.findWords(this.e!);
    if (act === 'suggest') {
      const w = el.closest<HTMLElement>('[data-word]')!.dataset.word!;
      this.plan.options.words = [...this.plan.options.words, w];
      const box = this.h.panel.querySelector<HTMLTextAreaElement>('#rv-words');
      if (box) box.value = this.plan.options.words.join('\n');
      const row = el.closest('.rv-suggest');
      el.closest('button')?.remove();
      if (row && !row.querySelector('button')) row.remove();
      return this.wordsChanged();
    }
    if (act === 'word-keep') return this.toggleWord(el.closest<HTMLElement>('[data-key]')!.dataset.key!);
    if (act === 'hear') {
      const s = this.spans[Number(el.closest<HTMLElement>('[data-i]')!.dataset.i)];
      // From a moment before, with the preview on, so the cover is heard in context.
      this.preview = true;
      const box = this.h.bar.querySelector<HTMLInputElement>('input[data-act=preview]');
      if (box) box.checked = true;
      this.h.video.currentTime = Math.max(0, s.t0 - 1);
      void this.h.video.play();
      return;
    }
    if (act === 'more-words') {
      this.wordsListed += WORDS_LISTED * 4;
      return this.renderMatches();
    }
    if (act === 'keep' && sel) return this.toggleKeep(this.selected!);
    if (act === 'start' && sel) return this.replaceHand(setEnd(sel, this.frame, 'start'));
    if (act === 'end' && sel) return this.replaceHand(setEnd(sel, this.frame, 'end'));
    if (act === 'delete' && sel) return this.deleteHand(this.selected!);
    if (act === 'show' && sel) return this.seekFrame(Math.round((sel.f0 + sel.f1) / 2));
    if (act === 'more') {
      const g = el.closest<HTMLElement>('[data-group]')!.dataset.group!;
      this.listed.set(g, (this.listed.get(g) ?? PER_SHOT) + MORE);
      return this.renderPanel();
    }
    const face = el.closest<HTMLElement>('.face');
    if (face) {
      const t = this.byId.get(face.dataset.track!)!;
      this.h.video.pause();
      this.selected = face.dataset.track;
      this.renderPanel();
      // The middle of the track (where it is most likely a good look), unless it is on screen.
      if (this.frame < t.f0 || this.frame > t.f1) this.seekFrame(Math.round((t.f0 + t.f1) / 2));
      else this.draw();
      return;
    }
    const head = el.closest<HTMLElement>('.shot-head');
    if (head) {
      this.h.video.pause();
      this.seekFrame(this.shots[Number(head.dataset.shot)].f0);
    }
  }

  // ---- spoken words ----

  /** The word list and its settings (rendered with the panel; typing does not redraw it). */
  private wordsForm(): string {
    const o = this.plan.options;
    const t = this.doc.layers.transcript;
    const sp = this.doc.layers.speakers?.items ?? [];
    const who =
      sp.length > 1
        ? `<p class="rv-speakers"><span class="muted">Everything one person says:</span> ${sp.map((s) => `<label><input type="checkbox" data-bleep-speaker="${esc(speakerName(s))}"${o.speakers.includes(speakerName(s)) ? ' checked' : ''}> ${esc(speakerName(s))}</label>`).join(' ')}</p>`
        : '';
    if (!t) return `${who}<p class="muted">No transcript: analyze this video with <b>Transcript</b> on to bleep spoken words.</p>`;
    if (!t.items.length) return '<p class="muted">Nobody speaks in this video, so there is nothing to bleep.</p>';
    const listed = new Set(o.words.map((w) => w.toLowerCase()));
    const names = suggestNames(this.doc)
      .filter((s) => !listed.has(s.word.toLowerCase()))
      .slice(0, 16);
    const need = this.hasWords
      ? ''
      : `<p class="rv-warn">This transcript was made without word times, which bleeping needs. <button data-act="find-words">Transcribe again with word times</button> (about a third of the video's length; the review reopens when it is done).</p>`;
    return `${need}${who}
      <label class="rv-words-label" for="rv-words">Words and phrases to bleep, one per line. Plurals, possessives, -ed and -ing count too; <code>*</code> matches any letters (<code>fuck*</code>); in "quotes", only that word.</label>
      <textarea id="rv-words" rows="3" spellcheck="false" placeholder="Okafor&#10;Daniel Reyes">${esc(o.words.join('\n'))}</textarea>
      ${names.length ? `<p class="rv-suggest"><span class="muted">Names in the transcript:</span> ${names.map((n) => `<button data-act="suggest" data-word="${esc(n.word)}" title="Said ${n.count} times: add it">${esc(n.word)} <span class="muted">${n.count}</span></button>`).join('')}</p>` : ''}
      <div class="rv-actions"><label><input type="checkbox" id="rv-profanity"${o.profanity ? ' checked' : ''}> Strong language too</label>
        <label>Sound <select id="rv-sound">${REDACT_SOUNDS.map((s) => `<option value="${s}"${s === o.sound ? ' selected' : ''}>${SOUND_NAMES[s]}</option>`).join('')}</select></label></div>`;
  }

  /** The matches: each one plays from a moment before; any can be left audible. */
  private renderMatches() {
    const el = this.h.panel.querySelector('.rv-matches');
    if (!el) return;
    const o = this.plan.options;
    const matches = this.spans.filter((s) => s.key);
    const kept = matches.filter((s) => s.kept).length;
    const secs = this.bleeps.reduce((n, [a, b]) => n + b - a, 0);
    const verb = o.sound === 'bleep' ? 'bleeped' : 'silenced';
    const turns = this.spans.filter((s) => s.speaker).length;
    const spoken = turns
      ? `<p class="muted">${turns} turns of ${esc(o.speakers.join(', '))} ${verb} (${secs.toFixed(1)} s in all). Turns are found automatically: some speech where people talk over each other is missed, so listen to the copy.</p>`
      : '';
    const head =
      spoken +
      (!o.words.length && !o.profanity
        ? ''
        : !matches.length
          ? `<p class="muted">None of these words is in the transcript. Names are often spelled differently there: search the transcript, or pick from the names above.</p>`
          : `<p class="muted">${matches.length - kept} ${verb} (${secs.toFixed(1)} s)${kept ? ` · ${kept} left audible` : ''}. Word times are approximate, so a little before and after each word is covered; play one to check it.</p>`);
    const rows = this.spans.slice(0, this.wordsListed).map((s, i) => {
      if (!s.key) return '';
      const say = s.term === s.text || !s.term ? '' : ` <span class="muted">${esc(s.term === 'profanity' ? 'strong language' : s.term)}</span>`;
      return `<li class="${s.kept ? 'kept' : ''}" data-i="${i}" data-key="${s.key}"><button class="ts" data-act="hear" title="Play it">${fmtT(s.t0 + o.lead)}</button> <span class="said">${esc(s.text ?? '')}</span>${say}
        <button data-act="word-keep">${s.kept ? (o.sound === 'bleep' ? 'Bleep it' : 'Silence it') : 'Leave audible'}</button></li>`;
    });
    const more = this.spans.length - this.wordsListed;
    el.innerHTML = `${head}<ol class="rv-wordlist">${rows.join('')}</ol>${more > 0 ? `<button class="more" data-act="more-words">${more} more</button>` : ''}`;
  }

  /** The heading and the summary line under it. */
  private renderTitle() {
    const h = this.h.panel.querySelector('.rv-head h1');
    if (h) h.textContent = `Review ${this.what()}`;
    const b = this.h.panel.querySelector('[data-act=redact]');
    if (b) b.textContent = `${this.bleeps.length ? `Blur and ${this.plan.options.sound === 'bleep' ? 'bleep' : 'silence'}` : `Blur ${this.what(false)}`}…`;
  }

  private what(words = true) {
    const names = this.kinds().map((k) => noun(k, 2));
    if (words && this.bleeps.length) names.push('spoken words');
    return names.length === 1 ? names[0] : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
  }

  // ---- the face list ----

  private renderPanel() {
    const e = this.e!;
    const p = this.plan;
    const kinds = this.kinds();
    const found = kinds.map((k) => ({ k, n: tracksOf(this.doc, k)!.items.length }));
    const kept = kinds.reduce((n, k) => n + keptIds(p.options, k).length, 0);
    const what = this.what(false);
    const thumbs = this.thumbs.get(e)!;
    const span = (t: Track) => `${fmtT(t.f0 / this.fps)}–${fmtT((t.f1 + 1) / this.fps)}`;
    const secs = (t: Track) => {
      const n = t.f1 - t.f0 + 1;
      const s = n / this.fps;
      return s < 0.1 ? `${n} frame${n === 1 ? '' : 's'}` : s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`;
    };
    const title = (key: string) => {
      const { kind, id } = parseKey(key);
      const n = noun(kind);
      return isHand(key) ? 'Added by hand' : `${n[0].toUpperCase()}${n.slice(1)} #${id}`;
    };
    const chip = (key: string) => {
      const t = this.byId.get(key)!;
      const k = this.isKept(key);
      const img = thumbs.get(key);
      const cls = `face${k ? ' kept' : ''}${isHand(key) ? ' hand' : ''}${parseKey(key).kind !== 'faces' ? ' plate' : ''}${key === this.selected ? ' sel' : ''}`;
      return `<button class="${cls}" data-track="${key}" title="${title(key)}, ${span(t)}">${img ? `<img src="${img}" alt="">` : '<span class="ph"></span>'}<span class="lab">${fmtT(t.f0 / this.fps)} · ${secs(t)}</span><span class="state">${isHand(key) ? 'added' : k ? 'visible' : 'blurred'}</span></button>`;
    };
    const count = (ids: string[]) =>
      kinds
        .map((g) => [g, ids.filter((k) => parseKey(k).kind === g).length] as const)
        .filter(([, n]) => n)
        .map(([g, n]) => `${n} ${noun(g, n)}`)
        .join(', ');
    const withFaces = this.shots.map((s, i) => ({ s, i, ids: this.shotTracks[i] })).filter((x) => x.ids.length);
    const shots = withFaces
      .map(({ s, i, ids }) => {
        // Each layer pages separately (faces with hand boxes), so a crowd does not bury the plates.
        const group = (g: RedactKind) => {
          const list = ids.filter((k) => parseKey(k).kind === g);
          if (!list.length) return '';
          const n = this.listed.get(`${i}:${g}`) ?? PER_SHOT;
          const more = list.length - n;
          return `<div class="faces" data-group="${i}:${g}">${list.slice(0, n).map(chip).join('')}${more > 0 ? `<button class="more" data-act="more">${more} shorter: show ${Math.min(more, MORE)} more</button>` : ''}</div>`;
        };
        return `<li class="rv-shot${i === this.shot ? ' cur' : ''}" data-shot="${i}">
          <div class="shot-head" data-shot="${i}">Shot ${i + 1} · ${fmtT(s.t0)}–${fmtT(s.t1)} · ${count(ids)}</div>
          ${kinds.map(group).join('')}</li>`;
      })
      .join('');
    const key = this.selected;
    const sel = key !== undefined ? this.byId.get(key) : undefined;
    const selHtml = !sel
      ? ''
      : isHand(key!)
        ? `<b>Added by hand</b>, ${span(sel)} (${secs(sel)}). If it moves, pause where it has moved and drag a new box: the box follows. <span class="btns"><button data-act="show">Show</button><button data-act="start">Starts here</button><button data-act="end">Ends here</button><button data-act="delete">Delete</button></span>`
        : `<b>${title(key!)}</b>, ${span(sel)} (${secs(sel)}): ${this.isKept(key!) ? 'left visible' : 'blurred'}. <span class="btns"><button data-act="show">Show</button><button data-act="keep">${this.isKept(key!) ? 'Blur it' : 'Leave visible'}</button></span>`;
    const shotsWithout = this.shots.length - withFaces.length;
    // The doc's other track layers, to cover or not (faces always are).
    const others = trackLayers(this.doc).filter((k) => k !== 'faces');
    const toggles = others.length
      ? `<span class="rv-cover">Cover ${others.map((k) => `<label><input type="checkbox" data-cover="${esc(k)}"${kinds.includes(k) ? ' checked' : ''}> ${noun(k, 2)}</label>`).join('')}</span>`
      : '';
    this.h.panel.innerHTML = `<div class="rv-head">
        <h1>Review ${this.what()}</h1>
        <p>Boxes on the video are what the copy covers${
          kinds.length > 1
            ? ` (${kinds
                .slice(1)
                .map((k) => `${noun(k, 2)} in ${this.color(k)[1]}`)
                .join(', ')})`
            : ''
        }. <b>Click a box</b> to leave it visible (click it again to cover it), e.g. a sign taken for a plate. <b>Drag a box</b> over anything the detector missed. Dashed boxes are held a moment before and after something is seen, for faces turning away.</p>
        <p class="muted keys">Space plays · ← → step a frame (Shift: 10) · [ ] previous / next shot · Esc deselects · Delete removes a box added by hand</p>
        <p class="muted">${found.map(({ k, n }) => `${n} ${noun(k, n)}`).join(' and ')} found · ${kept} left visible · ${p.added.length} added by hand · ${esc(this.h.keptWhere(e))}</p>
      </div>
      <div class="rv-sticky">
        <div class="rv-actions"><label>Cover with <select id="rv-style">${REDACT_STYLES.map((s) => `<option value="${s}"${s === p.options.style ? ' selected' : ''}>${STYLE_NAMES[s]}</option>`).join('')}</select></label>
          ${toggles}<button data-act="redact" class="primary">Blur ${what}…</button><button data-act="done">Done</button></div>
        <div class="rv-sel"${sel ? '' : ' hidden'}>${selHtml}</div>
      </div>
      <details class="rv-audio"${this.doc.layers.transcript?.items.length ? ' open' : ''}><summary>Spoken words${(this.doc.layers.speakers?.items.length ?? 0) > 1 ? ' and speakers' : ''}</summary>
        ${this.wordsForm()}<div class="rv-matches"></div></details>
      <ol class="rv-shots">${shots || `<li class="muted">No ${what} were found. Drag a box over anything to cover it.</li>`}</ol>
      ${shotsWithout ? `<p class="muted rv-foot">${shotsWithout} shot${shotsWithout === 1 ? '' : 's'} without ${what} not listed.</p>` : ''}`;
    this.renderMatches();
    this.renderTitle();
    this.seen.disconnect();
    this.wanted = [];
    for (const ph of this.h.panel.querySelectorAll('.face .ph')) this.seen.observe(ph.parentElement!);
  }
}
