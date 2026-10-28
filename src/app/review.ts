// Face review: the regions drawn over the player frame by frame, with the cover previewed live
// (the same drawing the redacted copy gets). Click a face to leave it visible, drag to cover
// one the detector missed. The choices are the entry's RedactPlan (saved as <video>.redact.json
// for folder entries), which Blur faces and `videodoc redact` use. Nothing is decoded here: the
// boxes come from the doc, the picture from the <video> element.
import {
  cursorFor, handTrack, newPlan, REDACT_STYLES, setBox, setEnd, trackBox,
  type Box, type RedactionCursor, type RedactPlan, type RedactStyle, type Region, type SceneItem, type Track, type VideoDoc,
} from '../doc/index.ts';
import { coverer } from '../redact';
import { esc } from './render';
import type { Entry } from './library';

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
  close: () => void;
}

const STYLE_NAMES: Record<RedactStyle, string> = { blur: 'Blur', pixelate: 'Pixelate', fill: 'Solid box' };
/** Faces listed per shot (longest first) until more are asked for, a page at a time (a crowded
 *  shot has thousands). */
const PER_SHOT = 24;
const MORE = 120;
/** Default length of a face added by hand, seconds (cut at the end of its shot). */
const HAND_SECONDS = 2;

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
  private byId = new Map<number, Track>();
  private shots: SceneItem[] = [];
  /** Track ids by shot, longest first (hand tracks first). */
  private shotTracks: number[][] = [];
  private fps = 25;
  /** Presentation time of frame 0 (first keyframe's t minus its frame time). */
  private off = 0;
  private frame = 0;
  private shot = -1;
  private regions: Region[] = [];
  private preview = true;
  private selected?: number;
  /** Faces listed per shot, when more than PER_SHOT were asked for. */
  private listed = new Map<number, number>();
  private hover?: number;
  private press?: { x: number; y: number; cx: number; cy: number; moved: boolean };
  private drag?: Box;
  private thumbs = new WeakMap<Entry, Map<number, string>>();
  private rvfc = 0;
  private unlisten: (() => void)[] = [];
  private resize = new ResizeObserver(() => this.layout());
  private thumbCanvas = document.createElement('canvas');
  /** A second, hidden player that seeks to faces in the list scrolled into view, for their
   *  thumbnails (the main player stays where the person is). */
  private filler = document.createElement('video');
  private wanted: number[] = [];
  private filling = false;
  private seen = new IntersectionObserver((items) => {
    for (const it of items) {
      const id = Number((it.target as HTMLElement).dataset.track);
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
    const doc = e.doc!;
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
      const t = ev.target as HTMLSelectElement;
      if (t.id !== 'rv-style') return;
      this.plan.options.style = t.value as RedactStyle;
      this.cover = coverer(overlay, this.plan.options.style);
      this.changed();
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
    for (const id of this.plan.options.keep) {
      const t = this.byId.get(id);
      const b = t && trackBox(t, this.frame);
      const g = b && this.cursor.grow(b);
      if (g) out.push({ id, box: g, held: false });
    }
    return out;
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
    for (const r of this.regions) {
      const on = r.id === this.selected || r.id === this.hover;
      const color = r.id < 0 ? '#f59e0b' : '#ffffff';
      stroke(r.box, on ? '#7ea2ff' : color, on ? 3 : 1.5, r.held ? [4, 3] : []);
      if (on) label(r.box, r.id < 0 ? 'added by hand' : `#${r.id} blurred · click: leave visible`, '#2b5fd9');
    }
    for (const r of kept) {
      const on = r.id === this.selected || r.id === this.hover;
      stroke(r.box, '#22c55e', on ? 3 : 2, [6, 4]);
      label(r.box, on ? `#${r.id} visible · click: blur` : 'visible', '#15803d');
    }
    if (this.drag) stroke(this.drag, '#f59e0b', 2, [6, 4]);
  }

  /** Thumbnails for the face list, cropped from the frame on screen as faces come by. */
  private capture(regions: Region[]) {
    const thumbs = this.thumbs.get(this.e!)!;
    let n = 0;
    for (const r of regions) {
      if (r.held || thumbs.has(r.id) || n >= 4) continue;
      n++;
      this.thumb(r.id, this.h.video, r.box);
    }
  }

  /** Crop `box` (source pixels, square around its center) from `v`'s frame as face `id`'s thumbnail. */
  private thumb(id: number, v: HTMLVideoElement, [x, y, w, h]: Box) {
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
    const faces = this.doc.layers.faces!.items;
    this.byId = new Map([...faces, ...this.plan.added].map((t) => [t.id, t]));
    this.cursor = cursorFor(this.doc, this.plan.options, this.plan.added);
    const len = (id: number) => (id < 0 ? Infinity : this.byId.get(id)!.f1 - this.byId.get(id)!.f0);
    this.shotTracks = this.shots.map(() => []);
    for (const t of [...this.plan.added, ...faces]) this.shotTracks[this.shotOf(t.f0)].push(t.id);
    for (const ids of this.shotTracks) ids.sort((a, b) => len(b) - len(a));
    this.regions = this.cursor.regions(this.frame);
  }

  private changed() {
    this.rebuild();
    this.h.changed(this.e!);
    this.renderPanel();
    this.draw();
  }

  private toggleKeep(id: number) {
    const keep = this.plan.options.keep;
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

  private deleteHand(id: number) {
    this.plan.added = this.plan.added.filter((x) => x.id !== id);
    if (this.selected === id) this.selected = undefined;
    this.thumbs.get(this.e!)!.delete(id);
    this.changed();
  }

  /** A box drawn on the frame: moves the selected hand track there, or adds a new one. */
  private drawn(box: Box) {
    const sel = this.selected !== undefined && this.selected < 0 ? this.byId.get(this.selected) : undefined;
    this.thumbs.get(this.e!)!.delete(sel?.id ?? 0);
    if (sel) return this.replaceHand(setBox(sel, this.frame, box));
    const id = Math.min(0, ...this.plan.added.map((t) => t.id)) - 1;
    const end = Math.min(this.shots[this.shotOf(this.frame)].f1, this.frame + Math.round(HAND_SECONDS * this.fps));
    this.plan.added = [...this.plan.added, handTrack(id, this.frame, box, end)];
    this.selected = id;
    this.changed();
  }

  // ---- input ----

  private point(ev: PointerEvent) {
    const r = this.h.overlay.getBoundingClientRect();
    return { x: ((ev.clientX - r.left) / r.width) * this.doc.source.width, y: ((ev.clientY - r.top) / r.height) * this.doc.source.height };
  }

  /** The smallest face box under (x, y): covered ones and kept ones. */
  private hit(x: number, y: number): number | undefined {
    let best: Region | undefined;
    for (const r of [...this.regions, ...this.keptBoxes()]) {
      const [bx, by, bw, bh] = r.box;
      if (x >= bx && x <= bx + bw && y >= by && y <= by + bh && (!best || bw * bh < best.box[2] * best.box[3])) best = r;
    }
    return best?.id;
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
    if (id !== undefined && id > 0) return this.toggleKeep(id);
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
      Delete: () => this.selected !== undefined && this.selected < 0 && this.deleteHand(this.selected),
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
    if (act === 'keep' && sel) return this.toggleKeep(sel.id);
    if (act === 'start' && sel) return this.replaceHand(setEnd(sel, this.frame, 'start'));
    if (act === 'end' && sel) return this.replaceHand(setEnd(sel, this.frame, 'end'));
    if (act === 'delete' && sel) return this.deleteHand(sel.id);
    if (act === 'show' && sel) return this.seekFrame(Math.round((sel.f0 + sel.f1) / 2));
    if (act === 'more') {
      const i = Number(el.closest<HTMLElement>('.rv-shot')!.dataset.shot);
      this.listed.set(i, (this.listed.get(i) ?? PER_SHOT) + MORE);
      return this.renderPanel();
    }
    const face = el.closest<HTMLElement>('.face');
    if (face) {
      const t = this.byId.get(Number(face.dataset.track))!;
      this.h.video.pause();
      this.selected = t.id;
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

  // ---- the face list ----

  private renderPanel() {
    const e = this.e!;
    const p = this.plan;
    const faces = this.doc.layers.faces!.items.length;
    const kept = p.options.keep.length;
    const thumbs = this.thumbs.get(e)!;
    const span = (t: Track) => `${fmtT(t.f0 / this.fps)}–${fmtT((t.f1 + 1) / this.fps)}`;
    const secs = (t: Track) => {
      const n = t.f1 - t.f0 + 1;
      const s = n / this.fps;
      return s < 0.1 ? `${n} frame${n === 1 ? '' : 's'}` : s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`;
    };
    const chip = (id: number) => {
      const t = this.byId.get(id)!;
      const k = p.options.keep.includes(id);
      const img = thumbs.get(id);
      const cls = `face${k ? ' kept' : ''}${id < 0 ? ' hand' : ''}${id === this.selected ? ' sel' : ''}`;
      return `<button class="${cls}" data-track="${id}" title="${id < 0 ? 'Added by hand' : `Face #${id}`}, ${span(t)}">${img ? `<img src="${img}" alt="">` : '<span class="ph"></span>'}<span class="lab">${fmtT(t.f0 / this.fps)} · ${secs(t)}</span><span class="state">${id < 0 ? 'added' : k ? 'visible' : 'blurred'}</span></button>`;
    };
    const withFaces = this.shots.map((s, i) => ({ s, i, ids: this.shotTracks[i] })).filter((x) => x.ids.length);
    const shots = withFaces
      .map(({ s, i, ids }) => {
        const n = this.listed.get(i) ?? PER_SHOT;
        const more = ids.length - n;
        return `<li class="rv-shot${i === this.shot ? ' cur' : ''}" data-shot="${i}">
          <div class="shot-head" data-shot="${i}">Shot ${i + 1} · ${fmtT(s.t0)}–${fmtT(s.t1)} · ${ids.length} face${ids.length === 1 ? '' : 's'}</div>
          <div class="faces">${ids.slice(0, n).map(chip).join('')}${more > 0 ? `<button class="more" data-act="more">${more} shorter: show ${Math.min(more, MORE)} more</button>` : ''}</div></li>`;
      })
      .join('');
    const sel = this.selected !== undefined ? this.byId.get(this.selected) : undefined;
    const selHtml = !sel ? ''
      : sel.id < 0
        ? `<b>Face added by hand</b>, ${span(sel)} (${secs(sel)}). If it moves, pause where it has moved and drag a new box: the box follows. <span class="btns"><button data-act="show">Show</button><button data-act="start">Starts here</button><button data-act="end">Ends here</button><button data-act="delete">Delete</button></span>`
        : `<b>Face #${sel.id}</b>, ${span(sel)} (${secs(sel)}): ${p.options.keep.includes(sel.id) ? 'left visible' : 'blurred'}. <span class="btns"><button data-act="show">Show</button><button data-act="keep">${p.options.keep.includes(sel.id) ? 'Blur it' : 'Leave visible'}</button></span>`;
    const shotsWithout = this.shots.length - withFaces.length;
    this.h.panel.innerHTML = `<div class="rv-head">
        <h1>Review faces</h1>
        <p>Boxes on the video are what the copy covers. <b>Click a face</b> to leave it visible (click it again to cover it). <b>Drag a box</b> over a face the detector missed. Dashed boxes are held a moment before and after a face is seen, for faces turning away.</p>
        <p class="muted keys">Space plays · ← → step a frame (Shift: 10) · [ ] previous / next shot · Esc deselects · Delete removes a face added by hand</p>
        <p class="muted">${faces} face${faces === 1 ? '' : 's'} found · ${kept} left visible · ${p.added.length} added by hand · ${esc(this.h.keptWhere(e))}</p>
      </div>
      <div class="rv-sticky">
        <div class="rv-actions"><label>Cover with <select id="rv-style">${REDACT_STYLES.map((s) => `<option value="${s}"${s === p.options.style ? ' selected' : ''}>${STYLE_NAMES[s]}</option>`).join('')}</select></label>
          <button data-act="redact" class="primary">Blur faces…</button><button data-act="done">Done</button></div>
        <div class="rv-sel"${sel ? '' : ' hidden'}>${selHtml}</div>
      </div>
      <ol class="rv-shots">${shots || '<li class="muted">No faces were found. Drag a box over any face to cover it.</li>'}</ol>
      ${shotsWithout ? `<p class="muted rv-foot">${shotsWithout} shot${shotsWithout === 1 ? '' : 's'} without faces not listed.</p>` : ''}`;
    this.seen.disconnect();
    this.wanted = [];
    for (const ph of this.h.panel.querySelectorAll('.face .ph')) this.seen.observe(ph.parentElement!);
  }
}
