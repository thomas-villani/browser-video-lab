// The "read the video" view: a doc rendered as Markdown for agents and people. Shots become
// sections; the transcript runs through them in timed paragraphs; keyframe images (when the
// caller has them) and silent gaps sit where they happen. Needs no captions and no LLM, and
// works with whatever layers the doc has. Pure TS.
import { silentGaps } from './keyframes.ts';
import type { KeyframeItem, SceneItem, Track, VideoDoc } from './schema.ts';

export interface MarkdownOptions {
  /** Link to a keyframe's image, relative to where the Markdown will live; undefined = none.
   *  Without images, keyframes still break paragraphs, so a timestamp appears at least every
   *  keyframes.maxGap seconds. */
  image?: (k: KeyframeItem, index: number) => string | undefined;
  /** Mark silent gaps at least this long, seconds (default: the keyframes layer's, else 4). */
  silence?: number;
  /** Start a new paragraph when speech pauses this long, seconds. */
  pause?: number;
}

/** 83.4 -> "01:23"; with `long`, "0:01:23". */
export function fmtTime(t: number, long = false): string {
  const s = Math.floor(t);
  const mm = String(Math.floor(s / 60) % 60).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return long ? `${Math.floor(s / 3600)}:${mm}:${ss}` : `${String(Math.floor(s / 60)).padStart(2, '0')}:${ss}`;
}

/** Most faces visible at once in frames [f0, f1]. */
export function maxFaces(tracks: Track[], f0: number, f1: number): number {
  const delta = new Map<number, number>();
  for (const t of tracks) {
    if (t.f1 < f0 || t.f0 > f1) continue;
    const a = Math.max(t.f0, f0);
    delta.set(a, (delta.get(a) ?? 0) + 1);
    delta.set(Math.min(t.f1, f1) + 1, (delta.get(Math.min(t.f1, f1) + 1) ?? 0) - 1);
  }
  let n = 0;
  let max = 0;
  for (const k of [...delta.keys()].sort((x, y) => x - y)) max = Math.max(max, (n += delta.get(k)!));
  return max;
}

type Event =
  | { t: number; kind: 'silence'; t1: number }
  | { t: number; kind: 'image'; k: KeyframeItem; i: number }
  | { t: number; kind: 'text'; t1: number; text: string };
const ORDER = { silence: 0, image: 1, text: 2 };

export function toMarkdown(doc: VideoDoc, opts: MarkdownOptions = {}): string {
  const { source, layers } = doc;
  const { speech, transcript, keyframes, faces } = layers;
  const long = source.duration >= 3600;
  const ts = (t: number) => fmtTime(t, long);
  const pause = opts.pause ?? 2;
  const silence = opts.silence ?? (keyframes?.producer.config.silence as number | undefined) ?? 4;
  const scenes: SceneItem[] = layers.scenes?.items.length
    ? layers.scenes.items
    : [{ t0: 0, t1: source.duration, f0: 0, f1: source.frames - 1, score: 0 }];

  const events: Event[] = [];
  for (const s of transcript?.items ?? []) events.push({ t: s.t0, kind: 'text', t1: s.t1, text: s.text.trim() });
  keyframes?.items.forEach((k, i) => events.push({ t: k.t, kind: 'image', k, i }));
  if (speech?.items.length) for (const [t, t1] of silentGaps(speech.items, source.duration, silence)) events.push({ t, kind: 'silence', t1 });
  events.sort((a, b) => a.t - b.t || ORDER[a.kind] - ORDER[b.kind]);

  // Header: what the video is and which layers this view was made from.
  const words = transcript?.items.reduce((n, x) => n + x.text.split(/\s+/).filter(Boolean).length, 0);
  const speechS = speech?.items.reduce((n, x) => n + x.t1 - x.t0, 0);
  const facts = [
    ts(source.duration),
    `${source.width}×${source.height}`,
    `${source.fps} fps`,
    layers.scenes && `${scenes.length} ${scenes.length === 1 ? 'shot' : 'shots'}`,
    speechS !== undefined && (speechS ? `${Math.round((100 * speechS) / source.duration)}% speech` : 'no speech detected'),
    !!words && `${words.toLocaleString('en-US')} words`,
    faces && `${faces.items.length} face tracks`,
  ].filter(Boolean);
  const made = Object.entries(layers).map(([n, l]) => `${n}${l.producer.model ? ` (${l.producer.model})` : ''}`);
  const out = [`# ${source.name}`, '', facts.join(' · '), `<!-- ${doc.schema} · ${made.join(' · ')} -->`];

  let e = 0;
  scenes.forEach((s, si) => {
    const last = si === scenes.length - 1;
    const mine: Event[] = [];
    while (e < events.length && (last || events[e].t < s.t1)) mine.push(events[e++]);
    const nFaces = faces && maxFaces(faces.items, s.f0, s.f1);
    const tail = nFaces === undefined ? '' : nFaces ? ` · up to ${nFaces} ${nFaces === 1 ? 'face' : 'faces'}` : ' · no faces';
    out.push('', `## ${scenes.length > 1 ? `Shot ${si + 1} · ` : ''}${ts(s.t0)}–${ts(s.t1)}${tail}`);

    // A keyframe inside running speech waits for the sentence to end, then breaks the paragraph.
    let para: { t: number; t1: number; text: string[] } | undefined;
    let waiting: Extract<Event, { kind: 'image' }>[] = [];
    const flush = () => {
      if (para) out.push('', `**[${ts(para.t)}]** ${para.text.join(' ')}`);
      para = undefined;
      for (const w of waiting) {
        const src = opts.image?.(w.k, w.i);
        if (src) out.push('', `![${ts(w.t)}](${src.replace(/ /g, '%20')})`);
      }
      waiting = [];
    };
    for (const ev of mine) {
      if (ev.kind === 'text') {
        if (para && ev.t - para.t1 > pause) flush();
        para ??= { t: ev.t, t1: ev.t1, text: [] };
        para.text.push(ev.text);
        para.t1 = ev.t1;
        if (waiting.length && /[.?!…]["')\]]*$/.test(ev.text)) flush();
      } else if (ev.kind === 'image') {
        waiting.push(ev);
        if (!para) flush();
      } else {
        flush();
        out.push('', `*(no speech ${ts(ev.t)}–${ts(ev.t1)})*`);
      }
    }
    flush();
  });
  return out.join('\n') + '\n';
}
