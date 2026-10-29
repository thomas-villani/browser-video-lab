// HTML for the app, from the same Markdown agents read (toMarkdown): people and agents see one
// view of a video. Only the subset toMarkdown writes is handled: headings, timed paragraphs,
// italic notes, keyframe images, the provenance comment. Times become seek targets (data-t).

export const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);

/** "01:23" or "1:01:23" -> seconds. */
export const secs = (ts: string) => ts.split(':').reduce((s, p) => s * 60 + Number(p), 0);

/** **bold** -> <mark> (search snippets), everything else escaped. */
export const marks = (s: string) => esc(s).replace(/\*\*(.+?)\*\*/g, '<mark>$1</mark>');

/**
 * Render toMarkdown output. Keyframe images must have been written as `kf:<frame>`; `image`
 * maps a frame to an object URL (or undefined while it is not extracted yet). `mark` wraps
 * search terms in **bold** (highlight() from search) so they show as <mark>.
 */
export function markdownToHtml(md: string, image: (frame: number) => string | undefined, mark: (text: string) => string = (t) => t): string {
  const out: string[] = [];
  const blocks = md.split(/\n{2,}/).map((b) => b.trim()).filter(Boolean);
  for (const [i, b] of blocks.entries()) {
    let m: RegExpExecArray | null;
    if ((m = /^# (.+)$/.exec(b))) {
      out.push(`<h1>${esc(m[1])}</h1>`);
    } else if ((m = /^(.*)\n<!--\s*(.*?)\s*-->$/.exec(b))) {
      // The facts line and the provenance comment under the title.
      out.push(`<p class="facts">${esc(m[1])}</p>`, `<p class="made" title="layers this view was made from">${esc(m[2])}</p>`);
    } else if ((m = /^## (.+)$/.exec(b))) {
      const t = /(\d+(?::\d\d){1,2})–/.exec(m[1]);
      // A shot with nothing in it (no speech, no keyframe) gets a compact heading.
      const bare = i + 1 === blocks.length || blocks[i + 1].startsWith('## ');
      out.push(`<h2${bare ? ' class="bare"' : ''}${t ? ` data-t="${secs(t[1])}"` : ''}>${esc(m[1])}</h2>`);
    } else if ((m = /^\*\*\[([\d:]+)\](?: ([^*]+):)?\*\* ([\s\S]+)$/.exec(b))) {
      const who = m[2] ? ` <b class="who">${esc(m[2])}:</b>` : '';
      out.push(`<p class="said" data-t="${secs(m[1])}"><button class="ts" data-t="${secs(m[1])}">${m[1]}</button>${who} ${marks(mark(m[3]))}</p>`);
    } else if ((m = /^!\[([\d:]+)\]\(kf:(\d+)\)$/.exec(b))) {
      const src = image(Number(m[2]));
      out.push(`<figure class="kf" data-t="${secs(m[1])}" data-frame="${m[2]}">${src ? `<img src="${src}" alt="frame at ${m[1]}">` : '<div class="pending"></div>'}<figcaption>${m[1]}</figcaption></figure>`);
    } else if ((m = /^\*\((.+)\)\*$/.exec(b))) {
      const t = /(\d+(?::\d\d){1,2})–/.exec(m[1]);
      out.push(`<p class="note"${t ? ` data-t="${secs(t[1])}"` : ''}>${esc(m[1])}</p>`);
    } else if ((m = /^\*(.+)\*$/.exec(b))) {
      out.push(`<p class="note">${esc(m[1])}</p>`);
    } else {
      out.push(`<p>${esc(b)}</p>`);
    }
  }
  return out.join('\n');
}

// ---- guidance for first-time users: time estimates, analysis steps, search suggestions ----

/**
 * Rough analysis time for `seconds` of video on an integrated GPU (Iris Xe, NOTES.md: the
 * transcript runs at 4-6x real time, shots alone at 10-25x, faces at 2.4-12x, plates 1.1-2.6x). A
 * rough number beats none: it tells people whether to wait or come back later.
 */
/** Plates, seconds per second of video (measured: see NOTES.md). */
const PLATES_COST = 0.6;
/** A pipeline's track layer on a DETR model (RF-DETR nano: ~9 s per second of busy 1080p, NOTES.md). */
const DETR_COST = 9;

/** Seconds per second of video a pipeline's track layer adds, by its model's decoder. */
export const trackCost = (decoder?: string) => (decoder === 'detr' ? DETR_COST : PLATES_COST);

/** Speakers, seconds per second of speech-filled audio (step 2 s, NOTES.md). */
const SPEAKERS_COST = 0.11;

/** `extra`: seconds per second of video for a pipeline's track layers (trackCost). */
export function estimate(seconds: number, o: { transcript: boolean; speakers?: boolean; faces: boolean; plates?: boolean; extra?: number }): string {
  const s = seconds * ((o.transcript ? 0.22 : 0.02) + (o.speakers ? SPEAKERS_COST : 0) + (o.faces ? 0.2 : 0.05) + (o.plates ? PLATES_COST : 0) + (o.extra ?? 0));
  if (s < 60) return 'under a minute';
  const m = Math.round(s / 60);
  return m < 90 ? `about ${m} min` : `about ${(s / 3600).toFixed(1)} h`;
}

/** "a, b and c". */
export const list = (xs: string[]) => (xs.length < 2 ? (xs[0] ?? '') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);

/** The steps a person sees for computing `want` when the doc already `has` some layers; `tracks`
 *  are the track layers (found together in one pass over the frames). */
export function analysisSteps(want: string[], has: Set<string>, tracks: string[] = ['faces', 'plates']): string[] {
  const missing = (n: string) => want.includes(n) && !has.has(n);
  const steps: string[] = [];
  if (missing('speech')) steps.push('Listening');
  if (missing('transcript')) steps.push('Transcribing');
  if (missing('speakers')) steps.push('Finding speakers');
  const found = want.filter((n) => tracks.includes(n) && missing(n));
  if (found.length) steps.push(`Finding ${list(found)}`);
  else if (missing('scenes')) steps.push('Finding shots');
  return steps;
}

/** Which of `steps` a progress report from analyze() belongs to (-1: before the first). */
export function stepOf(p: { pass: string; stage: string }, steps: string[]): number {
  const name =
    p.pass === 'frames' ? (steps.find((s) => s.startsWith('Finding ')) ?? '')
    : p.pass === 'audio' ? (/speaker/.test(p.stage) ? 'Finding speakers' : /^(loading|transcrib)/.test(p.stage) ? 'Transcribing' : 'Listening')
    : '';
  return steps.indexOf(name);
}

/**
 * Example queries from the transcripts: names and places (capitalized mid-sentence, which
 * Whisper writes reliably) said most often, taking turns between videos.
 */
export function suggestions(texts: string[][], n = 4): string[] {
  const per = texts.map((segs) => {
    const count = new Map<string, number>();
    for (const s of segs)
      for (const m of s.matchAll(/(?<=[a-z,;] )([A-Z][a-z]{3,})/g)) count.set(m[1], (count.get(m[1]) ?? 0) + 1);
    return [...count].filter(([, c]) => c >= 2).sort((a, b) => b[1] - a[1]).map(([w]) => w);
  });
  const out: string[] = [];
  for (let i = 0; out.length < n && per.some((p) => p.length > i); i++)
    for (const p of per) if (p[i] && !out.includes(p[i]) && out.length < n) out.push(p[i]);
  return out;
}
