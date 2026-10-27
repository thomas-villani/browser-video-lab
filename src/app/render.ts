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
    } else if ((m = /^\*\*\[([\d:]+)\]\*\* ([\s\S]+)$/.exec(b))) {
      out.push(`<p class="said" data-t="${secs(m[1])}"><button class="ts" data-t="${secs(m[1])}">${m[1]}</button> ${marks(mark(m[2]))}</p>`);
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
