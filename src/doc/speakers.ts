// Who said what: the speakers layer (diarization) applied to the transcript. Words go to the
// speaker talking over most of them; a segment without word times goes to the speaker talking
// over most of it. Pure.
import type { SpeakerItem, TranscriptItem } from './schema.ts';

/** A stretch of the transcript said by one speaker (undefined: nobody was found speaking). */
export interface Said {
  t0: number;
  t1: number;
  text: string;
  speaker?: number;
}

/** Seconds of [t0, t1] each speaker talks over; the one talking most, or undefined. */
export function speakerAt(speakers: SpeakerItem[], t0: number, t1: number): number | undefined {
  let best: number | undefined;
  let most = 0;
  for (const s of speakers) {
    let n = 0;
    for (const [a, b] of s.turns) if (b > t0 && a < t1) n += Math.min(b, t1) - Math.max(a, t0);
    if (n > most) {
      most = n;
      best = s.id;
    }
  }
  return best;
}

/** Split transcript segments where the speaker changes (by word, when the words have times).
 *  Words in no one's turn go with the word before (or after) them. */
export function whoSaid(items: TranscriptItem[], speakers: SpeakerItem[]): Said[] {
  const out: Said[] = [];
  for (const it of items) {
    if (!it.words?.length) {
      out.push({ t0: it.t0, t1: it.t1, text: it.text.trim(), speaker: speakerAt(speakers, it.t0, it.t1) });
      continue;
    }
    const who = it.words.map(([a, b]) => speakerAt(speakers, a, Math.max(b, a + 0.01)));
    for (let i = 1; i < who.length; i++) who[i] ??= who[i - 1];
    for (let i = who.length - 2; i >= 0; i--) who[i] ??= who[i + 1];
    let from = 0;
    for (let i = 1; i <= it.words.length; i++) {
      if (i < it.words.length && who[i] === who[from]) continue;
      const words = it.words.slice(from, i);
      out.push({ t0: words[0][0], t1: words[words.length - 1][1], text: words.map((w) => w[2].trim()).join(' ').replace(/\s+([,.?!;:…])/g, '$1'), speaker: who[from] });
      from = i;
    }
  }
  return out;
}

/** "Speaker 2", or the name given to that speaker. */
export const speakerName = (id: number, names?: Record<string, string>) => names?.[id] ?? `Speaker ${id}`;
