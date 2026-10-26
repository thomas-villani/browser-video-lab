// Transcript-derived signals. Pure TS (browser, worker, Node).
//
// Speech coverage drives visual sampling: where someone is talking, the transcript carries
// most of the meaning; silent stretches are where image analysis has to do the work.

export interface Span {
  start: number;
  end: number;
}

/** Merge overlapping or touching spans (within `join` seconds). */
export function mergeSpans(spans: Span[], join = 0): Span[] {
  const sorted = [...spans].filter((s) => s.end > s.start).sort((a, b) => a.start - b.start);
  const out: Span[] = [];
  for (const s of sorted) {
    const last = out[out.length - 1];
    if (last && s.start <= last.end + join) last.end = Math.max(last.end, s.end);
    else out.push({ start: s.start, end: s.end });
  }
  return out;
}

export interface SpeechCoverage {
  duration: number;
  speechSeconds: number;
  speechPct: number;
  /** Silent stretches of at least `minGap` seconds: candidates for denser visual sampling. */
  gaps: Span[];
}

export function speechCoverage(segments: Span[], duration: number, minGap = 2): SpeechCoverage {
  const speech = mergeSpans(segments, 0.3);
  const speechSeconds = speech.reduce((s, x) => s + Math.min(x.end, duration) - x.start, 0);
  const gaps: Span[] = [];
  let t = 0;
  for (const s of [...speech, { start: duration, end: duration }]) {
    if (s.start - t >= minGap) gaps.push({ start: r2(t), end: r2(s.start) });
    t = Math.max(t, s.end);
  }
  return { duration: r2(duration), speechSeconds: r2(speechSeconds), speechPct: r2((100 * speechSeconds) / duration), gaps };
}

const r2 = (v: number) => Math.round(v * 100) / 100;

export interface VadOptions {
  /** Speech starts at this probability... */
  threshold: number;
  /** ...and ends after `minSilence` seconds below threshold - 0.15. */
  minSilence: number;
  /** Drop speech shorter than this, seconds. */
  minSpeech: number;
  /** Grow each span by this much on both sides, seconds. */
  pad: number;
}

export const DEFAULT_VAD: VadOptions = { threshold: 0.5, minSilence: 0.3, minSpeech: 0.25, pad: 0.2 };

/**
 * Per-window speech probabilities → speech spans (after Silero's get_speech_timestamps).
 * `step` = seconds per probability window; `t0` = media time of the first window.
 */
export function speechSpans(probs: ArrayLike<number>, step: number, t0 = 0, opts: VadOptions = DEFAULT_VAD): Span[] {
  const spans: Span[] = [];
  const off = opts.threshold - 0.15;
  let start = -1;
  let quietSince = -1;
  const close = (end: number) => {
    if (end - start >= opts.minSpeech) spans.push({ start, end });
    start = -1;
    quietSince = -1;
  };
  for (let i = 0; i < probs.length; i++) {
    const t = i * step;
    const p = probs[i];
    if (start < 0) {
      if (p >= opts.threshold) start = t;
      continue;
    }
    if (p >= off) quietSince = -1;
    else if (quietSince < 0) quietSince = t;
    else if (t + step - quietSince >= opts.minSilence) close(quietSince);
  }
  if (start >= 0) close(probs.length * step);
  const end = probs.length * step;
  return mergeSpans(spans.map((s) => ({ start: Math.max(0, s.start - opts.pad), end: Math.min(end, s.end + opts.pad) }))).map((s) => ({
    start: r2(t0 + s.start),
    end: r2(t0 + s.end),
  }));
}

/**
 * Pack speech spans into chunks for a model with a fixed context (Whisper: 30 s): neighbors
 * closer than `joinGap` share a chunk while it stays within `maxLen`; longer spans are split
 * evenly. Each chunk is transcribed on its own, so silence never reaches the model.
 */
export function groupSpans(spans: Span[], maxLen = 28, joinGap = 1.5): Span[] {
  const out: Span[] = [];
  for (const s of mergeSpans(spans)) {
    const parts = Math.ceil((s.end - s.start) / maxLen);
    for (let i = 0; i < parts; i++) {
      const piece = { start: s.start + ((s.end - s.start) * i) / parts, end: s.start + ((s.end - s.start) * (i + 1)) / parts };
      const last = out[out.length - 1];
      if (last && piece.start - last.end < joinGap && piece.end - last.start <= maxLen) last.end = piece.end;
      else out.push(piece);
    }
  }
  return out.map((s) => ({ start: r2(s.start), end: r2(s.end) }));
}

/** Whisper's classic failure is a loop that repeats one segment; keep the first of each run. */
export function dropRepeats<T extends { text: string }>(segs: T[]): T[] {
  const key = (s: T) => s.text.toLowerCase().replace(/[^a-z0-9]/g, '');
  return segs.filter((s, i) => !key(s) || i === 0 || key(s) !== key(segs[i - 1]));
}
