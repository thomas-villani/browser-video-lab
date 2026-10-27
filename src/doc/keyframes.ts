// Keyframe selection: which frames are worth looking at (and later captioning), chosen from the
// scenes and speech layers without decoding anything. Pure TS, so the CLI derives it in Node.
//
// Heuristics from the bodycam search work: every shot gets a frame; silent stretches get one
// (nobody is saying what is on screen there); long shots get fill-ins so nothing goes unseen for
// more than `maxGap` seconds.
import type { KeyframeItem, SceneItem, SpeechItem } from './schema.ts';

export interface KeyframeConfig {
  /** Seconds into a shot for its frame (skips fades and motion blur at the cut). */
  offset: number;
  /** Silent gaps at least this long get a frame at their middle, seconds. */
  silence: number;
  /** Candidates closer than this to an accepted frame are dropped, seconds. */
  minGap: number;
  /** Longest stretch of a shot without a frame, seconds. */
  maxGap: number;
}

const PRIORITY = { scene: 0, silence: 1, coverage: 2 } as const;
const round3 = (v: number) => Math.round(v * 1000) / 1000;

/** Gaps between speech spans (and before the first and after the last) of at least `min` s. */
export function silentGaps(speech: SpeechItem[], duration: number, min: number): [number, number][] {
  const gaps: [number, number][] = [];
  let prev = 0;
  for (const s of [...speech, { t0: duration, t1: duration }]) {
    if (s.t0 - prev >= min) gaps.push([prev, s.t0]);
    prev = Math.max(prev, s.t1);
  }
  return gaps;
}

export function selectKeyframes(scenes: SceneItem[], speech: SpeechItem[], duration: number, fps: number, c: KeyframeConfig): KeyframeItem[] {
  if (!scenes.length) return [];
  // Shots at least minGap long always keep their frame; quick cuts and silences are thinned.
  const kept: { t: number; why: KeyframeItem['why'] }[] = [];
  const cands: typeof kept = [];
  for (const s of scenes) (s.t1 - s.t0 >= c.minGap ? kept : cands).push({ t: s.t0 + Math.min(c.offset, (s.t1 - s.t0) / 2), why: 'scene' });
  for (const [a, b] of silentGaps(speech, duration, c.silence)) cands.push({ t: (a + b) / 2, why: 'silence' });
  cands.sort((x, y) => PRIORITY[x.why] - PRIORITY[y.why] || x.t - y.t);
  for (const k of cands) if (kept.every((x) => Math.abs(x.t - k.t) >= c.minGap)) kept.push(k);

  // Fill-ins: evenly spaced inside every stretch of a shot longer than maxGap.
  const out: typeof cands = [...kept];
  for (const s of scenes) {
    const pts = [s.t0, ...kept.filter((k) => k.t >= s.t0 && k.t < s.t1).map((k) => k.t).sort((x, y) => x - y), s.t1];
    for (let i = 1; i < pts.length; i++) {
      const gap = pts[i] - pts[i - 1];
      const n = Math.ceil(gap / c.maxGap) - 1;
      for (let j = 1; j <= n; j++) out.push({ t: pts[i - 1] + (j * gap) / (n + 1), why: 'coverage' });
    }
  }

  out.sort((x, y) => x.t - y.t);
  return out.map(({ t, why }) => {
    let scene = scenes.findIndex((s) => t < s.t1);
    if (scene < 0) scene = scenes.length - 1;
    const s = scenes[scene];
    const frame = Math.min(s.f1, s.f0 + Math.round((t - s.t0) * fps));
    return { t: round3(t), frame, scene, why };
  });
}
