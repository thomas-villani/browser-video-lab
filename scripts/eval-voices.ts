// How far apart one person's voices are in separate recordings, against two people's: the
// threshold for suggesting a name from a voice named in another video. Each AMI meeting
// (scripts/fetch-ami.mjs, models cached by scripts/eval-speakers.ts) is cut into parts that
// stand in for separate videos; each part is diarized alone, its speakers are mapped to the
// reference's, and the voices of every pair of speakers in different parts are compared.
//
//   node scripts/eval-voices.ts [--parts 2,3,4] [--step 2]
//
// stdout: JSON (cosine distances of same-person and different-person pairs, by part length,
// and the error rates of thresholds). stderr: a table.
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { ROOT } from '../cli/lib.ts';
import { der, parseRttm } from '../src/audio/der.ts';
import { diarize, type WindowSeg } from '../src/audio/diarize.ts';
import { PRODUCERS } from '../src/doc/index.ts';
// @ts-expect-error untyped JS
import { AMI, MEETINGS } from './fetch-ami.mjs';

const { values: a } = parseArgs({ options: { parts: { type: 'string', default: '2,3,4' }, step: { type: 'string', default: String(PRODUCERS.speakers.defaults.step) } } });

interface Voice {
  meeting: string;
  part: number;
  who: string;
  voice: Float32Array;
  seconds: number;
}
const dist = (x: Float32Array, y: Float32Array) => 1 - x.reduce((s, v, i) => s + v * y[i], 0);
const out: Record<string, unknown> = {};
for (const parts of a.parts.split(',').map(Number)) {
  const voices: Voice[] = [];
  for (const m of MEETINGS as string[]) {
    const cache = join(ROOT, 'results/speakers', `${m}-step${a.step}-webgpu.json`);
    if (!existsSync(cache)) throw new Error(`${cache} missing: node scripts/eval-speakers.ts --meetings ${m} --step ${a.step}`);
    const inputs = JSON.parse(readFileSync(cache, 'utf8')) as { t0: number; windows: { start: number; act: string }[]; embeddings: (number[] | null)[][] };
    const ref = parseRttm(readFileSync(join(ROOT, AMI, `${m}.rttm`), 'utf8'));
    const [, , u0, u1] = readFileSync(join(ROOT, AMI, `${m}.uem`), 'utf8')
      .trim()
      .split(/\s+/)
      .map(Number);
    const len = (u1 - u0) / parts;
    for (let p = 0; p < parts; p++) {
      const [p0, p1] = [u0 + p * len, u0 + (p + 1) * len];
      // Windows wholly inside the part, as if the video were cut there.
      const keep = inputs.windows.map((_, i) => i).filter((i) => inputs.windows[i].start + inputs.t0 >= p0 && inputs.windows[i].start + inputs.t0 + 10 <= p1);
      const windows: WindowSeg[] = keep.map((i) => ({ start: inputs.windows[i].start, act: Uint8Array.from(atob(inputs.windows[i].act), (c) => c.charCodeAt(0)) }));
      const embeddings = keep.map((i) => inputs.embeddings[i].map((v) => (v ? Float32Array.from(v) : undefined)));
      const speakers = diarize(windows, embeddings);
      const hyp = speakers.flatMap((s) => s.turns.map(([t0, t1]) => ({ t0: t0 + inputs.t0, t1: t1 + inputs.t0, speaker: String(s.id) })));
      const r = der(ref, hyp, [p0, p1]);
      for (const s of speakers) {
        const who = r.mapping[String(s.id)];
        if (who) voices.push({ meeting: m, part: p, who: `${m}/${who}`, voice: s.voice, seconds: s.turns.reduce((n, [t0, t1]) => n + t1 - t0, 0) });
      }
    }
  }
  // Pairs in different parts: of one meeting (same room and microphones: the hard case), and
  // across meetings.
  const same: number[] = [];
  const other: number[] = [];
  const otherMeeting: number[] = [];
  for (let i = 0; i < voices.length; i++)
    for (let j = i + 1; j < voices.length; j++) {
      const [x, y] = [voices[i], voices[j]];
      if (x.meeting === y.meeting && x.part === y.part) continue;
      const d = dist(x.voice, y.voice);
      if (x.who === y.who) same.push(d);
      else (x.meeting === y.meeting ? other : otherMeeting).push(d);
    }
  const q = (xs: number[], f: number) => +[...xs].sort((p, r) => p - r)[Math.min(xs.length - 1, Math.floor(f * xs.length))].toFixed(3);
  const thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6].map((t) => ({
    threshold: t,
    sameMissed: +(same.filter((d) => d >= t).length / same.length).toFixed(3),
    otherAccepted: +(other.filter((d) => d < t).length / other.length).toFixed(3),
    otherMeetingAccepted: +(otherMeeting.filter((d) => d < t).length / Math.max(1, otherMeeting.length)).toFixed(3),
  }));
  const minutes = +(
    MEETINGS.reduce((n: number, m: string) => {
      const [, , u0, u1] = readFileSync(join(ROOT, AMI, `${m}.uem`), 'utf8')
        .trim()
        .split(/\s+/)
        .map(Number);
      return n + (u1 - u0) / 60;
    }, 0) /
    MEETINGS.length /
    parts
  ).toFixed(1);
  const row = {
    parts,
    minutesPerPart: minutes,
    voices: voices.length,
    same: { n: same.length, min: q(same, 0), median: q(same, 0.5), p90: q(same, 0.9), max: q(same, 1) },
    other: { n: other.length, min: q(other, 0), p10: q(other, 0.1), median: q(other, 0.5) },
    otherMeeting: { n: otherMeeting.length, min: q(otherMeeting, 0), median: q(otherMeeting, 0.5) },
    thresholds,
  };
  out[parts] = row;
  console.error(`\n${parts} parts (~${minutes} min each), ${voices.length} voices`);
  console.error(`  same person:      n ${row.same.n}, median ${row.same.median}, p90 ${row.same.p90}, max ${row.same.max}`);
  console.error(`  other, same room: n ${row.other.n}, min ${row.other.min}, p10 ${row.other.p10}, median ${row.other.median}`);
  console.error(`  other meeting:    n ${row.otherMeeting.n}, min ${row.otherMeeting.min}, median ${row.otherMeeting.median}`);
  for (const t of thresholds) console.error(`  < ${t.threshold}: same missed ${t.sameMissed}, other accepted ${t.otherAccepted} (other meeting ${t.otherMeetingAccepted})`);
}
console.log(JSON.stringify(out));
