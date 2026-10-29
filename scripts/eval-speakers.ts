// Diarization error rate of the speakers layer on AMI meetings (scripts/fetch-ami.mjs), and the
// clustering settings that minimize it. The models run once per meeting and window step (in the
// browser; cached in results/speakers/); clustering and scoring run here in Node, so a grid of
// clustering settings costs seconds.
//
//   node scripts/eval-speakers.ts [--meetings ES2004a,IS1009a] [--step 1] [--device webgpu]
//                                
//                                 [--threshold 0.6,0.7046,0.8] [--min-cluster 12] [--max 0]
//
// stdout: JSONL, one row per meeting and setting (DER, missed, false alarm, confusion, speakers
// found and in the reference, model time). stderr: progress and a summary table.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { analyze, ROOT, warmBrowser, type Context } from '../cli/lib.ts';
import { diarize, DIARIZE_DEFAULTS, type WindowSeg } from '../src/audio/diarize.ts';
import { der, parseRttm } from '../src/audio/der.ts';
import { parseDoc, PRODUCERS } from '../src/doc/index.ts';
// @ts-expect-error untyped JS
import { AMI, MEETINGS } from './fetch-ami.mjs';

const { values: a } = parseArgs({
  options: {
    meetings: { type: 'string' },
    step: { type: 'string', default: String(PRODUCERS.speakers.defaults.step) },
    device: { type: 'string', default: 'webgpu' },
    threshold: { type: 'string', default: String(DIARIZE_DEFAULTS.threshold) },
    'min-cluster': { type: 'string', default: String(DIARIZE_DEFAULTS.minClusterSize) },
    max: { type: 'string', default: '0' },
  },
});
const meetings: string[] = a.meetings ? a.meetings.split(',') : MEETINGS;
const step = Number(a.step);
const list = (s: string) => s.split(',').map(Number);
const CACHE = join(ROOT, 'results/speakers');
mkdirSync(CACHE, { recursive: true });

interface Inputs {
  t0: number;
  windows: { start: number; act: string }[];
  embeddings: (number[] | null)[][];
  stats: { windows: number; skipped: number; embeddings: number; segMs: number; embMs: number; fbankMs?: number; ms: number };
}

const ctx: Context = { browser: warmBrowser({ browser: 'msedge', headed: false }), report: { log: (l) => console.error(l), progress() {} }, cwd: ROOT };
const d = PRODUCERS.speakers.defaults;
const rows: Record<string, unknown>[] = [];
try {
  for (const m of meetings) {
    const video = join(ROOT, AMI, `${m}.mp4`);
    if (!existsSync(video)) throw new Error(`${video} missing: node scripts/fetch-ami.mjs ${m}`);
    const cache = join(CACHE, `${m}-step${step}-${a.device}.json`);
    if (!existsSync(cache)) {
      await analyze(ctx, video, { layers: ['speech'] });
      const speech = parseDoc(readFileSync(`${video}.videodoc.json`, 'utf8')).layers.speech!.items;
      console.error(`${m}: running the speaker models (step ${step} s)...`);
      let last = 0;
      const inputs = await ctx.browser.run(
        { video },
        (page, base) =>
          page.evaluate(
            ([u, sp, c]: [string, unknown, unknown]) => (window as unknown as { videodoc: { speakerInputs(...x: unknown[]): unknown } }).videodoc.speakerInputs(u, sp, c),
            [`${base}@file/video`, speech, { segmentation: d.segmentation, embedding: d.embedding, step, device: a.device }],
          ),
        (p) => {
          if (p.done - last >= 100 || p.done === p.total) console.error(`  ${p.done}/${p.total} windows`);
          last = p.done;
        },
      );
      writeFileSync(cache, JSON.stringify(inputs));
    }
    const inputs = JSON.parse(readFileSync(cache, 'utf8')) as Inputs;
    const windows: WindowSeg[] = inputs.windows.map((w) => ({ start: w.start, act: Uint8Array.from(atob(w.act), (c) => c.charCodeAt(0)) }));
    const embeddings = inputs.embeddings.map((e) => e.map((v) => (v ? Float32Array.from(v) : undefined)));
    const ref = parseRttm(readFileSync(join(ROOT, AMI, `${m}.rttm`), 'utf8'));
    const [, , u0, u1] = readFileSync(join(ROOT, AMI, `${m}.uem`), 'utf8').trim().split(/\s+/);
    for (const threshold of list(a.threshold))
      for (const minClusterSize of list(a['min-cluster']))
        for (const maxSpeakers of list(a.max)) {
          const t = performance.now();
          const speakers = diarize(windows, embeddings, { threshold, minClusterSize, maxSpeakers });
          const clusterMs = performance.now() - t;
          const hyp = speakers.flatMap((s) => s.turns.map(([t0, t1]) => ({ t0: t0 + inputs.t0, t1: t1 + inputs.t0, speaker: String(s.id) })));
          const r = der(ref, hyp, [Number(u0), Number(u1)]);
          const row = {
            meeting: m, step, threshold, minClusterSize, maxSpeakers,
            der: +(100 * r.der).toFixed(1), missed: +((100 * r.missed) / r.total).toFixed(1), falseAlarm: +((100 * r.falseAlarm) / r.total).toFixed(1), confusion: +((100 * r.confusion) / r.total).toFixed(1),
            found: speakers.length, reference: new Set(ref.map((x) => x.speaker)).size,
            minutes: +((Number(u1) - Number(u0)) / 60).toFixed(1), modelS: +(inputs.stats.ms / 1000).toFixed(1), segS: +(inputs.stats.segMs / 1000).toFixed(1), fbankS: +((inputs.stats.fbankMs ?? 0) / 1000).toFixed(1), embS: +(inputs.stats.embMs / 1000).toFixed(1),
            windows: inputs.stats.windows, embeddings: inputs.stats.embeddings, clusterMs: Math.round(clusterMs),
          };
          rows.push(row);
          console.log(JSON.stringify(row));
        }
  }
} finally {
  await ctx.browser.close();
}
// Per setting: DER over all meetings, weighted by reference speech (as benchmarks report it).
console.error('\nthreshold minCluster max |  DER  missed  FA   conf | per meeting');
const key = (r: Record<string, unknown>) => `${r.threshold} ${r.minClusterSize} ${r.maxSpeakers}`;
for (const k of [...new Set(rows.map(key))]) {
  const rs = rows.filter((r) => key(r) === k);
  const mean = (f: string) => (rs.reduce((s, r) => s + (r[f] as number) * (r.minutes as number), 0) / rs.reduce((s, r) => s + (r.minutes as number), 0)).toFixed(1);
  console.error(`${k.padEnd(24)} | ${mean('der').padStart(5)} ${mean('missed').padStart(6)} ${mean('falseAlarm').padStart(5)} ${mean('confusion').padStart(5)} | ${rs.map((r) => `${r.meeting} ${r.der} (${r.found}/${r.reference})`).join(', ')}`);
}
