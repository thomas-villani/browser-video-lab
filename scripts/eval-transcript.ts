// Word error rate of a transcript run against a reference WebVTT (e.g. YouTube captions via
// `yt-dlp --write-auto-subs --sub-format vtt --skip-download`). Only reference words inside
// the run's audio range are scored. Text is normalized: lowercase, punctuation stripped.
//
//   node scripts/eval-transcript.ts --pred results/<clip>-transcript-<name>.json --ref results/ref/<clip>.en.vtt
//
// stdout: JSONL (one row per --pred file). stderr: summary table.
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';

const { values: a } = parseArgs({ options: { pred: { type: 'string' }, ref: { type: 'string' } } });
if (!a.pred || !a.ref) {
  console.error('usage: node scripts/eval-transcript.ts --pred <run.json>[,...] --ref <captions.vtt>');
  process.exit(2);
}

const ts = (s: string) => s.split(':').reduce((acc, v) => acc * 60 + Number(v), 0);

/** Reference words with times. Handles YouTube's rolling auto-captions (word-timed <c> tags). */
function vttWords(vtt: string): { t: number; w: string }[] {
  const words: { t: number; w: string }[] = [];
  const cues = vtt.split(/\r?\n\r?\n/);
  const timed = cues.some((c) => /<\d\d:\d\d:\d\d\.\d+><c>/.test(c));
  for (const cue of cues) {
    const m = cue.match(/(\d\d:\d\d:\d\d\.\d+) -->/);
    if (!m) continue;
    const lines = cue.split(/\r?\n/).slice(1);
    // Rolling captions repeat the previous line; only lines with word timings are new text.
    const fresh = timed ? lines.filter((l) => /<c>/.test(l)) : lines.slice(lines.findIndex((l) => l.includes('-->')) + 1);
    for (const line of fresh) {
      let t = ts(m[1]);
      for (const part of line.split(/(<\d\d:\d\d:\d\d\.\d+>)/)) {
        const tm = part.match(/^<(\d\d:\d\d:\d\d\.\d+)>$/);
        if (tm) t = ts(tm[1]);
        else for (const w of norm(part.replace(/<[^>]+>/g, ''))) words.push({ t, w });
      }
    }
  }
  return words;
}

function norm(s: string): string[] {
  return s
    .toLowerCase()
    .replace(/&(gt|lt|amp|nbsp|quot);/g, ' ') // WebVTT entities; &gt;&gt; marks speaker changes
    .replace(/[’']/g, '')
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

/** Levenshtein distance over words. */
function editDistance(ref: string[], hyp: string[]): number {
  let prev = Array.from({ length: hyp.length + 1 }, (_, j) => j);
  for (let i = 1; i <= ref.length; i++) {
    const cur = [i];
    for (let j = 1; j <= hyp.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[hyp.length];
}

const refAll = vttWords(readFileSync(a.ref, 'utf8'));
const rows = [];
for (const file of a.pred.split(',')) {
  const run = JSON.parse(readFileSync(file, 'utf8'));
  const end = run.audio.t0 + run.audio.seconds;
  const ref = refAll.filter((x) => x.t >= run.audio.t0 && x.t < end).map((x) => x.w);
  const hyp = run.segments.flatMap((s: { text: string }) => norm(s.text));
  const errors = editDistance(ref, hyp);
  const row = {
    pred: file,
    model: run.config.model,
    audioSeconds: run.audio.seconds,
    speed: run.speed,
    refWords: ref.length,
    hypWords: hyp.length,
    wer: Math.round((1000 * errors) / Math.max(1, ref.length)) / 10,
  };
  process.stdout.write(JSON.stringify(row) + '\n');
  rows.push(row);
}
const cols = ['audioSeconds', 'speed', 'refWords', 'hypWords', 'wer'] as const;
console.error(`${'pred'.padEnd(52)}${cols.map((c) => c.padStart(13)).join('')}`);
for (const r of rows) console.error(`${r.pred.slice(-52).padEnd(52)}${cols.map((c) => String(r[c]).padStart(13)).join('')}`);
