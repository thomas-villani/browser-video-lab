// Join the turns tests/fixtures/tts-dialog.ps1 spoke (NN-<voice>.wav in <dir>) into the speakers
// fixture: tests/fixtures/tts-dialog.mp4 (a gray picture, the turns 0.6 s apart) and its exact
// turns, tests/fixtures/tts-dialog.rttm: where each voice is audible (TTS pads its WAVs and pauses
// between sentences; pauses over 0.25 s split a turn, as in AMI's word-based references).
//
//   pwsh -File tests/fixtures/tts-dialog.ps1 <dir>      (PowerShell 7: the Mark voice is OneCore)
//   node scripts/tts-dialog.mjs <dir>
import { execFileSync } from 'node:child_process';
import { readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const dir = process.argv[2];
if (!dir) {
  console.error('usage: node scripts/tts-dialog.mjs <dir with NN-<voice>.wav>');
  process.exit(2);
}
const RATE = 16000;
const GAP = 0.6;
const ffmpeg = (args) => execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', ...args]);
/** Audible stretches of 16-bit PCM, seconds: 10 ms frames above -46 dBFS, gaps under 0.25 s closed. */
function audible(pcm) {
  const hop = RATE / 100;
  const spans = [];
  for (let i = 0; i * hop * 2 < pcm.length; i++) {
    let e = 0;
    const n = Math.min(hop, pcm.length / 2 - i * hop);
    for (let k = 0; k < n; k++) e += (pcm.readInt16LE((i * hop + k) * 2) / 32768) ** 2;
    if (Math.sqrt(e / n) < 0.005) continue;
    const last = spans.at(-1);
    if (last && i / 100 - last[1] < 0.25) last[1] = (i + 1) / 100;
    else spans.push([i / 100, (i + 1) / 100]);
  }
  return spans;
}

const parts = [];
const rttm = [];
let t = GAP;
for (const f of readdirSync(dir).filter((f) => /^\d\d-.+\.wav$/.test(f)).sort()) {
  // Each turn as 16 kHz mono 16-bit PCM.
  const pcm = ffmpeg(['-i', join(dir, f), '-f', 's16le', '-ac', '1', '-ar', String(RATE), '-']);
  const seconds = pcm.length / 2 / RATE;
  parts.push(Buffer.alloc(Math.round(GAP * RATE) * 2), pcm);
  for (const [a, b] of audible(pcm)) rttm.push(`SPEAKER tts-dialog 1 ${(t + a).toFixed(3)} ${(b - a).toFixed(3)} <NA> <NA> ${f.slice(3, -4)} <NA> <NA>`);
  t += seconds + GAP;
}
parts.push(Buffer.alloc(Math.round(GAP * RATE) * 2));
const raw = join(dir, 'dialog.raw');
writeFileSync(raw, Buffer.concat(parts));
ffmpeg(['-f', 'lavfi', '-i', 'color=gray:s=640x360:r=25', '-f', 's16le', '-ar', String(RATE), '-ac', '1', '-i', raw, '-shortest', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '96k', 'tests/fixtures/tts-dialog.mp4']);
rmSync(raw);
writeFileSync('tests/fixtures/tts-dialog.rttm', `${rttm.join('\n')}\n`);
console.log(JSON.stringify({ turns: rttm.length, seconds: +t.toFixed(1), bytes: readFileSync('tests/fixtures/tts-dialog.mp4').length }));
