// Fetch AMI meetings (CC BY 4.0) with their reference speaker turns, for measuring the speakers
// layer: the Mix-Headset audio from the AMI mirror, the RTTM and UEM of pyannote's
// AMI-diarization-setup ("only_words": turns from the word alignments, the usual benchmark), and
// an MP4 of the audio over a small gray picture (videodoc analyzes videos). Into public/media/ami/
// (not in git).
//
//   node scripts/fetch-ami.mjs [meeting ...]      default: the four below, from the test set
//
// stdout: JSON (meeting -> { video, rttm, uem, minutes }); progress on stderr.
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

export const AMI = 'public/media/ami';
export const MEETINGS = ['ES2004a', 'IS1009a', 'TS3003a', 'EN2002a'];

const SETUP = 'https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words';
const MIRROR = 'https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus';

async function get(url, dest) {
  if (existsSync(dest)) return;
  console.error(`fetching ${url}...`);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  writeFileSync(dest, Buffer.from(await r.arrayBuffer()));
}

if (import.meta.filename === resolve(process.argv[1] ?? '')) {
  const meetings = process.argv.slice(2).length ? process.argv.slice(2) : MEETINGS;
  mkdirSync(AMI, { recursive: true });
  const out = {};
  for (const m of meetings) {
    const base = join(AMI, m);
    for (const set of ['test', 'development', 'train']) {
      try {
        await get(`${SETUP}/rttms/${set}/${m}.rttm`, `${base}.rttm`);
        await get(`${SETUP}/../uems/${set}/${m}.uem`, `${base}.uem`);
        break;
      } catch (e) {
        if (set === 'train') throw new Error(`${m}: not in AMI-diarization-setup (${e.message})`);
      }
    }
    const wav = `${base}.wav`;
    if (!existsSync(`${base}.mp4`)) {
      await get(`${MIRROR}/${m}/audio/${m}.Mix-Headset.wav`, wav);
      console.error(`encoding ${m}.mp4...`);
      execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', 'color=gray:s=320x180:r=5', '-i', wav, '-shortest', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', `${base}.mp4`]);
      rmSync(wav);
    }
    const [, , start, end] = readFileSync(`${base}.uem`, 'utf8').trim().split(/\s+/);
    out[m] = { video: `${base}.mp4`, rttm: `${base}.rttm`, uem: `${base}.uem`, minutes: Math.round((Number(end) - Number(start)) / 6) / 10 };
  }
  console.log(JSON.stringify(out, null, 2));
}
