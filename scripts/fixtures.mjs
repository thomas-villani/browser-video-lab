// Cut the browser tests' video fixtures from the measured footage in public/media (not in git:
// see README) into tests/fixtures/local/ (gitignored). Each is a short window re-encoded with
// ffmpeg so it starts on a keyframe and has an exact length. The TTS fixture
// (tests/fixtures/tts-words.mp4) is synthetic and committed.
//
//   node scripts/fixtures.mjs [--force]
//
// stdout: JSON (fixture -> { path, status }); progress on stderr.
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

/** name -> [source in public/media, start s, length s, what it exercises] */
export const FIXTURES = {
  'walk-crowd': ['walk.mp4', 30, 10, 'a crowded street: many small faces, two scene cuts'],
  'dashcam-plates': ['dashcam-KHkqJpPJBgs.mp4', 0, 10, 'traffic: plates near and far, some faces'],
  'rickroll-cuts': ['rickroll.mp4', 0, 12, 'a music video: quick cuts, close faces, sung audio'],
};

export const LOCAL = 'tests/fixtures/local';

export function fixturePath(name) {
  return join(LOCAL, `${name}.mp4`);
}

function ffmpeg(args) {
  try {
    execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', ...args], { stdio: ['ignore', 'ignore', 'inherit'] });
  } catch (e) {
    if (e.code === 'ENOENT') throw new Error('ffmpeg not found on PATH (it cuts the fixtures; install it, e.g. winget install Gyan.FFmpeg)');
    throw e;
  }
}

if (import.meta.filename === resolve(process.argv[1] ?? '')) {
  const force = process.argv.includes('--force');
  mkdirSync(LOCAL, { recursive: true });
  const out = {};
  for (const [name, [src, start, len]] of Object.entries(FIXTURES)) {
    const from = join('public/media', src);
    const to = fixturePath(name);
    if (!existsSync(from)) {
      console.error(`${name}: skipped, ${from} not found`);
      out[name] = { path: to, status: 'missing source' };
      continue;
    }
    if (existsSync(to) && !force && statSync(to).size > 0) {
      out[name] = { path: to, status: 'exists' };
      continue;
    }
    console.error(`${name}: ${src} ${start}-${start + len} s`);
    ffmpeg(['-ss', String(start), '-i', from, '-t', String(len), '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', to]);
    out[name] = { path: to, status: 'cut' };
  }
  console.log(JSON.stringify(out, null, 1));
}
