// Redaction end to end in headless Edge: the copy has the source's frames and length, faces
// found in the source are not found again in the copy, and bleeped words carry the tone exactly
// over the logged spans. Also the keyframe images. Run: npm run test:browser
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { after, test } from 'node:test';
import { analyze, markdown, redact, ROOT, speakers } from '../../cli/lib.ts';
import { parseRttm } from '../../src/audio/der.ts';
import { summarize } from '../../src/doc/index.ts';
import { context, fixture, fresh, loadDoc, near, notReady, saveBaseline, speed } from './lib.ts';

const ctx = context();
after(async () => {
  saveBaseline();
  await ctx.browser.close();
});

function ffprobe(path: string) {
  const r = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-show_streams', '-of', 'json', path], { encoding: 'utf8' }));
  return r.streams as { codec_type: string; codec_name: string; nb_frames?: string; duration?: string }[];
}

const hasFfmpeg = (() => {
  try {
    execFileSync('ffmpeg', ['-version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
})();

/** Mono 16 kHz samples of a file's audio, decoded by ffmpeg. */
function audio(path: string): Float32Array {
  const buf = execFileSync('ffmpeg', ['-v', 'error', '-i', path, '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'], { maxBuffer: 1 << 28 });
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

/** Per 10 ms frame: is most of its power a 1 kHz tone (Goertzel)? */
function toned(x: Float32Array, rate = 16000): boolean[] {
  const n = rate / 100;
  const w = (2 * Math.PI * 1000) / rate;
  const out: boolean[] = [];
  for (let i = 0; i + n <= x.length; i += n) {
    let s1 = 0;
    let s2 = 0;
    let energy = 0;
    for (let k = 0; k < n; k++) {
      const v = x[i + k];
      energy += v * v;
      const s = v + 2 * Math.cos(w) * s1 - s2;
      s2 = s1;
      s1 = s;
    }
    const power = (s1 * s1 + s2 * s2 - 2 * Math.cos(w) * s1 * s2) * (2 / n);
    out.push(energy > 1e-4 && power / energy > 0.8);
  }
  return out;
}

test('walk-crowd: blurred copy keeps frames and length, and its faces are not found again', async (t) => {
  const why = (await notReady(ctx)) ?? (hasFfmpeg ? undefined : 'ffmpeg not found (needed to inspect the output)');
  if (why) return t.skip(why);
  const path = fixture('walk-crowd');
  if (!path) return t.skip('fixture not cut: node scripts/fixtures.mjs');
  const video = fresh(path);
  const r = (await redact(ctx, video, { cover: 'faces' })) as { out: string; frames: number; ms: number; totals: { framesWithRegions: number } };
  const doc = loadDoc(video);
  assert.equal(r.frames, doc.source.frames);
  const [src, out] = [ffprobe(video), ffprobe(r.out)];
  const v = (s: typeof src) => s.find((x) => x.codec_type === 'video')!;
  const a = (s: typeof src) => s.find((x) => x.codec_type === 'audio')!;
  assert.equal(v(out).codec_name, 'h264');
  assert.equal(Number(v(out).nb_frames), doc.source.frames);
  assert.ok(Math.abs(Number(a(out).duration) - Number(a(src).duration)) < 0.05, 'audio length kept');
  assert.ok(existsSync(`${r.out}.redactions.json`));
  speed(t, 'walk-crowd.redact.fps', Math.round((r.frames * 1000) / r.ms));

  // Faces in the copy, by the same detector: what the blur leaves recognizable as a face.
  const copy = fresh(r.out);
  await analyze(ctx, copy, { layers: ['faces'] });
  const before = summarize(doc).faces.faceFrames as number;
  const left = summarize(loadDoc(copy)).faces.faceFrames as number;
  t.diagnostic(`face-frames: ${before} in the source, ${left} in the blurred copy`);
  assert.ok(left < before * 0.2, 'blurred faces are mostly not detected');
  near(t, 'walk-crowd.redact.facesLeftPct', Math.round((1000 * left) / before) / 10, { abs: 2 });
});

test('tts-words: bleeped words carry the tone over the logged spans and nowhere else; the video is copied', async (t) => {
  const why = (await notReady(ctx)) ?? (hasFfmpeg ? undefined : 'ffmpeg not found (needed to inspect the output)');
  if (why) return t.skip(why);
  const video = fresh(fixture('tts-words')!);
  const r = (await redact(ctx, video, { cover: 'none', words: 'Ackover,Daniel Reyes,Lindquist', profanity: true })) as {
    out: string; log: string; frames: number; audio: { matches: number; seconds: number };
  };
  assert.equal(r.frames, 0, 'video copied, not re-encoded');
  const log = JSON.parse(readFileSync(r.log, 'utf8')) as { audio: { spans: { t0: number; t1: number; kept: boolean; text: string }[] } };
  const spans = log.audio.spans.filter((s) => !s.kept);
  near(t, 'tts-words.bleep.matches', r.audio.matches, { abs: 0 });
  near(t, 'tts-words.bleep.seconds', r.audio.seconds, { abs: 0.3 });

  const tone = toned(audio(r.out));
  const at = (s: number) => Math.round(s * 100);
  const inside = new Set<number>();
  for (const s of spans) {
    let hit = 0;
    let n = 0;
    for (let f = at(s.t0) + 2; f < at(s.t1) - 2; f++, n++) {
      inside.add(f);
      hit += tone[f] ? 1 : 0;
    }
    assert.ok(hit / n > 0.95, `"${s.text}" ${s.t0}-${s.t1}: tone in ${hit}/${n} frames`);
  }
  // Frames well outside every span: no tone (the fades take 8 ms at each edge).
  const near0 = (f: number) => spans.some((s) => f >= at(s.t0) - 3 && f <= at(s.t1) + 3);
  const stray = tone.filter((v, f) => v && !near0(f)).length;
  assert.equal(stray, 0, 'no tone outside the spans');
  // The source never has the tone.
  assert.equal(toned(audio(video)).filter(Boolean).length, 0);
  // The video stream is byte-for-byte the source's.
  const md5 = (p: string) => execFileSync('ffmpeg', ['-v', 'error', '-i', p, '-map', '0:v', '-c', 'copy', '-f', 'md5', '-'], { encoding: 'utf8' }).trim();
  assert.equal(md5(r.out), md5(video));
});

test('tts-dialog: a speaker named in the speakers file is bleeped over their turns and nowhere else', async (t) => {
  const why = (await notReady(ctx)) ?? (hasFfmpeg ? undefined : 'ffmpeg not found (needed to inspect the output)');
  if (why) return t.skip(why);
  const video = fresh(fixture('tts-dialog')!);
  await analyze(ctx, video, { layers: ['speakers'] });
  speakers(ctx, video, { name: { 3: 'Cal' } });
  const r = (await redact(ctx, video, { cover: 'none', speakers: 'Cal' })) as { out: string; log: string; audio: { turns: number; seconds: number } };
  const log = JSON.parse(readFileSync(r.log, 'utf8')) as { audio: { spans: { t0: number; t1: number; speaker?: string }[] }; caveat: string };
  const spans = log.audio.spans;
  assert.ok(spans.length > 0 && spans.every((s) => s.speaker === 'Cal'));
  assert.match(log.caveat, /diarization/);
  near(t, 'tts-dialog.bleepSpeaker.seconds', r.audio.seconds, { abs: 1 });
  const tone = toned(audio(r.out));
  const at = (s: number) => Math.round(s * 100);
  const near0 = (f: number) => spans.some((s) => f >= at(s.t0) - 3 && f <= at(s.t1) + 3);
  assert.equal(tone.filter((v, f) => v && !near0(f)).length, 0, 'no tone outside the spans');
  // The voice that is Cal (the reference speaker overlapping Cal's spans most) is covered.
  const ref = parseRttm(readFileSync(join(ROOT, 'tests/fixtures/tts-dialog.rttm'), 'utf8'));
  const overlap = (a: number, b: number) => spans.reduce((n, s) => n + Math.max(0, Math.min(b, s.t1) - Math.max(a, s.t0)), 0);
  const voices = [...new Set(ref.map((x) => x.speaker))];
  const cal = voices.sort((x, y) => ref.filter((z) => z.speaker === y).reduce((n, z) => n + overlap(z.t0, z.t1), 0) - ref.filter((z) => z.speaker === x).reduce((n, z) => n + overlap(z.t0, z.t1), 0))[0];
  const turns = ref.filter((x) => x.speaker === cal);
  const said = turns.reduce((n, x) => n + x.t1 - x.t0, 0);
  const toneIn = turns.reduce((n, x) => n + tone.slice(at(x.t0), at(x.t1)).filter(Boolean).length / 100, 0);
  t.diagnostic(`Cal is ${cal}: ${toneIn.toFixed(1)} of ${said.toFixed(1)} s bleeped`);
  assert.ok(toneIn / said > 0.95, 'all that voice says is bleeped');
});

test('rickroll-cuts: keyframe images are written and linked from the Markdown', async (t) => {
  const why = await notReady(ctx);
  if (why) return t.skip(why);
  const path = fixture('rickroll-cuts');
  if (!path) return t.skip('fixture not cut: node scripts/fixtures.mjs');
  const video = fresh(path);
  const r = await markdown(ctx, video, { images: true, layers: ['scenes'] });
  assert.ok(r.imageFiles!.length > 0);
  for (const f of r.imageFiles!) {
    const b = readFileSync(f);
    assert.equal(b.readUInt16BE(0), 0xffd8, `${f} is a JPEG`);
    assert.ok(r.text!.includes(f.replace(/\\/g, '/')), `${f} linked`);
  }
  assert.match(r.text!, /^## Shot 1 /m);
});
