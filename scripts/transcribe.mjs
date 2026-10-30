// Headless runner for the transcript page (transcribe.html): Whisper in the browser.
//
// stdout: one JSON summary per run (JSONL). stderr: progress + a summary table.
// Each run's segments go to results/<clip>-transcript-<name>.json (the JSONL line points at it).
//
//   npm run transcribe -- --video /media/briefing-full.mp4 --seconds 120
//   npm run transcribe -- --video /media/briefing-full.mp4 --model onnx-community/whisper-base,onnx-community/whisper-small
//
// Options:
//   --video    URL path under public/ (required)
//   --model    repo id(s) under public/models (matrix; default onnx-community/whisper-base);
//              fetch with: node scripts/fetch-model.mjs <repo> encoder_model decoder_model_merged_q4
//   --encoder / --decoder   ONNX dtypes (default fp32 / q4)
//   --device   webgpu | wasm (matrix; default webgpu)
//   --seconds  transcribe only the first N seconds (default 0 = all)
//   --lang     (default en)   --mingap  min silent gap reported, seconds (default 2)
//   --vad      1 = Silero VAD gates Whisper and measures speech coverage (default); 0 = off
//   --browser msedge|chrome   --headed   --no-build
import { mkdirSync, writeFileSync } from 'node:fs';
import { basename, extname } from 'node:path';
import { parseArgs } from 'node:util';
import { withBrowser } from './lib/harness.mjs';

const { values: a } = parseArgs({
  options: {
    video: { type: 'string' },
    model: { type: 'string', default: 'onnx-community/whisper-base' },
    encoder: { type: 'string', default: 'fp32' },
    decoder: { type: 'string', default: 'q4' },
    device: { type: 'string', default: 'webgpu' },
    seconds: { type: 'string', default: '0' },
    lang: { type: 'string', default: 'en' },
    mingap: { type: 'string', default: '2' },
    vad: { type: 'string', default: '1' },
    words: { type: 'string', default: '0' },
    browser: { type: 'string', default: 'msedge' },
    headed: { type: 'boolean', default: false },
    'no-build': { type: 'boolean', default: false },
  },
});
if (!a.video) {
  console.error('usage: npm run transcribe -- --video /media/<clip>.mp4 [--model <repo>,...] [--seconds 120]');
  process.exit(2);
}
const list = (s) =>
  s
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean);
const clip = basename(a.video, extname(a.video));

const rows = [];
await withBrowser({ build: !a['no-build'], browser: a.browser, headed: a.headed }, async (open) => {
  for (const model of list(a.model))
    for (const device of list(a.device)) {
      const name = `${model.split('/').pop()}-${a.encoder}-${a.decoder}-${device}${a.vad === '0' ? '-novad' : ''}${a.words === '1' ? '-words' : ''}${a.seconds !== '0' ? `-${a.seconds}s` : ''}`;
      console.error(`running ${clip} ${name}...`);
      const params = { video: a.video, model, encoder: a.encoder, decoder: a.decoder, device, seconds: a.seconds, lang: a.lang, mingap: a.mingap, vad: a.vad, words: a.words };
      const { result, page } = await open('/transcribe.html', params, { timeoutMs: 60 * 60_000 });
      await page.close();
      result.name = name;
      if (result.segments) {
        mkdirSync('results', { recursive: true });
        const file = `results/${clip}-transcript-${name}.json`;
        writeFileSync(file, JSON.stringify(result, null, 1));
        delete result.segments;
        delete result.speech;
        result.coverage.gaps = result.coverage.gaps.length;
        result.file = file;
      }
      process.stdout.write(JSON.stringify(result) + '\n');
      rows.push(result);
    }
});

console.error(`\n${'name'.padEnd(44)}${['audio s', 'load ms', 'asr ms', 'speed x', 'words', 'speech %', 'gaps'].map((c) => c.padStart(10)).join('')}`);
for (const r of rows) {
  if (r.error) console.error(`${r.name.padEnd(44)} ERROR ${r.error}`);
  else
    console.error(
      `${r.name.padEnd(44)}${[r.audio.seconds, r.loadMs, r.asrMs, r.speed, r.words, r.coverage.speechPct, r.coverage.gaps].map((c) => String(c).padStart(10)).join('')}`,
    );
}
