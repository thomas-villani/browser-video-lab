// Fetch the default models into public/models/ from their upstream sources and check each ONNX
// file's SHA-256 against the files the measurements in NOTES.md were made with. Files already
// present and matching are kept. YuNet is relaxed to any input size with uv (tools/dynamic_yunet.py).
// Used by a fresh checkout (docs/getting-started.md) and the GitHub Pages build.
//
//   node scripts/fetch-models.mjs [name...]     (default: all; names below)
//
// stdout: JSON (name -> { files, bytes }); progress on stderr.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream, existsSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const HF = 'https://huggingface.co';

/** name -> files: [path under public/models, URL, sha256 (undefined: not checked)]; `hf` fetches
 *  a Hugging Face repo's config files (and the listed ONNX files) with fetch-model.mjs. */
export const MODELS = {
  vad: {
    files: [
      [
        'silero_vad.onnx',
        'https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx',
        '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3',
      ],
    ],
  },
  whisper: {
    hf: ['onnx-community/whisper-base_timestamped', 'encoder_model', 'decoder_model_merged_q4'],
    files: [
      ['onnx-community/whisper-base_timestamped/onnx/encoder_model.onnx', undefined, '7fcea817bb2be4d86729b521e5a7fcbec28fa743edfed67e882b33ff15852540'],
      ['onnx-community/whisper-base_timestamped/onnx/decoder_model_merged_q4.onnx', undefined, 'fc1902ce2e42c69b2346d8e2a98898c60c01da1e6a64ae90f41d22350ac7db13'],
    ],
  },
  segmentation: {
    hf: ['onnx-community/pyannote-segmentation-3.0', 'model'],
    files: [['onnx-community/pyannote-segmentation-3.0/onnx/model.onnx', undefined, '057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25']],
  },
  voices: {
    hf: ['onnx-community/wespeaker-voxceleb-resnet34-LM', 'model'],
    files: [['onnx-community/wespeaker-voxceleb-resnet34-LM/onnx/model.onnx', undefined, '3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb']],
  },
  faces: {
    files: [
      ['yunet_face.onnx', `${HF}/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx`, '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'],
    ],
    make: () =>
      run('uv', ['run', '--with', 'onnx', '--with', 'onnxruntime', 'python', '-I', 'tools/dynamic_yunet.py', 'public/models/yunet_face.onnx', 'public/models/yunet_face_dyn.onnx']),
    made: 'yunet_face_dyn.onnx',
  },
  plates: {
    files: [
      [
        'yolo_plate_t640.onnx',
        'https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx',
        'c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076',
      ],
    ],
  },
};

const run = (cmd, args) => execFileSync(cmd, args, { stdio: ['ignore', 'ignore', 'inherit'] });

async function sha256(path) {
  const h = createHash('sha256');
  await pipeline(createReadStream(path), h);
  return h.digest('hex');
}

async function download(url, dest) {
  mkdirSync(dirname(dest), { recursive: true });
  console.error(`fetching ${url}...`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  await pipeline(Readable.fromWeb(res.body), createWriteStream(dest));
}

/** Every file the default models serve from public/models (paths below it): the Pages site ships these. */
export function modelFiles() {
  return Object.values(MODELS).flatMap((m) => [
    ...m.files.map(([p]) => p),
    ...(m.made ? [m.made] : []),
    ...(m.hf
      ? readdirSync(`public/models/${m.hf[0]}`)
          .filter((f) => f.endsWith('.json'))
          .map((f) => `${m.hf[0]}/${f}`)
      : []),
  ]);
}

if (import.meta.filename === resolve(process.argv[1] ?? '')) await main();

async function main() {
  const names = process.argv.slice(2);
  const unknown = names.filter((n) => !MODELS[n]);
  if (unknown.length) {
    console.error(`unknown: ${unknown.join(', ')} (known: ${Object.keys(MODELS).join(', ')})`);
    process.exit(2);
  }
  const out = {};
  for (const name of names.length ? names : Object.keys(MODELS)) {
    const m = MODELS[name];
    if (m.hf) run(process.execPath, ['scripts/fetch-model.mjs', ...m.hf]);
    let bytes = 0;
    for (const [path, url, sum] of m.files) {
      const dest = `public/models/${path}`;
      if (url && !(existsSync(dest) && (!sum || (await sha256(dest)) === sum))) await download(url, dest);
      if (sum && (await sha256(dest)) !== sum) throw new Error(`${dest}: SHA-256 differs from the measured file (the upstream file changed?)`);
      bytes += statSync(dest).size;
    }
    if (m.made && !existsSync(`public/models/${m.made}`)) m.make();
    out[name] = { files: [...m.files.map(([p]) => p), ...(m.made ? [m.made] : [])], bytes };
    console.error(`${name}: ok`);
  }
  console.log(JSON.stringify(out));
}
