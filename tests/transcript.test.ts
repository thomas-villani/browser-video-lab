// Run: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { dropRepeats, groupSpans, mergeSpans, speechCoverage, speechSpans } from '../src/transcript.ts';

const step = 0.032; // Silero window at 16 kHz
/** Probabilities: speech (0.9) inside the given [start, end) second ranges, else 0.05. */
function probs(seconds: number, speech: [number, number][]): number[] {
  return Array.from({ length: Math.round(seconds / step) }, (_, i) => (speech.some(([a, b]) => i * step >= a && i * step < b) ? 0.9 : 0.05));
}

test('vad spans: onsets/offsets within a window, padded, short blips dropped', () => {
  const spans = speechSpans(
    probs(10, [
      [1, 3],
      [5, 5.1],
      [6, 8],
    ]),
    step,
    0,
    { threshold: 0.5, minSilence: 0.3, minSpeech: 0.25, pad: 0 },
  );
  assert.equal(spans.length, 2); // the 0.1 s blip at 5 s is dropped
  assert.ok(Math.abs(spans[0].start - 1) < step && Math.abs(spans[0].end - 3) < 2 * step, JSON.stringify(spans));
  assert.ok(Math.abs(spans[1].start - 6) < step, JSON.stringify(spans));
});

test('vad spans: pauses shorter than minSilence do not split speech', () => {
  const spans = speechSpans(
    probs(6, [
      [1, 2],
      [2.2, 3],
    ]),
    step,
    10,
    { threshold: 0.5, minSilence: 0.3, minSpeech: 0.25, pad: 0 },
  );
  assert.equal(spans.length, 1);
  assert.ok(spans[0].start >= 10.99 && spans[0].start <= 11.04, `t0 offset applied: ${spans[0].start}`);
});

test('groupSpans: packs neighbors up to maxLen and splits long speech evenly', () => {
  assert.deepEqual(
    groupSpans(
      [
        { start: 0, end: 10 },
        { start: 11, end: 20 },
        { start: 30, end: 35 },
      ],
      28,
      1.5,
    ),
    [
      { start: 0, end: 20 },
      { start: 30, end: 35 },
    ],
  );
  const long = groupSpans([{ start: 0, end: 70 }], 28);
  assert.equal(long.length, 3);
  assert.ok(long.every((s) => s.end - s.start <= 28));
});

test('speech coverage: gaps are the silent stretches', () => {
  const c = speechCoverage(
    mergeSpans([
      { start: 5, end: 20 },
      { start: 21, end: 30 },
      { start: 50, end: 55 },
    ]),
    60,
    2,
  );
  assert.deepEqual(c.gaps, [
    { start: 0, end: 5 },
    { start: 30, end: 50 },
    { start: 55, end: 60 },
  ]);
  assert.equal(c.speechSeconds, 29);
});

test('dropRepeats: collapses a repetition loop, keeps legitimate repeats apart', () => {
  const t = (text: string) => ({ text });
  const out = dropRepeats([t('Allena Sharazi.'), t('Allena Sharazi.'), t('allena sharazi'), t('Hi.'), t('Allena Sharazi.')]);
  assert.deepEqual(
    out.map((s) => s.text),
    ['Allena Sharazi.', 'Hi.', 'Allena Sharazi.'],
  );
});
