// Diarization error rate, for measuring the speakers layer against reference turns (AMI's RTTM
// files: scripts/eval-speakers.ts). Frame-based at 10 ms, no collar, overlapped speech scored
// (the setting pyannote reports): reference speakers are mapped one-to-one to found speakers to
// maximize their shared time, then per frame
//   missed    = reference speakers beyond the number found,
//   false     = found speakers beyond the number in the reference,
//   confusion = the rest that are not matched by the mapping,
// each summed and divided by the reference's total speaker time. Pure.

export interface Turn {
  t0: number;
  t1: number;
  speaker: string;
}

export interface Der {
  der: number;
  missed: number;
  falseAlarm: number;
  confusion: number;
  /** Reference speaker time, seconds. */
  total: number;
  /** Found speaker -> the reference speaker it was mapped to. */
  mapping: Record<string, string>;
}

/** Parse RTTM lines (`SPEAKER file 1 start duration <NA> <NA> name <NA> <NA>`). */
export function parseRttm(text: string): Turn[] {
  return text
    .split('\n')
    .map((l) => l.trim().split(/\s+/))
    .filter((f) => f[0] === 'SPEAKER')
    .map((f) => ({ t0: Number(f[3]), t1: Number(f[3]) + Number(f[4]), speaker: f[7] }));
}

/** Optimal one-to-one assignment maximizing the total (Hungarian, on a padded square matrix).
 *  Returns, per row, its column or -1. */
export function hungarianMax(score: number[][]): number[] {
  const rows = score.length;
  const cols = score[0]?.length ?? 0;
  const n = Math.max(rows, cols);
  const max = Math.max(0, ...score.flat());
  // Minimize max - score on an n x n matrix (padding costs max: worth nothing).
  const cost = (i: number, j: number) => (i < rows && j < cols ? max - score[i][j] : max);
  const u = new Float64Array(n + 1);
  const v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1);
  const way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(n + 1).fill(Infinity);
    const used = new Uint8Array(n + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  const out = new Array<number>(rows).fill(-1);
  for (let j = 1; j <= n; j++) if (p[j] - 1 < rows && j - 1 < cols) out[p[j] - 1] = j - 1;
  return out;
}

/** DER of `hyp` against `ref` within `uem` ([start, end], default: the reference's extent). */
export function der(ref: Turn[], hyp: Turn[], uem?: [number, number], frame = 0.01): Der {
  const start = uem?.[0] ?? Math.min(...ref.map((r) => r.t0));
  const end = uem?.[1] ?? Math.max(...ref.map((r) => r.t1));
  const n = Math.max(0, Math.round((end - start) / frame));
  const grid = (turns: Turn[]) => {
    const names = [...new Set(turns.map((t) => t.speaker))];
    const on = names.map(() => new Uint8Array(n));
    for (const t of turns) {
      const a = Math.max(0, Math.round((t.t0 - start) / frame));
      const b = Math.min(n, Math.round((t.t1 - start) / frame));
      on[names.indexOf(t.speaker)].fill(1, a, Math.max(a, b));
    }
    return { names, on };
  };
  const r = grid(ref);
  const h = grid(hyp);
  const overlap = r.on.map((ro) => h.on.map((ho) => ro.reduce((s, v, i) => s + (v & ho[i]), 0)));
  const map = hungarianMax(overlap);
  let missed = 0;
  let falseAlarm = 0;
  let confusion = 0;
  let total = 0;
  for (let i = 0; i < n; i++) {
    let nr = 0;
    let nh = 0;
    let hit = 0;
    for (let a = 0; a < r.names.length; a++) {
      if (!r.on[a][i]) continue;
      nr++;
      if (map[a] >= 0 && h.on[map[a]][i]) hit++;
    }
    for (let b = 0; b < h.names.length; b++) nh += h.on[b][i];
    total += nr;
    missed += Math.max(0, nr - nh);
    falseAlarm += Math.max(0, nh - nr);
    confusion += Math.min(nr, nh) - hit;
  }
  const mapping: Record<string, string> = {};
  map.forEach((b, a) => b >= 0 && overlap[a][b] > 0 && (mapping[h.names[b]] = r.names[a]));
  const s = (x: number) => x * frame;
  return { der: total ? (missed + falseAlarm + confusion) / total : 0, missed: s(missed), falseAlarm: s(falseAlarm), confusion: s(confusion), total: s(total), mapping };
}
