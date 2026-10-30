// Constant-velocity Kalman filter over box state (cx, cy, aspect, h) — a port of the
// filter used by SORT/DeepSORT/ByteTrack. State is 8-dim: position (4) + velocity (4).
// Pure TypeScript, no dependencies; runs in browsers, workers, and Node.

export type Vec = number[];
export type Mat = number[][];

const STD_POS = 1 / 20;
const STD_VEL = 1 / 160;

function zeros(r: number, c: number): Mat {
  return Array.from({ length: r }, () => new Array<number>(c).fill(0));
}

function diag(v: Vec): Mat {
  const m = zeros(v.length, v.length);
  v.forEach((x, i) => (m[i][i] = x));
  return m;
}

function matmul(a: Mat, b: Mat): Mat {
  const out = zeros(a.length, b[0].length);
  for (let i = 0; i < a.length; i++)
    for (let k = 0; k < b.length; k++) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < b[0].length; j++) out[i][j] += aik * b[k][j];
    }
  return out;
}

function transpose(a: Mat): Mat {
  return a[0].map((_, j) => a.map((row) => row[j]));
}

/** Gauss-Jordan inverse for the small (4x4) innovation covariance. */
function inverse(a: Mat): Mat {
  const n = a.length;
  const m = a.map((row, i) => [...row, ...diag(new Array<number>(n).fill(1))[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
    [m[c], m[p]] = [m[p], m[c]];
    const pivot = m[c][c];
    for (let j = 0; j < 2 * n; j++) m[c][j] /= pivot;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = m[r][c];
      if (f !== 0) for (let j = 0; j < 2 * n; j++) m[r][j] -= f * m[c][j];
    }
  }
  return m.map((row) => row.slice(n));
}

// Transition: position += velocity (one time step = one tracker update).
const F: Mat = (() => {
  const f = diag(new Array<number>(8).fill(1));
  for (let i = 0; i < 4; i++) f[i][i + 4] = 1;
  return f;
})();
const FT = transpose(F);

export interface KalmanState {
  mean: Vec;
  cov: Mat;
}

/** Measurement = [cx, cy, aspect (w/h), h]. */
export function initiate(z: Vec): KalmanState {
  const h = z[3];
  const std = [2 * STD_POS * h, 2 * STD_POS * h, 1e-2, 2 * STD_POS * h, 10 * STD_VEL * h, 10 * STD_VEL * h, 1e-5, 10 * STD_VEL * h];
  return { mean: [...z, 0, 0, 0, 0], cov: diag(std.map((s) => s * s)) };
}

export function predict(s: KalmanState): KalmanState {
  const h = s.mean[3];
  const std = [STD_POS * h, STD_POS * h, 1e-2, STD_POS * h, STD_VEL * h, STD_VEL * h, 1e-5, STD_VEL * h];
  const mean = matmul(
    F,
    s.mean.map((v) => [v]),
  ).map((r) => r[0]);
  const cov = matmul(matmul(F, s.cov), FT);
  std.forEach((v, i) => (cov[i][i] += v * v));
  return { mean, cov };
}

export function update(s: KalmanState, z: Vec): KalmanState {
  const h = s.mean[3];
  const r = [STD_POS * h, STD_POS * h, 1e-1, STD_POS * h];
  // Project to measurement space: H selects the first 4 state dims.
  const S = s.cov.slice(0, 4).map((row, i) => row.slice(0, 4).map((v, j) => v + (i === j ? r[i] * r[i] : 0)));
  const PHt = s.cov.map((row) => row.slice(0, 4)); // 8x4
  const K = matmul(PHt, inverse(S)); // 8x4
  const innovation = z.map((v, i) => v - s.mean[i]);
  const mean = s.mean.map((v, i) => v + K[i].reduce((acc, k, j) => acc + k * innovation[j], 0));
  const KSKt = matmul(matmul(K, S), transpose(K));
  const cov = s.cov.map((row, i) => row.map((v, j) => v - KSKt[i][j]));
  return { mean, cov };
}

/** [x, y, w, h] (top-left) <-> [cx, cy, aspect, h]. */
export function toXyah([x, y, w, h]: Vec): Vec {
  return [x + w / 2, y + h / 2, w / h, h];
}

export function fromXyah([cx, cy, a, h]: Vec): Vec {
  const w = a * h;
  return [cx - w / 2, cy - h / 2, w, h];
}
