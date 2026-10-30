// Minimum-cost linear assignment (Hungarian algorithm with potentials, O(n^2 m)),
// with a cost limit like lap.lapjv(cost_limit=...) as used by ByteTrack.

export interface Assignment {
  matches: [row: number, col: number][];
  unmatchedRows: number[];
  unmatchedCols: number[];
}

/** Solve for n <= m. Returns col index assigned to each row. */
function hungarian(cost: number[][], n: number, m: number): number[] {
  const INF = Number.POSITIVE_INFINITY;
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(m + 1).fill(0);
  const p = new Array<number>(m + 1).fill(0); // p[j] = row (1-based) matched to col j
  const way = new Array<number>(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(m + 1).fill(INF);
    const used = new Array<boolean>(m + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = INF;
      let j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) [minv[j], way[j]] = [cur, j0];
        if (minv[j] < delta) [delta, j1] = [minv[j], j];
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  const rowToCol = new Array<number>(n).fill(-1);
  for (let j = 1; j <= m; j++) if (p[j]) rowToCol[p[j] - 1] = j - 1;
  return rowToCol;
}

/**
 * Assign rows to cols minimizing total cost; pairs with cost > limit stay unmatched.
 * `cols` must be given explicitly because a matrix with zero rows can't carry it.
 */
export function linearAssignment(cost: number[][], limit: number, cols: number): Assignment {
  const rows = cost.length;
  if (!rows || !cols) {
    return { matches: [], unmatchedRows: [...Array(rows).keys()], unmatchedCols: [...Array(cols).keys()] };
  }
  // Costs above the limit are clamped so they never beat a real match, then discarded.
  const BIG = limit + 1e5;
  const clamp = (c: number) => (c > limit ? BIG : c);
  const transposed = rows > cols;
  const c = transposed ? cost[0].map((_, j) => cost.map((row) => clamp(row[j]))) : cost.map((row) => row.map(clamp));
  const assign = hungarian(c, c.length, c[0].length);

  const matches: [number, number][] = [];
  assign.forEach((j, i) => {
    if (j < 0) return;
    const [r, col] = transposed ? [j, i] : [i, j];
    if (cost[r][col] <= limit) matches.push([r, col]);
  });
  const mr = new Set(matches.map((m) => m[0]));
  const mc = new Set(matches.map((m) => m[1]));
  return {
    matches,
    unmatchedRows: [...Array(rows).keys()].filter((i) => !mr.has(i)),
    unmatchedCols: [...Array(cols).keys()].filter((j) => !mc.has(j)),
  };
}
