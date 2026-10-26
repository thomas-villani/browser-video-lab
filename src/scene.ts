// Scene-change scoring from frame thumbnails, following ffmpeg's `select=gt(scene,T)`:
//   mafd  = mean absolute frame difference, in percent of full range
//   score = clip(min(mafd, |mafd - previous mafd|) / 100, 0, 1)
// Subtracting the previous mafd suppresses sustained motion (a pan scores high every frame);
// a cut is a one-frame spike. Pure TS: runs in the browser, a worker, or Node.

export class SceneScorer {
  private prev?: Float32Array;
  private prevMafd = 0;

  /** Feed the next frame's thumbnail (same size every frame); returns its score in [0, 1]. */
  push(thumb: Float32Array): number {
    const prev = this.prev;
    this.prev = thumb;
    if (!prev) return 0;
    let sad = 0;
    for (let i = 0; i < thumb.length; i++) sad += Math.abs(thumb[i] - prev[i]);
    const mafd = (100 * sad) / thumb.length;
    const score = Math.min(1, Math.max(0, Math.min(mafd, Math.abs(mafd - this.prevMafd)) / 100));
    this.prevMafd = mafd;
    return score;
  }
}
