// Face-labeling review page (label.html). Shows each task frame with candidate faces from
// RetinaFace + YuNet; the reviewer accepts/rejects/adds boxes; each frame is saved to the dev
// server's label store (vite.config.ts → results/labels/faces.jsonl). Run with `npm run dev`.
//
//   ?start=<index>  open a specific task (default: first unlabeled)

export {};

type Status = 'face' | 'rejected' | 'uncertain';
type Box = [number, number, number, number];

interface Candidate {
  box: Box;
  retina?: number;
  yunet?: number;
}
interface Task {
  clip: string;
  frame: number;
  kind: string;
  image: string;
  candidates: Candidate[];
}
interface Face {
  box: Box;
  status: Status;
  source: 'both' | 'retina' | 'yunet' | 'human';
}
interface Saved {
  clip: string;
  frame: number;
  faces: Face[];
}

const canvas = document.getElementById('c') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const $ = (id: string) => document.getElementById(id)!;

const tasks: Task[] = await (await fetch('/label/tasks.json')).json();
const saved: Record<string, Saved> = await (await fetch('/api/labels')).json();
const key = (t: Task) => `${t.clip}/${t.frame}`;

let index = Number(new URLSearchParams(location.search).get('start') ?? NaN);
if (!Number.isFinite(index)) index = Math.max(0, tasks.findIndex((t) => !saved[key(t)]));
let faces: Face[] = [];
let img = new Image();
let scale = 1;
let mouse = { x: -1, y: -1 };
let drag: { x: number; y: number } | null = null;
let zoom = false;
let hide = false;
let shownAt = 0;

/** Initial status: on when either detector is confident or both agree; otherwise a suggestion. */
function initial(c: Candidate): Face {
  const r = c.retina ?? 0;
  const y = c.yunet ?? 0;
  const on = r >= 0.7 || y >= 0.7 || (r >= 0.5 && y >= 0.5);
  return { box: c.box, status: on ? 'face' : 'rejected', source: c.retina && c.yunet ? 'both' : c.retina ? 'retina' : 'yunet' };
}

async function show(i: number) {
  index = Math.min(Math.max(0, i), tasks.length - 1);
  const t = tasks[index];
  faces = saved[key(t)]?.faces.map((f) => ({ ...f })) ?? t.candidates.map(initial);
  img = new Image();
  img.src = t.image;
  await img.decode();
  const maxW = document.body.clientWidth;
  const maxH = window.innerHeight - 150;
  scale = Math.min(maxW / img.width, maxH / img.height, 1);
  canvas.width = Math.round(img.width * scale);
  canvas.height = Math.round(img.height * scale);
  history.replaceState(null, '', `?start=${index}`);
  shownAt = performance.now();
  draw();
}

const COLORS: Record<Status, string> = { face: '#00e676', rejected: '#ff5252', uncertain: '#ffab40' };

function drawBoxes(s: number, ox = 0, oy = 0) {
  for (const f of faces) {
    const [x, y, w, h] = f.box;
    ctx.strokeStyle = f.source === 'human' && f.status === 'face' ? '#40c4ff' : COLORS[f.status];
    ctx.lineWidth = f === hovered() ? 3 : 2;
    ctx.setLineDash(f.status === 'rejected' ? [4, 4] : []);
    ctx.strokeRect((x - ox) * s, (y - oy) * s, w * s, h * s);
  }
  ctx.setLineDash([]);
}

function draw() {
  const t = tasks[index];
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  if (!hide) drawBoxes(scale);
  if (drag) {
    ctx.strokeStyle = '#40c4ff';
    ctx.strokeRect(drag.x * scale, drag.y * scale, (mouse.x - drag.x) * scale, (mouse.y - drag.y) * scale);
  }
  if (zoom && mouse.x >= 0) {
    // 4x magnifier lens around the cursor, drawn in the corner away from it.
    const z = 4;
    const lens = 320;
    const src = lens / z / scale;
    const sx = mouse.x - src / 2;
    const sy = mouse.y - src / 2;
    const lx = mouse.x * scale > canvas.width / 2 ? 8 : canvas.width - lens - 8;
    ctx.save();
    ctx.beginPath();
    ctx.rect(lx, 8, lens, lens);
    ctx.clip();
    ctx.drawImage(img, sx, sy, src, src, lx, 8, lens, lens);
    ctx.translate(lx, 8);
    if (!hide) drawBoxes(lens / src, sx, sy);
    ctx.restore();
    ctx.strokeStyle = '#fff';
    ctx.strokeRect(lx, 8, lens, lens);
  }
  const done = tasks.filter((x) => saved[key(x)]).length;
  const c = (s: Status) => faces.filter((f) => f.status === s).length;
  $('title').textContent = `${t.clip} · frame ${t.frame} · ${t.kind}`;
  $('counts').textContent = `faces ${c('face')} · uncertain ${c('uncertain')} · rejected ${c('rejected')}`;
  $('progress').textContent = `task ${index + 1}/${tasks.length} · ${done} labeled`;
  $('saved').textContent = saved[key(t)] ? '✓ saved' : '';
  ($('bar') as HTMLDivElement).style.width = `${(100 * done) / tasks.length}%`;
}

/** Smallest box under the cursor (tiny boxes get a few pixels of slack). */
function hovered(): Face | undefined {
  const pad = 4 / scale;
  return faces
    .filter(({ box: [x, y, w, h] }) => mouse.x >= x - pad && mouse.x <= x + w + pad && mouse.y >= y - pad && mouse.y <= y + h + pad)
    .sort((a, b) => a.box[2] * a.box[3] - b.box[2] * b.box[3])[0];
}

async function save() {
  const t = tasks[index];
  const label = {
    clip: t.clip,
    frame: t.frame,
    kind: t.kind,
    faces: faces.map((f) => ({ ...f, box: f.box.map((v) => Math.round(v * 10) / 10) as Box })),
    candidates: t.candidates.length,
    reviewMs: Math.round(performance.now() - shownAt),
  };
  await fetch('/api/labels', { method: 'POST', body: JSON.stringify(label) });
  saved[key(t)] = label as Saved;
}

const toImage = (e: MouseEvent) => {
  const r = canvas.getBoundingClientRect();
  return { x: ((e.clientX - r.left) / r.width) * canvas.width / scale, y: ((e.clientY - r.top) / r.height) * canvas.height / scale };
};

canvas.addEventListener('mousemove', (e) => {
  mouse = toImage(e);
  draw();
});
canvas.addEventListener('mouseleave', () => {
  mouse = { x: -1, y: -1 };
  draw();
});
canvas.addEventListener('mousedown', (e) => {
  drag = toImage(e);
});
canvas.addEventListener('mouseup', (e) => {
  const end = toImage(e);
  const start = drag;
  drag = null;
  if (!start) return;
  const w = Math.abs(end.x - start.x);
  const h = Math.abs(end.y - start.y);
  if (w * scale > 4 && h * scale > 4) {
    faces.push({ box: [Math.min(start.x, end.x), Math.min(start.y, end.y), w, h], status: 'face', source: 'human' });
  } else {
    const f = hovered();
    if (f) {
      const next: Record<Status, Status> = { face: 'rejected', rejected: 'uncertain', uncertain: 'face' };
      if (f.source === 'human' && f.status === 'uncertain') faces.splice(faces.indexOf(f), 1);
      else f.status = f.source === 'human' && f.status === 'face' ? 'uncertain' : next[f.status];
    }
  }
  draw();
});

window.addEventListener('keydown', async (e) => {
  const k = e.key.toLowerCase();
  if (k === 'enter' || k === 'arrowright' || k === ' ') {
    e.preventDefault();
    await save();
    await show(index + 1);
  } else if (k === 'arrowleft') {
    await save();
    await show(index - 1);
  } else if (k === 'g') {
    await show(tasks.findIndex((t) => !saved[key(t)]));
  } else if (k === 'z') {
    zoom = !zoom;
  } else if (k === 'h') {
    hide = true;
  } else if (k === 'a') {
    for (const f of faces) f.status = 'face';
  } else if (k === 'x') {
    for (const f of faces) if (f.source !== 'human') f.status = 'rejected';
  } else if (k === 'u') {
    const f = hovered();
    if (f) f.status = 'uncertain';
  } else if (k === 'delete' || k === 'backspace') {
    const f = hovered();
    if (f?.source === 'human') faces.splice(faces.indexOf(f), 1);
    else if (f) f.status = 'rejected';
  }
  draw();
});
window.addEventListener('keyup', (e) => {
  if (e.key.toLowerCase() === 'h') {
    hide = false;
    draw();
  }
});
window.addEventListener('resize', () => show(index));

await show(index);
