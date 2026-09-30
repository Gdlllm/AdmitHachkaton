// Review a recorded game: how far the aim was from the rock being burned, over time,
// with and without the in-game shift. node ml/gaze/play_review.mjs [gameId]
import { readFileSync, readdirSync } from 'node:fs';
import { createGaze } from '../../src/games/gaze.js';
const all = readdirSync('ml/gaze/data').filter(f => f.startsWith('play-'));
const id = process.argv[2] ?? all.map(f => f.slice(5, -9)).sort().at(-1);
const parts = all.filter(f => f.includes(id)).sort().map(f => JSON.parse(readFileSync(`ml/gaze/data/${f}`, 'utf8')));
const frames = parts.flatMap(p => p.frames), { w: W, h: H } = parts[0].screen;
globalThis.innerWidth = W; globalThis.innerHeight = H;
const g = createGaze(); g.load(parts[0].model);
const bias = parts[0].bias ?? { x: 0, y: 0 };
const med = a => a.length ? [...a].sort((p, q) => p - q)[Math.floor(a.length / 2)] : NaN;
console.log(`game ${id}: ${frames.length} frames, screen ${W}x${H}, start bias ${JSON.stringify(bias)}`);
const t0 = frames[0].t;
const windows = new Map();
for (const r of frames) {
  if (!r.target || r.target.heat < .3 || !r.point) continue;
  const raw = g.aimOf(r.f);                       // the model alone (no centring, no in-game shift)
  const k = Math.floor((r.t - t0) / 20000);
  if (!windows.has(k)) windows.set(k, []);
  windows.get(k).push({ rawDx: (raw.x + bias.x - r.target.x) * W, rawDy: (raw.y + bias.y - r.target.y) * H, shownDx: (r.point.x - r.target.x) * W, shownDy: (r.point.y - r.target.y) * H, hy: r.f.hy, hx: r.f.hx, ty: r.target.y, tx: r.target.x });
}
console.log('window   n   model: dx   dy   | shown (with shift): dx   dy  | head hx   hy');
for (const [k, v] of [...windows].sort((a, b) => a[0] - b[0])) {
  console.log(`${String(k * 20).padStart(4)}s ${String(v.length).padStart(4)}   ${String(Math.round(med(v.map(e => e.rawDx)))).padStart(5)} ${String(Math.round(med(v.map(e => e.rawDy)))).padStart(5)}   | ${String(Math.round(med(v.map(e => e.shownDx)))).padStart(8)} ${String(Math.round(med(v.map(e => e.shownDy)))).padStart(5)}  | ${med(v.map(e => e.hx)).toFixed(3)} ${med(v.map(e => e.hy)).toFixed(3)}`);
}
// Where on screen the model misses (all locked frames), by rock position.
const locked = [...windows.values()].flat();
for (const [label, sel] of [['rock left third', e => e.tx < .33], ['rock middle', e => e.tx >= .33 && e.tx < .66], ['rock right third', e => e.tx >= .66], ['rock top third', e => e.ty < .33], ['rock middle row', e => e.ty >= .33 && e.ty < .66], ['rock bottom third', e => e.ty >= .66]]) {
  const v = locked.filter(sel);
  console.log(`${label.padEnd(18)} n ${String(v.length).padStart(4)}  model dx ${String(Math.round(med(v.map(e => e.rawDx)))).padStart(5)} dy ${String(Math.round(med(v.map(e => e.rawDy)))).padStart(5)}`);
}

// Replay the in-game correction causally on this game: the aim shown with each adapter,
// against the rock being burned. (Only frames that were locked in the real game.)
const { createAdapter } = await import('../../src/games/gaze.js');
for (const [label, make] of [['shift 60 + zones', () => createAdapter({ keep: 60 })], ['shift 40 + zones', () => createAdapter({ keep: 40 })], ['shift 30 + zones', () => createAdapter({ keep: 30 })], ['shift 40, no zones', () => createAdapter({ keep: 40, zones: false })]]) {
  const ad = make(), errs = [];
  for (const r of frames) {
    if (!r.target || r.target.heat < .3 || !r.f) continue;
    const a = g.aimOf(r.f), centred = { x: a.x + bias.x, y: a.y + bias.y }, shown = ad.apply(centred);
    errs.push([(shown.x - r.target.x) * W, (shown.y - r.target.y) * H]);
    ad.learn(centred, r.f, r.target);
  }
  const e = errs.slice(200);
  console.log(`${label.padEnd(26)} |dx| ${Math.round(med(e.map(v => Math.abs(v[0]))))} |dy| ${Math.round(med(e.map(v => Math.abs(v[1]))))} px, within 80 px: ${Math.round(e.filter(v => Math.hypot(v[0], v[1]) < 80).length / e.length * 100)} %`);
}
