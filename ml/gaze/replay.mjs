// Replays a recorded calibration through the game's own gaze code and reports the aim
// error on the check dots (never used for fitting): node ml/gaze/replay.mjs [session.json]
import { readFileSync, readdirSync } from 'node:fs';
import { trainGaze, fitGaze, createGaze } from '../../src/games/gaze.js';
globalThis.innerWidth ??= 1000; globalThis.innerHeight ??= 1000;
const file = process.argv[2] ?? `ml/gaze/data/${readdirSync('ml/gaze/data').filter(f => f.endsWith('.json')).sort().at(-1)}`;
const d = JSON.parse(readFileSync(file, 'utf8')), { w: W, h: H } = d.screen;
const glide = t => ({ x: .5 + .42 * Math.sin(2 * Math.PI * t / 11), y: .5 + .37 * Math.sin(2 * Math.PI * t / 7.3 + Math.PI / 2) });
// The glide's clock: recover its start from the recorded dot positions.
const pur = d.frames.filter(r => r.phase === 'pursuit');
let t0 = pur[0].t, best = Infinity;
for (let o = -3000; o < 3000; o += 5) {
  let e = 0; for (const r of pur.filter((_, i) => i % 5 === 0)) { const g = glide((r.t - pur[0].t - o) / 1000); e += (g.x - r.target.x) ** 2 + (g.y - r.target.y) ** 2; }
  if (e < best) { best = e; t0 = pur[0].t + o; }
}
function samples({ lag = .2, dots = true, glideOn = true } = {}) {
  const out = [];
  for (const r of d.frames) {
    if (!r.f || !r.target?.settled) continue;
    if (dots && r.phase === 'fix') out.push({ f: r.f, target: r.target, group: `fix${r.index}` });
    if (glideOn && r.phase === 'pursuit') out.push({ f: r.f, target: glide((r.t - t0) / 1000 - lag), group: `pur${Math.floor(r.t / 400)}` });
  }
  return out;
}
function check(model) {
  const byDot = new Map();
  for (const r of d.frames) if (r.phase === 'check' && r.target?.settled && r.f) { if (!byDot.has(r.index)) byDot.set(r.index, []); byDot.get(r.index).push(r); }
  const errs = [];
  for (const rs of byDot.values()) {
    const g = createGaze(); g.load(model);
    const aims = rs.map(r => g.aimOf(r.f));
    const med = a => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
    const ax = med(aims.map(a => a.x)), ay = med(aims.map(a => a.y));
    errs.push({ x: Math.abs(ax - rs[0].target.x) * W, y: Math.abs(ay - rs[0].target.y) * H });
  }
  const med = a => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
  return { px: Math.round(med(errs.map(e => Math.hypot(e.x, e.y)))), x: Math.round(med(errs.map(e => e.x))), y: Math.round(med(errs.map(e => e.y))), meanPx: Math.round(errs.reduce((a, e) => a + Math.hypot(e.x, e.y), 0) / errs.length) };
}
console.log(file, `screen ${W}x${H}`);
console.log('page model then (recorded):', d.error);
const old = fitGaze(samples({ lag: .12 }));
if (process.argv.includes('--save')) { const m = trainGaze(samples()); console.log(JSON.stringify(m)); process.exit(0); }
console.log('old plain fit (as the page did):', check(old));
for (const [label, opts] of [['new model, dots + glide', {}], ['new model, glide only (like the game)', { dots: false }], ['new model, dots only', { glideOn: false }]]) {
  const m = trainGaze(samples(opts));
  console.log(`${label}:`, m ? check(m) : 'no fit (falls back to the plain one)');
}
