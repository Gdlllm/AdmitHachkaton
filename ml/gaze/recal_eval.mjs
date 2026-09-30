// Replays the in-play recalibration on a recorded game: each rock burned towards the end
// becomes labels; the aim error is measured on later locked frames (never seen yet).
// node ml/gaze/recal_eval.mjs <gameId>
import { readFileSync, readdirSync } from 'node:fs';
import { createGaze, createRecalibrator } from '../../src/games/gaze.js';
const id = process.argv[2], dir = 'ml/gaze/data';
const parts = readdirSync(dir).filter(f => f.startsWith('play-') && f.includes(id)).sort().map(f => JSON.parse(readFileSync(`${dir}/${f}`, 'utf8')));
if (!parts[0].model) { console.log('part 0 (with the model) missing'); process.exit(1); }
const frames = parts.flatMap(p => p.frames), { w: W, h: H } = parts[0].screen, bias = parts[0].bias ?? { x: 0, y: 0 };
globalThis.innerWidth = W; globalThis.innerHeight = H;
const g = createGaze(); g.load(parts[0].model);
// Runs of locking one rock: the target barely moves between frames; a run that got hot is a kill.
const runs = []; let cur = null;
for (const r of frames) {
  if (!r.target || !r.f) { cur = null; continue; }
  const a = g.aimOf(r.f), aim = { x: a.x + bias.x, y: a.y + bias.y };
  if (!cur || Math.hypot(r.target.x - cur.last.x, r.target.y - cur.last.y) > .03) { cur = { pairs: [], maxHeat: 0, t: r.t }; runs.push(cur); }
  cur.pairs.push({ aim, target: { x: r.target.x, y: r.target.y }, t: r.t }); cur.last = r.target; cur.maxHeat = Math.max(cur.maxHeat, r.target.heat ?? 0);
}
const kills = runs.filter(r => r.maxHeat > .85 && r.pairs.length > 5);
console.log(`${frames.length} frames, ${runs.length} lock runs, ${kills.length} burned down`);
const med = a => [...a].sort((p, q) => p - q)[Math.floor(a.length / 2)];
for (const opts of [{}, { halfLife: 45 }, { halfLife: 25 }, { halfLife: 15 }, { halfLife: 25, prior: 20 }, { halfLife: 25, maxB: 1.6 }]) {
  const rc = createRecalibrator(opts), before = [], after = [];
  for (const run of kills) {
    // Per rock: the median aim over its burn (what the smoothed reticle shows), against the rock.
    const mx = med(run.pairs.map(p => p.aim.x)), my = med(run.pairs.map(p => p.aim.y)), tx = med(run.pairs.map(p => p.target.x)), ty = med(run.pairs.map(p => p.target.y));
    const c = rc.apply({ x: mx, y: my });
    before.push(Math.hypot((mx - tx) * W, (my - ty) * H)); after.push(Math.hypot((c.x - tx) * W, (c.y - ty) * H));
    rc.add(run.pairs, run.t / 1000);
  }
  const tail = a => a.slice(Math.floor(a.length / 4));
  console.log(`${JSON.stringify(opts).padEnd(18)} error px (after the first quarter): without ${Math.round(med(tail(before)))}  with ${Math.round(med(tail(after)))}   fix x ${rc.fix.x.map(v => v.toFixed(2))} y ${rc.fix.y.map(v => v.toFixed(2))}`);
}
