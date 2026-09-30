import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGaze, gazeFeatures } from '../src/games/gaze.js';

// A face in the (not mirrored) camera picture of a player looking at screen point (x, y):
// the head turns and the pupils move towards it.
function frameAt({ x, y }, { head = 1, eyes = 1, noise = .002 } = {}) {
  const r = () => (Math.random() - .5) * noise;
  const hx = (x - .5) * .25 * head + r(), ix = (x - .5) * .12 * eyes + r(), hy = .45 + (y - .5) * .2 * head + r(), iy = (y - .5) * .1 * eyes + r();
  const p = Array.from({ length: 478 }, () => ({ x: .5, y: .5, z: 0 }));
  p[234] = { x: .4, y: .5 }; p[454] = { x: .6, y: .5 };                 // cheeks, 0.2 apart
  p[33] = { x: .44, y: .45 }; p[263] = { x: .56, y: .45 }; p[13] = { x: .5, y: .6 }; p[14] = { x: .5, y: .6 };
  p[1] = { x: .5 - hx * .2, y: .45 + hy * .15 };                         // the player's right is the picture's left
  for (const [outer, inner, top, bottom, iris, cx] of [[33, 133, 159, 145, 468, .47], [263, 362, 386, 374, 473, .53]]) {
    // Looking down, the upper lid follows the eye and the opening narrows.
    p[outer] = { x: cx - .03, y: .45 }; p[inner] = { x: cx + .03, y: .45 }; p[top] = { x: cx, y: .44 + (y - .5) * .006 * eyes }; p[bottom] = { x: cx, y: .46 };
    p[iris] = { x: cx - ix * .06, y: .45 + iy * .06 };
  }
  return { timestamp: 0, face: { landmarks: p, blendshapes: [] } };
}
const DOTS = [{ x: .5, y: .5 }, { x: .12, y: .5 }, { x: .88, y: .5 }, { x: .5, y: .16 }, { x: .5, y: .84 }];
function aimAt(gaze, target, how = {}) {
  let p;
  for (let i = 0; i < 60; i++) { const f = frameAt(target, { noise: 0, ...how }); f.timestamp = i * 33; p = gaze.point(f, f.timestamp, 1000, 1000); }
  return p;
}

test('features grow towards the screen right and down', () => {
  const left = gazeFeatures(frameAt({ x: .1, y: .5 }, { noise: 0 })), right = gazeFeatures(frameAt({ x: .9, y: .5 }, { noise: 0 }));
  const up = gazeFeatures(frameAt({ x: .5, y: .1 }, { noise: 0 })), down = gazeFeatures(frameAt({ x: .5, y: .9 }, { noise: 0 }));
  assert.ok(right.hx > left.hx && right.ix > left.ix && down.hy > up.hy && down.iy > up.iy);
});

test('calibration on five dots recovers where the player looks', () => {
  const gaze = createGaze();
  for (const dot of DOTS) for (let i = 0; i < 20; i++) gaze.addSample(gaze.features(frameAt(dot)), dot);
  assert.equal(gaze.solve(), true);
  const p = aimAt(gaze, { x: .8, y: .25 });
  assert.ok(Math.abs(p.x - .8) < .06 && Math.abs(p.y - .25) < .06, `looked at .8,.25, got ${p.x.toFixed(2)},${p.y.toFixed(2)}`);
});

test('never mirrored: a player who kept the head still still aims the right way when turning it later', () => {
  const gaze = createGaze();
  for (const dot of DOTS) for (let i = 0; i < 20; i++) gaze.addSample(gaze.features(frameAt(dot, { head: 0, noise: .02 })), dot);
  gaze.solve();
  const left = aimAt(gaze, { x: .15, y: .5 }), right = aimAt(gaze, { x: .85, y: .5 });
  assert.ok(right.x > left.x, `left ${left.x.toFixed(2)} right ${right.x.toFixed(2)}`);
});

test('without calibration the default aim is not mirrored either', () => {
  const gaze = createGaze();
  const left = aimAt(gaze, { x: .15, y: .5 }), right = aimAt(createGaze(), { x: .85, y: .5 });
  const up = aimAt(createGaze(), { x: .5, y: .15 }), down = aimAt(createGaze(), { x: .5, y: .85 });
  assert.ok(right.x > left.x && down.y > up.y);
});

test('eyes only: with the head still, the pupils and the lids alone aim across the screen', () => {
  const gaze = createGaze();
  const grid = [.12, .5, .88].flatMap(x => [.16, .5, .84].map(y => ({ x, y })));
  for (const dot of grid) for (let i = 0; i < 16; i++) gaze.addSample(gaze.features(frameAt(dot, { head: 0, eyes: .6, noise: .003 })), dot);
  assert.equal(gaze.solve(), true);
  for (const target of [{ x: .3, y: .3 }, { x: .7, y: .75 }]) {
    let p;
    for (let i = 0; i < 60; i++) { const f = frameAt(target, { head: 0, eyes: .6, noise: 0 }); f.timestamp = i * 33; p = gaze.point(f, f.timestamp, 1000, 1000); }
    assert.ok(Math.abs(p.x - target.x) < .08 && Math.abs(p.y - target.y) < .08, `looked at ${target.x},${target.y}, got ${p.x.toFixed(2)},${p.y.toFixed(2)}`);
  }
});

test('the calibration page model: a gliding dot trains the expanded aim, and it holds off-range', async () => {
  const { trainGaze } = await import('../src/games/gaze.js');
  const glide = t => ({ x: .5 + .42 * Math.sin(2 * Math.PI * t / 11), y: .5 + .37 * Math.sin(2 * Math.PI * t / 7.3 + Math.PI / 2) });
  const samples = [];
  for (let t = 0; t < 30; t += 1 / 30) { const at = glide(t); samples.push({ f: createGaze().features(frameAt(at, { head: .3, noise: .002 })), target: at }); }
  const model = trainGaze(samples);
  assert.ok(model && model.v === 2, 'expanded model fitted');
  const gaze = createGaze(); gaze.load(model);
  const p = aimAt(gaze, { x: .3, y: .7 }, { head: .3 });
  assert.ok(Math.abs(p.x - .3) < .06 && Math.abs(p.y - .7) < .06, `looked at .3,.7, got ${p.x.toFixed(2)},${p.y.toFixed(2)}`);
  // A head turned much further than during calibration does not throw the aim across the screen.
  const g2 = createGaze(); g2.load(model);
  let q; for (let i = 0; i < 60; i++) { const f = frameAt({ x: .5, y: .5 }, { head: 4, noise: 0 }); q = g2.point(f, i * 33, 1000, 1000); }
  assert.ok(q.x > .3 && q.x < .7 && q.y > .3 && q.y < .7, `centre with a big head turn: ${q.x.toFixed(2)},${q.y.toFixed(2)}`);
});

test('recalibration during play: labels with an offset and a squashed scale are undone', async () => {
  const { createRecalibrator } = await import('../src/games/gaze.js');
  const rc = createRecalibrator();
  // The aim sits .3 too high and moves only 0.6 as far as the eyes: aim = −.3 + .6·true (+ noise).
  for (let k = 0; k < 40; k++) {
    const pairs = Array.from({ length: 20 }, () => { const t = { x: Math.random(), y: Math.random() }; return { target: t, aim: { x: t.x + (Math.random() - .5) * .08, y: -.3 + .6 * t.y + (Math.random() - .5) * .08 } }; });
    rc.add(pairs, k);
  }
  for (const t of [{ x: .2, y: .2 }, { x: .8, y: .8 }]) {
    const c = rc.apply({ x: t.x, y: -.3 + .6 * t.y });
    assert.ok(Math.abs(c.x - t.x) < .04 && Math.abs(c.y - t.y) < .05, `true ${t.x},${t.y} → ${c.x.toFixed(2)},${c.y.toFixed(2)}`);
  }
});
