import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRacketArm } from '../src/games/controls.js';

// A pose frame with shoulders in the middle and the right wrist at (wx, wy) in the camera picture.
const frame = (t, wx, wy, nose = .3) => {
  const lm = Array.from({ length: 33 }, () => ({ x: .5, y: .5, visibility: .9 }));
  lm[0] = { x: .5, y: nose, visibility: .9 };
  lm[11] = { x: .6, y: .45, visibility: .9 }; lm[12] = { x: .4, y: .45, visibility: .9 };
  lm[14] = { x: .35, y: .6, visibility: .9 }; lm[16] = { x: wx, y: wy, visibility: .9 };
  return { timestamp: t, pose: { landmarks: lm } };
};

test('a fast sweep of the wrist is one swing, reported at its fastest frame', () => {
  const arm = createRacketArm({ hand: 'right' }), swings = [];
  // Hold still, then sweep across the picture in ~0.3 s, then hold again (30 fps).
  const xs = [...Array(10).fill(.25), .27, .32, .4, .5, .6, .68, .72, .74, ...Array(10).fill(.74)];
  xs.forEach((x, i) => { const r = arm.update(frame(1000 + i * 33, x, .6), 16 / 9); if (r.swing) swings.push(r.swing); });
  assert.equal(swings.length, 1);
  const s = swings[0];
  assert.ok(s.t >= 1000 + 12 * 33 && s.t <= 1000 + 15 * 33, `peak at frame ${(s.t - 1000) / 33}`);
  assert.ok(s.vx < 0, 'the camera picture is mirrored: moving to the image right is the player\'s left');
});

test('slow movements are not swings; a raised hand is seen', () => {
  const arm = createRacketArm({ hand: 'right' });
  let swings = 0, last;
  for (let i = 0; i < 40; i++) { last = arm.update(frame(i * 33, .3 + i * .004, .6 - i * .01), 16 / 9); if (last.swing) swings++; }
  assert.equal(swings, 0);
  assert.equal(last.raisedHand, 'right');
  assert.equal(last.raised, true);
});

test('pause gesture: both wrists above the face, never from shoulders guessed below the frame', async () => {
  const { handsAboveShoulders } = await import('../src/games/controls.js');
  const pose = (l, r, shoulderY = .7, nose = .4) => {
    const lm = Array.from({ length: 33 }, () => ({ x: .5, y: .5, visibility: .9 }));
    lm[0] = { x: .5, y: nose, visibility: .9 };
    lm[11] = { x: .65, y: shoulderY, visibility: .9 }; lm[12] = { x: .35, y: shoulderY, visibility: .9 };
    lm[15] = { x: .75, y: l, visibility: .9 }; lm[16] = { x: .25, y: r, visibility: .9 };
    return { pose: { landmarks: lm } };
  };
  assert.equal(handsAboveShoulders(pose(.2, .6)), false, 'one hand up');
  assert.equal(handsAboveShoulders(pose(.2, .25)), true, 'both hands up');
  // At a laptop: shoulders guessed below the picture, hands in the middle holding the bat.
  assert.equal(handsAboveShoulders(pose(.55, .6, 1.15, .45)), false, 'hands at chest height, shoulders out of frame');
  assert.equal(handsAboveShoulders(pose(.3, .32, 1.15, .45)), true, 'both hands above the face, shoulders out of frame');
});

test('1€ filter: a shaking still hand calms down, a fast move goes through', async () => {
  const { createOneEuro } = await import('../src/games/controls.js');
  const f = createOneEuro();
  let out = [];
  for (let i = 0; i < 60; i++) out.push(f.filter(.5 + (i % 2 ? .02 : -.02), 1 / 30));
  const shake = Math.max(...out.slice(30)) - Math.min(...out.slice(30));
  assert.ok(shake < .012, `still hand shakes ${shake.toFixed(3)} instead of .04`);
  let v;
  for (let i = 0; i < 6; i++) v = f.filter(.5 + (i + 1) * .08, 1 / 30);
  assert.ok(v > .5 + 6 * .08 - .1, `fast move lags: ${v.toFixed(2)} vs ${(.5 + 6 * .08).toFixed(2)}`);
});
