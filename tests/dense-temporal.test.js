import test from 'node:test';
import assert from 'node:assert/strict';
import { SurfaceActivation, OneEuroVector } from '../src/dense/inference/temporal.js';

const FULL = { enough: true, tracked: true }, TORSO = { enough: false, tracked: true }, NONE = { enough: false, tracked: false };

test('enough of a body must persist for the arming time before the surface switches on', () => {
  const a = new SurfaceActivation({ armMs: 150 });
  assert.deepEqual(a.update(FULL, 0), { state: 'arming', active: false, changed: null });
  assert.equal(a.update(FULL, 100).active, false);
  assert.deepEqual(a.update(FULL, 150), { state: 'active', active: true, changed: 'on' });
  assert.equal(a.update(FULL, 180).changed, null, 'on is reported once');
});

test('one incomplete frame during arming restarts the count', () => {
  const a = new SurfaceActivation({ armMs: 150 });
  a.update(FULL, 0); a.update(FULL, 120);
  assert.equal(a.update(TORSO, 130).state, 'idle');
  a.update(FULL, 140);
  assert.equal(a.update(FULL, 250).active, false);
  assert.equal(a.update(FULL, 290).changed, 'on');
});

test('short occlusions and shoulders-only frames do not blink an active surface', () => {
  const a = new SurfaceActivation({ armMs: 0, holdMs: 450, graceMs: 1200 });
  a.update(FULL, 0);
  for (let t = 33; t < 400; t += 33) assert.equal(a.update(NONE, t).active, true, `lost for ${t} ms`);
  for (let t = 400; t < 1150; t += 33) assert.equal(a.update(TORSO, t).active, true, `shoulders only at ${t} ms`);
  assert.equal(a.update(FULL, 1180).active, true);
});

test('the surface switches off once the person is gone or too little is visible for too long', () => {
  const gone = new SurfaceActivation({ armMs: 0, holdMs: 450 });
  gone.update(FULL, 0); gone.update(NONE, 300);
  assert.deepEqual(gone.update(NONE, 460), { state: 'idle', active: false, changed: 'off' });
  assert.equal(gone.update(NONE, 500).changed, null);
  const close = new SurfaceActivation({ armMs: 0, holdMs: 450, graceMs: 1200 });
  close.update(FULL, 0);
  for (let t = 33; t <= 1200; t += 33) close.update(TORSO, t);
  assert.equal(close.update(TORSO, 1234).changed, 'off', 'head and shoulders alone do not keep the surface');
});

test('a clock that goes backwards starts over and reports the loss', () => {
  const a = new SurfaceActivation({ armMs: 0 });
  a.update(FULL, 1000);
  assert.equal(a.update(FULL, 10).changed, 'off');
  assert.equal(a.update(FULL, 20).changed, 'on');
  assert.throws(() => a.update(FULL, NaN), TypeError);
});

test('1 Euro vector passes the first sample, smooths jitter at rest and follows fast motion', () => {
  const f = new OneEuroVector(2, { minCutoff: 1, beta: 2 });
  assert.deepEqual(Array.from(f.update([1, -1], 0)), [1, -1]);
  let value;
  for (let i = 1; i <= 30; i++) value = f.update([1 + (i % 2 ? 0.02 : -0.02), -1], i / 30);
  assert.ok(Math.abs(value[0] - 1) < 0.012, 'jitter is attenuated');
  const moving = new OneEuroVector(1, { minCutoff: 1, beta: 2 });
  for (let i = 0; i <= 15; i++) value = moving.update([i * 0.2], i / 30);
  assert.ok(value[0] > 2.4, `fast motion is followed closely (${value[0]})`);
  assert.throws(() => moving.update([1, 2], 1), TypeError);
});

test('1 Euro vector restarts on reset or a non-increasing time', () => {
  const f = new OneEuroVector(1, { minCutoff: 0.3 });
  f.update([0], 0); f.update([10], 0.1);
  assert.deepEqual(Array.from(f.update([5], 0.1)), [5]);
  f.reset();
  assert.deepEqual(Array.from(f.update([7], 5)), [7]);
});
