import test from 'node:test';
import assert from 'node:assert/strict';
import { fitCameraTranslation } from '../src/dense/inference/fit-camera.js';

const rig = Array.from({ length: 18 }, (_, i) => ({ x: ((i * 7) % 11 - 5) * .13, y: (i - 8) * .1, z: ((i * 3) % 7 - 3) * .08 }));
const camera = [.13, -.09, 3.2], focal = 800, principalPoint = [320, 240];
const project = (points, t) => points.map(p => ({ x: focal * (p.x + t[0]) / (p.z + t[2]) + 320, y: focal * (p.y + t[1]) / (p.z + t[2]) + 240 }));
const options = { joints3d: rig, joints2d: project(rig, camera), initialTranslation: [.02, .13, 4.1], focal, principalPoint, indices: rig.map((_, i) => i) };

test('recovers known perspective translation, keeps geometry and input camera intact', () => {
  const snapshot = structuredClone(options), result = fitCameraTranslation(options);
  assert.equal(result.accepted, true);
  result.translation.forEach((v, i) => assert.ok(Math.abs(v - camera[i]) < 1e-8));
  assert.ok(result.after.meanPixels < 1e-8);
  assert.deepEqual(options, snapshot);
});

test('Huber fit resists two severe 2D outliers among distributed body anchors', () => {
  const targets = project(rig, camera);
  targets[2].x += 180; targets[2].y -= 130;
  targets[12].x -= 140; targets[12].y += 160;
  const result = fitCameraTranslation({ ...options, joints2d: targets, huberPixels: 4 });
  const projected = project(rig, result.translation), clean = project(rig, camera);
  const cleanErrors = projected.map((p, i) => Math.hypot(p.x - clean[i].x, p.y - clean[i].y)).filter((_, i) => ![2, 12].includes(i));
  assert.ok(cleanErrors.reduce((a, b) => a + b) / cleanErrors.length < 1.5);
  assert.ok(result.after.robustObjective < result.before.robustObjective);
});

test('insufficient, degenerate and off-screen correspondences retain original camera', () => {
  for (const extra of [{ indices: [0, 1, 2] }, { joints3d: rig.map(() => ({ x: 0, y: 0, z: 0 })), joints2d: rig.map(() => ({ x: 320, y: 240 })) }, { joints3d: rig.map(() => ({ x: 0, y: 0, z: 0 })) }, { imageSize: [1, 1] }]) {
    const result = fitCameraTranslation({ ...options, ...extra });
    assert.equal(result.accepted, false); assert.deepEqual(result.translation, options.initialTranslation);
  }
});

test('depth and lateral constraints prevent an arbitrary crop-scale correction', () => {
  const targets = project(rig, [2, 1, .9]);
  const result = fitCameraTranslation({ ...options, joints2d: targets, depthRatio: [.9, 1.1], maxPixelShift: 10 });
  assert.ok(result.translation[2] >= 4.1 * .9 && result.translation[2] <= 4.1 * 1.1);
  for (const axis of [0, 1]) assert.ok(Math.abs(result.translation[axis] - options.initialTranslation[axis]) <= 4.1 * 10 / focal + 1e-9);
  assert.ok(result.constraints.atBoundary.some(Boolean));
});

test('exact original camera remains unchanged and invalid or behind-camera inputs fail safely', () => {
  const result = fitCameraTranslation({ ...options, initialTranslation: camera });
  assert.equal(result.accepted, false); assert.equal(result.reason, 'no-improvement');
  assert.deepEqual(result.translation, camera);
  const hidden = fitCameraTranslation({ ...options, joints3d: rig.map(p => ({ ...p, z: -20 })) });
  assert.equal(hidden.reason, 'invalid-initial-depth');
  assert.throws(() => fitCameraTranslation({ ...options, focal: 0 }), /focal/);
  assert.throws(() => fitCameraTranslation({ ...options, initialTranslation: [0, 0, -1] }), /positive/);
});
