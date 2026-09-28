import test from 'node:test';
import assert from 'node:assert/strict';
import { CaptureStabilizer, landmarkConfidence } from '../src/tracking/stabilizer.js';

const points = (count, x = 0.45, y = 0.45, pose = false) => Array.from({ length: count }, (_, i) => ({
  x: x + (i % 5) * 0.008, y: y + Math.floor(i / 5) * 0.001, z: i * 0.0001,
  ...(pose ? { visibility: 0.95, presence: 0.98 } : {}),
}));
const hand = (x, handedness = 'Left') => ({ landmarks: points(21, x), handedness, score: 0.99 });
const frame = (timestamp, overrides = {}) => ({ timestamp, inferenceMs: 12, pose: { landmarks: points(33, 0.45, 0.45, true) }, face: { landmarks: points(478) }, hands: [], ...overrides });
const meanSquare = values => values.reduce((sum, value) => sum + value * value, 0) / values.length;

test('keeps 33/478/21 landmarks, optional data, source coordinates and input untouched', () => {
  const stabilizer = new CaptureStabilizer();
  const source = frame(0, { hands: [hand(0.3)] });
  source.pose.worldLandmarks = points(33).map(p => ({ ...p, x: p.x * 4 - 2, y: p.y * 4 - 2 }));
  source.face.blendshapes = [{ categoryName: 'eyeBlinkLeft', score: 0.2 }];
  const before = structuredClone(source);
  const output = stabilizer.update(source);
  assert.deepEqual(source, before);
  assert.equal(output.pose.landmarks.length, 33);
  assert.equal(output.face.landmarks.length, 478);
  assert.equal(output.hands[0].landmarks.length, 21);
  assert.equal(output.pose.worldLandmarks[0].x, source.pose.worldLandmarks[0].x);
  assert.ok(output.pose.worldLandmarks[0].drawConfidence > 0);
  assert.deepEqual(output.face.blendshapes, source.face.blendshapes);
  assert.notEqual(output.face.landmarks[0], source.face.landmarks[0]);
});

test('pose confidence gates immediately; unpopulated face/hand confidence does not hide them', () => {
  const stabilizer = new CaptureStabilizer();
  stabilizer.update(frame(0));
  const next = frame(33, { hands: [hand(0.3)] });
  Object.assign(next.pose.landmarks[15], { visibility: 0.1 });
  Object.assign(next.pose.landmarks[16], { x: 1.1 });
  Object.assign(next.pose.landmarks[25], { y: NaN });
  Object.assign(next.face.landmarks[0], { visibility: 0, presence: 0 });
  Object.assign(next.hands[0].landmarks[0], { visibility: 0, presence: 0 });
  next.hands[0].score = 0.1; // handedness score, not geometry confidence
  const output = stabilizer.update(next);
  for (const i of [15, 16, 25]) assert.equal(output.pose.landmarks[i].drawConfidence, 0);
  assert.equal(output.face.landmarks[0].drawConfidence, 1);
  assert.equal(output.hands[0].landmarks[0].drawConfidence, 1);
  assert.equal(landmarkConfidence({ x: 0.4, y: 0.5, presence: 0.1 }), 0);
});

test('explicit loss clears geometry immediately and reacquisition never blends stale positions', () => {
  const stabilizer = new CaptureStabilizer();
  const first = stabilizer.update(frame(0, { hands: [hand(0.25)] }));
  const lost = stabilizer.update(frame(33, { pose: null, face: { landmarks: [] }, hands: [] }));
  assert.equal(lost.pose, null);
  assert.equal(lost.face, null);
  assert.deepEqual(lost.hands, []);
  const reacquired = frame(66, { hands: [hand(0.35)] });
  reacquired.pose.landmarks = points(33, 0.65, 0.45, true);
  reacquired.face.landmarks = points(478, 0.3);
  const output = stabilizer.update(reacquired);
  assert.equal(output.pose.landmarks[0].x, 0.65);
  assert.equal(output.face.landmarks[0].x, 0.3);
  assert.equal(output.hands[0].landmarks[0].x, 0.35);
  assert.equal(output.hands[0].track.id, first.hands[0].track.id);
  assert.equal(output.hands[0].track.reacquired, true);
});

test('invalid points reset locally and restore instantly without dragging the mesh', () => {
  const stabilizer = new CaptureStabilizer();
  stabilizer.update(frame(0));
  const hidden = frame(33);
  hidden.pose.landmarks[15].visibility = 0.1;
  hidden.face.landmarks[0] = null;
  assert.equal(stabilizer.update(hidden).face.landmarks[0].drawConfidence, 0);
  const restored = frame(66);
  restored.pose.landmarks[15].x = 0.7;
  const output = stabilizer.update(restored);
  assert.equal(output.pose.landmarks[15].x, 0.7);
  assert.ok(output.pose.landmarks[15].drawConfidence > 0);
});

test('display smoothing stays within source-pixel budgets even at a low capture rate', () => {
  const stabilizer = new CaptureStabilizer();
  const first = frame(0, { sourceWidth: 1920, sourceHeight: 1080, hands: [hand(0.2)] });
  first.pose.worldLandmarks = first.pose.landmarks.map(p => ({ x: p.x * 2, y: p.y * 2, z: p.z * 2 }));
  stabilizer.update(first);
  const current = frame(125, { sourceWidth: 1920, sourceHeight: 1080, hands: [hand(0.4)] });
  current.face.landmarks = points(478, 0.62);
  current.pose.landmarks = points(33, 0.6, 0.6, true);
  current.pose.worldLandmarks = current.pose.landmarks.map(p => ({ x: p.x * 2, y: p.y * 2, z: p.z * 2 }));
  const output = stabilizer.update(current);
  for (const [raw, filtered, budget] of [[current.pose, output.pose, 4], [current.face, output.face, 2], [current.hands[0], output.hands[0], 3]]) {
    for (let i = 0; i < raw.landmarks.length; i++) {
      const residual = Math.hypot((raw.landmarks[i].x - filtered.landmarks[i].x) * 1920, (raw.landmarks[i].y - filtered.landmarks[i].y) * 1080);
      assert.ok(residual <= budget + 1e-9, `${residual}px exceeds ${budget}px`);
    }
  }
  assert.ok(output.tracking.smoothing.maxResidualPx <= 4 + 1e-9);
  assert.ok(Math.abs(output.pose.worldLandmarks[0].x - output.pose.landmarks[0].x * 2) < 1e-9);
  assert.throws(() => stabilizer.update(frame(150, { sourceWidth: -1 })), /dimensions/);
});

test('isolated face teleports are hidden, while coherent rapid translation remains visible', () => {
  const stabilizer = new CaptureStabilizer();
  stabilizer.update(frame(0));
  const bad = frame(33);
  bad.face.landmarks[100].x = 0.9;
  const rejected = stabilizer.update(bad);
  assert.equal(rejected.face.landmarks[100].drawConfidence, 0);
  assert.equal(rejected.tracking.rejectedPoints, 1);
  const moved = frame(66);
  moved.face.landmarks = points(478, 0.65);
  const accepted = stabilizer.update(moved);
  assert.ok(accepted.face.landmarks.every(p => p.drawConfidence > 0));
  assert.equal(accepted.face.landmarks[100].x, moved.face.landmarks[100].x);
});

test('stationary jitter is reduced while a coherent fast hand movement catches up promptly', () => {
  const stabilizer = new CaptureStabilizer();
  const rawErrors = [], filteredErrors = [];
  for (let i = 0; i < 90; i++) {
    const jitter = Math.sin(i * 2.2) * 0.002;
    const output = stabilizer.update(frame(i * 16.667, { hands: [hand(0.3 + jitter)] }));
    if (i > 10) { rawErrors.push(jitter); filteredErrors.push(output.hands[0].landmarks[0].x - 0.3); }
  }
  assert.ok(meanSquare(filteredErrors) < meanSquare(rawErrors) * 0.35);
  const first = stabilizer.update(frame(1500, { hands: [hand(0.45)] }));
  assert.ok(first.hands[0].landmarks[0].x > 0.4, 'fast movement should not retain the stationary filter lag');
  const second = stabilizer.update(frame(1533, { hands: [hand(0.45)] }));
  assert.ok(Math.abs(second.hands[0].landmarks[0].x - 0.45) < 0.012);
});

test('hand identity survives reversed detector order and erroneous handedness flips', () => {
  const stabilizer = new CaptureStabilizer();
  const first = stabilizer.update(frame(0, { hands: [hand(0.2, 'Left'), hand(0.7, 'Right')] }));
  const second = stabilizer.update(frame(33, { hands: [hand(0.71, 'Left'), hand(0.21, 'Right')] }));
  assert.equal(second.hands[0].track.id, first.hands[1].track.id);
  assert.equal(second.hands[1].track.id, first.hands[0].track.id);
  assert.equal(second.hands[0].track.handedness, 'right');
  assert.ok(second.hands[0].landmarks[0].x > 0.7);
  assert.ok(second.hands[1].landmarks[0].x < 0.22);
});

test('opposing hand trajectories keep identity through a crossing', () => {
  const stabilizer = new CaptureStabilizer();
  const initial = stabilizer.update(frame(0, { hands: [hand(0.3, 'Left'), hand(0.7, 'Right')] }));
  let output;
  for (let i = 1; i <= 5; i++) {
    const hands = [hand(0.3 + i * 0.05, 'Left'), hand(0.7 - i * 0.05, 'Right')];
    if (i % 2) hands.reverse();
    output = stabilizer.update(frame(i * 33, { hands }));
    assert.equal(output.hands.find(h => h.handedness === 'Left').track.id, initial.hands[0].track.id);
    assert.equal(output.hands.find(h => h.handedness === 'Right').track.id, initial.hands[1].track.id);
  }
});

test('long frame gaps reset interpolation; duplicate/regressed clocks reset safely', () => {
  const stabilizer = new CaptureStabilizer();
  stabilizer.update(frame(100));
  const distant = frame(600);
  distant.face.landmarks = points(478, 0.6);
  assert.equal(stabilizer.update(distant).face.landmarks[0].x, 0.6);
  const duplicate = frame(600);
  duplicate.face.landmarks = points(478, 0.3);
  const output = stabilizer.update(duplicate);
  assert.equal(output.tracking.clockReset, true);
  assert.equal(output.face.landmarks[0].x, 0.3);
  assert.equal(stabilizer.update(frame(50)).tracking.clockReset, true);
  assert.throws(() => stabilizer.update(frame(NaN)), /timestamp/);
});

test('expired hand identity is not attached to a new distant detection', () => {
  const stabilizer = new CaptureStabilizer();
  const first = stabilizer.update(frame(0, { hands: [hand(0.2)] }));
  stabilizer.update(frame(33));
  const newHand = stabilizer.update(frame(1000, { hands: [hand(0.7)] }));
  assert.notEqual(first.hands[0].track.id, newHand.hands[0].track.id);
  assert.equal(newHand.hands[0].landmarks[0].x, 0.7);
});
