import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { reconcileDetections } from '../src/tracking/fusion.js';

const body = () => ({ landmarks: Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.99 })) });
const hand = (minX, maxX, minY, maxY) => ({ landmarks: Array.from({ length: 21 }, (_, i) => ({ x: minX + (maxX - minX) * (i % 5) / 4, y: minY + (maxY - minY) * Math.floor(i / 5) / 4, z: 0 })) });

test('regression: reject confidently hallucinated torso wholly inside a close-up hand', () => {
  // Measured Pose Heavy torso + Hand Landmarker bounds, TFJS hand-signs.mp4,
  // sampled frame index 2. No ground-truth accuracy is inferred from confidence.
  const pose = body();
  [[11, 0.612893, 0.466358], [12, 0.594418, 0.485929], [23, 0.509680, 0.590122], [24, 0.506669, 0.594774]]
    .forEach(([i, x, y]) => Object.assign(pose.landmarks[i], { x, y, visibility: 0.9995 }));
  const source = { timestamp: 100, pose, face: null, hands: [hand(0.416487, 0.689823, 0.405546, 0.613477)] };
  const output = reconcileDetections(source);
  assert.equal(output.pose, null);
  assert.equal(output.fusion.reason, 'torso-contained-in-hand');
  assert.equal(output.hands, source.hands);
  assert.equal(source.pose, pose, 'never mutate the raw frame');
});

test('a real-sized torso remains when hands are in front of it', () => {
  const pose = body();
  [[11, 0.35, 0.3], [12, 0.65, 0.3], [23, 0.4, 0.7], [24, 0.6, 0.7]]
    .forEach(([i, x, y]) => Object.assign(pose.landmarks[i], { x, y }));
  const source = { pose, hands: [hand(0.4, 0.5, 0.4, 0.5), hand(0.5, 0.6, 0.4, 0.5)] };
  assert.equal(reconcileDetections(source), source);
});

test('one torso joint well outside the hand support envelope preserves the pose', () => {
  const source = { pose: body(), hands: [hand(0.4, 0.6, 0.4, 0.6)] };
  source.pose.landmarks[24].y = 0.7;
  assert.equal(reconcileDetections(source), source);
});

test('missing, partial or invalid detections cannot trigger the suppression', () => {
  const noBody = { pose: null, hands: [hand(0, 1, 0, 1)] };
  assert.equal(reconcileDetections(noBody), noBody);
  const sparseHand = { pose: body(), hands: [{ landmarks: [{ x: 0, y: 0 }, { x: 1, y: 1 }] }] };
  assert.equal(reconcileDetections(sparseHand), sparseHand);
  const hiddenCore = { pose: body(), hands: [hand(0, 1, 0, 1)] };
  hiddenCore.pose.landmarks[23].visibility = 0.1;
  assert.equal(reconcileDetections(hiddenCore), hiddenCore);
});

const measured = JSON.parse(readFileSync(new URL('./data/hand-confusion.json', import.meta.url), 'utf8'));
const measuredFrame = sample => ({
  sourceWidth: measured.sourceWidth, sourceHeight: measured.sourceHeight,
  pose: { landmarks: sample.pose.map(([x, y, visibility]) => ({ x, y, visibility })) },
  face: { landmarks: sample.face.map(([x, y]) => ({ x, y })) },
  hands: sample.hands.map(points => ({ landmarks: points.map(([x, y]) => ({ x, y })) })),
});

test('measured Full regressions: palm face and torso beyond wrist joint centres are suppressed', () => {
  for (const sample of measured.samples) {
    const source = measuredFrame(sample), before = structuredClone(source);
    const output = reconcileDetections(source);
    assert.equal(output.pose, null, `false body at ${sample.mediaTime}s`);
    if (source.face.landmarks.length) assert.equal(output.face, null, 'false dense face over palm');
    assert.deepEqual(source, before);
    assert.equal(output.hands, source.hands);
  }
});

test('the support envelope uses source pixels consistently across aspect ratios', () => {
  const source = measuredFrame(measured.samples[2]);
  const expanded = structuredClone(source);
  expanded.sourceWidth *= 2;
  for (const part of [expanded.pose, expanded.face, ...expanded.hands]) {
    for (const point of part.landmarks) point.x /= 2;
  }
  const a = reconcileDetections(source), b = reconcileDetections(expanded);
  assert.equal(a.pose, null);
  assert.equal(b.pose, null);
  assert.ok(Math.abs(a.fusion.paddingPx - b.fusion.paddingPx) < 1e-9);
});

test('a body with most confident joints outside a foreground hand remains visible', () => {
  const source = measuredFrame(measured.samples[1]);
  for (let i = 0; i < source.pose.landmarks.length; i++) {
    if (![11, 12, 23, 24].includes(i)) Object.assign(source.pose.landmarks[i], { x: 0.15, y: 0.2, visibility: 1 });
  }
  assert.equal(reconcileDetections(source), source);
});

test('a face beside or only partly overlapped by a hand is retained', () => {
  const source = measuredFrame(measured.samples[0]);
  source.pose = null;
  for (const point of source.face.landmarks) point.x = Math.max(0.01, point.x - 0.25);
  assert.equal(reconcileDetections(source), source);
});

test('a face between spread fingertips is retained even inside the whole-hand bounding box', () => {
  const source = measuredFrame(measured.samples[0]);
  source.pose = null;
  // A spread-hand box covers space between fingers; it does not imply skin.
  // Put a compact visible face above the metacarpal edge, inside the hand box.
  source.face.landmarks = source.face.landmarks.map((point, index) => ({ x: 0.53 + Math.sin(index) * 0.025, y: 0.40 + Math.cos(index) * 0.025 }));
  assert.equal(reconcileDetections(source), source);
});

test('a credible torso outside a close foreground hand preserves its associated face', () => {
  const source = measuredFrame(measured.samples[0]);
  [[11, 0.3, 0.65], [12, 0.7, 0.65], [23, 0.35, 0.9], [24, 0.65, 0.9]]
    .forEach(([i, x, y]) => Object.assign(source.pose.landmarks[i], { x, y, visibility: 1 }));
  assert.equal(reconcileDetections(source), source);
});
