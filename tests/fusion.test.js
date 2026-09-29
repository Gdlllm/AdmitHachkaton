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

test('one torso joint outside the envelope preserves a pose with its head independently visible', () => {
  const source = { pose: body(), hands: [hand(0.4, 0.6, 0.4, 0.6)] };
  source.pose.landmarks[24].y = 0.7;
  for (let i = 0; i < 11; i++) source.pose.landmarks[i].y = 0.2;
  assert.equal(reconcileDetections(source), source);
});

test('missing, partial or invalid detections cannot trigger the suppression', () => {
  const noBody = { pose: null, hands: [hand(0, 1, 0, 1)] };
  assert.equal(reconcileDetections(noBody), noBody);
  const sparseHand = { pose: body(), hands: [{ landmarks: [{ x: 0, y: 0 }, { x: 1, y: 1 }] }] };
  assert.equal(reconcileDetections(sparseHand), sparseHand);
  const hiddenCore = { pose: body(), hands: [hand(0, 1, 0, 1)] };
  hiddenCore.pose.landmarks[23].visibility = 0.1;
  hiddenCore.pose.landmarks[0].visibility = 0.1;
  assert.equal(reconcileDetections(hiddenCore), hiddenCore);
});

const measured = JSON.parse(readFileSync(new URL('./data/hand-confusion.json', import.meta.url), 'utf8'));
const sampleById = id => {
  const sample = measured.samples.find(sample => sample.id === id);
  assert.ok(sample, `missing measured sample ${id}`);
  return sample;
};
const measuredFrame = sample => ({
  sourceWidth: sample.sourceWidth ?? measured.sourceWidth, sourceHeight: sample.sourceHeight ?? measured.sourceHeight,
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
  const source = measuredFrame(sampleById('hand-clip-12.552089'));
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

test('face containment is invariant to normalized x rescaling with matching source width', () => {
  const source = measuredFrame(sampleById('enlarged-palm-face'));
  const expanded = structuredClone(source);
  expanded.sourceWidth *= 2;
  for (const part of [expanded.pose, expanded.face, ...expanded.hands]) {
    for (const point of part.landmarks) point.x /= 2;
  }
  assert.equal(reconcileDetections(source).face, null);
  assert.equal(reconcileDetections(expanded).face, null);
});

test('a body with independently visible head and most limbs outside a foreground hand remains visible', () => {
  const source = measuredFrame(sampleById('hand-clip-2.377557'));
  // Actual elbows/wrists/legs/feet supply the body-coverage outside evidence.
  // The independently visible head avoids the separate upper-body ambiguity.
  for (const i of [13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32]) {
    Object.assign(source.pose.landmarks[i], { x: 0.15, y: 0.2, visibility: 1 });
  }
  for (let i = 0; i < 11; i++) Object.assign(source.pose.landmarks[i], { x: 0.15, y: 0.15 });
  assert.equal(reconcileDetections(source), source);
});

test('head and both shoulders inside one hand remain ambiguous even with apparent legs outside', () => {
  const source = measuredFrame(sampleById('hand-whole-12.52'));
  const output = reconcileDetections(source);
  assert.equal(output.pose, null);
  assert.equal(output.fusion.reason, 'head-and-shoulders-contained-in-hand');
  assert.equal(output.fusion.poseAmbiguous, true);
  // The same coordinates could describe a real body behind a huge foreground
  // hand. This intentionally documents that conservative occlusion tradeoff.
  assert.ok(source.pose.landmarks[32].y > 0.9);
});

test('a hand covering the head alone does not suppress independently visible shoulders', () => {
  const source = measuredFrame(sampleById('hand-whole-12.52'));
  source.pose.landmarks[11].x = 0.15;
  source.pose.landmarks[12].x = 0.85;
  assert.equal(reconcileDetections(source), source);
});

test('the upper-body ambiguity rule uses raw hand bounds, not the padded envelope', () => {
  const source = measuredFrame(sampleById('hand-whole-12.52'));
  const minX = Math.min(...source.hands[0].landmarks.map(point => point.x));
  source.pose.landmarks[0].x = minX - 0.001;
  assert.equal(reconcileDetections(source), source);
});

test('missing or low-confidence head evidence cannot trigger upper-body suppression', () => {
  for (const value of [null, { x: 0.5, y: 0.5, visibility: 0.1 }]) {
    const source = measuredFrame(sampleById('hand-whole-12.52'));
    source.pose.landmarks[0] = value;
    assert.equal(reconcileDetections(source), source);
  }
});

test('a spatially matching dense face between fingers independently supports its body', () => {
  const source = measuredFrame(sampleById('hand-whole-12.52'));
  const head = source.pose.landmarks.slice(0, 11);
  const minX = Math.min(...head.map(p => p.x)), maxX = Math.max(...head.map(p => p.x));
  const minY = Math.min(...head.map(p => p.y)), maxY = Math.max(...head.map(p => p.y));
  source.face.landmarks = Array.from({ length: 478 }, (_, i) => ({
    x: (minX + maxX) / 2 + Math.cos(i) * (maxX - minX) * 0.6,
    y: (minY + maxY) / 2 + Math.sin(i) * (maxY - minY) * 0.6,
  }));
  assert.equal(reconcileDetections(source), source);
});

test('an unrelated distant face cannot confirm a tiny body inside the hand', () => {
  const source = measuredFrame(sampleById('hand-whole-12.52'));
  source.face.landmarks = Array.from({ length: 478 }, (_, i) => ({ x: 0.15 + Math.cos(i) * 0.05, y: 0.15 + Math.sin(i) * 0.05 }));
  const output = reconcileDetections(source);
  assert.equal(output.pose, null);
  assert.equal(output.face, source.face);
});

test('a coincident palm-face and miniature body cannot mutually corroborate each other', () => {
  const source = measuredFrame(sampleById('hand-whole-12.52'));
  const palm = [0, 1, 5, 9, 13, 17].map(i => source.hands[0].landmarks[i]);
  const center = { x: palm.reduce((sum, p) => sum + p.x, 0) / palm.length,
    y: palm.reduce((sum, p) => sum + p.y, 0) / palm.length };
  for (const i of [0, 2, 5]) Object.assign(source.pose.landmarks[i], center);
  source.face.landmarks = Array.from({ length: 478 }, (_, i) => ({ x: center.x + Math.cos(i) * 0.002, y: center.y + Math.sin(i) * 0.002 }));
  const output = reconcileDetections(source);
  assert.equal(output.pose, null);
  assert.equal(output.face, null);
});

test('two separate hands cannot combine their bounds to hide the upper body', () => {
  const source = measuredFrame(sampleById('hand-whole-12.52'));
  for (let i = 0; i < 13; i++) Object.assign(source.pose.landmarks[i], { x: 0.35 + 0.3 * i / 12, y: 0.4 });
  source.hands = [hand(0.3, 0.5, 0.3, 0.5), hand(0.5, 0.7, 0.3, 0.5)];
  assert.equal(reconcileDetections(source), source);
});

test('a face beside or only partly overlapped by a hand is retained', () => {
  const source = measuredFrame(sampleById('hand-clip-2.015274'));
  source.pose = null;
  for (const point of source.face.landmarks) point.x = Math.max(0.01, point.x - 0.25);
  assert.equal(reconcileDetections(source), source);
});

test('a face extending outside the physical hand envelope remains visible', () => {
  const source = measuredFrame(sampleById('enlarged-palm-face'));
  source.pose = null;
  for (let i = 0; i < 50; i++) source.face.landmarks[i].x = 0.25;
  assert.equal(reconcileDetections(source), source);
});

test('an out-of-frame facial point prevents the entire-face containment heuristic', () => {
  const source = measuredFrame(sampleById('enlarged-palm-face'));
  source.pose = null;
  source.face.landmarks[0].x = -0.01;
  assert.equal(reconcileDetections(source), source);
});

test('a non-finite facial point prevents the entire-face containment heuristic', () => {
  const source = measuredFrame(sampleById('enlarged-palm-face'));
  source.pose = null;
  source.face.landmarks[0].x = NaN;
  assert.equal(reconcileDetections(source), source);
});

test('a face between spread fingertips is retained even inside the whole-hand bounding box', () => {
  const source = measuredFrame(sampleById('hand-clip-2.015274'));
  source.pose = null;
  // A spread-hand box covers space between fingers; it does not imply skin.
  // Put a compact visible face above the metacarpal edge, inside the hand box.
  source.face.landmarks = source.face.landmarks.map((point, index) => ({ x: 0.53 + Math.sin(index) * 0.025, y: 0.40 + Math.cos(index) * 0.025 }));
  assert.equal(reconcileDetections(source), source);
});

test('a credible torso outside a close foreground hand preserves its associated face', () => {
  const source = measuredFrame(sampleById('hand-clip-2.015274'));
  [[11, 0.3, 0.65], [12, 0.7, 0.65], [23, 0.35, 0.9], [24, 0.65, 0.9]]
    .forEach(([i, x, y]) => Object.assign(source.pose.landmarks[i], { x, y, visibility: 1 }));
  assert.equal(reconcileDetections(source), source);
});
