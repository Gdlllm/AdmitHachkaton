import test from 'node:test';
import assert from 'node:assert/strict';
import { FaceLandmarker, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import { CaptureRenderer, containViewport, projectLandmark, isDrawableLandmark, trackingOpacity, matchBodyWrists } from '../src/rendering/renderer.js';
import * as topology from '../src/rendering/topology.js';

const entries = source => source.map(({ start, end }) => [start, end]);
const edgeKeys = values => new Set(values.map(([a, b]) => [Math.min(a, b), Math.max(a, b)].join(':')));

test('landmarks project to exactly the contained camera image in portrait and landscape', () => {
  const landscape = containViewport(1000, 1000, 1920, 1080);
  assert.deepEqual(landscape, { x: 0, y: 218.75, width: 1000, height: 562.5, scale: 1000 / 1920 });
  assert.deepEqual(projectLandmark({ x: 0, y: 0 }, landscape, false), { x: 0, y: 218.75 });
  assert.deepEqual(projectLandmark({ x: 0, y: 0 }, landscape, true), { x: 1000, y: 218.75 });
  assert.deepEqual(projectLandmark({ x: 1, y: 1 }, landscape, true), { x: 0, y: 781.25 });
  const portrait = containViewport(1200, 800, 720, 1280);
  assert.deepEqual(projectLandmark({ x: 0.5, y: 0.5 }, portrait), { x: 600, y: 400 });
  assert.equal(portrait.x, 375);
  assert.equal(portrait.height, 800);
  assert.equal(containViewport(1200, 800, 0, 0), null);
  assert.equal(projectLandmark({ x: NaN, y: 0.5 }, landscape), null);
});

test('valid edge points survive but pose occlusion and invalid tracks never draw', () => {
  assert.equal(isDrawableLandmark({ x: 0, y: 1 }), true);
  assert.equal(isDrawableLandmark({ x: -0.01, y: 0.5 }), true);
  assert.equal(isDrawableLandmark({ x: -0.3, y: 0.5 }), false);
  assert.equal(isDrawableLandmark({ x: 0.5, y: 0.5, visibility: 0.1 }, { pose: true }), false);
  assert.equal(isDrawableLandmark({ x: 0.5, y: 0.5, presence: 0.1 }, { pose: true }), false);
  assert.equal(isDrawableLandmark({ x: 0.5, y: 0.5, valid: false }), false);
  assert.equal(isDrawableLandmark({ x: 0.5, y: 0.5, predicted: true }), false);
  assert.equal(isDrawableLandmark({ x: 0.5, y: 0.5, drawConfidence: 0 }), false);
  // The face/hand model's missing visibility field is not an occlusion signal.
  assert.equal(isDrawableLandmark({ x: 0.5, y: 0.5, visibility: 0 }), true);
});

test('late inference is visually fresh when received, then stale geometry fades away', () => {
  const frame = { timestamp: 100, receivedAt: 650 };
  assert.equal(trackingOpacity(frame, 650), 1);
  assert.equal(trackingOpacity(frame, 800), 1);
  assert.ok(trackingOpacity(frame, 1000) > 0 && trackingOpacity(frame, 1000) < 1);
  assert.equal(trackingOpacity(frame, 1200), 0);
});

test('all exported facial connections match the installed official topology', () => {
  const mapping = {
    FACE_OVAL: 'FACE_LANDMARKS_FACE_OVAL', FACE_LIPS: 'FACE_LANDMARKS_LIPS',
    FACE_LEFT_EYE: 'FACE_LANDMARKS_LEFT_EYE', FACE_RIGHT_EYE: 'FACE_LANDMARKS_RIGHT_EYE',
    FACE_LEFT_BROW: 'FACE_LANDMARKS_LEFT_EYEBROW', FACE_RIGHT_BROW: 'FACE_LANDMARKS_RIGHT_EYEBROW',
    FACE_LEFT_IRIS: 'FACE_LANDMARKS_LEFT_IRIS', FACE_RIGHT_IRIS: 'FACE_LANDMARKS_RIGHT_IRIS',
  };
  for (const [name, official] of Object.entries(mapping)) assert.deepEqual(topology[name], entries(FaceLandmarker[official]));
  assert.deepEqual(edgeKeys(topology.FACE_TESSELLATION), edgeKeys(entries(FaceLandmarker.FACE_LANDMARKS_TESSELATION)));
  assert.equal(topology.FACE_TESSELLATION.length, edgeKeys(topology.FACE_TESSELLATION).size);
  assert.deepEqual(topology.HAND_CONNECTIONS, entries(HandLandmarker.HAND_CONNECTIONS));
  assert.deepEqual(topology.BODY_CONNECTIONS, entries(PoseLandmarker.POSE_CONNECTIONS).filter(([a, b]) => a >= 11 && b >= 11));
  assert.ok(topology.FACE_LEFT_IRIS.flat().every(i => i >= 468 && i < 478));
  assert.ok(topology.FACE_RIGHT_IRIS.flat().every(i => i >= 468 && i < 478));
});

function mockCanvas() {
  const calls = [];
  const methods = ['setTransform', 'clearRect', 'save', 'restore', 'beginPath', 'rect', 'clip', 'translate', 'scale', 'drawImage', 'moveTo', 'lineTo', 'arc', 'stroke', 'fill'];
  const context = Object.fromEntries(methods.map(name => [name, (...args) => calls.push([name, ...args])]));
  return { width: 1, height: 1, style: {}, getContext: () => context, calls };
}

test('DPR does not alter coordinates and the source uses the same mirrored contain rectangle', () => {
  const canvas = mockCanvas();
  const renderer = new CaptureRenderer(canvas);
  renderer.setSize(1000, 1000, 2);
  const source = { width: 1920, height: 1080 };
  renderer.draw({ timestamp: 100, pose: { landmarks: [] } }, { videoWidth: 1920, videoHeight: 1080, now: 100, source });
  assert.equal(canvas.width, 2000); assert.equal(canvas.height, 2000);
  assert.equal(canvas.style.width, '1000px');
  assert.ok(canvas.calls.some(c => c[0] === 'rect' && c[1] === 0 && c[2] === 218.75 && c[3] === 1000 && c[4] === 562.5));
  assert.ok(canvas.calls.some(c => c[0] === 'translate' && c[1] === 1000 && c[2] === 218.75));
  assert.ok(canvas.calls.some(c => c[0] === 'scale' && c[1] === -1 && c[2] === 1));
  assert.ok(canvas.calls.some(c => c[0] === 'drawImage' && c[1] === source && c[4] === 1000 && c[5] === 562.5));
  assert.equal(canvas.calls.filter(c => c[0] === 'save').length, canvas.calls.filter(c => c[0] === 'restore').length);
});

test('body overlay cannot manufacture a coarse face or hand from pose landmarks', () => {
  const canvas = mockCanvas();
  const renderer = new CaptureRenderer(canvas);
  renderer.setSize(1000, 800);
  const landmarks = Array.from({ length: 33 }, (_, index) => ({ x: index / 40, y: 0.5, visibility: index < 11 || index >= 17 && index <= 22 ? 1 : 0 }));
  renderer.draw({ timestamp: 1, pose: { landmarks } }, { videoWidth: 1000, videoHeight: 800, now: 1 });
  assert.equal(canvas.calls.filter(c => c[0] === 'lineTo').length, 0);
  assert.equal(canvas.calls.filter(c => c[0] === 'arc').length, 0);
});

test('missing points do not bridge adjacent hand bones', () => {
  const canvas = mockCanvas();
  const renderer = new CaptureRenderer(canvas);
  const hand = Array.from({ length: 21 }, (_, index) => ({ x: 0.3 + index / 100, y: 0.5 }));
  hand[2] = null;
  renderer.setSize(1000, 800);
  renderer.draw({ timestamp: 1, hands: [{ handedness: 'Left', landmarks: hand }] }, { videoWidth: 1000, videoHeight: 800, now: 1 });
  // Two of the 21 official connections touch missing joint #2; shadow + color pass.
  assert.equal(canvas.calls.filter(c => c[0] === 'lineTo').length, 19 * 2);
  assert.ok(canvas.calls.filter(c => c[0] === 'arc').length > 0);
});

function wristFixture() {
  const pose = Array(33).fill(null);
  pose[13] = { x: 0.2, y: 0.3, visibility: 1 };
  pose[15] = { x: 0.2, y: 0.5, visibility: 1 };
  pose[14] = { x: 0.8, y: 0.3, visibility: 1 };
  pose[16] = { x: 0.8, y: 0.5, visibility: 1 };
  const hand = (x, y = 0.5, palm = 0.05) => {
    const landmarks = Array(21).fill(null);
    landmarks[0] = { x, y, visibility: 0, drawConfidence: 1 };
    landmarks[9] = { x, y: y - palm, visibility: 0, drawConfidence: 1 };
    return { landmarks };
  };
  return { pose, hand };
}

test('nearby unambiguous measured wrists stitch without mutating model coordinates', () => {
  const { pose, hand } = wristFixture();
  const hands = [hand(0.21), hand(0.79)];
  const before = JSON.stringify({ pose, hands });
  const matches = matchBodyWrists(pose, hands, 1000, 1000);
  assert.deepEqual(matches.map(match => [match.poseIndex, match.handIndex]), [[15, 0], [16, 1]]);
  assert.ok(matches.every(match => Math.abs(match.distancePx - 10) < 1e-8));
  assert.equal(JSON.stringify({ pose, hands }), before);
});

test('wrist association uses source pixels and rejects distant or tiny detailed hands', () => {
  const { pose, hand } = wristFixture();
  assert.equal(matchBodyWrists(pose, [hand(0.26)], 1000, 1000).length, 0);
  assert.equal(matchBodyWrists(pose, [hand(0.204, 0.5, 0.005)], 1000, 1000).length, 0);
  // A .03 image-width displacement is 30px here but 60px in the wider source.
  assert.equal(matchBodyWrists(pose, [hand(0.229)], 1000, 1000).length, 1);
  assert.equal(matchBodyWrists(pose, [hand(0.229)], 2000, 1000).length, 0);
});

test('crossing or competing hands leave the original arm endpoints untouched', () => {
  const { pose, hand } = wristFixture();
  pose[15].x = 0.497; pose[16].x = 0.503;
  assert.deepEqual(matchBodyWrists(pose, [hand(0.495), hand(0.505)], 1000, 1000), []);
  assert.deepEqual(matchBodyWrists(pose, [hand(0.5)], 1000, 1000), []);
});

test('occluded pose wrists and invalid detailed wrists cannot be used for stitching', () => {
  const { pose, hand } = wristFixture();
  pose[15].visibility = 0.1;
  assert.deepEqual(matchBodyWrists(pose, [hand(0.21)], 1000, 1000), []);
  pose[15].visibility = 1;
  const missing = hand(0.21); missing.landmarks[0].drawConfidence = 0;
  assert.deepEqual(matchBodyWrists(pose, [missing], 1000, 1000), []);
});

test('renderer joins the forearm to the actual projected hand wrist without changing source points', () => {
  const { pose, hand } = wristFixture();
  const hands = [hand(0.21), hand(0.79)];
  const snapshot = JSON.stringify({ pose, hands });
  const renderer = new CaptureRenderer(mockCanvas());
  renderer.setSize(1000, 1000);
  let projectedBody;
  renderer.drawBody = points => { projectedBody = points; };
  renderer.draw({ timestamp: 1, pose: { landmarks: pose }, hands }, { videoWidth: 1000, videoHeight: 1000, now: 1, mirror: true });
  assert.equal(projectedBody[15].x, 790);
  assert.ok(Math.abs(projectedBody[16].x - 210) < 1e-8);
  assert.equal(projectedBody[15].y, 500);
  assert.equal(JSON.stringify({ pose, hands }), snapshot);
});
