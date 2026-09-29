import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { assessBody, personBox, landmarkScore, FULL_BODY_JOINTS } from '../src/dense/inference/body-gate.js';
import { reconcileDetections } from '../src/tracking/fusion.js';

function person() {
  // Face points around the head, coarse hand points next to the wrists.
  const p = Array.from({ length: 33 }, (_, i) => ({ x: 0.5, y: i <= 10 ? 0.11 : 0.15, visibility: 0.99, presence: 0.99 }));
  const xy = { 0:[.5,.12],2:[.48,.1],5:[.52,.1],17:[.19,.52],19:[.19,.52],21:[.2,.51],18:[.81,.52],20:[.81,.52],22:[.8,.51],11:[.4,.3],12:[.6,.3],13:[.3,.4],14:[.7,.4],15:[.2,.5],16:[.8,.5],
    23:[.43,.55],24:[.57,.55],25:[.43,.72],26:[.57,.72],27:[.43,.9],28:[.57,.9],
    29:[.42,.91],30:[.58,.91],31:[.40,.93],32:[.60,.93] };
  for (const [i,[x,y]] of Object.entries(xy)) Object.assign(p[i],{x,y});
  return p;
}
// A Face Landmarker result around the Pose nose (478 points on a small ring).
const faceAround = ({ x, y }, radius = 0.04) => Array.from({ length: 478 }, (_, i) => ({ x: x + radius * Math.cos(i), y: y + radius * Math.sin(i) }));
const FACE = faceAround({ x: .5, y: .12 });
const run = (pose, options) => assessBody(pose, 1000, 1000, { face: FACE, ...options });
const hide = (p, ids, value = 0.1) => { for (const i of ids) p[i].visibility = value; return p; };
const LEGS = [25, 26, 27, 28, 29, 30, 31, 32];

test('a complete, confident body is full, unclipped, with a detector-like box, and input is not mutated', () => {
  const p = person(), before = JSON.stringify(p), g = run(p);
  assert.equal(g.level, 'full'); assert.equal(g.enough, true); assert.equal(g.tracked, true);
  assert.deepEqual(g.missing, []); assert.equal(g.clipY, null);
  const [x0, y0, x1, y1] = g.bbox;
  assert.ok(x0 < 200 && x1 > 800, 'box covers the outstretched arms');
  assert.ok(y0 < 100, 'box extends above the eyes to the scalp');
  assert.ok(y1 > 930 && y1 <= 1000, 'box reaches the feet and stays in the image');
  assert.equal(JSON.stringify(p), before);
  assert.equal(run(p, { face: [] }).level, 'full', 'a full body needs no face (far away or turned around)');
});

test('standing close: legs cut by the bottom edge give an unclipped upper body', () => {
  const p = person();
  for (const i of LEGS) Object.assign(p[i], { y: p[i].y + 0.3, visibility: 0.2 });
  const g = run(p);
  assert.equal(g.level, 'upper'); assert.equal(g.enough, true);
  assert.equal(g.clipY, null, 'the thighs down to the frame edge are real'); assert.equal(g.clipKnown, true);
  assert.ok(g.bbox[3] <= 1000);
});

test('sitting at a desk: hips and legs hidden inside the frame, elbows seen, drawn only down to the hands', () => {
  const g = run(hide(person(), [23, 24, ...LEGS]));
  assert.equal(g.level, 'upper'); assert.equal(g.reason, 'upper-body');
  // Lowest visible joint: wrists at y = 500; margin 0.15 * (1.3 * shoulder width 200).
  assert.ok(Math.abs(g.clipY - 539) < 1e-6, String(g.clipY)); assert.equal(g.clipKnown, true);
  assert.ok(g.bbox, 'the network crop needs no hips');
});

test('one hidden knee inside the frame gives an upper body clipped below the lowest visible joint', () => {
  const g = run(hide(person(), [25]));
  assert.equal(g.level, 'upper'); assert.deepEqual(g.missing, [25]);
  assert.ok(g.clipY > 930, 'the feet are still visible, so almost nothing is hidden');
});

test('a hand spanning both shoulders needs the face around the Pose nose; otherwise no face is needed', () => {
  const desk = hide(person(), [23, 24, ...LEGS]);
  // 21 hand points covering the shoulder span, like a hand held close to the camera.
  const handOver = [{ landmarks: Array.from({ length: 21 }, (_, i) => ({ x: .37 + .26 * (i % 7) / 6, y: .2 + .2 * Math.floor(i / 7) / 2 })) }];
  for (const face of [undefined, [], faceAround({ x: .2, y: .5 }), FACE.slice(0, 100)]) {
    const g = run(desk, { face, hands: handOver });
    assert.equal(g.level, 'none'); assert.equal(g.reason, 'hand-over-shoulders-without-face'); assert.equal(g.tracked, true);
    assert.equal(run(desk, { face }).level, 'upper', 'a face too small to detect is fine without a hand over the body');
  }
  assert.equal(run(desk, { hands: handOver }).level, 'upper', 'a matching face confirms a real person');
  const wristHand = [{ landmarks: Array.from({ length: 21 }, (_, i) => ({ x: .18 + .04 * (i % 7) / 6, y: .48 + .06 * Math.floor(i / 7) / 2 })) }];
  assert.equal(run(desk, { face: [], hands: wristHand }).level, 'upper', 'a hand at the wrist is not over the shoulders');
});

test('head and shoulders alone are not enough, but keep an active surface tracked and clipped', () => {
  const g = run(hide(person(), [13, 14, 15, 16, 23, 24, ...LEGS]));
  assert.equal(g.level, 'none'); assert.equal(g.reason, 'needs-hips-or-elbows'); assert.equal(g.tracked, true);
  assert.ok(Math.abs(g.clipY - 339) < 1e-6, 'drawn only down to just below the shoulders');
  const lost = run(hide(person(), [11, 12], 0.2));
  assert.equal(lost.clipKnown, false, 'without shoulders the previous clip line is kept');
});

test('close-up at a laptop: head and shoulders with the body leaving the frame need a matching face', () => {
  // Shoulders low in the frame; elbows, hips and legs predicted below the bottom edge.
  const p = person().map((point, i) => ({ ...point, y: point.y + 0.45 }));
  for (const i of [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, ...LEGS]) p[i].visibility = 0.1;
  const face = faceAround({ x: p[0].x, y: p[0].y });
  const g = run(p, { face });
  assert.equal(g.level, 'upper'); assert.equal(g.reason, 'close-up'); assert.equal(g.clipY, null);
  assert.equal(g.bbox[3], 1000, 'the network crop reaches down to the frame edge');
  assert.equal(run(p, { face: [] }).reason, 'close-up-without-face');
  const covered = p.map((point, i) => [23, 24, 13, 14].includes(i) ? { ...point, y: 0.9 } : point);
  assert.equal(run(covered, { face }).reason, 'needs-hips-or-elbows', 'hidden inside the frame is not a close-up');
});

test('stabilizer-hidden points count as unseen regardless of their raw visibility', () => {
  const p = person(); p[25].drawConfidence = 0;
  assert.equal(landmarkScore(p[25]), 0);
  assert.equal(run(p).level, 'upper');
});

test('presence and non-finite values are respected', () => {
  const p = person(); p[23].presence = 0.2;
  assert.equal(run(p).full, false);
  const q = person(); q[24].visibility = NaN; q[12].x = NaN;
  const g = run(q); assert.equal(g.level, 'none'); assert.equal(g.tracked, false); assert.equal(g.bbox, null);
});

test('shoulders below the keep threshold are not tracked and get no box', () => {
  const g = run(hide(person(), [11, 12], 0.2));
  assert.equal(g.tracked, false); assert.equal(g.bbox, null); assert.equal(g.level, 'none');
});

test('empty, short and malformed candidates never become a body', () => {
  for (const pose of [undefined, [], person().slice(0, 20)]) {
    const g = run(pose); assert.equal(g.enough, false); assert.equal(g.tracked, false); assert.equal(g.bbox, null);
  }
  assert.equal(assessBody(person(), 0, 100).reason, 'invalid-source-size');
});

test('collapsed anchors and tiny bodies are rejected', () => {
  const collapsed = person().map(p => ({ ...p, x: .5, y: .5 }));
  assert.equal(run(collapsed, { face: faceAround({ x: .5, y: .5 }) }).reason, 'degenerate-body');
  const tiny = person().map(p => ({ ...p, x: .5 + (p.x - .5) * .12, y: .5 + (p.y - .5) * .12 }));
  assert.equal(run(tiny, { face: faceAround(tiny[0], .005) }).reason, 'body-too-small');
});

test('rotated and squatting bodies are still full bodies', () => {
  const rotated = person().map(p => ({ ...p, x: 1 - p.y, y: p.x }));
  assert.equal(run(rotated).level, 'full');
  const squat = person(); Object.assign(squat[25], { x: .65, y: .53 }); Object.assign(squat[26], { x: .35, y: .53 });
  assert.equal(run(squat).level, 'full');
});

test('thresholds are configurable', () => {
  const p = hide(person(), FULL_BODY_JOINTS, 0.45);
  assert.equal(run(p).level, 'none');
  assert.equal(run(p, { enter: 0.4 }).level, 'full');
});

test('the person box adds the whole head when no facial point is confident, and needs no hips', () => {
  const withFace = personBox(person(), 1000, 1000);
  const withoutFace = personBox(hide(person(), [...Array(11).keys()]), 1000, 1000);
  assert.ok(withoutFace[1] < 300 && withoutFace[1] > 0, 'head allowance above the shoulders');
  assert.ok(withFace[1] < withoutFace[1] + 200);
  const seated = personBox(hide(person(), [23, 24, ...LEGS]), 1000, 1000);
  assert.ok(seated[3] < 600, 'box ends at the visible upper body');
  assert.equal(personBox(person(), 1000, 0), null);
});

test('the 36 recorded hand-shaped Pose bodies never reach the surface', () => {
  const measured = JSON.parse(readFileSync(new URL('./data/hand-confusion.json', import.meta.url), 'utf8'));
  const frames = measured.samples.map(sample => ({ sourceWidth: measured.sourceWidth, sourceHeight: measured.sourceHeight,
    pose: { landmarks: sample.pose.map(([x, y, visibility]) => ({ x, y, visibility })) },
    face: { landmarks: sample.face.map(([x, y]) => ({ x, y })) },
    hands: sample.hands.map(points => ({ landmarks: points.map(([x, y]) => ({ x, y })) })) }));
  assert.equal(frames.length, 36);
  // Pose alone is no person detector: 4 hand bodies look full, 1 upper body even
  // has a hallucinated face on it (30 more are stopped by the hand/face rule).
  // The hand/face fusion in front removes all of them.
  const gate = frame => assessBody(frame.pose?.landmarks, frame.sourceWidth, frame.sourceHeight, { face: frame.face?.landmarks, hands: frame.hands }).enough;
  assert.equal(frames.filter(gate).length, 5);
  assert.equal(frames.map(reconcileDetections).filter(gate).length, 0);
});
