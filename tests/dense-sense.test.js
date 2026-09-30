import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { createDecoderFromArrays, viewsFromManifest } from '../src/dense/decoder/mhr-decoder.js';
import { flipFrontBack } from '../src/dense/inference/mhr-flip.js';
import { createBodySense, parseBodySense, swapPairs, torsoForward, faceForward, frameNormalization, labelFacing, FEATURES } from '../src/dense/inference/body-sense.js';
import { createSensePolicy } from '../src/dense/inference/sense-policy.js';
import { createLiftPolicy } from '../src/dense/inference/lift-policy.js';

const directory = new URL('../public/dense/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('mhr-lod3.json', directory)));
const bytes = gunzipSync(readFileSync(new URL('mhr-lod3.bin.gz', directory)));
const decoder = createDecoderFromArrays(manifest, viewsFromManifest(manifest, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
const swapName = name => name.replace(/(^|_)([lr])_/g, (_, pre, side) => `${pre}${side === 'l' ? 'r' : 'l'}_`);
const jointPair = manifest.jointNames.map(name => manifest.jointNames.indexOf(swapName(name)));
let seed = 7;
const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const randomPose = () => Float32Array.from({ length: 204 }, (_, i) => (random() - .5) * (i < 3 ? .5 : i < 6 ? 3 : .6));

test('front/back flip is the depth mirror of the same body about its root, left and right relabelled', () => {
  for (let trial = 0; trial < 12; trial++) {
    const q = randomPose(), X = decoder.decodeSkeleton(q).positions, Y = decoder.decodeSkeleton(flipFrontBack(q)).positions;
    const rootZ = X[5];
    let worst = 0;
    for (let j = 1; j < manifest.jointNames.length; j++) {
      const k = jointPair[j];
      worst = Math.max(worst, Math.hypot(Y[j * 3] - X[k * 3], Y[j * 3 + 1] - X[k * 3 + 1], Y[j * 3 + 2] - (2 * rootZ - X[k * 3 + 2])));
    }
    assert.ok(worst < 1e-4, `joint error ${worst} m`);
    const back = decoder.decodeSkeleton(flipFrontBack(flipFrontBack(q))).positions;
    let again = 0; for (let i = 3; i < X.length; i++) again = Math.max(again, Math.abs(back[i] - X[i]));
    assert.ok(again < 1e-4, 'flipping twice returns the same body');
  }
});

test('torso and face directions use camera axes (X right, Y down, Z forward)', () => {
  const rest = decoder.decodeKeypoints(new Float32Array(204), [5, 6, 9, 10]);
  const k = [5, 6, 9, 10].flatMap(i => [rest[i * 3], rest[i * 3 + 1], rest[i * 3 + 2]]);
  const f = torsoForward(k);
  assert.ok(f[2] < -.95, 'the rest pose faces the camera');
  const turned = decoder.decodeKeypoints(Float32Array.from({ length: 204 }, (_, i) => i === 4 ? Math.PI : 0), [5, 6, 9, 10]);
  assert.ok(torsoForward([5, 6, 9, 10].flatMap(i => [turned[i * 3], turned[i * 3 + 1], turned[i * 3 + 2]]))[2] > .95, 'turned round it shows its back');
  // Column-major MediaPipe matrix of a frontal face, then one turned to its left (towards image right).
  const frontal = { data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -33, 1] };
  assert.deepEqual(faceForward(frontal).map(v => Math.round(v * 1000) / 1000), [0, -0, -1]);
  const a = .5, left = { data: [Math.cos(a), 0, -Math.sin(a), 0, 0, 1, 0, 0, Math.sin(a), 0, Math.cos(a), 0, 0, 0, -33, 1] };
  assert.ok(faceForward(left)[0] > .4);
  assert.equal(faceForward(null), null);
});

test('swapping pairs exchanges whole landmarks and leaves the input alone', () => {
  const pose = Array.from({ length: 33 }, (_, i) => ({ x: i, y: i, visibility: 1, presence: 1 }));
  const swapped = swapPairs(pose, Array.from({ length: 16 }, (_, i) => i === 5));
  assert.equal(swapped[11].x, 12); assert.equal(swapped[12].x, 11); assert.equal(swapped[13].x, 13);
  assert.equal(pose[11].x, 11);
});

test('the model referees a lasting MediaPipe/network disagreement, never both at once', () => {
  const policy = createSensePolicy();
  const out = (back, swap = .1) => ({ back, swap: Array.from({ length: 16 }, () => swap) });
  const FRONT = [0, 0, -1], BACK = [0, 0, 1], SIDE = [1, 0, .1];
  let t = 0; const step = (...args) => policy.update(...args, t += 1 / 30);
  // MediaPipe sees a back, the network a front: after 0.3 s the model sides with MediaPipe.
  let d = step(out(.95), FRONT, -1);
  assert.equal(d.referee, false, 'a disagreement of one frame is flicker');
  assert.equal(d.flipped, false, 'the network is not turned round for a flicker');
  for (let i = 0; i < 10; i++) d = step(out(.95), FRONT, -1);
  assert.equal(d.referee, true); assert.equal(d.flipped, true); assert.equal(d.swapped.some(Boolean), false);
  // A brief agreement keeps the referee; a lasting one hands back.
  d = step(out(.95), FRONT, 1); assert.equal(d.flipped, true);
  for (let i = 0; i < 10; i++) d = step(out(.95, .9), FRONT, 1);
  assert.equal(d.referee, false); assert.equal(d.flipped, false, 'both see a front: the model is outvoted');
  assert.equal(d.swapped.some(Boolean), false);
  // MediaPipe mirrored a back into a front; the network saw the back: labels exchanged, network kept.
  for (let i = 0; i < 11; i++) d = step(out(.95, .9), BACK, 1);
  assert.equal(d.flipped, false); assert.equal(d.swapped.every(Boolean), true);
  d = step(out(.95, .5), BACK, 1); assert.equal(d.swapped.every(Boolean), true, 'held inside the swap band');
  // A side-on network keeps the last reading; an unsure model changes nothing.
  policy.reset(); t = 0;
  for (let i = 0; i < 11; i++) d = step(out(.9), FRONT, -1);
  assert.equal(d.flipped, true);
  assert.equal(step(out(.7), SIDE, -1).flipped, true, 'side-on network: last reading kept');
  d = step(out(.55), FRONT, -1);
  assert.equal(d.back, null); assert.equal(d.flipped, false);
});

test('a person who turned away (face found steadily, then gone) lets the model overrule both', () => {
  const out = (back, swap = .9) => ({ back, swap: Array.from({ length: 16 }, () => swap) });
  const FRONT = [0, 0, -1];
  let t = 0, policy = createSensePolicy();
  const step = (o, face) => policy.update(o, FRONT, 1, t += 1 / 30, face);
  for (let i = 0; i < 40; i++) step(out(.1, .1), true);                     // facing the camera, face found
  let d;
  for (let i = 0; i < 6; i++) d = step(out(.95, .1), false);
  assert.equal(d.referee, false, 'not before the face has been gone for 0.3 s');
  for (let i = 0; i < 6; i++) d = step(out(.95, .1), false);
  assert.equal(d.away, true); assert.equal(d.referee, true); assert.equal(d.flipped, true);
  assert.equal(d.swapped.every(Boolean), true, 'MediaPipe still reads a front: every pair goes to the decided side');
  for (let i = 0; i < 6; i++) d = step(out(.85, .1), false);
  assert.equal(d.away, true, 'held while the back decision stands'); assert.equal(d.flipped, true);
  for (let i = 0; i < 12; i++) d = step(out(.1, .1), true);
  assert.equal(d.referee, false, 'face back: the others agree again');
  // Never seen a face on this track: the detector may simply not work here.
  policy = createSensePolicy(); t = 0;
  for (let i = 0; i < 30; i++) d = step(out(.97), false);
  assert.equal(d.away, false); assert.equal(d.referee, false); assert.equal(d.flipped, false);
  // A small or blurred face the detector finds on one frame in four: losing it says nothing.
  policy = createSensePolicy(); t = 0;
  for (let i = 0; i < 60; i++) step(out(.97), i % 4 === 0);
  for (let i = 0; i < 20; i++) d = step(out(.97), false);
  assert.equal(d.away, false); assert.equal(d.referee, false); assert.equal(d.flipped, false);
});

test('flickering MediaPipe labels are put back at once when the model sides with the network', () => {
  const policy = createSensePolicy();
  const out = (back, swap) => ({ back, swap: Array.from({ length: 16 }, (_, i) => i === 5 ? swap : .1) });
  let t = 0; const step = (...args) => policy.update(...args, t += 1 / 30);
  for (let i = 0; i < 5; i++) step(out(.05, .1), [0, 0, -1], 1);         // a front, all agree
  let d = step(out(.05, .9), [0, 0, -1], -1);                              // one frame of mirrored shoulders
  assert.equal(d.flicker, true); assert.equal(d.swapped[5], true); assert.equal(d.flipped, false);
  d = step(out(.05, .1), [0, 0, -1], 1);
  assert.equal(d.swapped[5], false, 'labels back to normal');
  d = step(out(.9, .9), [0, 0, -1], -1);
  assert.equal(d.swapped.some(Boolean), false, 'a model siding with MediaPipe waits for the referee hold');
});

test('label facing reads MediaPipe left/right, and gives up side-on', () => {
  const pose = Array.from({ length: 33 }, () => ({ x: .5, y: .5, visibility: 1 }));
  const set = (i, x, y) => Object.assign(pose[i], { x, y });
  set(11, .6, .3); set(12, .4, .3); set(23, .57, .6); set(24, .43, .6); set(0, .5, .2);
  assert.equal(labelFacing(pose, 1000, 1000), 1, 'left shoulder on the image right: a front');
  set(11, .4, .3); set(12, .6, .3); set(23, .43, .6); set(24, .57, .6);
  assert.equal(labelFacing(pose, 1000, 1000), -1);
  set(11, .51, .3); set(12, .49, .3);
  assert.equal(labelFacing(pose, 1000, 1000), null, 'side-on');
  set(11, .6, .3); pose[12].visibility = 0;
  assert.equal(labelFacing(pose, 1000, 1000), null, 'a shoulder not seen');
});

test('frame normalization survives a side view (shoulders overlapping)', () => {
  const pose = Array.from({ length: 33 }, () => ({ x: .5, y: .5 }));
  Object.assign(pose[11], { x: .5, y: .4 }); Object.assign(pose[12], { x: .5, y: .4 });
  Object.assign(pose[23], { x: .5, y: .7 }); Object.assign(pose[24], { x: .5, y: .7 });
  Object.assign(pose[0], { x: .52, y: .3 });
  const { scale } = frameNormalization(pose, 1000, 1000);
  assert.ok(Math.abs(scale - 300) < 1e-6, String(scale));
});

for (const name of ['body-sense', 'body-lift']) {
const exported = new URL(`${name}.json`, directory);
test(`the exported ${name} model runs in the browser runtime exactly like PyTorch`, { skip: !existsSync(exported) && 'no exported model' }, () => {
  const meta = JSON.parse(readFileSync(exported));
  const weights = readFileSync(new URL(`${name}.bin`, directory));
  const model = parseBodySense(meta, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength));
  const { cases } = JSON.parse(readFileSync(new URL(`./data/${name}-parity.json`, import.meta.url)));
  assert.ok(cases.length >= 3);
  const sense = createBodySense(model);
  for (const c of cases) {
    sense.reset();
    let result;
    for (const frame of c.frames) {
      result = sense.step({ pose: frame.pose, width: c.width, height: c.height, dt: frame.dt, face: frame.face, network: frame.network });
      for (let i = 0; i < model.features; i++) assert.ok(Math.abs(sense._features[i] - frame.features[i]) < 2e-4, `feature ${i}: ${sense._features[i]} vs ${frame.features[i]}`);
    }
    for (let k = 0; k < 3; k++) assert.ok(Math.abs(result.forward[k] - c.expected.forward[k]) < 2e-3);
    assert.ok(Math.abs(result.back - c.expected.back) < 2e-3);
    result.swap.forEach((p, i) => assert.ok(Math.abs(p - c.expected.swap[i]) < 2e-3));
    // Joints come back in normalized image coordinates.
    const { cx, cy, scale } = frameNormalization(c.frames.at(-1).pose, c.width, c.height);
    c.expected.joints.forEach(([x, y], i) => {
      assert.ok(Math.abs(result.joints[i * 2] - (x * scale + cx) / c.width) < 5e-3);
      assert.ok(Math.abs(result.joints[i * 2 + 1] - (y * scale + cy) / c.height) < 5e-3);
    });
    if (c.expected.joints3d) {
      c.expected.joints3d.forEach((p, i) => p.forEach((v, k) => assert.ok(Math.abs(result.joints3d[i * 3 + k] - v) < 5e-3, `joint3d ${i}`)));
      c.expected.sigma3d.forEach((v, i) => assert.ok(Math.abs(result.sigma3d[i] - v) < 5e-3 * Math.max(1, v)));
    }
  }
  const began = performance.now();
  for (let i = 0; i < 300; i++) sense.step({ pose: cases[0].frames[i % 32].pose, width: cases[0].width, height: cases[0].height });
  const perFrame = (performance.now() - began) / 300;
  assert.ok(perFrame < 5, `${perFrame.toFixed(2)} ms per frame`);
});
}

test('lift policy: the 3D skeleton decides the side, the network is turned round, MediaPipe follows', () => {
  const policy = createLiftPolicy();
  // A skeleton with its back to the camera: left shoulder on the image left (x<0), chest facing +Z.
  const skeleton = back => {
    const j = new Float32Array(99);
    const set = (i, x, y, z) => { j[i * 3] = x; j[i * 3 + 1] = y; j[i * 3 + 2] = z; };
    const s = back ? -1 : 1;              // facing the camera: its left is on the image right (+x)
    set(11, .35 * s, -1, 0); set(12, -.35 * s, -1, 0); set(23, .2 * s, 0, 0); set(24, -.2 * s, 0, 0);
    return j;
  };
  const out = (back, swap = .1) => ({ joints3d: skeleton(back), swap: Array.from({ length: 16 }, () => swap) });
  const FRONT = [0, 0, -1], BACK = [0, 0, 1];
  let d = policy.update(out(false), FRONT, 1);
  assert.equal(d.back, false); assert.equal(d.flipped, false); assert.equal(d.swapped.some(Boolean), false);
  // Turned away; the network still reads a front and MediaPipe mirrored the labels into a front.
  d = policy.update(out(true), FRONT, 1);
  assert.equal(d.back, true); assert.equal(d.flipped, true);
  assert.equal(d.swapped.every(Boolean), true, 'MediaPipe shoulders brought to the back side');
  // MediaPipe reads the back correctly and the network agrees: nothing to change.
  d = policy.update(out(true), BACK, -1);
  assert.equal(d.flipped, false); assert.equal(d.swapped.some(Boolean), false);
  // Side-on (no clear depth): the last reading stands.
  const side = { joints3d: new Float32Array(99), swap: Array(16).fill(.5) };
  side.joints3d.set([0, -1, .35], 11 * 3); side.joints3d.set([0, -1, -.35], 12 * 3); side.joints3d.set([0, 0, .2], 23 * 3); side.joints3d.set([0, 0, -.2], 24 * 3);
  assert.equal(policy.update(side, [1, 0, 0], null).back, true);
  // A face in the pixels overrules a skeleton that reads a back (close-ups), and keeps doing so briefly.
  d = policy.update(out(true), FRONT, 1, 10, true);
  assert.equal(d.back, false); assert.equal(d.flipped, false);
  assert.equal(policy.update(out(true), FRONT, 1, 10.2, false).back, false);
  assert.equal(policy.update(out(true), FRONT, 1, 10.5, false).back, true);
  // Head and shoulders: MediaPipe's labels decide; a skeleton reading the other side is mirrored.
  d = policy.update(out(true), FRONT, 1, 20, false, true);
  assert.equal(d.back, false); assert.equal(d.mirrorSkeleton, true); assert.equal(d.flipped, false);
  d = policy.update(out(true), FRONT, -1, 20.1, false, true);
  assert.equal(d.back, true); assert.equal(d.mirrorSkeleton, false); assert.equal(d.flipped, true);
});
