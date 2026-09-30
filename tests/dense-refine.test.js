import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { createDecoderFromArrays, viewsFromManifest } from '../src/dense/decoder/mhr-decoder.js';
import { refineBodyPose, poseTargets } from '../src/dense/inference/refine-pose.js';

// Real official assets and known forward-generated observations. These tests
// check solver constraints, not whether estimated image joints are ground truth.
const manifest = JSON.parse(readFileSync(new URL('../public/dense/mhr-lod3.json', import.meta.url)));
const bytes = gunzipSync(readFileSync(new URL('../public/dense/mhr-lod3.bin.gz', import.meta.url)));
const decoder = createDecoderFromArrays(manifest, viewsFromManifest(manifest, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
const names = manifest.modelParameterNames;
const camera = { translation: [0, 1, 4], focal: 800, width: 960, height: 720 };
const body = [0,5,6,7,8,9,10,11,12,13,14,15,17,18,20,41,62];
const neutral = () => new Float32Array(204);
const changed = (values) => {
  const parameters = neutral();
  for (const [name, value] of Object.entries(values)) parameters[names.indexOf(name)] = value;
  return parameters;
};
const targetsFrom = (parameters, keys = body) => {
  const points = decoder.decodeSkeleton(parameters).keypoints70;
  return keys.map(keypoint => ({ keypoint,
    x: camera.width / 2 + camera.focal * (points[keypoint * 3] + camera.translation[0]) / (points[keypoint * 3 + 2] + camera.translation[2]),
    y: camera.height / 2 + camera.focal * (points[keypoint * 3 + 1] + camera.translation[1]) / (points[keypoint * 3 + 2] + camera.translation[2]), weight: 1 }));
};
const refine = (parameters, targets, extra = {}) => refineBodyPose({ decoder, mhrParams: parameters, camera, targets, ...extra });

test('known visible elbow motion reduces image error without changing root, fingers or proportions', () => {
  const prior = neutral(), truth = changed({ l_elbow_bend: .3 });
  const targets = targetsFrom(truth), inputSnapshot = structuredClone(targets), cameraSnapshot = structuredClone(camera);
  const result = refine(prior, targets);
  assert.equal(result.accepted, true);
  assert.ok(result.meanErrorPx < result.beforeErrorPx * .5);
  for (let i = 0; i < prior.length; i++) {
    if (i < 6 || i >= 68) assert.equal(result.mhrParams[i], prior[i], names[i]);
    assert.ok(Math.abs(result.mhrParams[i] - prior[i]) <= .450001, names[i]);
  }
  assert.deepEqual(prior, neutral());
  assert.deepEqual(targets, inputSnapshot); assert.deepEqual(camera, cameraSnapshot);
});

test('unobserved opposite arm and legs retain their exact local prior rotations', () => {
  const prior = neutral();
  const targets = targetsFrom(changed({ l_elbow_bend: .3 }), [0,5,6,9,10,7,62]);
  const result = refine(prior, targets);
  assert.equal(result.accepted, true);
  assert.ok(result.meanErrorPx < result.beforeErrorPx * .5);
  names.forEach((name, i) => {
    if (/^r_(clavicle|uparm|elbow)|^[lr]_(upleg|knee|foot)/.test(name)) assert.equal(result.mhrParams[i], prior[i], `hidden chain ${name}`);
  });
});

test('duplicate, nonfinite, zero-weight and off-screen anchors cannot manufacture six observations', () => {
  const prior = neutral(), sample = targetsFrom(prior)[0];
  const sets = [Array.from({ length: 12 }, () => ({ ...sample })),
    targetsFrom(prior).map(p => ({ ...p, weight: 0 })),
    targetsFrom(prior).map(p => ({ ...p, weight: NaN })),
    targetsFrom(prior).map(p => ({ ...p, x: Infinity })),
    targetsFrom(prior).map(p => ({ ...p, y: camera.height + 1 }))];
  for (const targets of sets) {
    const result = refine(prior, targets);
    assert.equal(result.accepted, false); assert.deepEqual(result.mhrParams, prior);
  }
});

test('six visible points without a complete supported body chain do not justify pose adjustment', () => {
  const prior = neutral();
  const result = refine(prior, targetsFrom(changed({ l_elbow_bend: .3 }), [0,1,2,3,4,5]));
  assert.equal(result.accepted, false);
  assert.equal(result.reason, 'no-observed-body-chains');
  assert.deepEqual(result.mhrParams, prior);
});

test('exact prior, zero iterations and zero trust radius are genuine no-ops', () => {
  const prior = neutral();
  for (const [targets, extra] of [[targetsFrom(prior), {}], [targetsFrom(changed({ l_elbow_bend: .3 })), { iterations: 0 }],
    [targetsFrom(changed({ l_elbow_bend: .3 })), { maxAngleChange: 0 }]]) {
    const result = refine(prior, targets, extra);
    assert.equal(result.accepted, false); assert.deepEqual(result.mhrParams, prior);
  }
});

test('large inconsistent observations stay inside the selected trust radius', () => {
  const prior = neutral(), targets = targetsFrom(prior).map(p => ({ ...p,
    x: Math.min(camera.width - 1, Math.max(1, p.x + (p.keypoint % 2 ? 120 : -120))),
    y: Math.min(camera.height - 1, Math.max(1, p.y + (p.keypoint % 3 ? 80 : -80))) }));
  const result = refine(prior, targets, { maxAngleChange: .08, iterations: 8 });
  assert.ok([...result.mhrParams].every(Number.isFinite));
  result.mhrParams.forEach((value, i) => assert.ok(Math.abs(value - prior[i]) <= .080001, names[i]));
});

test('behind-camera geometry and invalid solver controls do not produce a correction', () => {
  const prior = neutral(), targets = targetsFrom(prior);
  const result = refineBodyPose({ decoder, mhrParams: prior, targets, camera: { ...camera, translation: [0,1,-4] } });
  assert.equal(result.accepted, false); assert.deepEqual(result.mhrParams, prior);
  for (const extra of [{ iterations: -1 }, { maxAngleChange: NaN }, { maxAngleChange: 2 }, { bodyHeight: 0 }]) assert.throws(() => refine(prior, targets, extra));
});

test('Pose observation adapter discards occlusion, cropped points and nonfinite confidence', () => {
  const pose = Array.from({ length: 33 }, () => ({ x: .5, y: .5, visibility: .9, presence: .9 }));
  pose[11].visibility = .1; pose[12].x = -0.01; pose[13].y = NaN; pose[14].visibility = NaN;
  pose[15].presence = 0; pose[16].visibility = Infinity;
  const result = poseTargets(pose, camera.width, camera.height);
  assert.ok(result.every(p => Number.isFinite(p.weight) && p.weight > 0));
  for (const keypoint of [5,6,7,8,62]) assert.ok(!result.some(p => p.keypoint === keypoint));
});

test('a warm start from the previous frame keeps the trust region and regularization on the prior', () => {
  const prior = neutral(), truth = changed({ l_elbow_bend: .5, r_elbow_bend: -.3 });
  const targets = targetsFrom(truth);
  const cold = refine(prior, targets, { iterations: 2, maxAngleChange: .8 });
  const previous = changed({ l_elbow_bend: .45, r_elbow_bend: -.25 });
  const warm = refine(prior, targets, { iterations: 2, maxAngleChange: .8, initial: previous });
  assert.ok(warm.meanErrorPx <= cold.meanErrorPx + 1e-6, `${warm.meanErrorPx} vs ${cold.meanErrorPx}`);
  const far = changed({ l_elbow_bend: 2 });
  const clamped = refine(prior, targets, { iterations: 0, maxAngleChange: .3, initial: far });
  assert.ok(Math.abs(clamped.mhrParams[names.indexOf('l_elbow_bend')]) <= .300001, 'warm start is clamped into the trust region');
  for (let i = 0; i < 6; i++) assert.equal(warm.mhrParams[i], prior[i], 'root stays with the prior');
  assert.throws(() => refine(prior, targets, { initial: new Float32Array(10) }), /204/);
});

test('a hidden arm follows 3D skeleton targets when the image has no points for it', () => {
  const prior = neutral(), truth = changed({ l_elbow_bend: .6, l_uparm_rz: .4 });
  // The image sees everything except the left elbow and wrist (hand behind the back).
  const targets = targetsFrom(truth, body.filter(k => k !== 7 && k !== 62));
  const points = decoder.decodeSkeleton(truth).keypoints70;
  const root = [0, 1, 2].map(c => (points[9 * 3 + c] + points[10 * 3 + c]) / 2);
  const targets3d = [5, 6, 7, 62, 9, 10].map(keypoint => ({ keypoint, weight: .5,
    x: points[keypoint * 3] - root[0], y: points[keypoint * 3 + 1] - root[1], z: points[keypoint * 3 + 2] - root[2] }));
  const without = refine(prior, targets), withSkeleton = refine(prior, targets, { targets3d, maxAngleChange: .8, iterations: 6 });
  const error = parameters => {
    const p = decoder.decodeSkeleton(parameters).keypoints70;
    return Math.hypot(p[62 * 3] - points[62 * 3], p[62 * 3 + 1] - points[62 * 3 + 1], p[62 * 3 + 2] - points[62 * 3 + 2]);
  };
  assert.ok(error(withSkeleton.mhrParams) < error(without.mhrParams) * .5, `${error(withSkeleton.mhrParams)} vs ${error(without.mhrParams)}`);
});
