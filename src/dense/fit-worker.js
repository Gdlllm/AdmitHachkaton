/** Per-frame MHR surface for the optional body mesh: 1 Euro smoothing of the
 * network predictions (one person track) and a bounded fit of the rig to each
 * frame's stabilized Pose landmarks, warm-started from the previous frame.
 * The network runs slower than the camera; this keeps the surface on the body
 * in between. No ONNX Runtime here, so a busy network never delays a fit.
 *
 * Body sense (optional, ~0.6 MB): a small temporal model that decides which
 * way the torso faces and which MediaPipe left/right labels are swapped. The
 * fit then uses corrected labels, and reads the network's pose the other way
 * round (front ↔ back) when the model is sure the network got it backwards.
 *
 *   init                         → ready {faces, allFaces, jointNames, keypointNames, sense} | error
 *   reset {generation}           → start a new person track
 *   prior {generation, mhrParams, shapeParams, camTrans, focal, width, height, timestamp}
 *   pose  {generation, pose, width, height, timestamp, face} → mesh {generation, timestamp, mesh | null, unchanged}
 */
import { createBodyMeshDecoder } from './decoder/mhr-decoder.js';
import { correctBody } from './inference/correct-body.js';
import { OneEuroVector } from './inference/temporal.js';
import { createBodySense, labelFacing, parseBodySense, swapPairs, torsoForward, NET_JOINT_IDS, POSE_PAIRS } from './inference/body-sense.js';
import { createSensePolicy } from './inference/sense-policy.js';
import { createLiftPolicy } from './inference/lift-policy.js';
import { POSE_TO_MHR } from './inference/refine-pose.js';
import { flipFrontBack, rootMatrix, rootParams } from './inference/mhr-flip.js';
import { fetchAsset } from '../shared/assets.js';

const ALL_REGIONS = { body: true, leftArm: true, rightArm: true, leftLeg: true, rightLeg: true };
const TORSO_KEYPOINTS = [5, 6, 9, 10];   // MHR left/right shoulder, left/right hip
// A fit this far off the landmarks is not shown (controller MAX_FIT_ERROR) and
// must not warm-start the next one: a few solver steps from a wrong pose can
// stay wrong for the rest of the track.
const MAX_WARM_ERROR = 0.12;
// Lift mode: the model's 3D skeleton pulls every body joint the fit can move
// (hidden ones too), weighted by the model's own sigma; visible 2D points still
// set the in-image position. Body shape is frozen after the first network runs.
const LIFT_WEIGHT = 0.5, LIFT_SIGMA = 0.12, SHAPE_RUNS = 20;
const SCALES = [136, 204];               // MHR bone-length scales, frozen with the shape
const ROOT_TAU = 0.08;                   // s, root rotation smoothing (SLERP)

// Rotation matrix ↔ unit quaternion [w, x, y, z].
function toQuaternion(m) {
  const t = m[0][0] + m[1][1] + m[2][2];
  let q;
  if (t > 0) { const s = Math.sqrt(t + 1) * 2; q = [s / 4, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]; }
  else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) { const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2; q = [(m[2][1] - m[1][2]) / s, s / 4, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]; }
  else if (m[1][1] > m[2][2]) { const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2; q = [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, s / 4, (m[1][2] + m[2][1]) / s]; }
  else { const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2; q = [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, s / 4]; }
  const n = Math.hypot(...q); return q.map(v => v / n);
}
function toMatrix([w, x, y, z]) {
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
}
function slerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if (d < 0) { b = b.map(v => -v); d = -d; }                // the same rotation, nearer hemisphere
  if (d > .9995) { const q = a.map((v, i) => v + t * (b[i] - v)); const n = Math.hypot(...q); return q.map(v => v / n); }
  const th = Math.acos(d), s = Math.sin(th);
  return a.map((v, i) => (Math.sin((1 - t) * th) * v + Math.sin(t * th) * b[i]) / s);
}
const rootQuaternion = p => toQuaternion(rootMatrix(p[3], p[4], p[5]));
const angleBetween = (a, b) => 2 * Math.acos(Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])));
const NET_MHR = NET_JOINT_IDS.map(id => POSE_TO_MHR.find(([p]) => p === id)[1]);
const BODY_TARGETS = POSE_TO_MHR.filter(([p]) => p >= 11);
let decoder = null, generation = 0, prior = null, previous = null, filters = null;
let priorVersion = 0, lastInput = null;
let sense = null, policy = null, track = null, lift = false, drawMesh = true;

const post = (message, transfer = []) => self.postMessage(message, transfer);

function newTrack() {
  prior = null; previous = null; lastInput = null;
  // Per-field 1 Euro settings of the upstream InstantHMR live demo.
  filters = {
    pose: new OneEuroVector(204, { minCutoff: 1, beta: 2 }),
    shape: new OneEuroVector(45, { minCutoff: 0.3, beta: 0 }),
    camera: new OneEuroVector(3, { minCutoff: 0.6, beta: 2 }),
  };
  sense?.reset(); policy?.reset();
  track = { network: null, networkJoints: null, shapeRuns: 0, rootQ: null, priorTime: null, lastTimestamp: null, decision: { swapped: new Array(16).fill(false), flipped: false }, output: null };
}

function updatePrior({ mhrParams, shapeParams, camTrans, focal, width, height, timestamp }) {
  if (prior && (prior.width !== width || prior.height !== height)) newTrack();
  const raw = mhrParams;
  if (lift && track.rootQ) {
    // A network run that read the body the other way round is turned back to the
    // current side before smoothing; the skeleton decides the side afterwards.
    const q = rootQuaternion(mhrParams);
    if (angleBetween(q, track.rootQ) > 2.1) {
      const other = flipFrontBack(mhrParams);
      if (angleBetween(rootQuaternion(other), track.rootQ) < angleBetween(q, track.rootQ)) mhrParams = other;
    }
  } else if (prior && [3, 4, 5].some(i => Math.abs(mhrParams[i] - prior.rawRoot[i - 3]) > 1.2)) {
    // Averaging Euler roots across a flip would sweep through a wrong pose.
    filters.pose.reset(); previous = null;
  }
  const time = timestamp / 1000;
  priorVersion++;
  // Lift mode: the body's proportions settle once instead of breathing with every run.
  track.shapeRuns++;
  const shape = lift && prior && track.shapeRuns > SHAPE_RUNS ? prior.shapeParams : filters.shape.update(shapeParams, time);
  let pose = filters.pose.update(mhrParams, time);
  if (lift) {
    pose = Float32Array.from(pose);
    // Root rotation: SLERP on the rotation itself, not per Euler angle.
    const q = rootQuaternion(mhrParams), dt = track.priorTime === null ? 1 : Math.max(0, time - track.priorTime);
    track.rootQ = track.rootQ ? slerp(track.rootQ, q, 1 - Math.exp(-dt / ROOT_TAU)) : q;
    pose.set(rootParams(toMatrix(track.rootQ)), 3);
    if (prior && track.shapeRuns > SHAPE_RUNS) pose.set(prior.mhrParams.subarray(SCALES[0], SCALES[1]), SCALES[0]);
  }
  track.priorTime = time;
  prior = {
    mhrParams: pose, shapeParams: shape,
    camTrans: filters.camera.update(camTrans, time), rawRoot: Array.from(mhrParams.slice(3, 6)), focal, width, height,
  };
  if (sense) {
    // The network's own body direction (unfiltered, as the model was trained).
    // The model was trained on the network's raw readings (flips included); the flip decision
    // compares the skeleton with the pose the fit starts from.
    const torso = params => { const k = decoder.decodeKeypoints(params, TORSO_KEYPOINTS); return torsoForward(TORSO_KEYPOINTS.flatMap(i => [k[i * 3], k[i * 3 + 1], k[i * 3 + 2]])); };
    track.networkRaw = torso(raw);
    track.network = lift ? torso(prior.mhrParams) : track.networkRaw;
    if (lift) {
      const all = decoder.decodeKeypoints(raw, NET_MHR);
      track.networkJoints = Float32Array.from(NET_MHR.flatMap(i => [all[i * 3], all[i * 3 + 1], all[i * 3 + 2]]));
    }
  }
}

// Largest movement of a confident landmark since the last fit, in pixels.
function movedPx(pose, width, height) {
  if (!lastInput || lastInput.version !== priorVersion) return Infinity;
  let moved = 0;
  for (let i = 0; i < pose.length; i++) {
    const a = pose[i], b = lastInput.pose[i];
    if (Math.min(a.visibility ?? 0, a.presence ?? 1) < 0.5) continue;
    moved = Math.max(moved, Math.abs(a.x - b.x) * width, Math.abs(a.y - b.y) * height);
  }
  return moved;
}

// Runs every frame, fit or not, so the model's memory keeps the camera's pace.
function senseFrame({ pose, width, height, timestamp, face }) {
  const dt = track.lastTimestamp === null ? 1 / 30 : Math.max(0, timestamp - track.lastTimestamp) / 1000;
  track.lastTimestamp = timestamp;
  const network = track.networkRaw ? { forward: track.networkRaw, joints: track.networkJoints } : null;
  const output = sense.step({ pose, width, height, dt, face: face?.forward ? face : null, network });
  const labels = labelFacing(pose, width, height);
  const decision = lift ? policy.update(output, track.network, labels, timestamp / 1000, Boolean(face?.forward), closeUp(pose))
    : policy.update(output, track.network, labels, timestamp / 1000, Boolean(face?.forward));
  const changed = decision.flipped !== track.decision.flipped || decision.swapped.some((s, i) => s !== track.decision.swapped[i]);
  // The warm start belongs to the old reading of the network's pose.
  if (decision.flipped !== track.decision.flipped && previous) previous = { ...previous, mhrParams: flipFrontBack(previous.mhrParams) };
  track.decision = decision; track.output = output;
  return changed;
}

function fit(message) {
  const { width, height, timestamp } = message;
  let pose = message.pose;
  if (!Array.isArray(pose)) return null;
  if (sense && pose.length >= 33 && senseFrame(message)) lastInput = null;
  // Skeleton only: the model's answer for this frame, no surface fit.
  if (!drawMesh) return skeletonOnly(timestamp);
  if (!prior || prior.width !== width || prior.height !== height) return null;
  const decision = track.decision;
  // A body standing still keeps its surface; nothing to recompute.
  if (movedPx(pose, width, height) < 0.75) return 'unchanged';
  lastInput = { pose, version: priorVersion };
  const began = performance.now();
  if (sense && decision.swapped.some(Boolean)) pose = swapPairs(pose, decision.swapped);
  const mhrParams = sense && decision.flipped ? flipFrontBack(prior.mhrParams) : prior.mhrParams;
  const targets3d = lift && track.output?.joints3d ? skeletonTargets(mhrParams, track.output, decision.mirrorSkeleton) : [];
  const correction = correctBody({ decoder, pose, warm: previous, maxAngleChange: 0.8, targets3d,
    result: { mhrParams, camTrans: prior.camTrans, focal: prior.focal, sourceWidth: width, sourceHeight: height } });
  const { refinement } = correction;
  const fitted = { mhrParams: correction.mhrParams, translation: Array.from(correction.camera.translation) };
  previous = refinement.normalizedError > MAX_WARM_ERROR ? null : fitted;
  const decoded = decoder.decode(correction.mhrParams, prior.shapeParams);
  const out = track.output;
  return {
    timestamp, vertices: decoded.vertices, keypoints70: decoded.keypoints70, skeleton: decoded.skeleton.positions,
    camera: { width, height, focal: correction.camera.focal, translation: fitted.translation },
    fit: { accepted: refinement.accepted, reason: refinement.reason, meanErrorPx: refinement.meanErrorPx ?? null,
      normalizedError: refinement.normalizedError ?? null, anchors: refinement.anchorCount ?? 0 },
    sense: sense && out ? { mode: lift ? 'lift' : 'referee',
      back: lift && decision.back !== null ? (decision.back ? 1 : 0) : out.back, forward: (lift && decision.forward) || out.forward,
      skeleton3d: out.joints3d ?? null, sigma3d: out.sigma3d ?? null, mirrorSkeleton: decision.mirrorSkeleton ?? false, network: track.network, flipped: decision.flipped,
      swapped: decision.swapped.filter(Boolean).length, swappedPairs: decision.swapped, referee: decision.referee, flicker: decision.flicker, away: decision.away } : null,
    fitMs: performance.now() - began,
  };
}

// Head and shoulders only: both hips out of the picture.
const closeUp = pose => [23, 24].every(i => !(pose[i]?.x >= 0 && pose[i]?.x <= 1 && pose[i]?.y >= 0 && pose[i]?.y <= 1));

/** Left/right exchanged for 33 points of `size` values each. */
function relabel(values, size) {
  const out = Float32Array.from(values);
  for (const [a, b] of POSE_PAIRS) for (let c = 0; c < size; c++) { out[a * size + c] = values[b * size + c]; out[b * size + c] = values[a * size + c]; }
  return out;
}

function skeletonOnly(timestamp) {
  const out = track.output;
  if (!out) return null;
  const decision = track.decision, mirror = Boolean(decision.mirrorSkeleton);
  // Mirrored reading (the other side): depth negated, left and right relabelled.
  let joints3d = out.joints3d ?? null;
  if (joints3d && mirror) { joints3d = relabel(joints3d, 3); for (let i = 0; i < 33; i++) joints3d[i * 3 + 2] *= -1; }
  return {
    timestamp, skeletonOnly: true, fit: null,
    sense: { mode: lift ? 'lift' : 'referee', back: lift && decision.back !== null ? (decision.back ? 1 : 0) : out.back,
      forward: (lift && decision.forward) || out.forward, flipped: decision.flipped,
      swapped: decision.swapped.filter(Boolean).length, swappedPairs: decision.swapped,
      joints2d: mirror ? relabel(out.joints, 2) : Float32Array.from(out.joints),
      skeleton3d: joints3d, sigma3d: out.sigma3d ? (mirror ? relabel(out.sigma3d, 1) : out.sigma3d) : null, mirrorSkeleton: mirror },
  };
}

/** The model's skeleton as fit targets, in this body's size (MHR torso length).
 * mirror: use its depth mirror, left and right relabelled (the other side). */
function skeletonTargets(mhrParams, output, mirror = false) {
  const k = decoder.decodeKeypoints(mhrParams, TORSO_KEYPOINTS);
  const mid = (a, b) => [0, 1, 2].map(c => (k[a * 3 + c] + k[b * 3 + c]) / 2);
  const sh = mid(5, 6), hip = mid(9, 10);
  const torso = Math.hypot(sh[0] - hip[0], sh[1] - hip[1], sh[2] - hip[2]);
  if (!(torso > .1)) return [];
  const other = i => { const pair = POSE_PAIRS.find(([a, b]) => a === i || b === i); return pair ? (pair[0] === i ? pair[1] : pair[0]) : i; };
  return BODY_TARGETS.map(([p, keypoint]) => {
    const q = mirror ? other(p) : p, sz = mirror ? -1 : 1;
    const sigma = output.sigma3d[q];
    const weight = LIFT_WEIGHT * Math.max(.05, Math.min(1, (LIFT_SIGMA / Math.max(1e-3, sigma)) ** 2));
    return { keypoint, x: output.joints3d[q * 3] * torso, y: output.joints3d[q * 3 + 1] * torso, z: sz * output.joints3d[q * 3 + 2] * torso, weight };
  });
}

async function loadModel(name) {
  const response = await fetch(new URL(`/dense/${name}.json`, self.location.href));
  if (!response.ok) throw new Error(`${name} manifest: HTTP ${response.status}`);
  const manifest = await response.json();
  const bytes = await fetchAsset(`/dense/${name}.bin`, { bytes: manifest.bytes, sha256: manifest.sha256 });
  return parseBodySense(manifest, bytes);
}

async function init({ sense: wanted = true, lift: wantLift = true, mesh: wantMesh = true } = {}) {
  drawMesh = wantMesh;
  decoder = await createBodyMeshDecoder({ assetUrl: new URL('/dense/mhr-lod3.json', self.location.href).href });
  // The surface works without it, as before; the model only corrects it.
  let senseError = wanted ? null : 'off';
  if (wanted) {
    // The 3D-skeleton model when available (and wanted); otherwise the front/back referee.
    let model = null;
    if (wantLift) { try { model = await loadModel('body-lift'); } catch { model = null; } }
    try {
      model ??= await loadModel('body-sense');
      lift = model.lift === true;
      sense = createBodySense(model); policy = lift ? createLiftPolicy() : createSensePolicy();
    } catch (error) { sense = null; policy = null; lift = false; senseError = error?.message ?? String(error); }
  }
  newTrack();
  // Lines are drawn for the body and limbs; head and hands (drawn by their own
  // models) still hide what is behind them, e.g. the neck behind the chin.
  const faces = decoder.visibleSurface.select(ALL_REGIONS).slice(), allFaces = decoder.faces.slice();
  // Per region, so a mistake can light up the leg or arm it concerns.
  const regionFaces = Object.fromEntries(Object.keys(ALL_REGIONS).map(region => [region, decoder.visibleSurface.select({ [region]: true }).slice()]));
  post({ type: 'ready', faces, allFaces, regionFaces, jointNames: decoder.metadata.jointNames, keypointNames: decoder.metadata.keypointNames,
    sense: sense ? (lift ? 'lift' : 'ready') : senseError }, [faces.buffer, allFaces.buffer, ...Object.values(regionFaces).map(f => f.buffer)]);
}

// The controller terminates this worker; there is nothing to release first.
self.onmessage = ({ data }) => {
  if (data.type === 'init') init(data).catch(error => post({ type: 'error', message: error?.message ?? String(error) }));
  else if (data.type === 'reset') { generation = data.generation; newTrack(); }
  else if (data.type === 'prior') { if (decoder && data.generation === generation) updatePrior(data); }
  else if (data.type === 'pose') {
    let mesh = null, error = null;
    try { if (decoder && data.generation === generation) mesh = fit(data); }
    catch (failure) { error = failure?.message ?? String(failure); previous = null; lastInput = null; }
    const unchanged = mesh === 'unchanged';
    if (unchanged) mesh = null;
    post({ type: 'mesh', generation: data.generation, timestamp: data.timestamp, mesh, unchanged, error },
      mesh?.vertices ? [mesh.vertices.buffer, mesh.keypoints70.buffer, mesh.skeleton.buffer] : []);
  }
};
