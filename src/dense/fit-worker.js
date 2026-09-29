/** Per-frame MHR surface for the optional body mesh: 1 Euro smoothing of the
 * network predictions (one person track) and a bounded fit of the rig to each
 * frame's stabilized Pose landmarks, warm-started from the previous frame.
 * The network runs slower than the camera; this keeps the surface on the body
 * in between. No ONNX Runtime here, so a busy network never delays a fit.
 *
 *   init                         → ready {faces, allFaces, jointNames, keypointNames} | error
 *   reset {generation}           → start a new person track
 *   prior {generation, mhrParams, shapeParams, camTrans, focal, width, height, timestamp}
 *   pose  {generation, pose, width, height, timestamp} → mesh {generation, timestamp, mesh | null, unchanged}
 */
import { createBodyMeshDecoder } from './decoder/mhr-decoder.js';
import { correctBody } from './inference/correct-body.js';
import { OneEuroVector } from './inference/temporal.js';

const ALL_REGIONS = { body: true, leftArm: true, rightArm: true, leftLeg: true, rightLeg: true };
let decoder = null, generation = 0, prior = null, previous = null, filters = null;
let priorVersion = 0, lastInput = null;

const post = (message, transfer = []) => self.postMessage(message, transfer);

function newTrack() {
  prior = null; previous = null; lastInput = null;
  // Per-field 1 Euro settings of the upstream InstantHMR live demo.
  filters = {
    pose: new OneEuroVector(204, { minCutoff: 1, beta: 2 }),
    shape: new OneEuroVector(45, { minCutoff: 0.3, beta: 0 }),
    camera: new OneEuroVector(3, { minCutoff: 0.6, beta: 2 }),
  };
}

function updatePrior({ mhrParams, shapeParams, camTrans, focal, width, height, timestamp }) {
  if (prior && (prior.width !== width || prior.height !== height)) newTrack();
  // Averaging Euler roots across a flip would sweep through a wrong pose.
  if (prior && [3, 4, 5].some(i => Math.abs(mhrParams[i] - prior.rawRoot[i - 3]) > 1.2)) { filters.pose.reset(); previous = null; }
  const time = timestamp / 1000;
  priorVersion++;
  prior = {
    mhrParams: filters.pose.update(mhrParams, time), shapeParams: filters.shape.update(shapeParams, time),
    camTrans: filters.camera.update(camTrans, time), rawRoot: Array.from(mhrParams.slice(3, 6)), focal, width, height,
  };
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

function fit({ pose, width, height, timestamp }) {
  if (!prior || prior.width !== width || prior.height !== height || !Array.isArray(pose)) return null;
  // A body standing still keeps its surface; nothing to recompute.
  if (movedPx(pose, width, height) < 0.75) return 'unchanged';
  lastInput = { pose, version: priorVersion };
  const began = performance.now();
  const correction = correctBody({ decoder, pose, warm: previous, maxAngleChange: 0.8,
    result: { mhrParams: prior.mhrParams, camTrans: prior.camTrans, focal: prior.focal, sourceWidth: width, sourceHeight: height } });
  previous = { mhrParams: correction.mhrParams, translation: Array.from(correction.camera.translation) };
  const decoded = decoder.decode(correction.mhrParams, prior.shapeParams);
  const { refinement } = correction;
  return {
    timestamp, vertices: decoded.vertices, keypoints70: decoded.keypoints70, skeleton: decoded.skeleton.positions,
    camera: { width, height, focal: correction.camera.focal, translation: previous.translation },
    fit: { accepted: refinement.accepted, reason: refinement.reason, meanErrorPx: refinement.meanErrorPx ?? null,
      normalizedError: refinement.normalizedError ?? null, anchors: refinement.anchorCount ?? 0 },
    fitMs: performance.now() - began,
  };
}

async function init() {
  decoder = await createBodyMeshDecoder({ assetUrl: new URL('/dense/mhr-lod3.json', self.location.href).href });
  newTrack();
  // Lines are drawn for the body and limbs; head and hands (drawn by their own
  // models) still hide what is behind them, e.g. the neck behind the chin.
  const faces = decoder.visibleSurface.select(ALL_REGIONS).slice(), allFaces = decoder.faces.slice();
  post({ type: 'ready', faces, allFaces, jointNames: decoder.metadata.jointNames, keypointNames: decoder.metadata.keypointNames },
    [faces.buffer, allFaces.buffer]);
}

// The controller terminates this worker; there is nothing to release first.
self.onmessage = ({ data }) => {
  if (data.type === 'init') init().catch(error => post({ type: 'error', message: error?.message ?? String(error) }));
  else if (data.type === 'reset') { generation = data.generation; newTrack(); }
  else if (data.type === 'prior') { if (decoder && data.generation === generation) updatePrior(data); }
  else if (data.type === 'pose') {
    let mesh = null, error = null;
    try { if (decoder && data.generation === generation) mesh = fit(data); }
    catch (failure) { error = failure?.message ?? String(failure); previous = null; lastInput = null; }
    const unchanged = mesh === 'unchanged';
    if (unchanged) mesh = null;
    post({ type: 'mesh', generation: data.generation, timestamp: data.timestamp, mesh, unchanged, error },
      mesh ? [mesh.vertices.buffer, mesh.keypoints70.buffer, mesh.skeleton.buffer] : []);
  }
};
