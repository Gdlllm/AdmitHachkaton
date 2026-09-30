/** Body kinematics for one frame: joint angles in 3D, which way the body faces
 * and how it sits in the picture. Scenario-agnostic: games, trainers and the
 * "wrong movement" hints all read these numbers.
 *
 * Angles come from MediaPipe's world landmarks (metres, hip-centred, camera
 * axes X right / Y down / Z away from the camera), so a squat reads the same
 * facing the camera or side-on; flat image angles shrink when a thigh points
 * at the lens. Checked on the squat fixture: 162-167° standing, 81-88° at the
 * bottom, left and right within a few degrees. The dense surface is not used
 * for angles: its knee lags the body by ~1 s and never fully straightens.
 *
 * Every angle has a confidence (the lowest image visibility of its joints, 0
 * where the stabilizer hides a point); below 0.5 the value is a guess.
 * The camera is assumed upright unless calibrate() saw the person stand straight.
 */
export const JOINTS = Object.freeze({
  nose: 0, leftEar: 7, rightEar: 8, leftShoulder: 11, rightShoulder: 12, leftElbow: 13, rightElbow: 14,
  leftWrist: 15, rightWrist: 16, leftHip: 23, rightHip: 24, leftKnee: 25, rightKnee: 26,
  leftAnkle: 27, rightAnkle: 28, leftHeel: 29, rightHeel: 30, leftFoot: 31, rightFoot: 32,
});
// Named body parts for hints and highlights → the Pose joints they consist of.
export const BODY_PARTS = Object.freeze({
  leftKnee: [23, 25, 27], rightKnee: [24, 26, 28], leftHip: [11, 23, 25], rightHip: [12, 24, 26],
  leftArm: [11, 13, 15], rightArm: [12, 14, 16], leftElbow: [11, 13, 15], rightElbow: [12, 14, 16],
  leftLeg: [23, 25, 27, 29, 31], rightLeg: [24, 26, 28, 30, 32], leftFoot: [27, 29, 31], rightFoot: [28, 30, 32],
  torso: [11, 12, 23, 24], back: [11, 12, 23, 24], shoulders: [11, 12], hips: [23, 24],
});
const FULL_BODY = [11, 12, 23, 24, 25, 26, 27, 28];

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => Math.hypot(a[0], a[1], a[2]);
const unit = a => { const n = norm(a); return n > 1e-9 ? scale(a, 1 / n) : null; };
const mid = (a, b) => scale(add(a, b), .5);
const deg = r => r * 180 / Math.PI;
const angleBetween = (u, v) => {
  const n = norm(u) * norm(v);
  return n > 1e-12 ? deg(Math.acos(Math.max(-1, Math.min(1, dot(u, v) / n)))) : null;
};
const round = v => v === null || !Number.isFinite(v) ? null : Math.round(v * 10) / 10;

/** Interior angle at b of the chain a-b-c, degrees (180 = straight). */
export function jointAngle(a, b, c) { return angleBetween(sub(a, b), sub(c, b)); }

const finite3 = p => p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z);
const PAIRS = [[1, 4], [2, 5], [3, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22],
  [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]];
function swapPairs(points, swapped) {
  const out = points.slice();
  PAIRS.forEach(([a, b], i) => { if (swapped[i]) { out[a] = points[b]; out[b] = points[a]; } });
  return out;
}
const seen = p => !p ? 0 : p.drawConfidence === 0 ? 0 : Math.max(0, Math.min(1, Math.min(p.visibility ?? 1, p.presence ?? 1)));

/** frame: a capture frame (pose.landmarks + pose.worldLandmarks, optional face,
 * surface.sense). options.up: the "up" direction in camera axes (default
 * [0,-1,0]). Returns null without a usable 3D pose. */
export function measureBody(frame, { up = [0, -1, 0] } = {}) {
  let image = frame?.pose?.landmarks, world = frame?.pose?.worldLandmarks;
  if (!Array.isArray(image) || image.length < 33 || !Array.isArray(world) || world.length < 33 || !world.slice(0, 33).every(finite3)) return null;
  // Left/right pairs the body-sense model found swapped (e.g. a back read as a front).
  const swaps = frame?.surface?.active ? frame.surface.sense?.swappedPairs : null;
  if (Array.isArray(swaps) && swaps.some(Boolean)) { image = swapPairs(image, swaps); world = swapPairs(world, swaps); }
  const W = world.map(p => [p.x, p.y, p.z]);
  const C = image.map(seen);
  // Joints in the picture but hidden (a hand behind the back, a leg behind the
  // other): the body-sense skeleton knows where they are. Joints out of the
  // picture stay unknown, so framing hints still ask to step back.
  const skeleton = frame?.surface?.active ? frame.surface.sense?.skeleton3d : null;
  if (skeleton?.length >= 99) {
    const torso = Math.hypot(...[0, 1, 2].map(k => (W[11][k] + W[12][k]) / 2 - (W[23][k] + W[24][k]) / 2));
    const root = [0, 1, 2].map(k => (W[23][k] + W[24][k]) / 2);
    for (let i = 11; i < 33; i++) {
      const p = image[i], inside = p && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;
      if (!inside || C[i] >= .5 || !(torso > .05)) continue;
      W[i] = [0, 1, 2].map(k => root[k] + skeleton[i * 3 + k] * torso);
      C[i] = .6;                                                 // inferred: usable, never "sure"
    }
  }
  const conf = (...i) => Math.min(...i.map(k => C[k]));
  const angle = (a, b, c) => ({ value: round(jointAngle(W[a], W[b], W[c])), confidence: conf(a, b, c) });
  const pair = (left, right) => ({ left: left(), right: right() });
  const upUnit = unit(up) ?? [0, -1, 0];

  // Body axes: left (from the right hip/shoulder to the left), up (gravity), forward (left × up).
  const hipMid = mid(W[23], W[24]), shoulderMid = mid(W[11], W[12]);
  const across = add(sub(W[23], W[24]), sub(W[11], W[12]));
  const leftAxis = unit(sub(across, scale(upUnit, dot(across, upUnit))));
  const forward = leftAxis ? unit(cross(leftAxis, upUnit)) : null;
  const trunk = sub(shoulderMid, hipMid);
  const trunkConf = conf(11, 12, 23, 24);
  const lean = forward ? {
    forward: round(deg(Math.atan2(dot(trunk, forward), dot(trunk, upUnit)))),
    side: round(deg(Math.atan2(dot(trunk, leftAxis), dot(trunk, upUnit)))),
    total: round(angleBetween(trunk, upUnit)), confidence: trunkConf,
  } : null;

  // Knee drift in the frontal plane, degrees from the hip-ankle line: positive
  // outwards, negative inwards ("valgus", knees caving in).
  const kneeDrift = (hip, knee, ankle, outward) => {
    if (!leftAxis) return { value: null, confidence: 0 };
    const line = unit(sub(W[ankle], W[hip])), thigh = sub(W[knee], W[hip]);
    if (!line) return { value: null, confidence: 0 };
    const off = sub(thigh, scale(line, dot(thigh, line)));
    const side = dot(off, scale(leftAxis, outward)), along = dot(thigh, line);
    return { value: round(deg(Math.atan2(side, Math.max(1e-6, along)))), confidence: conf(hip, knee, ankle) };
  };
  const spread = (a, b) => leftAxis ? Math.abs(dot(sub(W[a], W[b]), leftAxis)) : null;
  const hipWidth = spread(23, 24), kneeWidth = spread(25, 26), ankleWidth = spread(27, 28);
  const height = (i, from) => dot(sub(W[i], W[from]), upUnit);

  const angles = {
    knee: pair(() => angle(23, 25, 27), () => angle(24, 26, 28)),
    hip: pair(() => angle(11, 23, 25), () => angle(12, 24, 26)),
    elbow: pair(() => angle(11, 13, 15), () => angle(12, 14, 16)),
    // Arm raise against gravity: 0 hanging, 90 horizontal, 180 straight
    // overhead. Needs only shoulder and elbow, so it works in a close-up.
    shoulder: pair(() => ({ value: round(angleBetween(sub(W[13], W[11]), scale(upUnit, -1))), confidence: conf(11, 13) }),
      () => ({ value: round(angleBetween(sub(W[14], W[12]), scale(upUnit, -1))), confidence: conf(12, 14) })),
    kneeDrift: pair(() => kneeDrift(23, 25, 27, 1), () => kneeDrift(24, 26, 28, -1)),
    trunk: lean,
  };
  const measures = {
    // Metres; positive when the hip is above the knee (parallel squat ≈ 0).
    hipAboveKnee: pair(() => ({ value: round(100 * height(23, 25)) / 100, confidence: conf(23, 25) }),
      () => ({ value: round(100 * height(24, 26)) / 100, confidence: conf(24, 26) })),
    // Wrist height over the nose, metres (hands above the head > 0).
    wristAboveHead: pair(() => ({ value: round(100 * height(15, 0)) / 100, confidence: conf(15, 0) }),
      () => ({ value: round(100 * height(16, 0)) / 100, confidence: conf(16, 0) })),
    kneeToAnkleWidth: kneeWidth !== null && ankleWidth > 0.05 ? round(100 * kneeWidth / ankleWidth) / 100 : null,
    stanceToHipWidth: ankleWidth !== null && hipWidth > 0.05 ? round(100 * ankleWidth / hipWidth) / 100 : null,
    // Left minus right, degrees.
    kneeAsymmetry: angles.knee.left.value !== null && angles.knee.right.value !== null ? round(angles.knee.left.value - angles.knee.right.value) : null,
    // How far (metres) the knees are ahead of the toes along the body's forward
    // axis: both knees (squat), and the front leg only (lunge).
    kneePastToe: kneeAhead(W, C, forward, [25, 31], [26, 32]),
    frontKneePastToe: frontKneeAhead(W, C, forward),
  };
  return { angles, measures, facing: facingOf(frame, forward), framing: framingOf(image, C), axes: { up: upUnit, left: leftAxis, forward } };
}

/** Which way the body faces. Yaw 0: towards the camera, ±90 side-on, 180 back.
 * With the body surface on, the body-sense model's answer is used (it catches
 * MediaPipe reading a back as a front); otherwise MediaPipe's own 3D pose,
 * which is reliable except for backs close to the camera. */
function facingOf(frame, forward) {
  const sense = frame?.surface?.sense;
  let vector = null, source = null;
  if (Array.isArray(sense?.forward) && sense.forward.length === 3 && frame?.surface?.active) { vector = sense.forward; source = 'body-sense'; }
  else if (forward) { vector = forward; source = 'pose'; }
  if (!vector) return { view: null, yaw: null, source: null, face: Boolean(frame?.face?.landmarks?.length) };
  // Camera axes: a body facing the camera points towards -Z.
  const yaw = deg(Math.atan2(vector[0], -vector[2]));
  const abs = Math.abs(yaw);
  // Side-on: which side of the body is towards the camera (yaw > 0: the right).
  const view = abs < 45 ? 'front' : abs > 135 ? 'back' : yaw > 0 ? 'right' : 'left';
  return { view, yaw: round(yaw), source, face: Boolean(frame?.face?.landmarks?.length) };
}

function framingOf(image, C) {
  const inFrame = i => { const p = image[i]; return p && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1; };
  const missing = FULL_BODY.filter(i => !(C[i] > 0 && inFrame(i)));
  const ys = [0, 27, 28].map(i => image[i]?.y).filter(Number.isFinite);
  const bodyHeight = ys.length === 3 ? Math.max(image[27].y, image[28].y) - image[0].y : null;
  const hipX = (image[23].x + image[24].x) / 2;
  return {
    fullBody: missing.length === 0,
    missing,
    feetOut: [27, 28].some(i => image[i].y > 1) || [27, 28].every(i => C[i] === 0),
    headOut: image[0].y < 0,
    // Nose-to-ankle height as a fraction of the picture height (full body only).
    bodyHeight: bodyHeight === null || missing.length ? null : round(100 * bodyHeight) / 100,
    centreX: Number.isFinite(hipX) ? round(100 * hipX) / 100 : null,
  };
}

function kneeAhead(W, C, forward, ...legs) {
  if (!forward) return null;
  const values = legs.filter(([knee, toe]) => Math.min(C[knee], C[toe]) >= .5).map(([knee, toe]) => dot(sub(W[knee], W[toe]), forward));
  return values.length ? round(100 * Math.max(...values)) / 100 : null;
}

function frontKneeAhead(W, C, forward) {
  if (!forward) return null;
  // The front leg has its ankle further forward.
  const front = dot(sub(W[27], W[28]), forward) > 0 ? [25, 31] : [26, 32];
  return kneeAhead(W, C, forward, front);
}

/** Up direction from a person standing straight (mid-hip → mid-shoulder). */
export function uprightFrom(frame) {
  const world = frame?.pose?.worldLandmarks;
  if (!Array.isArray(world) || world.length < 33 || ![11, 12, 23, 24].every(i => finite3(world[i]))) return null;
  const W = i => [world[i].x, world[i].y, world[i].z];
  return unit(sub(mid(W(11), W(12)), mid(W(23), W(24))));
}
