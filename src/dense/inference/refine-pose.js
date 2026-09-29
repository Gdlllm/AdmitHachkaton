// Bounded image-space refinement of a learned 3D prior. It cannot determine
// hidden anatomy from one camera. Scale, root and finger parameters stay fixed.
// Pose index -> MHR70 keypoint. Eyes and ears anchor close-ups where only the
// head and shoulders are in view.
const POSE_TO_MHR = [[0,0],[2,1],[5,2],[7,3],[8,4],[11,5],[12,6],[13,7],[14,8],[15,62],[16,41],[23,9],[24,10],[25,11],[26,12],[27,13],[28,14],[29,17],[30,20],[31,15],[32,18]];
const ELIGIBLE = /^(spine_(lean|bend|twist)[01]|[lr]_clavicle_r[xyz]|[lr]_uparm_r[yz]|[lr]_elbow_bend|[lr]_upleg_r[yz]|[lr]_knee_bend|[lr]_foot_bend)$/;

export function poseTargets(pose, width, height, minConfidence = .65) {
  if (!pose?.length) return [];
  return POSE_TO_MHR.flatMap(([index, keypoint]) => {
    const p = pose[index];
    const confidence = Math.min(p?.visibility ?? 0, p?.presence ?? 1);
    if (!p || !Number.isFinite(confidence) || confidence < minConfidence || !Number.isFinite(p.x) || !Number.isFinite(p.y) || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) return [];
    return [{ keypoint, x: p.x * width, y: p.y * height, weight: confidence, source: 'MediaPipe Pose image observation' }];
  });
}

function solve(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let column = 0; column < n; column++) {
    let pivot = column;
    for (let row = column + 1; row < n; row++) if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    if (Math.abs(a[pivot][column]) < 1e-10) return null;
    [a[column], a[pivot]] = [a[pivot], a[column]];
    const divisor = a[column][column];
    for (let c = column; c <= n; c++) a[column][c] /= divisor;
    for (let row = 0; row < n; row++) {
      if (row === column) continue;
      const factor = a[row][column];
      for (let c = column; c <= n; c++) a[row][c] -= factor * a[column][c];
    }
  }
  return a.map(row => row[n]);
}

function project(points, targets, camera) {
  return targets.map(target => {
    const i = target.keypoint * 3;
    const x = points[i] + camera.translation[0], y = points[i + 1] + camera.translation[1], z = points[i + 2] + camera.translation[2];
    if (!(z > .01) || ![x,y,z].every(Number.isFinite)) return null;
    return { x: camera.width / 2 + camera.focal * x / z, y: camera.height / 2 + camera.focal * y / z };
  });
}

/** `initial` warm-starts the solve (e.g. from the previous video frame); the
 * trust region and regularization stay centred on the `mhrParams` prior. */
export function refineBodyPose({ decoder, mhrParams, camera, targets, iterations = 4, maxAngleChange = .45, bodyHeight, initial }) {
  const began = performance.now();
  if (!Number.isInteger(iterations) || iterations < 0 || iterations > 12 || !Number.isFinite(maxAngleChange) || maxAngleChange < 0 || maxAngleChange > 1) throw new RangeError('Invalid refinement trust region');
  if (!mhrParams || mhrParams.length !== 204 || !Array.from(mhrParams).every(Number.isFinite)) throw new TypeError('204 finite MHR parameters required');
  if (initial !== undefined && (initial?.length !== 204 || !Array.from(initial).every(Number.isFinite))) throw new TypeError('A warm start needs 204 finite MHR parameters');
  if (!camera || ![camera.width, camera.height, camera.focal].every(v => Number.isFinite(v) && v > 0) || camera.translation?.length !== 3 || !Array.from(camera.translation).every(Number.isFinite)) throw new TypeError('Finite camera required');
  if (bodyHeight !== undefined && !(Number.isFinite(bodyHeight) && bodyHeight > 0)) throw new RangeError('Invalid body height');
  const prior = Float32Array.from(mhrParams);
  const current = Float32Array.from(prior);
  const unique = new Map();
  for (const target of targets ?? []) {
    if (!Number.isInteger(target?.keypoint) || target.keypoint < 0 || target.keypoint >= 70 || ![target.x,target.y,target.weight ?? 1].every(Number.isFinite) || (target.weight ?? 1) <= 0) continue;
    if (target.x < 0 || target.x > camera.width || target.y < 0 || target.y > camera.height) continue;
    if (!unique.has(target.keypoint)) unique.set(target.keypoint, target);
  }
  targets = [...unique.values()];
  if (targets.length < 6) return { mhrParams: prior, accepted: false, reason: 'insufficient-visible-anchors', timings: { refineMs: performance.now() - began } };
  const names = decoder.metadata.modelParameterNames ?? decoder.metadata.parameterNames;
  const has = (...keys) => keys.every(key => unique.has(key));
  // A segment is adjusted only when both of its ends are observed: an upper arm
  // (hand behind the head) needs shoulder + elbow, the elbow bend also a wrist.
  const supported = name => {
    if (name.startsWith('spine')) return has(5,6,9,10);
    const left = name.startsWith('l_');
    if (/clavicle|uparm/.test(name)) return left ? has(5,7) : has(6,8);
    if (/elbow/.test(name)) return left ? has(5,7,62) : has(6,8,41);
    if (/upleg/.test(name)) return left ? has(9,11) : has(10,12);
    const leg = left ? has(9,11,13) : has(10,12,14);
    if (/knee/.test(name)) return leg;
    if (name.includes('foot')) return leg && (left ? has(15) || has(17) : has(18) || has(20));
    return false;
  };
  const indices = names.flatMap((name, i) => i < 130 && ELIGIBLE.test(name) && supported(name) ? [i] : []);
  if (!indices.length) return { mhrParams: prior, accepted: false, reason: 'no-observed-body-chains', timings: { refineMs: performance.now() - began } };
  if (initial) for (const index of indices) current[index] = Math.max(prior[index] - maxAngleChange, Math.min(prior[index] + maxAngleChange, initial[index]));
  const scale = bodyHeight ?? Math.max(80, Math.max(...targets.map(t => t.y)) - Math.min(...targets.map(t => t.y)));
  const huber = Math.max(3, scale * .035);
  const regularization = scale * scale * .0015;
  const damping = scale * scale * .0002;
  const keypoints = targets.map(target => target.keypoint);
  const evaluate = parameters => {
    const points = decoder.decodeKeypoints ? decoder.decodeKeypoints(parameters, keypoints) : decoder.decodeSkeleton(parameters).keypoints70;
    const pixels = project(points, targets, camera);
    let cost = 0;
    const distances = [];
    for (let i = 0; i < targets.length; i++) {
      if (!pixels[i]) return { cost: Infinity, pixels, distances: [] };
      const distance = Math.hypot(pixels[i].x - targets[i].x, pixels[i].y - targets[i].y);
      distances.push(distance);
      cost += (targets[i].weight ?? 1) * (distance <= huber ? distance * distance : 2 * huber * distance - huber * huber);
    }
    for (const index of indices) cost += regularization * (parameters[index] - prior[index]) ** 2;
    return { cost, pixels, distances, points };
  };
  let state = evaluate(current);
  const beforeError = state.distances.reduce((a,b) => a+b,0) / targets.length;
  let completed = 0;
  if (!Number.isFinite(state.cost)) return { mhrParams: prior, accepted: false, reason: 'invalid-initial-projection', timings: { refineMs: performance.now() - began } };
  for (let step = 0; step < iterations; step++) {
    const n = indices.length;
    const jacobian = [];
    const epsilon = .002;
    for (const index of indices) {
      const perturbed = Float32Array.from(current); perturbed[index] += epsilon;
      const other = evaluate(perturbed);
      if (!Number.isFinite(other.cost)) break;
      jacobian.push(other.pixels.flatMap((p, i) => [(p.x-state.pixels[i].x)/epsilon,(p.y-state.pixels[i].y)/epsilon]));
    }
    if (jacobian.length !== n) break;
    const normal = Array.from({length:n}, () => new Float64Array(n));
    const rhs = new Float64Array(n);
    for (let p = 0; p < targets.length; p++) {
      const weight = (targets[p].weight ?? 1) * Math.min(1, huber / Math.max(1e-8, state.distances[p]));
      const residual = [targets[p].x-state.pixels[p].x, targets[p].y-state.pixels[p].y];
      for (let a = 0; a < n; a++) {
        const ax = jacobian[a][p*2], ay = jacobian[a][p*2+1];
        rhs[a] += weight * (ax * residual[0] + ay * residual[1]);
        for (let b = 0; b < n; b++) normal[a][b] += weight * (ax*jacobian[b][p*2]+ay*jacobian[b][p*2+1]);
      }
    }
    for (let i = 0; i < n; i++) {
      normal[i][i] += regularization + damping;
      rhs[i] += regularization * (prior[indices[i]]-current[indices[i]]);
    }
    const delta = solve(normal, rhs);
    if (!delta || !delta.every(Number.isFinite)) break;
    let improved = false;
    for (const rate of [1,.5,.25]) {
      const proposal = Float32Array.from(current);
      for (let i = 0; i < n; i++) {
        const index = indices[i];
        const change = Math.max(-.18,Math.min(.18,delta[i])) * rate;
        proposal[index] = Math.max(prior[index]-maxAngleChange,Math.min(prior[index]+maxAngleChange,current[index]+change));
      }
      const next = evaluate(proposal);
      if (next.cost < state.cost) { current.set(proposal); state = next; improved = true; break; }
    }
    if (!improved) break;
    completed++;
  }
  const meanErrorPx = state.distances.reduce((a,b)=>a+b,0) / targets.length;
  return { mhrParams: current, accepted: completed > 0, reason: completed ? 'bounded-visible-joint-fit' : 'no-improvement', anchorCount: targets.length,
    beforeErrorPx: beforeError, meanErrorPx, maxErrorPx: Math.max(...state.distances), normalizedError: meanErrorPx / scale,
    iterations: completed, adjustableParameters: indices.length,
    timings: { refineMs: performance.now() - began },
    limitations: 'Image-space fit to estimated visible joints; hidden geometry and depth remain inferred.' };
}
