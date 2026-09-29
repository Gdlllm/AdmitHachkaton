/** Body anchors in the official MHR70 ordering. Finger density must not make
 * camera placement primarily a fit to one hand. These are not confidences. */
export const CAMERA_BODY_INDICES = Object.freeze([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62, 69]);

const finite = Number.isFinite;
const point = (points, i, dimension) => typeof points?.[0] === 'number'
  ? Array.from({ length: dimension }, (_, d) => points[i * dimension + d])
  : Array.isArray(points?.[i]) ? points[i].slice(0, dimension)
    : dimension === 3 ? [points?.[i]?.x, points?.[i]?.y, points?.[i]?.z] : [points?.[i]?.x, points?.[i]?.y];
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const huber = (r, delta) => r <= delta ? r * r / 2 : delta * (r - delta / 2);

// Pivoted 3x3 solve. A rank-deficient point configuration is rejected instead
// of making up a depth through a diagonal regularizer.
function solve(matrix, rhs) {
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  const scale = Math.max(...matrix.flat().map(Math.abs));
  if (!(scale > 0)) return null;
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let row = col + 1; row < 3; row++) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    if (Math.abs(a[pivot][col]) < scale * 1e-10) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    for (let c = col; c < 4; c++) a[col][c] /= divisor;
    for (let row = 0; row < 3; row++) if (row !== col) {
      const factor = a[row][col];
      for (let c = col; c < 4; c++) a[row][c] -= factor * a[col][c];
    }
  }
  const result = a.map(row => row[3]);
  return result.every(finite) ? result : null;
}

function normalSystem() {
  const matrix = Array.from({ length: 3 }, () => [0, 0, 0]), rhs = [0, 0, 0];
  return {
    add(row, target, weight) {
      for (let i = 0; i < 3; i++) {
        rhs[i] += weight * row[i] * target;
        for (let j = 0; j < 3; j++) matrix[i][j] += weight * row[i] * row[j];
      }
    },
    solve: () => solve(matrix, rhs),
  };
}

/** Estimate camera translation only, with focal and rig geometry fixed.
 * Independent image-head targets are predictions, not calibration or GT.
 * The returned camera is for an optional overlay correction; original model
 * camera and geometry remain unchanged. No temporal state or pose detection.
 */
export function fitCameraTranslation({ joints3d, joints2d, initialTranslation, focal, principalPoint,
  indices = CAMERA_BODY_INDICES, weights, imageSize, huberPixels = 8,
  depthRatio = [0.5, 2], maxPixelShift = 200, minDepth = 0.05, maxIterations = 15 } = {}) {
  const originalTranslation = Array.from(initialTranslation ?? []);
  if (originalTranslation.length !== 3 || !originalTranslation.every(finite) || !(originalTranslation[2] > 0)) throw new Error('initialTranslation must be finite [tx,ty,tz] with positive tz');
  if (!(finite(focal) && focal > 0) || principalPoint?.length !== 2 || !Array.from(principalPoint).every(finite)) throw new Error('A fixed positive focal and finite principalPoint are required');
  if (!(huberPixels > 0 && finite(huberPixels) && maxPixelShift > 0 && finite(maxPixelShift) && minDepth > 0 && finite(minDepth))) throw new Error('Invalid camera-fit constraints');
  if (!(depthRatio?.length === 2 && depthRatio.every(finite) && depthRatio[0] > 0 && depthRatio[0] <= 1 && depthRatio[1] >= 1)) throw new Error('depthRatio must bracket the original positive depth');
  if (!Number.isInteger(maxIterations) || maxIterations < 1 || maxIterations > 100) throw new Error('maxIterations must be 1..100');
  if (imageSize && !(imageSize.length === 2 && imageSize.every(v => finite(v) && v > 0))) throw new Error('imageSize must be [width,height]');

  const selected = [];
  for (const index of new Set(indices)) {
    if (!Number.isInteger(index) || index < 0) continue;
    const xyz = point(joints3d, index, 3), uv = point(joints2d, index, 2), weight = weights?.[index] ?? 1;
    if (!xyz.every(finite) || !uv.every(finite) || !(finite(weight) && weight > 0)) continue;
    if (imageSize && (uv[0] < 0 || uv[1] < 0 || uv[0] >= imageSize[0] || uv[1] >= imageSize[1])) continue;
    selected.push({ index, xyz, uv, weight });
  }
  const fail = reason => ({ accepted: false, reason, originalTranslation, translation: [...originalTranslation], selectedIndices: selected.map(p => p.index), before: null, after: null });
  if (selected.length < 4) return fail('insufficient-correspondences');
  const minZ = Math.min(...selected.map(p => p.xyz[2]));
  const depthBounds = [Math.max(originalTranslation[2] * depthRatio[0], minDepth - minZ), originalTranslation[2] * depthRatio[1]];
  if (depthBounds[0] >= depthBounds[1] || originalTranslation[2] + minZ < minDepth) return fail('invalid-initial-depth');
  const lateral = originalTranslation[2] * maxPixelShift / focal;
  const bounds = [[originalTranslation[0] - lateral, originalTranslation[0] + lateral], [originalTranslation[1] - lateral, originalTranslation[1] + lateral], depthBounds];
  const constrain = t => t.map((value, i) => clamp(value, ...bounds[i]));
  const residuals = translation => selected.map(({ xyz, uv, weight }) => {
    const z = xyz[2] + translation[2];
    const rx = focal * (xyz[0] + translation[0]) / z + principalPoint[0] - uv[0];
    const ry = focal * (xyz[1] + translation[1]) / z + principalPoint[1] - uv[1];
    return { rx, ry, r: Math.hypot(rx, ry), weight };
  });
  const observability = normalSystem();
  for (const { xyz: [x, y, z], weight } of selected) {
    const depth = z + originalTranslation[2];
    observability.add([focal / depth, 0, -focal * (x + originalTranslation[0]) / depth ** 2], 0, weight);
    observability.add([0, focal / depth, -focal * (y + originalTranslation[1]) / depth ** 2], 0, weight);
  }
  if (!observability.solve()) return fail('degenerate-correspondences');
  const objective = values => values.reduce((sum, { r, weight }) => sum + weight * huber(r, huberPixels), 0);
  const summary = values => {
    const ordered = values.map(p => p.r).sort((a, b) => a - b);
    return { meanPixels: ordered.reduce((a, b) => a + b, 0) / ordered.length,
      medianPixels: ordered[Math.floor(ordered.length / 2)], maxPixels: ordered.at(-1),
      robustObjective: objective(values), count: ordered.length };
  };

  // f*tx - (u-cx)*tz = (u-cx)*Z - f*X. This algebraic solve
  // initializes the perspective fit; final optimization is in source pixels.
  const linear = normalSystem();
  for (const { xyz: [x, y, z], uv, weight } of selected) {
    const u = uv[0] - principalPoint[0], v = uv[1] - principalPoint[1];
    linear.add([focal, 0, -u], u * z - focal * x, weight);
    linear.add([0, focal, -v], v * z - focal * y, weight);
  }
  const initialization = linear.solve();
  if (!initialization) return fail('degenerate-correspondences');
  const before = summary(residuals(originalTranslation));
  let translation = [...originalTranslation], score = before.robustObjective;
  const candidate = constrain(initialization), candidateScore = objective(residuals(candidate));
  if (candidateScore < score) { translation = candidate; score = candidateScore; }
  let iterations = 0;
  for (; iterations < maxIterations; iterations++) {
    const values = residuals(translation), system = normalSystem();
    selected.forEach(({ xyz: [x, y, z], weight }, i) => {
      const depth = z + translation[2], { rx, ry, r } = values[i];
      const robustWeight = weight * Math.min(1, huberPixels / Math.max(r, 1e-12));
      system.add([focal / depth, 0, -focal * (x + translation[0]) / (depth * depth)], -rx, robustWeight);
      system.add([0, focal / depth, -focal * (y + translation[1]) / (depth * depth)], -ry, robustWeight);
    });
    const delta = system.solve();
    if (!delta || Math.hypot(...delta) < 1e-8) break;
    let improved = false;
    for (let step = 1; step >= 1 / 128; step /= 2) {
      const next = constrain(translation.map((v, i) => v + delta[i] * step));
      const nextScore = objective(residuals(next));
      if (nextScore < score - 1e-8) { translation = next; score = nextScore; improved = true; break; }
    }
    if (!improved) break;
  }
  const accepted = score < before.robustObjective - 1e-6;
  if (!accepted) translation = [...originalTranslation];
  return { accepted, reason: accepted ? 'reprojection-improved' : 'no-improvement', originalTranslation, translation,
    selectedIndices: selected.map(p => p.index), before, after: summary(residuals(translation)), iterations,
    constraints: { depthRatio: [...depthRatio], depthBounds, maxPixelShift, minDepth, huberPixels,
      atBoundary: translation.map((v, i) => Math.abs(v - bounds[i][0]) < 1e-7 || Math.abs(v - bounds[i][1]) < 1e-7) },
    targetSource: 'independent image predictions; not ground truth', corrected: 'camera translation only; fixed focal and rig' };
}
