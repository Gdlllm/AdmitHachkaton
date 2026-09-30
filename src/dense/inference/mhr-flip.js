import { MIRROR_SIGN, MIRROR_TARGET } from './mhr-mirror.js';

// MHR root rotation in camera axes (X right, Y down, Z forward), about the root
// joint: R = Rz(-rz) · Ry(-ry) · Rx(rx) (measured in ml/body-sense/mhr_mirror.mjs).
export function rootMatrix(rx, ry, rz) {
  const [cx, sx, cy, sy, cz, sz] = [Math.cos(rx), Math.sin(rx), Math.cos(-ry), Math.sin(-ry), Math.cos(-rz), Math.sin(-rz)];
  // Rz·Ry·Rx
  return [
    [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
    [-sy, cy * sx, cy * cx],
  ];
}

/** Root parameters [rx, ry, rz] of a rotation matrix (inverse of rootMatrix). */
export function rootParams(F) {
  const b = -Math.asin(Math.max(-1, Math.min(1, F[2][0])));
  return [Math.atan2(F[2][1], F[2][2]), -b, -Math.atan2(F[1][0], F[0][0])];
}

/** The other reading of the same picture: the body reflected in depth about
 * its root and relabelled left↔right, i.e. a front seen as a back (or the
 * reverse). Its 2D projection is nearly unchanged (exact for a distant camera). */
export function flipFrontBack(mhrParams) {
  const out = new Float32Array(mhrParams.length);
  for (let i = 6; i < mhrParams.length; i++) out[MIRROR_TARGET[i]] = MIRROR_SIGN[i] * mhrParams[i];
  out[0] = mhrParams[0]; out[1] = mhrParams[1]; out[2] = mhrParams[2];
  // R' = D·R·S: D reflects camera depth, S mirrors the body's left/right axis (X at rest).
  const R = rootMatrix(mhrParams[3], mhrParams[4], mhrParams[5]);
  const F = R.map((row, i) => row.map((v, j) => v * (i === 2 ? -1 : 1) * (j === 0 ? -1 : 1)));
  // F = Rz(c)·Ry(b)·Rx(a) with a = rx, b = -ry, c = -rz.
  const b = -Math.asin(Math.max(-1, Math.min(1, F[2][0])));
  out[3] = Math.atan2(F[2][1], F[2][2]);
  out[4] = -b;
  out[5] = -Math.atan2(F[1][0], F[0][0]);
  return out;
}
