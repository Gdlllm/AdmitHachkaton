/**
 * Browser MHR forward kinematics + blend shapes + pose correctives + LBS.
 * The algorithm follows Meta's pinned Apache-2.0 MHR and MIT PyMomentum.
 * Assets are losslessly exported by tools/decoder-export.py. No landmark
 * interpolation or arbitrary body geometry is used here.
 */
import { createVisibleSurface } from './visible-surface.js';

const TYPES = { Float32Array, Uint32Array, Int32Array };
// Pinned exported array payload; also verified by scripts/setup-dense.mjs.
const MHR_ARRAY_SHA256 = '879273a56eb36ea8eaf112c99666d0e1bc2a6846609dc15fc7d52aa7f533a779';
const sha256 = async bytes => [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
  .map(value => value.toString(16).padStart(2, '0')).join('');

/** Fetch may transparently decompress a .gz response when its HTTP server sends
 * Content-Encoding: gzip. Accept either representation, but verify the pinned
 * decoded bytes in both cases; a successful decompression alone is not trust.
 */
export async function decodeMhrAsset(metadata, bytes) {
  let buffer;
  if (bytes instanceof ArrayBuffer) buffer = bytes;
  else if (ArrayBuffer.isView(bytes)) buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  else throw new TypeError('Expected MHR asset ArrayBuffer or typed array');

  if (buffer.byteLength === metadata.compressedBytes) {
    if (await sha256(buffer) !== metadata.sha256) throw new Error('MHR compressed asset SHA-256 mismatch');
    if (typeof DecompressionStream === 'undefined') throw new Error('This browser requires gzip DecompressionStream support');
    const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
    buffer = await new Response(stream).arrayBuffer();
  } else if (buffer.byteLength !== metadata.uncompressedBytes) {
    throw new Error('MHR asset size mismatch');
  }
  if (buffer.byteLength !== metadata.uncompressedBytes) throw new Error('MHR decompressed asset size mismatch');
  if (await sha256(buffer) !== MHR_ARRAY_SHA256) throw new Error('MHR decompressed asset SHA-256 mismatch');
  return buffer;
}

export function viewsFromManifest(manifest, buffer) {
  if (!(buffer instanceof ArrayBuffer)) throw new TypeError('Expected decoded ArrayBuffer');
  if (buffer.byteLength !== manifest.uncompressedBytes) throw new Error('MHR asset byte length mismatch');
  const arrays = {};
  for (const [name, descriptor] of Object.entries(manifest.arrays)) {
    const Type = TYPES[descriptor.type];
    if (!Type || descriptor.byteOffset % 4 || descriptor.byteOffset + descriptor.length * 4 > buffer.byteLength) {
      throw new Error(`Invalid MHR array descriptor: ${name}`);
    }
    arrays[name] = new Type(buffer, descriptor.byteOffset, descriptor.length);
  }
  return arrays;
}

function input(array, size, name) {
  if (!array || array.length !== size) throw new TypeError(`${name} must contain ${size} values`);
  for (let i = 0; i < size; i++) if (!Number.isFinite(array[i])) throw new TypeError(`${name}[${i}] is not finite`);
}

/** No network dependency: useful for numerical tests and controlled asset loading. */
export function createDecoderFromArrays(metadata, arrays) {
  if (metadata.format !== 'mhr-browser-v1') throw new Error('Unsupported MHR asset format');
  const count = metadata.vertexCount * 3;
  const joints = metadata.jointCount;
  if (joints !== 127 || metadata.modelParameterCount !== 204 || metadata.identityCount !== 45 || metadata.expressionCount !== 72) {
    throw new Error('Unsupported MHR rig dimensions');
  }
  for (let j = 0; j < joints; j++) if (arrays.parents[j] >= j) throw new Error('MHR joints must be in parent-before-child order');
  let data = arrays;
  const parameters = new Float64Array(joints * 7);
  const states = new Float64Array(joints * 8);
  const matrices = new Float64Array(joints * 12);
  const rest = new Float64Array(count);
  // Identity/expression blend of the rest shape. Per-frame fitting keeps the
  // same identity for many frames, so this 117-vector blend is reused.
  const identityRest = new Float64Array(count);
  let identityKey = null;
  const features = new Float64Array(750);
  const activations = new Float64Array(3000);
  const zeroExpression = new Float32Array(72);
  const { parents } = arrays;
  const visibleSurface = createVisibleSurface({ faces: arrays.faces, skinIndices: arrays.skinIndices, skinWeights: arrays.skinWeights,
    parents, jointNames: metadata.jointNames, vertexCount: metadata.vertexCount });

  function computeSkeleton(mhrParams) {
    if (!data) throw new Error('MHR decoder has been disposed');
    input(mhrParams, 204, 'mhrParams');
    // PyMomentum's parameter transform: local translation, Euler XYZ, log2 scale.
    const { parameterTransformOffsets: po, parameterTransformIndices: pi, parameterTransformValues: pv } = data;
    for (let row = 0; row < parameters.length; row++) {
      let value = 0;
      for (let k = po[row]; k < po[row + 1]; k++) value += pv[k] * mhrParams[pi[k]];
      parameters[row] = value;
    }

    for (let j = 0; j < joints; j++) {
      const p = j * 7, t = j * 3, q = j * 4, s = j * 8;
      const lx = parameters[p] + data.jointOffsets[t];
      const ly = parameters[p + 1] + data.jointOffsets[t + 1];
      const lz = parameters[p + 2] + data.jointOffsets[t + 2];
      const cx = Math.cos(parameters[p + 3] * .5), sx = Math.sin(parameters[p + 3] * .5);
      const cy = Math.cos(parameters[p + 4] * .5), sy = Math.sin(parameters[p + 4] * .5);
      const cz = Math.cos(parameters[p + 5] * .5), sz = Math.sin(parameters[p + 5] * .5);
      const ex = sx * cy * cz - cx * sy * sz, ey = cx * sy * cz + sx * cy * sz;
      const ez = cx * cy * sz - sx * sy * cz, ew = cx * cy * cz + sx * sy * sz;
      const ax = data.prerotations[q], ay = data.prerotations[q + 1], az = data.prerotations[q + 2], aw = data.prerotations[q + 3];
      const qx = aw * ex + ax * ew + ay * ez - az * ey;
      const qy = aw * ey - ax * ez + ay * ew + az * ex;
      const qz = aw * ez + ax * ey - ay * ex + az * ew;
      const qw = aw * ew - ax * ex - ay * ey - az * ez;
      const scale = Math.exp(Math.LN2 * parameters[p + 6]);
      const parent = parents[j];
      if (parent < 0) {
        states[s] = lx; states[s + 1] = ly; states[s + 2] = lz;
        states[s + 3] = qx; states[s + 4] = qy; states[s + 5] = qz; states[s + 6] = qw; states[s + 7] = scale;
      } else {
        const a = parent * 8;
        const px = states[a + 3], py = states[a + 4], pz = states[a + 5], pw = states[a + 6], ps = states[a + 7];
        const ux = 2 * (py * lz - pz * ly), uy = 2 * (pz * lx - px * lz), uz = 2 * (px * ly - py * lx);
        states[s] = states[a] + ps * (lx + pw * ux + py * uz - pz * uy);
        states[s + 1] = states[a + 1] + ps * (ly + pw * uy + pz * ux - px * uz);
        states[s + 2] = states[a + 2] + ps * (lz + pw * uz + px * uy - py * ux);
        states[s + 3] = pw * qx + px * qw + py * qz - pz * qy;
        states[s + 4] = pw * qy - px * qz + py * qw + pz * qx;
        states[s + 5] = pw * qz + px * qy - py * qx + pz * qw;
        states[s + 6] = pw * qw - px * qx - py * qy - pz * qz;
        states[s + 7] = ps * scale;
      }
    }

  }

  function decode(mhrParams, shapeParams, expression = zeroExpression, { applyCorrectives = true } = {}) {
    input(shapeParams, 45, 'shapeParams'); input(expression, 72, 'expression');
    computeSkeleton(mhrParams);
    const sameIdentity = identityKey && identityKey.every((value, c) => value === (c < 45 ? shapeParams[c] : expression[c - 45]));
    if (!sameIdentity) {
      identityKey = Float64Array.from({ length: 117 }, (_, c) => c < 45 ? shapeParams[c] : expression[c - 45]);
      identityRest.set(data.baseShape);
      for (let c = 0; c < 117; c++) {
        const coefficient = identityKey[c];
        if (coefficient === 0) continue;
        const offset = c * count;
        for (let i = 0; i < count; i++) identityRest[i] += coefficient * data.shapeVectors[offset + i];
      }
    }
    rest.set(identityRest);

    if (applyCorrectives) {
      // Official batch6DFromXYZ: concatenate first two rotation-matrix columns,
      // subtract identity; joints 0 and 1 are global and intentionally excluded.
      for (let j = 2; j < joints; j++) {
        const p = j * 7, f = (j - 2) * 6;
        const cx = Math.cos(parameters[p + 3]), sx = Math.sin(parameters[p + 3]);
        const cy = Math.cos(parameters[p + 4]), sy = Math.sin(parameters[p + 4]);
        const cz = Math.cos(parameters[p + 5]), sz = Math.sin(parameters[p + 5]);
        features[f] = cy * cz - 1; features[f + 1] = cy * sz; features[f + 2] = -sy;
        features[f + 3] = -cx * sz + sx * sy * cz;
        features[f + 4] = cx * cz + sx * sy * sz - 1; features[f + 5] = sx * cy;
      }
      const { correctiveActivationOffsets: ao, correctiveActivationIndices: ai, correctiveActivationValues: av,
        correctiveShapeOffsets: co, correctiveShapeIndices: ci, correctiveShapeValues: cv } = data;
      for (let row = 0; row < activations.length; row++) {
        let value = 0;
        for (let k = ao[row]; k < ao[row + 1]; k++) value += av[k] * features[ai[k]];
        activations[row] = Math.max(value, 0);
      }
      for (let row = 0; row < activations.length; row++) {
        const value = activations[row];
        if (value === 0) continue;
        for (let k = co[row]; k < co[row + 1]; k++) rest[ci[k]] += value * cv[k];
      }
    }

    // Combine each global skeleton transform with its inverse bind transform.
    for (let j = 0; j < joints; j++) {
      const s = j * 8, m = j * 12;
      const px = states[s + 3], py = states[s + 4], pz = states[s + 5], pw = states[s + 6], ps = states[s + 7];
      const ix = data.inverseBindPose[s], iy = data.inverseBindPose[s + 1], iz = data.inverseBindPose[s + 2];
      const qx = data.inverseBindPose[s + 3], qy = data.inverseBindPose[s + 4], qz = data.inverseBindPose[s + 5], qw = data.inverseBindPose[s + 6];
      const x = pw * qx + px * qw + py * qz - pz * qy;
      const y = pw * qy - px * qz + py * qw + pz * qx;
      const z = pw * qz + px * qy - py * qx + pz * qw;
      const w = pw * qw - px * qx - py * qy - pz * qz;
      const scale = ps * data.inverseBindPose[s + 7];
      const ux = 2 * (py * iz - pz * iy), uy = 2 * (pz * ix - px * iz), uz = 2 * (px * iy - py * ix);
      matrices[m] = scale * (1 - 2 * (y * y + z * z)); matrices[m + 1] = scale * 2 * (x * y - z * w); matrices[m + 2] = scale * 2 * (x * z + y * w);
      matrices[m + 3] = states[s] + ps * (ix + pw * ux + py * uz - pz * uy);
      matrices[m + 4] = scale * 2 * (x * y + z * w); matrices[m + 5] = scale * (1 - 2 * (x * x + z * z)); matrices[m + 6] = scale * 2 * (y * z - x * w);
      matrices[m + 7] = states[s + 1] + ps * (iy + pw * uy + pz * ux - px * uz);
      matrices[m + 8] = scale * 2 * (x * z - y * w); matrices[m + 9] = scale * 2 * (y * z + x * w); matrices[m + 10] = scale * (1 - 2 * (x * x + y * y));
      matrices[m + 11] = states[s + 2] + ps * (iz + pw * uz + px * uy - py * ux);
    }
    const vertices = new Float32Array(count);
    for (let v = 0; v < metadata.vertexCount; v++) {
      const r = v * 3, skin = v * 8, x = rest[r], y = rest[r + 1], z = rest[r + 2];
      let vx = 0, vy = 0, vz = 0;
      for (let i = 0; i < 8; i++) {
        const weight = data.skinWeights[skin + i];
        if (weight === 0) continue;
        const m = data.skinIndices[skin + i] * 12;
        vx += weight * (matrices[m] * x + matrices[m + 1] * y + matrices[m + 2] * z + matrices[m + 3]);
        vy += weight * (matrices[m + 4] * x + matrices[m + 5] * y + matrices[m + 6] * z + matrices[m + 7]);
        vz += weight * (matrices[m + 8] * x + matrices[m + 9] * y + matrices[m + 10] * z + matrices[m + 11]);
      }
      vertices[r] = vx * .01; vertices[r + 1] = vy * -.01; vertices[r + 2] = vz * -.01;
    }
    const rig = readSkeleton();
    return { vertices, skeleton: { positions: rig.positions, parents, jointNames: metadata.jointNames },
      keypoints70: rig.keypoints70, keypointNames70: rig.keypointNames70,
      keypoints70Source: rig.keypoints70Source, units: rig.units, axes: rig.axes, inferred: true };
  }

  function readSkeleton() {
    const positions = new Float32Array(joints * 3);
    for (let j = 0; j < joints; j++) {
      positions[j * 3] = states[j * 8] * .01;
      positions[j * 3 + 1] = states[j * 8 + 1] * -.01;
      positions[j * 3 + 2] = states[j * 8 + 2] * -.01;
    }
    // Upstream's 70x127 learned linear regressor, separate from the 70 joints
    // independently predicted by InstantHMR's image head. It is NOT an index map.
    let keypoints70 = null;
    if (data.keypointRegressor) {
      keypoints70 = new Float32Array(70 * 3);
      for (let k = 0; k < 70; k++) {
        let x = 0, y = 0, z = 0;
        for (let j = 0; j < joints; j++) {
          const weight = data.keypointRegressor[k * joints + j];
          x += weight * positions[j * 3]; y += weight * positions[j * 3 + 1]; z += weight * positions[j * 3 + 2];
        }
        keypoints70[k * 3] = x; keypoints70[k * 3 + 1] = y; keypoints70[k * 3 + 2] = z;
      }
    }
    return { positions, parents, jointNames: metadata.jointNames, keypoints70, keypointNames70: metadata.keypointNames,
      keypoints70Source: 'linear regression from decoded MHR rig', units: 'metres', axes: 'X-right/Y-down/Z-forward', inferred: true };
  }

  /** Same official FK as decode(), without vertices, blend shapes or correctives.
   * Identity and expression affect the skin, not this MHR joint transform chain.
   * Each call owns its returned position/keypoint arrays; safe for numerical IK.
   */
  function decodeSkeleton(mhrParams) {
    computeSkeleton(mhrParams);
    return readSkeleton();
  }

  /** Same FK, but regresses only the requested keypoints (others are NaN).
   * Iterative image fitting needs ~17 of the 70 keypoints per evaluation. */
  function decodeKeypoints(mhrParams, indices) {
    computeSkeleton(mhrParams);
    const keypoints70 = new Float32Array(70 * 3).fill(NaN);
    for (const k of indices) {
      if (!Number.isInteger(k) || k < 0 || k >= 70) continue;
      let x = 0, y = 0, z = 0;
      for (let j = 0; j < joints; j++) {
        const weight = data.keypointRegressor[k * joints + j], s = j * 8;
        // Rounded like readSkeleton's Float32 positions, so results are identical.
        x += weight * Math.fround(states[s] * .01);
        y += weight * Math.fround(states[s + 1] * -.01);
        z += weight * Math.fround(states[s + 2] * -.01);
      }
      keypoints70[k * 3] = x; keypoints70[k * 3 + 1] = y; keypoints70[k * 3 + 2] = z;
    }
    return keypoints70;
  }

  return { decode, decodeSkeleton, decodeKeypoints, faces: arrays.faces, visibleSurface, metadata, dispose() { data = null; } };
}

/** Load ~losslessly compressed assets once, with no Python/server computation. */
export async function createBodyMeshDecoder({ assetUrl = '/dense/mhr-lod3.json', signal } = {}) {
  const manifestResponse = await fetch(assetUrl, { signal });
  if (!manifestResponse.ok) throw new Error(`MHR manifest HTTP ${manifestResponse.status}`);
  const metadata = await manifestResponse.json();
  const binaryUrl = new URL(metadata.binaryFile, new URL(assetUrl, globalThis.location?.href ?? import.meta.url));
  const response = await fetch(binaryUrl, { signal });
  if (!response.ok) throw new Error(`MHR assets HTTP ${response.status}`);
  const buffer = await decodeMhrAsset(metadata, await response.arrayBuffer());
  return createDecoderFromArrays(metadata, viewsFromManifest(metadata, buffer));
}
