/** Body sense: a small causal temporal network (trained in ml/body-sense on
 * motion capture filmed by virtual cameras) that looks at the last second of
 * Pose landmarks, the face and the surface network's body direction, and says
 * which way the torso really faces, which MediaPipe left/right labels are
 * swapped, and where every joint is, hidden ones included.
 *
 * Runs one frame at a time (dilated causal convolutions with cached history):
 * about 0.3 M multiply-adds per frame, no runtime library. Features must match
 * ml/body-sense/simulate.py exactly.
 *
 *   const model = parseBodySense(manifest, arrayBuffer);
 *   const sense = createBodySense(model);
 *   sense.step({ pose, width, height, dt, face, network }) → { forward, back, swap, joints }
 * pose: 33 × {x, y, visibility} (visibility 0 where the stabilizer hides a point).
 */
export const POSE_PAIRS = [[1, 4], [2, 5], [3, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22],
  [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]];
export const FEATURES = 33 * 4 + 3 + 4 + 4 + 1 + 1;
// Lift models (v9+) also read the surface network's 3D joints (these MediaPipe
// points, which MHR has too) and output the whole 3D skeleton with a sigma per joint.
export const NET_JOINT_IDS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
const HEAD = 3 + 1 + 16 + 66, LIFT_HEAD = 99 + 33;

const halfToFloat = (() => {
  const exponent = new Float32Array(32);
  for (let e = 0; e < 32; e++) exponent[e] = e ? 2 ** (e - 15) : 2 ** -14;
  return h => {
    const sign = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0x1f) return m ? NaN : sign * Infinity;
    return sign * exponent[e] * (e ? 1 + m / 1024 : m / 1024);
  };
})();

/** manifest: {format, config, tensors: [{name, shape, offset, length}]}; buffer: fp16 little-endian. */
export function parseBodySense(manifest, buffer) {
  if (manifest?.format !== 'body-sense-v1') throw new Error('Unsupported body-sense model');
  const { channels, dilations, kernel, features } = manifest.config;
  const lift = manifest.config.lift === true, netJoints = manifest.config.netJoints === true;
  const expected = FEATURES + (netJoints ? NET_JOINT_IDS.length * 3 : 0);
  if (features !== expected || kernel !== 3) throw new Error('Body-sense model does not match this runtime');
  const headSize = HEAD + (lift ? LIFT_HEAD : 0);
  const halves = new Uint16Array(buffer);
  const tensors = {};
  for (const { name, shape, offset, length } of manifest.tensors) {
    if (offset + length > halves.length || length !== shape.reduce((a, b) => a * b, 1)) throw new Error(`Bad tensor ${name}`);
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) out[i] = halfToFloat(halves[offset + i]);
    tensors[name] = out;
  }
  const need = (name, length) => {
    if (tensors[name]?.length !== length) throw new Error(`Missing tensor ${name}`);
    return tensors[name];
  };
  const C = channels;
  return {
    channels: C, dilations, features, lift, netJoints, headSize,
    // Models trained without the 11 face points see zeros there (ml/body-sense/simulate.py FACE_POINTS).
    facePoints: manifest.config.facePoints !== false,
    inW: need('inp.weight', C * features), inB: need('inp.bias', C),
    blocks: dilations.map((dilation, i) => ({
      dilation,
      normW: need(`blocks.${i}.norm.weight`, C), normB: need(`blocks.${i}.norm.bias`, C),
      convW: need(`blocks.${i}.conv.weight`, C * C * 3), convB: need(`blocks.${i}.conv.bias`, C),
      projW: need(`blocks.${i}.proj.weight`, C * C), projB: need(`blocks.${i}.proj.bias`, C),
    })),
    normW: need('norm.weight', C), normB: need('norm.bias', C),
    headW: need('head.weight', headSize * C), headB: need('head.bias', headSize),
  };
}

function layerNorm(x, w, b, out) {
  const n = x.length;
  let mean = 0; for (let i = 0; i < n; i++) mean += x[i]; mean /= n;
  let variance = 0; for (let i = 0; i < n; i++) { const d = x[i] - mean; variance += d * d; } variance /= n;
  const inv = 1 / Math.sqrt(variance + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
  return out;
}

// out = W x + b for a row-major (rows × x.length) matrix.
function linear(W, b, x, out) {
  const cols = x.length;
  for (let r = 0; r < out.length; r++) {
    let s = b[r];
    for (let c = 0, o = r * cols; c < cols; c++) s += W[o + c] * x[c];
    out[r] = s;
  }
  return out;
}

const gelu = v => 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
const sigmoid = v => 1 / (1 + Math.exp(-v));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** Body centre (mid-shoulder) and a size that survives turning sideways, in pixels. */
export function frameNormalization(pose, width, height) {
  const px = i => [pose[i].x * width, pose[i].y * height];
  const [lx, ly] = px(11), [rx, ry] = px(12), [lhx, lhy] = px(23), [rhx, rhy] = px(24), [nx, ny] = px(0);
  const cx = (lx + rx) / 2, cy = (ly + ry) / 2;
  const tall = Math.hypot(cx - (lhx + rhx) / 2, cy - (lhy + rhy) / 2);
  const wide = 1.3 * Math.hypot(lx - rx, ly - ry), neck = 2.5 * Math.hypot(nx - cx, ny - cy);
  return { cx, cy, scale: Math.max(8, tall, wide, neck) };
}

export function createBodySense(model) {
  const C = model.channels, receptive = 1 + 2 * model.dilations.reduce((a, b) => a + b, 0);
  const x = new Float32Array(C), n = new Float32Array(C), conv = new Float32Array(C), h = new Float32Array(C), proj = new Float32Array(C);
  const z = new Float32Array(C), head = new Float32Array(model.headSize ?? HEAD), features = new Float32Array(model.features ?? FEATURES);
  // Per block: normalized inputs of the last 2·dilation+1 frames (ring).
  const rings = model.blocks.map(block => ({ size: 2 * block.dilation + 1, head: 0, data: Array.from({ length: 2 * block.dilation + 1 }, () => new Float32Array(C)) }));
  let previous = null;

  function forward(f) {
    linear(model.inW, model.inB, f, x);
    model.blocks.forEach((block, bi) => {
      const ring = rings[bi], d = block.dilation;
      layerNorm(x, block.normW, block.normB, ring.data[ring.head]);
      const at = back => ring.data[(ring.head - back + ring.size) % ring.size];
      const taps = [at(2 * d), at(d), at(0)];          // Conv1d kernel index 0,1,2 = t-2d, t-d, t
      for (let o = 0; o < C; o++) {
        let s = block.convB[o];
        for (let k = 0; k < 3; k++) {
          const tap = taps[k];
          for (let c = 0, w = (o * C) * 3 + k; c < C; c++, w += 3) s += block.convW[w] * tap[c];
        }
        h[o] = gelu(s);
      }
      linear(block.projW, block.projB, h, proj);
      for (let o = 0; o < C; o++) x[o] += proj[o];
      ring.head = (ring.head + 1) % ring.size;
    });
    return linear(model.headW, model.headB, layerNorm(x, model.normW, model.normB, z), head);
  }

  function reset() {
    previous = null;
    for (const ring of rings) { ring.head = 0; for (const row of ring.data) row.fill(0); }
    // Before a track starts, training saw empty (all-zero) frames.
    features.fill(0);
    for (let i = 0; i < receptive; i++) forward(features);
  }

  /** frame: {pose: 33 × {x,y,visibility}, width, height, dt (s), face: {forward:[3]} | null,
   * network: {forward:[3], joints?: 21×3 camera-axes positions (NET_JOINT_IDS order)} | null}.
   * Returns the current frame's estimate; lift models add joints3d (33×3, root-relative,
   * camera axes, in torso lengths) and sigma3d (33, same unit). */
  function step({ pose, width, height, dt = 1 / 30, face = null, network = null }) {
    const { cx, cy, scale } = frameNormalization(pose, width, height);
    for (let i = 0; i < 33; i++) {
      if (i < 11 && !model.facePoints) { features.fill(0, i * 4, i * 4 + 4); continue; }
      const p = pose[i];
      features[i * 4] = clamp((p.x * width - cx) / scale, -6, 6);
      features[i * 4 + 1] = clamp((p.y * height - cy) / scale, -6, 6);
      features[i * 4 + 2] = p.visibility ?? 0;
      // The web landmarker has no presence score; whether the point is in the picture stands in.
      features[i * 4 + 3] = p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1 ? 1 : 0;
    }
    const m = 132;
    if (previous) {
      features[m] = clamp((cx - previous.cx) / scale, -3, 3);
      features[m + 1] = clamp((cy - previous.cy) / scale, -3, 3);
      features[m + 2] = clamp(Math.log(scale / previous.scale), -1, 1);
    } else features[m] = features[m + 1] = features[m + 2] = 0;
    const put = (at, direction) => {
      features[at] = direction ? 1 : 0;
      for (let k = 0; k < 3; k++) features[at + 1 + k] = direction ? direction[k] : 0;
    };
    put(m + 3, face?.forward); put(m + 7, network?.forward);
    features[m + 11] = clamp(dt, 0, 0.1) * 10;
    features[m + 12] = 1;
    if (model.netJoints) putNetworkJoints(features, FEATURES, network?.forward ? network.joints : null);
    previous = { cx, cy, scale };
    const out = forward(features);
    const length = Math.hypot(out[0], out[1], out[2]) || 1;
    const joints = new Float32Array(66);
    for (let i = 0; i < 33; i++) {
      joints[i * 2] = (out[20 + i * 2] * scale + cx) / width;
      joints[i * 2 + 1] = (out[21 + i * 2] * scale + cy) / height;
    }
    const result = { forward: [out[0] / length, out[1] / length, out[2] / length], back: sigmoid(out[3]),
      swap: Array.from({ length: 16 }, (_, i) => sigmoid(out[4 + i])), joints };
    if (model.lift) {
      result.joints3d = Float32Array.from(out.subarray(86, 185));
      result.sigma3d = Float32Array.from(out.subarray(185, 218), v => Math.exp(clamp(v, -5, 1)));
    }
    return result;
  }

  reset();
  return { step, reset, _features: features };
}

/** Network joints as the model saw them in training: root (mid-hip) relative,
 * in the network body's own torso lengths, clamped; zeros when absent. */
function putNetworkJoints(features, at, joints) {
  const n = NET_JOINT_IDS.length;
  if (!joints || joints.length < n * 3) { features.fill(0, at, at + n * 3); return; }
  const j = k => [joints[k * 3], joints[k * 3 + 1], joints[k * 3 + 2]];
  const idx = id => NET_JOINT_IDS.indexOf(id);
  const [ls, rs, lh, rh] = [j(idx(11)), j(idx(12)), j(idx(23)), j(idx(24))];
  const root = [0, 1, 2].map(k => (lh[k] + rh[k]) / 2), sh = [0, 1, 2].map(k => (ls[k] + rs[k]) / 2);
  const torso = Math.max(.1, Math.hypot(sh[0] - root[0], sh[1] - root[1], sh[2] - root[2]));
  for (let k = 0; k < n; k++) for (let c = 0; c < 3; c++) features[at + k * 3 + c] = clamp((joints[k * 3 + c] - root[c]) / torso, -4, 4);
}

/** +1 when MediaPipe's left/right labels describe a body facing the camera,
 * -1 its back, null when side-on or the shoulders are not seen. */
export function labelFacing(pose, width, height) {
  if (!(pose[11]?.visibility > 0 && pose[12]?.visibility > 0)) return null;
  const p = i => [pose[i].x * width, pose[i].y * height];
  const [a, b, c, d] = [p(11), p(12), p(23), p(24)];
  const lr = [(a[0] + c[0]) / 2 - (b[0] + d[0]) / 2, (a[1] + c[1]) / 2 - (b[1] + d[1]) / 2];
  const up = [(a[0] + b[0]) / 2 - (c[0] + d[0]) / 2, (a[1] + b[1]) / 2 - (c[1] + d[1]) / 2];
  const { scale } = frameNormalization(pose, width, height);
  if (Math.hypot(a[0] - b[0], a[1] - b[1]) < .25 * scale) return null;
  const cross = lr[0] * up[1] - lr[1] * up[0];
  return cross < 0 ? 1 : cross > 0 ? -1 : null;
}

/** A copy of the landmarks with the given left/right pairs exchanged. */
export function swapPairs(pose, swapped) {
  const out = pose.slice();
  POSE_PAIRS.forEach(([a, b], i) => { if (swapped[i]) { out[a] = pose[b]; out[b] = pose[a]; } });
  return out;
}

/** Torso direction (unit, camera axes X right/Y down/Z forward) from MHR
 * keypoints 5,6,9,10 (left/right shoulder, left/right hip) given as 4×3. */
export function torsoForward(k) {
  const lr = [(k[0] + k[6]) / 2 - (k[3] + k[9]) / 2, (k[1] + k[7]) / 2 - (k[4] + k[10]) / 2, (k[2] + k[8]) / 2 - (k[5] + k[11]) / 2];
  const up = [(k[0] + k[3]) / 2 - (k[6] + k[9]) / 2, (k[1] + k[4]) / 2 - (k[7] + k[10]) / 2, (k[2] + k[5]) / 2 - (k[8] + k[11]) / 2];
  const f = [lr[1] * up[2] - lr[2] * up[1], lr[2] * up[0] - lr[0] * up[2], lr[0] * up[1] - lr[1] * up[0]];
  const length = Math.hypot(f[0], f[1], f[2]);
  return length > 1e-9 ? f.map(v => v / length) : null;
}

/** Face direction in camera axes from a MediaPipe facial transformation matrix
 * (column-major 4×4, OpenGL camera: Y up, Z towards the viewer). */
export function faceForward(matrix) {
  const d = matrix?.data ?? matrix;
  if (!d || d.length < 11) return null;
  const f = [d[8], -d[9], -d[10]], length = Math.hypot(f[0], f[1], f[2]);
  return length > 1e-6 && Number.isFinite(length) ? f.map(v => v / length) : null;
}
