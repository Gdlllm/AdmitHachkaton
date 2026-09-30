import {
  BODY_CONNECTIONS, HAND_CONNECTIONS, FACE_TESSELLATION,
  FACE_OVAL, FACE_LIPS, FACE_LEFT_EYE, FACE_RIGHT_EYE,
  FACE_LEFT_BROW, FACE_RIGHT_BROW, FACE_LEFT_IRIS, FACE_RIGHT_IRIS,
} from './topology.js';

const TAU = Math.PI * 2;
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const finitePositive = value => Number.isFinite(value) && value > 0;
const COLORS = { left: '#68e6ef', right: '#ffc393', neutral: '#e6fafa', face: '#d5fff4' };
// Pose's six coarse hand landmarks are replaced by the detailed hand model.
const BODY_ONLY = BODY_CONNECTIONS.filter(([a, b]) => ![a, b].some(i => i >= 17 && i <= 22));
const BODY_INDICES = [...new Set(BODY_ONLY.flat())];
const LEFT_BODY = new Set([11, 13, 15, 23, 25, 27, 29, 31]);
const RIGHT_BODY = new Set([12, 14, 16, 24, 26, 28, 30, 32]);
const MISTAKE = '#ff4d5e';
const GUESS = '#b9a2ff';                 // joints the model filled in (hidden or out of frame)
const POSE_PAIRS = [[1, 4], [2, 5], [3, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22],
  [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]];

/** Body points to draw: MediaPipe's where it sees the joint (labels corrected by
 * the model), the model's own estimate where it does not. Returns {points, guessed}. */
export function modelBodyPoints(frame, viewport, mirror) {
  const sense = frame?.surface?.active ? frame.surface.sense : null;
  const model = sense?.joints2d, landmarks = frame?.pose?.landmarks;
  if (!model || model.length < 66 || !Array.isArray(landmarks) || landmarks.length < 33) return null;
  const pose = landmarks.slice();
  (sense.swappedPairs ?? []).forEach((swap, i) => { if (swap) { const [a, b] = POSE_PAIRS[i]; pose[a] = landmarks[b]; pose[b] = landmarks[a]; } });
  const guessed = new Set(), points = [];
  for (let i = 0; i < 33; i++) {
    const p = pose[i];
    if (isDrawableLandmark(p, { pose: true })) { points.push(projectLandmark(p, viewport, mirror)); continue; }
    // Face points are only drawn when seen: the model places the head from the body alone.
    if (i <= 10) { points.push(null); continue; }
    points.push(projectLandmark({ x: model[i * 2], y: model[i * 2 + 1] }, viewport, mirror));
    guessed.add(i);
  }
  return { points, guessed };
}
const FINGERTIPS = new Set([4, 8, 12, 16, 20]);

/** CSS pixel rectangle that exactly matches object-fit: contain. */
export function containViewport(width, height, videoWidth, videoHeight) {
  if (![width, height, videoWidth, videoHeight].every(finitePositive)) return null;
  const scale = Math.min(width / videoWidth, height / videoHeight);
  const fittedWidth = Math.min(width, videoWidth * scale), fittedHeight = Math.min(height, videoHeight * scale);
  return { x: (width - fittedWidth) / 2, y: (height - fittedHeight) / 2,
    width: fittedWidth, height: fittedHeight, scale };
}

/** Projection never mutates model coordinates. Mirroring happens exactly once. */
export function projectLandmark(point, viewport, mirror = true) {
  if (!viewport || !point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return null;
  return { x: viewport.x + (mirror ? 1 - point.x : point.x) * viewport.width,
    y: viewport.y + point.y * viewport.height };
}

/** Hand/face models do not provide reliable per-point visibility probabilities. */
export function isDrawableLandmark(point, { pose = false, margin = 0.025 } = {}) {
  if (!point || point.valid === false || point.tracked === false || point.predicted === true || point.drawConfidence === 0 ||
      !Number.isFinite(point.x) || !Number.isFinite(point.y)) return false;
  if (point.x < -margin || point.x > 1 + margin || point.y < -margin || point.y > 1 + margin) return false;
  if (pose && ((point.visibility ?? 1) < 0.45 || (point.presence ?? 1) < 0.45)) return false;
  return true;
}

/** Use arrival time for visual freshness; capture timestamp remains suitable for filtering. */
export function trackingOpacity(frame, now, freshFor = 180, expiresAfter = 480) {
  const timestamp = frame?.receivedAt ?? frame?.timestamp;
  if (!Number.isFinite(timestamp) || !Number.isFinite(now)) return 1;
  const age = Math.max(0, now - timestamp);
  return clamp((expiresAfter - age) / Math.max(1, expiresAfter - freshFor), 0, 1);
}

/** Match measured wrist points for drawing only. Image/world tracking data is
 * never altered, and ambiguous crossings retain the pose model's own endpoint.
 * Distances use source pixels so non-square images do not distort the gate. */
export function matchBodyWrists(poseLandmarks, hands, videoWidth, videoHeight) {
  if (!Array.isArray(poseLandmarks) || !Array.isArray(hands) ||
      ![videoWidth, videoHeight].every(finitePositive)) return [];
  const distance = (a, b) => Math.hypot((a.x - b.x) * videoWidth, (a.y - b.y) * videoHeight);
  const wrists = [15, 16].filter(index => isDrawableLandmark(poseLandmarks[index], { pose: true }) &&
    isDrawableLandmark(poseLandmarks[index - 2], { pose: true }));
  const detailed = hands.map((hand, handIndex) => {
    const wrist = hand.landmarks?.[0], palm = hand.landmarks?.[9];
    if (!isDrawableLandmark(wrist) || !isDrawableLandmark(palm)) return null;
    const palmLength = distance(wrist, palm);
    // Tiny hands do not support a reliable cross-model anatomical association.
    return palmLength >= 12 ? { handIndex, wrist, palmLength } : null;
  }).filter(Boolean);
  const pairs = wrists.flatMap(poseIndex => detailed.map(hand => {
    const forearmLength = distance(poseLandmarks[poseIndex], poseLandmarks[poseIndex - 2]);
    return { poseIndex, handIndex: hand.handIndex, distancePx: distance(poseLandmarks[poseIndex], hand.wrist),
      thresholdPx: Math.max(8, Math.min(0.3 * forearmLength, 0.6 * hand.palmLength)),
      separationPx: Math.max(4, 0.25 * hand.palmLength) };
  }));
  return pairs.filter(pair => {
    if (pair.distancePx > pair.thresholdPx) return false;
    // Mutual uniqueness prevents attaching an arm to the other hand when wrists
    // cross, or letting two arms share one detailed wrist.
    return pairs.every(other => other === pair ||
      (other.poseIndex !== pair.poseIndex && other.handIndex !== pair.handIndex) ||
      other.distancePx - pair.distancePx >= pair.separationPx);
  });
}

function bounds(points) {
  const valid = points.filter(Boolean);
  if (!valid.length) return null;
  return { width: Math.max(...valid.map(p => p.x)) - Math.min(...valid.map(p => p.x)),
    height: Math.max(...valid.map(p => p.y)) - Math.min(...valid.map(p => p.y)) };
}

function handednessLabel(hand) {
  const value = hand.track?.handedness || hand.handedness;
  if (typeof value === 'string') return value.toLowerCase();
  if (Array.isArray(value)) return String(value[0]?.categoryName ?? value[0]?.displayName ?? '').toLowerCase();
  return String(value?.categoryName ?? value?.displayName ?? '').toLowerCase();
}

// Pose joints of the named body parts in frame.motion.highlight (see src/motion/kinematics.js BODY_PARTS).
const PART_JOINTS = {
  leftKnee: [23, 25, 27], rightKnee: [24, 26, 28], leftHip: [11, 23, 25], rightHip: [12, 24, 26],
  leftArm: [11, 13, 15], rightArm: [12, 14, 16], leftElbow: [11, 13, 15], rightElbow: [12, 14, 16],
  leftLeg: [23, 25, 27, 29, 31], rightLeg: [24, 26, 28, 30, 32], leftFoot: [27, 29, 31], rightFoot: [28, 30, 32],
  torso: [11, 12, 23, 24], back: [11, 12, 23, 24], shoulders: [11, 12], hips: [23, 24],
};
function highlightJoints(parts) {
  if (!Array.isArray(parts) || !parts.length) return null;
  return new Set(parts.flatMap(part => PART_JOINTS[part] ?? []));
}

export class CaptureRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d', { alpha: true });
    if (!this.context) throw new Error('Canvas 2D is not available.');
    this.width = 1; this.height = 1; this.dpr = 1;
  }

  setSize(width, height, dpr = 1) {
    if (![width, height].every(finitePositive)) return;
    this.width = width; this.height = height;
    this.dpr = finitePositive(dpr) ? dpr : 1;
    const pixelWidth = Math.max(1, Math.round(width * this.dpr));
    const pixelHeight = Math.max(1, Math.round(height * this.dpr));
    if (this.canvas.width !== pixelWidth) this.canvas.width = pixelWidth;
    if (this.canvas.height !== pixelHeight) this.canvas.height = pixelHeight;
    if (this.canvas.style) {
      this.canvas.style.width = `${width}px`;
      this.canvas.style.height = `${height}px`;
    }
  }

  clear() {
    const ctx = this.context;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  /**
   * source is an optional ImageBitmap of the exact inference frame. The caller
   * owns/closes it. Keeping it alive until the next result avoids drawing joints
   * on a different live video frame. With no source this draws a transparent overlay.
   * Optional frame.surfaceImage is a transparent surface overlay of any pixel
   * size in source orientation, stretched over the image like the source.
   * Its lifetime also belongs to the caller.
   */
  draw(frame, { videoWidth, videoHeight, mirror = true, now = performance.now(), source = null } = {}) {
    this.clear();
    if (!frame) return;
    const viewport = containViewport(this.width, this.height, videoWidth, videoHeight);
    if (!viewport) return;
    const opacity = trackingOpacity(frame, now);
    if (opacity <= 0) return;
    const ctx = this.context;
    ctx.save();
    ctx.beginPath(); ctx.rect(viewport.x, viewport.y, viewport.width, viewport.height); ctx.clip();
    const image = source ?? frame.source;
    if (image && image.width > 0 && image.height > 0) {
      ctx.save();
      ctx.translate(mirror ? viewport.x + viewport.width : viewport.x, viewport.y);
      ctx.scale(mirror ? -1 : 1, 1);
      ctx.drawImage(image, 0, 0, viewport.width, viewport.height);
      ctx.restore();
    }
    const surfaceImage = frame.surfaceImage;
    if (surfaceImage && finitePositive(surfaceImage.width) && finitePositive(surfaceImage.height)) {
      ctx.save();
      ctx.translate(mirror ? viewport.x + viewport.width : viewport.x, viewport.y);
      ctx.scale(mirror ? -1 : 1, 1);
      ctx.globalAlpha = opacity;
      ctx.drawImage(surfaceImage, 0, 0, viewport.width, viewport.height);
      ctx.restore();
    }
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    const project = (landmarks, pose = false) => (landmarks ?? []).map(point =>
      isDrawableLandmark(point, { pose }) ? projectLandmark(point, viewport, mirror) : null);
    const model = modelBodyPoints(frame, viewport, mirror);
    const bodyPoints = model ? model.points : project(frame.pose?.landmarks, true);
    for (const match of matchBodyWrists(frame.pose?.landmarks, frame.hands, videoWidth, videoHeight)) {
      if (!model?.guessed.has(match.poseIndex)) bodyPoints[match.poseIndex] = projectLandmark(frame.hands[match.handIndex].landmarks[0], viewport, mirror);
    }
    this.drawBody(bodyPoints, opacity, highlightJoints(frame.motion?.highlight), model?.guessed ?? null);
    this.drawFace(project(frame.face?.landmarks), opacity);
    for (const hand of frame.hands ?? []) {
      const side = handednessLabel(hand);
      const color = side === 'left' ? COLORS.left : side === 'right' ? COLORS.right : COLORS.neutral;
      this.drawHand(project(hand.landmarks), color, opacity);
    }
    ctx.restore();
  }

  connections(points, connections, color, width, opacity) {
    if (opacity <= 0) return;
    const ctx = this.context;
    ctx.beginPath();
    let count = 0;
    for (const [a, b] of connections) {
      const p = points[a], q = points[b];
      if (!p || !q) continue;
      ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); count++;
    }
    if (count) {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.globalAlpha = opacity;
      ctx.stroke(); ctx.globalAlpha = 1;
    }
  }

  dots(points, indices, radius, color, opacity) {
    const ctx = this.context;
    ctx.beginPath();
    let count = 0;
    for (const index of indices) {
      const p = points[index];
      if (!p) continue;
      ctx.moveTo(p.x + radius, p.y); ctx.arc(p.x, p.y, radius, 0, TAU); count++;
    }
    if (count) {
      ctx.fillStyle = color; ctx.globalAlpha = opacity; ctx.fill(); ctx.globalAlpha = 1;
    }
  }

  drawBody(points, opacity, highlight = null, guessed = null) {
    // With the model's skeleton (guessed set) the bones are bolder, and the
    // filled-in ones are dashed amber so it is clear what the camera did not see.
    const bold = guessed ? 1.7 : 1;
    const known = guessed ? BODY_ONLY.filter(([a, b]) => !guessed.has(a) && !guessed.has(b)) : BODY_ONLY;
    const torso = known.filter(([a, b]) => LEFT_BODY.has(a) !== LEFT_BODY.has(b));
    const left = known.filter(([a, b]) => LEFT_BODY.has(a) && LEFT_BODY.has(b));
    const right = known.filter(([a, b]) => RIGHT_BODY.has(a) && RIGHT_BODY.has(b));
    // A restrained dark underlay keeps thin bones legible on bright clothing.
    this.connections(points, BODY_ONLY, '#092127', 3.3 * bold, opacity * 0.30);
    this.connections(points, torso, COLORS.neutral, 1.65 * bold, opacity * 0.82);
    this.connections(points, left, COLORS.left, 1.8 * bold, opacity * 0.95);
    this.connections(points, right, COLORS.right, 1.8 * bold, opacity * 0.95);
    if (guessed?.size) {
      const ctx = this.context;
      ctx.save(); ctx.setLineDash([6, 5]);
      this.connections(points, BODY_ONLY.filter(([a, b]) => guessed.has(a) || guessed.has(b)), GUESS, 2.2, opacity * 0.85);
      ctx.restore();
      this.dots(points, [...guessed], 3, GUESS, opacity * 0.9);
    }
    const shown = guessed ? BODY_INDICES.filter(i => !guessed.has(i)) : BODY_INDICES;
    this.dots(points, shown, 2.6 * bold, '#10282a', opacity * 0.50);
    this.dots(points, shown, 1.7 * bold, COLORS.neutral, opacity * 0.96);
    if (!highlight?.size) return;
    // Body parts with a mistake: thick red bones with a soft glow.
    const bones = BODY_ONLY.filter(([a, b]) => highlight.has(a) && highlight.has(b));
    const ctx = this.context;
    ctx.save(); ctx.shadowColor = MISTAKE; ctx.shadowBlur = 12;
    this.connections(points, bones, MISTAKE, 5, opacity * 0.9);
    ctx.restore();
    this.dots(points, [...highlight].filter(i => BODY_INDICES.includes(i)), 4, MISTAKE, opacity);
  }

  drawFace(points, opacity) {
    const size = bounds(points);
    if (!size) return;
    const scale = clamp(size.width / 260, 0.65, 1.25);
    // Small dots only (no mesh lines); the pupils are tracked with a ring and a centre.
    const face = points.slice(0, 468).map((_, index) => index);
    this.dots(points, face, 0.9 * scale, '#e2fff6', opacity * 0.7);
    const ctx = this.context;
    for (const [centre, ring, color] of [[468, [469, 470, 471, 472], COLORS.right], [473, [474, 475, 476, 477], COLORS.left]]) {
      const c = points[centre], rim = ring.map(i => points[i]).filter(Boolean);
      if (!c || rim.length < 3) continue;
      const radius = rim.reduce((sum, p) => sum + Math.hypot(p.x - c.x, p.y - c.y), 0) / rim.length;
      ctx.beginPath(); ctx.arc(c.x, c.y, Math.max(1.5, radius), 0, TAU);
      ctx.strokeStyle = color; ctx.lineWidth = 1.4 * scale; ctx.globalAlpha = opacity * 0.95; ctx.stroke(); ctx.globalAlpha = 1;
    }
    this.dots(points, [468, 473], 1.6 * scale, '#ffffff', opacity);
  }

  drawHand(points, color, opacity) {
    const size = bounds(points);
    if (!size) return;
    const scale = clamp(Math.max(size.width, size.height) / 210, 0.68, 1.25);
    const indices = points.map((_, index) => index);
    this.connections(points, HAND_CONNECTIONS, '#092127', 3.6 * scale, opacity * 0.30);
    this.connections(points, HAND_CONNECTIONS, color, 1.8 * scale, opacity * 0.96);
    this.dots(points, indices, 2.25 * scale, '#142e2d', opacity * 0.60);
    this.dots(points, indices, 1.5 * scale, '#eafff8', opacity * 0.95);
    this.dots(points, [...FINGERTIPS], 2.25 * scale, color, opacity);
  }
}
