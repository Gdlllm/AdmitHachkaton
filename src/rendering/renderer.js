import {
  BODY_CONNECTIONS, HAND_CONNECTIONS, FACE_TESSELLATION,
  FACE_OVAL, FACE_LIPS, FACE_LEFT_EYE, FACE_RIGHT_EYE,
  FACE_LEFT_BROW, FACE_RIGHT_BROW, FACE_LEFT_IRIS, FACE_RIGHT_IRIS,
} from './topology.js';

const TAU = Math.PI * 2;
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const finitePositive = value => Number.isFinite(value) && value > 0;
const COLORS = { left: '#68e6ef', right: '#ffc393', neutral: '#e6fafa', face: '#a4ecdf' };
// Pose's six coarse hand landmarks are replaced by the detailed hand model.
const BODY_ONLY = BODY_CONNECTIONS.filter(([a, b]) => ![a, b].some(i => i >= 17 && i <= 22));
const BODY_INDICES = [...new Set(BODY_ONLY.flat())];
const LEFT_BODY = new Set([11, 13, 15, 23, 25, 27, 29, 31]);
const RIGHT_BODY = new Set([12, 14, 16, 24, 26, 28, 30, 32]);
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
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    const project = (landmarks, pose = false) => (landmarks ?? []).map(point =>
      isDrawableLandmark(point, { pose }) ? projectLandmark(point, viewport, mirror) : null);
    const bodyPoints = project(frame.pose?.landmarks, true);
    for (const match of matchBodyWrists(frame.pose?.landmarks, frame.hands, videoWidth, videoHeight)) {
      bodyPoints[match.poseIndex] = projectLandmark(frame.hands[match.handIndex].landmarks[0], viewport, mirror);
    }
    this.drawBody(bodyPoints, opacity);
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

  drawBody(points, opacity) {
    const torso = BODY_ONLY.filter(([a, b]) => LEFT_BODY.has(a) !== LEFT_BODY.has(b));
    const left = BODY_ONLY.filter(([a, b]) => LEFT_BODY.has(a) && LEFT_BODY.has(b));
    const right = BODY_ONLY.filter(([a, b]) => RIGHT_BODY.has(a) && RIGHT_BODY.has(b));
    // A restrained dark underlay keeps thin bones legible on bright clothing.
    this.connections(points, BODY_ONLY, '#092127', 3.3, opacity * 0.30);
    this.connections(points, torso, COLORS.neutral, 1.65, opacity * 0.82);
    this.connections(points, left, COLORS.left, 1.8, opacity * 0.95);
    this.connections(points, right, COLORS.right, 1.8, opacity * 0.95);
    this.dots(points, BODY_INDICES, 2.6, '#10282a', opacity * 0.50);
    this.dots(points, BODY_INDICES, 1.7, COLORS.neutral, opacity * 0.96);
  }

  drawFace(points, opacity) {
    const size = bounds(points);
    if (!size) return;
    const scale = clamp(size.width / 260, 0.65, 1.25);
    // At a distance a triangle mesh becomes a solid mask. Actual contours remain
    // visible, with the measured vertices/tessellation appearing in close-up.
    const detail = clamp((size.width - 65) / 105, 0, 1);
    if (detail > 0) {
      this.connections(points, FACE_TESSELLATION, COLORS.face, 0.45, opacity * detail * 0.17);
      this.dots(points, points.map((_, index) => index), 0.63 * scale, '#e2fff6', opacity * detail * 0.60);
    }
    this.connections(points, FACE_OVAL, COLORS.face, 0.9 * scale, opacity * 0.62);
    this.connections(points, FACE_LIPS, '#fff1e8', 0.95 * scale, opacity * 0.85);
    this.connections(points, FACE_LEFT_BROW, COLORS.face, 0.95 * scale, opacity * 0.76);
    this.connections(points, FACE_RIGHT_BROW, COLORS.face, 0.95 * scale, opacity * 0.76);
    this.connections(points, FACE_LEFT_EYE, '#f0fff8', 1.05 * scale, opacity * 0.90);
    this.connections(points, FACE_RIGHT_EYE, '#f0fff8', 1.05 * scale, opacity * 0.90);
    this.connections(points, FACE_LEFT_IRIS, COLORS.left, 1.25 * scale, opacity * 0.95);
    this.connections(points, FACE_RIGHT_IRIS, COLORS.right, 1.25 * scale, opacity * 0.95);
    this.dots(points, [468, 473], 1.1 * scale, '#ffffff', opacity * 0.95);
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
