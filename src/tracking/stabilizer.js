/**
 * Temporal stabilization for the capture pipeline. Coordinates keep their source
 * units: image landmarks are normalized; world landmarks remain model metres.
 *
 * new CaptureStabilizer().update(frame) accepts:
 * { timestamp (monotonic ms), sourceWidth, sourceHeight,
 *   pose: {landmarks, worldLandmarks} | null,
 *   face: {landmarks, blendshapes, transformationMatrix} | null,
 *   hands: [{landmarks, worldLandmarks, handedness, score}], ...metadata }
 *
 * Returns fresh point objects, with `drawConfidence` in [0,1]. Consumers must
 * skip points with drawConfidence === 0, and never infer gestures from them.
 * Each detected part gains `track: {id, ageMs, quality, reacquired}`. Hand ids
 * follow spatial trajectories rather than the order/handedness of detections.
 * An empty detection removes the part immediately. There is no visual coast or
 * interpolation across missing detections. Track identity alone survives a
 * short gap; its landmark filters do not. Blendshapes/matrices are left intact.
 * Source dimensions make display smoothing bounds pixel-based. `width/height`
 * aliases are accepted; missing dimensions default to 1280×720. Smoothing
 * residual metrics compare filtered landmarks to detector output, NOT truth.
 *
 * Adaptive filtering follows Casiez et al.'s 1€ filter, with independent tuning
 * for each coordinate space: https://gery.casiez.net/1euro/ . Tuning is a latency
 * and jitter tradeoff, not a claim of improved model measurement accuracy.
 */

const clamp = (value, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, value));
const alpha = (cutoff, dt) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
const finiteXY = point => point && Number.isFinite(point.x) && Number.isFinite(point.y);
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const median = values => {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

const DEFAULTS = {
  resetAfterMs: 250,
  handIdentityAfterMs: 650,
  poseVisibility: 0.45,
  posePresence: 0.45,
  // Beta uses normalized-image units. World filters have separate parameters.
  pose: { minCutoff: 1.8, beta: 14, derivativeCutoff: 1.5, outlierDistance: 0.48, maxLagPx: 4 },
  face: { minCutoff: 3.2, beta: 22, derivativeCutoff: 2, outlierDistance: 0.14, maxLagPx: 2 },
  hand: { minCutoff: 2.5, beta: 20, derivativeCutoff: 2, outlierDistance: 0.25, maxLagPx: 3 },
  world: { minCutoff: 2.2, beta: 5, derivativeCutoff: 1.5 },
};

/** Face and hand landmark messages often have no meaningful confidence fields.
 * Their optional handedness score must not be treated as geometry confidence. */
export function landmarkConfidence(point, kind = 'pose', options = DEFAULTS) {
  if (!finiteXY(point) || (point.z !== undefined && !Number.isFinite(point.z))) return 0;
  if (point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) return 0;
  if (kind !== 'pose') return 1;
  const visibility = point.visibility ?? 1;
  const presence = point.presence ?? 1;
  if (!Number.isFinite(visibility) || !Number.isFinite(presence)) return 0;
  if (visibility < options.poseVisibility || presence < options.posePresence) return 0;
  return clamp(Math.min(visibility, presence));
}

class VectorFilter {
  constructor(settings) { this.settings = settings; this.reset(); }
  reset() { this.raw = null; this.filtered = null; this.derivative = null; this.timestamp = null; this.lastAlpha = 1; }
  update(point, timestamp, dimensions = null, alphaOverride = null) {
    const axes = ['x', 'y', 'z'].filter(axis => Number.isFinite(point[axis]));
    if (!this.raw || timestamp <= this.timestamp) {
      this.raw = { ...point };
      this.filtered = { ...point };
      this.derivative = Object.fromEntries(axes.map(axis => [axis, 0]));
      this.timestamp = timestamp;
      this.lastAlpha = 1;
      return { ...point };
    }
    const dt = Math.max(1 / 240, (timestamp - this.timestamp) / 1000);
    const derivativeAlpha = alpha(this.settings.derivativeCutoff, dt);
    let squaredSpeed = 0;
    for (const axis of axes) {
      const rawDerivative = (point[axis] - (this.raw[axis] ?? point[axis])) / dt;
      const previousDerivative = this.derivative[axis] ?? 0;
      this.derivative[axis] = previousDerivative + derivativeAlpha * (rawDerivative - previousDerivative);
      squaredSpeed += this.derivative[axis] ** 2;
    }
    const cutoff = this.settings.minCutoff + this.settings.beta * Math.sqrt(squaredSpeed);
    let smoothing = alphaOverride ?? alpha(cutoff, dt);
    if (dimensions && Number.isFinite(this.settings.maxLagPx)) {
      const distancePx = Math.hypot(
        (point.x - this.filtered.x) * dimensions.width,
        (point.y - this.filtered.y) * dimensions.height,
      );
      // Synced bitmap + old smoothed landmarks is still visibly misaligned.
      // Bound that residual in source pixels, without predicting new positions.
      if (distancePx > 0) smoothing = Math.max(smoothing, 1 - this.settings.maxLagPx / distancePx);
    }
    this.lastAlpha = smoothing;
    const result = { ...point };
    for (const axis of axes) {
      const previous = this.filtered[axis] ?? point[axis];
      result[axis] = previous + smoothing * (point[axis] - previous);
    }
    this.timestamp = timestamp;
    this.raw = { ...point };
    this.filtered = result;
    return result;
  }
}

class PartFilter {
  constructor(kind, options) {
    this.kind = kind;
    this.options = options;
    this.image = [];
    this.world = [];
    this.raw = [];
    this.timestamp = null;
  }
  reset() {
    this.image = [];
    this.world = [];
    this.raw = [];
    this.timestamp = null;
  }
  update(part, timestamp, dimensions) {
    const points = part.landmarks;
    const gap = this.timestamp === null || timestamp - this.timestamp > this.options.resetAfterMs;
    const reset = gap || points.length !== this.image.length;
    if (reset) this.reset();
    const confidence = points.map(point => landmarkConfidence(point, this.kind, this.options));
    const movement = [];
    for (let i = 0; i < points.length; i++) {
      if (confidence[i] && this.raw[i]) {
        movement.push({ x: points[i].x - this.raw[i].x, y: points[i].y - this.raw[i].y });
      }
    }
    // A whole part can move quickly. Only isolated, extreme jumps relative to
    // the group's median translation are rejected, never ordinary fast motion.
    const translation = { x: median(movement.map(p => p.x)), y: median(movement.map(p => p.y)) };
    const residuals = movement.map(p => Math.hypot(p.x - translation.x, p.y - translation.y));
    const threshold = Math.max(this.options[this.kind].outlierDistance, median(residuals) * 8);
    let rejected = 0, residualSum = 0, residualMax = 0, validCount = 0;
    const landmarks = points.map((point, i) => {
      const old = this.raw[i];
      const residual = old && finiteXY(point) ? Math.hypot(point.x - old.x - translation.x, point.y - old.y - translation.y) : 0;
      const outlier = movement.length >= 5 && residual > threshold;
      if (!confidence[i] || outlier) {
        this.image[i] = null;
        this.raw[i] = null;
        confidence[i] = 0;
        if (outlier) rejected++;
        return { ...point, drawConfidence: 0 };
      }
      this.image[i] ??= new VectorFilter(this.options[this.kind]);
      const filtered = this.image[i].update(point, timestamp, dimensions);
      const residualPx = Math.hypot((point.x - filtered.x) * dimensions.width, (point.y - filtered.y) * dimensions.height);
      residualSum += residualPx ** 2;
      residualMax = Math.max(residualMax, residualPx);
      validCount++;
      this.raw[i] = { ...point };
      return { ...filtered, drawConfidence: confidence[i] };
    });
    // Do not apply normalized image bounds or image-scale thresholds to metres.
    const worldLandmarks = Array.isArray(part.worldLandmarks) ? part.worldLandmarks.map((point, i) => {
      if (!confidence[i] || !finiteXY(point) || !Number.isFinite(point.z)) {
        this.world[i] = null;
        return { ...point, drawConfidence: 0 };
      }
      this.world[i] ??= new VectorFilter(this.options.world);
      // Same temporal blend as the corresponding image point, without mixing
      // the two coordinate spaces or treating model metres as image pixels.
      return { ...this.world[i].update(point, timestamp, null, this.image[i].lastAlpha), drawConfidence: confidence[i] };
    }) : part.worldLandmarks;
    if (!Array.isArray(part.worldLandmarks) || !part.worldLandmarks.length) this.world = [];
    this.timestamp = timestamp;
    const quality = confidence.length ? confidence.reduce((sum, value) => sum + value, 0) / confidence.length : 0;
    return {
      part: { ...part, landmarks, ...(worldLandmarks !== undefined ? { worldLandmarks } : {}) },
      quality, rejected, reset,
      smoothing: { maxResidualPx: residualMax, squaredResidualPx: residualSum, pointCount: validCount },
    };
  }
}

function anchorOf(hand) {
  const points = hand.landmarks;
  const wrist = points[0];
  if (landmarkConfidence(wrist, 'hand')) return { x: wrist.x, y: wrist.y };
  const palm = [0, 5, 9, 13, 17].map(i => points[i]).filter(p => landmarkConfidence(p, 'hand'));
  if (!palm.length) return null;
  return { x: median(palm.map(p => p.x)), y: median(palm.map(p => p.y)) };
}

function shapeOf(hand) {
  const points = hand.landmarks;
  const anchor = anchorOf(hand);
  if (!anchor) return [];
  const scale = Math.max(0.025, finiteXY(points[9]) ? distance(anchor, points[9]) : 0.08);
  return [4, 8, 12, 16, 20].map(i => finiteXY(points[i]) ? distance(anchor, points[i]) / scale : null);
}

function handLabel(hand) {
  const value = hand.handedness;
  if (typeof value === 'string') return value.toLowerCase();
  if (Array.isArray(value)) return handLabel({ handedness: value[0] });
  return (value?.categoryName ?? value?.displayName ?? '').toLowerCase();
}

/** Single-person full-body, face and two-hand stabilization. */
export class CaptureStabilizer {
  constructor(options = {}) {
    this.options = { ...DEFAULTS, ...options };
    for (const kind of ['pose', 'face', 'hand', 'world']) {
      this.options[kind] = { ...DEFAULTS[kind], ...options[kind] };
    }
    this.reset();
  }

  reset() {
    this.pose = new PartFilter('pose', this.options);
    this.face = new PartFilter('face', this.options);
    this.bodyState = { pose: null, face: null };
    this.hands = new Map();
    this.nextHand = 1;
    this.timestamp = null;
  }

  update(frame) {
    if (!Number.isFinite(frame?.timestamp)) throw new TypeError('Capture frame timestamp must be finite milliseconds.');
    const timestamp = frame.timestamp;
    const dimensions = {
      width: frame.sourceWidth ?? frame.width ?? 1280,
      height: frame.sourceHeight ?? frame.height ?? 720,
    };
    if (!(Number.isFinite(dimensions.width) && dimensions.width > 0 && Number.isFinite(dimensions.height) && dimensions.height > 0)) {
      throw new TypeError('Capture source dimensions must be positive finite pixels.');
    }
    const clockReset = this.timestamp !== null && timestamp <= this.timestamp;
    if (clockReset) this.reset();
    const output = { ...frame, timestamp, hands: [] };
    const tracking = { timestamp, clockReset, rejectedPoints: 0, pose: null, face: null, hands: [], smoothing: { maxResidualPx: 0, rmsResidualPx: 0, pointCount: 0 } };
    let squaredResidualPx = 0;
    const collectResidual = result => {
      tracking.smoothing.maxResidualPx = Math.max(tracking.smoothing.maxResidualPx, result.smoothing.maxResidualPx);
      tracking.smoothing.pointCount += result.smoothing.pointCount;
      squaredResidualPx += result.smoothing.squaredResidualPx;
    };
    for (const kind of ['pose', 'face']) {
      const source = frame[kind];
      if (!Array.isArray(source?.landmarks) || source.landmarks.length === 0) {
        this[kind].reset();
        this.bodyState[kind] = null;
        output[kind] = null;
        continue;
      }
      const result = this[kind].update(source, timestamp, dimensions);
      const previous = this.bodyState[kind];
      const start = previous && !result.reset ? previous.start : timestamp;
      const track = { id: kind, ageMs: timestamp - start, quality: result.quality, reacquired: !previous || result.reset };
      output[kind] = { ...result.part, track };
      tracking[kind] = track;
      tracking.rejectedPoints += result.rejected;
      collectResidual(result);
      this.bodyState[kind] = { start };
    }

    // Retain identities briefly, but never retain their rendered positions.
    for (const [id, state] of this.hands) {
      if (timestamp - state.lastSeen > this.options.handIdentityAfterMs) this.hands.delete(id);
    }
    const detected = (frame.hands ?? []).filter(hand => Array.isArray(hand?.landmarks) && hand.landmarks.length && anchorOf(hand));
    const candidates = [...this.hands.values()];
    const assignment = this.assignHands(detected, candidates, timestamp);
    const seen = new Set();
    for (let index = 0; index < detected.length; index++) {
      const hand = detected[index];
      const currentAnchor = anchorOf(hand);
      let state = assignment[index];
      const reacquired = !state || !state.active || timestamp - state.lastSeen > this.options.resetAfterMs;
      if (!state) {
        state = { id: `hand-${this.nextHand++}`, filter: new PartFilter('hand', this.options), start: timestamp, lastSeen: timestamp, anchor: currentAnchor, velocity: { x: 0, y: 0 }, label: handLabel(hand), shape: shapeOf(hand) };
        this.hands.set(state.id, state);
      }
      if (reacquired) {
        state.filter.reset();
        state.velocity = { x: 0, y: 0 };
        state.start = timestamp;
      } else {
        const dt = Math.max(1 / 240, (timestamp - state.lastSeen) / 1000);
        // A short, bounded prediction supports crossing hands without flinging
        // identities far away after a dropout or a detector jump.
        state.velocity = {
          x: clamp((currentAnchor.x - state.anchor.x) / dt, -3, 3),
          y: clamp((currentAnchor.y - state.anchor.y) / dt, -3, 3),
        };
      }
      const result = state.filter.update(hand, timestamp, dimensions);
      const track = { id: state.id, ageMs: timestamp - state.start, quality: result.quality, reacquired, handedness: state.label || handLabel(hand) };
      output.hands.push({ ...result.part, track });
      tracking.hands.push(track);
      tracking.rejectedPoints += result.rejected;
      collectResidual(result);
      state.anchor = currentAnchor;
      state.shape = shapeOf(hand);
      state.lastSeen = timestamp;
      state.active = true;
      // Classifier flips do not replace a confirmed label used for tie-breaking.
      state.label ||= handLabel(hand);
      seen.add(state.id);
    }
    for (const state of this.hands.values()) {
      if (!seen.has(state.id)) { state.active = false; state.filter.reset(); }
    }
    this.timestamp = timestamp;
    tracking.smoothing.rmsResidualPx = tracking.smoothing.pointCount ? Math.sqrt(squaredResidualPx / tracking.smoothing.pointCount) : 0;
    output.tracking = tracking;
    return output;
  }

  assignHands(detections, candidates, timestamp) {
    const costs = detections.map(hand => candidates.map(state => {
      const anchor = anchorOf(hand);
      const elapsed = timestamp - state.lastSeen;
      const dt = state.active ? Math.min(0.07, elapsed / 1000) : 0;
      const predicted = { x: state.anchor.x + state.velocity.x * dt, y: state.anchor.y + state.velocity.y * dt };
      const spatial = distance(anchor, predicted);
      if (spatial > (state.active ? 0.45 : 0.28)) return Infinity;
      const shape = shapeOf(hand);
      const differences = shape.map((value, i) => value !== null && state.shape[i] !== null && state.shape[i] !== undefined ? Math.abs(value - state.shape[i]) : 0);
      const shapeCost = Math.min(0.06, median(differences) * 0.03);
      const labelCost = state.label && handLabel(hand) && state.label !== handLabel(hand) ? 0.035 : 0;
      return spatial + shapeCost + labelCost;
    }));
    // Exhaustive matching is tiny for two hands and avoids greedy swaps.
    let bestCost = Infinity;
    let best = [];
    const visit = (index, used, chosen, cost) => {
      if (cost >= bestCost) return;
      if (index === detections.length) { bestCost = cost; best = chosen.slice(); return; }
      for (let c = 0; c < candidates.length; c++) {
        if (!used.has(c) && Number.isFinite(costs[index][c])) {
          used.add(c); chosen.push(candidates[c]);
          visit(index + 1, used, chosen, cost + costs[index][c]);
          chosen.pop(); used.delete(c);
        }
      }
      chosen.push(null);
      visit(index + 1, used, chosen, cost + 0.5);
      chosen.pop();
    };
    visit(0, new Set(), [], 0);
    return best;
  }
}
