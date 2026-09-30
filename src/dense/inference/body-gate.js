/**
 * Body gate for the optional surface. It switches on for a full body
 * (shoulders, hips, knees and ankles) or an upper body: both shoulders plus
 * both hips or both elbows (standing close, sitting at a desk), or a close-up
 * where only head and shoulders are in view and the body leaves the frame at
 * the bottom; a close-up needs the Face Landmarker's face around the Pose nose
 * to start. Once a face has confirmed the track (`options.confirmed`), it
 * continues without one, e.g. when the person turns away from a laptop, unless
 * a hand covers the shoulders. Without legs, Pose readily fits a body into a close hand: when a
 * detected hand spans both shoulders, any upper body needs that face. For an
 * upper body the surface is
 * hidden below the lowest visible joint, unless the body simply continues past
 * the bottom edge of the frame. This is not an independent person detector:
 * run the cross-model hand/face fusion first.
 *
 * Landmarks are MediaPipe Pose points: normalized x/y plus visibility/presence.
 * Stabilized points with drawConfidence === 0 count as missing. All returned
 * boxes and clip lines are in source pixels.
 */
export const FULL_BODY_JOINTS = Object.freeze([11, 12, 23, 24, 25, 26, 27, 28]);
const SHOULDERS = [11, 12], HIPS = [23, 24], ELBOWS = [13, 14];
// Arms, torso and legs; Pose's face and coarse palm points are not body evidence.
const BODY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
// Parent -> next joint down the body: tells "cut by the frame" from "hidden".
const DOWNWARD = [[11, 23], [12, 24], [23, 25], [24, 26], [25, 27], [26, 28]];
// A hidden joint predicted this low has left the frame rather than being covered.
const BOTTOM = 0.97;

const DEFAULTS = { enter: 0.5, keep: 0.3, edge: 0.004, minExtent: 0.15, minShoulder: 0.04, boxConfidence: 0.3 };

export function landmarkScore(point) {
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || point.drawConfidence === 0) return 0;
  const visibility = point.visibility ?? 0, presence = point.presence ?? 1;
  if (!Number.isFinite(visibility) || !Number.isFinite(presence)) return 0;
  return Math.max(0, Math.min(1, visibility, presence));
}

// Body scale in pixels: torso length, or a shoulder-width estimate of it.
function bodyUnit(landmarks, score, width, height, minConfidence) {
  const px = i => ({ x: landmarks[i].x * width, y: landmarks[i].y * height });
  const [ls, rs] = SHOULDERS.map(px), shoulderWidth = Math.hypot(ls.x - rs.x, ls.y - rs.y);
  if (!HIPS.every(i => score[i] >= minConfidence)) return Math.max(1.3 * shoulderWidth, 4);
  const [lh, rh] = HIPS.map(px);
  return Math.max(Math.hypot((ls.x + rs.x - lh.x - rh.x) / 2, (ls.y + rs.y - lh.y - rh.y) / 2), shoulderWidth, 4);
}

/** Detector-like person box from joint centres: Pose points stop at the eyes,
 * joint centres and feet, so the box is extended to the scalp and silhouette.
 * InstantHMR was trained on 1.2x square crops of such tight person boxes. */
export function personBox(landmarks, width, height, { minConfidence = DEFAULTS.boxConfidence, toBottom = false } = {}) {
  if (!Array.isArray(landmarks) || landmarks.length < 33 || !(width > 0 && height > 0)) return null;
  const score = landmarks.slice(0, 33).map(landmarkScore);
  const used = score.flatMap((value, i) => value >= minConfidence ? [{ x: landmarks[i].x * width, y: landmarks[i].y * height }] : []);
  if (used.length < 4 || !SHOULDERS.every(i => score[i] >= minConfidence)) return null;
  const unit = bodyUnit(landmarks, score, width, height, minConfidence);
  let x0 = Math.min(...used.map(p => p.x)), x1 = Math.max(...used.map(p => p.x));
  let y0 = Math.min(...used.map(p => p.y)), y1 = Math.max(...used.map(p => p.y));
  const headSeen = score.slice(0, 11).some(value => value >= minConfidence);
  // Scalp above the eyes, or the whole head when no facial point is confident.
  y0 -= headSeen ? 0.24 * unit : 0.75 * unit;
  // A body cut by the bottom edge fills the frame down to it, like a detector box.
  y1 = toBottom ? height : y1 + 0.05 * unit;
  x0 -= 0.12 * unit; x1 += 0.12 * unit;
  const box = [Math.max(0, x0), Math.max(0, y0), Math.min(width, x1), Math.min(height, y1)];
  return box[2] - box[0] > 1 && box[3] - box[1] > 1 ? box : null;
}

/** Where to stop drawing an incomplete body: {known, y} with y in source
 * pixels, or y = null when nothing needs hiding. Unknown without visible joints. */
function clipBelow(landmarks, score, width, height, enter) {
  const visible = new Set(BODY_JOINTS.filter(i => score[i] >= enter &&
    landmarks[i].x >= 0 && landmarks[i].x <= 1 && landmarks[i].y >= 0 && landmarks[i].y <= 1));
  if (!visible.size) return { known: false, y: null };
  // A body cut by the bottom edge has nothing hidden below; a desk does.
  if (DOWNWARD.some(([parent, child]) => visible.has(parent) && !visible.has(child) && landmarks[child].y > BOTTOM)) return { known: true, y: null };
  const lowest = Math.max(...[...visible].map(i => landmarks[i].y * height));
  // Room for the flesh around the lowest joint (hand, buttocks), not much more:
  // right below it there is often the table that hides the rest.
  return { known: true, y: lowest + 0.15 * bodyUnit(landmarks, score, width, height, enter) };
}

// The Pose nose inside the detected face (10% margin), in normalized units.
function faceAgrees(landmarks, face) {
  if (!Array.isArray(face) || face.length < 468) return false;
  const nose = landmarks[0];
  if (!(landmarkScore(nose) > 0)) return false;
  const xs = face.map(p => p.x).filter(Number.isFinite), ys = face.map(p => p.y).filter(Number.isFinite);
  if (xs.length < 468 || ys.length < 468) return false;
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const mx = (x1 - x0) * 0.1, my = (y1 - y0) * 0.1;
  return nose.x >= x0 - mx && nose.x <= x1 + mx && nose.y >= y0 - my && nose.y <= y1 + my;
}

// Both Pose shoulders inside one detected hand (padded by 20%): the typical
// false body fitted into a hand held close to the camera.
function handOverShoulders(landmarks, hands) {
  const shoulders = SHOULDERS.map(i => landmarks[i]);
  return (hands ?? []).some(hand => {
    const points = (hand?.landmarks ?? []).filter(p => Number.isFinite(p?.x) && Number.isFinite(p?.y));
    if (points.length < 15) return false;
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const mx = (x1 - x0) * 0.2, my = (y1 - y0) * 0.2;
    return shoulders.every(p => p.x >= x0 - mx && p.x <= x1 + mx && p.y >= y0 - my && p.y <= y1 + my);
  });
}

/** Stateless per-frame evidence. `enough` (full or upper body) switches the
 * surface on; `tracked` (confident shoulders) keeps an active surface alive.
 * `options.face` / `options.hands` are the frame's face landmarks and hands;
 * `options.confirmed`: a face already matched this track's Pose (see above).
 * `face` in the result: this frame's face matches the Pose head. */
export function assessBody(landmarks, width, height, options = {}) {
  const { enter, keep, edge, minExtent, minShoulder, hands } = { ...DEFAULTS, ...options };
  const none = { level: 'none', full: false, enough: false, tracked: false, missing: [...FULL_BODY_JOINTS], extent: 0, bbox: null, clipY: null, clipKnown: false };
  if (!(Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0)) return { ...none, reason: 'invalid-source-size' };
  if (!Array.isArray(landmarks) || landmarks.length < 33) return { ...none, reason: 'no-pose' };
  const score = landmarks.slice(0, 33).map(landmarkScore);
  const seen = i => { const p = landmarks[i]; return score[i] >= enter && p.x >= edge && p.x <= 1 - edge && p.y >= edge && p.y <= 1 - edge; };
  const missing = FULL_BODY_JOINTS.filter(i => !seen(i));
  const points = FULL_BODY_JOINTS.map(i => ({ x: landmarks[i].x * width, y: landmarks[i].y * height }));
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const extent = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const distance = (a, b) => Math.hypot(points[a].x - points[b].x, points[a].y - points[b].y);
  const size = Math.min(width, height), minSegment = Math.max(2, size * 0.01);
  // Collapsed anchors (all joints on one spot) are not a body.
  const torsoShaped = distance(0, 2) >= minSegment && distance(1, 3) >= minSegment;
  const shoulderWidth = distance(0, 1);
  const full = missing.length === 0 && shoulderWidth >= minSegment && torsoShaped && extent >= minExtent * size;
  const shoulders = SHOULDERS.every(seen) && shoulderWidth >= minShoulder * size;
  const upperShape = !full && shoulders && ((HIPS.every(seen) && torsoShaped) || ELBOWS.every(seen));
  // Close-up: hips or elbows predicted at/below the bottom edge, not seen.
  const leavesFrame = [...HIPS, ...ELBOWS].some(i => !seen(i) && landmarks[i].y > BOTTOM);
  const face = faceAgrees(landmarks, options.face);
  const upper = upperShape && (face || !handOverShoulders(landmarks, hands));
  const closeShape = !full && !upperShape && shoulders && leavesFrame;
  const kept = closeShape && !face && Boolean(options.confirmed) && !handOverShoulders(landmarks, hands);
  const closeUp = closeShape && (face || kept);
  const level = full ? 'full' : upper || closeUp ? 'upper' : 'none';
  const reason = full ? 'full-body' : upper ? 'upper-body' : closeUp ? (kept ? 'close-up-kept' : 'close-up')
    : upperShape ? 'hand-over-shoulders-without-face'
    : !SHOULDERS.every(seen) ? 'shoulders-missing' : shoulderWidth < minSegment ? 'degenerate-body'
    : shoulderWidth < minShoulder * size ? 'body-too-small' : leavesFrame ? 'close-up-without-face' : 'needs-hips-or-elbows';
  const tracked = SHOULDERS.every(i => score[i] >= keep);
  // Any incomplete body is clipped, also while an active surface waits out a
  // moment with too little visible. `clipKnown` false: keep the previous line.
  const clip = full ? { known: true, y: null } : tracked ? clipBelow(landmarks, score, width, height, enter) : { known: false, y: null };
  return { level, full, enough: level !== 'none', tracked, face, missing, extent, reason,
    bbox: tracked ? personBox(landmarks, width, height, { toBottom: !full && clip.known && clip.y === null }) : null,
    clipY: clip.y, clipKnown: clip.known };
}
