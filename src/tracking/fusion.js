/** A conservative check of agreement between independently inferred regions.
 *
 * Pose Full/Heavy can confidently fit a miniature body inside a close-up hand.
 * Hand landmarks describe joint centres, not the outside edge of the skin. Use
 * a conservative support envelope padded by 20% of measured palm width, then
 * require all four torso joints AND >=80% of confident body joints inside.
 * Padding is a heuristic in source pixels, not a model accuracy guarantee.
 * A separate ambiguity check rejects a body when its entire confident head
 * AND both shoulders fit inside one RAW hand box, without an independently
 * supported matching face. This also hides a real body whose head/shoulders
 * are completely occluded by a large foreground hand; visible legs alone do
 * not resolve the ambiguity (the model can fit legs along a real forearm).
 *
 * This is an explicit heuristic, not an anatomical or accuracy guarantee.
 * A face is suppressed only if all its measured points fit in the same hand
 * support envelope, its centre is inside the palm polygon (not between fingertips), and
 * there is no credible torso outside that hand. This deliberately prefers no
 * mesh to a guessed mesh over a palm; unusual occlusions remain ambiguous.
 * Never joins different world-coordinate origins. Run before stabilization.
 */
const TORSO = [11, 12, 23, 24];
const HEAD_AND_SHOULDERS = Array.from({ length: 13 }, (_, index) => index);
// Match the body anatomy drawn by the renderer when detailed hands are present.
// Coarse pose-face/palm anchors are not evidence about the torso or limbs.
const BODY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
const validImagePoint = point => point && Number.isFinite(point.x) && Number.isFinite(point.y)
  && point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1;
const confidentPosePoint = point => validImagePoint(point) && (point.visibility ?? 1) >= 0.45 && (point.presence ?? 1) >= 0.45;
const inside = (point, bounds) => point.x >= bounds.minX && point.x <= bounds.maxX && point.y >= bounds.minY && point.y <= bounds.maxY;
const boundsOf = points => ({
  minX: Math.min(...points.map(point => point.x)), maxX: Math.max(...points.map(point => point.x)),
  minY: Math.min(...points.map(point => point.y)), maxY: Math.max(...points.map(point => point.y)),
});

function inPolygon(point, polygon) {
  let contained = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i], b = polygon[j];
    if ((a.y > point.y) !== (b.y > point.y) && point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x) contained = !contained;
  }
  return contained;
}

function handRegion(hand, width, height) {
  const source = hand.landmarks ?? [];
  const points = source.filter(validImagePoint);
  if (points.length < 15) return null;
  const bounds = boundsOf(points);
  const palm = [0, 1, 5, 9, 13, 17].map(index => source[index]);
  const validPalm = palm.every(validImagePoint);
  const palmWidth = validPalm ? Math.hypot((source[5].x - source[17].x) * width, (source[5].y - source[17].y) * height) : 0;
  const padding = 0.2 * palmWidth;
  const envelope = { minX: bounds.minX - padding / width, maxX: bounds.maxX + padding / width,
    minY: bounds.minY - padding / height, maxY: bounds.maxY + padding / height };
  return { bounds, envelope, palm: validPalm ? palm : null, paddingPx: padding };
}

export function reconcileDetections(frame) {
  const width = frame.sourceWidth ?? frame.width ?? 1280, height = frame.sourceHeight ?? frame.height ?? 720;
  if (!(width > 0 && height > 0 && Number.isFinite(width) && Number.isFinite(height))) return frame;
  const torso = TORSO.map(index => frame.pose?.landmarks?.[index]);
  const credibleTorso = torso.every(confidentPosePoint);
  const posePoints = BODY_JOINTS.map(index => frame.pose?.landmarks?.[index]).filter(confidentPosePoint);
  const upperBody = HEAD_AND_SHOULDERS.map(index => frame.pose?.landmarks?.[index]);
  const facePoints = (frame.face?.landmarks ?? []).filter(validImagePoint);
  const completeFace = facePoints.length >= 468 && facePoints.length === frame.face?.landmarks?.length;
  const faceBounds = completeFace ? boundsOf(facePoints) : null;
  const faceCenter = completeFace ? {
    x: facePoints.reduce((sum, point) => sum + point.x, 0) / facePoints.length,
    y: facePoints.reduce((sum, point) => sum + point.y, 0) / facePoints.length,
  } : null;
  let pose = frame.pose, face = frame.face;
  const fusion = { ...(frame.fusion ?? {}) };
  for (let handIndex = 0; handIndex < (frame.hands?.length ?? 0); handIndex++) {
    const region = handRegion(frame.hands[handIndex], width, height);
    if (!region) continue;
    const poseCoverage = posePoints.length ? posePoints.filter(point => inside(point, region.envelope)).length / posePoints.length : 0;
    const containedTorso = credibleTorso && torso.every(point => inside(point, region.envelope));
    const faceOverPalm = completeFace && region.palm && facePoints.every(point => inside(point, region.envelope))
      && inPolygon(faceCenter, region.palm);
    // Pose nose and both eyes must lie inside the independent dense face.
    // A distant second face or another guessed mesh over this palm is no veto.
    const matchingFace = face && completeFace && !faceOverPalm
      && [0, 2, 5].every(index => confidentPosePoint(upperBody[index]) && inside(upperBody[index], faceBounds));
    const ambiguousUpperBody = upperBody.every(point => confidentPosePoint(point) && inside(point, region.bounds))
      && !matchingFace;
    if (pose && containedTorso && poseCoverage >= 0.8) {
      pose = null;
      Object.assign(fusion, { rejectedPose: true, reason: 'torso-contained-in-hand', handIndex,
        handBounds: region.bounds, supportBounds: region.envelope, paddingPx: region.paddingPx, poseCoverage });
    } else if (pose && ambiguousUpperBody) {
      pose = null;
      Object.assign(fusion, { rejectedPose: true, reason: 'head-and-shoulders-contained-in-hand', handIndex,
        handBounds: region.bounds, poseAmbiguous: true, poseCoverage });
    }
    const externalBody = pose && credibleTorso && !containedTorso;
    if (face && faceOverPalm && !externalBody) {
      face = null;
      Object.assign(fusion, { rejectedFace: true, faceReason: 'face-contained-over-palm', faceHandIndex: handIndex });
    }
  }
  return pose !== frame.pose || face !== frame.face ? { ...frame, pose, face, fusion } : frame;
}
