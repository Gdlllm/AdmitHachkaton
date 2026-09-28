/** A conservative check of agreement between independently inferred regions.
 *
 * Pose Full/Heavy can confidently fit a miniature body inside a close-up hand.
 * Hand landmarks describe joint centres, not the outside edge of the skin. Use
 * a conservative support envelope padded by 20% of measured palm width, then
 * require all four torso joints AND >=80% of confident pose landmarks inside.
 * Padding is a heuristic in source pixels, not a model accuracy guarantee.
 *
 * This is an explicit heuristic, not an anatomical or accuracy guarantee.
 * A face is suppressed only if all its measured points fit in one hand's raw
 * bounds, its centre is inside the palm polygon (not between fingertips), and
 * there is no credible torso outside that hand. This deliberately prefers no
 * mesh to a guessed mesh over a palm; unusual occlusions remain ambiguous.
 * Never joins different world-coordinate origins. Run before stabilization.
 */
const TORSO = [11, 12, 23, 24];
const validImagePoint = point => point && Number.isFinite(point.x) && Number.isFinite(point.y)
  && point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1;
const confidentPosePoint = point => validImagePoint(point) && (point.visibility ?? 1) >= 0.45 && (point.presence ?? 1) >= 0.45;
const inside = (point, bounds) => point.x >= bounds.minX && point.x <= bounds.maxX && point.y >= bounds.minY && point.y <= bounds.maxY;

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
  const bounds = {
    minX: Math.min(...points.map(point => point.x)), maxX: Math.max(...points.map(point => point.x)),
    minY: Math.min(...points.map(point => point.y)), maxY: Math.max(...points.map(point => point.y)),
  };
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
  const posePoints = (frame.pose?.landmarks ?? []).filter(confidentPosePoint);
  const facePoints = (frame.face?.landmarks ?? []).filter(validImagePoint);
  let pose = frame.pose, face = frame.face;
  const fusion = { ...(frame.fusion ?? {}) };
  for (let handIndex = 0; handIndex < (frame.hands?.length ?? 0); handIndex++) {
    const region = handRegion(frame.hands[handIndex], width, height);
    if (!region) continue;
    const poseCoverage = posePoints.length ? posePoints.filter(point => inside(point, region.envelope)).length / posePoints.length : 0;
    const containedTorso = credibleTorso && torso.every(point => inside(point, region.envelope));
    if (pose && containedTorso && poseCoverage >= 0.8) {
      pose = null;
      Object.assign(fusion, { rejectedPose: true, reason: 'torso-contained-in-hand', handIndex,
        handBounds: region.bounds, supportBounds: region.envelope, paddingPx: region.paddingPx, poseCoverage });
    }
    const externalBody = pose && credibleTorso && !containedTorso;
    if (face && facePoints.length >= 468 && region.palm && !externalBody && facePoints.every(point => inside(point, region.bounds))) {
      const center = { x: facePoints.reduce((sum, point) => sum + point.x, 0) / facePoints.length,
        y: facePoints.reduce((sum, point) => sum + point.y, 0) / facePoints.length };
      if (inPolygon(center, region.palm)) {
        face = null;
        Object.assign(fusion, { rejectedFace: true, faceReason: 'face-contained-over-palm', faceHandIndex: handIndex });
      }
    }
  }
  return pose !== frame.pose || face !== frame.face ? { ...frame, pose, face, fusion } : frame;
}
