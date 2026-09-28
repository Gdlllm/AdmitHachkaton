export const REGION_NAMES = ['pose', 'face', 'hands'];

export function modelOptions(region, config = {}) {
  const common = { runningMode: 'VIDEO' };
  if (region === 'pose') {
    const model = ['heavy', 'full', 'lite'].includes(config.model) ? config.model : 'full';
    return {
      ...common, baseOptions: { modelAssetPath: `/models/pose_${model}.task` },
      numPoses: 1, minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
      outputSegmentationMasks: Boolean(config.segmentation),
    };
  }
  if (region === 'face') return {
    ...common, baseOptions: { modelAssetPath: '/models/face_landmarker.task' },
    numFaces: 1, minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
  };
  if (region === 'hands') return {
    ...common, baseOptions: { modelAssetPath: '/models/hand_landmarker.task' },
    numHands: 2, minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  };
  throw new Error(`Unknown tracking region: ${region}`);
}

export function normalizeResult(region, raw) {
  if (region === 'pose') return {
    landmarks: raw.landmarks?.[0] ?? [], worldLandmarks: raw.worldLandmarks?.[0] ?? [],
  };
  if (region === 'face') return {
    landmarks: raw.faceLandmarks?.[0] ?? [],
    blendshapes: raw.faceBlendshapes?.[0]?.categories ?? [],
    transformationMatrix: raw.facialTransformationMatrixes?.[0] ?? null,
  };
  if (region === 'hands') return (raw.landmarks ?? []).map((landmarks, index) => {
    const category = raw.handedness?.[index]?.[0];
    return {
      landmarks, worldLandmarks: raw.worldLandmarks?.[index] ?? [],
      handedness: category?.categoryName ?? 'Unknown',
      // Left/Right classification confidence is not landmark accuracy.
      score: category?.score ?? null,
    };
  });
  throw new Error(`Unknown tracking region: ${region}`);
}

// This is a person silhouette, not a dense body mesh. Copy into owned memory;
// transferring a MediaPipe/WASM-owned mask buffer could invalidate the runtime.
export function compactMask(mask, maxSide = 320) {
  if (!mask) return null;
  const ratio = Math.min(1, maxSide / Math.max(mask.width, mask.height));
  const width = Math.max(1, Math.round(mask.width * ratio));
  const height = Math.max(1, Math.round(mask.height * ratio));
  const values = mask.getAsFloat32Array();
  const data = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const sy = Math.min(mask.height - 1, Math.floor((y + 0.5) * mask.height / height));
    for (let x = 0; x < width; x++) {
      const sx = Math.min(mask.width - 1, Math.floor((x + 0.5) * mask.width / width));
      data[y * width + x] = Math.round(Math.max(0, Math.min(1, values[sy * mask.width + sx])) * 255);
    }
  }
  return { width, height, data };
}
