import { fitCameraTranslation } from './fit-camera.js';
import { poseTargets, refineBodyPose } from './refine-pose.js';

export function projectBodyPoints(points, camera) {
  return Array.from({ length: points.length / 3 }, (_, i) => {
    const z = points[i * 3 + 2] + camera.translation[2];
    return { x: camera.width / 2 + camera.focal * (points[i * 3] + camera.translation[0]) / z,
      y: camera.height / 2 + camera.focal * (points[i * 3 + 1] + camera.translation[1]) / z };
  });
}

// Align the learned 3D prior to independent, confident image observations.
// No ground truth or fixture metadata enters this function. Preserve the raw
// parameters in the caller for diagnostics; a fit is not proof of 3D accuracy.
// `warm` ({mhrParams, translation}) is the previous frame's fit: between network
// runs it starts both solves near the current pose, while the trust region and
// regularization stay centred on the network prior.
export function correctBody({ decoder, result, pose, bodyHeight, warm = null, maxAngleChange, iterations, targets3d = [] }) {
  const start = performance.now();
  const targets = poseTargets(pose, result.sourceWidth, result.sourceHeight);
  const startParams = warm?.mhrParams ?? result.mhrParams;
  const points = decoder.decodeSkeleton(startParams).keypoints70;
  const imageTargets = Array.from({ length: 70 }, () => ({ x: NaN, y: NaN }));
  const weights = new Float32Array(70);
  for (const target of targets) { imageTargets[target.keypoint] = target; weights[target.keypoint] = target.weight; }
  const cameraFit = fitCameraTranslation({ joints3d: points, joints2d: imageTargets,
    indices: targets.map(t => t.keypoint), weights,
    initialTranslation: warm?.translation ?? result.camTrans, focal: result.focal,
    principalPoint: [result.sourceWidth / 2, result.sourceHeight / 2],
    imageSize: [result.sourceWidth, result.sourceHeight],
    // The network's depth is least reliable for cut-off close-ups; the image
    // anchors (shoulders, head, visible limbs) decide the scale.
    depthRatio: [.4, 2.5], maxPixelShift: Math.max(result.sourceWidth, result.sourceHeight) * .2 });
  const camera = { width: result.sourceWidth, height: result.sourceHeight, focal: result.focal, translation: cameraFit.translation };
  const refinement = refineBodyPose({ decoder, mhrParams: result.mhrParams, camera, targets, bodyHeight, targets3d,
    ...(warm?.mhrParams ? { initial: warm.mhrParams } : {}),
    ...(maxAngleChange !== undefined ? { maxAngleChange } : {}), ...(iterations !== undefined ? { iterations } : {}) });
  const skeleton = decoder.decodeSkeleton(refinement.mhrParams);
  return { mhrParams: refinement.mhrParams, camera, cameraFit, refinement,
    keypoints70: skeleton.keypoints70, projectedJoints2d: projectBodyPoints(skeleton.keypoints70, camera),
    rawProjectedJoints2d: warm ? null : projectBodyPoints(points, { ...camera, translation: result.camTrans }),
    timings: { correctionMs: performance.now() - start } };
}
