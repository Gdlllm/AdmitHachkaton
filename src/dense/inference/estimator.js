import * as ort from 'onnxruntime-web/webgpu';
import { INPUT_SIZE, prepareRGBA, inverseCrop, projectCameraPoints } from './preprocess.js';
import { JOINT_NAMES, SKELETON_EDGES } from './skeleton.js';

export { JOINT_NAMES, SKELETON_EDGES };
const OUTPUT_LENGTHS = { mhr_params: 204, shape_params: 45, cam_trans: 3, joints_2d: 140, joints_3d: 210 };

/** Single-person research regressor. This does not contain a person detector:
 * supplied boxes are explicit, and blank/cropped inputs can still produce bodies.
 * `model` is a URL or the model bytes. The graph's free batch dimension is fixed
 * to 1: shape subgraphs then fold into constants and nothing falls back to CPU.
 */
export async function createBodyEstimator({
  model = '/models/instanthmr.onnx', executionProviders = ['webgpu', 'wasm'],
  wasmPaths, numThreads = 1, graphOptimizationLevel = 'all',
} = {}) {
  ort.env.wasm.numThreads = numThreads;
  if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths;
  const started = performance.now();
  const session = await ort.InferenceSession.create(model, {
    executionProviders, graphOptimizationLevel, preferredOutputLocation: 'cpu', freeDimensionOverrides: { batch: 1 },
    // Constant folding reports float16 nodes it leaves to the GPU; not errors.
    logSeverityLevel: 3,
  });
  if (!['image', 'cliff_cond'].every(name => session.inputNames.includes(name)) ||
    !Object.keys(OUTPUT_LENGTHS).every(name => session.outputNames.includes(name))) {
    await session.release();
    throw new Error('InstantHMR checkpoint I/O differs from the pinned five-output model');
  }
  const initMs = performance.now() - started;
  let disposed = false, active = null;
  let canvas = null, context = null;

  /** `input` is a normalized 1x3x224x224 crop, `crop` comes from cropFromBox(). */
  const run = async (input, cliff, crop, { frameId = null } = {}) => {
    if (disposed) throw new Error('Estimator is disposed');
    if (active) throw new Error('One InstantHMR frame is already in flight');
    const task = (async () => {
      const began = performance.now();
      const inputs = {
        image: new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]),
        cliff_cond: new ort.Tensor('float32', Float32Array.from(cliff), [1, 3]),
      };
      let outputs;
      try {
        outputs = await session.run(inputs);
        const inferred = performance.now();
        const raw = {};
        for (const [name, size] of Object.entries(OUTPUT_LENGTHS)) {
          const value = await outputs[name].getData();
          if (value.length !== size) throw new Error(`Unexpected ${name} length ${value.length}; expected ${size}`);
          raw[name] = Float32Array.from(value);
          if (!raw[name].every(Number.isFinite)) throw new Error(`Non-finite model output ${name}`);
        }
        const focal = crop.focalLength ?? Math.hypot(crop.width, crop.height);
        const principalPoint = crop.principalPoint ?? [crop.width / 2, crop.height / 2];
        const joints3d = Array.from({ length: 70 }, (_, i) => ({ x: raw.joints_3d[i * 3], y: raw.joints_3d[i * 3 + 1], z: raw.joints_3d[i * 3 + 2] }));
        const joints3dCamera = joints3d.map(point => ({ x: point.x + raw.cam_trans[0], y: point.y + raw.cam_trans[1], z: point.z + raw.cam_trans[2] }));
        const finished = performance.now();
        return {
          frameId, sourceWidth: crop.width, sourceHeight: crop.height,
          mhrParams: raw.mhr_params, shapeParams: raw.shape_params, camTrans: raw.cam_trans,
          joints2d: inverseCrop(raw.joints_2d, crop), joints3d, joints3dCamera,
          projectedJoints2d: projectCameraPoints(joints3dCamera, focal, principalPoint),
          jointNames: JOINT_NAMES, skeletonEdges: SKELETON_EDGES,
          crop, focal, focalLength: focal, focalAssumed: crop.focalAssumed ?? true, principalPoint, cliffCondition: Array.from(cliff),
          conventions: { cliff: 'pixel-normalized (pinned graph has no metadata)', joints2d: 'source pixels, direct 2D head', joints3d: 'rig-local metres, Y-down Z-forward', projectedJoints2d: '3D head + camera translation, perspective, source pixels', confidence: 'not provided by regressor', bboxSource: 'caller supplied; no person detector in estimator' },
          raw, timings: { inferenceMs: inferred - began, postprocessMs: finished - inferred, totalMs: finished - began },
        };
      } finally {
        for (const value of Object.values(inputs)) value.dispose();
        if (outputs) for (const value of Object.values(outputs)) value.dispose();
      }
    })();
    active = task;
    try { return await task; } finally { active = null; }
  };

  /** Convenience path for a whole image source (tests and offline tools). */
  const estimate = async (imageSource, { bbox, focal, frameId = null } = {}) => {
    let rgba;
    if (imageSource?.data && imageSource.width && imageSource.height) rgba = imageSource;
    else {
      const width = imageSource.videoWidth || imageSource.naturalWidth || imageSource.width;
      const height = imageSource.videoHeight || imageSource.naturalHeight || imageSource.height;
      if (!(width > 0 && height > 0)) throw new Error('Source has no decoded dimensions');
      canvas ??= typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(1, 1) : document.createElement('canvas');
      context ??= canvas.getContext('2d', { willReadFrequently: true });
      canvas.width = width; canvas.height = height;
      context.drawImage(imageSource, 0, 0, width, height);
      rgba = context.getImageData(0, 0, width, height);
    }
    const prepared = prepareRGBA(rgba, { bbox, focal });
    const crop = { ...prepared.crop, focalLength: prepared.focalLength, focalAssumed: prepared.focalAssumed, principalPoint: prepared.principalPoint };
    return { ...await run(prepared.input, prepared.cliff, crop, { frameId }), bbox: bbox.slice() };
  };

  return {
    run, estimate,
    info: { initMs, executionProviders, inputNames: session.inputNames, outputNames: session.outputNames },
    async dispose() {
      if (disposed) return;
      disposed = true;
      if (active) await active.catch(() => {});
      await session.release();
      if (canvas) canvas.width = canvas.height = 1;
    },
  };
}
