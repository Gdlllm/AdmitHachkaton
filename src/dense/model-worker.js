/** InstantHMR network for the optional body surface (WebGPU; the WASM backend
 * only when forced). It only sees 224x224 person crops and returns the raw
 * prediction; smoothing and per-frame fitting live in fit-worker.js, so a busy
 * GPU never delays the surface following the body.
 *
 *   init {mode: 'auto'|'force'} → progress*, ready {provider} | unsupported | error
 *   infer {id, bitmap, crop}    → result {id, mhrParams, shapeParams, camTrans, focal, inferenceMs} | result {id, error}
 * The controller terminates the worker, which also releases its GPU device.
 */
import * as ort from 'onnxruntime-web/webgpu';
import { createBodyEstimator } from './inference/estimator.js';
import { cropFromBox, normalizeInput, INPUT_SIZE } from './inference/preprocess.js';
import { fetchAsset } from '../shared/assets.js';

const ORT_FILES = { mjs: 'ort-wasm-simd-threaded.asyncify.mjs', wasm: 'ort-wasm-simd-threaded.asyncify.wasm' };
let estimator = null;
const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
const context = canvas.getContext('2d', { willReadFrequently: true });

const post = (message, transfer = []) => self.postMessage(message, transfer);
const url = path => new URL(path, self.location.href).href;
const json = async path => { const response = await fetch(url(path)); if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`); return response.json(); };

async function init({ mode = 'auto' } = {}) {
  const adapter = await navigator.gpu?.requestAdapter?.().catch(() => null);
  // The pinned graph stores float16 weights; without shader-f16 it would run on CPU.
  const webgpu = Boolean(adapter?.features?.has('shader-f16'));
  if (!webgpu && mode !== 'force') {
    post({ type: 'unsupported', reason: !navigator.gpu ? 'no-webgpu' : adapter ? 'webgpu-without-shader-f16' : 'no-webgpu-adapter' });
    return;
  }
  const provider = webgpu ? 'webgpu' : 'wasm';
  const [model, runtime] = await Promise.all([json('/dense/model-manifest.json'), json('/dense/ort/manifest.json').catch(() => null)]);
  const wasmPin = runtime?.files?.[ORT_FILES.wasm] ?? {};
  const loaded = { runtime: 0, model: 0 }, totals = { runtime: wasmPin.bytes ?? 0, model: model.bytes };
  let reported = 0;
  const progress = key => (value, total) => {
    loaded[key] = value; totals[key] = total || totals[key];
    if (performance.now() - reported < 200) return;
    reported = performance.now();
    post({ type: 'progress', loaded: loaded.runtime + loaded.model, total: totals.runtime + totals.model });
  };
  const [wasmBinary, weights] = await Promise.all([
    fetchAsset(`/dense/ort/${ORT_FILES.wasm}`, { ...wasmPin, onProgress: progress('runtime') }),
    fetchAsset('/dense/instanthmr.onnx', { bytes: model.bytes, sha256: model.sha256, onProgress: progress('model') }),
  ]);
  ort.env.logLevel = 'error';
  ort.env.wasm.wasmBinary = new Uint8Array(wasmBinary);
  ort.env.wasm.wasmPaths = { mjs: url(`/dense/ort/${ORT_FILES.mjs}`) };
  const started = performance.now();
  estimator = await createBodyEstimator({ model: new Uint8Array(weights), executionProviders: [provider] });
  // Compile the GPU kernels now instead of on the first person in view.
  await estimator.run(new Float32Array(3 * INPUT_SIZE * INPUT_SIZE), [0, 0, 0.5], cropFromBox([0, 0, 64, 128], 256, 256));
  post({ type: 'ready', provider, initMs: performance.now() - started });
}

async function infer({ id, bitmap, crop }) {
  try {
    context.fillStyle = '#000'; context.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    // Outside the source the crop is transparent; over black that is the
    // upstream zero padding.
    try { context.drawImage(bitmap, 0, 0, INPUT_SIZE, INPUT_SIZE); } finally { bitmap.close(); }
    const input = normalizeInput(context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data);
    const output = await estimator.run(input, crop.cliff, crop);
    post({ type: 'result', id, mhrParams: output.mhrParams, shapeParams: output.shapeParams, camTrans: output.camTrans,
      focal: output.focal, inferenceMs: output.timings.inferenceMs },
    [output.mhrParams.buffer, output.shapeParams.buffer, output.camTrans.buffer]);
  } catch (error) {
    post({ type: 'result', id, error: error?.message ?? String(error) });
  }
}

self.onmessage = ({ data }) => {
  if (data.type === 'init') init(data).catch(error => post({ type: 'error', message: error?.message ?? String(error) }));
  else if (data.type === 'infer') {
    if (estimator) void infer(data);
    else { data.bitmap?.close(); post({ type: 'result', id: data.id, error: 'not-ready' }); }
  }
};
