import { assessBody } from './inference/body-gate.js';
import { SurfaceActivation } from './inference/temporal.js';
import { cropFromBox, INPUT_SIZE } from './inference/preprocess.js';
import { MeshOverlay } from './view/mesh-overlay.js';
import { faceForward } from './inference/body-sense.js';

// The network runs at most this often and keeps the GPU at most ~half busy;
// every camera frame in between is re-fitted by the fit worker.
const MIN_INFERENCE_INTERVAL_MS = { webgpu: 66, wasm: 1000 };
const INFERENCE_DUTY = 0.5;
// How long a frame may wait for its own surface before it is shown with the
// previous one. A fit takes a few ms, far less than the next camera frame.
const FRAME_WAIT_MS = 24;
const FADE_IN_MS = 180, FADE_OUT_MS = 260, CLIP_MS = 120;
// Mean reprojection error / body height beyond which a fit is not shown.
const MAX_FIT_ERROR = 0.12;

function edgesOf(faces) {
  const set = new Set(), edges = [];
  for (let i = 0; i < faces.length; i += 3) for (let j = 0; j < 3; j++) {
    const a = faces[i + j], b = faces[i + (j + 1) % 3], key = a < b ? `${a}:${b}` : `${b}:${a}`;
    if (!set.has(key)) { set.add(key); edges.push(a, b); }
  }
  return Uint32Array.from(edges);
}

// Only what the fit needs; points hidden by the stabilizer count as unseen.
const compactPose = landmarks => landmarks.slice(0, 33).map(p => ({ x: p.x, y: p.y,
  visibility: p.drawConfidence === 0 ? 0 : p.visibility, presence: p.presence }));
// Which way a found face looks (camera axes), for the body-sense model.
const faceCue = face => {
  const forward = face?.landmarks?.length ? faceForward(face.transformationMatrix) : null;
  return forward ? { forward } : null;
};

const defaultWorkers = () => ({
  model: new Worker(new URL('./model-worker.js', import.meta.url), { type: 'module', name: 'motion-surface-model' }),
  fit: new Worker(new URL('./fit-worker.js', import.meta.url), { type: 'module', name: 'motion-surface-fit' }),
});

/** Optional dense body surface (InstantHMR -> MHR). The network and the rig
 * run in two workers; update() and render() are cheap and never block capture.
 * The surface switches on by itself for a full or upper body and off when the
 * person leaves; for an upper body nothing is drawn below the visible joints.
 * The original Pose/Face/Hands observations remain the source of gesture data;
 * inferred vertices are not observed landmarks.
 */
export function createDenseSurface({ mode = 'auto', sense = true, lift = true, drawMesh = true, workerFactory = defaultWorkers, overlayFactory = () => new MeshOverlay(),
  createBitmap = (...args) => createImageBitmap(...args), now = () => performance.now(), setTimer = setTimeout, clearTimer = clearTimeout } = {}) {
  const activation = new SurfaceActivation();
  let workers = workerFactory();
  let state = 'loading', provider = null, failure = null, progress = null, overlay = null, closed = false;
  let modelReady = false, fitReady = false;
  let faces = null, depthFaces = null, edges = null, names = {}, generation = 0, inferenceId = 0;
  let poseBusy = false, pendingPose = null, inferBusy = false, inflight = null, lastInferAt = -Infinity;
  let mesh = null, meshSerial = 0, shown = null, shownSerial = -1, drawn = null;
  let visibility = 0, lastRender = null, badFits = 0, fitOk = true, clipY = null, clipTarget = null;
  let senseState = null, confirmed = false, regions = {};
  let lastBody = null, lastStep = { state: 'idle', active: false, changed: null };
  const waiting = new Map(); // frame timestamp -> resolve, for frames held for their own surface
  const stats = { inferences: 0, inferenceMs: null, lastInferenceMs: null, fits: 0, fitMs: null, errors: 0, lastError: null, lateFrames: 0 };
  let settle;
  const ready = new Promise((resolve, reject) => { settle = { resolve, reject }; });
  ready.catch(() => {});

  const average = (previous, value) => previous === null ? value : previous * 0.85 + value * 0.15;
  const note = error => { stats.errors++; stats.lastError = error; };

  function releaseWaiting() { for (const resolve of waiting.values()) resolve(); waiting.clear(); }

  function release() {
    workers?.model.terminate(); workers?.fit.terminate(); workers = null;
    try { overlay?.dispose(); } catch {}
    overlay = null; mesh = shown = null; poseBusy = inferBusy = false; pendingPose = null;
    releaseWaiting();
  }

  function fail(error, nextState = 'error') {
    if (closed || state === 'error' || state === 'unsupported') return;
    state = nextState; failure = error.message;
    release();
    settle.reject(error);
  }

  function restartTrack() {
    generation++; pendingPose = null; mesh = null; badFits = 0; fitOk = true;
    workers?.fit.postMessage({ type: 'reset', generation });
    releaseWaiting();
  }

  function becomeReady() {
    if (!modelReady || !fitReady || state !== 'loading') return;
    try { overlay = overlayFactory(); } catch (error) { fail(error); return; }
    state = 'ready'; progress = null;
    settle.resolve({ provider });
  }

  function sendPose(message) {
    if (poseBusy) { pendingPose = message; return; }
    poseBusy = true;
    workers.fit.postMessage({ type: 'pose', generation, ...message });
  }

  function onFit({ data }) {
    if (closed) return;
    if (data.type === 'ready') {
      faces = data.faces; depthFaces = data.allFaces ?? data.faces; edges = edgesOf(faces);
      names = { jointNames: data.jointNames, keypointNames: data.keypointNames };
      senseState = data.sense ?? null;
      regions = data.regionFaces ? Object.fromEntries(Object.entries(data.regionFaces).map(([name, list]) => [name, { faces: list, edges: edgesOf(list) }])) : {};
      fitReady = true; becomeReady(); return;
    }
    if (data.type === 'error') { fail(new Error(data.message || 'Surface fit worker failed')); return; }
    if (data.type !== 'mesh') return;
    poseBusy = false;
    if (data.error) note(data.error);
    if (data.mesh && data.generation === generation && lastStep.active) {
      const error = data.mesh.fit?.normalizedError;
      if (Number.isFinite(error) && error > MAX_FIT_ERROR) { if (++badFits >= 3) fitOk = false; }
      else { badFits = 0; fitOk = true; }
      mesh = { ...data.mesh, receivedAt: now() }; meshSerial++;
      stats.fits++; stats.fitMs = average(stats.fitMs, data.mesh.fitMs);
    }
    // Release frames up to this one; a newer frame's pose is sent next.
    for (const [timestamp, resolve] of waiting) if (timestamp <= data.timestamp) { resolve(); waiting.delete(timestamp); }
    if (pendingPose) { const next = pendingPose; pendingPose = null; sendPose(next); }
  }

  function onModel({ data }) {
    if (closed) return;
    if (data.type === 'progress') { progress = { loaded: data.loaded, total: data.total }; return; }
    if (data.type === 'ready') { provider = data.provider; modelReady = true; becomeReady(); return; }
    if (data.type === 'unsupported') { fail(new Error(`Body surface needs WebGPU with shader-f16 (${data.reason})`), 'unsupported'); return; }
    if (data.type === 'error') { fail(new Error(data.message || 'Surface model worker failed')); return; }
    if (data.type !== 'result') return;
    inferBusy = false;
    if (data.error) { note(data.error); return; }
    stats.inferences++; stats.lastInferenceMs = data.inferenceMs;
    stats.inferenceMs = average(stats.inferenceMs, data.inferenceMs);
    const request = inflight;
    inflight = null;
    if (!request || request.id !== data.id || request.generation !== generation) return;
    workers.fit.postMessage({ type: 'prior', generation, mhrParams: data.mhrParams, shapeParams: data.shapeParams,
      camTrans: data.camTrans, focal: data.focal, width: request.width, height: request.height, timestamp: request.timestamp },
    [data.mhrParams.buffer, data.shapeParams.buffer, data.camTrans.buffer]);
  }

  workers.model.onmessage = onModel;
  workers.fit.onmessage = onFit;
  const onWorkerError = event => { event.preventDefault?.(); fail(new Error(event.message || 'Surface worker failed')); };
  workers.model.onerror = onWorkerError; workers.fit.onerror = onWorkerError;
  workers.model.postMessage({ type: 'init', mode });
  workers.fit.postMessage({ type: 'init', sense, lift, mesh: drawMesh });

  function requestInference(frame, bbox, width, height) {
    let crop;
    try { crop = cropFromBox(bbox, width, height); } catch { return; }
    const [x0, y0, x1, y1] = crop.rasterBounds;
    const request = { id: ++inferenceId, generation, width, height, timestamp: frame.timestamp };
    inferBusy = true; lastInferAt = now();
    // The GPU crops and resizes; the worker gets 224x224 pixels, not a frame.
    createBitmap(frame.source, x0, y0, x1 - x0, y1 - y0, { resizeWidth: INPUT_SIZE, resizeHeight: INPUT_SIZE, resizeQuality: 'low' })
      .then(bitmap => {
        if (closed || !workers || request.generation !== generation) { bitmap.close(); inferBusy = false; return; }
        inflight = request;
        workers.model.postMessage({ type: 'infer', id: request.id, bitmap, crop }, [bitmap]);
      }, error => { inferBusy = false; note(error?.message ?? String(error)); });
  }

  function inferenceDue() {
    const minimum = MIN_INFERENCE_INTERVAL_MS[provider] ?? MIN_INFERENCE_INTERVAL_MS.webgpu;
    return now() - lastInferAt >= Math.max(minimum, (stats.inferenceMs ?? 0) / INFERENCE_DUTY);
  }

  /** Per published (stabilized) frame. `wait` resolves when this frame's own
   * surface arrived (or after a short timeout); `geometry()` then reads it. */
  function update(frame) {
    if (state !== 'ready' || closed || !frame || !Number.isFinite(frame.timestamp)) return { wait: null };
    const width = frame.sourceWidth, height = frame.sourceHeight;
    const landmarks = frame.pose?.landmarks;
    lastBody = assessBody(landmarks, width, height, { face: frame.face?.landmarks, hands: frame.hands, confirmed });
    // A face confirms the person once; the confirmation lasts while the shoulders stay tracked.
    confirmed = lastBody.tracked && (confirmed || lastBody.face);
    lastStep = activation.update(lastBody, frame.timestamp);
    // The last known clip line holds through frames that cannot place one
    // (e.g. shoulders briefly lost); a full body or a frame edge lifts it.
    if (lastBody.clipKnown) clipTarget = lastBody.clipY;
    if (lastStep.changed === 'on') { restartTrack(); shown = null; visibility = 0; clipY = null; }
    else if (lastStep.changed === 'off') restartTrack();
    if (!lastStep.active || !Array.isArray(landmarks) || landmarks.length < 33) return { wait: null };
    if (!inferBusy && lastBody.bbox && frame.source && inferenceDue()) requestInference(frame, lastBody.bbox, width, height);
    sendPose({ pose: compactPose(landmarks), width, height, timestamp: frame.timestamp, face: faceCue(frame.face) });
    if (!mesh && !shown) return { wait: null }; // nothing to keep in sync yet
    const timestamp = frame.timestamp;
    const wait = new Promise(resolve => {
      const timer = setTimer(() => { if (waiting.delete(timestamp)) { stats.lateFrames++; resolve(); } }, FRAME_WAIT_MS);
      waiting.set(timestamp, () => { clearTimer(timer); resolve(); });
    });
    return { wait };
  }

  function geometry() {
    if (!lastStep.active || !fitOk || !mesh || mesh.skeletonOnly) return null;
    return { vertices: mesh.vertices, faces, skeleton: { positions: mesh.skeleton, jointNames: names.jointNames },
      keypoints70: mesh.keypoints70, keypointNames: names.keypointNames, camera: mesh.camera,
      timestamp: mesh.timestamp, units: 'metres', axes: 'X-right/Y-down/Z-forward', inferred: true,
      representation: 'Estimated MHR anatomical surface; not a measured clothed contour' };
  }

  /** Wireframe for the on-screen video rectangle, or null. `highlight`: body
   * regions (body, leftArm, rightArm, leftLeg, rightLeg) drawn red. */
  function render(width, height, time = now(), highlight = []) {
    if (!overlay || closed) return null;
    const elapsed = lastRender === null ? 0 : Math.max(0, time - lastRender);
    lastRender = time;
    const target = lastStep.active && fitOk && mesh && !mesh.skeletonOnly;
    if (target && shownSerial !== meshSerial) { shown = mesh; shownSerial = meshSerial; }
    visibility = target ? Math.min(1, visibility + elapsed / FADE_IN_MS) : Math.max(0, visibility - elapsed / FADE_OUT_MS);
    if (!shown || visibility <= 0) {
      if (!target) shown = null;
      return null;
    }
    // The clip line glides; with no clip it moves out below the frame.
    const clipTo = clipTarget ?? shown.camera.height * 1.5;
    clipY = clipY === null ? clipTo : clipY + (clipTo - clipY) * (1 - Math.exp(-elapsed / CLIP_MS));
    const lit = [...new Set(highlight)].filter(name => regions[name]).sort();
    const key = `${shownSerial}:${width}x${height}:${visibility.toFixed(3)}:${clipY.toFixed(1)}:${lit.join(',')}`;
    if (drawn !== key) {
      overlay.draw({ vertices: shown.vertices, faces: depthFaces }, edges, shown.camera, { width, height, alpha: visibility, clipY,
        highlight: lit.length ? highlightOf(lit) : null });
      drawn = key;
    }
    return overlay.canvas;
  }

  // Faces and edges of the lit regions, joined once per combination.
  const highlightCache = new Map();
  function highlightOf(names) {
    const key = names.join(',');
    if (!highlightCache.has(key)) {
      const join = field => { const parts = names.map(n => regions[n][field]); const out = new Uint32Array(parts.reduce((a, b) => a + b.length, 0)); let at = 0; for (const p of parts) { out.set(p, at); at += p.length; } return out; };
      highlightCache.set(key, { key, faces: join('faces'), edges: join('edges') });
    }
    return highlightCache.get(key);
  }

  function getDiagnostics() {
    return { enabled: !closed && state !== 'error' && state !== 'unsupported', state, provider, error: failure, progress,
      active: lastStep.active, activation: lastStep.state, visible: visibility > 0,
      body: lastBody?.level ?? 'none', reason: lastBody?.reason ?? null, missingJoints: lastBody?.missing ?? null,
      clipY: clipTarget,
      fitOk, fit: mesh?.fit ?? null, meshAgeMs: mesh ? now() - mesh.receivedAt : null,
      sense: { state: senseState, ...(mesh?.sense ?? {}) },
      inference: { count: stats.inferences, averageMs: stats.inferenceMs, lastMs: stats.lastInferenceMs },
      fits: stats.fits, fitMs: stats.fitMs, lateFrames: stats.lateFrames, errors: stats.errors, lastError: stats.lastError,
      representation: 'Estimated MHR anatomical surface; not a measured clothed contour' };
  }

  return {
    ready, update, geometry, render, getDiagnostics,
    /** New capture session: forget the person, keep the loaded models. */
    reset() {
      activation.reset(); lastStep = { state: 'idle', active: false, changed: null }; lastBody = null; confirmed = false;
      if (state === 'ready') restartTrack();
      releaseWaiting();
      shown = null; visibility = 0; lastRender = null; drawn = null; clipY = null; clipTarget = null;
    },
    async dispose() {
      if (closed) return;
      closed = true;
      if (state === 'loading') settle.reject(new Error('Surface disposed'));
      state = 'disposed';
      release();
    },
  };
}
