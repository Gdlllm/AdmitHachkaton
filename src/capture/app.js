import '../style.css';
import { CaptureSession } from './session.js';
import { reconcileDetections } from '../tracking/fusion.js';
import { CaptureStabilizer } from '../tracking/stabilizer.js';
import { CaptureRenderer, containViewport } from '../rendering/renderer.js';

const video = document.getElementById('camera');
const notice = document.getElementById('notice');
const renderer = new CaptureRenderer(document.getElementById('overlay'));
const stabilizer = new CaptureStabilizer();
const listeners = new Set();
const params = new URLSearchParams(location.search);
const defaultModel = ['full', 'heavy', 'lite'].includes(params.get('model')) ? params.get('model') : 'full';
// Body surface: on wherever WebGPU can run it, switched on by a full body in
// view. ?surface=0 turns it off; ?surface=1 also allows the slow CPU fallback.
// Data Saver skips the ~115 MB download unless it is requested explicitly.
const defaultSurface = { 0: false, 1: 'force' }[params.get('surface')] ?? (navigator.connection?.saveData ? false : 'auto');
let lastFrame = null, lastDrawn = null, paused = false, retryOptions = {};
let renderMs = 0, stabilizeMs = 0;
let dense = null, denseLoading = null, denseError = null, surfaceMode = false, startGeneration = 0;

const surfaceModeOf = option => option == null ? defaultSurface : option === 'auto' ? 'auto' : option ? 'force' : false;

function releaseDense() {
  const old = dense; dense = null; denseLoading = null;
  old?.dispose().catch(error => console.error('Surface cleanup failed:', error));
}

/** The surface files (~115 MB, cached after the first visit) load in the
 * background after capture is running; landmarks never wait for it. */
function loadDense(mode) {
  // A failed download may succeed on the next start; unsupported stays off.
  if (dense?.getDiagnostics().state === 'error') releaseDense();
  if (dense || denseLoading) return;
  denseError = null;
  const loading = denseLoading = import('../dense/controller.js').then(({ createDenseSurface }) => {
    if (loading !== denseLoading) return;
    dense = createDenseSurface({ mode });
    return dense.ready;
  }).catch(error => {
    if (loading !== denseLoading) return;
    denseError = error.message;
    console.warn('Body surface is off:', error.message);
  }).finally(() => { if (loading === denseLoading) denseLoading = null; });
}

let held = null; // a frame waiting a few ms for its own body surface

function publishFrame(frame) {
  if (paused) return;
  // A newer frame supersedes a held one: show that now, while its image lives.
  flushHeld();
  const started = performance.now();
  const stabilized = stabilizer.update(frame);
  stabilizeMs = performance.now() - started;
  const wait = dense && surfaceMode ? dense.update(stabilized).wait : null;
  if (!wait) { finishPublish(stabilized); return; }
  const token = session.generation;
  held = stabilized;
  wait.then(() => { if (held === stabilized && token === session.generation && !paused) { held = null; finishPublish(stabilized); } });
}

function flushHeld() { const frame = held; held = null; if (frame) finishPublish(frame); }

function finishPublish(frame) {
  if (dense && surfaceMode) { frame.surface = dense.getDiagnostics(); frame.bodySurface = dense.geometry(); }
  lastFrame = frame;
  drawFrame(lastFrame); lastDrawn = lastFrame;
  const token = session.generation;
  for (const listener of listeners) {
    try { listener(lastFrame); } catch (error) { console.error('Capture subscriber failed:', error); }
    if (session.generation !== token) return;
  }
  window.dispatchEvent(new CustomEvent('motionframe', { detail: lastFrame }));
}

function drawFrame(frame) {
  const started = performance.now();
  renderer.setSize(innerWidth, innerHeight, Math.min(devicePixelRatio || 1, 2));
  const videoWidth = frame?.sourceWidth ?? video.videoWidth, videoHeight = frame?.sourceHeight ?? video.videoHeight;
  let surfaceImage = null;
  if (frame && dense && surfaceMode) {
    // One CSS pixel per wireframe line; the renderer applies contain/mirror.
    const viewport = containViewport(innerWidth, innerHeight, videoWidth, videoHeight);
    if (viewport) surfaceImage = dense.render(Math.round(viewport.width), Math.round(viewport.height), started);
  }
  renderer.draw(surfaceImage ? { ...frame, surfaceImage } : frame, { videoWidth, videoHeight, mirror: true, now: started });
  renderMs = performance.now() - started;
}

const session = new CaptureSession({
  video,
  onState({ state, error }) {
    if (state === 'idle' || state === 'error' || state === 'starting') {
      lastFrame = null; lastDrawn = null; held = null; stabilizer.reset(); renderer.clear(); dense?.reset();
    }
    notice.hidden = state !== 'error';
    document.getElementById('message').textContent = error || '';
  },
  onFrame(frame) {
    if (!paused) publishFrame(reconcileDetections(frame));
  },
});

async function start(options = {}) {
  stop();
  const token = startGeneration;
  retryOptions = { ...options, model: options.model ?? defaultModel };
  surfaceMode = surfaceModeOf(options.surface);
  try {
    const result = await session.start(retryOptions);
    if (token === startGeneration && result && surfaceMode) loadDense(surfaceMode);
    return token === startGeneration ? result : null;
  } catch (error) {
    if (token !== startGeneration) return null;
    document.getElementById('message').textContent = error.message;
    notice.hidden = false;
    throw error;
  }
}
function stop() { startGeneration++; session.stop(); dense?.reset(); lastFrame = null; lastDrawn = null; held = null; renderer.clear(); }
function getDiagnostics() {
  return { ...session.getDiagnostics(), renderMs, stabilizeMs,
    surface: dense?.getDiagnostics() ?? { enabled: Boolean(surfaceMode), state: denseLoading ? 'loading' : surfaceMode ? 'waiting' : 'off', error: denseError },
    smoothing: lastFrame?.tracking?.smoothing ?? null,
    lastFrame: session.lastFrame ? { ...session.lastFrame,
      visiblePoseCount: lastFrame?.pose?.landmarks?.filter(p => p.drawConfidence > 0).length ?? 0,
      stabilizedFaceCount: lastFrame?.face?.landmarks?.length ?? 0,
      stabilizedHandCounts: (lastFrame?.hands ?? []).map(h => h.landmarks.length),
    } : null };
}

// This is also the reusable module interface. It has no on-screen debug panels,
// no recording and no network export. A subscriber must not close frame.source.
window.motionCapture = Object.freeze({
  start, stop, getDiagnostics,
  subscribe(listener) { if (typeof listener !== 'function') throw new TypeError('A callback is required.'); listeners.add(listener); return () => listeners.delete(listener); },
});

function tick() {
  if (lastFrame && !paused) {
    const resized = renderer.width !== innerWidth || renderer.height !== innerHeight;
    const fading = performance.now() - lastFrame.receivedAt > 180;
    if (resized || fading || lastFrame !== lastDrawn) {
      drawFrame(lastFrame); lastDrawn = lastFrame;
      if (performance.now() - lastFrame.receivedAt > 500) { lastFrame = null; lastDrawn = null; }
    }
  }
  requestAnimationFrame(tick);
}

document.getElementById('retry').addEventListener('click', () => start(retryOptions).catch(() => {}));
window.addEventListener('pagehide', () => { stop(); releaseDense(); });
window.addEventListener('pageshow', event => { if (event.persisted && !params.has('test')) start(retryOptions).catch(() => {}); });
document.addEventListener('visibilitychange', () => {
  paused = document.hidden;
  if (paused) { session.pause(); lastFrame = null; lastDrawn = null; held = null; stabilizer.reset(); renderer.clear(); dense?.reset(); }
  else session.resume();
});
requestAnimationFrame(tick);
if (!params.has('test')) start(params.has('video') ? { videoUrl: params.get('video') } : {}).catch(() => {});
