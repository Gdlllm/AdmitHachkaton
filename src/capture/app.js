import '../style.css';
import { CaptureSession } from './session.js';
import { reconcileDetections } from '../tracking/fusion.js';
import { CaptureStabilizer } from '../tracking/stabilizer.js';
import { CaptureRenderer } from '../rendering/renderer.js';

const video = document.getElementById('camera');
const notice = document.getElementById('notice');
const renderer = new CaptureRenderer(document.getElementById('overlay'));
const stabilizer = new CaptureStabilizer();
const listeners = new Set();
const params = new URLSearchParams(location.search);
const defaultModel = ['full', 'heavy', 'lite'].includes(params.get('model')) ? params.get('model') : 'full';
let lastFrame = null, lastDrawn = null, paused = false, retryOptions = {};
let renderMs = 0, stabilizeMs = 0;

function drawFrame(frame) {
  const started = performance.now();
  renderer.setSize(innerWidth, innerHeight, Math.min(devicePixelRatio || 1, 2));
  renderer.draw(frame, { videoWidth: frame?.sourceWidth ?? video.videoWidth,
    videoHeight: frame?.sourceHeight ?? video.videoHeight, mirror: true, now: started });
  renderMs = performance.now() - started;
}

const session = new CaptureSession({
  video,
  onState({ state, error }) {
    if (state === 'idle' || state === 'error' || state === 'starting') {
      lastFrame = null; lastDrawn = null; stabilizer.reset(); renderer.clear();
    }
    notice.hidden = state !== 'error';
    document.getElementById('message').textContent = error || '';
  },
  onFrame(frame) {
    if (paused) return;
    const started = performance.now();
    const token = session.generation;
    lastFrame = stabilizer.update(reconcileDetections(frame));
    stabilizeMs = performance.now() - started;
    drawFrame(lastFrame); lastDrawn = lastFrame;
    for (const listener of listeners) {
      try { listener(lastFrame); } catch (error) { console.error('Capture subscriber failed:', error); }
      if (session.generation !== token) return;
    }
    window.dispatchEvent(new CustomEvent('motionframe', { detail: lastFrame }));
  },
});

async function start(options = {}) {
  retryOptions = { ...options, model: options.model ?? defaultModel };
  return session.start(retryOptions);
}
function stop() { session.stop(); lastFrame = null; renderer.clear(); }
function getDiagnostics() {
  return { ...session.getDiagnostics(), renderMs, stabilizeMs, smoothing: lastFrame?.tracking?.smoothing ?? null,
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
window.addEventListener('pagehide', stop);
window.addEventListener('pageshow', event => { if (event.persisted && !params.has('test')) start(retryOptions).catch(() => {}); });
document.addEventListener('visibilitychange', () => {
  paused = document.hidden;
  if (paused) { session.pause(); lastFrame = null; lastDrawn = null; stabilizer.reset(); renderer.clear(); }
  else session.resume();
});
requestAnimationFrame(tick);
if (!params.has('test')) start(params.has('video') ? { videoUrl: params.get('video') } : {}).catch(() => {});
