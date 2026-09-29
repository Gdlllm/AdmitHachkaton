import { FilesetResolver, PoseLandmarker, FaceLandmarker, HandLandmarker } from '@mediapipe/tasks-vision';
import { REGION_NAMES, modelOptions, normalizeResult, compactMask } from './inference.js';
import { fetchAsset } from '../shared/assets.js';

// One captured frame is shared by independent, persistent region workers.
// No region can block detection of another region or run on the UI thread.
const region = new URL(self.location.href).searchParams.get('region');
if (REGION_NAMES.includes(region)) startRegion(region);
else startCoordinator();

function errorMessage(error) { return error?.message || String(error); }

function startRegion(name) {
  const Task = { pose: PoseLandmarker, face: FaceLandmarker, hands: HandLandmarker }[name];
  let detector, fileset, config, delegate, modelBuffer;
  let initialized = false, busy = false, frameWidth = 0, frameHeight = 0, maskWarningSent = false;

  async function create(requestedDelegate) {
    const options = modelOptions(name, config);
    // Loaded once per worker; a production build may have split large models.
    modelBuffer ??= new Uint8Array(await fetchAsset(options.baseOptions.modelAssetPath));
    if (name === 'pose' && requestedDelegate === 'CPU' && options.outputSegmentationMasks) {
      // Pinned MediaPipe 1.0.1 CPU masks abort on a tested portrait crop. Keep
      // landmark capture operational instead of enabling that optional output.
      options.outputSegmentationMasks = false;
      if (!maskWarningSent) {
        maskWarningSent = true;
        self.postMessage({ type: 'warning', region: name, code: 'cpu-mask-disabled', message: 'The optional silhouette is disabled on CPU; body landmarks remain enabled.' });
      }
    }
    const instance = await Task.createFromOptions(fileset, {
      ...options, baseOptions: { modelAssetBuffer: modelBuffer, delegate: requestedDelegate },
      canvas: new OffscreenCanvas(1280, 720),
    });
    delegate = requestedDelegate;
    return instance;
  }

  async function fallBackToCPU(cause) {
    try { detector?.close(); } catch { /* Closing a lost GPU context may fail. */ }
    detector = undefined;
    self.postMessage({ type: 'warning', region: name, code: 'cpu-fallback', message: errorMessage(cause) });
    detector = await create('CPU');
  }

  self.onmessage = async ({ data }) => {
    if (data.type === 'init') {
      if (initialized || busy) return;
      busy = true;
      config = data;
      const started = performance.now();
      try {
        fileset = await FilesetResolver.forVisionTasks('/wasm');
        const requested = config.delegate === 'CPU' ? 'CPU' : 'GPU';
        try { detector = await create(requested); }
        catch (error) {
          if (requested === 'CPU') throw error;
          await fallBackToCPU(error);
        }
        initialized = true;
        self.postMessage({ type: 'ready', region: name, delegate, initializationMs: performance.now() - started });
      } catch (error) {
        self.postMessage({ type: 'error', region: name, phase: 'initialization', message: errorMessage(error) });
      } finally { busy = false; }
      return;
    }

    if (data.type !== 'frame') return;
    const bitmap = data.bitmap;
    if (!initialized || busy) {
      bitmap?.close();
      self.postMessage({ type: 'error', region: name, message: 'Region received a frame before it was ready.' });
      return;
    }
    busy = true;
    let transferred = false, raw;
    const started = performance.now();
    try {
      if (frameWidth && (bitmap.width !== frameWidth || bitmap.height !== frameHeight)) {
        // Reusing VIDEO ROI state across a camera orientation/resolution change
        // can retain the wrong crop; CPU segmentation also rejects some changes.
        try { detector.close(); } catch { /* Recreate even if an old graph fails close. */ }
        detector = undefined;
        const requested = delegate;
        try { detector = await create(requested); }
        catch (error) {
          if (requested === 'CPU') throw error;
          await fallBackToCPU(error);
        }
      }
      frameWidth = bitmap.width;
      frameHeight = bitmap.height;
      try { raw = detector.detectForVideo(bitmap, data.timestamp); }
      catch (error) {
        if (delegate !== 'GPU') throw error;
        await fallBackToCPU(error);
        raw = detector.detectForVideo(bitmap, data.timestamp);
      }
      const result = normalizeResult(name, raw);
      const mask = name === 'pose' && config.segmentation ? compactMask(raw.segmentationMasks?.[0]) : null;
      const response = {
        type: 'result', region: name, frameId: data.frameId, timestamp: data.timestamp,
        inferenceMs: performance.now() - started, delegate, result, mask,
      };
      const transfers = mask ? [mask.data.buffer] : [];
      // Pose returns the original camera frame; face/hands own disposable clones.
      if (data.returnSource) { response.source = bitmap; transfers.push(bitmap); }
      self.postMessage(response, transfers);
      transferred = Boolean(data.returnSource);
    } catch (error) {
      self.postMessage({ type: 'error', region: name, frameId: data.frameId, phase: 'inference', message: errorMessage(error) });
    } finally {
      raw?.segmentationMasks?.forEach(mask => mask.close());
      if (!transferred) bitmap.close();
      busy = false;
    }
  };
}

class RegionClient {
  constructor(name, forward) {
    this.name = name;
    const url = new URL(self.location.href);
    url.searchParams.set('region', name);
    this.worker = new Worker(url.href, { name: `motion-${name}` });
    this.pending = null;
    this.worker.onmessage = ({ data }) => {
      if (data.type === 'warning' || data.type === 'progress') { forward(data); return; }
      const request = this.pending;
      if (!request) { data.source?.close(); return; }
      this.pending = null;
      clearTimeout(request.timer);
      if (data.type === 'error') request.reject(new Error(`${name}: ${data.message}`));
      else request.resolve(data);
    };
    this.worker.onerror = error => this.reject(new Error(`${name}: ${error.message || 'worker stopped'}`));
    this.worker.onmessageerror = () => this.reject(new Error(`${name}: invalid worker message`));
  }

  request(message, transfers = []) {
    if (this.pending) {
      message.bitmap?.close();
      return Promise.reject(new Error(`${this.name}: overlapping model requests`));
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.reject(new Error(`${this.name}: model timed out`)), 45000);
      this.pending = { resolve, reject, timer };
      try { this.worker.postMessage(message, transfers); }
      catch (error) { message.bitmap?.close(); this.reject(error); }
    });
  }

  reject(error) {
    const pending = this.pending;
    if (!pending) return;
    this.pending = null;
    clearTimeout(pending.timer);
    pending.reject(error);
  }

  close() {
    this.reject(new Error(`${this.name}: model stopped`));
    this.worker.terminate();
  }
}

function startCoordinator() {
  let clients = [], ready = false, busy = false, initializing = false;
  let lastTimestamp = -Infinity, frameId = 0, config = {};
  const forward = message => self.postMessage(message);
  const close = () => { ready = false; clients.forEach(client => client.close()); clients = []; };

  async function initialize(data) {
    if (initializing || ready) return;
    initializing = true;
    config = { model: 'full', segmentation: false, ...data };
    if (!['full', 'heavy', 'lite'].includes(config.model)) config.model = 'full';
    const started = performance.now();
    try {
      clients = REGION_NAMES.map(name => new RegionClient(name, forward));
      const responses = await Promise.all(clients.map(client => client.request(config)));
      ready = true;
      const delegates = Object.fromEntries(responses.map(result => [result.region, result.delegate]));
      self.postMessage({
        type: 'ready', model: config.model, workerCount: 4,
        models: { pose: `pose_${config.model}.task`, face: 'face_landmarker.task', hands: 'hand_landmarker.task' },
        delegate: new Set(Object.values(delegates)).size === 1 ? responses[0].delegate : 'mixed',
        delegates, initializationMs: performance.now() - started,
        capabilities: { pose: 33, hands: 42, face: 478, faceBlendshapes: 52, segmentation: Boolean(config.segmentation) && delegates.pose === 'GPU' },
      });
    } catch (error) {
      close();
      self.postMessage({ type: 'error', phase: 'initialization', message: errorMessage(error) });
    } finally { initializing = false; }
  }

  async function capture(data) {
    let source = data.bitmap;
    if (!ready || busy || !Number.isFinite(data.timestamp) || data.timestamp <= lastTimestamp) {
      source?.close();
      self.postMessage({ type: 'dropped', timestamp: data.timestamp, reason: !ready ? 'not-ready' : busy ? 'busy' : 'timestamp' });
      return;
    }
    busy = true;
    lastTimestamp = data.timestamp;
    const id = data.frameId ?? ++frameId;
    const started = performance.now();
    let faceCopy, handsCopy, frameClosed = false;
    try {
      const sourceWidth = source.width, sourceHeight = source.height;
      if (!sourceWidth || !sourceHeight) throw new Error('The camera frame is empty.');
      faceCopy = await createImageBitmap(source);
      handsCopy = await createImageBitmap(source);
      const request = { type: 'frame', timestamp: data.timestamp, frameId: id };
      const posePromise = clients[0].request({ ...request, bitmap: source, returnSource: true }, [source])
        .then(result => {
          if (frameClosed) { result.source?.close(); throw new Error('Frame was cancelled before pose completed.'); }
          source = result.source;
          return result;
        });
      source = null;
      const facePromise = clients[1].request({ ...request, bitmap: faceCopy }, [faceCopy]);
      faceCopy = null;
      const handsPromise = clients[2].request({ ...request, bitmap: handsCopy }, [handsCopy]);
      handsCopy = null;
      const [poseResult, faceResult, handsResult] = await Promise.all([posePromise, facePromise, handsPromise]);
      const results = [poseResult, faceResult, handsResult];
      if (results.some(result => result.frameId !== id || result.timestamp !== data.timestamp)) {
        throw new Error('Model results belong to different camera frames.');
      }
      const inferenceMs = performance.now() - started;
      const epoch = Number.isFinite(data.timeOrigin) ? data.timeOrigin : config.timeOrigin;
      const ageMs = Number.isFinite(epoch)
        ? Math.max(0, performance.timeOrigin + performance.now() - epoch - data.timestamp) : inferenceMs;
      const stamp = { timestamp: data.timestamp, ageMs };
      const pose = { ...poseResult.result, ...stamp };
      const face = { ...faceResult.result, ...stamp };
      const hands = handsResult.result.map(hand => ({ ...hand, ...stamp }));
      const response = {
        type: 'result', frameId: id, mediaTime: data.mediaTime, timestamp: data.timestamp, inferenceMs, ageMs,
        source, sourceWidth, sourceHeight, pose, face, hands, segmentation: poseResult.mask,
        timings: { poseMs: poseResult.inferenceMs, faceMs: faceResult.inferenceMs, handsMs: handsResult.inferenceMs },
        delegates: Object.fromEntries(results.map(result => [result.region, result.delegate])),
        landmarks: pose.landmarks, worldLandmarks: pose.worldLandmarks,
      };
      const transfers = [source];
      if (poseResult.mask) transfers.push(poseResult.mask.data.buffer);
      self.postMessage(response, transfers);
      source = null;
    } catch (error) {
      close();
      self.postMessage({ type: 'error', phase: 'inference', timestamp: data.timestamp, message: errorMessage(error) });
    } finally {
      frameClosed = true;
      source?.close(); faceCopy?.close(); handsCopy?.close(); busy = false;
    }
  }

  self.onmessage = ({ data }) => {
    if (data.type === 'init') void initialize(data);
    else if (data.type === 'frame') void capture(data);
    else if (data.type === 'dispose') { close(); self.close(); }
  };
}
