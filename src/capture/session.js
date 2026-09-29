/** Owns camera, inference workers and transferred bitmaps for one capture session.
 * Only one canonical source frame is in flight. A generation invalidates every
 * asynchronous continuation on restart; stale ImageBitmaps are always closed.
 */
export class CaptureSession {
  constructor({ video, onFrame = () => {}, onState = () => {}, workerFactory = url => new Worker(url),
    mediaDevices = navigator.mediaDevices, createBitmap = source => createImageBitmap(source), now = () => performance.now() } = {}) {
    if (!video) throw new TypeError('A video element is required.');
    Object.assign(this, { video, onFrame, onState, workerFactory, mediaDevices, createBitmap, now });
    this.generation = 0; this.worker = null; this.stream = null; this.currentFrame = null;
    this.callback = null; this.rafCallback = false; this.pending = null;
    this.resetDiagnostics();
  }

  resetDiagnostics() {
    this.state = 'idle'; this.inFlight = false; this.frameId = 0; this.activeFrameId = null;
    this.frameCount = 0; this.lastVideoTime = -1; this.source = { kind: null, width: 0, height: 0 };
    this.lastError = null; this.lastFailure = null; this.workerCount = 0; this.samples = []; this.arrivals = [];
    this.startedAt = null; this.lastFrame = null; this.warnings = []; this.delegates = null;
    this.models = null; this.videoWarmup = false; this.videoEnded = false; this.modelsReady = false;
    this.latestPresentedSerial = 0; this.lastCapturedSerial = -1; this.latestMediaTime = 0;
    this.lastCapturedMediaTime = null; this.activeSourceFrameId = null; this.lastFallbackFrame = -1;
  }

  setState(state, error = null) {
    this.state = state;
    if (error) this.lastError = error;
    this.onState({ state, error, diagnostics: this.getDiagnostics() });
  }

  stop() {
    this.generation++;
    clearTimeout(this.startTimer); clearTimeout(this.frameTimer);
    this.startTimer = this.frameTimer = null;
    this.cancelScheduledFrame();
    const worker = this.worker; this.worker = null;
    if (worker) {
      // The coordinator owns three children; ask it to dispose, then terminate.
      try { worker.postMessage({ type: 'dispose' }); } catch {}
      worker.terminate();
    }
    this.stream?.getTracks().forEach(track => track.stop()); this.stream = null;
    this.video.pause(); this.video.srcObject = null;
    this.video.removeAttribute('src'); this.video.load();
    this.video.onended = null; this.video.onplay = null;
    this.currentFrame?.source?.close(); this.currentFrame = null;
    this.pendingPrevious?.source?.close(); this.pendingPrevious = null;
    this.pending?.resolve(null); this.pending = null;
    this.inFlight = false; this.workerCount = 0; this.activeFrameId = null; this.modelsReady = false;
    this.state = 'idle';
    this.onState({ state: 'idle', error: null, diagnostics: this.getDiagnostics() });
  }

  cancelScheduledFrame() {
    if (this.callback !== null) {
      if (this.rafCallback) cancelAnimationFrame(this.callback);
      else this.video.cancelVideoFrameCallback?.(this.callback);
    }
    this.callback = null;
  }

  start({ videoUrl = null, stream = null, model = 'full', delegate = 'GPU', loop = false, segmentation = false } = {}) {
    this.stop(); this.resetDiagnostics();
    const token = this.generation;
    this.setState('starting');
    this.startedAt = this.now();
    const completed = new Promise((resolve, reject) => { this.pending = { resolve, reject }; });
    // Attach immediately: early failures otherwise race the caller's await.
    completed.catch(() => {});
    this.startTimer = setTimeout(() => this.fail(new Error('Не удалось запустить модели. Повтори запуск камеры.'), token), 90000);
    // Return immediately so stop/timeout settles the public Promise even while
    // the browser permission prompt remains open. Late streams still get closed.
    void this.prepareSource({ videoUrl, stream, model, delegate, loop, segmentation }, token);
    return completed;
  }

  async prepareSource({ videoUrl, stream, model, delegate, loop, segmentation }, token) {
    try {
      if (videoUrl) {
        const url = new URL(videoUrl, location.href);
        if (url.origin !== location.origin && url.protocol !== 'blob:') throw new Error('Тестовое видео должно быть локальным файлом или находиться на этом сайте.');
        this.source.kind = 'video'; this.videoWarmup = true;
        this.video.autoplay = false; this.video.loop = loop; this.video.src = url.href;
        await this.waitForVideo(token);
        if (token !== this.generation) return;
        this.video.pause(); // warm the models on frame zero before playback
      } else {
        this.source.kind = 'camera';
        if (!stream && !this.mediaDevices?.getUserMedia) throw new Error('Для камеры нужен localhost или HTTPS. Открой ссылку в Chrome или Safari.');
        const incoming = stream ?? await this.mediaDevices.getUserMedia({
          video: { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 60, max: 60 }, facingMode: 'user' }, audio: false,
        });
        if (token !== this.generation) { incoming.getTracks().forEach(track => track.stop()); return null; }
        this.stream = incoming; this.video.srcObject = incoming;
        const track = incoming.getVideoTracks()[0];
        if (!track) throw new Error('В источнике нет видеопотока.');
        track.addEventListener('ended', () => this.fail(new Error('Камера отключилась. Подключи её и повтори запуск.'), token));
        await this.video.play();
      }
      if (token !== this.generation) return null;
      this.source.width = this.video.videoWidth; this.source.height = this.video.videoHeight;
      this.video.onended = () => { if (token === this.generation) { this.videoEnded = true; this.cancelScheduledFrame(); } };
      this.video.onplay = () => { if (token === this.generation) { this.videoEnded = false; if (this.state === 'running') this.schedule(token); } };
      const worker = this.workerFactory('/pose-worker.js');
      this.worker = worker; this.workerCount = 1;
      worker.onerror = event => this.fail(new Error(event.message || 'Не удалось запустить захват движения.'), token);
      worker.onmessageerror = () => this.fail(new Error('Не удалось получить результат распознавания.'), token);
      worker.onmessage = event => this.handleMessage(event.data, token);
      worker.postMessage({ type: 'init', model, delegate, segmentation, timeOrigin: performance.timeOrigin });
    } catch (error) { this.fail(error, token); }
  }

  waitForVideo(token) {
    return new Promise((resolve, reject) => {
      const clean = () => { clearInterval(poll); this.video.removeEventListener('loadeddata', ready); this.video.removeEventListener('error', bad); };
      const ready = () => { clean(); resolve(); };
      const bad = () => { clean(); reject(new Error('Не удалось открыть тестовое видео.')); };
      const poll = setInterval(() => { if (token !== this.generation) { clean(); resolve(); } }, 100);
      this.video.addEventListener('loadeddata', ready, { once: true });
      this.video.addEventListener('error', bad, { once: true });
      this.video.load();
      if (this.video.readyState >= 2) ready();
    });
  }

  fail(error, token) {
    if (token !== this.generation) return;
    const pending = this.pending; this.pending = null;
    this.stop();
    this.lastFailure = { name: error?.name ?? 'Error', message: error?.message ?? String(error) };
    const normalized = normalizeCaptureError(error);
    this.setState('error', normalized);
    pending?.reject(new Error(normalized));
  }

  handleMessage(data, token) {
    if (token !== this.generation) { data.source?.close(); data.bitmap?.close(); return; }
    if (data.type === 'ready') {
      this.workerCount = data.workerCount ?? 4; this.modelsReady = true;
      this.delegates = data.delegates ?? data.delegate; this.models = data.models ?? null;
      this.latestMediaTime = this.video.currentTime;
      this.capture(token, { force: true, mediaTime: this.video.currentTime });
      this.schedule(token);
      return;
    }
    if (data.type === 'warning') { this.warnings.push(data.message); this.warnings = this.warnings.slice(-10); return; }
    if (data.type === 'error') { this.fail(new Error(data.message || 'Ошибка модели захвата.'), token); return; }
    if (data.type === 'dropped') {
      // A correctly scheduled session should never queue a second frame.
      this.inFlight = false; this.activeFrameId = null; clearTimeout(this.frameTimer); this.schedule(token); return;
    }
    if (data.type !== 'result') return;
    if (data.frameId !== undefined && data.frameId !== this.activeFrameId) { data.source?.close(); return; }
    this.inFlight = false; this.activeFrameId = null; clearTimeout(this.frameTimer);
    const receivedAt = this.now();
    const source = data.source ?? data.bitmap;
    if (!source) { this.fail(new Error('Модель не вернула исходный кадр для синхронизации.'), token); return; }
    // GPU shader compilation can make the first webcam frame seconds old. It is
    // useful for warming the models, but must never flash over the current view.
    if (this.source.kind === 'camera' && receivedAt - data.timestamp > 1200) {
      source.close(); this.schedule(token); return;
    }
    const width = source.width, height = source.height;
    const frame = { ...data, source, receivedAt, sourceWidth: width, sourceHeight: height,
      latencyMs: receivedAt - data.timestamp, mediaTime: data.mediaTime ?? this.lastMediaTime,
      sourceFrameId: this.activeSourceFrameId };
    const previous = this.currentFrame;
    this.pendingPrevious = previous;
    this.currentFrame = frame;
    this.inFlight = true;
    this.frameCount++; this.source.width = width; this.source.height = height;
    this.samples.push({ latencyMs: frame.latencyMs, inferenceMs: data.inferenceMs });
    if (this.samples.length > 180) this.samples.shift();
    this.arrivals.push(receivedAt); this.arrivals = this.arrivals.filter(value => receivedAt - value < 3000);
    this.lastFrame = { timestamp: frame.timestamp, receivedAt, frameId: data.frameId, sourceFrameId: frame.sourceFrameId, mediaTime: frame.mediaTime,
      poseCount: frame.pose?.landmarks?.length ?? 0, faceCount: frame.face?.landmarks?.length ?? 0,
      handCounts: (frame.hands ?? []).map(hand => hand.landmarks.length),
      inferenceMs: data.inferenceMs, latencyMs: frame.latencyMs, timings: data.timings };
    const complete = () => {
      if (this.pendingPrevious === previous) { previous?.source?.close(); this.pendingPrevious = null; }
      // An asynchronous extension or subscriber may stop/replace this session.
      if (token !== this.generation) return;
      this.inFlight = false; clearTimeout(this.frameTimer);
      if (this.state !== 'running') {
        clearTimeout(this.startTimer); this.setState('running'); this.pending?.resolve(this.getDiagnostics()); this.pending = null;
      }
      if (this.videoWarmup) {
        this.videoWarmup = false;
        this.video.play().then(() => this.schedule(token)).catch(error => this.fail(error, token));
      } else this.schedule(token);
    };
    try {
      const processing = this.onFrame(frame);
      if (processing && typeof processing.then === 'function') {
        // Keep the canonical bitmap alive and the single-frame barrier closed
        // until every optional surface model has used the exact same image.
        if (token === this.generation) {
          this.inFlight = true;
          this.frameTimer = setTimeout(() => this.fail(new Error('Обработка поверхности не ответила вовремя.'), token), 15000);
        }
        Promise.resolve(processing).then(complete, error => { this.fail(error, token); });
      } else complete();
    } catch (error) { this.fail(error, token); }
  }

  schedule(token) {
    if (token !== this.generation || !this.worker || !this.modelsReady || document.hidden || this.videoEnded) return;
    // Keep the decoded-frame pump alive even while inference is busy. A video
    // clock advances between decoded frames; currentTime alone is NOT a frame ID.
    if (this.callback === null) {
      if (this.video.requestVideoFrameCallback) {
        this.rafCallback = false;
        this.callback = this.video.requestVideoFrameCallback((_, metadata) => {
          if (token !== this.generation) return;
          this.callback = null;
          this.latestPresentedSerial++;
          this.latestMediaTime = metadata.mediaTime;
          this.schedule(token);
        });
      } else {
        this.rafCallback = true;
        this.callback = requestAnimationFrame(() => {
          if (token !== this.generation) return;
          this.callback = null;
          const decoded = this.video.getVideoPlaybackQuality?.().totalVideoFrames;
          const frameKey = Number.isFinite(decoded) ? decoded : Math.floor(this.video.currentTime * 30);
          if (frameKey !== this.lastFallbackFrame) {
            this.lastFallbackFrame = frameKey;
            this.latestPresentedSerial++;
            this.latestMediaTime = this.video.currentTime;
          }
          this.schedule(token);
        });
      }
    }
    if (!this.inFlight && this.latestPresentedSerial > this.lastCapturedSerial) void this.capture(token);
  }

  async capture(token, { force = false, mediaTime = this.latestMediaTime } = {}) {
    if (token !== this.generation || !this.worker || !this.modelsReady || this.inFlight) return;
    if (this.video.readyState < 2 || (!force && (document.hidden ||
      this.latestPresentedSerial <= this.lastCapturedSerial || mediaTime === this.lastCapturedMediaTime))) return;
    this.inFlight = true; this.lastVideoTime = this.video.currentTime;
    this.lastCapturedSerial = this.latestPresentedSerial;
    this.activeSourceFrameId = this.latestPresentedSerial;
    this.lastCapturedMediaTime = mediaTime; this.lastMediaTime = mediaTime;
    const timestamp = this.now();
    let bitmap = null;
    try {
      bitmap = await this.createBitmap(this.video);
      if (token !== this.generation || !this.worker) { bitmap.close(); return; }
      const frameId = ++this.frameId; this.activeFrameId = frameId;
      this.worker.postMessage({ type: 'frame', frameId, timestamp, mediaTime, bitmap }, [bitmap]);
      bitmap = null; // ownership transferred to the coordinator
      this.frameTimer = setTimeout(() => this.fail(new Error('Распознавание перестало отвечать. Повтори запуск.'), token), this.frameCount ? 15000 : 60000);
    } catch (error) { bitmap?.close(); this.fail(error, token); }
  }

  pause() { this.cancelScheduledFrame(); }
  resume() { if (this.worker && !this.videoEnded) this.schedule(this.generation); }

  getDiagnostics() {
    const latency = this.samples.map(sample => sample.latencyMs).sort((a, b) => a - b);
    const percentile = p => latency.length ? latency[Math.min(latency.length - 1, Math.floor((latency.length - 1) * p))] : null;
    const fps = this.arrivals.length > 1 ? (this.arrivals.length - 1) * 1000 / (this.arrivals.at(-1) - this.arrivals[0]) : 0;
    return { state: this.state, source: { ...this.source }, frameCount: this.frameCount, inFlight: this.inFlight,
      workerCount: this.workerCount, lastFrame: this.lastFrame, error: this.lastError, failureDetails: this.lastFailure,
      fps, latencyP50: percentile(.5), latencyP95: percentile(.95), delegates: this.delegates, models: this.models,
      warnings: [...this.warnings], videoEnded: this.videoEnded, generation: this.generation };
  }
}

export function normalizeCaptureError(error) {
  return ({
    NotAllowedError: 'Разреши камеру в браузере. Если запрос не появился, открой эту ссылку в Chrome или Safari.',
    NotFoundError: 'Камера не найдена. Подключи её и попробуй снова.',
    NotReadableError: 'Камера занята. Закрой другое приложение, использующее её.',
    OverconstrainedError: 'Камера не поддерживает запрошенный режим. Попробуй другую камеру.',
  })[error?.name] ?? (/[^\u0000-\u007f]/.test(error?.message ?? '') ? error.message : 'Не удалось запустить захват. Повтори запуск или открой ссылку в Chrome.');
}
