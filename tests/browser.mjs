/** Real-model browser QA. Run against `npm run dev`; no mocked detector.
 *
 * PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node tests/browser.mjs
 * CAPTURE_FIXTURES_DIR points at locally downloaded test media. Optional
 * CAPTURE_CLOSEUP_IMAGE is a local image used only in this browser; it is never
 * posted to a service. Screenshots and JSON go in tests/artifacts (gitignored).
 * This is a correctness/lifecycle test, not a controlled inference benchmark.
 */
import assert from 'node:assert/strict';
import { readFile, mkdir, writeFile, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const project = fileURLToPath(new URL('../', import.meta.url));
const base = process.env.CAPTURE_BASE_URL ?? 'http://127.0.0.1:5173';
const fixtures = process.env.CAPTURE_FIXTURES_DIR ?? path.join(project, 'tests/fixtures');
const artifacts = process.env.CAPTURE_QA_OUT ?? path.join(project, 'tests/artifacts');
const playwrightPath = process.env.PLAYWRIGHT_MODULE;
const systemChrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
let chromePath = process.env.CAPTURE_CHROME_PATH;
if (!chromePath && process.platform === 'darwin') {
  try { await access(systemChrome); chromePath = systemChrome; } catch {}
}
const closeupPath = process.env.CAPTURE_CLOSEUP_IMAGE;
const playwright = playwrightPath ? await import(pathToFileURL(playwrightPath).href) : await import('playwright');
const { chromium } = playwright;
await mkdir(artifacts, { recursive: true });
const report = { startedAt: new Date().toISOString(), base, browser: chromePath ?? 'Playwright Chromium', tests: [], pageErrors: [], fixtureReports: [], scope: 'Real local MediaPipe models; one desktop browser; no measured ground-truth accuracy.' };
let browser, page, browserCDP;

async function captureWorkerTargets() {
  const { targetInfos } = await browserCDP.send('Target.getTargets');
  return targetInfos.filter(target => target.type === 'worker' && target.url.includes('/pose-worker.js')).map(target => ({ id: target.targetId, url: target.url }));
}

async function expectWorkerTargets(expected) {
  const deadline = Date.now() + 5000;
  let targets;
  do {
    targets = await captureWorkerTargets();
    if (targets.length === expected) return targets;
    await new Promise(resolve => setTimeout(resolve, 100));
  } while (Date.now() < deadline);
  assert.equal(targets.length, expected, `actual browser worker targets: ${JSON.stringify(targets)}`);
}

async function test(name, fn) {
  const started = Date.now();
  try {
    const details = await fn();
    report.tests.push({ name, passed: true, durationMs: Date.now() - started, details });
    console.log(`PASS ${name}`);
  } catch (error) {
    report.tests.push({ name, passed: false, durationMs: Date.now() - started, error: error.stack ?? String(error) });
    console.error(`FAIL ${name}: ${error.message}`);
    if (page) await page.screenshot({ path: path.join(artifacts, `failure-${report.tests.length}.png`) }).catch(() => {});
  }
}

async function stop() {
  await page.evaluate(() => {
    window.motionCapture.stop();
    clearInterval(window.__qaCanvasTimer);
    window.__qaCanvasTimer = null;
  });
}

async function waitFrames(count = 8, timeout = 30000) {
  await page.waitForFunction(count => window.__qaFrames.length >= count, count, { timeout });
  return page.evaluate(() => ({ frames: window.__qaFrames, diagnostics: window.motionCapture.getDiagnostics(), workers: window.__qaWorkers }));
}

async function installFrameRecorder() {
  await page.evaluate(() => {
    window.__qaUnsubscribe?.();
    window.__qaFrames = [];
    window.__qaEventCount = 0;
    if (!window.__qaEventInstalled) {
      addEventListener('motionframe', () => { window.__qaEventCount++; });
      window.__qaEventInstalled = true;
    }
    window.__qaUnsubscribe = window.motionCapture.subscribe(frame => {
      const parts = [frame.pose, frame.face, ...(frame.hands ?? [])].filter(Boolean);
      window.__qaFrames.push({
        timestamp: frame.timestamp,
        mediaTime: frame.mediaTime,
        sourceFrameId: frame.sourceFrameId,
        sourceWidth: frame.sourceWidth ?? frame.width,
        sourceHeight: frame.sourceHeight ?? frame.height,
        poseCount: frame.pose?.landmarks?.length ?? 0,
        faceCount: frame.face?.landmarks?.length ?? 0,
        handCounts: frame.hands?.map(hand => hand.landmarks.length) ?? [],
        handIds: frame.hands?.map(hand => hand.track?.id) ?? [],
        inferenceMs: frame.inferenceMs,
        badVisiblePoints: parts.flatMap(part => part.landmarks).filter(p => p.drawConfidence > 0 && (!Number.isFinite(p.x) || !Number.isFinite(p.y))).length,
        smoothing: frame.tracking?.smoothing,
      });
    });
  });
}

async function startVideo(filename, frameCount = 10) {
  await stop();
  await installFrameRecorder();
  await page.evaluate(videoUrl => window.motionCapture.start({ videoUrl }), `/__fixtures/${filename}`);
  await waitFrames(frameCount);
  const duration = await page.evaluate(() => document.getElementById('camera').duration);
  // Cover distinct poses/lighting later in each real clip, not just its opening.
  // This is sampled coverage: inference continues on actual decoded frames.
  for (const fraction of [0.25, 0.5, 0.75, 0.9]) {
    const target = duration * fraction;
    const before = await page.evaluate(target => {
      document.getElementById('camera').currentTime = target;
      return window.__qaFrames.length;
    }, target);
    await page.waitForFunction(({ before, target }) => window.__qaFrames.slice(before).filter(frame => frame.mediaTime >= target - 0.1).length >= 2, { before, target }, { timeout: 30000 });
  }
  const result = await page.evaluate(() => ({ frames: window.__qaFrames, diagnostics: window.motionCapture.getDiagnostics(), workers: window.__qaWorkers }));
  assert.ok(result.frames.every(frame => frame.sourceWidth > 0 && frame.sourceHeight > 0), 'source dimensions must accompany every frame');
  assert.ok(result.frames.every(frame => frame.badVisiblePoints === 0), 'visible points must be finite');
  assert.equal(new Set(result.frames.map(frame => frame.sourceFrameId)).size, result.frames.length, 'each accepted result must correspond to a new decoded source frame');
  for (let i = 1; i < result.frames.length; i++) assert.ok(result.frames[i].timestamp > result.frames[i - 1].timestamp, 'source timestamps must increase');
  assert.equal(result.workers.active, 1, 'one coordinator may run at a time');
  return result;
}

async function createCanvasStream({ imageUrl, width = 1280, height = 720 } = {}) {
  return page.evaluate(async ({ imageUrl, width, height }) => {
    clearInterval(window.__qaCanvasTimer);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    let image;
    if (imageUrl) {
      image = new Image();
      image.src = imageUrl;
      await image.decode();
    }
    window.__qaBlank = !image;
    const draw = () => {
      ctx.fillStyle = '#111';
      ctx.fillRect(0, 0, width, height);
      if (image && !window.__qaBlank) ctx.drawImage(image, 0, 0, width, height);
    };
    draw();
    window.__qaCanvasTimer = setInterval(draw, 1000 / 15);
    window.__qaStream = canvas.captureStream(15);
    window.__qaStream.getVideoTracks()[0].requestFrame?.();
    return { width, height };
  }, { imageUrl, width, height });
}

try {
  browser = await chromium.launch({ ...(chromePath ? { executablePath: chromePath } : {}), headless: true });
  browserCDP = await browser.newBrowserCDPSession();
  await browserCDP.send('Target.setDiscoverTargets', { discover: true });
  page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.setDefaultTimeout(90000);
  page.on('pageerror', error => report.pageErrors.push(error.message));
  await page.addInitScript(() => {
    const NativeWorker = window.Worker;
    const stats = { created: 0, terminated: 0, active: 0, peak: 0 };
    window.__qaWorkers = stats;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        this.__qaAlive = true;
        stats.created++;
        stats.active++;
        stats.peak = Math.max(stats.peak, stats.active);
      }
      terminate() {
        if (this.__qaAlive) { this.__qaAlive = false; stats.active--; stats.terminated++; }
        return super.terminate();
      }
    };
  });
  await page.route('**/__fixtures/*', async route => {
    try {
      const filename = path.basename(new URL(route.request().url()).pathname);
      const isCloseup = filename === 'qa-closeup.png';
      const bytes = await readFile(isCloseup && closeupPath ? closeupPath : path.join(fixtures, filename));
      const contentType = isCloseup ? 'image/png' : 'video/mp4';
      const range = route.request().headers().range?.match(/^bytes=(\d+)-(\d*)$/);
      if (range) {
        const start = Number(range[1]);
        const end = range[2] ? Math.min(Number(range[2]), bytes.length - 1) : bytes.length - 1;
        await route.fulfill({ status: 206, contentType, body: bytes.subarray(start, end + 1), headers: { 'Accept-Ranges': 'bytes', 'Content-Range': `bytes ${start}-${end}/${bytes.length}` } });
      } else {
        await route.fulfill({ status: 200, contentType, body: bytes, headers: { 'Accept-Ranges': 'bytes' } });
      }
    } catch {
      await route.fulfill({ status: 404, body: 'Fixture not found' });
    }
  });
  // Landmark regression scope; the body surface has its own suite (test:dense).
  await page.goto(`${base}/?test=1&surface=0`);
  await page.waitForFunction(() => window.motionCapture?.start && window.motionCapture?.subscribe);

  await test('test mode is idle, camera-only UI has no visible controls or text', async () => {
    const result = await page.evaluate(() => ({ diagnostics: window.motionCapture.getDiagnostics(), workers: window.__qaWorkers, text: document.body.innerText.trim(), canvases: document.querySelectorAll('canvas').length }));
    assert.equal(result.diagnostics.state, 'idle');
    assert.equal(result.workers.active, 0);
    assert.equal(result.text, '');
    assert.ok(result.canvases >= 1);
    return result;
  });

  await test('real body motion yields a 33-point pose with finite timestamped output', async () => {
    const result = await startVideo('body-motion.mp4');
    assert.ok(result.frames.some(frame => frame.poseCount === 33), 'body model should detect the reference person');
    assert.equal(result.diagnostics.workerCount, 4);
    result.actualBrowserWorkers = await expectWorkerTargets(4);
    report.fixtureReports.push({ fixture: 'body-motion.mp4', ...result });
    await page.screenshot({ path: path.join(artifacts, 'body-motion.png') });
    return { frames: result.frames.length, poseFrames: result.frames.filter(frame => frame.poseCount === 33).length };
  });

  await test('real hand motion yields complete 21-point hand detections', async () => {
    const result = await startVideo('hand-signs.mp4');
    assert.ok(result.frames.some(frame => frame.handCounts.some(count => count === 21)), 'hand model should detect the reference hand');
    assert.ok(result.frames.every(frame => frame.handCounts.every(count => count === 21)));
    assert.ok(result.frames.every(frame => frame.poseCount === 0 && frame.faceCount === 0), 'hand-only source must not display a guessed body or face');
    report.fixtureReports.push({ fixture: 'hand-signs.mp4', ...result });
    return { frames: result.frames.length, detectedHands: result.frames.map(frame => frame.handCounts.length) };
  });

  await test('real squat sequence continues to produce full body landmarks', async () => {
    const result = await startVideo('body-squats.mp4');
    assert.ok(result.frames.some(frame => frame.poseCount === 33));
    report.fixtureReports.push({ fixture: 'body-squats.mp4', ...result });
    return { frames: result.frames.length, poseFrames: result.frames.filter(frame => frame.poseCount === 33).length };
  });

  for (const filename of ['face-turns.mp4', 'face-lighting.mp4']) {
    await test(`real ${filename} yields a dense finite face through sampled clip positions`, async () => {
      const result = await startVideo(filename);
      assert.ok(result.frames.some(frame => frame.faceCount === 478), 'face must include 478 landmarks with iris points');
      assert.ok(result.frames.every(frame => frame.faceCount === 0 || frame.faceCount === 478));
      report.fixtureReports.push({ fixture: filename, ...result });
      await page.screenshot({ path: path.join(artifacts, `${path.parse(filename).name}.png`) });
      return { frames: result.frames.length, faceFrames: result.frames.filter(frame => frame.faceCount === 478).length };
    });
  }

  await test('two mirrored crops of a real hand are detected as two independent 21-point hands', async () => {
    await stop();
    await installFrameRecorder();
    await page.evaluate(async () => {
      const source = document.createElement('video');
      source.muted = true; source.playsInline = true;
      source.src = '/__fixtures/hand-signs.mp4';
      await new Promise((resolve, reject) => { source.onloadeddata = resolve; source.onerror = reject; source.load(); });
      source.currentTime = Math.min(0.5, source.duration / 4);
      await new Promise(resolve => { source.onseeked = resolve; });
      const canvas = document.createElement('canvas');
      canvas.width = 1280; canvas.height = 720;
      const ctx = canvas.getContext('2d');
      const sx = source.videoWidth * 0.34, sy = source.videoHeight * 0.3;
      const sw = source.videoWidth * 0.42, sh = source.videoHeight * 0.45;
      const scale = Math.min(560 / sw, 620 / sh), width = sw * scale, height = sh * scale;
      const draw = () => {
        ctx.fillStyle = '#111'; ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(source, sx, sy, sw, sh, 320 - width / 2, 360 - height / 2, width, height);
        ctx.save(); ctx.translate(1280, 0); ctx.scale(-1, 1);
        ctx.drawImage(source, sx, sy, sw, sh, 320 - width / 2, 360 - height / 2, width, height);
        ctx.restore();
      };
      draw(); clearInterval(window.__qaCanvasTimer);
      window.__qaCanvasTimer = setInterval(draw, 1000 / 15);
      window.__qaStream = canvas.captureStream(15);
    });
    await page.evaluate(() => window.motionCapture.start({ stream: window.__qaStream }));
    const result = await waitFrames(10);
    assert.ok(result.frames.some(frame => frame.handCounts.length === 2 && frame.handCounts.every(count => count === 21)), 'both separately visible hand crops should be tracked');
    const dual = result.frames.filter(frame => frame.handCounts.length === 2);
    assert.ok(dual.every(frame => new Set(frame.handIds).size === 2), 'hands must have separate persistent ids');
    report.fixtureReports.push({ fixture: 'two mirrored crops of hand-signs.mp4 (composed input)', ...result });
    await page.screenshot({ path: path.join(artifacts, 'two-hands.png') });
    return { frames: result.frames.length, twoHandFrames: dual.length };
  });

  if (closeupPath) {
    await test('user close-up yields dense 478-point face and detailed hand', async () => {
      await stop();
      await installFrameRecorder();
      await createCanvasStream({ imageUrl: '/__fixtures/qa-closeup.png', width: 1328, height: 1180 });
      await page.evaluate(() => window.motionCapture.start({ stream: window.__qaStream }));
      const result = await waitFrames(8);
      assert.ok(result.frames.some(frame => frame.faceCount === 478), 'close-up must use Face Landmarker, including irises');
      assert.ok(result.frames.some(frame => frame.handCounts.includes(21)), 'raised close-up hand must have individual fingers');
      report.fixtureReports.push({ fixture: 'local user close-up (not copied into report)', ...result });
      await page.screenshot({ path: path.join(artifacts, 'closeup.png') });
      assert.equal(await page.evaluate(() => document.body.innerText.trim()), '');
      return { faceFrames: result.frames.filter(frame => frame.faceCount === 478).length, handFrames: result.frames.filter(frame => frame.handCounts.includes(21)).length };
    });

    await test('blank frames remove all stale person geometry', async () => {
      const before = await page.evaluate(() => { window.__qaBlank = true; return window.__qaFrames.length; });
      await page.waitForFunction(before => {
        const next = window.__qaFrames.slice(before);
        return next.length >= 3 && next.slice(-2).every(frame => frame.poseCount === 0 && frame.faceCount === 0 && frame.handCounts.length === 0);
      }, before, { timeout: 30000 });
      const result = await page.evaluate(() => window.__qaFrames.slice(-3));
      await page.screenshot({ path: path.join(artifacts, 'blank-after-loss.png') });
      return result;
    });
  }

  await test('stop releases every supplied stream track and coordinator', async () => {
    await stop();
    await expectWorkerTargets(0);
    const result = await page.evaluate(() => ({ diagnostics: window.motionCapture.getDiagnostics(), workers: window.__qaWorkers, tracks: window.__qaStream?.getTracks().map(track => track.readyState) ?? [] }));
    assert.equal(result.diagnostics.state, 'idle');
    assert.equal(result.diagnostics.inFlight, false);
    assert.equal(result.diagnostics.workerCount, 0);
    assert.equal(result.workers.active, 0);
    assert.ok(result.tracks.every(state => state === 'ended'));
    result.actualBrowserWorkers = await expectWorkerTargets(0);
    return result;
  });

  await test('stop during startup cancels cleanly without leaking workers or a camera track', async () => {
    await createCanvasStream();
    const result = await page.evaluate(async () => {
      const pending = window.motionCapture.start({ stream: window.__qaStream });
      window.motionCapture.stop();
      const value = await pending;
      return { value, diagnostics: window.motionCapture.getDiagnostics(), workers: window.__qaWorkers, tracks: window.__qaStream.getTracks().map(track => track.readyState) };
    });
    assert.equal(result.value, null);
    assert.equal(result.diagnostics.state, 'idle');
    assert.equal(result.workers.active, 0);
    assert.ok(result.tracks.every(state => state === 'ended'));
    result.actualBrowserWorkers = await expectWorkerTargets(0);
    return result;
  });

  await test('stop resolves a pending permission start immediately and stops a late camera stream', async () => {
    await createCanvasStream();
    const result = await page.evaluate(async () => {
      const original = navigator.mediaDevices.getUserMedia;
      let grant;
      navigator.mediaDevices.getUserMedia = () => new Promise(resolve => { grant = resolve; });
      try {
        const pending = window.motionCapture.start();
        window.motionCapture.stop();
        const value = await Promise.race([pending, new Promise(resolve => setTimeout(() => resolve('still-pending'), 500))]);
        grant(window.__qaStream);
        await new Promise(resolve => setTimeout(resolve, 0));
        return { value, diagnostics: window.motionCapture.getDiagnostics(), workers: window.__qaWorkers, tracks: window.__qaStream.getTracks().map(track => track.readyState) };
      } finally { navigator.mediaDevices.getUserMedia = original; }
    });
    assert.equal(result.value, null);
    assert.equal(result.diagnostics.state, 'idle');
    assert.equal(result.workers.active, 0);
    assert.ok(result.tracks.every(state => state === 'ended'));
    return result;
  });

  await test('a subscriber can stop capture during delivery without reviving a stopped session', async () => {
    await createCanvasStream();
    const result = await page.evaluate(async () => {
      const unsubscribe = window.motionCapture.subscribe(() => window.motionCapture.stop());
      const value = await window.motionCapture.start({ stream: window.__qaStream });
      unsubscribe();
      return { value, diagnostics: window.motionCapture.getDiagnostics(), workers: window.__qaWorkers, tracks: window.__qaStream.getTracks().map(track => track.readyState) };
    });
    assert.equal(result.value, null);
    assert.equal(result.diagnostics.state, 'idle');
    assert.equal(result.workers.active, 0);
    assert.ok(result.tracks.every(state => state === 'ended'));
    return result;
  });

  await test('rapid replacement keeps only the newest source and one coordinator', async () => {
    await stop();
    await installFrameRecorder();
    const result = await page.evaluate(async () => {
      const make = (width, height) => {
        const canvas = document.createElement('canvas');
        canvas.width = width; canvas.height = height;
        const ctx = canvas.getContext('2d');
        const timer = setInterval(() => { ctx.fillStyle = '#111'; ctx.fillRect(0, 0, width, height); }, 66);
        return { stream: canvas.captureStream(15), timer };
      };
      const a = make(640, 480), b = make(960, 540);
      try {
        const oldStart = window.motionCapture.start({ stream: a.stream });
        const newStart = window.motionCapture.start({ stream: b.stream });
        const [oldResult] = await Promise.all([oldStart, newStart]);
        const diagnostics = window.motionCapture.getDiagnostics();
        const workers = { ...window.__qaWorkers };
        window.motionCapture.stop();
        return { oldResult, diagnostics, workers, oldTracks: a.stream.getTracks().map(track => track.readyState), newTracks: b.stream.getTracks().map(track => track.readyState) };
      } finally { clearInterval(a.timer); clearInterval(b.timer); }
    });
    assert.equal(result.oldResult, null);
    assert.equal(result.diagnostics.source.width, 960);
    assert.equal(result.diagnostics.source.height, 540);
    assert.equal(result.workers.active, 1);
    assert.ok(result.oldTracks.every(state => state === 'ended'));
    assert.ok(result.newTracks.every(state => state === 'ended'));
    return result;
  });

  await test('pagehide stops capture and releases supplied stream tracks', async () => {
    await createCanvasStream();
    await page.evaluate(() => window.motionCapture.start({ stream: window.__qaStream }));
    const result = await page.evaluate(() => {
      dispatchEvent(new Event('pagehide'));
      return { diagnostics: window.motionCapture.getDiagnostics(), workers: window.__qaWorkers, tracks: window.__qaStream.getTracks().map(track => track.readyState) };
    });
    assert.equal(result.diagnostics.state, 'idle');
    assert.equal(result.workers.active, 0);
    assert.ok(result.tracks.every(state => state === 'ended'));
    return result;
  });

  await test('unavailable video has a recoverable error and releases resources', async () => {
    const result = await page.evaluate(async () => {
      let rejected = false;
      try { await window.motionCapture.start({ videoUrl: '/__fixtures/does-not-exist.mp4' }); } catch { rejected = true; }
      return { rejected, diagnostics: window.motionCapture.getDiagnostics(), workers: window.__qaWorkers, text: document.body.innerText.trim() };
    });
    assert.equal(result.rejected, true);
    assert.equal(result.diagnostics.state, 'error');
    assert.equal(result.workers.active, 0);
    assert.ok(result.text.length > 0, 'error should explain recovery');
    return result;
  });

  await test('no uncaught page exceptions occurred', () => { assert.deepEqual(report.pageErrors, []); });
} catch (error) {
  report.fatalError = error.stack ?? String(error);
  console.error(report.fatalError);
} finally {
  if (page) await stop().catch(() => {});
  if (browser) await browser.close();
  report.finishedAt = new Date().toISOString();
  report.passed = !report.fatalError && report.tests.every(test => test.passed);
  await writeFile(path.join(artifacts, 'browser-report.json'), JSON.stringify(report, null, 2));
  console.log(`Report: ${path.join(artifacts, 'browser-report.json')}`);
  if (!report.passed) process.exitCode = 1;
}
