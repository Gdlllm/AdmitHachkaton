/** Real-model transition diagnostic: hand → body → face → blank in ONE stream.
 * No model/session reset between phases. Actual decoded fixture video is drawn
 * onto a fixed 1280×720 canvas. A tiny corner marker labels the returned source
 * bitmap, so an in-flight old frame cannot be assigned to a new phase by mistake.
 *
 * CAPTURE_FIXTURES_DIR=/path/to/fixtures node tests/transition.mjs
 * Output: tests/artifacts/transition-report.json and four screenshots.
 */
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const project = fileURLToPath(new URL('../', import.meta.url));
const base = process.env.CAPTURE_BASE_URL ?? 'http://127.0.0.1:5173';
const fixtures = process.env.CAPTURE_FIXTURES_DIR ?? path.join(project, 'tests/fixtures');
const artifacts = process.env.CAPTURE_QA_OUT ?? path.join(project, 'tests/artifacts');
const allowedFiles = new Set(['hand-signs.mp4', 'body-squats.mp4', 'face-turns.mp4']);
let executablePath = process.env.CAPTURE_CHROME_PATH;
if (!executablePath && process.platform === 'darwin') {
  const candidate = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  try { await access(candidate); executablePath = candidate; } catch {}
}
await mkdir(artifacts, { recursive: true });
const report = {
  startedAt: new Date().toISOString(), passed: false, phases: [], pageErrors: [],
  scope: 'Diagnostic on a composed fixed-resolution stream of real fixture videos. Counts and reacquisition delay do not measure ground-truth accuracy.',
  input: { width: 1280, height: 720, fps: 15, samplesPerPhase: 24,
    note: 'Hand video is cropped for close-up; body and face videos use aspect-preserving contain. A 24px corner marker identifies each returned source frame.' },
};
let browser, page, cdp;

async function workerTargets() {
  const { targetInfos } = await cdp.send('Target.getTargets');
  return targetInfos.filter(target => target.type === 'worker' && target.url.includes('/pose-worker.js'))
    .map(target => target.targetId).sort();
}

try {
  browser = await chromium.launch({ ...(executablePath ? { executablePath } : {}), headless: true });
  cdp = await browser.newBrowserCDPSession();
  await cdp.send('Target.setDiscoverTargets', { discover: true });
  page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.setDefaultTimeout(90000);
  page.on('pageerror', error => report.pageErrors.push(error.message));
  await page.route('**/__transition_assets/*', async route => {
    const filename = path.basename(new URL(route.request().url()).pathname);
    if (!allowedFiles.has(filename)) { await route.fulfill({ status: 404, body: 'Unknown test fixture' }); return; }
    const bytes = await readFile(path.join(fixtures, filename));
    const range = route.request().headers().range?.match(/^bytes=(\d+)-(\d*)$/);
    if (range) {
      const start = Number(range[1]);
      const end = range[2] ? Math.min(Number(range[2]), bytes.length - 1) : bytes.length - 1;
      await route.fulfill({ status: 206, contentType: 'video/mp4', body: bytes.subarray(start, end + 1),
        headers: { 'Accept-Ranges': 'bytes', 'Content-Range': `bytes ${start}-${end}/${bytes.length}` } });
    } else await route.fulfill({ status: 200, contentType: 'video/mp4', body: bytes, headers: { 'Accept-Ranges': 'bytes' } });
  });
  await page.goto(`${base}/?test=1`);
  await page.waitForFunction(() => window.motionCapture?.subscribe);
  await page.evaluate(async () => {
    const names = { hand: 'hand-signs.mp4', body: 'body-squats.mp4', face: 'face-turns.mp4' };
    const videos = {};
    for (const [name, filename] of Object.entries(names)) {
      const video = document.createElement('video');
      video.muted = true; video.playsInline = true; video.loop = true;
      video.src = `/__transition_assets/${filename}`;
      await new Promise((resolve, reject) => { video.onloadeddata = resolve; video.onerror = reject; video.load(); });
      videos[name] = video;
    }
    const canvas = document.createElement('canvas');
    canvas.width = 1280; canvas.height = 720;
    const ctx = canvas.getContext('2d');
    const marker = document.createElement('canvas'); marker.width = marker.height = 1;
    const markerContext = marker.getContext('2d', { willReadFrequently: true });
    const colors = { hand: '#f01414', body: '#14f014', face: '#1414f0', blank: '#f0f014' };
    window.__transition = { canvas, videos, phase: 'hand', frames: [], changedAt: { hand: performance.now() }, timer: null };
    const state = window.__transition;
    state.draw = () => {
      ctx.fillStyle = '#111'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      const source = videos[state.phase];
      if (source) {
        let sx = 0, sy = 0, sw = source.videoWidth, sh = source.videoHeight;
        if (state.phase === 'hand') { sx = sw * .34; sy = sh * .3; sw *= .42; sh *= .45; }
        const scale = Math.min(1200 / sw, 680 / sh), width = sw * scale, height = sh * scale;
        ctx.drawImage(source, sx, sy, sw, sh, (1280 - width) / 2, (720 - height) / 2, width, height);
      }
      ctx.fillStyle = colors[state.phase]; ctx.fillRect(0, 0, 24, 24);
    };
    state.select = async name => {
      Object.values(videos).forEach(video => video.pause());
      state.phase = name; state.changedAt[name] = performance.now();
      if (videos[name]) { videos[name].currentTime = 0; await videos[name].play(); }
      state.draw();
    };
    state.unsubscribe = motionCapture.subscribe(frame => {
      markerContext.drawImage(frame.source, 12, 12, 1, 1, 0, 0, 1, 1);
      const [r, g, b] = markerContext.getImageData(0, 0, 1, 1).data;
      const phase = r > 100 && g > 100 ? 'blank' : r > g && r > b ? 'hand' : g > r && g > b ? 'body' : 'face';
      state.frames.push({
        phase, timestamp: frame.timestamp, receivedAt: frame.receivedAt,
        frameId: frame.frameId, sourceFrameId: frame.sourceFrameId,
        width: frame.sourceWidth, height: frame.sourceHeight,
        pose: frame.pose?.landmarks?.length ?? 0, face: frame.face?.landmarks?.length ?? 0,
        hands: (frame.hands ?? []).map(hand => hand.landmarks.length),
        fusion: frame.tracking?.fusion ?? frame.fusion ?? null,
        inferenceMs: frame.inferenceMs, latencyMs: frame.latencyMs,
      });
    });
    await state.select('hand');
    state.stream = canvas.captureStream(15);
    state.timer = setInterval(state.draw, 1000 / 15);
    await motionCapture.start({ stream: state.stream });
  });
  const initial = await page.evaluate(() => motionCapture.getDiagnostics());
  report.workerIds = await workerTargets();
  assert.equal(report.workerIds.length, 4);

  for (const name of ['hand', 'body', 'face', 'blank']) {
    if (name !== 'hand') await page.evaluate(name => window.__transition.select(name), name);
    await page.waitForFunction(name => window.__transition.frames.filter(frame => frame.phase === name).length >= 24, name, { timeout: 30000 });
    const data = await page.evaluate(name => ({
      frames: window.__transition.frames.filter(frame => frame.phase === name),
      changedAt: window.__transition.changedAt[name], diagnostics: motionCapture.getDiagnostics(),
    }), name);
    const frames = data.frames.slice(0, 24);
    const expected = frame => name === 'hand' ? frame.hands.includes(21)
      : name === 'body' ? frame.pose === 33
        : name === 'face' ? frame.face === 478
          : frame.pose === 0 && frame.face === 0 && frame.hands.length === 0;
    const first = frames.findIndex(expected);
    const phase = { name, samples: frames.length, firstExpectedFrame: first < 0 ? null : first + 1,
      firstExpectedDelayMs: first < 0 ? null : frames[first].receivedAt - data.changedAt,
      poseFrames: frames.filter(frame => frame.pose === 33).length,
      faceFrames: frames.filter(frame => frame.face === 478).length,
      handFrames: frames.filter(frame => frame.hands.includes(21)).length,
      expectedFrames: frames.filter(expected).length, frames };
    report.phases.push(phase);
    console.log(JSON.stringify({ ...phase, frames: undefined }));
    await page.screenshot({ path: path.join(artifacts, `transition-${name}.png`) });
    assert.ok(frames.every(frame => frame.width === 1280 && frame.height === 720), 'source size must remain constant');
    assert.equal(data.diagnostics.generation, initial.generation, 'the session must never restart between phases');
    assert.deepEqual(await workerTargets(), report.workerIds, 'the same workers must survive all scene changes');
    assert.ok(first >= 0 && first < 8, `${name}: the expected region must reacquire within eight observed source frames`);
    assert.ok(frames.slice(-5).every(expected), `${name}: the last five frames must contain the expected region (or no geometry for blank)`);
  }
  report.finalDiagnostics = await page.evaluate(() => motionCapture.getDiagnostics());
  assert.deepEqual(report.pageErrors, []);
  report.passed = true;
} catch (error) {
  report.error = error.stack ?? String(error);
  process.exitCode = 1;
  console.error(report.error);
} finally {
  if (page) await page.evaluate(() => {
    motionCapture.stop();
    const state = window.__transition;
    if (state) { clearInterval(state.timer); state.unsubscribe(); Object.values(state.videos).forEach(video => video.pause()); }
  }).catch(() => {});
  if (cdp) {
    for (let attempt = 0; attempt < 30; attempt++) {
      report.remainingWorkerIds = await workerTargets();
      if (!report.remainingWorkerIds.length) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    if (report.remainingWorkerIds.length) { report.passed = false; process.exitCode = 1; }
  }
  report.finishedAt = new Date().toISOString();
  await writeFile(path.join(artifacts, 'transition-report.json'), JSON.stringify(report, null, 2));
  if (browser) await browser.close();
}
