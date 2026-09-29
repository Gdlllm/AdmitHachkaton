/** Continuous real-video QA: every clip plays from its first frame to `ended`.
 * No detector mocks, seeking or fixed sample cap. Run separately from other GPU
 * benchmarks. Measurements are local throughput/continuity, not ground-truth accuracy.
 * PLAYWRIGHT_MODULE and CAPTURE_CHROME_PATH are optional explicit tool paths.
 * CAPTURE_FIXTURES_DIR supplies the same five public fixtures as browser.mjs.
 * CAPTURE_SOAK_CLIPS optionally narrows a diagnostic run to comma-separated fixture names.
 */
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, access } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';

const project = fileURLToPath(new URL('../', import.meta.url));
const base = process.env.CAPTURE_BASE_URL ?? 'http://127.0.0.1:5173';
const fixtures = process.env.CAPTURE_FIXTURES_DIR ?? path.join(project, 'tests/fixtures');
const artifacts = process.env.CAPTURE_QA_OUT ?? path.join(project, 'tests/artifacts');
const knownClips = ['body-motion.mp4', 'body-squats.mp4', 'hand-signs.mp4', 'face-turns.mp4', 'face-lighting.mp4'];
const clips = process.env.CAPTURE_SOAK_CLIPS?.split(',').map(name => name.trim()).filter(Boolean) ?? knownClips;
assert.ok(clips.length > 0 && clips.every(name => knownClips.includes(name)), 'CAPTURE_SOAK_CLIPS must name existing fixture clips');
const playwright = process.env.PLAYWRIGHT_MODULE ? await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href) : await import('playwright');
let executablePath = process.env.CAPTURE_CHROME_PATH;
if (!executablePath && process.platform === 'darwin') {
  const systemChrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  try { await access(systemChrome); executablePath = systemChrome; } catch {}
}
await mkdir(artifacts, { recursive: true });
const report = {
  startedAt: new Date().toISOString(), base,
  scope: 'Sustained real-time playback of the selected whole real videos, no seeks and no fixed frame cap. Observed throughput and continuity only; fixtures have no ground-truth landmarks or identities. The JavaScript heap metric covers the main renderer realm, not total process/GPU/worker memory.',
  selectedClips: clips,
  hardware: { platform: process.platform, architecture: process.arch, osRelease: os.release(), cpu: os.cpus()[0]?.model, logicalCPUs: os.cpus().length, totalMemoryBytes: os.totalmem() },
  clips: [], pageErrors: [], screenshots: [],
};
const percentile = (values, fraction) => {
  const sorted = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor((sorted.length - 1) * fraction)] : null;
};
let browser, page, cdp, browserCDP;
async function workerTargets() {
  const { targetInfos } = await browserCDP.send('Target.getTargets');
  return targetInfos.filter(target => target.type === 'worker' && target.url.includes('/pose-worker.js')).map(target => ({ id: target.targetId, url: target.url }));
}
async function expectWorkers(count) {
  const deadline = Date.now() + 6000;
  let targets;
  do {
    targets = await workerTargets();
    if (targets.length === count) return targets;
    await new Promise(resolve => setTimeout(resolve, 100));
  } while (Date.now() < deadline);
  assert.equal(targets.length, count, JSON.stringify(targets));
}
async function heap() {
  const { metrics } = await cdp.send('Performance.getMetrics');
  return Object.fromEntries(metrics.filter(metric => ['JSHeapUsedSize', 'JSHeapTotalSize', 'Nodes', 'Documents'].includes(metric.name)).map(metric => [metric.name, metric.value]));
}
function absentWindows(frames, key) {
  const groups = []; let active = null;
  for (const frame of frames) {
    if (!frame[key]) {
      if (!active) active = { from: frame.mediaTime, to: frame.mediaTime, frames: 0 };
      active.to = frame.mediaTime; active.frames++;
    } else if (active) { groups.push(active); active = null; }
  }
  if (active) groups.push(active);
  return groups;
}
function analyzeContinuity(frames) {
  const largeAnchorSteps = [], possibleHandReassignments = [];
  for (let index = 1; index < frames.length; index++) {
    const previous = frames[index - 1], current = frames[index];
    const dt = current.mediaTime - previous.mediaTime;
    if (dt <= 0 || dt > 0.25) continue;
    const diagonal = Math.hypot(current.sourceWidth, current.sourceHeight);
    const distance = (a, b) => Math.hypot((a.x - b.x) * current.sourceWidth, (a.y - b.y) * current.sourceHeight);
    for (const part of current.anchors) {
      const old = previous.anchors.find(value => value.id === part.id);
      if (old && distance(old, part) / diagonal > 0.20) largeAnchorSteps.push({ time: current.mediaTime, id: part.id, imageDiagonalFraction: distance(old, part) / diagonal, elapsedSeconds: dt });
    }
    if (current.handIds.length === previous.handIds.length) {
      for (const hand of current.anchors.filter(value => value.kind === 'hand')) {
        if (previous.handIds.includes(hand.id)) continue;
        const nearby = previous.anchors.filter(value => value.kind === 'hand').sort((a, b) => distance(a, hand) - distance(b, hand))[0];
        if (nearby && distance(nearby, hand) / diagonal < 0.15) possibleHandReassignments.push({ time: current.mediaTime, from: nearby.id, to: hand.id, imageDiagonalFraction: distance(nearby, hand) / diagonal });
      }
    }
  }
  return {
    definitions: 'Large anchor step: >20% of image diagonal within <=250ms for the same track; can be true fast motion. Possible hand reassignment: consecutive equal hand counts, a new track ID within 15% of image diagonal of a previous hand; identity errors require manual/ground-truth verification.',
    largeAnchorSteps, possibleHandReassignments,
  };
}

try {
  browser = await playwright.chromium.launch({ ...(executablePath ? { executablePath } : {}), headless: true });
  report.browser = { executable: executablePath ?? 'Playwright Chromium', version: browser.version(), headless: true };
  browserCDP = await browser.newBrowserCDPSession();
  await browserCDP.send('Target.setDiscoverTargets', { discover: true });
  page = await browser.newPage({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(120000);
  page.on('pageerror', error => report.pageErrors.push(error.message));
  cdp = await page.context().newCDPSession(page);
  await cdp.send('Performance.enable');
  await page.route('**/__fixtures/*', async route => {
    try {
      const name = path.basename(new URL(route.request().url()).pathname);
      const bytes = await readFile(path.join(fixtures, name));
      const range = route.request().headers().range?.match(/^bytes=(\d+)-(\d*)$/);
      if (range) {
        const start = Number(range[1]), end = range[2] ? Math.min(Number(range[2]), bytes.length - 1) : bytes.length - 1;
        await route.fulfill({ status: 206, contentType: 'video/mp4', body: bytes.subarray(start, end + 1), headers: { 'Accept-Ranges': 'bytes', 'Content-Range': `bytes ${start}-${end}/${bytes.length}` } });
      } else await route.fulfill({ contentType: 'video/mp4', body: bytes, headers: { 'Accept-Ranges': 'bytes' } });
    } catch { await route.fulfill({ status: 404, body: 'Fixture not found' }); }
  });
  await page.addInitScript(() => {
    const NativeWorker = window.Worker;
    window.__soakRawCandidates = new Map();
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        // Registered before the application handler. Keep unmodified model
        // coordinates; the subscription later decides whether output is false.
        this.addEventListener('message', ({ data }) => {
          if (!window.__soakCollectHand || data.type !== 'result' ||
            (!data.pose?.landmarks?.length && !data.face?.landmarks?.length)) return;
          window.__soakRawCandidates.set(data.frameId, structuredClone({
            frameId: data.frameId, timestamp: data.timestamp, mediaTime: data.mediaTime,
            sourceWidth: data.sourceWidth, sourceHeight: data.sourceHeight,
            pose: data.pose, face: data.face, hands: data.hands,
          }));
          if (window.__soakRawCandidates.size > 16) window.__soakRawCandidates.delete(window.__soakRawCandidates.keys().next().value);
        });
      }
    };
  });
  // Landmark regression scope; the body surface has its own suite (test:dense).
  await page.goto(`${base}/?test=1&surface=0`);
  await page.waitForFunction(() => window.motionCapture?.subscribe);
  report.browser.environment = await page.evaluate(() => {
    const canvas = document.createElement('canvas'), gl = canvas.getContext('webgl2');
    const extension = gl?.getExtension('WEBGL_debug_renderer_info');
    const gpu = extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : null;
    gl?.getExtension('WEBGL_lose_context')?.loseContext();
    return { userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency, deviceMemoryGB: navigator.deviceMemory, gpu, viewport: { width: innerWidth, height: innerHeight, dpr: devicePixelRatio } };
  });
  for (const filename of clips) {
    const clip = { filename, startedAt: new Date().toISOString(), passed: false };
    report.clips.push(clip);
    try {
      await page.evaluate(filename => {
        window.motionCapture.stop(); window.__soakUnsubscribe?.();
        window.__soakFrames = []; window.__soakMissingImages = []; window.__soakSeenFace = false;
        window.__soakPhantoms = []; window.__soakRawCandidates.clear();
        window.__soakCollectHand = filename === 'hand-signs.mp4';
        window.__soakUnsubscribe = window.motionCapture.subscribe(frame => {
          const pointsValid = point => point && point.drawConfidence > 0 && Number.isFinite(point.x) && Number.isFinite(point.y);
          const anchors = [];
          const poseAnchor = frame.pose?.landmarks?.[11];
          if (pointsValid(poseAnchor)) anchors.push({ kind: 'pose', id: 'pose', x: poseAnchor.x, y: poseAnchor.y });
          const faceAnchor = frame.face?.landmarks?.[4];
          if (pointsValid(faceAnchor)) anchors.push({ kind: 'face', id: 'face', x: faceAnchor.x, y: faceAnchor.y });
          for (const hand of frame.hands ?? []) {
            const wrist = hand.landmarks[0];
            if (pointsValid(wrist)) anchors.push({ kind: 'hand', id: hand.track?.id, x: wrist.x, y: wrist.y });
          }
          const parts = [frame.pose, frame.face, ...(frame.hands ?? [])].filter(Boolean);
          const sample = {
            timestamp: frame.timestamp, receivedAt: frame.receivedAt, mediaTime: frame.mediaTime, frameId: frame.frameId, sourceFrameId: frame.sourceFrameId,
            sourceWidth: frame.sourceWidth, sourceHeight: frame.sourceHeight,
            poseCount: frame.pose?.landmarks?.length ?? 0, faceCount: frame.face?.landmarks?.length ?? 0,
            handIds: (frame.hands ?? []).map(hand => hand.track?.id), handCounts: (frame.hands ?? []).map(hand => hand.landmarks.length),
            anchors, latencyMs: frame.latencyMs, inferenceMs: frame.inferenceMs,
            smoothing: frame.tracking?.smoothing, rejectedPoints: frame.tracking?.rejectedPoints,
            badVisiblePoints: parts.flatMap(part => part.landmarks).filter(point => point.drawConfidence > 0 && (!Number.isFinite(point.x) || !Number.isFinite(point.y))).length,
          };
          window.__soakFrames.push(sample);
          if (window.__soakCollectHand && (sample.poseCount || sample.faceCount) && window.__soakPhantoms.length < 8) {
            const canvas = document.createElement('canvas'); canvas.width = frame.sourceWidth; canvas.height = frame.sourceHeight;
            canvas.getContext('2d').drawImage(frame.source, 0, 0);
            window.__soakPhantoms.push({
              sample, raw: window.__soakRawCandidates.get(frame.frameId) ?? null,
              output: structuredClone({ pose: frame.pose, face: frame.face, hands: frame.hands, tracking: frame.tracking }),
              png: canvas.toDataURL('image/png'),
            });
          }
          if (sample.faceCount) window.__soakSeenFace = true;
          const lastImage = window.__soakMissingImages.at(-1);
          if (filename === 'face-turns.mp4' && !sample.faceCount && window.__soakSeenFace && window.__soakMissingImages.length < 3 && (!lastImage || sample.mediaTime - lastImage.mediaTime > 0.35)) {
            // Save exact missing-detection source frames. No live-camera capture.
            const canvas = document.createElement('canvas'); canvas.width = frame.sourceWidth; canvas.height = frame.sourceHeight;
            canvas.getContext('2d').drawImage(frame.source, 0, 0);
            window.__soakMissingImages.push({ mediaTime: sample.mediaTime, dataUrl: canvas.toDataURL('image/jpeg', 0.9) });
          }
        });
      }, filename);
      clip.heapBefore = await heap();
      await page.evaluate(videoUrl => window.motionCapture.start({ videoUrl, loop: false }), `/__fixtures/${filename}`);
      clip.activeWorkers = await expectWorkers(4);
      clip.durationSeconds = await page.evaluate(() => document.getElementById('camera').duration);
      assert.ok(Number.isFinite(clip.durationSeconds) && clip.durationSeconds > 0);
      console.log(`Playing whole ${filename}: ${clip.durationSeconds.toFixed(2)}s`);
      await page.waitForFunction(() => {
        const diagnostics = window.motionCapture.getDiagnostics();
        return diagnostics.state === 'error' || (document.getElementById('camera').ended && !diagnostics.inFlight);
      }, null, { timeout: clip.durationSeconds * 1000 + 60000 });
      const result = await page.evaluate(() => ({ frames: window.__soakFrames, missingImages: window.__soakMissingImages, phantoms: window.__soakPhantoms, diagnostics: window.motionCapture.getDiagnostics(), ended: document.getElementById('camera').ended, mediaTime: document.getElementById('camera').currentTime, playbackQuality: (() => { const value = document.getElementById('camera').getVideoPlaybackQuality?.(); return value ? { totalVideoFrames: value.totalVideoFrames, droppedVideoFrames: value.droppedVideoFrames, corruptedVideoFrames: value.corruptedVideoFrames } : null; })() }));
      clip.frames = result.frames; clip.diagnostics = result.diagnostics;
      // Persist exact source + raw/processed geometry BEFORE assertions so a
      // red run leaves enough evidence to reproduce and diagnose the failure.
      clip.phantoms = [];
      for (const [index, phantom] of result.phantoms.entries()) {
        const { png, ...geometry } = phantom;
        const imageFile = `soak-hand-phantom-${index + 1}.png`;
        await writeFile(path.join(artifacts, imageFile), Buffer.from(png.split(',')[1], 'base64'));
        clip.phantoms.push({ ...geometry, imageFile });
      }
      assert.equal(result.ended, true, 'whole clip must reach its natural end');
      assert.ok(result.frames[0]?.mediaTime <= 0.05, 'model warmup must not skip the beginning of the video');
      assert.equal(result.diagnostics.state, 'running');
      assert.ok(result.frames.length > 2);
      assert.ok(result.frames.every(frame => frame.badVisiblePoints === 0));
      assert.ok(result.frames.every(frame => (frame.smoothing?.maxResidualPx ?? 0) <= 4.000001), 'smoothing residual must remain within the 4px maximum region bound');
      for (let index = 1; index < result.frames.length; index++) {
        assert.ok(result.frames[index].timestamp > result.frames[index - 1].timestamp, 'capture timestamps strictly increase');
        assert.ok(result.frames[index].mediaTime > result.frames[index - 1].mediaTime, 'each result must refer to a new decoded video frame, with no repeats or backward seeks');
        assert.ok(result.frames[index].sourceFrameId > result.frames[index - 1].sourceFrameId, 'decoded source frame identifiers strictly increase');
        assert.ok(result.frames[index].frameId > result.frames[index - 1].frameId, 'frame identifiers strictly increase');
      }
      const frames = result.frames, steady = frames.slice(1), first = steady[0], last = steady.at(-1);
      clip.summary = {
        frames: frames.length, sourceTimeFrom: frames[0].mediaTime, sourceTimeTo: last.mediaTime,
        sampledTimeSpanPercent: (last.mediaTime - frames[0].mediaTime) / clip.durationSeconds * 100,
        deliveredFramesPerPlaybackSecond: (steady.length - 1) / (last.mediaTime - first.mediaTime),
        deliveredFramesPerWallSecond: (steady.length - 1) * 1000 / (last.receivedAt - first.receivedAt),
        coldFirstLatencyMs: frames[0].latencyMs,
        steadyLatencyP50Ms: percentile(steady.map(frame => frame.latencyMs), 0.5), steadyLatencyP95Ms: percentile(steady.map(frame => frame.latencyMs), 0.95),
        steadyInferenceP50Ms: percentile(steady.map(frame => frame.inferenceMs), 0.5), steadyInferenceP95Ms: percentile(steady.map(frame => frame.inferenceMs), 0.95),
        poseFrames: frames.filter(frame => frame.poseCount === 33).length, faceFrames: frames.filter(frame => frame.faceCount === 478).length,
        anyHandFrames: frames.filter(frame => frame.handCounts.length).length, twoHandFrames: frames.filter(frame => frame.handCounts.length === 2).length,
        maxSmoothingResidualPx: Math.max(...frames.map(frame => frame.smoothing?.maxResidualPx ?? 0)),
        rejectedPoints: frames.reduce((sum, frame) => sum + (frame.rejectedPoints ?? 0), 0),
      };
      clip.continuity = analyzeContinuity(frames);
      clip.faceMissingWindows = absentWindows(frames, 'faceCount');
      clip.frames = frames; clip.diagnostics = result.diagnostics; clip.playbackQuality = result.playbackQuality;
      if (filename === 'hand-signs.mp4') {
        assert.equal(clip.summary.poseFrames, 0, 'a hand-only clip must not invent a human body');
        assert.equal(clip.summary.faceFrames, 0, 'a hand-only clip must not invent a face');
      }
      for (const [index, image] of result.missingImages.entries()) {
        const output = `soak-face-missing-${index + 1}.jpg`;
        await writeFile(path.join(artifacts, output), Buffer.from(image.dataUrl.split(',')[1], 'base64'));
        report.screenshots.push({ filename: output, source: filename, mediaTime: image.mediaTime });
      }
      await page.evaluate(() => { window.motionCapture.stop(); window.__soakUnsubscribe?.(); window.__soakUnsubscribe = null; window.__soakFrames = []; window.__soakMissingImages = []; });
      clip.workersAfterStop = await expectWorkers(0);
      await cdp.send('HeapProfiler.collectGarbage');
      clip.heapAfterStopAndGC = await heap();
      clip.passed = true;
      console.log(JSON.stringify({ filename, ...clip.summary, possibleHandReassignments: clip.continuity.possibleHandReassignments.length, largeAnchorSteps: clip.continuity.largeAnchorSteps.length }));
    } catch (error) { clip.error = error.stack ?? String(error); console.error(`FAIL ${filename}: ${error.message}`); await page.evaluate(() => window.motionCapture.stop()).catch(() => {}); }
    clip.finishedAt = new Date().toISOString();
    await writeFile(path.join(artifacts, 'soak-report.json'), JSON.stringify(report, null, 2));
  }
} catch (error) { report.fatalError = error.stack ?? String(error); console.error(report.fatalError); }
finally {
  await browser?.close();
  report.finishedAt = new Date().toISOString();
  report.passed = !report.fatalError && !report.pageErrors.length && report.clips.length === clips.length && report.clips.every(clip => clip.passed);
  await writeFile(path.join(artifacts, 'soak-report.json'), JSON.stringify(report, null, 2));
  console.log(`Report: ${path.join(artifacts, 'soak-report.json')}`);
  if (!report.passed) process.exitCode = 1;
}
