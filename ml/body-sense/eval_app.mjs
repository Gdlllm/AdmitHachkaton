/** Plays video files through the real app (Pose, Face, Hands, fusion, MHR
 * surface) in headless Chrome and records, per published frame, what the
 * body-sense model sees and what the surface did. No camera is used.
 *
 *   node ml/body-sense/eval_app.mjs OUT_DIR VIDEO_DIR name1.mp4 [name2.mp4 ...]
 *
 * OUT_DIR/<name>.json: frames [{t, pose[33][x,y,v,p], face {yaw,pitch} | null,
 * body, reason, active, fitError, sense, mesh {kp: [[x,y,z] x 70]} | null}]
 * and OUT_DIR/<name>-<i>.jpg snapshots every SNAP_S seconds of video.
 * EVAL_QUERY is appended to the page URL (e.g. "&sense=0").
 * EVAL_CROP=file.json maps a video name to {zoom, x, y}: the video is drawn
 * enlarged around that point (normalized video coordinates) into a 1280x720
 * stream, like a webcam close to the person.
 */
import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';

const [outDir, videoDir, ...names] = process.argv.slice(2);
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../..');
const port = Number(process.env.EVAL_PORT ?? 5176);
const snapEvery = Number(process.env.SNAP_S ?? 1.5);
const query = process.env.EVAL_QUERY ?? '';
const crops = process.env.EVAL_CROP ? JSON.parse(await readFile(process.env.EVAL_CROP, 'utf8')) : {};
let server, browser;
try {
  await mkdir(outDir, { recursive: true });
  server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
  // Never leave the dev server behind, whatever ends this process.
  process.on('exit', () => server?.kill('SIGTERM'));
  for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(signal, () => process.exit(130));
  process.on('uncaughtException', error => { console.error(error); process.exit(1); });
  process.on('unhandledRejection', error => { console.error(error); process.exit(1); });
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('server did not start')), 15000);
    server.stdout.on('data', data => { if (String(data).includes(String(port))) { clearTimeout(timeout); resolve(); } });
    server.once('error', reject);
  });
  browser = await chromium.launch({ executablePath: process.env.CAPTURE_CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.setDefaultTimeout(180000);
  page.on('pageerror', error => console.error('pageerror', error.message));
  await page.addInitScript(() => { navigator.mediaDevices.getUserMedia = () => { throw new Error('no camera in eval'); }; });
  await page.route('**/__videos/*', async route => {
    let bytes;
    try { bytes = await readFile(path.join(videoDir, decodeURIComponent(path.basename(new URL(route.request().url()).pathname)))); }
    catch { await route.fulfill({ status: 404, body: '' }); return; }
    const match = route.request().headers().range?.match(/^bytes=(\d+)-(\d*)$/);
    if (match) {
      const start = Number(match[1]), end = match[2] ? Math.min(Number(match[2]), bytes.length - 1) : bytes.length - 1;
      await route.fulfill({ status: 206, contentType: 'video/mp4', body: bytes.subarray(start, end + 1), headers: { 'Accept-Ranges': 'bytes', 'Content-Range': `bytes ${start}-${end}/${bytes.length}` } });
    } else await route.fulfill({ contentType: 'video/mp4', body: bytes });
  });
  await page.goto(`http://127.0.0.1:${port}/?test=1${query}`);
  await page.waitForFunction(() => window.motionCapture);
  const load = await page.evaluate(async () => {
    const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 480;
    const context = canvas.getContext('2d'), timer = setInterval(() => { context.fillStyle = '#222'; context.fillRect(0, 0, 640, 480); }, 66);
    await window.motionCapture.start({ stream: canvas.captureStream(15), surface: true });
    while (!['ready', 'error', 'unsupported'].includes(window.motionCapture.getDiagnostics().surface.state)) await new Promise(r => setTimeout(r, 100));
    clearInterval(timer); const state = window.motionCapture.getDiagnostics().surface; window.motionCapture.stop(); return state;
  });
  if (load.state !== 'ready') throw new Error(`surface ${load.state}: ${load.error}`);
  for (const name of names) {
    const began = Date.now();
    const crop = crops[name.replace(/\.\w+$/, '')] ?? null;
    await page.evaluate(async ({ name, crop, skeleton }) => {
      window.__evalUnsubscribe?.(); window.__evalFrames = []; window.__evalClip = null;
      const round = v => Math.round(v * 1e4) / 1e4;
      let stream = null;
      if (crop) {
        const clip = document.createElement('video'); clip.src = `/__videos/${encodeURIComponent(name)}`; clip.muted = true;
        await new Promise(resolve => { clip.onloadeddata = resolve; });
        const canvas = document.createElement('canvas'); canvas.width = 1280; canvas.height = 720;
        const context = canvas.getContext('2d');
        const scale = crop.zoom * Math.max(1280 / clip.videoWidth, 720 / clip.videoHeight);
        const draw = () => {
          context.fillStyle = '#000'; context.fillRect(0, 0, 1280, 720);
          context.drawImage(clip, 640 - crop.x * clip.videoWidth * scale, 360 - crop.y * clip.videoHeight * scale, clip.videoWidth * scale, clip.videoHeight * scale);
          if (!clip.ended) requestAnimationFrame(draw);
        };
        window.__evalClip = clip; stream = canvas.captureStream(30);
        await clip.play(); draw();
      }
      window.__evalUnsubscribe = window.motionCapture.subscribe(frame => {
        const m = frame.face?.transformationMatrix?.data ?? frame.face?.transformationMatrix;
        // Column-major 4x4: yaw from the rotated forward axis (third column).
        const face = frame.face?.landmarks?.length && m ? { yaw: round(Math.atan2(m[8], m[10])), pitch: round(Math.asin(Math.max(-1, Math.min(1, -m[9])))) } : null;
        const g = frame.bodySurface, kp = g?.keypoints70;
        const s = frame.surface ?? {};
        window.__evalFrames.push({ t: round(window.__evalClip ? window.__evalClip.currentTime : frame.mediaTime ?? 0),
          pose: (frame.pose?.landmarks ?? []).slice(0, 33).map(p => [round(p.x), round(p.y), round(p.drawConfidence === 0 ? 0 : p.visibility ?? 0), round(p.presence ?? 0)]),
          world: (frame.pose?.worldLandmarks ?? []).slice(0, 33).map(p => [round(p.x), round(p.y), round(p.z)]),
          motion: frame.motion ? { exercise: frame.motion.exercise, hint: frame.motion.hint, highlight: frame.motion.highlight,
            events: frame.motion.events, facing: frame.motion.facing, gestures: Object.entries(frame.motion.gestures ?? {}).filter(([, v]) => v).map(([k]) => k),
            knee: frame.motion.angles ? [frame.motion.angles.knee.left.value, frame.motion.angles.knee.right.value] : null,
            drift: frame.motion.angles ? [frame.motion.angles.kneeDrift.left.value, frame.motion.angles.kneeDrift.right.value] : null,
            trunk: frame.motion.angles?.trunk?.forward ?? null, kneePastToe: frame.motion.measures?.kneePastToe ?? null,
            kneeToAnkle: frame.motion.measures?.kneeToAnkleWidth ?? null } : null,
          face, body: s.body ?? null, reason: s.reason ?? null, active: s.active ?? false, fitError: s.fit?.normalizedError ?? null,
          sense: s.sense?.back === undefined ? null : { back: round(s.sense.back), flipped: s.sense.flipped, swapped: s.sense.swapped, referee: s.sense.referee, flicker: s.sense.flicker, away: s.sense.away,
            net: s.sense.network?.map(round) ?? null, forward: s.sense.forward?.map(round) ?? null,
            ...(skeleton ? { skeleton3d: s.sense.skeleton3d ? Array.from(s.sense.skeleton3d, v => round(v)) : null,
              sigma3d: s.sense.sigma3d ? Array.from(s.sense.sigma3d, v => round(v)) : null, mirror: s.sense.mirrorSkeleton ?? null } : {}) },
          mesh: kp ? { kp: Array.from({ length: 70 }, (_, i) => [round(kp[i * 3]), round(kp[i * 3 + 1]), round(kp[i * 3 + 2])]), t: g.camera.translation.map(round), focal: round(g.camera.focal) } : null });
      });
      await window.motionCapture.start(stream ? { stream, surface: true } : { videoUrl: `/__videos/${encodeURIComponent(name)}`, surface: true });
    }, { name, crop, skeleton: Boolean(process.env.EVAL_SKELETON) });
    const clipTime = () => page.evaluate(() => window.__evalClip ? window.__evalClip.currentTime : document.getElementById('camera').currentTime);
    const done = () => page.evaluate(() => (window.__evalClip ? window.__evalClip.ended : document.getElementById('camera').ended) && !window.motionCapture.getDiagnostics().inFlight);
    let snap = 0;
    while (!(await done())) {
      if (await clipTime() >= snap * snapEvery) { await page.screenshot({ path: path.join(outDir, `${name.replace(/\.\w+$/, '')}-${snap}.jpg`), quality: 60 }); snap++; }
      await new Promise(r => setTimeout(r, 100));
      if (Date.now() - began > 600000) break;
    }
    const frames = await page.evaluate(() => window.__evalFrames);
    await page.evaluate(() => window.motionCapture.stop());
    await writeFile(path.join(outDir, name.replace(/\.\w+$/, '') + '.json'), JSON.stringify({ name, crop, frames }));
    console.log(name, frames.length, 'frames', frames.filter(f => f.mesh).length, 'with mesh', `${((Date.now() - began) / 1000).toFixed(0)} s`);
  }
} finally {
  await browser?.close().catch(() => {});
  server?.kill('SIGTERM');
}
