/** Real production app, full video playback, all regional models and the body
 * surface. Opens no webcam; always owns and stops its temporary server. */
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const built = process.env.DENSE_TEST_BUILT === '1';
const clips = (process.env.DENSE_TEST_CLIPS ?? 'body-motion.mp4,body-squats.mp4,hand-signs.mp4').split(',');
assert.ok(clips.length && clips.every(name=>['body-motion.mp4','body-squats.mp4','hand-signs.mp4'].includes(name)));
const artifacts = path.join(root, built ? 'tests/artifacts/dense-built' : 'tests/artifacts/dense');
const fixtures = process.env.CAPTURE_FIXTURES_DIR ?? path.join(root, 'tests/fixtures');
const report = { startedAt: new Date().toISOString(), clips: [], pageErrors: [], cleanup: {}, cameraRequests: 0,
  scope: 'Whole public-video playback through production Pose, Face, Hands, fusion and the asynchronous MHR surface; no 3D ground truth.' };
let server, browser, page;
let deadline;
const surfaceWorkers = async cdp => (await cdp.send('Target.getTargets')).targetInfos.filter(t => t.type === 'worker' && /model-worker|fit-worker/.test(t.url));
try {
  await mkdir(artifacts, { recursive: true });
  server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', ...(built ? ['preview'] : []), '--host', '127.0.0.1', '--port', '5175', '--strictPort'], { cwd: root, stdio: ['ignore','pipe','pipe'] });
  await new Promise((resolve,reject) => {
    const timeout = setTimeout(() => reject(new Error('Test server did not start')), 10000);
    server.stdout.on('data', data => { if (String(data).includes('5175')) { clearTimeout(timeout); resolve(); } });
    server.stderr.on('data', data => process.stderr.write(data));
    server.once('error', reject);
  });
  browser = await chromium.launch({ executablePath: process.env.CAPTURE_CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true });
  deadline = setTimeout(() => { browser?.close().catch(()=>{}); server?.kill('SIGTERM'); }, 300000);
  page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.setDefaultTimeout(120000);
  page.on('pageerror', error => report.pageErrors.push(error.message));
  await page.addInitScript(() => {
    window.__cameraRequests = 0;
    navigator.mediaDevices.getUserMedia = () => { window.__cameraRequests++; throw new Error('Camera forbidden in video QA'); };
  });
  await page.route('**/__fixtures/*', async route => {
    const name = path.basename(new URL(route.request().url()).pathname);
    const bytes = await readFile(path.join(fixtures,name));
    const match = route.request().headers().range?.match(/^bytes=(\d+)-(\d*)$/);
    if (match) {
      const start = Number(match[1]), end = match[2] ? Math.min(Number(match[2]),bytes.length-1) : bytes.length-1;
      await route.fulfill({ status:206,contentType:'video/mp4',body:bytes.subarray(start,end+1),headers:{'Accept-Ranges':'bytes','Content-Range':`bytes ${start}-${end}/${bytes.length}`} });
    } else await route.fulfill({ contentType:'video/mp4',body:bytes });
  });
  const cdp = await browser.newBrowserCDPSession();
  await cdp.send('Target.setDiscoverTargets',{discover:true});
  await page.goto('http://127.0.0.1:5175/?test=1');
  await page.waitForFunction(() => window.motionCapture);
  assert.equal((await page.evaluate(()=>window.motionCapture.getDiagnostics())).state,'idle');

  // The surface loads in the background after capture runs and then persists
  // across stop/start. Load it once on a blank source so clips start with it.
  const load = await page.evaluate(async () => {
    const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 480;
    const context = canvas.getContext('2d'), timer = setInterval(() => { context.fillStyle = '#222'; context.fillRect(0, 0, 640, 480); }, 66);
    const began = performance.now();
    await window.motionCapture.start({ stream: canvas.captureStream(15), surface: true });
    const capturing = performance.now() - began;
    while (!['ready', 'error', 'unsupported'].includes(window.motionCapture.getDiagnostics().surface.state)) await new Promise(r => setTimeout(r, 100));
    clearInterval(timer);
    return { capturingMs: capturing, surfaceReadyMs: performance.now() - began, surface: window.motionCapture.getDiagnostics().surface };
  });
  report.surfaceLoad = load;
  assert.equal(load.surface.state, 'ready', `surface failed to load: ${load.surface.error}`);
  assert.ok(load.capturingMs < load.surfaceReadyMs, 'capture ran before the surface model finished loading');
  console.log(`PASS surface loaded in background (${Math.round(load.surfaceReadyMs)} ms, ${load.surface.provider}); capture started after ${Math.round(load.capturingMs)} ms`);

  for (const filename of clips) {
    const clip = { filename }; report.clips.push(clip);
    await page.evaluate(async filename => {
      window.__denseUnsubscribe?.(); window.__denseFrames = [];
      window.__denseUnsubscribe = window.motionCapture.subscribe(frame => {
        window.__denseFrames.push({ frameId:frame.frameId, timestamp: frame.timestamp, mediaTime:frame.mediaTime,
          sourceWidth:frame.source?.width, sourceHeight:frame.source?.height,
          active: frame.surface?.active ?? false, meshTimestamp: frame.bodySurface?.timestamp ?? null,
          vertices: frame.bodySurface?.vertices?.length ?? 0, faces: frame.bodySurface?.faces?.length ?? 0,
          fitError: frame.bodySurface ? frame.surface?.fit?.normalizedError ?? null : null,
          pose:frame.pose?.landmarks?.length ?? 0, hands:(frame.hands ?? []).map(h=>h.landmarks.length),
          captureToPublishMs:performance.now()-frame.timestamp });
      });
      await window.motionCapture.start({videoUrl:`/__fixtures/${filename}`,surface:true});
    },filename);
    await page.waitForFunction(()=>document.getElementById('camera').ended && !window.motionCapture.getDiagnostics().inFlight,null,{timeout:90000});
    clip.frames = await page.evaluate(()=>window.__denseFrames);
    clip.diagnostics = await page.evaluate(()=>window.motionCapture.getDiagnostics());
    assert.ok(clip.frames.length>3,'Expected continued whole-video capture');
    for (let i=0;i<clip.frames.length;i++) {
      const frame = clip.frames[i];
      assert.ok(frame.sourceWidth>0 && frame.sourceHeight>0,'Source bitmap remained alive until publish');
      if (i) { assert.ok(frame.frameId>clip.frames[i-1].frameId); assert.ok(frame.mediaTime>clip.frames[i-1].mediaTime); }
      if (frame.vertices) { assert.equal(frame.vertices, 4899 * 3); assert.ok(frame.faces > 0); assert.ok(frame.meshTimestamp <= frame.timestamp); }
    }
    clip.surfaceFrames = clip.frames.filter(frame=>frame.vertices).length;
    clip.ownFrameSurfaces = clip.frames.filter(frame=>frame.vertices && frame.meshTimestamp===frame.timestamp).length;
    clip.activeFrames = clip.frames.filter(frame=>frame.active).length;
    if (filename.startsWith('body-')) {
      assert.ok(clip.surfaceFrames >= clip.frames.length * 0.6, `Dense surface on only ${clip.surfaceFrames}/${clip.frames.length} frames`);
      assert.ok(clip.ownFrameSurfaces >= clip.surfaceFrames * 0.8, 'Most frames should be shown with their own fitted surface');
    } else assert.equal(clip.activeFrames,0,'An isolated hand must not become a dense body');
    await page.screenshot({path:path.join(artifacts,`${filename}.png`)});
    await page.evaluate(()=>window.motionCapture.stop());
    await page.waitForFunction(()=>window.motionCapture.getDiagnostics().state==='idle' && !window.motionCapture.getDiagnostics().surface.active);
    let workers;
    for(let attempt=0;attempt<30;attempt++) {
      workers=(await cdp.send('Target.getTargets')).targetInfos.filter(t=>t.type==='worker' && t.url.includes('pose-worker'));
      if(!workers.length)break;
      await new Promise(resolve=>setTimeout(resolve,100));
    }
    assert.equal(workers.length,0,'Regional workers survived stop');
    clip.passed=true;
    console.log(`PASS ${filename}: ${clip.frames.length} frames, ${clip.surfaceFrames} with surface (${clip.ownFrameSurfaces} fitted to their own frame)`);
  }
  // Standing close to the camera: the real squat clip enlarged so the legs are
  // cut by the bottom edge of a 1280x720 stream. No legs, still a surface.
  const close = await page.evaluate(async () => {
    window.__denseUnsubscribe?.();
    const video = document.createElement('video'); video.src = '/__fixtures/body-squats.mp4'; video.muted = true; video.loop = true;
    await new Promise(resolve => { video.onloadeddata = resolve; });
    const canvas = document.createElement('canvas'); canvas.width = 1280; canvas.height = 720;
    const context = canvas.getContext('2d');
    let drawing = true;
    const draw = () => {
      if (!drawing) return;
      context.fillStyle = '#6b6f73'; context.fillRect(0, 0, 1280, 720);
      const scale = 2 * 720 / video.videoHeight, width = video.videoWidth * scale, height = video.videoHeight * scale;
      context.drawImage(video, 640 - width / 2, 0.78 * 720 - height / 2, width, height);
      requestAnimationFrame(draw);
    };
    await video.play(); draw();
    const frames = [];
    const unsubscribe = window.motionCapture.subscribe(frame => frames.push({ body: frame.surface?.body ?? null, surface: Boolean(frame.bodySurface) }));
    await window.motionCapture.start({ stream: canvas.captureStream(30), surface: true });
    await new Promise(resolve => setTimeout(resolve, 6000));
    unsubscribe(); drawing = false; video.pause(); window.motionCapture.stop();
    return { frames: frames.length, upper: frames.filter(f => f.body === 'upper').length, surfaceFrames: frames.filter(f => f.surface).length };
  });
  report.upperBody = close;
  assert.ok(close.frames > 60, 'the close-up stream kept capturing');
  assert.ok(close.upper >= close.frames * 0.8, `upper body recognised on only ${close.upper}/${close.frames} frames`);
  assert.ok(close.surfaceFrames >= close.frames * 0.7, `surface on only ${close.surfaceFrames}/${close.frames} frames`);
  console.log(`PASS upper body with legs out of frame: ${close.surfaceFrames}/${close.frames} frames with surface`);
  // Laptop close-up: only head and shoulders (OpenFace clip, needs the ffmpeg-converted fixture).
  const faceClip = await readFile(path.join(fixtures, 'face-turns.mp4')).then(() => true, () => false);
  if (faceClip) {
    const closeUp = await page.evaluate(async () => {
      const video = document.createElement('video'); video.src = '/__fixtures/face-turns.mp4'; video.muted = true; video.loop = true;
      await new Promise(resolve => { video.onloadeddata = resolve; });
      const canvas = document.createElement('canvas'); canvas.width = 1280; canvas.height = 720;
      const context = canvas.getContext('2d');
      let drawing = true;
      const draw = () => {
        if (!drawing) return;
        const scale = 720 / video.videoHeight, width = video.videoWidth * scale;
        context.fillStyle = '#6b6f73'; context.fillRect(0, 0, 1280, 720);
        context.drawImage(video, 640 - width / 2, 0, width, 720);
        requestAnimationFrame(draw);
      };
      await video.play(); draw();
      const frames = [];
      const unsubscribe = window.motionCapture.subscribe(frame => frames.push({ reason: frame.surface?.reason ?? null, surface: Boolean(frame.bodySurface) }));
      await window.motionCapture.start({ stream: canvas.captureStream(30), surface: true });
      await new Promise(resolve => setTimeout(resolve, 6000));
      unsubscribe(); drawing = false; video.pause(); window.motionCapture.stop();
      return { frames: frames.length, closeUp: frames.filter(f => f.reason === 'close-up').length, surfaceFrames: frames.filter(f => f.surface).length };
    });
    report.closeUp = closeUp;
    assert.ok(closeUp.closeUp >= closeUp.frames * 0.8, `close-up recognised on only ${closeUp.closeUp}/${closeUp.frames} frames`);
    assert.ok(closeUp.surfaceFrames >= closeUp.frames * 0.7, `surface on only ${closeUp.surfaceFrames}/${closeUp.frames} frames`);
    console.log(`PASS head-and-shoulders close-up: ${closeUp.surfaceFrames}/${closeUp.frames} frames with surface`);
  } else console.log('SKIP head-and-shoulders close-up: tests/fixtures/face-turns.mp4 missing (npm run fixtures -- --ffmpeg …)');
  assert.equal((await surfaceWorkers(cdp)).length, 2, 'the loaded surface is kept across stop/start');
  await page.evaluate(() => dispatchEvent(new Event('pagehide')));
  let left;
  for (let attempt = 0; attempt < 30; attempt++) { left = await surfaceWorkers(cdp); if (!left.length) break; await new Promise(r => setTimeout(r, 100)); }
  assert.equal(left.length, 0, 'pagehide releases the surface workers');
  report.releasedOnPagehide = true;
  console.log('PASS pagehide releases the surface model and fit workers');
  report.cameraRequests=await page.evaluate(()=>window.__cameraRequests);
  assert.equal(report.cameraRequests,0); assert.deepEqual(report.pageErrors,[]);
  report.passed=true;
} catch(error) { report.error=error.stack;report.passed=false;process.exitCode=1;console.error(error); }
finally {
  clearTimeout(deadline);
  if(page && !page.isClosed())try { await page.evaluate(()=>window.motionCapture?.stop());report.cleanup.stopped=true; }catch{}
  if(browser) { await browser.close();report.cleanup.browserClosed=true; }
  if(server && server.exitCode===null && server.signalCode===null) {
    const exited=once(server,'exit');server.kill('SIGTERM');
    const force=setTimeout(()=>server.kill('SIGKILL'),3000);await exited;clearTimeout(force);
  }
  report.cleanup.serverStopped=Boolean(server && (server.exitCode!==null || server.signalCode!==null));
  report.finishedAt=new Date().toISOString();
  await mkdir(artifacts,{recursive:true});await writeFile(path.join(artifacts,'report.json'),JSON.stringify(report,null,2));
}
