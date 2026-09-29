import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createDenseSurface } from '../src/dense/controller.js';

// Surface controller with scripted workers: activation, scheduling, frame sync
// and ownership. Actual ORT/WebGL/browser integration is in tests/dense-browser.mjs.
class FakeWorker {
  constructor() { this.sent = []; this.terminated = 0; this.onmessage = null; this.onerror = null; }
  postMessage(message, transfer = []) { this.sent.push({ message, transfer }); }
  terminate() { this.terminated++; }
  emit(data) { this.onmessage?.({ data }); }
  of(type) { return this.sent.map(item => item.message).filter(message => message.type === type); }
}

function person(scale = 1) {
  const p = Array.from({ length: 33 }, () => ({ x: .5, y: .15, visibility: .99, presence: .99 }));
  const xy = { 11:[.4,.3],12:[.6,.3],13:[.3,.4],14:[.7,.4],15:[.2,.5],16:[.8,.5],23:[.43,.55],24:[.57,.55],
    25:[.43,.72],26:[.57,.72],27:[.43,.9],28:[.57,.9],29:[.42,.91],30:[.58,.91],31:[.4,.93],32:[.6,.93] };
  for (const [i, [x, y]] of Object.entries(xy)) Object.assign(p[i], { x: .5 + (x - .5) * scale, y: .5 + (y - .5) * scale });
  return p;
}
const flush = () => new Promise(resolve => setImmediate(resolve));

function harness({ overlayFails = false } = {}) {
  let clock = 0;
  const workers = { model: new FakeWorker(), fit: new FakeWorker() };
  const timers = new Map(); let timerId = 0;
  const bitmaps = [], overlays = [];
  const surface = createDenseSurface({
    workerFactory: () => workers,
    overlayFactory: () => {
      if (overlayFails) throw new Error('WebGL2 unavailable');
      const overlay = { draws: 0, disposed: 0, canvas: { width: 1, height: 1 }, draw(mesh, edges, camera, options) { this.draws++; this.lastOptions = options; return this.canvas; }, dispose() { this.disposed++; } };
      overlays.push(overlay); return overlay;
    },
    createBitmap: async (...args) => { const bitmap = { args, closed: 0, close() { this.closed++; } }; bitmaps.push(bitmap); return bitmap; },
    now: () => clock,
    setTimer: (fn, ms) => { timers.set(++timerId, { fn, at: clock + ms }); return timerId; },
    clearTimer: id => timers.delete(id),
  });
  const ready = async () => {
    workers.model.emit({ type: 'ready', provider: 'webgpu' });
    workers.fit.emit({ type: 'ready', faces: new Uint32Array([0, 1, 2, 1, 2, 3]), jointNames: [], keypointNames: [] });
    await surface.ready;
  };
  const frame = (t, pose = person()) => ({ timestamp: t, sourceWidth: 1000, sourceHeight: 1000, source: { width: 1000, height: 1000 }, pose: pose && { landmarks: pose } });
  const advance = ms => { clock += ms; for (const [id, timer] of [...timers]) if (timer.at <= clock) { timers.delete(id); timer.fn(); } };
  const mesh = timestamp => ({ timestamp, vertices: new Float32Array(12), keypoints70: new Float32Array(210), skeleton: new Float32Array(381),
    camera: { width: 1000, height: 1000, focal: 1414, translation: [0, 0, 3] }, fit: { normalizedError: 0.02 }, fitMs: 3 });
  return { surface, workers, bitmaps, overlays, ready, frame, advance, mesh, setClock: value => { clock = value; }, get clock() { return clock; } };
}

// Arms the surface: the first frame starts the count, 150 ms later it is on.
function activate(h, from = 0) {
  h.setClock(from); h.surface.update(h.frame(from));
  h.setClock(from + 160); return h.surface.update(h.frame(from + 160));
}

test('ready waits for both workers; nothing is inferred before a full body is armed', async () => {
  const h = harness();
  assert.equal(h.surface.getDiagnostics().state, 'loading');
  assert.deepEqual(h.surface.update(h.frame(0)), { wait: null }, 'frames pass through while loading');
  h.workers.model.emit({ type: 'ready', provider: 'webgpu' });
  assert.equal(h.surface.getDiagnostics().state, 'loading');
  await h.ready();
  assert.equal(h.surface.getDiagnostics().state, 'ready');
  assert.equal(h.surface.update(h.frame(10)).wait, null);
  assert.equal(h.workers.model.of('infer').length + h.workers.fit.of('pose').length, 0, 'arming sends nothing');
  activate(h, 20);
  await flush();
  assert.equal(h.workers.fit.of('reset').length, 1, 'a new person track starts');
  assert.equal(h.workers.model.of('infer').length, 1);
  assert.equal(h.workers.fit.of('pose').length, 1);
  const [x, y, w, height, options] = h.bitmaps[0].args.slice(1);
  assert.ok(w > 0 && height > 0 && Number.isInteger(x) && Number.isInteger(y));
  assert.deepEqual(options, { resizeWidth: 224, resizeHeight: 224, resizeQuality: 'low' });
  assert.equal(h.workers.model.sent.at(-1).transfer[0], h.bitmaps[0], 'the crop bitmap is transferred, not copied');
});

test('unsupported devices and broken WebGL fail softly and release both workers', async () => {
  const h = harness();
  h.workers.model.emit({ type: 'unsupported', reason: 'no-webgpu' });
  await assert.rejects(h.surface.ready, /WebGPU/);
  assert.equal(h.surface.getDiagnostics().state, 'unsupported');
  assert.equal(h.workers.model.terminated + h.workers.fit.terminated, 2);
  assert.deepEqual(h.surface.update(h.frame(0)), { wait: null });
  assert.equal(h.surface.render(100, 100), null);
  const broken = harness({ overlayFails: true });
  await assert.rejects(broken.ready(), /WebGL2 unavailable/);
  assert.equal(broken.workers.model.terminated + broken.workers.fit.terminated, 2);
});

test('only the newest pose waits while a fit is running, and one network run is in flight', async () => {
  const h = await (async () => { const h = harness(); await h.ready(); return h; })();
  activate(h); await flush();
  for (const t of [200, 233, 266]) { h.setClock(t); h.surface.update(h.frame(t)); }
  await flush();
  assert.equal(h.workers.fit.of('pose').length, 1, 'later poses are coalesced');
  assert.equal(h.workers.model.of('infer').length, 1, 'no second network run while one is in flight');
  h.workers.fit.emit({ type: 'mesh', generation: 1, timestamp: 160, mesh: h.mesh(160) });
  const poses = h.workers.fit.of('pose');
  assert.equal(poses.length, 2); assert.equal(poses[1].timestamp, 266, 'the latest pending pose is sent next');
  const infer = h.workers.model.of('infer')[0];
  h.workers.model.emit({ type: 'result', id: infer.id, mhrParams: new Float32Array(204), shapeParams: new Float32Array(45), camTrans: new Float32Array([0, 0, 3]), focal: 1414, inferenceMs: 100 });
  const prior = h.workers.fit.of('prior');
  assert.equal(prior.length, 1); assert.equal(prior[0].generation, 1); assert.equal(prior[0].timestamp, 160);
  // A 100 ms network run keeps the GPU at most half busy: next run 200 ms later.
  h.setClock(280); h.surface.update(h.frame(280)); await flush();
  assert.equal(h.workers.model.of('infer').length, 1, 'the next run waits for the duty-cycle interval');
  h.setClock(400); h.surface.update(h.frame(400)); await flush();
  assert.equal(h.workers.model.of('infer').length, 2);
});

test('a frame is released by its own fit or, at the latest, by the short timeout', async () => {
  const h = harness(); await h.ready();
  activate(h); await flush();
  h.workers.fit.emit({ type: 'mesh', generation: 1, timestamp: 160, mesh: h.mesh(160) });
  h.setClock(193);
  const { wait } = h.surface.update(h.frame(193));
  assert.ok(wait, 'with a surface on screen the frame waits for its own fit');
  let released = false; wait.then(() => { released = true; });
  h.workers.fit.emit({ type: 'mesh', generation: 1, timestamp: 193, mesh: h.mesh(193) });
  await flush();
  assert.equal(released, true);
  assert.equal(h.surface.geometry().timestamp, 193, 'geometry belongs to the released frame');
  const late = h.surface.update(h.frame(226)).wait;
  let timedOut = false; late.then(() => { timedOut = true; });
  h.advance(10); await flush(); assert.equal(timedOut, false);
  h.advance(20); await flush(); assert.equal(timedOut, true);
  assert.equal(h.surface.getDiagnostics().lateFrames, 1);
});

test('leaving the frame switches the surface off and ignores fits from the old track', async () => {
  const h = harness(); await h.ready();
  activate(h); await flush();
  h.workers.fit.emit({ type: 'mesh', generation: 1, timestamp: 160, mesh: h.mesh(160) });
  assert.ok(h.surface.geometry());
  h.surface.render(640, 360, h.clock + 100);
  assert.ok(h.surface.render(640, 360, h.clock + 300), 'fades in');
  for (let t = 200; t <= 700; t += 33) { h.setClock(t); h.surface.update(h.frame(t, null)); }
  assert.equal(h.surface.getDiagnostics().active, false);
  assert.equal(h.surface.geometry(), null);
  assert.equal(h.workers.fit.of('reset').at(-1).generation, 2);
  h.workers.fit.emit({ type: 'mesh', generation: 1, timestamp: 690, mesh: h.mesh(690) });
  assert.equal(h.surface.geometry(), null, 'a late fit of the previous track is ignored');
  assert.equal(h.surface.render(640, 360, h.clock + 1000), null, 'faded out');
});

test('an upper body with its face switches the surface on, clipped below the visible joints', async () => {
  const h = harness(); await h.ready();
  const seated = person();
  for (const i of [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]) seated[i].visibility = 0.1;
  const face = { landmarks: Array.from({ length: 478 }, (_, i) => ({ x: .5 + .03 * Math.cos(i), y: .15 + .03 * Math.sin(i) })) };
  const frame = t => ({ ...h.frame(t, seated), face });
  h.setClock(0); h.surface.update(frame(0));
  h.setClock(160); h.surface.update(frame(160)); await flush();
  assert.equal(h.surface.getDiagnostics().active, true);
  assert.equal(h.surface.getDiagnostics().body, 'upper');
  h.workers.fit.emit({ type: 'mesh', generation: 1, timestamp: 160, mesh: h.mesh(160) });
  h.surface.render(640, 360, 200); h.surface.render(640, 360, 400);
  // Wrists are the lowest visible joints (y = 500 px); the surface stops below them.
  assert.ok(Math.abs(h.overlays[0].lastOptions.clipY - 539) < 1, String(h.overlays[0].lastOptions.clipY));
  // Squatting behind the desk: elbows vanish too. The surface stays on for the
  // grace period, but is clipped under the shoulders instead of showing legs.
  const lower = seated.map((p, i) => [13, 14, 15, 16].includes(i) ? { ...p, visibility: 0.1 } : p);
  h.setClock(200); h.surface.update({ ...h.frame(200, lower), face });
  h.surface.render(640, 360, 2000);
  assert.equal(h.surface.getDiagnostics().active, true);
  assert.ok(h.overlays[0].lastOptions.clipY < 400, String(h.overlays[0].lastOptions.clipY));
  // A hand over both shoulders without a face is the typical false body in a hand.
  const hands = [{ landmarks: Array.from({ length: 21 }, (_, i) => ({ x: .37 + .26 * (i % 7) / 6, y: .2 + .2 * Math.floor(i / 7) / 2 })) }];
  const suspicious = harness(); await suspicious.ready();
  for (const t of [0, 160, 320]) { suspicious.setClock(t); suspicious.surface.update({ ...suspicious.frame(t, seated), hands }); }
  assert.equal(suspicious.surface.getDiagnostics().active, false, 'a hand over the shoulders needs a matching face');
});

test('a crop that resolves after the track ended is closed instead of sent', async () => {
  const h = harness(); await h.ready();
  h.setClock(0); h.surface.update(h.frame(0));
  h.setClock(160); h.surface.update(h.frame(160)); // activation requests a crop
  h.surface.reset(); // a new session before createImageBitmap resolved
  await flush();
  assert.equal(h.bitmaps[0].closed, 1);
  assert.equal(h.workers.model.of('infer').length, 0);
});

test('dispose releases workers, overlay and waiting frames, and is idempotent', async () => {
  const loading = harness();
  await loading.surface.dispose();
  await assert.rejects(loading.surface.ready, /disposed/);
  const h = harness(); await h.ready();
  activate(h); await flush();
  h.workers.fit.emit({ type: 'mesh', generation: 1, timestamp: 160, mesh: h.mesh(160) });
  h.setClock(193);
  const { wait } = h.surface.update(h.frame(193));
  await h.surface.dispose(); await h.surface.dispose();
  await wait;
  assert.equal(h.workers.model.terminated, 1); assert.equal(h.workers.fit.terminated, 1);
  assert.equal(h.overlays[0].disposed, 1);
  assert.equal(h.surface.getDiagnostics().enabled, false);
});

// ---- app integration (module body with explicit stubs) ----
const read = path => readFileSync(new URL(path, import.meta.url), 'utf8');
const stripImports = source => source.replace(/^import .*;\n/gm, '');
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };

function appHarness({ search = '?test=1', saveData = false } = {}) {
  let source = stripImports(read('../src/capture/app.js'));
  const lazyImport = "import('../dense/controller.js')";
  assert.equal(source.split(lazyImport).length, 2, 'update the explicit lazy import stub if the module contract changes');
  source = source.replace(lazyImport, 'denseModule()');
  const events = [], draws = [], counts = { moduleRequests: 0, created: 0 };
  class Session {
    constructor(options) { this.options = options; this.generation = 0; this.pending = null; this.source = { kind: 'camera' }; }
    stop() { this.generation++; this.pending?.resolve(null); this.pending = null; }
    start() { this.pending = deferred(); return this.pending.promise; }
    getDiagnostics() { return {}; }
  }
  class Renderer { setSize() {} draw(frame) { draws.push(frame); } clear() {} }
  class Stabilizer { update(frame) { return { ...frame }; } reset() {} }
  const elements = new Map();
  const document = { hidden: false, getElementById(id) { if (!elements.has(id)) elements.set(id, { addEventListener() {} }); return elements.get(id); }, addEventListener() {} };
  const window = { addEventListener() {}, dispatchEvent(event) { events.push(event.detail); } };
  const surfaces = [];
  const denseModule = () => { counts.moduleRequests++; return Promise.resolve({ createDenseSurface(options) {
    counts.created++;
    const surface = { options, updates: [], waits: [], disposed: 0, resets: 0, ready: Promise.resolve(),
      update(frame) { this.updates.push(frame); const next = this.waits.shift(); return { wait: next ?? null }; },
      geometry: () => ({ mesh: true }), getDiagnostics: () => ({ state: 'ready' }), render: () => null,
      reset() { this.resets++; }, async dispose() { this.disposed++; } };
    surfaces.push(surface); return surface;
  } }); };
  const load = new Function('CaptureSession', 'reconcileDetections', 'CaptureStabilizer', 'CaptureRenderer', 'containViewport', 'document', 'window', 'location', 'navigator', 'requestAnimationFrame', 'denseModule', 'performance', 'innerWidth', 'innerHeight', 'devicePixelRatio', 'CustomEvent',
    source + '\nreturn { session };');
  const hooks = load(Session, frame => frame, Stabilizer, Renderer, () => ({ x: 0, y: 0, width: 1280, height: 720 }), document, window, { search }, { connection: { saveData } }, () => {},
    denseModule, { now: () => 0 }, 1280, 720, 1, class { constructor(type, { detail }) { this.type = type; this.detail = detail; } });
  return { ...hooks, api: window.motionCapture, events, draws, counts, surfaces, run: value => hooks.session.pending.resolve(value) };
}

test('capture starts without waiting for the surface model, which loads only after capture runs', async () => {
  const h = appHarness();
  const pending = h.api.start({ stream: {} });
  await flush();
  assert.equal(h.counts.moduleRequests, 0, 'the heavy model never delays the camera and landmarks');
  h.run({ state: 'running' });
  assert.deepEqual(await pending, { state: 'running' });
  await flush();
  assert.equal(h.counts.created, 1);
  assert.equal(h.surfaces[0].options.mode, 'auto');
});

test('stop before capture runs never loads the surface, and surface:false opts out', async () => {
  const h = appHarness();
  const pending = h.api.start({ stream: {} });
  h.api.stop();
  assert.equal(await pending, null);
  await flush();
  assert.equal(h.counts.moduleRequests, 0);
  const off = appHarness();
  const started = off.api.start({ stream: {}, surface: false });
  off.run({ state: 'running' }); await started; await flush();
  assert.equal(off.counts.moduleRequests, 0);
  const forced = appHarness({ search: '?test=1&surface=1', saveData: true });
  const run = forced.api.start({ stream: {} });
  forced.run({ state: 'running' }); await run; await flush();
  assert.equal(forced.surfaces[0].options.mode, 'force', 'an explicit request overrides Data Saver');
  const saver = appHarness({ saveData: true });
  const saving = saver.api.start({ stream: {} });
  saver.run({ state: 'running' }); await saving; await flush();
  assert.equal(saver.counts.moduleRequests, 0, 'Data Saver skips the surface model by default');
});

test('a frame waits for its own surface, and a newer frame flushes a held one first', async () => {
  const h = appHarness();
  const run = h.api.start({ stream: {} });
  h.run({ state: 'running' }); await run; await flush();
  const surface = h.surfaces[0], first = deferred();
  surface.waits.push(first.promise);
  h.session.options.onFrame({ frameId: 1, timestamp: 10, receivedAt: 10, sourceWidth: 1280, sourceHeight: 720 });
  assert.equal(h.events.length, 0, 'held until its surface arrives');
  first.resolve(); await flush();
  assert.deepEqual(h.events.map(frame => frame.frameId), [1]);
  assert.deepEqual(h.events[0].bodySurface, { mesh: true });
  const slow = deferred();
  surface.waits.push(slow.promise);
  h.session.options.onFrame({ frameId: 2, timestamp: 43, receivedAt: 43, sourceWidth: 1280, sourceHeight: 720 });
  h.session.options.onFrame({ frameId: 3, timestamp: 76, receivedAt: 76, sourceWidth: 1280, sourceHeight: 720 });
  assert.deepEqual(h.events.map(frame => frame.frameId), [1, 2, 3], 'frame 2 is shown before frame 3, while its image is alive');
  slow.resolve(); await flush();
  assert.equal(h.events.length, 3, 'a flushed frame is not published twice');
});
