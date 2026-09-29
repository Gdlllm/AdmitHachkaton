import test from 'node:test';
import assert from 'node:assert/strict';
import { CaptureSession } from '../src/capture/session.js';

const turn = () => new Promise(resolve => setImmediate(resolve));
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function videoStub() {
  const callbacks = new Map();
  let callbackId = 0;
  return {
    readyState: 4, videoWidth: 1280, videoHeight: 720, currentTime: 1,
    pause() {}, load() {}, removeAttribute() {}, play: async () => {},
    requestVideoFrameCallback(callback) { const id = ++callbackId; callbacks.set(id, callback); return id; },
    cancelVideoFrameCallback(id) { callbacks.delete(id); },
    get pendingFrameCallbacks() { return [...callbacks.values()]; },
    emitFrame(mediaTime, presentedFrames) {
      this.currentTime = mediaTime;
      const waiting = [...callbacks.values()];
      callbacks.clear();
      for (const callback of waiting) callback(performance.now(), { mediaTime, presentedFrames });
    },
  };
}
function streamStub() {
  const track = { stopped: 0, stop() { this.stopped++; }, addEventListener() {} };
  return { track, getTracks: () => [track], getVideoTracks: () => [track] };
}

test('stop settles start even while the camera permission request remains unanswered', async () => {
  const permission = deferred();
  const stream = streamStub();
  let workers = 0;
  const session = new CaptureSession({ video: videoStub(), mediaDevices: { getUserMedia: () => permission.promise },
    workerFactory: () => { workers++; throw new Error('A cancelled session must not create a worker.'); } });
  const started = session.start();
  session.stop();
  const stateAfterStop = await Promise.race([
    started.then(value => ({ settled: true, value })),
    delay(50).then(() => ({ settled: false })),
  ]);
  // Always release the simulated late stream and remove timers, even on failure.
  permission.resolve(stream);
  await started;
  await turn();
  session.stop();
  assert.equal(stream.track.stopped, 1, 'a permission result arriving after stop must be closed');
  assert.equal(workers, 0);
  assert.deepEqual(stateAfterStop, { settled: true, value: null });
});

test('an old GPU warmup frame is closed and a fresh camera frame completes startup', async () => {
  const previousDocument = globalThis.document;
  globalThis.document = { hidden: false };
  const video = videoStub(), stream = streamStub(), received = [];
  let now = 100;
  const worker = { messages: [], postMessage(data) { this.messages.push(data); }, terminate() {} };
  const session = new CaptureSession({ video, onFrame: frame => received.push(frame),
    mediaDevices: {}, workerFactory: () => worker, now: () => now,
    createBitmap: async () => ({ width: 1280, height: 720, closed: 0, close() { this.closed++; } }),
  });
  try {
    const started = session.start({ stream });
    await turn();
    worker.onmessage({ data: { type: 'ready', workerCount: 4 } });
    await turn();
    const old = worker.messages.find(message => message.type === 'frame');
    now = 1500;
    video.emitFrame(2, 2);
    worker.onmessage({ data: { type: 'result', frameId: old.frameId, timestamp: old.timestamp,
      source: old.bitmap, pose: { landmarks: [] }, face: { landmarks: [] }, hands: [] } });
    assert.equal(old.bitmap.closed, 1);
    assert.equal(received.length, 0);
    await turn();
    const next = worker.messages.filter(message => message.type === 'frame').at(-1);
    assert.notEqual(next.frameId, old.frameId);
    now = 1530;
    worker.onmessage({ data: { type: 'result', frameId: next.frameId, timestamp: next.timestamp,
      source: next.bitmap, pose: { landmarks: [] }, face: { landmarks: [] }, hands: [], inferenceMs: 30 } });
    const diagnostics = await started;
    assert.equal(diagnostics.state, 'running');
    assert.equal(diagnostics.frameCount, 1);
    assert.equal(received.length, 1);
    assert.equal(received[0].source, next.bitmap);
  } finally {
    session.stop();
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  }
});

test('video clock advancement cannot duplicate inference; a decoded frame can', async () => {
  const previousDocument = globalThis.document;
  globalThis.document = { hidden: false };
  const video = videoStub(), stream = streamStub(), received = [];
  let now = 100;
  const worker = { messages: [], postMessage(data) { this.messages.push(data); }, terminate() {} };
  const session = new CaptureSession({ video, onFrame: frame => received.push(frame),
    mediaDevices: {}, workerFactory: () => worker, now: () => now,
    createBitmap: async () => ({ width: 1280, height: 720, closed: 0, close() { this.closed++; } }),
  });
  const sentFrames = () => worker.messages.filter(message => message.type === 'frame');
  const finish = request => worker.onmessage({ data: { type: 'result', frameId: request.frameId,
    timestamp: request.timestamp, mediaTime: request.mediaTime, source: request.bitmap,
    pose: { landmarks: [] }, face: { landmarks: [] }, hands: [], inferenceMs: 5 } });
  try {
    const started = session.start({ stream });
    await turn();
    worker.onmessage({ data: { type: 'ready', workerCount: 4 } });
    await turn();
    assert.equal(sentFrames().length, 1);

    // currentTime is a playback clock: it advances inside a 40ms decoded frame.
    video.currentTime += .008;
    now += 5;
    finish(sentFrames()[0]);
    await started;
    await turn();
    assert.equal(sentFrames().length, 1, 'completing inference must not reprocess the same decoded image');

    video.currentTime += .008;
    session.resume();
    await turn();
    assert.equal(sentFrames().length, 1, 'a changing playback clock alone is not a new camera image');

    video.emitFrame(1.04, 2);
    await turn();
    assert.equal(sentFrames().length, 2, 'a decoded-frame callback starts fresh inference');

    // Two newer frames arrive while inference is occupied. Keep only the latest.
    video.emitFrame(1.08, 3);
    video.emitFrame(1.12, 4);
    await turn();
    assert.equal(sentFrames().length, 2, 'only one frame can be in flight');
    now += 5;
    finish(sentFrames()[1]);
    await turn();
    assert.equal(sentFrames().length, 3);
    assert.equal(sentFrames()[2].mediaTime, 1.12, 'discard intermediate decoded frames instead of building a queue');
    now += 5;
    finish(sentFrames()[2]);
    await turn();
    assert.equal(sentFrames().length, 3);
    assert.equal(received.length, 3);
    assert.ok(received[1].sourceFrameId > received[0].sourceFrameId);
    assert.ok(received[2].sourceFrameId > received[1].sourceFrameId);
  } finally {
    session.stop();
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  }
});

test('a late callback from a cancelled generation cannot erase the current frame pump', async () => {
  const previousDocument = globalThis.document;
  globalThis.document = { hidden: false };
  const video = videoStub(), workers = [];
  const session = new CaptureSession({ video, mediaDevices: {},
    workerFactory: () => {
      const worker = { postMessage() {}, terminate() {} };
      workers.push(worker);
      return worker;
    },
    createBitmap: async () => ({ width: 1280, height: 720, close() {} }),
  });
  try {
    const firstStart = session.start({ stream: streamStub() });
    await turn();
    workers[0].onmessage({ data: { type: 'ready' } });
    await turn();
    const oldCallback = video.pendingFrameCallbacks[0];
    assert.equal(typeof oldCallback, 'function');
    session.stop();
    await firstStart;

    const nextStart = session.start({ stream: streamStub() });
    await turn();
    workers[1].onmessage({ data: { type: 'ready' } });
    await turn();
    const activeHandle = session.callback;
    assert.notEqual(activeHandle, null);
    // Simulate a browser callback already dispatched before cancellation won.
    oldCallback(performance.now(), { mediaTime: 1.04, presentedFrames: 2 });
    const retainedHandle = session.callback;
    session.stop();
    await nextStart;
    assert.equal(retainedHandle, activeHandle);
    assert.equal(video.pendingFrameCallbacks.length, 0, 'stop must cancel the current-generation pump');
  } finally {
    session.stop();
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  }
});

function asynchronousSession(onFrame) {
  const previousDocument = globalThis.document;
  globalThis.document = { hidden: false };
  const video = videoStub(), workers = [];
  const session = new CaptureSession({ video, onFrame, mediaDevices: {}, now: () => 100,
    workerFactory: () => {
      const worker = { messages: [], terminated: 0,
        postMessage(message) { this.messages.push(message); }, terminate() { this.terminated++; } };
      workers.push(worker); return worker;
    },
    createBitmap: async () => ({ width: 1280, height: 720, closed: 0, close() { this.closed++; } }),
  });
  return { session, video, workers,
    frames: worker => worker.messages.filter(message => message.type === 'frame'),
    finish: (worker, request) => worker.onmessage({ data: { type: 'result', frameId: request.frameId,
      timestamp: request.timestamp, mediaTime: request.mediaTime, source: request.bitmap,
      pose: { landmarks: [] }, face: { landmarks: [] }, hands: [], inferenceMs: 5 } }),
    cleanup() {
      session.stop();
      if (previousDocument === undefined) delete globalThis.document;
      else globalThis.document = previousDocument;
    },
  };
}

test('awaited frame processing retains both source images and permits only the latest decoded frame after settlement', async () => {
  const firstProcessing = deferred(), secondProcessing = deferred();
  const processing = [firstProcessing, secondProcessing];
  let calls = 0, startSettled = false;
  const h = asynchronousSession(() => processing[calls++]?.promise);
  try {
    const started = h.session.start({ stream: streamStub() });
    started.then(() => { startSettled = true; });
    await turn();
    const worker = h.workers[0];
    worker.onmessage({ data: { type: 'ready' } }); await turn();
    const first = h.frames(worker)[0];
    h.finish(worker, first);
    h.video.emitFrame(1.04, 2); h.video.emitFrame(1.08, 3); h.video.emitFrame(1.12, 4);
    await turn();
    assert.equal(h.session.inFlight, true);
    assert.equal(startSettled, false, 'startup includes the asynchronous surface stage');
    assert.equal(h.frames(worker).length, 1, 'no inference queue while the extension owns this frame');
    assert.equal(first.bitmap.closed, 0);

    firstProcessing.resolve();
    await started; await turn();
    assert.equal(h.frames(worker).length, 2);
    const second = h.frames(worker)[1];
    assert.equal(second.mediaTime, 1.12, 'intermediate frames are skipped');
    h.finish(worker, second); await turn();
    assert.equal(first.bitmap.closed, 0, 'previous displayed image survives until its replacement is processed');
    assert.equal(second.bitmap.closed, 0, 'processing image stays usable');
    secondProcessing.resolve(); await turn();
    assert.equal(first.bitmap.closed, 1);
    assert.equal(second.bitmap.closed, 0);
    assert.equal(h.session.currentFrame.source, second.bitmap);
    assert.equal(h.session.inFlight, false);
  } finally {
    firstProcessing.resolve(); secondProcessing.resolve(); h.cleanup(); await turn();
  }
});

test('late resolution or rejection after stop cannot revive the old generation or disturb a replacement session', async () => {
  for (const settlement of ['resolve', 'reject']) {
    const obsoleteProcessing = deferred(); let calls = 0;
    const h = asynchronousSession(() => ++calls === 1 ? obsoleteProcessing.promise : undefined);
    try {
      const oldStream = streamStub(), oldStart = h.session.start({ stream: oldStream });
      await turn();
      const oldWorker = h.workers[0]; oldWorker.onmessage({ data: { type: 'ready' } }); await turn();
      const oldFrame = h.frames(oldWorker)[0]; h.finish(oldWorker, oldFrame);
      h.session.stop();
      assert.equal(await oldStart, null);
      assert.equal(oldStream.track.stopped, 1);
      assert.equal(oldFrame.bitmap.closed, 1);

      const activeStream = streamStub(), activeStart = h.session.start({ stream: activeStream });
      await turn();
      const activeWorker = h.workers[1]; activeWorker.onmessage({ data: { type: 'ready' } }); await turn();
      const activeFrame = h.frames(activeWorker)[0]; h.finish(activeWorker, activeFrame);
      await activeStart;
      const generation = h.session.generation, callback = h.session.callback;
      obsoleteProcessing[settlement](new Error('Obsolete extension failure'));
      await turn();
      assert.equal(h.session.generation, generation);
      assert.equal(h.session.state, 'running');
      assert.equal(h.session.currentFrame.source, activeFrame.bitmap);
      assert.equal(h.session.callback, callback);
      assert.equal(h.session.inFlight, false);
      assert.equal(activeFrame.bitmap.closed, 0);
      assert.equal(activeWorker.terminated, 0);
      assert.equal(activeStream.track.stopped, 0);
    } finally { obsoleteProcessing.resolve(); h.cleanup(); await turn(); }
  }
});

test('asynchronous surface rejection shuts down its stream, source, worker and public startup promise', async () => {
  const processing = deferred(), h = asynchronousSession(() => processing.promise);
  try {
    const stream = streamStub(), started = h.session.start({ stream });
    await turn();
    const worker = h.workers[0]; worker.onmessage({ data: { type: 'ready' } }); await turn();
    const frame = h.frames(worker)[0]; h.finish(worker, frame);
    processing.reject(new Error('Surface inference failed'));
    await assert.rejects(started);
    await turn();
    assert.equal(stream.track.stopped, 1);
    assert.equal(frame.bitmap.closed, 1);
    assert.equal(worker.terminated, 1);
    assert.equal(h.session.state, 'error');
    assert.equal(h.session.lastFailure.message, 'Surface inference failed');
    assert.equal(h.session.inFlight, false);
    assert.equal(h.session.currentFrame, null);
    assert.equal(h.session.workerCount, 0);
    assert.equal(h.video.pendingFrameCallbacks.length, 0);
  } finally { processing.resolve(); h.cleanup(); await turn(); }
});

test('stop releases the previous displayed bitmap even when its asynchronous replacement never settles', async () => {
  const neverSettles = deferred(); let calls = 0;
  const h = asynchronousSession(() => ++calls === 2 ? neverSettles.promise : undefined);
  try {
    const started = h.session.start({ stream: streamStub() }); await turn();
    const worker = h.workers[0]; worker.onmessage({ data: { type: 'ready' } }); await turn();
    const previous = h.frames(worker)[0]; h.finish(worker, previous); await started;
    h.video.emitFrame(1.04, 2); await turn();
    const replacement = h.frames(worker)[1]; h.finish(worker, replacement);
    assert.equal(previous.bitmap.closed, 0); assert.equal(replacement.bitmap.closed, 0);
    h.session.stop();
    assert.equal(previous.bitmap.closed, 1, 'stop must not depend on an extension eventually resolving');
    assert.equal(replacement.bitmap.closed, 1);
  } finally { neverSettles.resolve(); h.cleanup(); await turn(); }
});

test('a subscriber cannot reopen the one-frame barrier by resuming during onFrame', async () => {
  const processing = deferred();
  let h;
  h = asynchronousSession(() => { h.session.resume(); return processing.promise; });
  try {
    const started = h.session.start({ stream: streamStub() }); await turn();
    const worker = h.workers[0]; worker.onmessage({ data: { type: 'ready' } }); await turn();
    const first = h.frames(worker)[0];
    h.video.emitFrame(1.04, 2); // a later decoded image is already available
    h.finish(worker, first); await turn();
    assert.equal(h.frames(worker).length, 1, 'reentrant resume cannot capture before async processing finishes');
    processing.resolve(); await started; await turn();
    assert.equal(h.frames(worker).length, 2);
  } finally { processing.resolve(); h.cleanup(); await turn(); }
});
