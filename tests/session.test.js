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
