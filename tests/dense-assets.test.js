import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { fetchAsset } from '../src/shared/assets.js';
import { cropFromBox, normalizeInput, prepareRGBA, INPUT_SIZE } from '../src/dense/inference/preprocess.js';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const payload = Uint8Array.from({ length: 1000 }, (_, i) => (i * 37) % 251);

// A static host: `files` maps absolute path -> bytes or {json}; everything else
// is a 404, or index.html with 200 when `spa` (an SPA rewrite).
function host(files, { spa = false } = {}) {
  const requests = [];
  globalThis.fetch = async url => {
    const path = new URL(url).pathname; requests.push(path);
    const file = files[path];
    if (file?.json) return new Response(JSON.stringify(file.json), { headers: { 'content-type': 'application/json' } });
    if (file) return new Response(file, { headers: { 'content-type': 'application/octet-stream' } });
    return spa ? new Response('<!doctype html>', { headers: { 'content-type': 'text/html' } }) : new Response('missing', { status: 404 });
  };
  return requests;
}
globalThis.location ??= { href: 'http://localhost/' };
const split = { '/big.bin.parts.json': { json: { file: 'big.bin', bytes: 1000, sha256: sha256(payload),
  parts: [{ file: 'big.bin.part0', bytes: 600 }, { file: 'big.bin.part1', bytes: 400 }] } },
  '/big.bin.part0': payload.slice(0, 600), '/big.bin.part1': payload.slice(600) };

test('a whole file is fetched directly, verified against its pin, with progress', async () => {
  const requests = host({ '/big.bin': payload });
  const progress = [];
  const buffer = await fetchAsset('/big.bin', { bytes: 1000, sha256: sha256(payload), onProgress: (loaded, total) => progress.push([loaded, total]) });
  assert.deepEqual(new Uint8Array(buffer), payload);
  assert.deepEqual(requests, ['/big.bin'], 'no probe for parts when the file exists');
  assert.deepEqual(progress.at(-1), [1000, 1000]);
});

test('a file split by the build is joined from its parts, also behind an SPA rewrite', async () => {
  for (const spa of [false, true]) {
    const requests = host(split, { spa });
    const buffer = await fetchAsset('/big.bin');
    assert.deepEqual(new Uint8Array(buffer), payload);
    assert.deepEqual(requests.sort(), ['/big.bin', '/big.bin.part0', '/big.bin.part1', '/big.bin.parts.json']);
  }
});

test('corrupt, truncated or missing assets are errors', async () => {
  host({ ...split, '/big.bin.part1': payload.slice(600, 999) });
  await assert.rejects(fetchAsset('/big.bin'), /expected 1000 bytes/);
  const corrupt = Uint8Array.from(payload); corrupt[5] ^= 1;
  host({ '/big.bin': corrupt });
  await assert.rejects(fetchAsset('/big.bin', { sha256: sha256(payload) }), /SHA-256 mismatch/);
  host({}, { spa: true });
  await assert.rejects(fetchAsset('/big.bin'), /big\.bin/);
});

test('the GPU crop path and the reference CPU crop use identical geometry and normalization', () => {
  const width = 320, height = 240;
  const data = Uint8ClampedArray.from({ length: width * height * 4 }, (_, i) => (i * 7919) % 256);
  const bbox = [100, 40, 190, 200];
  const crop = cropFromBox(bbox, width, height);
  const reference = prepareRGBA({ data, width, height }, { bbox });
  for (const key of ['x', 'y', 'cx', 'cy', 'size']) assert.equal(crop[key], reference.crop[key]);
  assert.deepEqual(crop.rasterBounds, reference.crop.rasterBounds);
  assert.deepEqual(Array.from(crop.cliff), Array.from(reference.cliff));
  assert.equal(crop.focalLength, Math.hypot(width, height));
  // The GPU path hands the worker 224x224 RGBA; normalization is ImageNet NCHW.
  const patch = Uint8ClampedArray.from({ length: INPUT_SIZE * INPUT_SIZE * 4 }, (_, i) => (i * 31) % 256);
  const input = normalizeInput(patch), plane = INPUT_SIZE * INPUT_SIZE;
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  for (const i of [0, 1, 12345, plane - 1]) for (let c = 0; c < 3; c++) {
    assert.ok(Math.abs(input[c * plane + i] - (patch[i * 4 + c] / 255 - mean[c]) / std[c]) < 1e-6);
  }
  assert.ok(Math.abs(normalizeInput(new Uint8Array(plane * 4))[0] + mean[0] / std[0]) < 1e-6, 'zero padding stays zero before normalization');
  assert.throws(() => normalizeInput(new Uint8Array(10)), /224x224/);
  assert.throws(() => cropFromBox([0, 0, 0, 10], width, height), /positive area/);
});
