import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { gunzipSync, gzipSync } from 'node:zlib';
import { decodeMhrAsset, viewsFromManifest } from '../src/dense/decoder/mhr-decoder.js';

const manifest = JSON.parse(readFileSync(new URL('../public/dense/mhr-lod3.json', import.meta.url)));
const compressed = readFileSync(new URL('../public/dense/mhr-lod3.bin.gz', import.meta.url));
const raw = gunzipSync(compressed);
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const arrayBuffer = bytes => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

test('raw gzip and an HTTP-decoded body produce the identical pinned MHR arrays', async () => {
  const fromGzip = await decodeMhrAsset(manifest, arrayBuffer(compressed));
  const fromHttpDecoded = await decodeMhrAsset(manifest, arrayBuffer(raw));
  assert.equal(fromGzip.byteLength, manifest.uncompressedBytes);
  assert.deepEqual(new Uint8Array(fromGzip), new Uint8Array(fromHttpDecoded));
  const arrays = viewsFromManifest(manifest, fromHttpDecoded);
  assert.equal(arrays.baseShape.length, 4899 * 3);
  assert.equal(arrays.keypointRegressor.length, 70 * 127);
});

test('corruption and truncation fail in either HTTP representation', async () => {
  for (const payload of [compressed, raw]) {
    const corrupted = Uint8Array.from(payload);
    corrupted[Math.floor(corrupted.length / 2)] ^= 1;
    await assert.rejects(decodeMhrAsset(manifest, corrupted), /SHA-256 mismatch/);
    await assert.rejects(decodeMhrAsset(manifest, payload.subarray(0, payload.length - 1)), /size mismatch/);
  }
});

test('a matching gzip checksum cannot bypass verification of its decompressed contents', async () => {
  const corrupted = Uint8Array.from(raw); corrupted[12345] ^= 1;
  const repacked = gzipSync(corrupted, { level: 1 });
  const alteredManifest = { ...manifest, compressedBytes: repacked.byteLength, sha256: sha256(repacked) };
  await assert.rejects(decodeMhrAsset(alteredManifest, repacked), /decompressed asset SHA-256 mismatch/);
});

test('decompressed byte length is verified after successful gzip authentication', async () => {
  const shortened = gzipSync(raw.subarray(0, raw.length - 4), { level: 1 });
  const alteredManifest = { ...manifest, compressedBytes: shortened.byteLength, sha256: sha256(shortened) };
  await assert.rejects(decodeMhrAsset(alteredManifest, shortened), /decompressed asset size mismatch/);
});

test('a typed-array window is honored and non-byte inputs fail clearly', async () => {
  const padded = new Uint8Array(raw.length + 16); padded.set(raw, 8);
  const result = await decodeMhrAsset(manifest, padded.subarray(8, padded.length - 8));
  assert.equal(sha256(new Uint8Array(result)), sha256(raw));
  await assert.rejects(decodeMhrAsset(manifest, 'not bytes'), /ArrayBuffer or typed array/);
});
