import { readFile, writeFile, mkdir, rename, rm } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const dense = path.join(root, 'public/dense');
const checkOnly = process.argv.includes('--check');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
// Pin the exported decoder metadata itself as well as its referenced payload.
// MHR assets are supplied with the project; changing their format is a deliberate
// exporter/version change, never an implicit download of a newer character.
const mhrPins = {
  'mhr-lod3.json': { bytes: 18723, sha256: 'fb31b0383d47338fbcdce7d053082c86e2205401cf95b92ce5bf0281fabc9f45' },
  'mhr-lod3.bin.gz': { bytes: 7442328, sha256: 'eba865d2b2a0637245971732ec7b93b991d55be53fd261998d54df3e44313335' },
};

async function atomicWrite(filename, bytes) {
  await mkdir(path.dirname(filename), { recursive: true });
  const temporary = `${filename}.${process.pid}.${randomUUID()}.part`;
  try {
    await writeFile(temporary, bytes, { flag: 'wx' });
    await rename(temporary, filename);
  } finally { await rm(temporary, { force: true }); }
}
async function existing(filename) {
  try { return await readFile(filename); }
  catch (error) { if (error.code === 'ENOENT') return null; throw error; }
}
function verify(bytes, expected, name) {
  if (!bytes || bytes.length !== expected.bytes || digest(bytes) !== expected.sha256) {
    throw new Error(`${name} checksum/size mismatch. Expected ${expected.bytes} bytes, SHA-256 ${expected.sha256}. Restore the pinned project asset; no replacement was installed.`);
  }
}

const model = JSON.parse(await readFile(path.join(dense, 'model-manifest.json'), 'utf8'));
if (!/^[a-f0-9]{40}$/.test(model.revision) || !/^[a-f0-9]{64}$/.test(model.sha256) || !Number.isInteger(model.bytes) || model.bytes <= 0) throw new Error('Invalid pinned InstantHMR manifest');
const remote = new URL(model.url);
if (remote.protocol !== 'https:' || remote.hostname !== 'huggingface.co' || !remote.pathname.includes(`/resolve/${model.revision}/`)) throw new Error('Model download URL must use the pinned Hugging Face revision');
for (const [name, pin] of Object.entries(mhrPins)) verify(await existing(path.join(dense, name)), pin, name);
const metadata = JSON.parse(await readFile(path.join(dense, 'mhr-lod3.json'), 'utf8'));
if (metadata.binaryFile !== 'mhr-lod3.bin.gz' || metadata.sha256 !== mhrPins['mhr-lod3.bin.gz'].sha256) throw new Error('MHR metadata references a different binary');
const unpacked = gunzipSync(await readFile(path.join(dense, 'mhr-lod3.bin.gz')));
if (unpacked.length !== metadata.uncompressedBytes || digest(unpacked) !== '879273a56eb36ea8eaf112c99666d0e1bc2a6846609dc15fc7d52aa7f533a779') throw new Error('MHR decompressed array payload mismatch');

const ortDirectory = path.join(root, 'node_modules/onnxruntime-web');
const ortPackage = JSON.parse(await readFile(path.join(ortDirectory, 'package.json'), 'utf8'));
if (ortPackage.version !== '1.30.0') throw new Error(`Expected onnxruntime-web 1.30.0, found ${ortPackage.version}. Run npm ci with the pinned lockfile.`);
// The dense worker imports onnxruntime-web/webgpu without embedded WASM
// (see vite.config.js); that entry runs the asyncify build of the WebGPU EP.
const runtimeFiles = {};
for (const name of ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm']) {
  const source = await readFile(path.join(ortDirectory, 'dist', name));
  const destination = path.join(dense, 'ort', name), installed = await existing(destination);
  runtimeFiles[name] = { bytes: source.length, sha256: digest(source) };
  if (installed && digest(installed) === digest(source)) continue;
  if (checkOnly) throw new Error(`${name} missing or differs from installed ONNX Runtime 1.30.0; run npm run setup:dense.`);
  await atomicWrite(destination, source);
  console.log(`Installed ONNX Runtime 1.30.0 ${name}`);
}
// Earlier setups also copied the unused JSEP runtime (28 MB); do not deploy it.
for (const name of ['ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm']) {
  if (!checkOnly) await rm(path.join(dense, 'ort', name), { force: true });
}
const runtimeManifest = JSON.stringify({ package: 'onnxruntime-web', version: ortPackage.version, files: runtimeFiles }, null, 2) + '\n';
const manifestPath = path.join(dense, 'ort', 'manifest.json');
if ((await existing(manifestPath))?.toString() !== runtimeManifest) {
  if (checkOnly) throw new Error('public/dense/ort/manifest.json is missing or stale; run npm run setup:dense.');
  await atomicWrite(manifestPath, runtimeManifest);
}

const filename = path.join(dense, 'instanthmr.onnx');
let bytes = await existing(filename);
if (!bytes || bytes.length !== model.bytes || digest(bytes) !== model.sha256) {
  if (checkOnly) verify(bytes, model, 'instanthmr.onnx');
  console.log(`Downloading InstantHMR at ${model.revision} (${model.bytes} bytes)…`);
  const response = await fetch(model.url, { signal: AbortSignal.timeout(180000) });
  if (!response.ok) throw new Error(`InstantHMR download failed: HTTP ${response.status}`);
  bytes = Buffer.from(await response.arrayBuffer());
  verify(bytes, model, 'Downloaded instanthmr.onnx');
  await atomicWrite(filename, bytes);
}
verify(bytes, model, 'instanthmr.onnx');
console.log('Verified pinned InstantHMR, MHR LOD3 metadata/arrays, and local ONNX Runtime files.');
