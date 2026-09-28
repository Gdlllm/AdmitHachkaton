import { cp, mkdir, readFile, writeFile, rename, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
const files = [
  ['pose_heavy.task', 'pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task', '64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b'],
  ['pose_full.task', 'pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task', '5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1'],
  ['pose_lite.task', 'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a'],
  ['hand_landmarker.task', 'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'],
  ['face_landmarker.task', 'face_landmarker/face_landmarker/float16/1/face_landmarker.task', '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'],
];
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
await mkdir(`${root}public/models`, { recursive: true });
// WASM comes from the exact package version in package-lock.json.
await cp(`${root}node_modules/@mediapipe/tasks-vision/wasm`, `${root}public/wasm`, { recursive: true });
for (const [name, remotePath, sha256] of files) {
  const path = `${root}public/models/${name}`;
  try {
    if (digest(await readFile(path)) === sha256) continue;
    console.log(`Replacing incomplete or outdated ${name}…`);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  console.log(`Downloading ${name} (pinned version 1)…`);
  const response = await fetch(`https://storage.googleapis.com/mediapipe-models/${remotePath}`, { signal: AbortSignal.timeout(180000) });
  if (!response.ok) throw new Error(`Could not download ${name}: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (digest(bytes) !== sha256) throw new Error(`Checksum mismatch for ${name}. The file was not installed.`);
  const temporary = `${path}.${process.pid}.part`;
  try { await writeFile(temporary, bytes); await rename(temporary, path); }
  finally { await rm(temporary, { force: true }); }
}
await writeFile(`${root}public/models/manifest.json`, JSON.stringify({
  library: '@mediapipe/tasks-vision', version: '1.0.1',
  models: files.map(([file, remotePath, sha256]) => ({ file, sha256, url: `https://storage.googleapis.com/mediapipe-models/${remotePath}` })),
}, null, 2) + '\n');
console.log('Verified body, hand and face models. Inference and camera frames stay in the browser.');
