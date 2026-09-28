import { mkdir, readFile, writeFile, rename, rm, access } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const args = process.argv.slice(2);
const option = name => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : undefined; };
const destination = path.resolve(option('--out') ?? process.env.CAPTURE_FIXTURES_DIR ?? fileURLToPath(new URL('../tests/fixtures', import.meta.url)));
const ffmpeg = option('--ffmpeg') ?? process.env.FFMPEG_PATH ?? 'ffmpeg';
const tfjs = 'https://raw.githubusercontent.com/tensorflow/tfjs-models/c731b9ebbd6f4c9e8bf99b0df76bbdbf9c25b07f/';
const openface = 'https://raw.githubusercontent.com/TadasBaltrusaitis/OpenFace/3d4b5cf8d96138be42bed229447f36cbb09a5a29/samples/';
const fixtures = [
  { name: 'body-motion.mp4', url: `${tfjs}pose-detection/test_data/pose_1.mp4`, sha256: 'a581dd3c6495c4257a8f7756c6c59b445e8949eec0bc2637ba5283bfa7d71930' },
  { name: 'body-squats.mp4', url: `${tfjs}pose-detection/test_data/pose_squats.mp4`, sha256: 'ea9151e447b301985d5d65666551ef863b369a2e0f3a71ddd58abef2e722f96a' },
  { name: 'hand-signs.mp4', url: `${tfjs}hand-pose-detection/test_data/asl_hand.25fps.mp4`, sha256: '57c10fb1eb76639edf43e9675213dcc495c51851e32a3592cacaa9437be4f37e' },
  { name: 'face-turns.wmv', output: 'face-turns.mp4', url: `${openface}default.wmv`, sha256: 'b6c4879cbe14f85412d33be9c0854d3920aacc26f90115868c94e40f836a316c' },
  { name: 'face-lighting.wmv', output: 'face-lighting.mp4', url: `${openface}changeLighting.wmv`, sha256: 'b656ad942a31edb413e5bea169bccccf4f32357c03f86903c1f761e472f3ea93' },
];
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
await mkdir(destination, { recursive: true });
for (const item of fixtures) {
  const filename = path.join(destination, item.name);
  let valid = false;
  try { valid = hash(await readFile(filename)) === item.sha256; } catch {}
  if (!valid) {
    console.log(`Downloading ${item.name}`);
    const response = await fetch(item.url, { signal: AbortSignal.timeout(60000) });
    if (!response.ok) throw new Error(`${item.name}: HTTP ${response.status}`);
    const bytes = Buffer.from(await response.arrayBuffer());
    if (hash(bytes) !== item.sha256) throw new Error(`${item.name}: checksum mismatch`);
    await writeFile(`${filename}.part`, bytes); await rename(`${filename}.part`, filename);
  }
  if (item.output) {
    const target = path.join(destination, item.output);
    try { await access(target); } catch {
      const temporary = `${target}.part.mp4`;
      console.log(`Converting ${item.name} to browser-compatible MP4`);
      try {
        await promisify(execFile)(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-i', filename, '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-y', temporary]);
        await rename(temporary, target);
      } catch (error) {
        await rm(temporary, { force: true });
        throw new Error(`FFmpeg is required for the OpenFace WMV clips. Pass --ffmpeg /path/to/ffmpeg. ${error.message}`);
      }
    }
  }
}
await writeFile(path.join(destination, 'sources.json'), JSON.stringify({ fixtures, note: 'Original source hashes; two OpenFace videos are transcoded locally for browser playback.' }, null, 2) + '\n');
console.log(`Five real-video fixtures ready in ${destination}`);
