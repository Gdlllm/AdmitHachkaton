// Static hosts limit the size of one file (Cloudflare Pages: 25 MiB; GitHub
// warns above 50 MB). After `vite build`, every dist file above the limit is
// replaced by <name>.part0..N plus <name>.parts.json; src/shared/assets.js
// fetches the parts when the whole file is absent and verifies the SHA-256.
import { readdir, readFile, writeFile, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const LIMIT = 24 * 1024 * 1024, PART = 20 * 1024 * 1024;

async function* files(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) yield* files(full);
    else yield full;
  }
}

let count = 0;
for await (const file of files(dist)) {
  const bytes = await readFile(file);
  if (bytes.length <= LIMIT) continue;
  const name = path.basename(file), parts = [];
  for (let offset = 0, index = 0; offset < bytes.length; offset += PART, index++) {
    const part = bytes.subarray(offset, offset + PART);
    parts.push({ file: `${name}.part${index}`, bytes: part.length });
    await writeFile(`${file}.part${index}`, part);
  }
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  await writeFile(`${file}.parts.json`, JSON.stringify({ file: name, bytes: bytes.length, sha256, parts }, null, 2) + '\n');
  await rm(file);
  count++;
  console.log(`Split ${path.relative(dist, file)} (${(bytes.length / 2 ** 20).toFixed(1)} MiB) into ${parts.length} parts`);
}
console.log(count ? `No file in dist/ exceeds ${LIMIT / 2 ** 20} MiB now.` : 'No file needed splitting.');
