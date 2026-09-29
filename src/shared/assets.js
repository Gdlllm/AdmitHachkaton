/** Large static files (model weights, WASM runtimes) for any browser context,
 * including workers. `npm run build` splits files that exceed common static
 * host limits (scripts/split-large-assets.mjs); when the whole file is absent
 * its parts are fetched and joined here. With a pinned SHA-256 the verified
 * bytes are kept in the Cache API, so a reload does not download ~100 MB again.
 * Nothing is uploaded; only same-site files are read.
 */
const CACHE_NAME = 'motion-lab-assets-v1';

const hex = buffer => [...new Uint8Array(buffer)].map(value => value.toString(16).padStart(2, '0')).join('');
export const sha256Hex = async buffer => hex(await crypto.subtle.digest('SHA-256', buffer));

async function openCache() {
  try { return typeof caches === 'undefined' ? null : await caches.open(CACHE_NAME); } catch { return null; }
}

// An SPA fallback answers a missing file with index.html and status 200.
const found = response => response.ok && !/text\/html/.test(response.headers.get('content-type') ?? '');

async function read(response, onChunk) {
  if (!response.body || !onChunk) return new Uint8Array(await response.arrayBuffer());
  const chunks = [];
  let length = 0;
  for (const reader = response.body.getReader(); ;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); length += value.length; onChunk(value.length);
  }
  return join(chunks, length);
}

function join(chunks, length = chunks.reduce((sum, chunk) => sum + chunk.length, 0)) {
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

async function download(url, { signal, onProgress, expectedBytes }) {
  let loaded = 0, total = expectedBytes ?? 0;
  const onChunk = onProgress ? size => { loaded += size; onProgress(loaded, total); } : null;
  const whole = await fetch(url, { signal });
  if (found(whole)) {
    total ||= Number(whole.headers.get('content-length')) || 0;
    return { data: await read(whole, onChunk) };
  }
  await whole.body?.cancel().catch(() => {});
  const listing = await fetch(`${url}.parts.json`, { signal, cache: 'no-cache' });
  const manifest = found(listing) ? await listing.json().catch(() => null) : null;
  if (!Array.isArray(manifest?.parts) || !manifest.parts.length) throw new Error(`${url.split('/').pop()}: HTTP ${whole.status}`);
  total ||= manifest.bytes ?? 0;
  const parts = await Promise.all(manifest.parts.map(async part => {
    const response = await fetch(new URL(part.file, url).href, { signal });
    if (!found(response)) throw new Error(`${part.file}: HTTP ${response.status}`);
    return read(response, onChunk);
  }));
  return { data: join(parts), manifest };
}

/** Returns an ArrayBuffer. `bytes`/`sha256` pin the expected content when known;
 * a mismatch is an error and is never cached. `onProgress(loaded, total)`. */
export async function fetchAsset(path, { bytes: expectedBytes, sha256, signal, onProgress } = {}) {
  const url = new URL(path, globalThis.location?.href).href;
  // The query only makes the cache key content-addressed; it is never fetched.
  const key = sha256 ? `${url}?sha256=${sha256}` : null;
  const cache = key ? await openCache() : null;
  if (cache) {
    const hit = await cache.match(key).catch(() => null);
    if (hit) {
      const buffer = await hit.arrayBuffer();
      if (!expectedBytes || buffer.byteLength === expectedBytes) { onProgress?.(buffer.byteLength, buffer.byteLength); return buffer; }
      await cache.delete(key).catch(() => {});
    }
  }
  const { data, manifest } = await download(url, { signal, onProgress, expectedBytes });
  const name = url.split('/').pop();
  const wantedBytes = expectedBytes ?? manifest?.bytes;
  if (wantedBytes && data.byteLength !== wantedBytes) throw new Error(`${name}: expected ${wantedBytes} bytes, received ${data.byteLength}`);
  const wantedHash = sha256 ?? manifest?.sha256;
  if (wantedHash && await sha256Hex(data) !== wantedHash) throw new Error(`${name}: SHA-256 mismatch`);
  // Response copies the bytes now; storing may finish in the background.
  if (cache) cache.put(key, new Response(data)).catch(() => {});
  return data.buffer;
}
