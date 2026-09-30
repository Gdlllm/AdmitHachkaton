import { defineConfig, defaultClientConditions } from 'vite';
import { mkdirSync, writeFileSync } from 'node:fs';

// Dev only: the gaze calibration page (?games=1&calibrate=gaze) posts its recording here,
// and it lands in ml/gaze/data/ for offline tuning. Nothing like it exists in a build.
const gazeRecorder = {
  name: 'gaze-recorder',
  configureServer(server) {
    server.middlewares.use('/__gaze-data', (req, res) => {
      if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }
      const chunks = [];
      req.on('data', c => chunks.push(c));
      req.on('end', () => {
        mkdirSync('ml/gaze/data', { recursive: true });
        // A game sends its recording in parts (?game=<id>&part=<n>), so a reload loses little.
        const q = new URL(req.url, 'http://x').searchParams, game = q.get('game')?.replace(/[^\w-]/g, ''), part = Number(q.get('part')) || 0;
        const file = game ? `ml/gaze/data/play-${game}-${String(part).padStart(3, '0')}.json` : `ml/gaze/data/session-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
        writeFileSync(file, Buffer.concat(chunks));
        res.setHeader('content-type', 'application/json'); res.end(JSON.stringify({ file }));
      });
    });
  },
};

export default defineConfig({
  plugins: [gazeRecorder],
  resolve: {
    // ONNX Runtime without its embedded WASM: the dense worker loads the one
    // runtime it uses from public/dense/ort, so no second 26 MB copy is bundled.
    conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions],
  },
  // The surface worker is an ES module (ONNX Runtime imports its glue module).
  worker: { format: 'es' },
});
