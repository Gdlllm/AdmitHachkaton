import { defineConfig, defaultClientConditions } from 'vite';

export default defineConfig({
  resolve: {
    // ONNX Runtime without its embedded WASM: the dense worker loads the one
    // runtime it uses from public/dense/ort, so no second 26 MB copy is bundled.
    conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions],
  },
  // The surface worker is an ES module (ONNX Runtime imports its glue module).
  worker: { format: 'es' },
});
