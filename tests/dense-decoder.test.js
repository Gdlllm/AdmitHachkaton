import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { createDecoderFromArrays, viewsFromManifest } from '../src/dense/decoder/mhr-decoder.js';

const directory = new URL('../public/dense/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('mhr-lod3.json', directory)));
const reference = JSON.parse(readFileSync(new URL('./dense-assets/reference.json', import.meta.url)));
const readBinary = filename => {
  const bytes = gunzipSync(readFileSync(new URL(filename, filename.startsWith('reference') ? new URL('./dense-assets/', import.meta.url) : directory)));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};
const data = viewsFromManifest(manifest, readBinary('mhr-lod3.bin.gz'));
const expected = viewsFromManifest(reference, readBinary('reference.bin.gz'));
const decoder = createDecoderFromArrays(manifest, data);

for (const sample of reference.cases) {
  test(`official CPU reference: ${sample.name}, all vertices and joints`, () => {
    const actual = decoder.decode(sample.parameters, sample.identity, sample.expression);
    for (const [name, output] of [['vertices', actual.vertices], ['positions', actual.skeleton.positions], ['keypoints70', actual.keypoints70]]) {
      const target = expected[`${sample.name}/${name}`];
      assert.equal(output.length, target.length);
      let maximum = 0;
      for (let i = 0; i < target.length; i++) maximum = Math.max(maximum, Math.abs(output[i] - target[i]));
      // 0.01 mm numerical tolerance, not real-world reconstruction accuracy.
      assert.ok(maximum <= 1e-5, `${name}: ${maximum}m exceeds 0.01mm tolerance`);
    }
    assert.equal(actual.vertices.length, 4899 * 3);
    assert.equal(actual.skeleton.positions.length, 127 * 3);
    assert.equal(actual.keypoints70.length, 70 * 3);
    assert.equal(actual.keypointNames70.length, 70);
    assert.equal(actual.axes, 'X-right/Y-down/Z-forward');
    const fast = decoder.decodeSkeleton(sample.parameters);
    assert.deepEqual(fast.positions, actual.skeleton.positions, 'fast FK matches full decode exactly');
    assert.deepEqual(fast.keypoints70, actual.keypoints70, 'fast FK keypoint regression matches full decode exactly');
  });
}

test('same identity is independent of earlier frames, and results are not mutated later', () => {
  const sample = reference.cases[0];
  const first = decoder.decode(sample.parameters, sample.identity, sample.expression);
  const copy = first.vertices.slice();
  const firstRig = decoder.decodeSkeleton(sample.parameters);
  const rigCopy = firstRig.positions.slice(), keypointCopy = firstRig.keypoints70.slice();
  const alternate = reference.cases.at(-1);
  decoder.decode(alternate.parameters, alternate.identity, alternate.expression);
  decoder.decodeSkeleton(alternate.parameters);
  assert.deepEqual(first.vertices, copy);
  assert.deepEqual(firstRig.positions, rigCopy);
  assert.deepEqual(firstRig.keypoints70, keypointCopy);
  assert.deepEqual(decoder.decode(sample.parameters, sample.identity, sample.expression).vertices, copy);
});

test('invalid parameter input and disposed decoder fail clearly', () => {
  const temporary = createDecoderFromArrays(manifest, data);
  assert.throws(() => temporary.decode(new Float32Array(203), new Float32Array(45)), /204/);
  const bad = new Float32Array(204); bad[10] = NaN;
  assert.throws(() => temporary.decode(bad, new Float32Array(45)), /not finite/);
  assert.throws(() => temporary.decodeSkeleton(bad), /not finite/);
  assert.throws(() => temporary.decodeSkeleton(new Float32Array(203)), /204/);
  temporary.dispose();
  assert.throws(() => temporary.decode(new Float32Array(204), new Float32Array(45)), /disposed/);
  assert.throws(() => temporary.decodeSkeleton(new Float32Array(204)), /disposed/);
});

test('model parameter names match the original 204-value transform order', () => {
  assert.equal(decoder.metadata.modelParameterNames.length, 204);
  assert.deepEqual(decoder.metadata.modelParameterNames, decoder.metadata.parameterNames);
});

test('subset keypoint regression used by fitting matches full FK exactly; the identity cache is transparent', () => {
  const indices = [0, 5, 6, 41, 62, 13, 14];
  for (const sample of reference.cases) {
    const full = decoder.decodeSkeleton(sample.parameters).keypoints70;
    const subset = decoder.decodeKeypoints(sample.parameters, indices);
    for (const k of indices) for (let d = 0; d < 3; d++) assert.equal(subset[k * 3 + d], full[k * 3 + d]);
    assert.ok(Number.isNaN(subset[3]), 'keypoints that were not requested are not computed');
  }
  // Same identity twice (cached blend) and after another identity: identical vertices.
  const [a, b] = [reference.cases[0], reference.cases.at(-1)];
  const first = decoder.decode(a.parameters, a.identity, a.expression).vertices;
  const cached = decoder.decode(a.parameters, a.identity, a.expression).vertices;
  decoder.decode(b.parameters, b.identity, b.expression);
  assert.deepEqual(cached, first);
  assert.deepEqual(decoder.decode(a.parameters, a.identity, a.expression).vertices, first);
});
