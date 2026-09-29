/** Anatomical patches derived from the actual MHR skinning support.
 * A region mask is supplied by an independent visibility/presence gate. This
 * module is not a detector, silhouette segmenter, or occlusion estimator.
 */
export const SURFACE_REGIONS = Object.freeze({ body: 1, leftArm: 2, rightArm: 4, leftLeg: 8, rightLeg: 16 });

export function createVisibleSurface({ faces, skinIndices, skinWeights, parents, jointNames, vertexCount }) {
  if (!Number.isInteger(vertexCount) || vertexCount < 1 || skinIndices?.length !== vertexCount * 8 || skinWeights?.length !== vertexCount * 8) {
    throw new TypeError('Expected the MHR eight-influence skinning arrays');
  }
  if (!faces || faces.length % 3 || !parents || parents.length !== jointNames?.length) throw new TypeError('Invalid MHR topology');
  const roots = { l_clavicle: 'leftArm', r_clavicle: 'rightArm', l_upleg: 'leftLeg', r_upleg: 'rightLeg' };
  const required = ['root', 'l_wrist', 'r_wrist', 'c_head', ...Object.keys(roots)];
  for (const name of required) if (!jointNames.includes(name)) throw new Error(`Missing MHR anatomical branch: ${name}`);
  const jointRegions = new Uint8Array(parents.length);
  for (let j = 0; j < parents.length; j++) {
    const parent = parents[j], name = jointNames[j];
    if (!Number.isInteger(parent) || parent < -1 || parent >= j) throw new Error('Expected MHR parent-before-child topology');
    // Head/jaw/eyes and complete palms/fingers are left to the dedicated models.
    if (name === 'c_head' || name === 'l_wrist' || name === 'r_wrist') jointRegions[j] = 0;
    else if (roots[name]) jointRegions[j] = SURFACE_REGIONS[roots[name]];
    else if (name === 'root') jointRegions[j] = SURFACE_REGIONS.body;
    else jointRegions[j] = parent < 0 ? 0 : jointRegions[parent];
  }

  const vertexRegions = new Uint8Array(vertexCount);
  const counts = Object.fromEntries([...Object.keys(SURFACE_REGIONS), 'excluded'].map(name => [name, 0]));
  const regionName = Object.fromEntries(Object.entries(SURFACE_REGIONS).map(([name, bit]) => [bit, name]));
  for (let v = 0; v < vertexCount; v++) {
    let dominant = -1, largest = 0;
    for (let k = 0; k < 8; k++) {
      const index = v * 8 + k, joint = skinIndices[index], weight = skinWeights[index];
      if (!Number.isInteger(joint) || joint < 0 || joint >= parents.length || !Number.isFinite(weight) || weight < 0) throw new Error('Invalid MHR skinning influence');
      if (weight > largest) { largest = weight; dominant = joint; }
    }
    const region = dominant < 0 ? 0 : jointRegions[dominant];
    vertexRegions[v] = region;
    counts[regionName[region] ?? 'excluded']++;
  }
  for (const vertex of faces) if (!Number.isInteger(vertex) || vertex < 0 || vertex >= vertexCount) throw new Error('Invalid MHR face index');

  // Only 32 possible masks. Returned buffers are shared read-only static data.
  const cache = new Map();
  function select(mask = {}) {
    let enabled = 0;
    for (const [name, bit] of Object.entries(SURFACE_REGIONS)) if (mask?.[name] === true) enabled |= bit;
    if (cache.has(enabled)) return cache.get(enabled);
    const selected = [];
    for (let i = 0; i < faces.length; i += 3) {
      const a = faces[i], b = faces[i + 1], c = faces[i + 2];
      // Do not retain a bridge triangle into a hidden/excluded patch or invent
      // caps at the cut. Every index remains an original official mesh vertex.
      if ((vertexRegions[a] & enabled) && (vertexRegions[b] & enabled) && (vertexRegions[c] & enabled)) selected.push(a, b, c);
    }
    const result = Uint32Array.from(selected);
    cache.set(enabled, result);
    return result;
  }
  return { select, vertexRegions, jointRegions, counts: Object.freeze(counts), regions: SURFACE_REGIONS,
    source: 'dominant official MHR skinning joint; anatomical support, not observed pixel visibility' };
}
