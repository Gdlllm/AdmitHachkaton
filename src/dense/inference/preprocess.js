export const INPUT_SIZE = 224;
export const CROP_EXPAND = 1.2;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

/** Square 1.2x crop around a tight person box, as upstream InstantHMR does.
 * Pixel-conditioned CLIFF for the pinned public checkpoint (metadata absent).
 * Do not substitute angular CLIFF just because a calibrated focal is provided.
 * Upstream source: InstantHMR/inference.py at upstream/revisions.json.
 */
export function cropFromBox(bbox, width, height, { focal } = {}) {
  if (!(width > 0 && height > 0)) throw new Error('Invalid source dimensions');
  if (!Array.isArray(bbox) || bbox.length !== 4 || !bbox.every(Number.isFinite)) throw new Error('A source-pixel person bbox [x1,y1,x2,y2] is required');
  const [x1, y1, x2, y2] = bbox, bw = x2 - x1, bh = y2 - y1;
  if (!(bw > 0 && bh > 0 && x2 > 0 && y2 > 0 && x1 < width && y1 < height)) throw new Error('The bbox must have positive area and intersect the source');
  if (focal !== undefined && !(Number.isFinite(focal) && focal > 0)) throw new Error('Focal length must be positive source pixels');
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, size = Math.max(bw, bh) * CROP_EXPAND;
  const x = cx - size / 2, y = cy - size / 2;
  // Match upstream floor/ceil padded raster extraction. The logical float crop
  // is retained separately because upstream inverse projection uses it.
  const rasterBounds = [Math.floor(x), Math.floor(y), Math.ceil(x + size), Math.ceil(y + size)];
  return {
    x, y, cx, cy, size, width, height, inputSize: INPUT_SIZE, rasterBounds,
    cliff: new Float32Array([2 * cx / width - 1, 2 * cy / height - 1, Math.max(bw, bh) / Math.max(width, height)]),
    focalLength: focal ?? Math.hypot(width, height), focalAssumed: focal === undefined,
    principalPoint: [width / 2, height / 2],
  };
}

/** A 224x224 RGBA crop (already resized, zero outside the source) to NCHW. */
export function normalizeInput(rgba) {
  if (rgba?.length !== INPUT_SIZE * INPUT_SIZE * 4) throw new Error(`Expected ${INPUT_SIZE}x${INPUT_SIZE} RGBA pixels`);
  const plane = INPUT_SIZE * INPUT_SIZE, input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) input[c * plane + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  }
  return input;
}

/** Reference CPU path for a full RGBA frame (tests and offline tools). */
export function prepareRGBA({ data, width, height }, { bbox, focal } = {}) {
  if (!(width > 0 && height > 0) || data.length !== width * height * 4) throw new Error('Invalid RGBA image dimensions');
  const crop = cropFromBox(bbox, width, height, { focal });
  const [ix1, iy1, ix2, iy2] = crop.rasterBounds;
  const patchWidth = ix2 - ix1, patchHeight = iy2 - iy1;
  const input = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const pixel = (px, py, channel) => px < 0 || py < 0 || px >= width || py >= height ? 0 : data[(py * width + px) * 4 + channel];
  for (let dy = 0; dy < INPUT_SIZE; dy++) {
    const sy = Math.max(0, Math.min(patchHeight - 1, (dy + 0.5) * patchHeight / INPUT_SIZE - 0.5));
    const py0 = Math.floor(sy), py1 = Math.min(patchHeight - 1, py0 + 1), fy = sy - py0;
    for (let dx = 0; dx < INPUT_SIZE; dx++) {
      const sx = Math.max(0, Math.min(patchWidth - 1, (dx + 0.5) * patchWidth / INPUT_SIZE - 0.5));
      const px0 = Math.floor(sx), px1 = Math.min(patchWidth - 1, px0 + 1), fx = sx - px0;
      for (let c = 0; c < 3; c++) {
        const top = pixel(ix1 + px0, iy1 + py0, c) * (1 - fx) + pixel(ix1 + px1, iy1 + py0, c) * fx;
        const bottom = pixel(ix1 + px0, iy1 + py1, c) * (1 - fx) + pixel(ix1 + px1, iy1 + py1, c) * fx;
        // cv2 INTER_LINEAR resizes uint8 before normalisation. Rounding is
        // deliberate; OpenCV's fixed-point implementation may differ by 1 RGB.
        const value = Math.round(top * (1 - fy) + bottom * fy);
        input[c * INPUT_SIZE * INPUT_SIZE + dy * INPUT_SIZE + dx] = (value / 255 - MEAN[c]) / STD[c];
      }
    }
  }
  const { cliff, focalLength, focalAssumed, principalPoint, ...geometry } = crop;
  return { input, cliff, crop: geometry, focalLength, focalAssumed, principalPoint };
}

export function inverseCrop(points, crop) {
  if (points.length % 2) throw new Error('2D points must be xy pairs');
  return Array.from({ length: points.length / 2 }, (_, i) => ({
    x: (points[i * 2] + 1) * 0.5 * crop.size + crop.x,
    y: (points[i * 2 + 1] + 1) * 0.5 * crop.size + crop.y,
  }));
}

/** Perspective projection of camera-space metres; never clamps off-screen joints. */
export function projectCameraPoints(points, focalLength, principalPoint) {
  return points.map(({ x, y, z }) => z > 1e-6 && [x, y, z].every(Number.isFinite)
    ? { x: x / z * focalLength + principalPoint[0], y: y / z * focalLength + principalPoint[1] }
    : null);
}
