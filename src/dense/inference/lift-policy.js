/** Decisions for the surface fit when a lift model (body-sense v9+) gives the
 * whole 3D skeleton: the skeleton is the reference, not a referee between
 * MediaPipe and the surface network.
 *
 * - Which way the torso faces comes from the skeleton (with a dead band while
 *   side-on, where the last reading is kept), except while the face tracker
 *   finds a face: it needs a real face in the pixels (MediaPipe Pose draws one
 *   on the back of the head too, the face tracker does not), and close up the
 *   skeleton cannot tell a face from the back of a head.
 * - Head and shoulders only (hips out of the picture): the skeleton cannot tell
 *   a face from the back of a head there (on stock close-ups it read fronts as
 *   backs), while MediaPipe's labels are nearly always right on fronts. The side
 *   then follows the labels, and a skeleton facing the other way is mirrored
 *   in depth before it is used.
 * - The surface network's pose is read the other way round (front ↔ back)
 *   whenever it disagrees with the skeleton.
 * - MediaPipe left/right pairs are exchanged as the model says, and the
 *   shoulders are always brought to the skeleton's side: the fit must see left
 *   and right where the mesh has them.
 */
import { torsoForward } from './body-sense.js';

export function createLiftPolicy({ swapOn = .65, swapOff = .35, depth = .25, faceHold = .3 } = {}) {
  const swapped = new Array(16).fill(false);
  let back = null, flipped = false, faceAt = null;
  return {
    /** out: lift step (joints3d, swap); network: its torso direction (camera axes) or null;
     * labels: labelFacing() of the raw landmarks (+1 front, -1 back, null unclear);
     * t: seconds; face: the face tracker found a face matching this body;
     * closeUp: head and shoulders only. */
    update(out, network, labels, t = 0, face = false, closeUp = false) {
      if (face) faceAt = t;
      const faceSeen = faceAt !== null && t - faceAt <= faceHold;
      const j = i => [out.joints3d[i * 3], out.joints3d[i * 3 + 1], out.joints3d[i * 3 + 2]];
      const forward = torsoForward([...j(11), ...j(12), ...j(23), ...j(24)]);
      // Camera Z points away from the camera: a torso facing it has forward Z < 0.
      const labelsBack = labels ? labels < 0 : null;
      const skeletonBack = forward && Math.abs(forward[2]) > depth ? forward[2] > 0 : null;
      if (faceSeen) back = false;
      else if (closeUp) { if (labelsBack !== null) back = labelsBack; }
      else if (skeletonBack !== null) back = skeletonBack;
      out.swap.forEach((p, i) => { if (p > swapOn) swapped[i] = true; else if (p < swapOff) swapped[i] = false; });
      if (closeUp) swapped.fill(false);
      if (back !== null && labelsBack !== null && (labelsBack !== swapped[5]) !== back) swapped.fill(labelsBack !== back);
      const networkBack = network && Math.abs(network[2]) > depth ? network[2] > 0 : null;
      if (back !== null && networkBack !== null) flipped = networkBack !== back;
      // The skeleton read the other side: use its depth mirror (left and right relabelled).
      const mirrorSkeleton = back !== null && skeletonBack !== null && skeletonBack !== back;
      return { swapped: swapped.slice(), flipped, back, forward, mirrorSkeleton, referee: true, flicker: false, away: false };
    },
    reset() { swapped.fill(false); back = null; flipped = false; faceAt = null; },
  };
}
