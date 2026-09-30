/** Turns body-sense probabilities into stable decisions for the surface fit:
 * which MediaPipe left/right pairs to exchange, and whether the surface
 * network's pose should be read the other way round (front ↔ back).
 *
 * Three opinions on front/back: MediaPipe's left/right labels, the surface
 * network, and the model. While MediaPipe and the network agree, nothing is
 * changed. When they keep disagreeing (for `hold` seconds; a few flickering
 * frames are normal around side views) the model referees, and it keeps doing
 * so until they have agreed again for `hold` seconds. The model overrules both
 * only when the person has turned away: the face detector had been finding the
 * face steadily (`steadyFace` of the frames over about `faceWindow` seconds),
 * lost it `hold` seconds ago, and the model is sure of a back. A face that only
 * flickers in and out (small, blurred) says nothing about turning. Close to the
 * camera both MediaPipe and the network mostly read a back as a front (stock
 * clips, head and shoulders: 62% and 90% of back views). Otherwise it never
 * overrules both: it has not seen the pixels, and a front-facing person whose
 * face is simply not detected can look like a back to it. */
const SHOULDERS = 5;                  // [11, 12] in POSE_PAIRS

export function createSensePolicy({ sure = .8, release = .6, swapOn = .65, swapOff = .35, networkMinDepth = .3, hold = .3, turnedAway = .9,
  faceWindow = 1, steadyFace = .6 } = {}) {
  const swapped = new Array(16).fill(false);
  let back = null, flipped = false, referee = false, conflictSince = null, accordSince = null;
  let faceRate = 0, faceRateAtLoss = 0, faceGoneSince = null, lastT = null, turned = false;
  return {
    /** out: body-sense step; network: the network's torso direction (camera axes) or null;
     * labels: labelFacing() of the raw landmarks (+1 front, -1 back, null unclear); t: seconds;
     * face: this frame has a face matching the body. */
    update(out, network, labels, t, face = false) {
      const dt = lastT === null ? 0 : Math.min(Math.max(t - lastT, 0), .5);
      lastT = t;
      if (face) faceGoneSince = null;
      else if (faceGoneSince === null) { faceGoneSince = t; faceRateAtLoss = faceRate; }
      faceRate += ((face ? 1 : 0) - faceRate) * Math.min(1, dt / faceWindow);
      if (out.back > sure) back = true;
      else if (out.back < 1 - sure) back = false;
      else if ((back === true && out.back < release) || (back === false && out.back > 1 - release)) back = null;
      // Turned away: needs a sure model once, then holds while the face stays lost and the back decision stands.
      const faceLost = faceGoneSince !== null && t - faceGoneSince >= hold && faceRateAtLoss >= steadyFace;
      turned = faceLost && (turned ? back === true : out.back > turnedAway);
      const networkBack = network && Math.abs(network[2]) > networkMinDepth ? network[2] > 0 : null;
      const labelsBack = labels ? labels < 0 : null;
      if (networkBack !== null && labelsBack !== null) {
        if (networkBack !== labelsBack) { conflictSince ??= t; accordSince = null; }
        else { accordSince ??= t; conflictSince = null; }
      }
      if (back === null) referee = false;
      else if (!referee && ((conflictSince !== null && t - conflictSince >= hold) || turned)) referee = true;
      else if (referee && accordSince !== null && t - accordSince >= hold && !turned) referee = false;
      // MediaPipe's labels flicker for a few frames around side views. When the
      // model sides with the network, putting the labels back is cheap and safe,
      // so it happens at once; turning the network's pose round waits for `hold`.
      const flicker = !referee && conflictSince !== null && back !== null && back === networkBack;
      if (flicker) {
        out.swap.forEach((p, i) => { if (p > swapOn) swapped[i] = true; else if (p < swapOff) swapped[i] = false; });
        flipped = false;
      } else if (!referee) { swapped.fill(false); flipped = false; }
      else {
        out.swap.forEach((p, i) => { if (p > swapOn) swapped[i] = true; else if (p < swapOff) swapped[i] = false; });
        // Side-on to the camera the network has no front/back to correct; keep the last reading.
        if (networkBack !== null) flipped = networkBack !== back;
      }
      // The fit needs MediaPipe's left and right on the decided side: when the
      // shoulders still read the other way after the model's exchanges, exchange every pair.
      if ((flicker || referee) && labelsBack !== null && (labelsBack !== swapped[SHOULDERS]) !== back) swapped.fill(labelsBack !== back);
      return { swapped: swapped.slice(), flipped, back, referee, flicker, away: turned };
    },
    reset() {
      swapped.fill(false); flipped = false; back = null; referee = false; conflictSince = null; accordSince = null;
      faceRate = 0; faceRateAtLoss = 0; faceGoneSince = null; lastT = null; turned = false;
    },
  };
}
