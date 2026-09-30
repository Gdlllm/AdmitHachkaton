/** Game controls read from a capture frame: an imaginary steering wheel held
 * in both hands, and face actions (mouth open, one eye closed). All positions
 * are in the mirrored 0..1 picture the player sees (x grows to their right).
 *
 * The coach turns what is wrong with the grip into one concrete hint, held for
 * a moment so it does not flicker.
 */

const PALM = [0, 5, 9, 13, 17];

export function palmCentre(hand) {
  const p = hand.landmarks;
  const x = PALM.reduce((s, i) => s + p[i].x, 0) / PALM.length, y = PALM.reduce((s, i) => s + p[i].y, 0) / PALM.length;
  return { x: 1 - x, y };
}

/** Both hands on the wheel: palms from the hand tracker, pose wrists as a fallback.
 * Returns { hands, angle (radians, + turns right), left, right, spread, height, edge }. */
export function readWheel(frame) {
  let points = (frame?.hands ?? []).filter(h => h?.landmarks?.length >= 21).map(palmCentre);
  if (points.length < 2) {
    const pose = frame?.pose?.landmarks;
    const wrists = pose ? [15, 16].map(i => pose[i]).filter(p => p && (p.visibility ?? 0) >= .5 && p.drawConfidence !== 0 && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1)
      .map(p => ({ x: 1 - p.x, y: p.y })) : [];
    if (wrists.length > points.length) points = wrists;
  }
  points = points.slice(0, 2).sort((a, b) => a.x - b.x);
  const edge = points.some(p => p.x < .05 || p.x > .95 || p.y > .96);
  if (points.length < 2) return { hands: points.length, angle: 0, left: points[0] ?? null, right: null, spread: 0, height: points[0]?.y ?? null, edge };
  const [left, right] = points;
  // The right hand lower than the left: the wheel is turned clockwise, to the right.
  const angle = Math.atan2(right.y - left.y, right.x - left.x);
  return { hands: 2, angle, left, right, spread: Math.hypot(right.x - left.x, right.y - left.y), height: (left.y + right.y) / 2, edge };
}

const shape = (frame, name) => frame?.face?.blendshapes?.find?.(c => c.categoryName === name)?.score ?? 0;

/** Face actions: mouth open (nitro), one eye closed (brake), both closed (a mistake). */
export function readFace(frame, { open = .45, closed = .5, openEye = .3 } = {}) {
  const found = Boolean(frame?.face?.landmarks?.length);
  if (!found) return { found, mouth: 0, nitro: false, brake: false, bothClosed: false };
  const mouth = shape(frame, 'jawOpen'), l = shape(frame, 'eyeBlinkLeft'), r = shape(frame, 'eyeBlinkRight');
  const bothClosed = l > closed && r > closed;
  const brake = !bothClosed && ((l > closed && r < openEye) || (r > closed && l < openEye));
  return { found, mouth, nitro: mouth > open, brake, bothClosed };
}

export const WHEEL_HINTS = {
  noHands: 'Покажи камере обе руки и возьмись за руль',
  oneHand: 'Держи руль двумя руками',
  outOfFrame: 'Рука вышла из кадра: держи руль ближе к центру',
  tooClose: 'Руки слишком близко: разведи их на ширину руля',
  tooWide: 'Руки слишком широко: сведи их, как на настоящем руле',
  tooLow: 'Подними руль до уровня груди',
  tooHigh: 'Опусти руль ниже, до уровня груди',
  bothEyes: 'Закрой только один глаз: с двумя закрытыми ты не видишь дорогу',
  overTurn: 'Слишком резко: руль поворачивается максимум на пол-оборота',
};

/** Which grip problem to show now (after `holdMs`), or null. */
export function createWheelCoach({ holdMs = 400 } = {}) {
  const since = new Map();
  return {
    update(wheel, face, t) {
      const problems = [];
      if (wheel.hands === 0) problems.push('noHands');
      else if (wheel.hands === 1) problems.push(wheel.edge ? 'outOfFrame' : 'oneHand');
      else {
        if (wheel.edge) problems.push('outOfFrame');
        if (wheel.spread < .12) problems.push('tooClose');
        else if (wheel.spread > .72) problems.push('tooWide');
        if (wheel.height > .88) problems.push('tooLow');
        else if (wheel.height < .12) problems.push('tooHigh');
        if (Math.abs(wheel.angle) > Math.PI * .45) problems.push('overTurn');
      }
      if (face?.bothClosed) problems.push('bothEyes');
      for (const key of [...since.keys()]) if (!problems.includes(key)) since.delete(key);
      for (const key of problems) if (!since.has(key)) since.set(key, t);
      const due = problems.find(key => t - since.get(key) >= holdMs);
      return due ? { code: due, text: WHEEL_HINTS[due] } : null;
    },
    reset() { since.clear(); },
  };
}

/** Wheel angle → steering in -1..1, with a small dead zone and full lock at `lock` radians. */
export function steering(angle, { dead = .07, lock = .75 } = {}) {
  const a = Math.abs(angle);
  if (a < dead) return 0;
  return Math.sign(angle) * Math.min(1, (a - dead) / (lock - dead));
}

/** A steady on-screen cursor from the hand (idea from handstick, rewritten):
 * the palm centre instead of the jittery fingertip, the middle of the picture
 * stretched to the whole screen, one-frame jumps ignored, then a 1€ filter in
 * screen pixels and a small dead zone. update() → {x, y} in 0..1 of the screen, or null. */
export function createHandCursor({ area = .7, minCutoff = 1.1, beta = .01, dCutoff = 1, dead = 2.5, jump = .25 } = {}) {
  const axis = () => ({ x: null, dx: 0 });
  const fx = axis(), fy = axis();
  let last = null, lastT = null, out = null, spike = 0;
  const alpha = (cutoff, dt) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
  function filter(f, value, dt) {
    if (f.x === null) { f.x = value; return value; }
    const dx = (value - f.x) / dt, ad = alpha(dCutoff, dt);
    f.dx = ad * dx + (1 - ad) * f.dx;
    const a = alpha(minCutoff + beta * Math.abs(f.dx), dt);
    f.x = a * value + (1 - a) * f.x;
    return f.x;
  }
  return {
    update(frame, t, width = innerWidth, height = innerHeight) {
      const hands = (frame?.hands ?? []).filter(h => h?.landmarks?.length >= 21);
      if (!hands.length) { if (lastT !== null && t - lastT > 400) { fx.x = fy.x = null; out = null; last = null; } return out; }
      const hand = hands.reduce((a, b) => (b.score ?? 0) > (a.score ?? 0) ? b : a);
      const palm = palmCentre(hand);
      const margin = (1 - area) / 2;
      const nx = Math.min(1, Math.max(0, (palm.x - margin) / area)), ny = Math.min(1, Math.max(0, (palm.y - margin) / area));
      // A single frame flung across the screen is a detector glitch: wait for a second one.
      if (last && Math.hypot(nx - last.x, ny - last.y) > jump && spike < 1) { spike++; return out; }
      spike = 0; last = { x: nx, y: ny };
      const dt = lastT === null ? 1 / 30 : Math.min(.1, Math.max(.001, (t - lastT) / 1000)); lastT = t;
      const px = filter(fx, nx * width, dt), py = filter(fy, ny * height, dt);
      if (!out || Math.hypot(px - out.x * width, py - out.y * height) > dead) out = { x: px / width, y: py / height };
      return out;
    },
    reset() { fx.x = fy.x = null; out = null; last = null; lastT = null; },
  };
}

/** Tennis: the racket arm from the pose (the wrist survives a fast swing better
 * than the hand tracker). Positions are in shoulder widths from the middle of
 * the shoulders, in the mirrored picture: x to the player's right, y up, so a
 * swing looks the same close to the camera and far from it. A swing is
 * reported once, at its fastest moment, with the capture time of that frame. */
const WRIST = { left: 15, right: 16 }, ELBOW = { left: 13, right: 14 };
export function createRacketArm({ hand = 'right', minSpeed = 4.2, cooldown = 420 } = {}) {
  let side = hand, centre = null, width = null, last = null, lastT = null, v = { x: 0, y: 0 }, peak = null, lastSwing = -1e9, out = { seen: false };
  const seen = p => p && (p.visibility ?? 0) >= .5 && p.x > -.05 && p.x < 1.05 && p.y > -.05 && p.y < 1.05;
  return {
    get hand() { return side; },
    setHand(h) { side = h; last = null; peak = null; },
    /** frame, aspect of the picture (width / height) → { seen, x, y, vx, vy, speed, raised, raisedHand, image, elbow, swing } */
    update(frame, aspect = 16 / 9) {
      const pose = frame?.pose?.landmarks, t = frame?.timestamp;
      if (!pose || t == null || t === lastT) return { ...out, swing: null };
      const pt = p => ({ x: (1 - p.x) * aspect, y: p.y });
      const [ls, rs] = [pose[11], pose[12]];
      if (seen(ls) && seen(rs)) {
        const a = pt(ls), b = pt(rs), w = Math.hypot(a.x - b.x, a.y - b.y);
        if (w > .05) { centre = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; width = width ? width * .8 + w * .2 : w; }
      }
      centre ??= { x: aspect / 2, y: .5 }; width ??= .3;
      const nose = pose[0];
      const raisedHand = ['right', 'left'].find(h => seen(pose[WRIST[h]]) && seen(nose) && pose[WRIST[h]].y < nose.y - .02) ?? null;
      const wrist = pose[WRIST[side]], elbow = pose[ELBOW[side]];
      const dt = lastT === null ? 1 / 30 : Math.min(.1, Math.max(.01, (t - lastT) / 1000)); lastT = t;
      if (!seen(wrist)) { last = null; peak = null; out = { seen: false, raisedHand }; return { ...out, swing: null }; }
      const w = pt(wrist), x = (w.x - centre.x) / width, y = -(w.y - centre.y) / width;
      if (last) { v = { x: v.x * .35 + (x - last.x) / dt * .65, y: v.y * .35 + (y - last.y) / dt * .65 }; } else v = { x: 0, y: 0 };
      last = { x, y };
      const speed = Math.hypot(v.x, v.y);
      let swing = null;
      if (speed >= minSpeed && t - lastSwing > cooldown) {
        if (!peak || speed > peak.speed) peak = { t, vx: v.x, vy: v.y, speed };
      } else if (peak) {
        swing = peak; lastSwing = peak.t; peak = null;
      }
      if (peak && speed < peak.speed * .6) { swing = peak; lastSwing = peak.t; peak = null; }
      out = { seen: true, x, y, vx: v.x, vy: v.y, speed, raised: raisedHand === side || y > 1.1, raisedHand, image: { x: 1 - wrist.x, y: wrist.y }, elbow: seen(elbow) ? { x: 1 - elbow.x, y: elbow.y } : null };
      return { ...out, swing };
    },
    reset() { last = null; lastT = null; peak = null; v = { x: 0, y: 0 }; },
  };
}

/** Both hands raised to the face or higher: the pause gesture. Both wrists
 * must be in the picture and above the nose — and above the shoulders when
 * those are in the picture. Close to a laptop the shoulders are often below
 * the frame, where the pose guesses them; every hand would be "above" them. */
export function handsAboveShoulders(frame) {
  const p = frame?.pose?.landmarks;
  if (!p) return false;
  const inFrame = i => p[i] && (p[i].visibility ?? 0) >= .5 && p[i].x > 0 && p[i].x < 1 && p[i].y > 0 && p[i].y < .97;
  if (!inFrame(15) || !inFrame(16) || !inFrame(0)) return false;
  let line = p[0].y - .02;
  if (inFrame(11) && inFrame(12)) line = Math.min(line, Math.min(p[11].y, p[12].y) - .35 * Math.abs(p[11].x - p[12].x));
  return p[15].y < line && p[16].y < line;
}

/** A 1€ filter for one value (Casiez et al.): strong smoothing while the value
 * moves slowly, almost none during fast moves, so a still hand does not shake
 * and a quick one does not lag. filter(value, dt) → smoothed value. */
export function createOneEuro({ minCutoff = 1.2, beta = 4, dCutoff = 1.5 } = {}) {
  let x = null, dx = 0;
  const alpha = (cutoff, dt) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
  return {
    filter(value, dt) {
      if (x === null) { x = value; return x; }
      const a = alpha(dCutoff, dt);
      dx = a * ((value - x) / dt) + (1 - a) * dx;
      const b = alpha(minCutoff + beta * Math.abs(dx), dt);
      x = b * value + (1 - b) * x;
      return x;
    },
    get speed() { return dx; },
    reset() { x = null; dx = 0; },
  };
}
