/** Motion understanding on top of capture: per frame, what the body is doing
 * (3D joint angles, facing, framing, gestures, a hand pointer), and for the
 * chosen exercise: phase, repetitions, and concrete hints about what is wrong
 * with the setup or the technique, with the body parts to highlight.
 *
 *   const motion = createMotionAnalyzer();
 *   motion.setExercise('squat');          // or null; returns the finished summary
 *   frame.motion = motion.update(frame);   // every published frame
 *
 * See README "Понимание движения" for the frame.motion fields.
 */
import { measureBody, uprightFrom } from './kinematics.js';
import { EXERCISES, createRepCounter } from './exercises.js';

const SETUP = {
  body: { code: 'no-body', text: 'Встань перед камерой так, чтобы я видел тебя целиком', parts: [] },
  feet: { code: 'step-back', text: 'Отойди на пару шагов назад: мне нужно видеть тебя целиком, до стоп', parts: ['leftFoot', 'rightFoot'] },
  head: { code: 'head-out', text: 'Отойди назад: голова не помещается в кадр', parts: [] },
  far: { code: 'come-closer', text: 'Подойди ближе к камере', parts: [] },
  centre: { code: 'centre', text: 'Встань по центру кадра', parts: [] },
  front: { code: 'face-camera', text: 'Повернись лицом к камере', parts: ['torso'] },
};
const GESTURES = ['handsUp', 'leftHandUp', 'rightHandUp', 'leanLeft', 'leanRight', 'crouch', 'jump', 'pinch'];

export function createMotionAnalyzer({ setupHoldMs = 400, hintLingerMs = 2500, gestureHoldMs = 200 } = {}) {
  let up = [0, -1, 0], exercise = null, counter = null, startedAt = null, lastT = 0, manualUp = false;
  const stance = [];               // upright samples before the first rep, for auto-calibration
  const hints = new Map();          // code → { code, text, parts, kind, since, seen }
  const pendingSetup = new Map();   // code → first time the problem was seen
  const gestureSince = Object.fromEntries(GESTURES.map(g => [g, null]));
  const gestureOn = Object.fromEntries(GESTURES.map(g => [g, false]));
  const ankleHistory = [];

  function note(hint, kind, t, events) {
    const known = hints.get(hint.code);
    if (!known) events.push({ type: 'hint', hint: { code: hint.code, text: hint.text, parts: hint.parts, kind } });
    hints.set(hint.code, { code: hint.code, text: hint.text, parts: hint.parts, kind, since: known?.since ?? t, seen: t });
  }

  function gestures(m, frame, t, events) {
    const raw = Object.fromEntries(GESTURES.map(g => [g, false]));
    if (m) {
      const w = m.measures.wristAboveHead, lean = m.angles.trunk;
      const handUp = a => a.value !== null && a.confidence >= .5 && a.value > .05;
      raw.leftHandUp = handUp(w.left); raw.rightHandUp = handUp(w.right); raw.handsUp = raw.leftHandUp && raw.rightHandUp;
      if (lean && lean.confidence >= .5) { raw.leanLeft = lean.side > 15; raw.leanRight = lean.side < -15; }
      const knees = [m.angles.knee.left, m.angles.knee.right].filter(a => a.value !== null && a.confidence >= .5).map(a => a.value);
      raw.crouch = knees.length === 2 && (knees[0] + knees[1]) / 2 < 120;
      // A jump: both feet rise clearly above where they stood a moment ago.
      const image = frame.pose.landmarks;
      if (m.framing.fullBody && m.framing.bodyHeight) {
        const y = (image[27].y + image[28].y) / 2;
        ankleHistory.push({ t, y });
        while (ankleHistory.length && t - ankleHistory[0].t > 1500) ankleHistory.shift();
        const floor = Math.max(...ankleHistory.map(h => h.y));
        raw.jump = floor - y > .07 * m.framing.bodyHeight;
      } else ankleHistory.length = 0;
    }
    const hand = pointerOf(frame);
    raw.pinch = Boolean(hand?.pinch);
    for (const g of GESTURES) {
      if (raw[g]) gestureSince[g] ??= t; else gestureSince[g] = null;
      const hold = g === 'jump' ? 0 : g === 'pinch' ? 120 : gestureHoldMs;
      const on = gestureSince[g] !== null && t - gestureSince[g] >= hold;
      if (on && !gestureOn[g]) events.push({ type: 'gesture', name: g });
      gestureOn[g] = on;
    }
    return { gestures: { ...gestureOn }, pointer: hand };
  }

  function setup(m, def, t, events) {
    const problems = [];
    if (!m) problems.push(SETUP.body);
    else {
      const f = m.framing;
      if (def.needs.fullBody && !f.fullBody) problems.push(f.headOut && !f.feetOut ? SETUP.head : SETUP.feet);
      else if (f.bodyHeight !== null && f.bodyHeight < .35) problems.push(SETUP.far);
      if (f.centreX !== null && (f.centreX < .15 || f.centreX > .85)) problems.push(SETUP.centre);
      if (def.needs.view === 'front' && m.facing.view && m.facing.view !== 'front') problems.push(SETUP.front);
    }
    const codes = new Set(problems.map(p => p.code));
    for (const code of [...pendingSetup.keys()]) if (!codes.has(code)) pendingSetup.delete(code);
    let blocking = false;
    for (const p of problems) {
      if (!pendingSetup.has(p.code)) pendingSetup.set(p.code, t);
      if (t - pendingSetup.get(p.code) >= setupHoldMs) { note(p, 'setup', t, events); blocking = true; }
    }
    return blocking;
  }

  function summary() {
    if (!exercise) return null;
    const mistakes = Object.entries(counter.tally).sort((a, b) => b[1] - a[1]).map(([code, count]) => {
      const rule = EXERCISES[exercise].rules.find(r => r.code === code);
      return { code, count, text: rule?.text ?? code };
    });
    return { exercise, title: EXERCISES[exercise].title, reps: counter.count, goodReps: counter.good,
      accuracy: counter.count ? Math.round(100 * counter.good / counter.count) : null, mistakes,
      durationMs: startedAt === null ? 0 : lastT - startedAt };
  }

  function calibrate(frame) {
    const u = uprightFrom(frame);
    if (!u || u[1] > -Math.cos(35 * Math.PI / 180)) return false;
    up = u; manualUp = true; return true;
  }

  // A tilted laptop camera skews every lean. Standing straight and still for a
  // second before the first rep teaches the real vertical (once per exercise).
  function autoCalibrate(frame, m, t) {
    if (manualUp || !m || counter.count > 0 || stance.done) return;
    const knees = [m.angles.knee.left, m.angles.knee.right];
    const straight = m.framing.fullBody && knees.every(k => k.value !== null && k.confidence >= .5 && k.value > 155);
    const u = straight ? uprightFrom(frame) : null;
    if (!u || u[1] > -Math.cos(25 * Math.PI / 180)) { stance.length = 0; return; }
    stance.push({ t, u });
    while (stance.length && t - stance[0].t > 1000) stance.shift();
    if (t - stance[0].t < 900) return;
    const mean = [0, 1, 2].map(i => stance.reduce((a, s) => a + s.u[i], 0) / stance.length);
    const spread = Math.max(...stance.map(s => Math.hypot(s.u[0] - mean[0], s.u[1] - mean[1], s.u[2] - mean[2])));
    if (spread > .05) return;                               // not still yet
    const n = Math.hypot(...mean); up = mean.map(v => v / n); stance.done = true;
  }

  return {
    exercises: Object.fromEntries(Object.entries(EXERCISES).map(([name, d]) => [name, d.title])),
    get exercise() { return exercise; },
    summary,
    /** Takes the current posture as upright (a tilted camera). False if the person is not standing straight enough. */
    calibrate,

    /** Starts counting `name` (a key of EXERCISES) or stops with null.
     * Returns the summary of the exercise that was running, if any. */
    setExercise(name, frame = null) {
      const finished = exercise ? summary() : null;
      if (name !== null && !EXERCISES[name]) throw new RangeError(`Unknown exercise: ${name}`);
      exercise = name; counter = name ? createRepCounter(EXERCISES[name]) : null; startedAt = null;
      hints.clear(); pendingSetup.clear(); stance.length = 0; stance.done = false;
      if (frame) calibrate(frame);
      return finished;
    },

    update(frame) {
      const t = frame?.timestamp ?? performance.now();
      lastT = t;
      const events = [];
      const m = measureBody(frame, { up });
      const { gestures: g, pointer } = gestures(m, frame ?? {}, t, events);
      let state = null;
      if (exercise) {
        startedAt ??= t;
        const def = EXERCISES[exercise];
        autoCalibrate(frame, m, t);
        const blocked = setup(m, def, t, events);
        const step = counter.update(blocked ? null : m, t);
        for (const a of step.active) note(a, 'technique', t, events);
        if (step.rep) {
          events.push({ type: 'rep', exercise, rep: step.rep });
          for (const e of step.rep.errors) note(e, 'technique', t, events);
        }
        state = { name: exercise, title: def.title, phase: step.phase, signal: step.signal === null ? null : Math.round(step.signal * 100) / 100,
          reps: counter.count, goodReps: counter.good, lastRep: step.rep ?? null };
      }
      for (const [code, h] of hints) if (t - h.seen > (h.kind === 'setup' ? 300 : hintLingerMs)) hints.delete(code);
      const active = [...hints.values()].sort((a, b) => (a.kind === 'setup' ? 0 : 1) - (b.kind === 'setup' ? 0 : 1) || b.seen - a.seen)
        .map(({ code, text, parts, kind }) => ({ code, text, parts, kind }));
      return {
        t, source: m ? 'pose-3d' : null, vertical: manualUp ? 'manual' : stance.done ? 'calibrated' : 'camera',
        angles: m?.angles ?? null, measures: m?.measures ?? null, facing: m?.facing ?? null, framing: m?.framing ?? null,
        gestures: g, pointer, exercise: state,
        hints: active, hint: active[0] ?? null, highlight: [...new Set(active.flatMap(h => h.parts))],
        events,
      };
    },
  };
}
/** Index fingertip of the most confident hand, in mirrored screen coordinates
 * (0..1 over the video, as the user sees it), with a thumb-index pinch. */
export function pointerOf(frame) {
  const hands = (frame?.hands ?? []).filter(h => Array.isArray(h.landmarks) && h.landmarks.length >= 21);
  if (!hands.length) return null;
  const hand = hands.reduce((a, b) => (b.score ?? 0) > (a.score ?? 0) ? b : a);
  const p = hand.landmarks, tip = p[8], thumb = p[4], wrist = p[0], knuckle = p[9];
  const size = Math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y);
  // Clasped hands (fingers touching the other hand) are not a pinch.
  const clasped = hands.length > 1 && hands.some(other => other !== hand
    && Math.hypot(other.landmarks[0].x - wrist.x, other.landmarks[0].y - wrist.y) < 1.5 * size);
  return { x: 1 - tip.x, y: tip.y, hand: hand.handedness ?? null,
    pinch: !clasped && size > 0 && Math.hypot(tip.x - thumb.x, tip.y - thumb.y) < .35 * size };
}
