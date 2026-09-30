import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { measureBody, jointAngle } from '../src/motion/kinematics.js';
import { createMotionAnalyzer } from '../src/motion/index.js';
import { EXERCISES, createRepCounter } from '../src/motion/exercises.js';

// A synthetic person in MediaPipe world axes (metres, hip-centred, X right,
// Y down, Z away from the camera), facing the camera unless turned.
const rad = d => d * Math.PI / 180;
function body({ knee = 180, lean = 0, kneesIn = 0, toeReach = .18, arms = 0, stance = 1, yaw = 0, feetOut = false, knees = null } = {}) {
  const W = Array.from({ length: 33 }, () => [0, -.6, -.08]);
  const put = (i, p) => { W[i] = p; };
  const leg = (side, angle) => {
    // Thigh a and shin b below the horizontal, a + b = knee angle; hips go back,
    // shins stay steeper (at 90°: thigh 30°, shin 60°).
    const b = rad(90 - (180 - angle) / 3), a = rad(angle) - b, x = .1 * side * stance;
    const hip = [.1 * side, 0, 0];
    const kneeP = [x - kneesIn * side * .06, .45 * Math.sin(a), -.45 * Math.cos(a)];
    const ankle = [x, kneeP[1] + .45 * Math.sin(b), kneeP[2] + .45 * Math.cos(b)];
    return { hip, knee: kneeP, ankle, heel: [ankle[0], ankle[1] + .05, ankle[2] + .05], toe: [ankle[0], ankle[1] + .06, ankle[2] - toeReach] };
  };
  for (const [side, [h, k, a, heel, toe]] of [[1, [23, 25, 27, 29, 31]], [-1, [24, 26, 28, 30, 32]]]) {
    const l = leg(side, knees ? knees[side > 0 ? 0 : 1] : knee);
    put(h, l.hip); put(k, l.knee); put(a, l.ankle); put(heel, l.heel); put(toe, l.toe);
  }
  const s = rad(lean);
  for (const [side, [sh, el, wr]] of [[1, [11, 13, 15]], [-1, [12, 14, 16]]]) {
    const shoulder = [.18 * side, -.5 * Math.cos(s), -.5 * Math.sin(s)];
    const r = rad(arms);   // 0 hanging, 180 overhead, out to the side
    const elbow = [shoulder[0] + .28 * Math.sin(r) * side, shoulder[1] + .28 * Math.cos(r), shoulder[2]];
    const wrist = [shoulder[0] + .55 * Math.sin(r) * side, shoulder[1] + .55 * Math.cos(r), shoulder[2]];
    put(sh, shoulder); put(el, elbow); put(wr, wrist);
  }
  W[0] = [0, W[11][1] - .15, W[11][2] - .08];
  // Turn about the vertical axis (yaw > 0: the body's right side towards the camera).
  const c = Math.cos(rad(yaw)), n = Math.sin(rad(yaw));
  const world = W.map(([x, y, z]) => ({ x: c * x - n * z, y, z: n * x + c * z }));
  const landmarks = world.map(p => ({ x: .5 + p.x / 2.6, y: .42 + p.y / 2.6, visibility: .99, presence: .99 }));
  if (feetOut) for (const i of [27, 28, 29, 30, 31, 32]) Object.assign(landmarks[i], { y: 1.08, visibility: .2, drawConfidence: 0 });
  return { pose: { landmarks, worldLandmarks: world } };
}
const frameAt = (t, options) => ({ ...body(options), timestamp: t });

test('joint angles are 3D and view-independent; facing and side come out right', () => {
  const standing = measureBody(body());
  assert.ok(Math.abs(standing.angles.knee.left.value - 180) < 1);
  const squat = measureBody(body({ knee: 90 }));
  assert.ok(Math.abs(squat.angles.knee.left.value - 90) < 1 && Math.abs(squat.angles.knee.right.value - 90) < 1);
  const turned = measureBody(body({ knee: 90, yaw: 70 }));
  assert.ok(Math.abs(turned.angles.knee.left.value - 90) < 1, 'same squat side-on reads the same');
  assert.equal(standing.facing.view, 'front'); assert.ok(Math.abs(standing.facing.yaw) < 1);
  assert.equal(turned.facing.view, 'right', 'yaw > 0: right side towards the camera');
  assert.equal(measureBody(body({ yaw: 180 })).facing.view, 'back');
  assert.ok(Math.abs(measureBody(body({ lean: 40 })).angles.trunk.forward - 40) < 1);
  assert.ok(Math.abs(measureBody(body({ arms: 180 })).angles.shoulder.left.value - 180) < 1);
  // Close-up: hips out of the picture, the arm raise still reads.
  const close = body({ arms: 175 }); for (const i of [23, 24, 25, 26, 27, 28]) close.pose.landmarks[i].drawConfidence = 0;
  assert.ok(measureBody(close).angles.shoulder.left.confidence > .5);
  assert.ok(measureBody(body({ arms: 180 })).measures.wristAboveHead.left.value > .1);
  assert.equal(Math.round(jointAngle([1, 0, 0], [0, 0, 0], [0, 1, 0])), 90);
  assert.equal(measureBody({ pose: { landmarks: [], worldLandmarks: [] } }), null);
});

test('knees caving in and knees past the toes are measured', () => {
  const inward = measureBody(body({ knee: 95, kneesIn: 1.5 }));
  assert.ok(inward.angles.kneeDrift.left.value < -12 && inward.angles.kneeDrift.right.value < -12, JSON.stringify(inward.angles.kneeDrift));
  assert.ok(Math.abs(measureBody(body({ knee: 95 })).angles.kneeDrift.left.value) < 3);
  assert.ok(measureBody(body({ knee: 90, toeReach: .05 })).measures.kneePastToe > .1);
  assert.ok(measureBody(body({ knee: 90, toeReach: .3 })).measures.kneePastToe < .05);
});

// stand → down to `depth` → hold → up → stand; returns the analyzer and all events.
function squatOnce(analyzer, { depth = 85, from = 0, ...faults } = {}) {
  const events = [], seq = [];
  for (let i = 0; i < 8; i++) seq.push({ knee: 175 });
  for (let i = 1; i <= 8; i++) seq.push({ knee: 175 - (175 - depth) * i / 8, ...faults });
  for (let i = 0; i < 4; i++) seq.push({ knee: depth, ...faults });
  for (let i = 1; i <= 8; i++) seq.push({ knee: depth + (175 - depth) * i / 8, ...(i < 4 ? faults : {}) });
  for (let i = 0; i < 6; i++) seq.push({ knee: 175 });
  let last;
  seq.forEach((o, i) => { last = analyzer.update(frameAt(from + i * 100, o)); events.push(...last.events); });
  return { events, last, end: from + seq.length * 100 };
}

test('a clean squat counts once, with no hint', () => {
  const a = createMotionAnalyzer(); a.setExercise('squat');
  const { events } = squatOnce(a);
  const reps = events.filter(e => e.type === 'rep');
  assert.equal(reps.length, 1); assert.equal(reps[0].rep.errors.length, 0); assert.equal(reps[0].rep.full, true);
  assert.equal(events.filter(e => e.type === 'hint').length, 0);
  assert.deepEqual(a.summary().reps, 1); assert.equal(a.summary().accuracy, 100);
});

test('someone whose knees read 158° standing still gets squats counted', () => {
  const a = createMotionAnalyzer(); a.setExercise('squat');
  const events = [];
  const seq = [...Array(20).fill(158), ...[150, 140, 125, 110, 95, 90, 90, 95, 110, 125, 140, 150], ...Array(10).fill(158)];
  seq.forEach((k, i) => events.push(...a.update(frameAt(i * 100, { knee: k })).events));
  assert.equal(events.filter(e => e.type === 'rep').length, 1);
});

test('each squat mistake gives its own concrete hint and body parts', () => {
  const cases = [
    [{ depth: 125 }, 'squat-depth', 'leftKnee'],
    [{ kneesIn: 1.4 }, 'knees-in', 'leftKnee'],
    [{ lean: 60 }, 'lean-forward', 'back'],
    [{ toeReach: .02 }, 'knees-past-toes', 'leftFoot'],
    [{ knees: [85, 130] }, 'uneven', 'hips'],
  ];
  for (const [fault, code, part] of cases) {
    const a = createMotionAnalyzer(); a.setExercise('squat');
    const { events, last } = squatOnce(a, fault);
    const rep = events.find(e => e.type === 'rep');
    assert.ok(rep, `${code}: the rep still counts`);
    assert.ok(rep.rep.errors.some(e => e.code === code), `${code}: got ${rep.rep.errors.map(e => e.code)}`);
    const hint = last.hints.find(h => h.code === code);
    assert.ok(hint && hint.text.length > 20, `${code}: a concrete hint`);
    assert.ok(last.highlight.includes(part), `${code}: highlights ${part}`);
    assert.equal(a.summary().accuracy, 0);
    assert.equal(a.summary().mistakes[0].count, 1);
  }
});

test('setup comes first: feet out of the picture stops counting and asks to step back', () => {
  const a = createMotionAnalyzer(); a.setExercise('squat');
  const events = [];
  let last;
  for (let t = 0; t < 1500; t += 100) { last = a.update(frameAt(t, { knee: t < 700 ? 175 : 90, feetOut: true })); events.push(...last.events); }
  assert.equal(last.hint.code, 'step-back'); assert.equal(last.hint.kind, 'setup');
  assert.match(last.hint.text, /назад/);
  assert.equal(events.filter(e => e.type === 'rep').length, 0);
});

test('jumping jacks: open and close counts, low arms are pointed out', () => {
  const a = createMotionAnalyzer(); a.setExercise('jumpingJack');
  const events = [];
  const cycle = (arms, stance, from) => {
    const poses = [[0, 1], [arms / 2, (1 + stance) / 2], [arms, stance], [arms, stance], [arms / 2, (1 + stance) / 2], [0, 1], [0, 1]];
    poses.forEach(([ar, st], i) => { events.push(...a.update(frameAt(from + i * 120, { arms: ar, stance: st })).events); });
    return from + poses.length * 120;
  };
  let t = cycle(175, 2.3, 0);
  t = cycle(120, 2.3, t);
  const reps = events.filter(e => e.type === 'rep');
  assert.equal(reps.length, 2);
  assert.equal(reps[0].rep.errors.length, 0);
  assert.ok(reps[1].rep.errors.some(e => e.code === 'arms-low'));
});

test('gestures fire once per raise and lean', () => {
  const a = createMotionAnalyzer();
  const names = [];
  for (let t = 0; t < 2000; t += 50) names.push(...a.update(frameAt(t, { arms: t > 500 && t < 1500 ? 175 : 0 })).events.filter(e => e.type === 'gesture').map(e => e.name));
  assert.deepEqual(names.filter(n => n === 'handsUp'), ['handsUp']);
  assert.ok(names.includes('leftHandUp') && names.includes('rightHandUp'));
});

test('every exercise definition is complete and its hints are specific', () => {
  for (const [name, def] of Object.entries(EXERCISES)) {
    assert.ok(def.title && ['down', 'up'].includes(def.direction), name);
    assert.ok(def.rules.length >= 2, `${name} has at least two kinds of mistakes`);
    for (const rule of def.rules) {
      assert.ok(['during', 'extreme', 'rep'].includes(rule.when), rule.code);
      assert.ok(rule.text.includes(':') || rule.text.split(' ').length >= 3, `${rule.code}: a concrete hint`);
      assert.ok(Array.isArray(rule.parts) && rule.parts.length, `${rule.code}: highlights something`);
    }
    const counter = createRepCounter(def);
    assert.equal(counter.update(null, 0).phase, 'ready');
  }
});

// The real squat clip (front view, two full squats, the third cut at the bottom):
// landmarks recorded through the app by ml/body-sense/eval_app.mjs.
test('the recorded squat clip counts two clean reps', () => {
  const frames = JSON.parse(readFileSync(new URL('./data/squat-clip-landmarks.json', import.meta.url))).frames;
  const a = createMotionAnalyzer(); a.setExercise('squat');
  const reps = [];
  for (const f of frames) {
    if (f.pose.length < 33 || !f.world?.length) continue;
    const frame = { timestamp: f.t * 1000, pose: { landmarks: f.pose.map(([x, y, v]) => ({ x, y, visibility: v })), worldLandmarks: f.world.map(([x, y, z]) => ({ x, y, z })) } };
    reps.push(...a.update(frame).events.filter(e => e.type === 'rep'));
  }
  assert.equal(reps.length, 2);
  assert.ok(reps.every(r => r.rep.full && r.rep.errors.length === 0));
});

test('the voice counts reps and says the short part of a new hint, not too often', async () => {
  const { createVoice } = await import('../src/motion/voice.js');
  const spoken = []; let clock = 0;
  const synth = { speak: u => spoken.push(u.text), cancel() {}, getVoices: () => [] };
  const voice = createVoice({ synth, Utterance: class { constructor(text) { this.text = text; } }, now: () => clock });
  voice.events([{ type: 'rep', rep: { index: 2 } }]);
  clock = 2000; voice.events([{ type: 'hint', hint: { code: 'squat-depth', text: 'Сядь глубже: опускай таз' } }]);
  clock = 3000; voice.events([{ type: 'hint', hint: { code: 'squat-depth', text: 'Сядь глубже: опускай таз' } }]);
  assert.deepEqual(spoken, ['два', 'Сядь глубже']);
  voice.enabled = false; voice.events([{ type: 'rep', rep: { index: 3 } }]);
  assert.equal(spoken.length, 2);
  assert.equal(createVoice({ synth: null }), null);
});

test('hands-free buttons: dwell or pinch selects, jitter does not reset progress', async () => {
  const { createDwellSelector } = await import('../src/motion/dwell.js');
  const buttons = createDwellSelector([{ id: 'start', x: .4, y: .4, w: .2, h: .2 }]);
  const on = { x: .5, y: .5 }, off = { x: .9, y: .9 };
  assert.equal(buttons.update(on, 0).hover, 'start');
  assert.equal(buttons.update(off, 400).hover, 'start', 'a short slip keeps the hover');
  let r = buttons.update(on, 500);
  assert.ok(r.progress > .45 && r.progress < .55);
  assert.equal(buttons.update(on, 1000).selected, 'start');
  assert.equal(buttons.update(on, 1500).selected, null, 'no repeat right away');
  const pinch = createDwellSelector([{ id: 'next', x: 0, y: 0, w: 1, h: 1 }]);
  pinch.update({ x: .5, y: .5 }, 0);
  assert.equal(pinch.update({ x: .5, y: .5, pinch: true }, 100).selected, 'next');
  assert.equal(buttons.update(null, 5000).hover, 'start'); assert.equal(buttons.update(null, 5400).hover, null);
});

test('a tilted camera is learned from a still, straight stance before the first rep', () => {
  const a = createMotionAnalyzer(); a.setExercise('squat');
  // Everything tipped 15° about the camera X axis (a laptop screen leaning back).
  const tilt = f => { const c = Math.cos(.26), n = Math.sin(.26);
    f.pose.worldLandmarks = f.pose.worldLandmarks.map(p => ({ x: p.x, y: c * p.y - n * p.z, z: n * p.y + c * p.z })); return f; };
  let last;
  for (let t = 0; t < 1500; t += 100) last = a.update(tilt(frameAt(t, { knee: 178 })));
  assert.equal(last.vertical, 'calibrated');
  assert.ok(Math.abs(last.angles.trunk.forward) < 2, `lean ${last.angles.trunk.forward}`);
});
