/** A table tennis player built from capsules, animated procedurally: steps
 * sideways, bounces in the ready stance, plays a forehand, a backhand or a
 * serve with a bat, and can hold the bat arm where the real player's hand is. Metres, feet on the ground,
 * facing +Z. Its right side is -X (it faces the viewer at +Z).
 *
 *   const p = createMannequin({ shirt: '#e8394a' });
 *   scene.add(p.group);
 *   p.swing('forehand');                         // contact comes after p.contactDelay('forehand') s
 *   p.update(dt, { speed, arm: { r, y, z } | null });
 */
import * as THREE from 'three';
import { createPaddle } from './paddle.js';

const UPPER = .3, LOWER = .28, HIP_H = .95;
const DOWN = new THREE.Vector3(0, -1, 0);

// Swing keyframes in the torso's frame: r = to the player's right, y up from the hips, z forward.
// `off` is the free hand. Contact is where u = CONTACT.
const CONTACT = .55;
const SWINGS = {
  forehand: [
    { u: 0, hand: [.48, .12, -.08], off: [-.25, .25, .3], yaw: -.65 },
    { u: .3, hand: [.5, .1, .15], off: [-.25, .22, .3], yaw: -.35 },
    { u: CONTACT, hand: [.38, .16, .4], off: [-.28, .2, .25], yaw: 0 },
    { u: 1, hand: [-.08, .52, .38], off: [-.3, .15, .1], yaw: .5 },
  ],
  backhand: [
    { u: 0, hand: [-.1, .12, .08], off: [-.3, .2, .2], yaw: .35 },
    { u: .3, hand: [-.18, .1, .2], off: [-.3, .2, .2], yaw: .2 },
    { u: CONTACT, hand: [-.12, .16, .42], off: [-.32, .18, .15], yaw: 0 },
    { u: 1, hand: [.3, .42, .38], off: [-.35, .15, .05], yaw: -.3 },
  ],
  serve: [
    { u: 0, hand: [.35, .25, .05], off: [-.1, .35, .38], yaw: -.3 },
    { u: .35, hand: [.4, .2, .1], off: [-.1, .55, .35], yaw: -.35 },
    { u: CONTACT, hand: [.22, .1, .38], off: [-.2, .3, .3], yaw: 0 },
    { u: 1, hand: [-.05, .35, .4], off: [-.25, .15, .2], yaw: .35 },
  ],
};
const DURATION = { forehand: .42, backhand: .38, serve: .9 };

function lerpKey(keys, u) {
  let i = 1; while (i < keys.length - 1 && keys[i].u < u) i++;
  const a = keys[i - 1], b = keys[i], f = Math.min(1, Math.max(0, (u - a.u) / (b.u - a.u))), s = f * f * (3 - 2 * f);
  const mix = (p, q) => p.map((v, k) => v + (q[k] - v) * s);
  return { hand: mix(a.hand, b.hand), off: mix(a.off, b.off), yaw: a.yaw + (b.yaw - a.yaw) * s, pitch: (a.pitch ?? 0) + ((b.pitch ?? 0) - (a.pitch ?? 0)) * s };
}

function capsule(radius, length, material, y) {
  const m = new THREE.Mesh(new THREE.CapsuleGeometry(radius, length, 6, 12), material);
  m.position.y = y; m.castShadow = true; return m;
}

export function createMannequin({ shirt = '#f2f2f2', shorts = '#1d2a4a', skin = '#d9a07a', hair = '#2a1d16', band = '#e8394a', lefty = false } = {}) {
  const mat = c => new THREE.MeshStandardMaterial({ color: c, roughness: .75 });
  const shirtM = mat(shirt), shortsM = mat(shorts), skinM = mat(skin), hairM = mat(hair), shoeM = mat('#f4f4f4'), bandM = mat(band), sockM = mat('#ffffff');
  const group = new THREE.Group();
  const hips = new THREE.Group(); hips.position.y = HIP_H; group.add(hips);
  const pelvis = new THREE.Mesh(new THREE.SphereGeometry(.16, 16, 12), shortsM); pelvis.scale.set(1, .75, .8); pelvis.castShadow = true; hips.add(pelvis);
  const torso = new THREE.Group(); hips.add(torso);
  const chest = capsule(.16, .3, shirtM, .3); chest.scale.set(1.12, 1, .78); torso.add(chest);
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(.05, .055, .12, 10), skinM); neck.position.y = .6; torso.add(neck);
  const head = new THREE.Group(); head.position.y = .74; torso.add(head);
  const skull = new THREE.Mesh(new THREE.SphereGeometry(.105, 18, 14), skinM); skull.scale.set(.92, 1.08, 1); skull.castShadow = true; head.add(skull);
  const hairCap = new THREE.Mesh(new THREE.SphereGeometry(.11, 18, 10, 0, Math.PI * 2, 0, Math.PI * .55), hairM); hairCap.position.set(0, .015, -.01); hairCap.scale.set(.95, 1.05, 1.05); head.add(hairCap);
  const headband = new THREE.Mesh(new THREE.CylinderGeometry(.106, .106, .028, 20, 1, true), bandM); headband.position.y = .035; head.add(headband);
  const nose = new THREE.Mesh(new THREE.SphereGeometry(.018, 8, 6), skinM); nose.position.set(0, -.01, .1); head.add(nose);

  function arm(side) {                                   // side: +1 right, -1 left
    const shoulder = new THREE.Group(); shoulder.position.set(-side * .21, .5, 0); torso.add(shoulder);
    const sleeve = new THREE.Mesh(new THREE.SphereGeometry(.058, 12, 10), shirtM); sleeve.position.x = side * .01; sleeve.castShadow = true; shoulder.add(sleeve);
    const upper = new THREE.Group(); shoulder.add(upper);
    upper.add(capsule(.048, UPPER - .08, skinM, -UPPER / 2));
    const elbow = new THREE.Group(); elbow.position.y = -UPPER; upper.add(elbow);
    elbow.add(capsule(.042, LOWER - .08, skinM, -LOWER / 2));
    const hand = new THREE.Group(); hand.position.y = -LOWER; elbow.add(hand);
    const palm = new THREE.Mesh(new THREE.SphereGeometry(.045, 10, 8), skinM); palm.scale.set(.8, 1.1, .6); hand.add(palm);
    const wristband = new THREE.Mesh(new THREE.CylinderGeometry(.046, .046, .05, 12), sockM); wristband.position.y = .03; hand.add(wristband);
    return { shoulder, upper, elbow, hand, side };
  }
  const right = arm(1), left = arm(-1);
  const racketArm = lefty ? left : right, freeArm = lefty ? right : left;
  const racket = createPaddle();
  racket.rotation.x = Math.PI;                           // handle in the hand, blade along the forearm
  racket.position.y = .02;
  racketArm.hand.add(racket);

  function leg(side) {
    const hip = new THREE.Group(); hip.position.set(-side * .1, -.04, 0); hips.add(hip);
    hip.add(capsule(.075, .3, shortsM, -.17));
    const knee = new THREE.Group(); knee.position.y = -.45; hip.add(knee);
    knee.add(capsule(.058, .32, skinM, -.2));
    const sock = new THREE.Mesh(new THREE.CylinderGeometry(.052, .05, .1, 10), sockM); sock.position.y = -.38; knee.add(sock);
    const ankle = new THREE.Group(); ankle.position.y = -.46; knee.add(ankle);
    const shoe = new THREE.Mesh(new THREE.BoxGeometry(.1, .07, .27), shoeM); shoe.position.set(0, -.01, .05); shoe.castShadow = true; ankle.add(shoe);
    return { hip, knee, ankle };
  }
  const legs = [leg(1), leg(-1)];

  // Two-bone arm: the hand goes to a target in torso space, the elbow bends out and down.
  const tmpS = new THREE.Vector3(), tmpT = new THREE.Vector3(), dir = new THREE.Vector3(), perp = new THREE.Vector3(), upDir = new THREE.Vector3(),
    elbowP = new THREE.Vector3(), foreDir = new THREE.Vector3(), q = new THREE.Quaternion(), qi = new THREE.Quaternion();
  function reach(a, r, y, z, pole) {
    tmpS.copy(a.shoulder.position);
    tmpT.set(-r, y, z);
    dir.subVectors(tmpT, tmpS);
    const d = Math.min(UPPER + LOWER - .01, Math.max(.12, dir.length())); dir.normalize();
    const a1 = Math.acos(Math.min(1, Math.max(-1, (UPPER * UPPER + d * d - LOWER * LOWER) / (2 * UPPER * d))));
    perp.copy(pole).addScaledVector(dir, -pole.dot(dir)).normalize();
    upDir.copy(dir).multiplyScalar(Math.cos(a1)).addScaledVector(perp, Math.sin(a1)).normalize();
    elbowP.copy(tmpS).addScaledVector(upDir, UPPER);
    tmpT.copy(tmpS).addScaledVector(dir, d);
    foreDir.subVectors(tmpT, elbowP).normalize();
    a.upper.quaternion.setFromUnitVectors(DOWN, upDir);
    qi.copy(a.upper.quaternion).invert();
    q.setFromUnitVectors(DOWN, foreDir.applyQuaternion(qi));
    a.elbow.quaternion.copy(q);
  }
  const POLE = side => new THREE.Vector3(side * .4, -.6, -.7);   // elbows point out, down and back
  const poleR = POLE(-1), poleL = POLE(1);

  let anim = null, runPhase = 0, time = 0, bounce = 0;
  const mirror = lefty ? -1 : 1;
  return {
    group, racket,
    get swinging() { return Boolean(anim); },
    contactDelay: kind => DURATION[kind] * CONTACT,
    /** Start a swing; `from` skips the start (0..1), used when the real swing is already under way. */
    swing(kind, { from = 0 } = {}) { anim = { kind, u: from }; },
    /** Start a swing so that its contact comes in `seconds`. */
    swingIn(kind, seconds) { anim = { kind, u: Math.max(0, CONTACT - seconds / DURATION[kind]) }; },
    update(dt, { speed = 0, arm = null, crouch = .25 } = {}) {
      time += dt;
      const run = Math.min(1, speed / 4);
      runPhase += dt * (4 + speed * 1.6);
      bounce = run < .15 ? Math.sin(time * 7) * .012 : 0;
      hips.position.y = HIP_H - crouch * .12 - run * .05 + bounce + Math.abs(Math.sin(runPhase)) * .04 * run;
      let yaw = 0, pitch = .12 + crouch * .15 + run * .1, hand, off;
      if (anim) {
        anim.u += dt / DURATION[anim.kind];
        const k = lerpKey(SWINGS[anim.kind], Math.min(1, anim.u));
        hand = k.hand; off = k.off; yaw = k.yaw; pitch += k.pitch;
        if (anim.u >= 1) anim = null;
      } else if (arm) {
        hand = [arm.r, arm.y, arm.z]; off = [-.12, .2, .38]; yaw = -arm.r * .5;
      } else {
        hand = [.22, .2, .38]; off = [-.22, .2, .32];       // ready: bat in front, free hand up
      }
      torso.rotation.set(pitch, yaw * mirror, 0);
      reach(racketArm, hand[0] * mirror, hand[1], hand[2], lefty ? poleL : poleR);
      reach(freeArm, off[0] * mirror, off[1], off[2], lefty ? poleR : poleL);
      // The racket turns with the swing so its face points where the ball goes.
      racket.rotation.set(Math.PI - (anim ? .7 : 1.0), 0, (anim ? .3 : .15) * mirror);
      // Legs: a running cycle, or knees bent in the ready stance.
      legs.forEach((l, i) => {
        const ph = runPhase + i * Math.PI, swingLeg = Math.sin(ph) * .75 * run;
        l.hip.rotation.x = -crouch * .5 - swingLeg;
        l.knee.rotation.x = crouch * .9 + Math.max(0, Math.cos(ph)) * 1.1 * run;
        l.ankle.rotation.x = -crouch * .4;
      });
    },
  };
}
