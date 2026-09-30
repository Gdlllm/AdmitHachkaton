/** «Пинг-понг» in 3D (three.js), first person: a table in a club hall, a bot
 * across the net, and the player's own bat right in front of the camera,
 * following the real hand. The whole game is putting the bat where the ball
 * is going: the ball is returned when it crosses the bat's plane close enough
 * to the blade. A fast hand hits harder; where the ball meets the blade and
 * the hand's sideways motion aim the return. Real rules: a serve bounces on
 * the own half first, one bounce on the other side, games to 11.
 *
 *   const game = createPingPong3D(container);
 *   game.update({ paddle: { x, y, vx, vy } | null, head, auto }, performance.now() / 1000);
 *   game.draw(); game.state.events → sounds and calls; game.dispose();
 */
import * as THREE from 'three';
import { createMannequin } from './mannequin.js';
import { createPaddle } from './paddle.js';
import { TABLE, TABLE_SCALE, BALL_R, NET_TOP, makeBall, stepBall, predict, crossing, aimShot, serveShot, createScore } from './pingpong-rules.js';

export const PLANE = TABLE.halfLength + .25; // the player's bat moves in the plane z = PLANE, behind the end line
export const HIT_R = .264;                  // how close to the blade centre the ball must pass
const BAT_SCALE = 1.74;                    // the player's bat is drawn larger than life, easier to aim
const SL = TABLE_SCALE.length, SW = TABLE_SCALE.width;   // shot targets were tuned on a real table
const SLOW = .8;                           // every shot 20% slower than a real one: more time to reach the ball
const BOT_PLANE = -PLANE, BOT_Z = -PLANE - .63, BOT_SPEED = 2.8, BOT_REACH = .62, STEP = 1 / 240;
const rand = (a, b) => a + Math.random() * (b - a);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const ROOM = { x: 9, z: 13, h: 6 };

// ---- scenery ------------------------------------------------------------
function canvasTexture(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t;
}
const noise = (g, w, h, n, a) => { for (let i = 0; i < n; i++) { g.fillStyle = `rgba(${Math.random() < .5 ? '255,255,255' : '0,0,0'},${Math.random() * a})`; g.fillRect(Math.random() * w, Math.random() * h, 2, 2); } };

function buildTable(scene, { x = 0, shadows = true } = {}) {
  const group = new THREE.Group(); group.position.x = x;
  const L = TABLE.halfLength * 2, W = TABLE.halfWidth * 2;
  const top = new THREE.Mesh(new THREE.BoxGeometry(W, .03, L), new THREE.MeshStandardMaterial({ color: '#173f7a', roughness: .5 }));
  top.position.y = TABLE.top - .015; top.receiveShadow = shadows; top.castShadow = shadows; group.add(top);
  const lines = canvasTexture(256, 460, (g, w, h) => {
    g.fillStyle = '#1b4a8f'; g.fillRect(0, 0, w, h); noise(g, w, h, 2500, .05);
    g.fillStyle = '#f4f4f0'; const e = 4;
    g.fillRect(0, 0, w, e); g.fillRect(0, h - e, w, e); g.fillRect(0, 0, e, h); g.fillRect(w - e, 0, e, h); g.fillRect(w / 2 - 1, 0, 2, h);
  });
  const face = new THREE.Mesh(new THREE.PlaneGeometry(W, L), new THREE.MeshStandardMaterial({ map: lines, roughness: .45 }));
  face.rotation.x = -Math.PI / 2; face.position.y = TABLE.top + .0005; face.receiveShadow = shadows; group.add(face);
  const dark = new THREE.MeshStandardMaterial({ color: '#23262d', roughness: .5, metalness: .4 });
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(.05, TABLE.top - .03, .05), dark); leg.position.set(sx * (W / 2 - .12), (TABLE.top - .03) / 2, sz * (L / 2 - .25)); leg.castShadow = shadows; group.add(leg);
  }
  const frame = new THREE.Mesh(new THREE.BoxGeometry(W - .2, .05, L - .3), dark); frame.position.y = TABLE.top - .06; group.add(frame);
  // Net: mesh between two clamps, white tape on top.
  const netW = W + 2 * TABLE.netOverhang;
  const mesh = canvasTexture(64, 64, (g) => { g.strokeStyle = 'rgba(15,15,20,1)'; g.lineWidth = 3; g.strokeRect(0, 0, 64, 64); });
  mesh.wrapS = mesh.wrapT = THREE.RepeatWrapping; mesh.repeat.set(netW / .015, TABLE.net / .015);
  const net = new THREE.Mesh(new THREE.PlaneGeometry(netW, TABLE.net), new THREE.MeshStandardMaterial({ map: mesh, transparent: true, alphaTest: .25, side: THREE.DoubleSide }));
  net.position.y = TABLE.top + TABLE.net / 2; group.add(net);
  const tape = new THREE.Mesh(new THREE.BoxGeometry(netW, .015, .006), new THREE.MeshStandardMaterial({ color: '#f4f4f0' })); tape.position.y = NET_TOP - .007; group.add(tape);
  for (const s of [-1, 1]) { const post = new THREE.Mesh(new THREE.BoxGeometry(.02, TABLE.net + .03, .04), dark); post.position.set(s * netW / 2, TABLE.top + TABLE.net / 2, 0); group.add(post); }
  scene.add(group);
  return group;
}

function buildHall(scene) {
  const floorTex = canvasTexture(512, 512, (g, w, h) => { g.fillStyle = '#86302b'; g.fillRect(0, 0, w, h); noise(g, w, h, 9000, .06); });
  floorTex.wrapS = floorTex.wrapT = THREE.RepeatWrapping; floorTex.repeat.set(8, 12);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(ROOM.x * 2, ROOM.z * 2), new THREE.MeshStandardMaterial({ map: floorTex, roughness: .7 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
  // Walls: light paint over a blue dado, high windows on the sides.
  const wallTex = canvasTexture(512, 256, (g, w, h) => {
    g.fillStyle = '#d5dbe3'; g.fillRect(0, 0, w, h); g.fillStyle = '#2a4f8f'; g.fillRect(0, h * .72, w, h * .28); g.fillStyle = '#f2f2f2'; g.fillRect(0, h * .71, w, 3); noise(g, w, h, 2000, .03);
  });
  const wallMat = new THREE.MeshStandardMaterial({ map: wallTex, roughness: .95 });
  const wall = (w, x, z, ry) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(w, ROOM.h), wallMat); m.position.set(x, ROOM.h / 2, z); m.rotation.y = ry; scene.add(m); };
  wall(ROOM.x * 2, 0, -ROOM.z, 0); wall(ROOM.x * 2, 0, ROOM.z, Math.PI); wall(ROOM.z * 2, -ROOM.x, 0, Math.PI / 2); wall(ROOM.z * 2, ROOM.x, 0, -Math.PI / 2);
  const glass = new THREE.MeshStandardMaterial({ color: '#9fb9d6', emissive: '#6f8fb3', emissiveIntensity: .5, roughness: .2 });
  for (const s of [-1, 1]) for (let z = -10; z <= 10; z += 4) { const win = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 1.2), glass); win.position.set(s * (ROOM.x - .01), 3.9, z); win.rotation.y = -s * Math.PI / 2; scene.add(win); }
  const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(ROOM.x * 2, ROOM.z * 2), new THREE.MeshStandardMaterial({ color: '#8d939d', roughness: 1, emissive: '#3a3d44', emissiveIntensity: .6 }));
  ceiling.rotation.x = Math.PI / 2; ceiling.position.y = ROOM.h; scene.add(ceiling);
  const panel = new THREE.MeshBasicMaterial({ color: '#fffaf0' });
  for (const x of [-5.2, 0, 5.2]) for (const z of [-6, -2.2, 2.2]) { const p = new THREE.Mesh(new THREE.PlaneGeometry(1.6, .5), panel); p.rotation.x = Math.PI / 2; p.position.set(x, ROOM.h - .02, z); scene.add(p); }
  // Court surrounds: low dark-blue boards around each table with the event name.
  const board = canvasTexture(512, 128, (g, w, h) => { g.fillStyle = '#1b2a57'; g.fillRect(0, 0, w, h); g.fillStyle = '#ffffff'; g.font = 'bold 54px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText('MOTION OPEN', w / 2, h / 2 + 3); });
  const boardMat = new THREE.MeshStandardMaterial({ map: board, roughness: .7 });
  const boardGeo = new THREE.BoxGeometry(1.4, .7, .04);
  for (const cx of [-5.2, 0, 5.2]) {
    for (let x = -2.1; x <= 2.1; x += 1.4) { const b = new THREE.Mesh(boardGeo, boardMat); b.position.set(cx + x, .35, -5.2); scene.add(b); }
    for (const s of [-1, 1]) for (let z = -4.2; z <= 4.2; z += 1.4) { const b = new THREE.Mesh(boardGeo, boardMat); b.position.set(cx + s * 2.6, .35, z); b.rotation.y = Math.PI / 2; scene.add(b); }
  }
  buildTable(scene, { x: -5.2, shadows: false }); buildTable(scene, { x: 5.2, shadows: false });
  // A few people on benches along the left wall and standing at the far boards.
  const spots = [];
  for (let z = -8; z <= 8; z += .7) if (Math.random() < .7) spots.push({ x: -ROOM.x + .7, y: .75, z, sit: true });
  for (let x = -3.5; x <= 3.5; x += .8) if (Math.random() < .55) spots.push({ x, y: 1.05, z: -6.2 + rand(-.3, .3), sit: false });
  const bench = new THREE.Mesh(new THREE.BoxGeometry(.45, .45, 17), new THREE.MeshStandardMaterial({ color: '#5a3d24', roughness: .8 })); bench.position.set(-ROOM.x + .7, .225, 0); scene.add(bench);
  const bodies = new THREE.InstancedMesh(new THREE.CapsuleGeometry(.19, .45, 3, 8), new THREE.MeshStandardMaterial({ roughness: .9 }), spots.length);
  const heads = new THREE.InstancedMesh(new THREE.SphereGeometry(.12, 10, 8), new THREE.MeshStandardMaterial({ roughness: .9 }), spots.length);
  const SHIRTS = ['#e8394a', '#f2f2f2', '#2c5aa0', '#f4c542', '#1f1f24', '#3f8f5a', '#f08a3c'], SKIN = ['#e7b894', '#c68a64', '#8d5a3b', '#f1c9a5'];
  const m = new THREE.Matrix4(), col = new THREE.Color();
  spots.forEach((s, i) => {
    s.phase = Math.random() * 6.28; s.y += s.sit ? .1 : 0;
    m.makeTranslation(s.x, s.y + .3, s.z); bodies.setMatrixAt(i, m); m.makeTranslation(s.x, s.y + .72, s.z); heads.setMatrixAt(i, m);
    bodies.setColorAt(i, col.set(SHIRTS[i % SHIRTS.length])); heads.setColorAt(i, col.set(SKIN[(i * 3) % SKIN.length]));
  });
  scene.add(bodies, heads);
  let excite = 0;
  return {
    cheer(a = 1) { excite = Math.max(excite, a); },
    update(dt, time) {
      if (excite < .02) return;
      excite = Math.max(0, excite - dt * .5);
      spots.forEach((s, i) => {
        const j = Math.max(0, Math.sin(time * 10 + s.phase)) * .12 * excite;
        m.makeTranslation(s.x, s.y + .3 + j, s.z); bodies.setMatrixAt(i, m); m.makeTranslation(s.x, s.y + .72 + j, s.z); heads.setMatrixAt(i, m);
      });
      bodies.instanceMatrix.needsUpdate = heads.instanceMatrix.needsUpdate = true;
    },
  };
}

/** The umpire's flip counter by the net, facing the player. */
function buildCounter(scene) {
  const c = document.createElement('canvas'); c.width = 256; c.height = 128;
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const g = c.getContext('2d');
  const set = (a, b) => {
    g.fillStyle = '#15171c'; g.fillRect(0, 0, 256, 128);
    for (const [i, v, colr] of [[0, a, '#f4f4f0'], [1, b, '#f4f4f0']]) {
      g.fillStyle = '#2b2f38'; g.fillRect(12 + i * 124, 10, 108, 108);
      g.fillStyle = colr; g.font = 'bold 84px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(String(v), 66 + i * 124, 68);
      g.fillStyle = '#000'; g.fillRect(12 + i * 124, 63, 108, 2);
    }
    tex.needsUpdate = true;
  };
  set(0, 0);
  const stand = new THREE.Group();
  const deskM = new THREE.MeshStandardMaterial({ color: '#1b2a57', roughness: .7 });
  const desk = new THREE.Mesh(new THREE.BoxGeometry(.9, .75, .5), deskM); desk.position.y = .375; stand.add(desk);
  const counter = new THREE.Mesh(new THREE.PlaneGeometry(.5, .25), new THREE.MeshBasicMaterial({ map: tex })); counter.position.set(0, .9, .05); counter.rotation.x = -.35; stand.add(counter);
  const base = new THREE.Mesh(new THREE.BoxGeometry(.52, .27, .06), new THREE.MeshStandardMaterial({ color: '#111' })); base.position.set(0, .89, .02); base.rotation.x = -.35; stand.add(base);
  stand.position.set(1.75, 0, .1); stand.rotation.y = -Math.PI / 2 + .5; scene.add(stand);
  return { set };
}

// ---- the game -------------------------------------------------------------
export function createPingPong3D(container) {
  const canvas = document.createElement('canvas');
  canvas.className = 'game-canvas game-canvas-3d';
  container.prepend(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(1.25, devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.1;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#2f3238');
  const camera = new THREE.PerspectiveCamera(58, 1, .05, 60);
  scene.add(new THREE.HemisphereLight('#fff8ee', '#5a3a30', 1.05));
  // Hall lights straight above the table: sharp ball shadow on the table, the best depth cue.
  const spot = new THREE.SpotLight('#fff6e8', 60, 12, 1.0, .6, 1.4);
  spot.position.set(0, 5.2, .3); spot.target.position.set(0, 0, 0); spot.castShadow = true; spot.shadow.mapSize.set(1024, 1024); spot.shadow.bias = -.0003;
  scene.add(spot, spot.target);
  const fill = new THREE.DirectionalLight('#dfe8ff', .6); fill.position.set(3, 6, 6); scene.add(fill);
  const hall = buildHall(scene);
  buildTable(scene);
  const counter = buildCounter(scene);

  const bot = createMannequin({ shirt: '#1e5bb8', shorts: '#15171c', skin: '#e0ac85', hair: '#3a2416', band: '#f4f4f0' });
  bot.group.position.set(0, 0, BOT_Z); scene.add(bot.group);
  const bat = createPaddle({ ghost: true }); bat.scale.setScalar(BAT_SCALE); scene.add(bat);
  const ballMesh = new THREE.Mesh(new THREE.SphereGeometry(BALL_R * 1.6, 18, 12), new THREE.MeshStandardMaterial({ color: '#ff9a2e', emissive: '#ff6a00', emissiveIntensity: .35, roughness: .5 }));
  ballMesh.castShadow = true; scene.add(ballMesh);
  const blob = new THREE.Mesh(new THREE.CircleGeometry(.03, 16), new THREE.MeshBasicMaterial({ color: '#000', transparent: true, opacity: .35, depthWrite: false }));
  blob.rotation.x = -Math.PI / 2; scene.add(blob);
  const TRAIL = 8, trailPos = new Float32Array(TRAIL * 3), trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
  const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: '#ffd29a', transparent: true, opacity: .45 })); trail.frustumCulled = false; scene.add(trail);
  // Where the ball will cross the bat's plane: a ring from the moment the bot hits.
  const aim = new THREE.Mesh(new THREE.RingGeometry(.05, .065, 32), new THREE.MeshBasicMaterial({ color: '#dcef3c', transparent: true, opacity: .0, depthWrite: false, side: THREE.DoubleSide }));
  scene.add(aim);

  const score = createScore({ target: 11, firstServer: 0 });
  const st = {
    phase: 'serve', time: null, events: [], ball: null, hitter: null, isServe: false, bounces: 0, legal: false, decided: false, nextAt: 0,
    rally: 0, paddle: { x: 0, y: 1.05, vx: 0, vy: 0 }, hitFlash: 0, bot: { x: 0, tx: 0, speed: 0 }, plan: null, cross: null, toss: null,
    stats: { hits: 0, returns: 0, chances: 0, center: 0, misses: [], best: 0, fastest: 0, aces: 0, errors: 0, botErrors: 0 },
    head: 0, lastKmh: 0,
  };
  const emit = (type, data = {}) => st.events.push({ type, ...data });

  function setupServe() {
    Object.assign(st, { phase: 'serve', ball: null, hitter: null, isServe: true, bounces: 0, legal: false, decided: false, plan: null, cross: null, toss: null, rally: 0 });
    st.nextAt = st.time + (score.server === 0 ? .9 : 1.2);
    emit('serve', { server: score.server });
  }

  function toss(by) {
    const x = by === 0 ? clamp(st.paddle.x + rand(-.22, .22), -.6 * SW, .6 * SW) : st.bot.x + .12;
    st.ball = makeBall({ x, y: by === 0 ? 1.12 : .98, z: by === 0 ? PLANE - .01 : BOT_PLANE }, { x: 0, y: by === 0 ? 1.6 : 2.2, z: 0 });
    st.toss = { by, at: st.time };
    st.phase = 'toss'; emit('toss', { by });
    if (by === 1) bot.swingIn('serve', .55);
  }

  function launch(shot, from, by, serve) {
    st.ball = makeBall(from, shot.v, shot.spin);
    Object.assign(st, { hitter: by, isServe: serve, bounces: 0, legal: false, phase: 'rally', cross: null, plan: null });
    st.rally++; st.stats.best = Math.max(st.stats.best, st.rally);
    const kmh = Math.round(Math.hypot(shot.v.x, shot.v.y, shot.v.z) * 3.6);
    if (by === 0) { st.stats.fastest = Math.max(st.stats.fastest, kmh); st.lastKmh = kmh; }
    emit('hit', { by, kmh, x: from.x, serve });
    // The receiver gets ready: where the ball will cross their plane.
    const pr = predict(st.ball, { until: 3 });
    if (by === 0) {
      const c = crossing(pr, BOT_PLANE, -1);
      if (c) { st.plan = { t: st.time + c.t, x: c.x, y: c.y, side: c.x < st.bot.x - .05 ? 'forehand' : 'backhand', swung: false }; st.bot.tx = clamp(c.x + (st.plan.side === 'forehand' ? .32 : -.25), -1.3 * SW, 1.3 * SW); }
    } else {
      const c = crossing(pr, PLANE, 1);
      // Show the ring as soon as the bot hits, when the ball is going to land on the player's half.
      const land = pr.events[serve ? 1 : 0];
      if (c) st.cross = { t: st.time + c.t, x: c.x, y: c.y, good: land?.type === 'table' && land.z > 0 };
      st.bot.tx = 0;
    }
  }

  function decide(winner, reason) {
    if (st.decided) return;
    st.decided = true; st.cross = null; st.plan = null;
    if (winner === 0) { if (reason === 'ace') st.stats.aces++; else st.stats.botErrors++; } else if (reason === 'out' || reason === 'own-side') st.stats.errors++;
    const result = score.point(winner);
    counter.set(score.state.points[0], score.state.points[1]);
    st.phase = 'dead'; st.nextAt = st.time + (result.match ? 2.5 : 1.6);
    hall.cheer(winner === 0 ? .5 + Math.min(.5, st.rally * .06) : .2);
    emit('point', { winner, reason, rally: st.rally, ...result });
    if (result.match) emit('match', { winner });
  }

  function onEvent(e) {
    emit(e.type === 'table' ? 'bounce' : e.type, { x: e.x, z: e.z });
    if (st.decided || st.hitter === null) return;
    const hitter = st.hitter, receiver = 1 - hitter, hitterSide = hitter === 0 ? 1 : -1;
    if (e.type === 'floor') { decide(st.legal ? hitter : receiver, st.legal ? (hitter === 0 ? (st.isServe ? 'ace' : 'winner') : 'missed') : (hitter === 0 ? 'out' : 'bot-out')); return; }
    // A bounce on the table.
    st.bounces++;
    const side = Math.sign(e.z);
    if (st.isServe && st.bounces === 1) { if (side !== hitterSide) decide(receiver, hitter === 0 ? 'serve-fault' : 'bot-serve-fault'); return; }
    if (!st.legal) {
      if (side === -hitterSide) st.legal = true;
      else decide(receiver, hitter === 0 ? 'own-side' : 'bot-out');
      return;
    }
    // A second bounce on the receiver's side: not returned.
    decide(hitter, hitter === 0 ? (st.isServe ? 'ace' : 'winner') : 'missed');
  }

  // ---- shots ----
  function playerHit(dx, dy) {
    const p = st.paddle, hand = Math.hypot(p.vx, p.vy), power = clamp(hand / 3, 0, 1), dist = Math.hypot(dx, dy);
    const b = st.ball.p, from = { x: b.x, y: Math.max(TABLE.top + .05, b.y), z: b.z };
    // Every return lands on the bot's half: the aim stays inside its side of the table.
    const inX = TABLE.halfWidth - .12, inZ = TABLE.halfLength - .12;
    const target = { x: clamp((-b.x * .3 + p.vx * .12 - dx * 2.2 + rand(-.12, .12)) * SW, -inX, inX), z: -clamp((.55 + power * .62 + rand(-.08, .16)) * SL, .35 * SL, inZ) };
    let shot = aimShot(from, target, (6.5 + power * 10) * SLOW, { spin: clamp(.3 + p.vy * .25, -.3, .7), clear: .05 + (1 - power) * .08 });
    const first = predict(makeBall(from, shot.v, shot.spin), { until: 1 }).events[0];
    if (first?.type !== 'table' || first.z >= 0) shot = aimShot(from, { x: 0, z: -.7 * SL }, 6.5 * SLOW, { spin: .3, clear: .1 });
    st.stats.hits++; st.stats.returns++; if (dist < .05) st.stats.center++;
    st.hitFlash = 1;
    emit('contact', { dist, power, center: dist < .05 });
    launch(shot, from, 0, false);
  }

  /** Would this shot pass the player's bat plane somewhere a hand can put the bat? */
  function reachable(shot, from) {
    const pr = predict(makeBall(from, shot.v, shot.spin), { seconds: 3 }), c = crossing(pr, PLANE, 1);
    if (!c || c.y < .85 || Math.abs(c.x) > TABLE.halfWidth || Math.abs(c.x - st.paddle.x) > 1.1) return false;
    // It must be on screen: where it passes the bat inside the bat's area, and the whole flight in view.
    const r = batRange();
    if (c.x < r.x0 || c.x > r.x1 || c.y < r.y0 || c.y > r.y1) return false;
    return pr.path.every(q => q.t > c.t || onScreen(q.x, q.y, q.z));
  }

  function botHit() {
    const plan = st.plan, b = st.ball.p, from = { x: b.x, y: Math.max(TABLE.top + .05, b.y), z: b.z };
    const level = Math.min(1, (score.state.points[0] + score.state.points[1]) / 18);
    const stretch = Math.abs(st.bot.x - plan.x) / BOT_REACH, incoming = Math.hypot(st.ball.v.x, st.ball.v.z);
    const errP = .05 + .18 * Math.max(0, stretch - .5) + (incoming > 12 ? .12 : 0) + (b.y > 1.4 || b.y < .8 ? .1 : 0);
    const speed = (rand(6, 8.5) + level * 2.5) * SLOW, spin = rand(.1, .5);
    st.stats.botShots = (st.stats.botShots ?? 0) + 1;
    if (Math.random() < errP) {
      // A real mistake, but a near one and in view: just past the end or just off the side.
      let miss = null;
      for (let k = 0; k < 8 && !miss; k++) {
        const side = Math.random() < .5 ? -1 : 1;
        const target = Math.random() < .5 ? { x: rand(-.5, .5), z: TABLE.halfLength + rand(.08, .3) } : { x: side * (TABLE.halfWidth + rand(.06, .2)), z: rand(.9, 1.8) };
        const shot = aimShot(from, target, speed * .9, { spin });
        const pr = predict(makeBall(from, shot.v, shot.spin), { seconds: 3, until: 1 });
        if (pr.path.every(q => q.z > PLANE || onScreen(q.x, q.y, q.z))) miss = shot;
      }
      if (miss) { launch(miss, from, 1, false); return; }
    }
    // A good shot, but one the player can reach: try placements (away from the bat first)
    // and keep the first whose path passes the player within reach.
    let shot = null;
    for (let i = 0; i < 10 && !shot; i++) {
      const away = i < 5 && Math.random() < .6 ? (st.paddle.x > 0 ? -1 : 1) : (Math.random() < .5 ? -1 : 1);
      const target = { x: away * rand(0, .55) * SW * (1 - i / 14), z: rand(.5, 1.15) * SL };
      const s = aimShot(from, target, speed, { spin });
      if (reachable(s, from)) shot = s;
    }
    // Nothing fit: softer and softer balls to the middle until one passes on screen.
    for (let k = 0; k < 8 && !shot; k++) {
      st.stats.fallbacks = (st.stats.fallbacks ?? 0) + 1;
      const s = aimShot(from, { x: st.paddle.x * .3, z: (.95 - k * .05) * SL }, speed * (.85 - k * .06), { spin: .45 + k * .05 });
      if (reachable(s, from) || k === 7) shot = s;
    }
    launch(shot, from, 1, false);
  }

  // ---- what the player can see ----
  // screenTop: how far down the screen the playable area starts (0..1), e.g. below the pause button.
  let screenTop = .3;
  const ndc = new THREE.Vector3(), ray = new THREE.Vector3();
  function onScreen(x, y, z) {
    camera.updateMatrixWorld(); ndc.set(x, y, z).project(camera);
    return Math.abs(ndc.x) <= .95 && ndc.y <= .95 && ndc.y >= -1;
  }
  /** The point of the bat's plane under a screen position (normalised device coordinates). */
  function onPlane(nx, ny) {
    camera.updateMatrixWorld();
    ray.set(nx, ny, .5).unproject(camera).sub(camera.position);
    const k = (PLANE - camera.position.z) / ray.z;
    return { x: camera.position.x + ray.x * k, y: camera.position.y + ray.y * k };
  }
  /** Where the bat may go: the visible part of its plane, over the table's width,
   * from the table up to `screenTop`. */
  function batRange() {
    // The camera looks down a little, so the visible width differs at the top and the bottom: take the narrower.
    const topY = 1 - 2 * screenTop, bl = onPlane(-.85, -.85), tl = onPlane(-.85, topY), br = onPlane(.85, -.85), tr = onPlane(.85, topY);
    const half = Math.min(TABLE.halfWidth, -Math.max(bl.x, tl.x), Math.min(br.x, tr.x));
    const y0 = Math.max(.8, bl.y, br.y), y1 = Math.max(y0 + .3, Math.min(tl.y, tr.y));
    return { x0: -half, x1: half, y0, y1 };
  }

  // ---- update ----
  function update({ paddle = null, head = 0, auto = false, top = null } = {}, now) {
    if (top !== null) screenTop = top;
    if (st.time === null) { st.time = now; setupServe(); }
    const dt = clamp(now - st.time, 0, .05);
    st.time = now; st.head = head;
    if (auto) {
      // Autopilot: the bat drifts to where the ball will cross, with a little error.
      const goal = st.cross ?? (st.phase === 'toss' && st.toss.by === 0 && st.ball ? { x: st.ball.p.x, y: st.ball.p.y } : { x: 0, y: 1.05 });
      const k = Math.min(1, dt * 9), nx = st.paddle.x + (goal.x + (st.autoErr ??= rand(-.08, .08)) - st.paddle.x) * k, ny = st.paddle.y + (clamp(goal.y, .8, 1.5) - st.paddle.y) * k;
      paddle = { x: nx, y: ny, vx: (nx - st.paddle.x) / Math.max(dt, 1e-3), vy: 1.2 + rand(0, 1.5) };
    }
    if (paddle) st.paddle = { ...paddle };
    st.hitFlash = Math.max(0, st.hitFlash - dt * 5);

    if (st.phase === 'serve' && now >= st.nextAt) { st.autoErr = null; toss(score.server); }
    if (st.phase === 'dead' && now >= st.nextAt) {
      if (score.state.over) st.phase = 'over';
      else { st.letServe = false; setupServe(); }
    }

    if (st.ball) {
      const steps = Math.max(1, Math.round(dt / STEP));
      for (let i = 0; i < steps; i++) {
        const b = st.ball.p, z0 = b.z, y0 = b.y;
        const e = stepBall(st.ball, dt / steps);
        if (st.phase === 'toss') {
          // Serving: the tossed ball falls onto the bat (player) or the bot strikes it.
          if (st.toss.by === 0 && st.ball.v.y < 0 && y0 >= st.paddle.y && b.y < st.paddle.y && Math.abs(b.x - st.paddle.x) < HIT_R) {
            const p = st.paddle, power = clamp(Math.hypot(p.vx, p.vy) / 3, 0, 1);
            st.stats.hits++; st.hitFlash = 1; emit('contact', { dist: Math.abs(b.x - p.x), power, serve: true });
            launch(serveShot({ x: b.x, y: Math.max(.86, b.y), z: PLANE - .05 }, 1, { x2: clamp(-p.vx * .15 + rand(-.4, .4), -.6, .6) * SW, speed: (5.5 + power * 2) * SLOW }), { x: b.x, y: Math.max(.86, b.y), z: PLANE - .05 }, 0, true);
            break;
          }
          if (st.toss.by === 0 && b.y < .78) { emit('retoss'); st.ball = null; st.phase = 'serve'; st.nextAt = now + .6; break; }
          if (st.toss.by === 1 && now - st.toss.at > .55) {
            const from = { x: b.x, y: Math.max(.9, b.y), z: BOT_PLANE + .05 };
            let serve = null;
            for (let k = 0; k < 16 && !serve; k++) {
              const sv = serveShot(from, -1, { x2: rand(-.5, .5) * SW * Math.max(0, 1 - k / 8), speed: (k < 8 ? rand(5, 6.5) : 4.5 - (k - 8) * .2) * SLOW, depth: k < 8 ? .75 : .85 });
              if (reachable(sv, from) || k === 15) serve = sv;
            }
            launch(serve, from, 1, true);
            break;
          }
          continue;
        }
        if (e) onEvent(e);
        // The ball reaches the player's plane.
        if (!st.decided && st.hitter === 1 && z0 < PLANE && b.z >= PLANE) {
          if (st.legal) {
            st.stats.chances++;
            const dx = b.x - st.paddle.x, dy = b.y - st.paddle.y;
            if (Math.hypot(dx, dy) <= HIT_R) { playerHit(dx, dy); break; }
            st.stats.misses.push({ dx, dy }); emit('miss', { dx, dy }); st.cross = null;
          }
        }
        // The ball reaches the bot.
        if (!st.decided && st.hitter === 0 && z0 > BOT_PLANE && b.z <= BOT_PLANE && st.legal && st.plan) {
          if (Math.abs(st.bot.x - b.x) <= BOT_REACH && b.y > .7 && b.y < 1.75) { botHit(); break; }
        }
      }
    }
    // The bot swings a moment before the ball arrives, and steps sideways.
    if (st.plan && !st.plan.swung && st.legal && now >= st.plan.t - bot.contactDelay(st.plan.side)) { st.plan.swung = true; bot.swing(st.plan.side); }
    const d = st.bot.tx - st.bot.x, stepLen = Math.min(Math.abs(d), BOT_SPEED * dt);
    st.bot.x += Math.sign(d) * stepLen; st.bot.speed = dt > 0 ? stepLen / dt : 0;
  }

  // ---- drawing ----
  const look = new THREE.Vector3(0, .8, -TABLE.halfLength * .9);
  let camX = 0, last = null;
  function draw() {
    const w = container.clientWidth || innerWidth, h = container.clientHeight || innerHeight;
    if (canvas.width !== Math.round(w * renderer.getPixelRatio()) || canvas.height !== Math.round(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    }
    const now = st.time ?? 0, dt = last === null ? 0 : clamp(now - last, 0, .05); last = now;
    // First person: eyes behind the end of the table, a little parallax with the real head.
    camX += (clamp(st.head * .5, -.2, .2) - camX) * .1;
    camera.position.set(camX, 1.55, PLANE + .95); camera.lookAt(look.x + camX * .3, look.y, look.z);
    // The player's bat: blade centre on the hand, handle towards the body, a punch forward on contact.
    const p = st.paddle;
    bat.position.set(p.x, p.y - .175 * BAT_SCALE, PLANE - st.hitFlash * .12);
    bat.rotation.set(-.2 - st.hitFlash * .3, 0, clamp(-p.x * .55, -.6, .6));
    bot.group.position.set(st.bot.x, 0, BOT_Z);
    bot.update(dt, { speed: st.bot.speed, crouch: .35 });
    if (st.ball) {
      const b = st.ball.p;
      ballMesh.visible = true; ballMesh.position.set(b.x, b.y, b.z);
      const overTable = Math.abs(b.x) <= TABLE.halfWidth && Math.abs(b.z) <= TABLE.halfLength && b.y >= TABLE.top;
      const ground = overTable ? TABLE.top + .002 : .002;
      blob.visible = true; blob.position.set(b.x, ground, b.z); blob.material.opacity = clamp(.45 - (b.y - ground) * .25, .08, .45);
      trailPos.copyWithin(3, 0, (TRAIL - 1) * 3); trailPos.set([b.x, b.y, b.z], 0); trailGeo.attributes.position.needsUpdate = true; trail.visible = true;
    } else { ballMesh.visible = blob.visible = trail.visible = false; trailPos.fill(0); }
    const c = st.cross;
    if (c && (c.good || st.legal) && !st.decided) { aim.position.set(c.x, c.y, PLANE - .02); aim.material.opacity = Math.min(.6, aim.material.opacity + dt * 5); }
    else aim.material.opacity = Math.max(0, aim.material.opacity - dt * 4);
    hall.update(dt, now);
    renderer.render(scene, camera);
  }

  function dispose() {
    renderer.dispose(); canvas.remove();
    scene.traverse(o => { o.geometry?.dispose?.(); o.material?.map?.dispose?.(); });
  }
  const scoreValue = () => {
    const s = score.state, x = st.stats;
    return Math.max(0, s.points[0] * 10 + (s.winner === 0 ? 100 : 0) + x.best * 5 + x.center * 3);
  };
  return { ready: Promise.resolve(), update, draw, dispose, score, scoreValue, batRange, get state() { return st; }, canvas };
}
