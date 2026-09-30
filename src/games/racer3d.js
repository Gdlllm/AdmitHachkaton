/** «Руль» in 3D (three.js): a five-lane city road at dusk, low-poly car and
 * building models (Kenney, CC0), a chase camera. The car drives itself; the
 * player steers, opens the mouth for nitro and closes one eye to brake.
 *
 *   const race = createRacer3D(container);   // appends its own WebGL canvas
 *   await race.ready;
 *   race.update({ steer, nitro, brake }, dt); race.draw(); race.dispose();
 *
 * Units are metres. The track is a centreline with gentle curves; everything
 * on it lives in track coordinates (s along it, d across it, + to the right).
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { createSportsCar } from './sportscar.js';


export const RACE_SECONDS = 90;
const LANES = 5, LANE = 3.5, HALF = LANES * LANE / 2, WALK = 5;
const MAX = 300 / 3.6, NITRO = 1.25, BRAKE = 26;              // m/s: 300 km/h top speed, ~375 on nitro
// Seven-speed automatic: top speed of each gear (km/h), shift points in rpm.
const GEARS = [55, 95, 135, 175, 215, 258, 300].map(v => v / 3.6), IDLE = 900, REDLINE = 7600, UPSHIFT = 7300, DOWNSHIFT = 3200;
const gearFloor = g => g === 0 ? 0 : GEARS[g - 1] * .72;
const TRACK = 9000, STEP = 2;
const CAR_SCALE = 1.3, CAR_LENGTH = 4.1, CAR_WIDTH = 2;
const TRAFFIC = ['sedan', 'taxi', 'suv', 'hatchback-sports', 'police', 'suv-luxury', 'delivery'];
const BUILDINGS = 'abcdefghijklmn'.split('').map(c => `building-${c}`), TOWERS = 'abcde'.split('').map(c => `building-skyscraper-${c}`);

function buildTrack() {
  // Curvature plan (1/m): straights and gentle bends, smoothed at the ends.
  const plan = [[300, 0], [260, 1 / 420], [200, 0], [240, -1 / 380], [160, 1 / 520], [280, 0], [300, -1 / 350], [180, 1 / 400],
    [260, 0], [240, 1 / 330], [200, -1 / 460], [300, 0], [320, -1 / 400], [220, 1 / 360]];
  const n = TRACK / STEP, points = new Float32Array(n * 2), heading = new Float32Array(n), curve = new Float32Array(n);
  let s = 0, i = 0;
  const bends = [];
  while (s < TRACK) for (const [len, k] of plan) { bends.push([s, len, k]); s += len; if (s >= TRACK) break; }
  const curvatureAt = x => {
    for (const [start, len, k] of bends) if (x >= start && x < start + len) {
      const f = Math.min(1, Math.min(x - start, start + len - x) / 40);   // ease in/out over 40 m
      return k * f * f * (3 - 2 * f);
    }
    return 0;
  };
  let x = 0, z = 0, h = 0;
  for (i = 0; i < n; i++) {
    curve[i] = curvatureAt(i * STEP); heading[i] = h; points[i * 2] = x; points[i * 2 + 1] = z;
    h += curve[i] * STEP; x += Math.sin(h) * STEP; z -= Math.cos(h) * STEP;
  }
  const at = (sAlong, d = 0, out = new THREE.Vector3()) => {
    const f = Math.max(0, Math.min(n - 1.001, sAlong / STEP)), k = Math.floor(f), t = f - k;
    const px = points[k * 2] + (points[k * 2 + 2] - points[k * 2]) * t, pz = points[k * 2 + 1] + (points[k * 2 + 3] - points[k * 2 + 1]) * t;
    const hh = heading[k] + (heading[k + 1] - heading[k]) * t;
    return out.set(px + Math.cos(hh) * d, 0, pz + Math.sin(hh) * d);   // right of the direction (sin h, -cos h) is (cos h, sin h)
  };
  const headingAt = sAlong => heading[Math.max(0, Math.min(n - 1, Math.round(sAlong / STEP)))];
  const curveAt = sAlong => curve[Math.max(0, Math.min(n - 1, Math.round(sAlong / STEP)))];
  return { n, at, headingAt, curveAt };
}

function roadTexture() {
  const c = document.createElement('canvas'); c.width = 512; c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#34363b'; g.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 2200; i++) { const v = 40 + Math.random() * 30; g.fillStyle = `rgb(${v},${v},${v + 3})`; g.fillRect(Math.random() * 512, Math.random() * 512, 2, 2); }
  const lane = 512 / LANES;
  g.fillStyle = '#f1f1ea';
  g.fillRect(6, 0, 8, 512); g.fillRect(512 - 14, 0, 8, 512);                       // solid edge lines
  for (let l = 1; l < LANES; l++) for (let y = 0; y < 512; y += 128) g.fillRect(l * lane - 3, y, 6, 64);   // dashed lanes
  const t = new THREE.CanvasTexture(c); t.wrapS = THREE.ClampToEdgeWrapping; t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function ribbon(track, from, to, left, right, y, repeatEvery) {
  const rows = Math.ceil((to - from) / STEP) + 1, pos = new Float32Array(rows * 6), uv = new Float32Array(rows * 4), index = [];
  const a = new THREE.Vector3(), b = new THREE.Vector3();
  for (let r = 0; r < rows; r++) {
    const s = from + r * STEP;
    track.at(s, left, a); track.at(s, right, b);
    pos.set([a.x, y, a.z, b.x, y, b.z], r * 6);
    uv.set([0, s / repeatEvery, 1, s / repeatEvery], r * 4);
    if (r) { const o = (r - 1) * 2; index.push(o, o + 1, o + 2, o + 1, o + 3, o + 2); }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geo.setIndex(index); geo.computeVertexNormals();
  return geo;
}

export function createRacer3D(container) {
  const canvas = document.createElement('canvas');
  canvas.className = 'game-canvas game-canvas-3d';
  container.prepend(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(1.25, devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFShadowMap;

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog('#b9a3c9', 120, 520);
  {
    const c = document.createElement('canvas'); c.width = 4; c.height = 256;
    const g = c.getContext('2d'), grad = g.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0, '#2d3a6b'); grad.addColorStop(.55, '#8f7fb6'); grad.addColorStop(.8, '#f0b08a'); grad.addColorStop(1, '#f6c89f');
    g.fillStyle = grad; g.fillRect(0, 0, 4, 256);
    const sky = new THREE.CanvasTexture(c); sky.colorSpace = THREE.SRGBColorSpace;
    scene.background = sky;
  }
  const camera = new THREE.PerspectiveCamera(62, 1, .5, 900);

  scene.add(new THREE.HemisphereLight('#c8d4ff', '#4a3b33', 1.25));
  const sun = new THREE.DirectionalLight('#ffd2a6', 2.2);
  sun.castShadow = true; sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -25, right: 25, top: 25, bottom: -25, near: 1, far: 120 });
  scene.add(sun, sun.target);

  const track = buildTrack();
  // Road and sidewalks, built once along the whole track.
  const roadMat = new THREE.MeshStandardMaterial({ map: roadTexture(), roughness: .92, metalness: 0 });
  roadMat.map.repeat.set(1, 1);
  const road = new THREE.Mesh(ribbon(track, 0, TRACK - STEP, -HALF, HALF, 0, 26), roadMat); road.receiveShadow = true; scene.add(road);
  const walkMat = new THREE.MeshStandardMaterial({ color: '#8f8b86', roughness: 1 });
  for (const side of [-1, 1]) {
    const inner = side * HALF, outer = side * (HALF + WALK);
    const walk = new THREE.Mesh(ribbon(track, 0, TRACK - STEP, Math.min(inner, outer), Math.max(inner, outer), .18, 4), walkMat);
    walk.receiveShadow = true; scene.add(walk);
  }
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), new THREE.MeshStandardMaterial({ color: '#4d4a47', roughness: 1 }));
  ground.rotation.x = -Math.PI / 2; ground.position.y = -.05; scene.add(ground);

  // Lamps: a recycled pool along both sides.
  const lampGeo = new THREE.CylinderGeometry(.09, .12, 7, 8), headGeo = new THREE.BoxGeometry(.5, .18, 1.4);
  const poleMat = new THREE.MeshStandardMaterial({ color: '#2f3238', metalness: .6, roughness: .4 });
  const glowMat = new THREE.MeshStandardMaterial({ color: '#fff3d0', emissive: '#ffd89a', emissiveIntensity: 2.4 });
  const lamps = Array.from({ length: 36 }, (_, i) => {
    const g = new THREE.Group(), pole = new THREE.Mesh(lampGeo, poleMat), head = new THREE.Mesh(headGeo, glowMat);
    pole.position.y = 3.5; head.position.set(0, 7, 0); g.add(pole, head); scene.add(g);
    return { g, side: i % 2 ? 1 : -1, s: 0 };
  });

  const state = {};
  const models = {};
  const loader = new GLTFLoader();
  const load = (path, name) => new Promise((resolve, reject) => loader.load(`${path}/${name}.glb`, gltf => {
    gltf.scene.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    models[name] = gltf.scene; resolve();
  }, undefined, reject));
  const buildingPool = [], traffic = [];
  let player = null;
  const ready = Promise.all([
    ...TRAFFIC.map(n => load('/games/cars', n)),
    ...[...BUILDINGS, ...TOWERS].map(n => load('/games/city', n)),
  ]).then(() => {
    player = createSportsCar(renderer); scene.add(player);

    for (let i = 0; i < 70; i++) {
      const tower = i % 5 === 0, name = tower ? TOWERS[i % TOWERS.length] : BUILDINGS[i % BUILDINGS.length];
      const m = models[name].clone(), scale = tower ? 16 + Math.random() * 6 : 13 + Math.random() * 5;
      m.scale.setScalar(scale); scene.add(m);

      // Turned side-on to the road, the model's depth (its Z extent) points across it:
      // the facade starts past the sidewalk, with a margin for curves.
      const box = new THREE.Box3().setFromObject(models[name]), half = Math.max(Math.abs(box.min.z), Math.abs(box.max.z)) * scale;
      buildingPool.push({ m, side: i % 2 ? 1 : -1, s: 0, back: half + (tower ? 8 + Math.random() * 12 : 2 + Math.random() * 3) });
    }
    for (let i = 0; i < 9; i++) {
      const m = models[TRAFFIC[i % TRAFFIC.length]].clone(); m.scale.setScalar(CAR_SCALE); scene.add(m);
      const box = new THREE.Box3().setFromObject(models[TRAFFIC[i % TRAFFIC.length]]);
      const tail = new THREE.MeshBasicMaterial({ color: new THREE.Color('#ff1e3c').multiplyScalar(2.5) }), head = new THREE.MeshBasicMaterial({ color: new THREE.Color('#fff4d6').multiplyScalar(2) });
      for (const x of [-.42, .42]) {
        const t = new THREE.Mesh(new THREE.BoxGeometry(.22, .08, .03), tail); t.position.set(x, box.max.y * .42, box.min.z - .01); m.add(t);
        const h = new THREE.Mesh(new THREE.BoxGeometry(.2, .08, .03), head); h.position.set(x, box.max.y * .38, box.max.z + .01); m.add(h);
      }
      traffic.push({ m, s: 0, lane: 0, speed: 0 });
    }
    reset();
  });

  const laneOffset = lane => -HALF + LANE * (lane + .5);
  function placeTraffic(car, ahead) {
    car.s = state.s + ahead; car.lane = Math.floor(Math.random() * LANES); car.d = laneOffset(car.lane);
    car.speed = 16 + Math.random() * 18; car.passed = false;
  }
  function reset() {
    Object.assign(state, { s: 30, d: laneOffset(2), speed: 0, steer: 0, time: RACE_SECONDS, distance: 0, overtakes: 0, crashes: 0,
      nitroFuel: 1, nitroOn: false, braking: false, crashFlash: 0, over: false, yaw: 0, events: [], gear: 0, rpm: IDLE, shiftCut: 0, throttle: 0 });
    traffic.forEach((car, i) => placeTraffic(car, 60 + i * 55 + Math.random() * 30));
    buildingPool.forEach((b, i) => { b.s = Math.floor(i / 2) * 22 - 40; });
    lamps.forEach((l, i) => { l.s = Math.floor(i / 2) * 30; });
  }

  function update({ steer = 0, nitro = false, brake = false } = {}, dt) {
    if (!player || state.over) return;
    dt = Math.min(.05, Math.max(0, dt));
    const s = state;
    s.steer += (steer - s.steer) * Math.min(1, dt * 8);
    const nitroNow = nitro && s.nitroFuel > .02;
    if (nitroNow && !s.nitroOn) s.events.push('nitro');
    s.nitroOn = nitroNow; s.braking = brake;
    s.nitroFuel = Math.max(0, Math.min(1, s.nitroFuel + (s.nitroOn ? -.35 : .06) * dt));
    const top = MAX * (s.nitroOn ? NITRO : 1);
    // Automatic gearbox: rpm from speed within the gear; shift at the top, down when slow.
    const span = v => (v - gearFloor(s.gear)) / (GEARS[s.gear] - gearFloor(s.gear));
    s.rpm = Math.max(IDLE, Math.min(REDLINE, DOWNSHIFT * .75 + span(s.speed) * (UPSHIFT - DOWNSHIFT * .75)));
    if (s.rpm >= UPSHIFT && s.gear < GEARS.length - 1 && !brake) { s.gear++; s.shiftCut = .16; s.events.push('shift'); }
    else if (s.gear > 0 && s.speed < gearFloor(s.gear) + 1) { s.gear--; s.events.push('downshift'); }
    s.shiftCut = Math.max(0, s.shiftCut - dt);
    // Pulling power falls with each gear; none for a moment while shifting.
    const pull = (s.shiftCut > 0 ? 0 : 1) * (11.5 - s.gear * 1.15) * (s.nitroOn ? 1.8 : 1);
    s.throttle = brake || s.shiftCut > 0 ? 0 : s.speed < top ? 1 : .3;
    if (brake) s.speed -= BRAKE * dt;
    else if (s.speed < top) s.speed += pull * dt;
    else s.speed -= 4 * dt;
    s.speed = Math.max(0, s.speed);
    // Lateral: the wheel moves the car across the lanes; curves push it outwards a little.
    s.d += s.steer * 9 * (.35 + .65 * Math.min(1, s.speed / MAX)) * dt;
    s.d -= track.curveAt(s.s) * s.speed * s.speed * .08 * dt;
    const limit = HALF - CAR_WIDTH / 2;
    if (Math.abs(s.d) > limit) { s.d = Math.sign(s.d) * limit; s.speed *= 1 - 1.5 * dt; if (s.speed > 8) s.events.push('curb'); }
    s.yaw += ((s.steer * .28) - s.yaw) * Math.min(1, dt * 6);
    for (const car of traffic) {
      car.s += car.speed * dt;
      const gap = car.s - s.s;
      if (Math.abs(gap) < CAR_LENGTH && Math.abs(car.d - s.d) < CAR_WIDTH * .95 && s.speed > car.speed) {
        s.speed = car.speed * .5; s.crashes++; s.crashFlash = 1; s.events.push('crash');
        s.s = car.s - CAR_LENGTH - .3; car.passed = true;
      }
      if (!car.passed && gap < -CAR_LENGTH) { car.passed = true; s.overtakes++; s.events.push('overtake'); }
      if (gap < -40 || gap > 700) placeTraffic(car, 420 + Math.random() * 200);
    }
    s.crashFlash = Math.max(0, s.crashFlash - dt * 2);
    s.s += s.speed * dt; s.distance += s.speed * dt;
    if (s.s > TRACK - 800) {   // endless: jump back to the start, everything on the road with it
      const back = s.s - 200; s.s -= back;
      for (const list of [traffic, buildingPool, lamps]) list.forEach(o => { o.s -= back; });
    }
    s.time = Math.max(0, s.time - dt);
    if (s.time === 0) { s.over = true; s.events.push('finish'); }
  }

  const tmp = new THREE.Vector3(), look = new THREE.Vector3();
  function place(object, sAlong, d, yaw = 0, y = 0) {
    track.at(sAlong, d, object.position); object.position.y = y;
    object.rotation.set(0, -track.headingAt(sAlong) + Math.PI + yaw, 0);   // models face +Z; the road runs along (sin h, -cos h)
  }

  function draw() {
    const w = container.clientWidth || innerWidth, h = container.clientHeight || innerHeight;
    if (canvas.width !== Math.round(w * renderer.getPixelRatio()) || canvas.height !== Math.round(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    }
    if (player) {
      const s = state;
      place(player, s.s, s.d, -s.yaw);
      for (const car of traffic) place(car.m, car.s, car.d);
      for (const b of buildingPool) {
        if (b.s < s.s - 40) b.s += Math.ceil(buildingPool.length / 2) * 22;
        place(b.m, b.s, b.side * (HALF + WALK + b.back), b.side > 0 ? Math.PI / 2 : -Math.PI / 2);
      }
      for (const l of lamps) {
        if (l.s < s.s - 30) l.s += lamps.length / 2 * 30;
        place(l.g, l.s, l.side * (HALF + .8), 0);
      }
      // Chase camera, a little higher and wider on nitro.
      const fov = 62 + (s.nitroOn ? 12 : 0) + Math.min(6, s.speed / MAX * 6);
      if (Math.abs(camera.fov - fov) > .05) { camera.fov += (fov - camera.fov) * .08; camera.updateProjectionMatrix(); }
      track.at(s.s - 9, s.d * .85, tmp); tmp.y = 3.4;
      camera.position.lerp(tmp, .25);
      track.at(s.s + 14, s.d * .6, look); look.y = 1.2;
      camera.lookAt(look);
      track.at(s.s + 10, s.d, sun.target.position); sun.position.copy(sun.target.position).add(new THREE.Vector3(-30, 45, 20));
      player.userData.tail.emissiveIntensity = s.braking ? 8 : 3;
    }
    renderer.render(scene, camera);
  }

  const score = () => Math.max(0, Math.round(state.distance / 10 + state.overtakes * 25 - state.crashes * 30));
  const kmh = () => Math.round(state.speed * 3.6);
  function dispose() {
    renderer.dispose(); canvas.remove();
    scene.traverse(o => { o.geometry?.dispose?.(); });
  }
  return { ready, update, draw, reset, dispose, score, kmh, get state() { return state; }, canvas };
}
