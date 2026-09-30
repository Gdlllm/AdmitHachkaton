/** «Космический лазер» in 3D (three.js): the player sits in a station facing
 * deep space; asteroids fly in from far away. Wherever the player looks, the
 * station's two cannons burn with a laser: hold the gaze on an asteroid until
 * it glows white and bursts. Big ones split into small ones. An asteroid that
 * reaches the station takes a bite out of the shields. Waves get faster; kills
 * charge a shockwave that an open mouth sets off.
 *
 *   const game = createSpace3D(container);
 *   game.update({ gaze: { x, y } | null, mouth, auto }, performance.now() / 1000);
 *   game.draw(); game.state.events → sounds and calls; game.dispose();
 */
import * as THREE from 'three';

const FAR = 150, HIT_DIST = 3, ASSIST_PX = 110;
// Picking the rock by movement ("Pursuits", Vidal et al. 2013): the rocks drift across the
// screen, and the one whose path over the last second the gaze followed is the target, even
// when the aim sits far off it. Correlation ignores an offset or a wrong scale of the aim.
const PURSUIT_WIN = 1, EYE_LAG = .15, PURSUIT_MIN = .7, PURSUIT_MARGIN = .2;
// Pace: a webcam gaze is rough (about 50 px off), so the rocks come few and slow.
const waveSize = w => 3 + w, maxOnScreen = w => Math.min(5, 2 + Math.floor(w / 2));
const travelTime = w => Math.max(7, 14 - w * .7), spawnGap = w => Math.max(1.4, 3.2 - w * .2);
const rand = (a, b) => a + Math.random() * (b - a);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

function canvasTexture(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

/** A lumpy rock: an icosphere pushed in and out by smooth pseudo-noise. */
function rockGeometry(seed) {
  const geo = new THREE.IcosahedronGeometry(1, 3), pos = geo.attributes.position, v = new THREE.Vector3();
  const bumps = Array.from({ length: 7 }, () => ({ d: new THREE.Vector3(rand(-1, 1), rand(-1, 1), rand(-1, 1)).normalize(), a: rand(-.28, .22), w: rand(1.5, 4) }));
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).normalize();
    let r = 1 + Math.sin(v.x * 5.1 + seed) * .04 + Math.sin(v.y * 6.3 - seed) * .04;
    for (const b of bumps) r += b.a * Math.exp(-b.w * (1 - v.dot(b.d)) * 3);
    v.multiplyScalar(r); pos.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  return geo;
}

function buildSky(scene) {
  // Nebula clouds on a big inside-out sphere.
  const sky = canvasTexture(1024, 512, (g, w, h) => {
    g.fillStyle = '#03040b'; g.fillRect(0, 0, w, h);
    for (const [x, y, r, c] of [[.3, .45, .35, 'rgba(90,40,160,.55)'], [.62, .38, .28, 'rgba(20,110,170,.45)'], [.48, .62, .3, 'rgba(170,40,110,.35)'], [.8, .55, .22, 'rgba(40,160,150,.3)'], [.12, .3, .25, 'rgba(60,70,190,.35)']]) {
      const gr = g.createRadialGradient(x * w, y * h, 0, x * w, y * h, r * w); gr.addColorStop(0, c); gr.addColorStop(1, 'rgba(0,0,0,0)');
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
    }
  });
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(900, 48, 24), new THREE.MeshBasicMaterial({ map: sky, side: THREE.BackSide, depthWrite: false })));
  // Stars.
  const n = 5000, pos = new Float32Array(n * 3), col = new Float32Array(n * 3), c = new THREE.Color();
  for (let i = 0; i < n; i++) {
    const v = new THREE.Vector3(rand(-1, 1), rand(-1, 1), rand(-1, 1)).normalize().multiplyScalar(rand(500, 850));
    pos.set([v.x, v.y, v.z], i * 3); c.setHSL(rand(.55, .7), rand(0, .4), rand(.6, 1)); col.set([c.r, c.g, c.b], i * 3);
  }
  const stars = new THREE.BufferGeometry(); stars.setAttribute('position', new THREE.BufferAttribute(pos, 3)); stars.setAttribute('color', new THREE.BufferAttribute(col, 3));
  scene.add(new THREE.Points(stars, new THREE.PointsMaterial({ size: 1.6, vertexColors: true, sizeAttenuation: false, depthWrite: false })));
  // A planet low on the left and a sun high on the right.
  const planetTex = canvasTexture(512, 256, (g, w, h) => {
    const gr = g.createLinearGradient(0, 0, 0, h); gr.addColorStop(0, '#2c5a9a'); gr.addColorStop(.5, '#3f8fb0'); gr.addColorStop(1, '#1d3a6a');
    g.fillStyle = gr; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 40; i++) { g.fillStyle = `rgba(255,255,255,${rand(.05, .2)})`; g.fillRect(0, rand(0, h), w, rand(2, 8)); }
    for (let i = 0; i < 25; i++) { g.fillStyle = `rgba(60,120,70,${rand(.3, .6)})`; g.beginPath(); g.ellipse(rand(0, w), rand(h * .2, h * .8), rand(15, 50), rand(8, 25), 0, 0, 7); g.fill(); }
  });
  const planet = new THREE.Mesh(new THREE.SphereGeometry(70, 48, 32), new THREE.MeshStandardMaterial({ map: planetTex, roughness: 1 }));
  planet.position.set(-190, -95, -380); scene.add(planet);
  const glowTex = canvasTexture(128, 128, (g, w) => { const gr = g.createRadialGradient(w / 2, w / 2, w * .3, w / 2, w / 2, w / 2); gr.addColorStop(0, 'rgba(120,190,255,.5)'); gr.addColorStop(1, 'rgba(120,190,255,0)'); g.fillStyle = gr; g.fillRect(0, 0, w, w); });
  const atmo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false })); atmo.scale.setScalar(200); atmo.position.copy(planet.position); scene.add(atmo);
  const sunTex = canvasTexture(128, 128, (g, w) => { const gr = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2); gr.addColorStop(0, 'rgba(255,250,230,1)'); gr.addColorStop(.15, 'rgba(255,220,160,.8)'); gr.addColorStop(1, 'rgba(255,160,80,0)'); g.fillStyle = gr; g.fillRect(0, 0, w, w); });
  const sun = new THREE.Sprite(new THREE.SpriteMaterial({ map: sunTex, blending: THREE.AdditiveBlending, depthWrite: false })); sun.scale.setScalar(110); sun.position.set(320, 210, -600); scene.add(sun);
  return { planet, glowTex };
}

/** Space dust streaming past: the feeling of moving forward. */
function buildDust(scene) {
  const n = 400, pos = new Float32Array(n * 3);
  const reset = (i, z = rand(-FAR, 0)) => pos.set([rand(-60, 60), rand(-40, 40), z], i * 3);
  for (let i = 0; i < n; i++) reset(i);
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(geo, new THREE.PointsMaterial({ size: .25, color: '#9fb6d9', transparent: true, opacity: .55, depthWrite: false })));
  return dt => { for (let i = 0; i < n; i++) { pos[i * 3 + 2] += 22 * dt; if (pos[i * 3 + 2] > 2) reset(i, -FAR); } geo.attributes.position.needsUpdate = true; };
}

/** The station's two laser cannons, low in the corners of the view. */
function buildCannons(camera) {
  const metal = new THREE.MeshStandardMaterial({ color: '#9aa6b8', metalness: .5, roughness: .35 });
  const hull = new THREE.MeshStandardMaterial({ color: '#4a5366', metalness: .35, roughness: .5 });
  const trim = new THREE.MeshBasicMaterial({ color: '#39d8ff' });
  const glow = new THREE.MeshBasicMaterial({ color: '#7cf2ff' });
  return [-1, 1].map(side => {
    const g = new THREE.Group();
    const body = new THREE.Mesh(new THREE.CylinderGeometry(.16, .2, .55, 16), hull); body.rotation.x = Math.PI / 2; g.add(body);
    const stripe = new THREE.Mesh(new THREE.TorusGeometry(.17, .012, 6, 24), trim); stripe.position.z = -.12; g.add(stripe);
    const barrel = new THREE.Mesh(new THREE.CylinderGeometry(.045, .06, .75, 14), metal); barrel.rotation.x = Math.PI / 2; barrel.position.z = -.6; g.add(barrel);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(.06, .015, 8, 20), glow); ring.position.z = -.98; g.add(ring);
    g.scale.setScalar(.75); g.position.set(side * 1.75, -1.12, -2.6); g.rotation.y = side * .16; g.rotation.x = .1;
    camera.add(g);
    return { g, muzzle: ring, side };
  });
}

export function createSpace3D(container) {
  const canvas = document.createElement('canvas');
  canvas.className = 'game-canvas game-canvas-3d';
  container.prepend(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(1.25, devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.1;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(70, 1, .1, 2000); scene.add(camera);
  scene.add(new THREE.AmbientLight('#6d7aa8', .7));
  const sunLight = new THREE.DirectionalLight('#fff0dc', 2.6); sunLight.position.set(320, 210, -100); scene.add(sunLight);
  const cockpitLight = new THREE.PointLight('#7cf2ff', 4, 8); cockpitLight.position.set(0, -.6, -1.5); camera.add(cockpitLight);
  const sky = buildSky(scene);
  const dust = buildDust(scene);
  const cannons = buildCannons(camera);
  const rocks = Array.from({ length: 6 }, (_, i) => rockGeometry(i * 1.7));

  // Laser beams: a bright core and a soft glow, from each muzzle to the target.
  const beamGeo = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true); beamGeo.translate(0, .5, 0); beamGeo.rotateX(Math.PI / 2);
  const beams = cannons.map(() => {
    const core = new THREE.Mesh(beamGeo, new THREE.MeshBasicMaterial({ color: '#e8ffff', transparent: true, opacity: .95, blending: THREE.AdditiveBlending, depthWrite: false }));
    const halo = new THREE.Mesh(beamGeo, new THREE.MeshBasicMaterial({ color: '#39d8ff', transparent: true, opacity: .35, blending: THREE.AdditiveBlending, depthWrite: false }));
    core.visible = halo.visible = false; scene.add(core, halo);
    return { core, halo };
  });
  const sparkTex = canvasTexture(64, 64, (g, w) => { const gr = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2); gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(.3, 'rgba(255,200,120,.8)'); gr.addColorStop(1, 'rgba(255,120,40,0)'); g.fillStyle = gr; g.fillRect(0, 0, w, w); });
  const spark = new THREE.Sprite(new THREE.SpriteMaterial({ map: sparkTex, blending: THREE.AdditiveBlending, depthWrite: false })); spark.visible = false; scene.add(spark);

  // Explosions: a burst of glowing particles, a flash and a few flying chunks.
  const booms = [];
  function explode(p, size) {
    const n = Math.round(60 + size * 25), pos = new Float32Array(n * 3), vel = [];
    for (let i = 0; i < n; i++) { pos.set([p.x, p.y, p.z], i * 3); vel.push(new THREE.Vector3(rand(-1, 1), rand(-1, 1), rand(-1, 1)).normalize().multiplyScalar(rand(4, 16) * (.6 + size * .25))); }
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ map: sparkTex, size: .9 + size * .35, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, color: '#ffc27a' }));
    const flash = new THREE.Sprite(new THREE.SpriteMaterial({ map: sparkTex, blending: THREE.AdditiveBlending, depthWrite: false })); flash.position.copy(p); flash.scale.setScalar(size * 6);
    const light = new THREE.PointLight('#ffb070', 30 * size, 40); light.position.copy(p);
    const chunks = Array.from({ length: 5 }, () => {
      const m = new THREE.Mesh(rocks[Math.floor(rand(0, 6))], new THREE.MeshStandardMaterial({ color: '#6b6058', roughness: 1, flatShading: true, emissive: '#ff5a10', emissiveIntensity: .8 }));
      m.scale.setScalar(size * rand(.15, .3)); m.position.copy(p); m.userData.v = new THREE.Vector3(rand(-1, 1), rand(-1, 1), rand(-1, 1)).normalize().multiplyScalar(rand(3, 9)); scene.add(m); return m;
    });
    scene.add(pts, flash, light);
    booms.push({ pts, vel, flash, light, chunks, age: 0 });
  }

  const st = {
    phase: 'intro', time: null, events: [], asteroids: [], wave: 0, waveLeft: 0, nextSpawn: 0, introUntil: 0,
    shields: 100, score: 0, combo: 0, lastKill: -10, charge: 0, gaze: null, target: null, shake: 0, mouthSince: null, gazeHist: [], how: null,
    stats: { kills: 0, big: 0, bestCombo: 0, lost: 0, shockwaves: 0, seconds: 0, burnTime: 0, aimTime: 0 },
  };
  const emit = (type, data = {}) => st.events.push({ type, ...data });
  let width = 1, height = 1;

  function startWave() {
    st.wave++; st.waveLeft = waveSize(st.wave); st.phase = 'intro'; st.introUntil = st.time + 2.2; st.nextSpawn = st.introUntil;
    emit('wave', { wave: st.wave });
  }

  // Asteroids come at the station while drifting across the screen along gentle arcs: each
  // keeps a spot on screen (np, in normalised device coordinates) that moves, and a distance.
  const tmp = new THREE.Vector3();
  let zoneBag = [], rockId = 0;
  const placeAlong = a => { tmp.set(a.np.x, a.np.y, .5).unproject(camera).sub(camera.position).normalize(); a.mesh.position.copy(camera.position).addScaledVector(tmp, a.dist); };
  function spawn({ size = Math.random() < .2 ? rand(3.6, 5) : rand(1.6, 2.6), at = null, speed = null } = {}) {
    // Evenly over the whole screen: the screen is cut into 3 × 3 parts and each rock takes the
    // next part from a shuffled bag, so every side gets its share; only the camera window
    // (top right), the score (top left) and the cannons' muzzles are avoided within a part.
    if (!zoneBag.length) zoneBag = [0, 1, 2, 3, 4, 5, 6, 7, 8].sort(() => Math.random() - .5);
    const zone = zoneBag.pop(), zx = zone % 3, zy = Math.floor(zone / 3);
    let nx, ny;
    for (let tries = 0; tries < 30; tries++) {
      nx = -.84 + (zx + rand(.1, .9)) * (1.68 / 3); ny = .78 - (zy + rand(.1, .9)) * (1.56 / 3);
      const hidden = (nx > .5 && ny > .5) || (nx < -.6 && ny > .6) || (Math.abs(nx) > .55 && ny < -.55);
      if (!hidden) break;
    }
    let p;
    if (at) { p = at.clone(); tmp.copy(at).project(camera); nx = tmp.x; ny = tmp.y; }
    else { tmp.set(nx, ny, .5).unproject(camera).sub(camera.position).normalize(); p = tmp.clone().multiplyScalar(-FAR / tmp.z); }
    const travel = travelTime(st.wave);
    const mesh = new THREE.Mesh(rocks[Math.floor(rand(0, 6))], new THREE.MeshStandardMaterial({ color: new THREE.Color().setHSL(rand(.05, .1), rand(.08, .2), rand(.32, .45)), roughness: 1, flatShading: true, emissive: '#ff4a00', emissiveIntensity: 0 }));
    mesh.scale.set(size * rand(.85, 1.15), size * rand(.8, 1.1), size * rand(.85, 1.15)); mesh.position.copy(p);
    scene.add(mesh);
    const heading = rand(0, Math.PI * 2), drift = rand(.07, .13) * (1 + st.wave * .04);
    const a = { id: ++rockId, mesh, size, heat: 0, big: size > 3.3, v: speed ?? (p.length() / travel), dist: p.length(),
      np: { x: nx, y: ny }, nv: { x: Math.cos(heading) * drift, y: Math.sin(heading) * drift }, turn: rand(-.5, .5),
      spin: new THREE.Vector3(rand(-1, 1), rand(-1, 1), rand(-1, 1)).multiplyScalar(.8), screen: { x: 0, y: 0, r: 0 }, hist: [], burn: [] };
    st.asteroids.push(a);
    return a;
  }

  function destroy(a, { shock = false } = {}) {
    st.asteroids.splice(st.asteroids.indexOf(a), 1); scene.remove(a.mesh); a.mesh.material.dispose();
    explode(a.mesh.position, a.size);
    const combo = st.time - st.lastKill < 2.5 ? st.combo + 1 : 0;
    st.combo = combo; st.lastKill = st.time; st.stats.bestCombo = Math.max(st.stats.bestCombo, combo);
    const points = Math.round((a.big ? 250 : 100) * (1 + combo * .25) * (shock ? .5 : 1));
    st.score += points; st.stats.kills++; if (a.big) st.stats.big++;
    if (!shock) st.charge = Math.min(1, st.charge + (a.big ? .14 : .085));
    emit('kill', { id: a.id, big: a.big, points, combo, x: a.screen.x, y: a.screen.y, burn: shock ? [] : a.burn });
    // A big rock breaks into small ones that carry on.
    if (a.big && !shock && st.wave >= 3) for (let i = 0; i < 2; i++) {
      const piece = spawn({ size: rand(1.3, 1.8), at: a.mesh.position.clone().add(new THREE.Vector3(rand(-2, 2), rand(-2, 2), 0)), speed: a.v * 1.15 });
      piece.v = a.v * 1.15;
    }
  }

  // ---- pursuit matching ----
  const valueAt = (hist, t, key) => {
    if (!hist.length || t < hist[0].t) return null;
    let i = hist.length - 1; while (i > 0 && hist[i - 1].t > t) i--;
    const b = hist[i], a = hist[i - 1] ?? b; if (b.t <= t || a === b) return b[key];
    const f = (t - a.t) / (b.t - a.t); return a[key] + (b[key] - a[key]) * f;
  };
  const stats = v => { const m = v.reduce((s, x) => s + x, 0) / v.length; return { m, v: v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length }; };
  const corr = (p, q) => { const a = stats(p), b = stats(q); let c = 0; p.forEach((x, i) => { c += (x - a.m) * (q[i] - b.m); }); return c / p.length / Math.sqrt(a.v * b.v + 1e-12); };
  /** How well the gaze followed each rock over the last second (−1..1), best first. */
  function pursuitScores(now) {
    const gh = st.gazeHist;
    if (!gh.length || now - gh[0].t < PURSUIT_WIN) return [];
    const ts = Array.from({ length: 20 }, (_, i) => now - PURSUIT_WIN + i * PURSUIT_WIN / 19);
    const gx = ts.map(t => valueAt(gh, t, 'x')), gy = ts.map(t => valueAt(gh, t, 'y'));
    if (stats(gx).v + stats(gy).v < 1e-5) return [];
    const out = [];
    for (const a of st.asteroids) {
      if (!a.screen.visible || !a.hist.length || a.hist[0].t > now - PURSUIT_WIN - EYE_LAG) continue;
      const rx = ts.map(t => valueAt(a.hist, t - EYE_LAG, 'x')), ry = ts.map(t => valueAt(a.hist, t - EYE_LAG, 'y'));
      const vx = stats(rx).v, vy = stats(ry).v;
      if (vx + vy < 2e-5) continue;                             // it hardly moved: nothing to follow
      out.push({ a, score: (vx * corr(gx, rx) + vy * corr(gy, ry)) / (vx + vy) });
    }
    return out.sort((p, q) => q.score - p.score);
  }

  function project(a) {
    tmp.copy(a.mesh.position).project(camera);
    const r = a.size * 1.05 / Math.max(.5, a.mesh.position.length()) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * height / 2;
    a.screen = { x: (tmp.x + 1) / 2 * width, y: (1 - tmp.y) / 2 * height, r, visible: tmp.z < 1 };
  }

  // ---- update ----
  function update({ gaze = null, mouth = false, auto = false, autoOffset = null } = {}, now) {
    if (st.time === null) { st.time = now; startWave(); }
    const dt = clamp(now - st.time, 0, .05); st.time = now;
    if (st.phase === 'over') return;
    st.stats.seconds += dt;
    for (const a of st.asteroids) project(a);
    // Autopilot: the gaze drifts to the nearest threat.
    if (auto) {
      const goal = st.asteroids.filter(a => a.screen.visible).sort((a, b) => b.mesh.position.z - a.mesh.position.z)[0];
      const g = st.autoGaze ?? { x: .5, y: .5 }, k = Math.min(1, dt * 5);
      st.autoGaze = goal ? { x: g.x + (goal.screen.x / width - g.x) * k, y: g.y + (goal.screen.y / height - g.y) * k } : g;
      gaze = autoOffset ? { x: st.autoGaze.x + autoOffset.x, y: st.autoGaze.y + autoOffset.y } : st.autoGaze;
    }
    st.gaze = gaze;
    if (gaze) { st.gazeHist.push({ t: now, x: gaze.x, y: gaze.y }); while (st.gazeHist.length && st.gazeHist[0].t < now - 2.5) st.gazeHist.shift(); }
    if (st.phase === 'intro' && now >= st.introUntil) st.phase = 'play';
    if (st.phase === 'play' && st.waveLeft > 0 && now >= st.nextSpawn && st.asteroids.length < maxOnScreen(st.wave)) {
      spawn(); st.waveLeft--; st.nextSpawn = now + spawnGap(st.wave) * rand(.8, 1.25);
    }
    if (st.phase === 'play' && st.waveLeft === 0 && st.asteroids.length === 0) { st.shields = Math.min(100, st.shields + 20); startWave(); }

    // Lock on. First by movement: the rock the gaze has been following. Otherwise the rock
    // under the gaze, with a little help around each rock. The current target is kept while
    // it is still followed or near, so the laser does not flicker between rocks.
    let target = null, how = null;
    const scores = gaze ? pursuitScores(now) : [];
    const nearOf = () => {
      if (!gaze) return null;
      const gx = gaze.x * width, gy = gaze.y * height;
      let best = Infinity, pick = null;
      for (const a of st.asteroids) {
        if (!a.screen.visible) continue;
        // Up and down the webcam gaze is rougher (the camera sits above the screen): twice the room vertically.
        const d = Math.hypot(a.screen.x - gx, (a.screen.y - gy) / 2) - a.screen.r;
        if (d < ASSIST_PX && d < best) { best = d; pick = a; }
      }
      return pick;
    };
    const [first, second] = scores;
    if (first && first.score > PURSUIT_MIN && first.score - (second?.score ?? 0) > PURSUIT_MARGIN) { target = first.a; how = 'pursuit'; }
    else {
      const keep = st.target && st.asteroids.includes(st.target) && (scores.find(s => s.a === st.target)?.score ?? 0) > .45;
      target = keep ? st.target : nearOf(); how = target ? (keep ? 'kept' : 'near') : null;
    }
    st.how = how;
    if (target) target.burn.push({ t: now, x: target.screen.x / width, y: target.screen.y / height, how });
    if (target !== st.target) { emit(target ? 'lock' : 'unlock'); st.target = target; }
    if (target) st.stats.burnTime += dt;
    if (st.asteroids.length) st.stats.aimTime += dt;
    for (const a of [...st.asteroids]) {
      if (a === target) a.heat += dt / (.35 + a.size * .12);
      else a.heat = Math.max(0, a.heat - dt * .12);         // a gaze that slips off keeps most of the heat
      a.mesh.material.emissiveIntensity = a.heat * a.heat * 3;
      a.mesh.material.emissive.setHSL(.07 - a.heat * .05, 1, .45 + a.heat * .3);
      if (a.heat >= 1) { destroy(a); continue; }
      // Closer, and along its arc across the screen, bouncing off the edges and the camera window.
      a.dist -= a.v * dt;
      const turn = a.turn * dt, c = Math.cos(turn), sn = Math.sin(turn);
      a.nv = { x: a.nv.x * c - a.nv.y * sn, y: a.nv.x * sn + a.nv.y * c };
      a.np.x += a.nv.x * dt; a.np.y += a.nv.y * dt;
      if (Math.abs(a.np.x) > .86) { a.nv.x = -Math.sign(a.np.x) * Math.abs(a.nv.x); a.np.x = Math.sign(a.np.x) * .86; }
      if (Math.abs(a.np.y) > .8) { a.nv.y = -Math.sign(a.np.y) * Math.abs(a.nv.y); a.np.y = Math.sign(a.np.y) * .8; }
      if (a.np.x > .5 && a.np.y > .5) { a.nv.x = -Math.abs(a.nv.x); a.nv.y = -Math.abs(a.nv.y); }
      placeAlong(a);
      project(a);
      a.hist.push({ t: now, x: a.screen.x / width, y: a.screen.y / height }); while (a.hist.length && a.hist[0].t < now - 2.5) a.hist.shift();
      a.mesh.rotation.x += a.spin.x * dt; a.mesh.rotation.y += a.spin.y * dt; a.mesh.rotation.z += a.spin.z * dt;
      if (a.dist < HIT_DIST + a.size) {
        st.asteroids.splice(st.asteroids.indexOf(a), 1); scene.remove(a.mesh); a.mesh.material.dispose();
        const damage = a.big ? 18 : 8;
        st.shields = Math.max(0, st.shields - damage); st.shake = 1; st.combo = 0; st.stats.lost++;
        emit('hit', { damage, shields: st.shields });
        if (st.shields === 0) { st.phase = 'over'; emit('over'); }
      }
    }
    // Shockwave: an open mouth, held a moment, when fully charged.
    if (mouth && st.charge >= 1) { st.mouthSince ??= now; if (now - st.mouthSince > .25) shockwave(); } else st.mouthSince = null;
    st.shake = Math.max(0, st.shake - dt * 2.5);
  }

  function shockwave() {
    st.charge = 0; st.mouthSince = null; st.stats.shockwaves++;
    const hit = st.asteroids.filter(a => a.screen.visible);
    emit('shockwave', { count: hit.length });
    for (const a of hit) destroy(a, { shock: true });
    st.shake = .6;
  }

  // ---- drawing ----
  let last = null;
  const muzzle = new THREE.Vector3(), aim = new THREE.Vector3(), dir = new THREE.Vector3();
  function draw() {
    const w = container.clientWidth || innerWidth, h = container.clientHeight || innerHeight;
    if (canvas.width !== Math.round(w * renderer.getPixelRatio()) || canvas.height !== Math.round(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    }
    width = w; height = h;
    const now = st.time ?? 0, dt = last === null ? 0 : clamp(now - last, 0, .05); last = now;
    // The view stays still (a moving view would drag the rocks away from the gaze); it only trembles on a hit.
    camera.rotation.set(Math.sin(now * 60) * .012 * st.shake, Math.cos(now * 47) * .012 * st.shake, 0);
    camera.updateMatrixWorld();
    dust(dt);
    sky.planet.rotation.y += dt * .01;
    const t = st.target;
    beams.forEach((b, i) => {
      const on = Boolean(t) && st.phase !== 'over';
      b.core.visible = b.halo.visible = on;
      cannons[i].muzzle.material.color.set(on ? '#ffffff' : '#7cf2ff');
      if (!on) return;
      cannons[i].muzzle.getWorldPosition(muzzle);
      aim.copy(t.mesh.position).addScaledVector(dir.copy(camera.position).sub(t.mesh.position).normalize(), t.size * .8);
      const len = muzzle.distanceTo(aim);
      for (const [m, r] of [[b.core, .012], [b.halo, .035 + Math.sin(now * 40 + i) * .008]]) { m.position.copy(muzzle); m.lookAt(aim); m.scale.set(r * (1 + len * .012), r * (1 + len * .012), len); }
      b.core.material.opacity = .8 + Math.random() * .2;
    });
    spark.visible = Boolean(t);
    if (t) { spark.position.copy(aim); spark.scale.setScalar(t.size * (1.2 + Math.random() * .5)); }
    for (const b of [...booms]) {
      b.age += dt;
      const pos = b.pts.geometry.attributes.position;
      b.vel.forEach((v, i) => { pos.setXYZ(i, pos.getX(i) + v.x * dt, pos.getY(i) + v.y * dt, pos.getZ(i) + v.z * dt); v.multiplyScalar(1 - dt * .8); });
      pos.needsUpdate = true;
      b.pts.material.opacity = Math.max(0, 1 - b.age / 1.3);
      b.flash.material.opacity = Math.max(0, 1 - b.age * 4); b.flash.scale.multiplyScalar(1 + dt * 3);
      b.light.intensity = Math.max(0, b.light.intensity - dt * 120);
      for (const c of b.chunks) { c.position.addScaledVector(c.userData.v, dt); c.rotation.x += dt * 3; c.material.emissiveIntensity = Math.max(0, .8 - b.age); c.material.opacity = 1; }
      if (b.age > 1.4) {
        scene.remove(b.pts, b.flash, b.light, ...b.chunks); b.pts.geometry.dispose(); b.pts.material.dispose(); b.flash.material.dispose(); b.chunks.forEach(c => c.material.dispose());
        booms.splice(booms.indexOf(b), 1);
      }
    }
    renderer.render(scene, camera);
  }

  function dispose() {
    renderer.dispose(); canvas.remove();
    scene.traverse(o => { if (!rocks.includes(o.geometry)) o.geometry?.dispose?.(); o.material?.map?.dispose?.(); });
    rocks.forEach(g => g.dispose());
  }
  return { ready: Promise.resolve(), update, draw, dispose, get state() { return st; }, canvas };
}
