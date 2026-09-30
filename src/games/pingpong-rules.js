/** Table tennis without the pictures: the table, a 40 mm ball with air drag
 * and spin, bounces off the table and the floor, aiming a shot, a
 * legal serve, and scoring (to 11, two clear, serve changes every two points,
 * every point from 10:10). The net is only drawn: shots are aimed over it and
 * nothing ever stops in it (a game rule here). Metres and seconds; the net is at z = 0, the
 * player's end is z > 0, x grows to the player's right, y up from the floor.
 */
// The table is bigger than a real one (2.74 × 1.525 m): 1.75× longer and 1.25× wider, so the
// ball flies about twice as long and there is time to put the bat in its way.
export const TABLE_SCALE = { length: 1.75, width: 1.25 };
export const TABLE = { halfLength: 1.37 * TABLE_SCALE.length, halfWidth: .7625 * TABLE_SCALE.width, top: .76, net: .1525, netOverhang: .1525 };
export const BALL_R = .02, G = 9.81;
export const NET_TOP = TABLE.top + TABLE.net;
// Drag k = ½·ρ·Cd·A / m for a 2.7 g, 40 mm ball is about 0.14 per metre; half of that here,
// so the ball does not die over the longer table.
const DRAG = .07, TABLE_BOUNCE = .8, TABLE_KEEP = .88, FLOOR_BOUNCE = .7, FLOOR_KEEP = .7, MAGNUS = 3.5;

export function makeBall(p, v, spin = 0) { return { p: { ...p }, v: { ...v }, spin }; }
export const onTable = (x, z) => Math.abs(x) <= TABLE.halfWidth && Math.abs(z) <= TABLE.halfLength;

/** Advance by dt; returns { type: 'table' | 'floor', x, z } or null. */
export function stepBall(b, dt) {
  const { p, v } = b, z0 = p.z, y0 = p.y;
  const speed = Math.hypot(v.x, v.y, v.z), drag = DRAG * speed * dt;
  v.x -= v.x * drag; v.y -= v.y * drag; v.z -= v.z * drag;
  v.y -= (G + MAGNUS * b.spin) * dt;                   // topspin dips the ball, backspin floats it
  p.x += v.x * dt; p.y += v.y * dt; p.z += v.z * dt;
  if (v.y < 0 && p.y <= TABLE.top + BALL_R && y0 > TABLE.top + BALL_R - .001 && onTable(p.x, p.z)) {
    p.y = TABLE.top + BALL_R; v.y = -v.y * TABLE_BOUNCE;
    const keep = TABLE_KEEP * (1 + .08 * b.spin);
    v.x *= keep; v.z *= keep; b.spin *= .6;
    return { type: 'table', x: p.x, z: p.z };
  }
  if (v.y < 0 && p.y <= BALL_R) {
    p.y = BALL_R; v.y = -v.y * FLOOR_BOUNCE; v.x *= FLOOR_KEEP; v.z *= FLOOR_KEEP; b.spin = 0;
    if (v.y < .3) v.y = 0;
    return { type: 'floor', x: p.x, z: p.z };
  }
  return null;
}

/** Fly a copy forward: the events on the way and the path (every step). */
export function predict(ball, { seconds = 3, dt = 1 / 240, until = 3 } = {}) {
  const b = makeBall(ball.p, ball.v, ball.spin), events = [], path = [];
  for (let t = dt; t <= seconds; t += dt) {
    const e = stepBall(b, dt);
    path.push({ t, x: b.p.x, y: b.p.y, z: b.p.z, vz: b.v.z });
    if (e) { events.push({ ...e, t }); if (events.length >= until || e.type === 'floor') break; }
  }
  return { events, path };
}

/** When and where the ball crosses the plane z = zPlane (heading `dir`: +1 towards +z). */
export function crossing(prediction, zPlane, dir) {
  const path = prediction.path;
  for (let i = 1; i < path.length; i++) {
    const a = path[i - 1], b = path[i];
    if ((dir > 0 ? a.z < zPlane && b.z >= zPlane : a.z > zPlane && b.z <= zPlane)) {
      const f = (zPlane - a.z) / (b.z - a.z);
      return { t: a.t + (b.t - a.t) * f, x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f };
    }
  }
  return null;
}

/** Where the ball first comes down through table height (on the table or past it). */
function landing(from, v, spin) {
  const b = makeBall(from, v, spin), dt = 1 / 150;
  let yNet = Infinity;
  for (let t = 0; t < 3; t += dt) {
    const z0 = b.p.z, e = stepBall(b, dt);
    if (z0 !== 0 && Math.sign(z0) !== Math.sign(b.p.z) && yNet === Infinity) yNet = b.p.y;
    if (e?.type === 'table' || (b.v.y < 0 && b.p.y <= TABLE.top + BALL_R)) return { yNet: yNet === Infinity ? -Infinity : yNet, x: b.p.x, z: b.p.z, t };
  }
  return { yNet, x: b.p.x, z: b.p.z };
}

/** Launch velocity from `from` so the ball first lands at `target` ({x, z}),
 * leaving at about `speed` m/s and clearing the net by `clear` metres.
 * Solved numerically with drag and spin. */
export function aimShot(from, target, speed, { spin = .3, clear = .06, overNet = true } = {}) {
  const dx = target.x - from.x, dz = target.z - from.z, d = Math.hypot(dx, dz), ux = dx / d, uz = dz / d;
  const velocity = (vh, vy) => ({ x: ux * vh, y: vy, z: uz * vh });
  const rangeFit = vy => {
    let lo = .5, hi = 25;
    for (let i = 0; i < 17; i++) {
      const mid = (lo + hi) / 2, f = landing(from, velocity(mid, vy), spin);
      const reach = Math.hypot(f.x - from.x, f.z - from.z);
      if (reach < d) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  };
  // Start low and flat for the pace, lift until the ball clears the net.
  let vy = -1.5, vh = rangeFit(vy), best = null;
  for (let i = 0; i < 40; i++) {
    const f = landing(from, velocity(vh, vy), spin);
    if (!overNet || f.yNet >= NET_TOP + clear) { best = velocity(vh, vy); if (Math.hypot(vh, vy) <= speed * 1.05) break; }
    vy += .2; vh = rangeFit(vy);
    if (best && Math.hypot(vh, vy) < speed * .8) break;
  }
  return { v: best ?? velocity(vh, vy), spin };
}

/** A serve: first bounce on the server's own half (side: +1 player, −1 bot), then over the net onto the other half. */
const rand = (a, b) => a + Math.random() * (b - a);
/** A serve: first bounce on the server's own half (side: +1 player, −1 bot),
 * then over the net onto the other half near `x2`, `depth` of the way to the
 * far end. A small grid search over launch speeds keeps it robust. */
export function serveShot(from, side, { x2 = 0, depth = .75, speed = 5, spin = .2 } = {}) {
  const want = -side * TABLE.halfLength * depth;
  let best = null;
  for (let vh = 2.5; vh <= 12; vh += .3) for (let vy = -3.5; vy <= 1.5; vy += .25) {
    const dz = -side, dx = (x2 - from.x) / Math.max(.5, Math.abs(want - from.z));
    const n = Math.hypot(dx, 1), v = { x: vh * dx / n, y: vy, z: vh * dz / n };
    const [a, b] = predict(makeBall(from, v, spin), { until: 2, seconds: 2, dt: 1 / 120 }).events;
    // With room to spare: not near the net, not near the ends or the sides.
    const inside = e => e?.type === 'table' && Math.abs(e.z) >= .3 && Math.abs(e.z) <= TABLE.halfLength - .15 && Math.abs(e.x) <= TABLE.halfWidth - .1;
    if (!inside(a) || Math.sign(a.z) !== side || !inside(b) || Math.sign(b.z) !== -side) continue;
    const cost = Math.abs(b.z - want) + .05 * Math.abs(Math.hypot(vh, vy) - speed);
    if (!best || cost < best.cost) best = { cost, v };
  }
  return { v: best?.v ?? { x: 0, y: -1, z: -side * 5 }, spin };
}

/** Score for one game to `target` (11), two clear. Side 0 is the player. */
export function createScore({ target = 11, firstServer = 0 } = {}) {
  const s = { points: [0, 0], over: false, winner: null };
  const server = () => {
    const total = s.points[0] + s.points[1];
    const deuce = s.points[0] >= target - 1 && s.points[1] >= target - 1;
    const turns = deuce ? (target - 1) + (total - 2 * (target - 1)) : Math.floor(total / 2);
    return (firstServer + turns) % 2;
  };
  return {
    state: s,
    get server() { return server(); },
    point(winner) {
      if (s.over) return {};
      s.points[winner]++;
      const [a, b] = [s.points[winner], s.points[1 - winner]];
      if (a >= target && a - b >= 2) { s.over = true; s.winner = winner; return { match: true }; }
      return { match: false, gamePoint: Math.max(...s.points) >= target - 1 && s.points[0] !== s.points[1] };
    },
  };
}
