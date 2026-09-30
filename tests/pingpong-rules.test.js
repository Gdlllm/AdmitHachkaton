import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TABLE, makeBall, predict, crossing, aimShot, serveShot, createScore } from '../src/games/pingpong-rules.js';

test('an aimed shot clears the net and lands on the target', () => {
  for (const [from, target, speed] of [[{ x: .3, y: 1, z: 1.6 }, { x: -.5, z: -1 }, 9], [{ x: -.4, y: .95, z: -1.6 }, { x: .4, z: .9 }, 7], [{ x: 0, y: 1.2, z: 1.6 }, { x: .6, z: -1.2 }, 16]]) {
    const shot = aimShot(from, target, speed);
    const e = predict(makeBall(from, shot.v, shot.spin)).events[0];
    assert.equal(e.type, 'table');
    assert.ok(Math.hypot(e.x - target.x, e.z - target.z) < .06, `landed ${e.x.toFixed(2)},${e.z.toFixed(2)}`);
  }
});

test('the net never stops a ball: a low shot flies through the net line', () => {
  const from = { x: 0, y: .8, z: TABLE.halfLength + .25 };
  const events = predict(makeBall(from, { x: 0, y: 0, z: -9 }, 0)).events;
  assert.ok(events.every(e => e.type !== 'net'));
});

test('a serve bounces on the own half, then on the other', () => {
  for (const side of [1, -1]) for (let i = 0; i < 5; i++) {
    const from = { x: (Math.random() - .5) * .8, y: .95, z: side * (TABLE.halfLength + .2) };
    const shot = serveShot(from, side, { x2: (Math.random() - .5) });
    const [a, b] = predict(makeBall(from, shot.v, shot.spin), { until: 2 }).events;
    assert.equal(a.type, 'table'); assert.equal(Math.sign(a.z), side);
    assert.equal(b.type, 'table'); assert.equal(Math.sign(b.z), -side);
  }
});

test('the ball comes up after the bounce and crosses the player plane at hitting height', () => {
  const from = { x: 0, y: .95, z: -TABLE.halfLength - .25 }, shot = aimShot(from, { x: .3, z: 1.6 }, 8);
  const c = crossing(predict(makeBall(from, shot.v, shot.spin)), TABLE.halfLength + .25, 1);
  assert.ok(c && c.y > TABLE.top && c.y < 1.5, `crosses at ${c?.y.toFixed(2)} m`);
});

test('scoring: to 11, two clear, serve every two points, every point from 10:10', () => {
  const sc = createScore();
  const servers = [];
  for (let i = 0; i < 6; i++) { servers.push(sc.server); sc.point(i % 2); }
  assert.deepEqual(servers, [0, 0, 1, 1, 0, 0]);
  const g = createScore();
  for (let i = 0; i < 10; i++) { g.point(0); g.point(1); }
  const a = g.server; g.point(0); const b = g.server; g.point(1); const c = g.server;
  assert.ok(a !== b && b !== c);
  assert.equal(g.point(0).match, false); assert.equal(g.point(0).match, true);
  assert.equal(g.state.winner, 0);
});
