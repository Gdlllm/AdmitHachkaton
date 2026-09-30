/** The table tennis screen of the game centre: picks the bat hand, moves the
 * bat with the real hand, turns the match into sounds, calls and hints
 * («мимо: ракетка ниже мяча на 12 см»), and ends with the results.
 *
 *   const game = startPingPong(host);   // host: shared pieces of the game centre
 *   game.tick(frame, now); game.pause(on); game.finishNow(); game.dispose();
 */
import { createRacketArm, palmCentre, createOneEuro } from './controls.js';
import { createPingPong3D } from './pingpong3d.js';

const BAT_SVG = `<svg viewBox="-20 -44 40 64" aria-hidden="true">
  <ellipse cx="0" cy="-22" rx="17" ry="18" fill="#d0202a" stroke="#c89a62" stroke-width="2"/>
  <rect x="-4" y="-5" width="8" height="22" rx="3" fill="#c89a62" stroke="#7a5a34" stroke-width=".8"/>
</svg>`;

const REASON = {
  ace: 'Эйс!', winner: 'Очко!', 'bot-out': 'Бот в аут', 'bot-serve-fault': 'Бот ошибся на подаче', missed: 'Пропустил',
  out: 'Аут', 'own-side': 'Мяч на своей половине', 'serve-fault': 'Ошибка подачи',
};
const ADVICE = {
  low: 'ракетка чаще была ниже мяча — держи руку выше',
  high: 'ракетка чаще была выше мяча — опускай руку ниже',
  left: 'ракетка чаще была левее мяча — сдвигай руку вправо',
  right: 'ракетка чаще была правее мяча — сдвигай руку влево',
  out: 'мячи улетали в аут — бей мягче',
};
export const PP_HINTS = {
  noArm: 'Рука с ракеткой не видна: отодвинься, чтобы камера видела её целиком',
  pick: 'Подними руку, в которой будет ракетка',
  serve: 'Твоя подача: подставь ракетку под падающий мяч',
};
const cm = v => Math.round(Math.abs(v) * 100);
/** The biggest miss direction, in the player's words. */
function missText({ dx, dy }) {
  if (Math.abs(dy) >= Math.abs(dx)) return dy > 0 ? `Мимо: ракетка ниже мяча на ${cm(dy)} см` : `Мимо: ракетка выше мяча на ${cm(dy)} см`;
  return dx > 0 ? `Мимо: ракетка левее мяча на ${cm(dx)} см` : `Мимо: ракетка правее мяча на ${cm(dx)} см`;
}
const missKind = ({ dx, dy }) => Math.abs(dy) >= Math.abs(dx) ? (dy > 0 ? 'low' : 'high') : (dx > 0 ? 'left' : 'right');

export function startPingPong(host) {
  const { stage, ui, sound, html, showHint, setButtons, showMenu, saveRecord, video, autopilot } = host;
  const arm = createRacketArm();
  let game = null, phase = 'loading', pickSince = null, pickHand = null, pickStartedAt = null, ambience = null, noArmSince = null;
  let paused = false, pausedAt = 0, pausedTotal = 0, bat = null, head = 0, lastMiss = false, lastFrameT = null;
  const fx = createOneEuro(), fy = createOneEuro();
  const mistakes = {};
  const hud = html('div', 'tn-hud');
  hud.innerHTML = `<div class="tn-board">
      <div class="tn-row" data-i="0"><i class="tn-serve"></i><span class="tn-name">Ты</span><b class="tn-points">0</b></div>
      <div class="tn-row" data-i="1"><i class="tn-serve"></i><span class="tn-name">Бот</span><b class="tn-points">0</b></div>
    </div>
    <p class="race-legend">Ракетка — твоя рука: подставь её туда, куда летит мяч · Резкое движение — сильнее удар · Пауза: обе руки выше лица · Игра до 11</p>`;
  const rally = html('div', 'tn-radar'); rally.innerHTML = '<span>Розыгрыш</span><b>0</b><small>ударов</small>';
  const call = html('div', 'tn-call'); call.hidden = true;
  const center = html('div', 'race-center');
  ui.append(hud, rally, call, center);
  const camBat = html('div', 'cam-racket'); camBat.innerHTML = BAT_SVG; camBat.hidden = true; document.body.append(camBat);
  const rows = [...hud.querySelectorAll('.tn-row')];
  center.replaceChildren(html('p', 'race-call', 'Захожу в клуб…'));

  let callTimer = null;
  function flash(text, tone = '') {
    call.textContent = text; call.className = `tn-call ${tone}`; call.hidden = false;
    clearTimeout(callTimer); callTimer = setTimeout(() => { call.hidden = true; }, 1400);
  }

  game = createPingPong3D(stage);
  if (autopilot) window.motionPingPong = game;                // for automated checks
  phase = autopilot ? 'play' : 'pick';
  if (autopilot) center.replaceChildren();

  function scoreboard() {
    const s = game.score.state, server = game.score.server;
    rows.forEach((row, i) => { row.querySelector('.tn-points').textContent = s.points[i]; row.classList.toggle('serving', server === i && !s.over); });
    rally.querySelector('b').textContent = game.state.rally;
  }

  function onEvents() {
    for (const ev of game.state.events.splice(0)) {
      const pan = (ev.x ?? 0) / 1.2;
      if (ev.type === 'hit') sound.play(`pp-hit${1 + Math.floor(Math.random() * 3)}`, { volume: ev.by === 0 ? 1 : .6, rate: .95 + Math.random() * .1, pan });
      else if (ev.type === 'bounce') sound.play(`pp-table${1 + Math.floor(Math.random() * 3)}`, { volume: ev.z > 0 ? .8 : .45, rate: .96 + Math.random() * .08, pan });
      else if (ev.type === 'floor') sound.play('pp-floor', { volume: .4, pan });
      else if (ev.type === 'contact') {
        if (ev.power > .6) sound.play('whoosh', { volume: .25 });
        if (ev.center) flash('В центр ракетки!', 'good');
      } else if (ev.type === 'miss') {
        const kind = missKind(ev); mistakes[kind] = (mistakes[kind] ?? 0) + 1;
        flash(missText(ev), 'bad'); lastMiss = true;
      } else if (ev.type === 'retoss') flash('Подставь ракетку под мяч');
      else if (ev.type === 'point') {
        const good = ev.winner === 0;
        if (!good && ev.reason === 'out') mistakes[ev.reason] = (mistakes[ev.reason] ?? 0) + 1;
        const tail = ev.match ? '' : ev.gamePoint ? ' · Геймбол' : '';
        // A missed ball already showed by how much: keep that on screen.
        if (!(ev.reason === 'missed' && lastMiss)) flash(`${REASON[ev.reason] ?? (good ? 'Очко тебе' : 'Очко боту')}${tail}`, good ? 'good' : 'bad');
        lastMiss = false;
        // Long rallies won get the big cheer; a normal point, polite applause.
        if (good) sound.play(ev.rally >= 6 ? 'cheer' : 'applause', { volume: ev.rally >= 6 ? .55 : .35 + Math.min(.3, ev.rally * .04) });
        else sound.play('applause', { volume: .12 });
      } else if (ev.type === 'match') sound.play('cheer', { volume: ev.winner === 0 ? .9 : .35 });
    }
  }

  /** The top of the playable area, as a share of the screen height: just below the pause button. */
  function playTop() {
    const pause = document.querySelector('.game-pause');
    const r = pause && !pause.hidden ? pause.getBoundingClientRect() : null;
    return r && r.bottom > 0 ? Math.min(.45, r.bottom / innerHeight + .02) : .3;
  }

  /** The bat hand in the mirrored picture (0..1): the palm from the hand tracker
   * (steadier than the pose wrist), the one nearest the chosen arm's wrist. */
  function batHand(frame, a) {
    const palms = (frame?.hands ?? []).filter(h => h?.landmarks?.length >= 21).map(palmCentre);
    if (palms.length) {
      if (a?.seen && a.image) return palms.reduce((p, q) => Math.hypot(q.x - a.image.x, q.y - a.image.y) < Math.hypot(p.x - a.image.x, p.y - a.image.y) ? q : p);
      if (palms.length === 1) return palms[0];
      return palms.sort((p, q) => p.x - q.x)[arm.hand === 'left' ? 0 : 1];
    }
    return a?.seen ? a.image : null;
  }

  // The small bat at the real wrist in the camera window.
  function drawCamBat(a) {
    if (!video?.videoWidth || !a?.seen || !a.image) { camBat.hidden = true; return; }
    const box = video.getBoundingClientRect(), scale = Math.min(box.width / video.videoWidth, box.height / video.videoHeight);
    const w = video.videoWidth * scale, h = video.videoHeight * scale, left = box.left + (box.width - w) / 2, top = box.top + (box.height - h) / 2;
    const x = left + a.image.x * w, y = top + a.image.y * h;
    const angle = a.elbow ? Math.atan2(a.image.y * h - a.elbow.y * h, a.image.x * w - a.elbow.x * w) + Math.PI / 2 : 0;
    const size = Math.max(34, h * .24);
    camBat.hidden = false;
    camBat.style.width = `${size * .62}px`; camBat.style.height = `${size}px`;
    camBat.style.transform = `translate(${x - size * .31}px, ${y - size * .8}px) rotate(${angle}rad)`;
    camBat.style.transformOrigin = '50% 80%';
  }

  function finish() {
    phase = 'results'; ambience?.stop(); ambience = null; showHint(null);
    const s = game.score.state, x = game.state.stats, stopped = !s.over, points = game.scoreValue(), list = saveRecord('pingpong', points);
    const card = html('section', 'game-results');
    card.append(html('p', 'game-eyebrow', stopped ? 'Игра прервана' : s.winner === 0 ? 'Победа' : 'Поражение'), html('h2', null, `${s.points[0]} : ${s.points[1]}`));
    const facts = html('dl', 'game-facts');
    const add = (k, v) => facts.append(html('dt', null, k), html('dd', null, String(v)));
    const acc = x.chances ? Math.round(x.returns / x.chances * 100) : 0;
    const misses = x.misses.length, avgMiss = misses ? Math.round(x.misses.reduce((a, m) => a + Math.hypot(m.dx, m.dy), 0) / misses * 100) : 0;
    add('Очки', points); add('Рекорд', list[0].score); add('Отбито мячей', x.returns); add('Точность', `${acc} %`);
    add('В центр ракетки', x.center); add('Лучший розыгрыш', `${x.best} уд.`); add('Промахов', misses ? `${misses} (в среднем ${avgMiss} см)` : '0');
    card.append(facts);
    const top = Object.entries(mistakes).sort((a, b) => b[1] - a[1]).slice(0, 2);
    card.append(html('p', 'game-advice', top.length ? `Чаще всего ${top.map(([k]) => ADVICE[k]).join('; ')}` : 'Ни одного промаха — рука точно шла за мячом.'));
    const row = html('div', 'game-actions');
    const again = html('button', 'game-button', 'Ещё игра'); again.dataset.id = 'again'; again.type = 'button';
    const menu = html('button', 'game-button', 'В меню'); menu.dataset.id = 'menu'; menu.type = 'button';
    again.append(html('span', 'game-ring')); menu.append(html('span', 'game-ring'));
    again.addEventListener('click', () => { sound.ui.confirm(); host.restart(); });
    menu.addEventListener('click', () => { sound.ui.confirm(); showMenu(); });
    row.append(again, menu); card.append(row);
    center.replaceChildren(card);
    requestAnimationFrame(() => setButtons([again, menu]));
  }

  function tick(frame, nowMs) {
    const aspect = video?.videoWidth ? video.videoWidth / video.videoHeight : 16 / 9;
    const a = arm.update(frame, aspect);
    drawCamBat(a);
    if (!game) return;
    if (paused || phase === 'results') { game.draw(); return; }

    if (phase === 'pick') {
      center.replaceChildren(html('p', 'race-call', PP_HINTS.pick));
      // Close to a laptop a raised hand can leave the picture: after a few seconds, any visible hand will do.
      pickStartedAt ??= nowMs;
      const hand = a.raisedHand ?? (nowMs - pickStartedAt > 3500 && batHand(frame, a) ? 'right' : null);
      if (hand && hand === pickHand) {
        if (nowMs - pickSince > 500) { arm.setHand(hand); sound.ui.confirm(); phase = 'play'; center.replaceChildren(); flash(hand === 'left' ? 'Ракетка в левой руке' : 'Ракетка в правой руке'); }
      } else { pickHand = hand; pickSince = nowMs; }
      game.draw();
      return;
    }

    // The bat follows the hand as seen in the camera picture: the picture's
    // middle is stretched over the part of the bat's plane that is on screen
    // (the table's width, from the table up to the pause button), then a 1€ filter.
    const hand = batHand(frame, a);
    if (hand && frame.timestamp !== lastFrameT) {
      const dt = lastFrameT === null ? 1 / 30 : Math.min(.1, Math.max(.01, (frame.timestamp - lastFrameT) / 1000)); lastFrameT = frame.timestamp;
      const r = game.batRange(), u = Math.max(0, Math.min(1, (hand.x - .12) / .76)), v = Math.max(0, Math.min(1, (hand.y - .12) / .72));
      const tx = r.x0 + u * (r.x1 - r.x0), ty = r.y1 - v * (r.y1 - r.y0);
      bat = { x: fx.filter(tx, dt), y: fy.filter(ty, dt) };
    } else if (!hand) { lastFrameT = null; fx.reset(); fy.reset(); }
    const nose = frame?.pose?.landmarks?.[0];
    if (nose && (nose.visibility ?? 0) > .5) head += ((1 - nose.x - .5) - head) * .2;
    ambience ??= sound.loop('club-loop', { volume: .35 });
    ambience.tick();
    game.update({ paddle: bat && { ...bat, vx: fx.speed, vy: fy.speed }, head, auto: autopilot, top: playTop() }, (nowMs - pausedTotal) / 1000);
    onEvents();
    scoreboard();
    if (!autopilot && !hand) noArmSince ??= nowMs; else noArmSince = null;
    const st = game.state;
    const serving = (st.phase === 'serve' || st.phase === 'toss') && game.score.server === 0;
    showHint(noArmSince && nowMs - noArmSince > 700 ? PP_HINTS.noArm : serving && game.score.state.points[0] + game.score.state.points[1] < 2 ? PP_HINTS.serve : null);
    if (st.phase === 'over') finish();
    game.draw();
  }

  function dispose() { ambience?.stop(); game?.dispose(); game = null; camBat.remove(); clearTimeout(callTimer); }
  return {
    tick, dispose,
    get playing() { return phase === 'play' || phase === 'pick'; },
    pause(on) {
      if (on === paused) return;
      paused = on;
      if (on) { pausedAt = performance.now(); ambience?.stop(); ambience = null; } else pausedTotal += performance.now() - pausedAt;
    },
    finishNow() { if (game && phase === 'play') finish(); else showMenu(); },
  };
}
