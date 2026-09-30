/** Game centre (?games=1): pick a game by pointing at it, play it with the body,
 * hands and face. The camera with the tracked skeleton sits in a corner.
 *
 *   mountGames(window.motionCapture)
 */
import './games.css';
import { createDwellSelector } from '../motion/dwell.js';
import { readWheel, readFace, createWheelCoach, steering, createHandCursor, handsAboveShoulders, WHEEL_HINTS } from './controls.js';
import { createSound } from './sound.js';

const GAMES = [
  { id: 'racer', title: 'Руль', text: 'Городская гонка на 90 секунд. Рули двумя руками, открой рот для нитро, закрой один глаз, чтобы тормознуть.', ready: true },
  { id: 'pingpong', title: 'Пинг-понг', text: 'Настольный теннис в клубе. Ракетка — твоя рука: подставь её туда, куда летит мяч. Игра до 11.', ready: true },
  { id: 'gaze', title: 'Взгляд', text: 'Целься глазами, моргай, чтобы стрелять.', ready: false },
];

const STORE = 'motion-games-records';
function records() { try { return JSON.parse(localStorage.getItem(STORE)) ?? {}; } catch { return {}; } }
function saveRecord(game, score) {
  const all = records(), list = [...(all[game] ?? []), { score, date: new Date().toISOString().slice(0, 10) }].sort((a, b) => b.score - a.score).slice(0, 5);
  all[game] = list;
  try { localStorage.setItem(STORE, JSON.stringify(all)); } catch { /* private mode: records stay for this visit only */ }
  return list;
}
const best = game => records()[game]?.[0]?.score ?? null;
const html = (tag, className, text) => { const el = document.createElement(tag); if (className) el.className = className; if (text != null) el.textContent = text; return el; };

// A real-looking steering wheel: leather rim with stitching, three brushed spokes, hub, top marker.
const WHEEL_SVG = `<svg class="real-wheel" viewBox="-50 -50 100 100" aria-hidden="true">
  <defs>
    <radialGradient id="rimLeather" r="0.6"><stop offset="0.78" stop-color="#3a3a3f"/><stop offset="0.9" stop-color="#1c1c20"/><stop offset="1" stop-color="#0c0c0e"/></radialGradient>
    <linearGradient id="spokeMetal" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#d9dce2"/><stop offset=".5" stop-color="#8a8f99"/><stop offset="1" stop-color="#4b4f57"/></linearGradient>
    <radialGradient id="hubGrad" r="0.7"><stop offset="0" stop-color="#3b3f47"/><stop offset="1" stop-color="#15171b"/></radialGradient>
  </defs>
  <circle r="42" fill="none" stroke="url(#rimLeather)" stroke-width="11"/>
  <circle r="42" fill="none" stroke="#6b6b72" stroke-width=".7" stroke-dasharray="2 2.2" opacity=".8"/>
  <path d="M-6.5 -44 h13 v8 h-13z" fill="#e8394a"/>
  <path d="M-37 2 C-24 -4 -14 -6 -9 -3 L-9 6 C-18 8 -28 10 -36 12z" fill="url(#spokeMetal)"/>
  <path d="M37 2 C24 -4 14 -6 9 -3 L9 6 C18 8 28 10 36 12z" fill="url(#spokeMetal)"/>
  <path d="M-7 9 L7 9 L10 37 L-10 37z" fill="url(#spokeMetal)"/>
  <circle r="13" fill="url(#hubGrad)" stroke="#9aa0aa" stroke-width="1.2"/>
  <circle r="5" fill="none" stroke="#c9ccd3" stroke-width="1.4"/>
</svg>`;

// Analog speedometer, 0–400 km/h over 240°, the gear in the middle.
const SPEEDO_MAX = 400;
function speedometerSvg() {
  const ticks = [];
  for (let v = 0; v <= SPEEDO_MAX; v += 20) {
    const a = (-210 + v / SPEEDO_MAX * 240) * Math.PI / 180, major = v % 50 === 0 || v % 100 === 0;
    const r1 = major ? 36 : 39, r2 = 44;
    ticks.push(`<line x1="${(Math.cos(a) * r1).toFixed(2)}" y1="${(Math.sin(a) * r1).toFixed(2)}" x2="${(Math.cos(a) * r2).toFixed(2)}" y2="${(Math.sin(a) * r2).toFixed(2)}" class="${v > 300 ? 'red' : ''} ${major ? 'major' : ''}"/>`);
    if (v % 100 === 0) ticks.push(`<text x="${(Math.cos(a) * 28).toFixed(2)}" y="${(Math.sin(a) * 28 + 3).toFixed(2)}">${v}</text>`);
  }
  return `<svg viewBox="-50 -50 100 100" aria-hidden="true"><circle r="48" class="face"/>${ticks.join('')}
    <line x1="0" y1="0" x2="-38" y2="0" class="needle"/><circle r="4.5" class="cap"/>
    <text y="-11" class="gear">N</text><text y="25" class="value">0</text><text y="33" class="unit">км/ч</text></svg>`;
}

export function mountGames(capture) {
  document.body.classList.add('games');
  const stage = html('div', 'game-stage');
  const menuCanvas = html('canvas', 'game-canvas');
  const ui = html('div', 'game-ui');
  const cursor = html('div', 'game-cursor');
  const hint = html('div', 'game-hint'); hint.hidden = true;
  const camWheel = html('div', 'cam-wheel'); camWheel.innerHTML = WHEEL_SVG; camWheel.hidden = true;
  const soundNote = html('div', 'game-sound', 'Кликни один раз, чтобы включить звук'); soundNote.hidden = true;
  // Pause: a button in the corner, both hands above the shoulders, or Esc.
  const pauseButton = html('button', 'game-pause'); pauseButton.type = 'button'; pauseButton.hidden = true;
  pauseButton.innerHTML = '<b>❚❚ Пауза</b><small>обе руки выше лица · Esc</small><span class="game-ring"></span>';
  const pauseLayer = html('div', 'game-pause-layer'); pauseLayer.hidden = true;
  stage.append(menuCanvas);
  document.body.append(stage, ui, hint, camWheel, soundNote, pauseButton, pauseLayer, cursor);
  const video = document.getElementById('camera');
  const sound = createSound();
  const handCursor = createHandCursor();

  let dwell = null, lastFrame = null, race = null, coach = createWheelCoach(), phase = 'menu';
  let raceUi = null, lastTick = performance.now(), countdownUntil = 0, lastCount = null, mistakes = {}, lastHint = null, lastProblem = null;

  function resize() {
    const dpr = Math.min(2, devicePixelRatio || 1);
    menuCanvas.width = Math.round(innerWidth * dpr); menuCanvas.height = Math.round(innerHeight * dpr);
  }
  addEventListener('resize', resize); resize();

  function targetsOf(buttons) {
    return buttons.map(el => { const r = el.getBoundingClientRect(); return { id: el.dataset.id, x: r.left / innerWidth, y: r.top / innerHeight, w: r.width / innerWidth, h: r.height / innerHeight }; });
  }
  function setButtons(buttons) { dwell = createDwellSelector(targetsOf(buttons)); dwell.buttons = buttons; }

  let pingpong = null, paused = false;
  const inGame = () => !paused && ((race && ['wait', 'countdown', 'race'].includes(phase)) || Boolean(pingpong?.playing));

  function pauseGame() {
    if (!inGame()) return;
    paused = true; pausedAt = performance.now();
    sound.engineAt(null); pingpong?.pause(true); showHint(null);
    const card = html('section', 'game-results');
    card.append(html('p', 'game-eyebrow', race ? 'Руль' : 'Пинг-понг'), html('h2', null, 'Пауза'));
    const row = html('div', 'game-actions');
    const button = (id, text, action) => { const b = html('button', 'game-button', text); b.type = 'button'; b.dataset.id = id; b.append(html('span', 'game-ring')); b.addEventListener('click', () => { sound.ui.confirm(); action(); }); row.append(b); return b; };
    const buttons = [
      button('resume', 'Продолжить', resume),
      button('finish', 'Закончить', () => { resume(); if (race) finishRace(); else pingpong?.finishNow(); }),
      button('menu', 'В меню', () => { closePause(); showMenu(); }),
    ];
    card.append(row, html('p', 'game-lead', 'Наведи ладонь на кнопку и подержи'));
    pauseLayer.replaceChildren(card); pauseLayer.hidden = false;
    requestAnimationFrame(() => setButtons(buttons));
  }
  function closePause() { paused = false; pauseLayer.hidden = true; pauseLayer.replaceChildren(); dwell = null; pauseArmed = false; }
  function resume() {
    if (!paused) return;
    const away = performance.now() - pausedAt;
    closePause(); pingpong?.pause(false);
    if (phase === 'countdown') { countdownUntil += away; lastCount = null; }
  }
  pauseButton.addEventListener('click', () => { sound.ui.confirm(); pauseGame(); });
  addEventListener('keydown', e => { if (e.key === 'Escape') { if (paused) resume(); else pauseGame(); } });

  function leaveRace() {
    sound.menuMusic(false); sound.ui.charge(0); if (paused) closePause();
    if (race) { race.dispose(); race = null; } sound.engineAt(null); raceUi = null; camWheel.hidden = true;
    if (pingpong) { pingpong.dispose(); pingpong = null; }
    document.body.classList.remove('pp-mode');
  }

  function showMenu() {
    leaveRace(); phase = 'menu'; showHint(null); ui.replaceChildren(); menuCanvas.hidden = false;
    sound.menuMusic(true);
    const menu = html('section', 'game-menu');
    menu.append(html('p', 'game-eyebrow', 'Камера вместо джойстика'), html('h1', null, 'Игровой центр'),
      html('p', 'game-lead', 'Наведи ладонь на игру и подержи секунду.'));
    const cards = html('div', 'game-cards'), buttons = [];
    for (const game of GAMES) {
      const card = html('button', 'game-card'); card.type = 'button'; card.dataset.id = game.id; card.disabled = !game.ready;
      card.append(html('span', 'game-card-title', game.title), html('span', 'game-card-text', game.text),
        html('span', 'game-card-foot', game.ready ? (best(game.id) != null ? `Рекорд: ${best(game.id)}` : 'Играть') : 'Скоро'), html('span', 'game-ring'));
      card.addEventListener('click', () => { if (game.ready) { sound.ui.confirm(); open(game.id); } });
      cards.append(card);
      if (game.ready) buttons.push(card);
    }
    menu.append(cards); ui.append(menu);
    requestAnimationFrame(() => setButtons(buttons));
  }

  function open(id) { if (id === 'racer') startRace(); else if (id === 'pingpong') openPingPong(); }

  async function openPingPong() {
    leaveRace();
    phase = 'pingpong'; ui.replaceChildren(); dwell = null; showHint(null); menuCanvas.hidden = true;
    document.body.classList.add('pp-mode');
    const { startPingPong } = await import('./pingpong-game.js');
    if (phase !== 'pingpong') return;
    pingpong = startPingPong({ stage, ui, sound, html, showHint, setButtons, showMenu, saveRecord, video, autopilot, restart: openPingPong });
  }

  async function startRace() {
    leaveRace();
    phase = 'loading'; ui.replaceChildren(); dwell = null; mistakes = {}; lastProblem = null; coach.reset(); menuCanvas.hidden = true;
    const hud = html('div', 'race-hud');
    hud.innerHTML = `<div class="race-stats"><span data-k="time"></span><span data-k="score"></span></div>
      <div class="race-nitro"><span>Нитро</span><i><b></b></i></div>
      <p class="race-legend">Руль: две руки · Нитро: открой рот · Тормоз: закрой один глаз · Пауза: обе руки выше лица</p>`;
    const wheel = html('div', 'race-wheel'); wheel.innerHTML = WHEEL_SVG;
    const speedo = html('div', 'race-speedo'); speedo.innerHTML = speedometerSvg();
    const center = html('div', 'race-center'); center.append(html('p', 'race-call', 'Загружаю город…'));
    ui.append(hud, wheel, speedo, center);
    raceUi = { hud, wheel, speedo, needle: speedo.querySelector('.needle'), value: speedo.querySelector('.value'), gear: speedo.querySelector('.gear'), center };
    const { createRacer3D } = await import('./racer3d.js');
    race = createRacer3D(stage);
    await race.ready;
    if (phase === 'loading') phase = 'wait';
  }

  function finishRace() {
    phase = 'results'; sound.engineAt(null); sound.play('finish');
    const score = race.score(), list = saveRecord('racer', score), s = race.state;
    const card = html('section', 'game-results');
    card.append(html('p', 'game-eyebrow', 'Финиш'), html('h2', null, `${score} очков`));
    const facts = html('dl', 'game-facts');
    const add = (k, v) => facts.append(html('dt', null, k), html('dd', null, String(v)));
    add('Рекорд', list[0].score); add('Обгоны', s.overtakes); add('Столкновения', s.crashes); add('Проехал', `${(s.distance / 1000).toFixed(1)} км`);
    card.append(facts);
    const top = Object.entries(mistakes).sort((a, b) => b[1] - a[1]).slice(0, 2);
    card.append(html('p', 'game-advice', top.length ? `Чаще всего: ${top.map(([code]) => WHEEL_HINTS[code].split(':')[0].toLowerCase()).join('; ')}` : 'Хват был чистым всю гонку.'));
    const row = html('div', 'game-actions');
    const again = html('button', 'game-button', 'Ещё раз'); again.dataset.id = 'again'; again.type = 'button';
    const menu = html('button', 'game-button', 'В меню'); menu.dataset.id = 'menu'; menu.type = 'button';
    again.append(html('span', 'game-ring')); menu.append(html('span', 'game-ring'));
    again.addEventListener('click', () => { sound.ui.confirm(); startRace(); });
    menu.addEventListener('click', () => { sound.ui.confirm(); showMenu(); });
    row.append(again, menu); card.append(row);
    raceUi.center.replaceChildren(card);
    showHint(null);
    requestAnimationFrame(() => setButtons([again, menu]));
  }

  capture.subscribe(frame => { lastFrame = frame; });

  function showHint(text) {
    if (text === lastHint) return;
    if (text && (phase === 'race' || phase === 'pingpong')) sound.play('hint', { volume: .5 });
    lastHint = text; hint.textContent = text ?? ''; hint.hidden = !text;
  }

  // The small wheel drawn between the hands in the camera window.
  function drawCamWheel(wheel) {
    if (!video || wheel.hands < 2 || !video.videoWidth) { camWheel.hidden = true; return; }
    const box = video.getBoundingClientRect(), scale = Math.min(box.width / video.videoWidth, box.height / video.videoHeight);
    const w = video.videoWidth * scale, h = video.videoHeight * scale, left = box.left + (box.width - w) / 2, top = box.top + (box.height - h) / 2;
    const cx = left + (wheel.left.x + wheel.right.x) / 2 * w, cy = top + (wheel.left.y + wheel.right.y) / 2 * h;
    const size = Math.max(22, wheel.spread * w * .6);
    camWheel.hidden = false;
    camWheel.style.width = camWheel.style.height = `${size}px`;
    camWheel.style.transform = `translate(${cx - size / 2}px, ${cy - size / 2}px) rotate(${wheel.angle}rad)`;
  }

  let handsUpSince = null, lastHover = null, brakeHeld = 0, pausedAt = 0, pauseArmed = true;
  function loop(now) {
    const dt = (now - lastTick) / 1000; lastTick = now;
    const frame = lastFrame, t = frame?.timestamp ?? now;
    soundNote.hidden = !sound.needsUnlock;
    sound.tick();
    // Pause gesture: both hands raised above the face for a moment (hands must come down between two pauses).
    pauseButton.hidden = !inGame();
    const up = !autopilot && handsAboveShoulders(frame);
    if (!up) { pauseArmed = true; handsUpSince = null; }
    if (up && pauseArmed && inGame()) {
      handsUpSince ??= now;
      if (now - handsUpSince > 1200) { pauseArmed = false; handsUpSince = null; sound.ui.confirm(); pauseGame(); }
    }
    pauseButton.style.setProperty('--p', handsUpSince && inGame() ? Math.min(1, (now - handsUpSince) / 1200) : 0);
    const pointer = handCursor.update(frame, t);
    if (dwell && pointer) {
      cursor.hidden = false;
      cursor.style.transform = `translate(${pointer.x * innerWidth}px, ${pointer.y * innerHeight}px)`;
      const { hover, progress, selected } = dwell.update(pointer, t);
      if (hover !== lastHover) { if (hover) sound.ui.hover(); lastHover = hover; }
      sound.ui.charge(hover && !selected ? progress : 0);
      for (const b of dwell.buttons) { b.classList.toggle('hover', b.dataset.id === hover); b.style.setProperty('--p', b.dataset.id === hover ? progress : 0); }
      if (selected) dwell.buttons.find(b => b.dataset.id === selected)?.click();
    } else { cursor.hidden = true; if (lastHover) { sound.ui.charge(0); lastHover = null; } }

    if (pingpong) pingpong.tick(frame, now);
    else if (race && paused) race.draw();
    else if (race && raceUi && phase !== 'loading') {
      // ?autopilot=1 (tests, demo recordings): both hands on a centred wheel, no face actions.
      const wheel = autopilot ? { hands: 2, angle: Math.sin(now / 1400) * .25, left: { x: .38, y: .6 }, right: { x: .62, y: .6 }, spread: .24, height: .6, edge: false } : readWheel(frame);
      const face = autopilot ? { found: false, nitro: false, brake: false, bothClosed: false } : readFace(frame);
      const problem = coach.update(wheel, face, t);
      const onWheel = wheel.hands === 2 && !wheel.edge;
      drawCamWheel(wheel);
      if (phase === 'wait') {
        raceUi.center.replaceChildren(html('p', 'race-call', 'Возьмись за руль двумя руками, чтобы начать'));
        if (onWheel) { phase = 'countdown'; countdownUntil = now + 3000; lastCount = null; }
      } else if (phase === 'countdown') {
        const left = Math.ceil((countdownUntil - now) / 1000);
        if (left !== lastCount) { sound.play(left > 0 ? 'tick' : 'go'); lastCount = left; }
        raceUi.center.replaceChildren(html('p', 'race-count', left > 0 ? String(left) : 'Поехали!'));
        if (now >= countdownUntil) { phase = 'race'; raceUi.center.replaceChildren(); }
      } else if (phase === 'race') {
        race.update({ steer: wheel.hands === 2 ? steering(wheel.angle) : 0, nitro: face.nitro, brake: face.brake }, dt);
        for (const event of race.state.events.splice(0)) {
          if (event === 'crash') { sound.play('crash'); sound.play('crash2', { volume: .6 }); }
          else if (event === 'shift') sound.shift();
        }
        const rs = race.state;
        // The tyres screech only for a real brake, not a blink.
        brakeHeld = rs.braking ? brakeHeld + dt : 0;
        sound.engineAt(rs.rpm, { throttle: rs.throttle, nitro: rs.nitroOn, brake: brakeHeld > .15 && rs.speed > 5 ? Math.min(1, .35 + rs.speed / 40) : 0 });
        // Each time a grip mistake appears, it counts once (for the results screen).
        if (problem && problem.code !== lastProblem) mistakes[problem.code] = (mistakes[problem.code] ?? 0) + 1;
        lastProblem = problem?.code ?? null;
        if (race.state.over) finishRace();
      }
      if (race && raceUi) {
        const s = race.state, [time, score] = raceUi.hud.querySelectorAll('[data-k]');
        time.textContent = `${Math.ceil(s.time)} с`; score.textContent = `${race.score()} очк.`;
        const kmh = race.kmh();
        raceUi.needle.setAttribute('transform', `rotate(${-30 + Math.min(SPEEDO_MAX, kmh) / SPEEDO_MAX * 240})`);
        raceUi.value.textContent = String(kmh);
        raceUi.gear.textContent = s.speed < .5 ? 'N' : String(s.gear + 1);
        raceUi.speedo.classList.toggle('redline', s.rpm > 7000);
        raceUi.speedo.classList.toggle('boost', s.nitroOn);
        raceUi.hud.querySelector('b').style.width = `${Math.round(s.nitroFuel * 100)}%`;
        raceUi.hud.classList.toggle('boost', s.nitroOn);
        raceUi.wheel.style.transform = `rotate(${wheel.hands === 2 ? wheel.angle : 0}rad)`;
        raceUi.wheel.classList.toggle('lost', wheel.hands < 2);
        race.draw();
        if (phase !== 'results') showHint(phase === 'wait' && !problem ? null : problem?.text ?? null);
      }
    } else if (!race && phase !== 'pingpong') {
      camWheel.hidden = true;
      const ctx = menuCanvas.getContext('2d'), g = ctx.createLinearGradient(0, 0, 0, menuCanvas.height);
      g.addColorStop(0, '#141a2e'); g.addColorStop(1, '#2b2140');
      ctx.fillStyle = g; ctx.fillRect(0, 0, menuCanvas.width, menuCanvas.height);
    }
    requestAnimationFrame(loop);
  }

  // ?game=racer opens a game straight away (demo links).
  const autopilot = new URLSearchParams(location.search).get('autopilot') === '1';
  const direct = new URLSearchParams(location.search).get('game');
  if (GAMES.some(g => g.id === direct && g.ready)) open(direct); else showMenu();
  requestAnimationFrame(t => { lastTick = t; loop(t); });
  return { showMenu, open };
}
