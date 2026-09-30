/** The «Космический лазер» screen of the game centre: a gaze
 * calibration before every game (~28 s, see STAGES), then waves of asteroids burned by looking
 * at them; sounds, hints and the results.
 *
 *   const game = startSpace(host);   // host: shared pieces of the game centre
 *   game.tick(frame, now); game.pause(on); game.finishNow(); game.dispose();
 */
import { readFace } from './controls.js';
import { createGaze } from './gaze.js';
import { createSpace3D } from './space3d.js';

// Calibration before every game (~28 s): the pose changes from game to game, and a fit made
// for another pose was far off in the recordings. The middle; the middle again while the head
// sways (so the fit learns that a turned head with eyes turned back still looks at the same
// spot — the cause of the aim drifting in play); the four corners (the edges were the worst);
// a dot gliding over the whole screen; then three check dots measure the aim.
const EYE_LAG = .2, CHECK_LIMIT_PX = 160;
const glideAt = t => ({ x: .5 + .44 * Math.sin(2 * Math.PI * t / 8), y: .5 + .4 * Math.sin(2 * Math.PI * t / 5.6) });
const C = { x: .5, y: .5 };
const STAGES = [
  { kind: 'dot', at: C, move: 0, settle: .6, collect: .9, step: 1 },
  { kind: 'sway', at: C, dur: 4, skip: .6, step: 2 },
  ...[{ x: .1, y: .13 }, { x: .9, y: .13 }, { x: .9, y: .87 }, { x: .1, y: .87 }].map(at => ({ kind: 'dot', at, move: .3, settle: .45, collect: .6, step: 3 })),
  { kind: 'glide', dur: 14, skip: 1, step: 4 },
  { kind: 'fit' },
  ...[{ x: .3, y: .35 }, { x: .72, y: .6 }, { x: .45, y: .8 }].map(at => ({ kind: 'check', at, move: .3, settle: .5, collect: .6, step: 5 })),
  { kind: 'verdict' },
];
// Before each step a short card says what to do; nothing is recorded while it is read
// (reading moves the eyes all over the text), then the card goes and only the dot remains.
const BRIEF = {
  1: { text: 'Смотри в центр точки', ms: 2600 },
  2: { text: 'Смотри на точку и медленно покачай головой: влево-вправо, вверх-вниз. Взгляд всё время на точке.', ms: 4200, sway: true },
  3: { text: 'Точка прыгнет по углам экрана — смотри на неё', ms: 2600 },
  4: { text: 'Точка поплывёт по экрану — веди её глазами', ms: 2600 },
  5: { text: 'Проверка прицела: смотри на точки', ms: 2400 },
};
const READY_MS = 3000;

export const SPACE_HINTS = {
  noFace: 'Лицо не видно: сядь прямо напротив камеры',
  calibrate: 'Смотри глазами на точку и веди её взглядом',
  retry: 'Ещё раз, поточнее',
};
const ADVICE = {
  lost: 'много астероидов долетело — переводи взгляд на ближайший, а не на дальний',
  slow: 'держи взгляд на астероиде дольше, пока он не раскалится добела',
  shock: 'не забывай про ударную волну: когда шкала полная, открой рот',
};

export function startSpace(host) {
  const { stage, ui, sound, html, showHint, setButtons, showMenu, saveRecord, video, autopilot } = host;
  const gaze = createGaze();
  // ?aimoff=0.3 (with ?autopilot=1): the autopilot looks at the rocks with its aim shifted down
  // by that share of the screen — a check that the movement matching still finds them.
  const autoOffset = { x: 0, y: Number(new URLSearchParams(location.search).get('aimoff')) || 0 };
  const rawHist = [];
  let calStage = 0, checks = [], briefStep = 0, briefUntil = 0, readyUntil = 0, mouthSince = null;
  const game = createSpace3D(stage);
  if (autopilot) window.motionSpace = game;                     // for automated checks
  // For tuning the aim with a real player: the calibration and the last seconds of play.
  const live = [];
  window.motionGaze = { gaze, live, game };
  let phase = autopilot ? 'play' : 'intro', stepStart = null, retried = false, paused = false, pausedAt = 0, pausedTotal = 0;
  let hum = null, noFaceSince = null, overAt = null;

  const hud = html('div', 'sp-hud');
  hud.innerHTML = `<div class="sp-stats"><span data-k="score">0</span><span data-k="wave">Волна 1</span></div>
    <div class="sp-bar sp-shield"><span>Щиты</span><i><b></b></i></div>
    <div class="sp-bar sp-charge"><span>Ударная волна</span><i><b></b></i></div>
    <p class="race-legend">Смотри на астероид — лазер жжёт его · Полная шкала: открой рот · Пауза: обе руки выше лица</p>`;
  const reticle = html('div', 'sp-reticle'); reticle.hidden = true;
  const lockRing = html('div', 'sp-lock'); lockRing.hidden = true;
  const banner = html('div', 'sp-banner'); banner.hidden = true;
  const flashLayer = html('div', 'sp-flash');
  const dot = html('div', 'sp-dot'); dot.hidden = true;
  const calibText = html('p', 'sp-calib'); calibText.hidden = true;
  const center = html('div', 'race-center');
  // The briefing card (what to do next) and the welcome card.
  const brief = html('section', 'sp-brief'); brief.hidden = true;
  ui.append(flashLayer, hud, reticle, lockRing, banner, dot, calibText, brief, center);
  hud.hidden = !autopilot;
  function showBrief(title, text, { ms = 0, sway = false } = {}) {
    brief.replaceChildren(html('p', 'game-eyebrow', title), html('h2', null, text));
    if (sway) { const head = html('div', 'sp-head'); head.append(html('i', 'sp-eye'), html('i', 'sp-eye'), html('b', 'sp-mouth')); brief.append(head); }
    if (ms) { const bar = html('div', 'sp-brief-bar'); bar.style.animationDuration = `${ms}ms`; brief.append(bar); }
    brief.hidden = false;
  }
  function showIntro() {
    brief.replaceChildren(html('p', 'game-eyebrow', 'Космический лазер'), html('h2', null, 'Управление — глазами'));
    const list = html('ul', 'sp-rules');
    for (const t of ['Смотри на астероид — лазер жжёт его, пока он не взорвётся', 'Шкала «ударная волна» полная — открой рот', 'Обе руки выше лица — пауза'])
      list.append(html('li', null, t));
    brief.append(list, html('p', 'sp-note', 'Сначала настроим прицел под твои глаза — около 30 секунд. Сядь так, как будешь играть, и держи голову в кадре.'));
    const go = html('button', 'game-button', 'Начать'); go.type = 'button'; go.dataset.id = 'start'; go.append(html('span', 'game-ring'));
    go.addEventListener('click', startCalibration);
    brief.append(go, html('p', 'sp-note', 'Наведи ладонь на кнопку, нажми мышкой или открой рот'));
    brief.hidden = false;
    requestAnimationFrame(() => setButtons([go]));
  }
  function startCalibration() {
    if (phase !== 'intro') return;
    sound.ui.confirm(); host.clearButtons?.(); phase = 'calibrate'; calStage = 0; briefStep = 0; stepStart = null; brief.hidden = true;
  }
  if (!autopilot) showIntro();
  const q = k => hud.querySelector(`[data-k="${k}"]`);
  let bannerTimer = null;
  function showBanner(text, ms = 1600, tone = '') {
    banner.textContent = text; banner.className = `sp-banner ${tone}`; banner.hidden = false;
    clearTimeout(bannerTimer); bannerTimer = setTimeout(() => { banner.hidden = true; }, ms);
  }
  function floatText(text, x, y) {
    const el = html('div', 'sp-float', text); el.style.left = `${x}px`; el.style.top = `${y}px`;
    ui.append(el); setTimeout(() => el.remove(), 1000);
  }

  // A rock burned until it exploded: the player surely looked at it all that time. Its path
  // paired with the aim of the same moments (the eyes a little behind) teaches the aim.
  const rawAt = t => { let best = null; for (const r of rawHist) if (!best || Math.abs(r.t - t) < Math.abs(best.t - t)) best = r; return best && Math.abs(best.t - t) < .1 ? best : null; };
  function learnFromKill(ev) {
    if (autopilot || !ev.burn?.length) return;
    const pairs = ev.burn.map(b => ({ aim: rawAt(b.t + .1), target: { x: b.x, y: b.y } })).filter(p => p.aim);
    if (pairs.length >= 5) gaze.recal.add(pairs, game.state.time);
  }

  function onEvents() {
    for (const ev of game.state.events.splice(0)) {
      if (ev.type === 'kill') learnFromKill(ev);
      if (ev.type === 'wave') { showBanner(`Волна ${ev.wave}`, 1800); sound.play('space-zap', { volume: .5, rate: .8 }); }
      else if (ev.type === 'kill') {
        sound.play(ev.big ? 'space-explode2' : 'space-explode1', { volume: ev.big ? .9 : .7, rate: .9 + Math.random() * .2, pan: (ev.x / innerWidth - .5) * 1.4 });
        floatText(`+${ev.points}${ev.combo ? ` ×${(1 + ev.combo * .25).toFixed(2).replace(/\.?0+$/, '')}` : ''}`, ev.x, ev.y);
        if (ev.combo >= 2) showBanner(`Комбо ×${ev.combo + 1}`, 900, 'good');
      } else if (ev.type === 'lock') sound.play('space-zap', { volume: .25, rate: 1.4 });
      else if (ev.type === 'hit') {
        sound.play('space-shield', { volume: .9 }); sound.play('space-hit', { volume: .6 });
        flashLayer.classList.remove('on'); void flashLayer.offsetWidth; flashLayer.classList.add('on');
        if (ev.shields <= 30 && ev.shields > 0) showBanner('Щиты на исходе!', 1400, 'bad');
      } else if (ev.type === 'shockwave') { sound.play('space-boom', { volume: 1 }); showBanner(`Ударная волна! −${ev.count}`, 1400, 'good'); }
      else if (ev.type === 'over') { sound.play('space-boom', { volume: .8, rate: .7 }); showBanner('Станция пала', 2500, 'bad'); overAt = performance.now(); }
    }
  }

  function hudUpdate() {
    const st = game.state;
    q('score').textContent = st.score.toLocaleString('ru-RU'); q('wave').textContent = `Волна ${st.wave}`;
    hud.querySelector('.sp-shield b').style.width = `${st.shields}%`;
    hud.querySelector('.sp-shield').classList.toggle('low', st.shields <= 30);
    hud.querySelector('.sp-charge b').style.width = `${Math.round(st.charge * 100)}%`;
    hud.querySelector('.sp-charge').classList.toggle('full', st.charge >= 1);
    hud.querySelector('.sp-charge span').textContent = st.charge >= 1 ? 'Ударная волна: открой рот!' : 'Ударная волна';
    const g = st.gaze;
    reticle.hidden = !g;
    if (g) reticle.style.transform = `translate(${g.x * innerWidth}px, ${g.y * innerHeight}px)`;
    const t = st.target;
    lockRing.hidden = !t;
    if (t) {
      const size = Math.max(46, t.screen.r * 2 + 24);
      lockRing.style.width = lockRing.style.height = `${size}px`;
      lockRing.style.transform = `translate(${t.screen.x - size / 2}px, ${t.screen.y - size / 2}px)`;
      lockRing.style.setProperty('--p', Math.min(1, t.heat));
    }
    reticle.classList.toggle('locked', Boolean(t));
    sound.laser(Boolean(t), t?.heat ?? 0);
  }

  function finish() {
    phase = 'results'; hum?.stop(); hum = null; sound.laser(false); showHint(null); reticle.hidden = lockRing.hidden = true;
    sendRecording(game.state.phase === 'over' ? 'over' : 'stopped');
    const st = game.state, x = st.stats, stopped = st.phase !== 'over', list = saveRecord('gaze', st.score);
    const card = html('section', 'game-results');
    card.append(html('p', 'game-eyebrow', stopped ? 'Игра прервана' : 'Станция пала'), html('h2', null, `${st.score.toLocaleString('ru-RU')} очков`));
    const facts = html('dl', 'game-facts');
    const add = (k, v) => facts.append(html('dt', null, k), html('dd', null, String(v)));
    add('Рекорд', list[0].score.toLocaleString('ru-RU')); add('Волна', st.wave); add('Сбито астероидов', `${x.kills} (крупных ${x.big})`);
    add('Лучшее комбо', x.bestCombo ? `×${x.bestCombo + 1}` : '—'); add('Ударных волн', x.shockwaves); add('Продержался', `${Math.round(x.seconds)} с`);
    card.append(facts);
    const tips = [];
    if (x.lost > x.kills / 3) tips.push(ADVICE.lost);
    if (x.aimTime && x.burnTime / x.aimTime < .5) tips.push(ADVICE.slow);
    if (!x.shockwaves && x.kills > 12) tips.push(ADVICE.shock);
    card.append(html('p', 'game-advice', tips.length ? `Совет: ${tips.slice(0, 2).join('; ')}` : 'Отличный взгляд — так держать.'));
    const row = html('div', 'game-actions');
    const again = html('button', 'game-button', 'Ещё раз'); again.dataset.id = 'again'; again.type = 'button';
    const menu = html('button', 'game-button', 'В меню'); menu.dataset.id = 'menu'; menu.type = 'button';
    again.append(html('span', 'game-ring')); menu.append(html('span', 'game-ring'));
    again.addEventListener('click', () => { sound.ui.confirm(); host.restart(); });
    menu.addEventListener('click', () => { sound.ui.confirm(); showMenu(); });
    row.append(again, menu); card.append(row);
    center.replaceChildren(card);
    requestAnimationFrame(() => setButtons([again, menu]));
  }

  function calibrate(frame, nowMs) {
    const f = gaze.features(frame);
    showHint(!f ? SPACE_HINTS.noFace : null);
    const st0 = STAGES[calStage];
    // A new step: its card first, for as long as it takes to read.
    if (st0.step && st0.step !== briefStep) {
      briefStep = st0.step; briefUntil = nowMs + BRIEF[st0.step].ms; stepStart = null;
      dot.hidden = true; calibText.hidden = true;
      showBrief(`${retried ? 'Ещё раз, поточнее · ' : ''}Настройка прицела · шаг ${st0.step} из 5`, BRIEF[st0.step].text, BRIEF[st0.step]);
    }
    if (nowMs < briefUntil) return;
    brief.hidden = true;
    if (!f) { stepStart = null; return; }                     // the clock waits while the face is lost
    stepStart ??= nowMs;
    const st = STAGES[calStage], s = (nowMs - stepStart) / 1000, prev = STAGES[calStage - 1];
    const next = () => { calStage++; stepStart = nowMs; };
    if (st.kind === 'fit') {
      if (!gaze.solve() && !retried) return restart();
      checks = []; next(); return;
    }
    if (st.kind === 'verdict') {
      const med = a => [...a].sort((p, q) => p - q)[Math.floor(a.length / 2)];
      const err = checks.length ? Math.round(med(checks)) : Infinity;
      if (err > CHECK_LIMIT_PX && !retried) { showBanner(`Промах ${Number.isFinite(err) ? `${err} px` : 'большой'} — ещё раз`, 1800, 'bad'); return restart(); }
      dot.hidden = true; calibText.hidden = true; showHint(null); sound.ui.confirm();
      showBrief('Прицел готов', gaze.calibrated ? `Промах около ${err} px. Поехали!` : 'Прицел по умолчанию: смотри на астероид и держи взгляд. Поехали!', { ms: READY_MS });
      phase = 'ready'; readyUntil = nowMs + READY_MS;
      return;
    }
    // Where the dot is: easing from the last spot, fixed, or gliding.
    const from = prev?.at ?? (prev?.kind === 'glide' ? glideAt(prev.dur) : C);
    let target;
    if (st.kind === 'glide') {
      const k = Math.min(1, s / .5), g = glideAt(s);
      target = k < 1 ? { x: from.x + (g.x - from.x) * k, y: from.y + (g.y - from.y) * k } : g;
    } else {
      const k = st.move ? Math.min(1, s / st.move) : 1, e = k * k * (3 - 2 * k);
      target = { x: from.x + (st.at.x - from.x) * e, y: from.y + (st.at.y - from.y) * e };
    }
    dot.hidden = false; dot.classList.toggle('glide', st.kind === 'glide' || st.kind === 'sway');
    dot.style.transform = `translate(${target.x * innerWidth}px, ${target.y * innerHeight}px)`;
    calibText.hidden = false; calibText.textContent = `${st.step} / 5`;   // small, at the bottom: nothing to read while recording
    if (st.kind === 'glide' || st.kind === 'sway') {
      dot.style.setProperty('--p', Math.min(1, s / st.dur));
      if (s > st.skip) gaze.addSample(f, st.kind === 'glide' ? glideAt(s - EYE_LAG) : st.at);
      if (s >= st.dur) { next(); sound.ui.hover(); }
      return;
    }
    const hold = (st.move ?? 0) + st.settle;
    dot.style.setProperty('--p', Math.min(1, Math.max(0, (s - hold) / st.collect)));
    if (s > hold && s < hold + st.collect) {
      if (st.kind === 'dot') gaze.addSample(f, st.at);
      else { const a = gaze.aimOf(f); checks.push(Math.hypot((a.x - st.at.x) * innerWidth, (a.y - st.at.y) * innerHeight)); }
    }
    if (s >= hold + st.collect) { next(); sound.ui.hover(); }
  }
  function restart() { retried = true; calStage = 0; stepStart = null; checks = []; briefStep = 0; gaze.clear(); }

  function tick(frame, nowMs) {
    if (paused || phase === 'results') { game.draw(); return; }
    if (phase === 'intro') {
      // Hands-free start: an open mouth, held a moment.
      if (readFace(frame).nitro) { mouthSince ??= nowMs; if (nowMs - mouthSince > 600) startCalibration(); } else mouthSince = null;
      game.draw(); return;
    }
    if (phase === 'calibrate') { calibrate(frame, nowMs); game.draw(); return; }
    if (phase === 'ready') {
      if (nowMs < readyUntil) { game.draw(); return; }
      brief.hidden = true; hud.hidden = false; phase = 'play';
    }
    hum ??= sound.loop('space-hum', { volume: .22 });
    hum.tick();
    const t = frame?.timestamp ?? nowMs;
    const point = autopilot ? null : gaze.point(frame, t);
    const face = autopilot ? { found: true, nitro: false } : readFace(frame);
    game.update({ gaze: point, mouth: face.nitro, auto: autopilot, autoOffset }, (nowMs - pausedTotal) / 1000);
    // The aim before the in-play recalibration, by game time: paired later with the rocks burned down.
    const gameT = (nowMs - pausedTotal) / 1000, lc = gaze.lastCentred;
    if (lc && !autopilot) { rawHist.push({ t: gameT, x: lc.x, y: lc.y }); while (rawHist.length && rawHist[0].t < gameT - 20) rawHist.shift(); }
    const locked = game.state.target;
    const f = gaze.features(frame);
    if (f) { live.push({ t: Math.round(nowMs), f, point, raw: lc, how: game.state.how, target: locked ? { x: locked.screen.x / innerWidth, y: locked.screen.y / innerHeight, heat: +locked.heat.toFixed(2) } : null, rocks: game.state.asteroids.map(r => [+(r.screen.x / innerWidth).toFixed(4), +(r.screen.y / innerHeight).toFixed(4), +(r.screen.r / innerWidth).toFixed(4)]) }); if (live.length > 20000) { live.shift(); sentUpTo = Math.max(0, sentUpTo - 1); } }
    onEvents();
    hudUpdate();
    if (!autopilot && !face.found) noFaceSince ??= nowMs; else noFaceSince = null;
    showHint(noFaceSince && nowMs - noFaceSince > 800 ? SPACE_HINTS.noFace : null);
    if (game.state.phase === 'over' && overAt && performance.now() - overAt > 2200) finish();
    game.draw();
  }

  // Dev only: send the game (model, calibration, every frame) to ml/gaze/data/ in parts every
  // 30 s, so a page reload loses little. Part 0 carries the model and the calibration.
  const gameId = `${new Date().toISOString().replace(/[:.]/g, '-')}-${Math.random().toString(36).slice(2, 6)}`;
  let part = 0, sentUpTo = 0, done = false;
  function sendRecording(reason) {
    if (autopilot || done) return;
    const frames = live.slice(sentUpTo); sentUpTo = live.length;
    if (!frames.length && reason === 'part') return;
    const st = game.state, body = { kind: 'play', reason, part, screen: { w: innerWidth, h: innerHeight }, stats: { ...st.stats, wave: st.wave, score: st.score, shields: st.shields }, recal: { fix: gaze.recal.fix, labels: gaze.recal.count }, how: game.state.how, frames,
      ...(part === 0 ? { model: gaze.model, calibration: gaze.debug().samples, bias: gaze.debug().bias } : {}) };
    // keepalive (surviving a page close) is only allowed for small bodies (64 KB): used just for the last one.
    const json = JSON.stringify(body);
    fetch(`/__gaze-data?game=${gameId}&part=${part++}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: json, keepalive: reason === 'left' && json.length < 60000 }).catch(() => {});
    if (reason !== 'part') done = true;
  }
  const flush = setInterval(() => { if (phase === 'play' && !paused) sendRecording('part'); }, 30000);
  addEventListener('pagehide', () => sendRecording('left'));

  function dispose() {
    sendRecording('closed'); clearInterval(flush);
    hum?.stop(); sound.laser(false); game.dispose(); clearTimeout(bannerTimer);
    if (window.motionSpace === game) delete window.motionSpace;
    if (window.motionGaze?.game === game) delete window.motionGaze;
  }
  return {
    tick, dispose,
    get playing() { return ['intro', 'calibrate', 'ready', 'play'].includes(phase); },
    pause(on) {
      if (on === paused) return;
      paused = on;
      if (on) { pausedAt = performance.now(); hum?.stop(); hum = null; sound.laser(false); } else pausedTotal += performance.now() - pausedAt;
    },
    finishNow() { if (phase === 'play') finish(); else showMenu(); },
  };
}
