/** A one-off gaze calibration page (?games=1&calibrate=gaze): the player
 * follows a dot for about a minute — fixed points all over the screen, then a
 * slowly gliding dot, then a check on fresh points. Every camera frame is
 * recorded with where the dot was; the page fits the player's aim, shows how
 * far off it is, keeps the fit for the game (localStorage), and in dev sends
 * the whole recording to ml/gaze/data/ for offline tuning.
 *
 *   mountGazeLab(window.motionCapture)
 */
import './games.css';
import { createGaze, fitGaze, trainGaze, GAZE_KEYS } from './gaze.js';

export const GAZE_MODEL_KEY = 'motion-gaze-model';
const FIX = [.08, .36, .64, .92].flatMap(x => [.12, .5, .88].map(y => ({ x, y })));
const MOVE = .35, SETTLE = .45, HOLD = 1.3;                 // seconds per fixed point: glide, settle, then recorded
const PURSUIT = 32;                                          // seconds of the gliding dot
const EYE_LAG = .2;                                          // the eyes trail a moving dot by about this much (measured)
const CHECK = [{ x: .25, y: .3 }, { x: .75, y: .3 }, { x: .5, y: .5 }, { x: .2, y: .75 }, { x: .8, y: .72 }, { x: .5, y: .15 }];
// Raw points kept in the recording: eye corners, lids, irises, nose, cheeks, lips, forehead, chin.
const RAW = [33, 133, 159, 145, 160, 144, 158, 153, 468, 469, 470, 471, 472, 263, 362, 386, 374, 387, 373, 385, 380, 473, 474, 475, 476, 477, 1, 234, 454, 13, 14, 10, 152];
const SHAPES = ['eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft', 'eyeLookDownLeft', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight', 'eyeLookDownRight',
  'eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight'];
const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
const pursuitAt = t => ({ x: .5 + .42 * Math.sin(2 * Math.PI * t / 11), y: .5 + .37 * Math.sin(2 * Math.PI * t / 7.3 + Math.PI / 2) });

export function mountGazeLab(capture) {
  document.body.classList.add('games', 'cam-corner', 'gaze-lab');
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const root = el('div', 'gl-root'), dot = el('div', 'gl-dot'), aim = el('div', 'sp-reticle'), panel = el('section', 'gl-panel');
  aim.hidden = true; dot.hidden = true;
  root.append(dot, aim, panel); document.body.append(root);

  const gaze = createGaze();
  let frame = null, phase = 'intro', phaseAt = 0, order = [], index = 0, faceSince = null;
  const rec = { started: new Date().toISOString(), screen: { w: innerWidth, h: innerHeight }, video: null, frames: [] };
  capture.subscribe(f => {
    if (!f || f.timestamp === frame?.timestamp) return;
    frame = f; onFrame(f);
  });

  function say(title, text, buttons = []) {
    panel.replaceChildren(el('h2', null, title), el('p', null, text));
    if (buttons.length) { const row = el('div', 'game-actions'); for (const [label, fn] of buttons) { const b = el('button', 'game-button', label); b.type = 'button'; b.addEventListener('click', fn); row.append(b); } panel.append(row); }
    panel.hidden = false;
  }
  const target = () => {
    const t = (performance.now() - phaseAt) / 1000;
    if (phase === 'fix' || phase === 'check') {
      const list = phase === 'fix' ? order : CHECK, p = list[index], prev = list[index - 1] ?? p;
      const k = Math.min(1, t / MOVE), e = k * k * (3 - 2 * k);
      return { x: prev.x + (p.x - prev.x) * e, y: prev.y + (p.y - prev.y) * e, settled: t > MOVE + SETTLE, t };
    }
    if (phase === 'pursuit') { const p = pursuitAt(t); return { ...p, settled: t > 1.5, t, lagged: pursuitAt(Math.max(0, t - EYE_LAG)) }; }
    return null;
  };

  // Every camera frame: record it with the dot's place.
  function onFrame(f) {
    const p = f.face?.landmarks;
    rec.video ??= (() => { const v = document.getElementById('camera'); return v?.videoWidth ? { w: v.videoWidth, h: v.videoHeight } : null; })();
    if (!['fix', 'pursuit', 'check', 'free'].includes(phase)) return;
    const tg = target(), feats = gaze.features(f);
    rec.frames.push({
      t: Math.round(f.timestamp), phase, index, target: tg && { x: +tg.x.toFixed(4), y: +tg.y.toFixed(4), settled: tg.settled, lagged: tg.lagged && { x: +tg.lagged.x.toFixed(4), y: +tg.lagged.y.toFixed(4) } },
      f: feats && Object.fromEntries(Object.entries(feats).map(([k, v]) => [k, +v.toFixed(5)])),
      raw: p?.length >= 478 ? RAW.map(i => [+p[i].x.toFixed(5), +p[i].y.toFixed(5), +(p[i].z ?? 0).toFixed(5)]) : null,
      bs: f.face?.blendshapes?.length ? SHAPES.map(n => +(f.face.blendshapes.find(c => c.categoryName === n)?.score ?? 0).toFixed(4)) : null,
      m: (() => { const m = f.face?.transformationMatrix?.data ?? f.face?.transformationMatrix; return m ? Array.from(m).map(v => +v.toFixed(5)) : null; })(),
      aim: phase === 'check' || phase === 'free' ? gaze.point(f, f.timestamp) : null,
    });
  }

  function begin(next) { phase = next; phaseAt = performance.now(); index = 0; }
  function loop() {
    const now = performance.now(), t = (now - phaseAt) / 1000;
    const face = Boolean(frame?.face?.landmarks?.length);
    if (phase === 'intro') {
      faceSince = face ? faceSince ?? now : null;
      say('Калибровка взгляда', face ? 'Сядь так, как будешь играть. Голову держи ровно, смотри только глазами. Следи за точкой: сначала она прыгает по экрану, потом плавно едет. Около минуты.' : 'Лицо не видно: сядь прямо напротив камеры.',
        face ? [['Начать', () => { order = shuffle([...FIX]); panel.hidden = true; dot.hidden = false; begin('fix'); }]] : []);
      phase = 'intro-wait';
    } else if (phase === 'intro-wait' && !face) phase = 'intro';
    const tg = target();
    if (tg) { dot.style.transform = `translate(${tg.x * innerWidth}px, ${tg.y * innerHeight}px)`; dot.classList.toggle('hold', tg.settled); }
    if (phase === 'fix' && t > MOVE + SETTLE + HOLD) { index++; phaseAt = now; if (index >= order.length) begin('pursuit'); }
    else if (phase === 'pursuit' && t > PURSUIT) { fit(); begin('check'); aim.hidden = false; }
    else if (phase === 'check' && t > MOVE + SETTLE + HOLD) { index++; phaseAt = now; if (index >= CHECK.length) finish(); }
    if (phase === 'check' || phase === 'free') {
      const p = rec.frames.at(-1)?.aim;
      if (p) aim.style.transform = `translate(${p.x * innerWidth}px, ${p.y * innerHeight}px)`;
    }
    requestAnimationFrame(loop);
  }

  let model = null;
  function fit() {
    // Fixed points once the eyes have settled, and the gliding dot where the eyes were a moment ago.
    const samples = [];
    for (const r of rec.frames) {
      if (!r.f || !r.target?.settled) continue;
      if (r.phase === 'fix') samples.push({ f: r.f, target: r.target, group: `fix${r.index}` });
      if (r.phase === 'pursuit') samples.push({ f: r.f, target: r.target.lagged, group: `pur${Math.floor(r.t / 400)}` });
    }
    // The expanded model when it holds up, the plain (never mirrored) one otherwise.
    model = trainGaze(samples) ?? fitGaze(samples);
    if (model) gaze.load(model);
  }

  async function finish() {
    phase = 'done'; dot.hidden = true;
    // How far the aim was from the check dots (settled part only), in screen widths and pixels.
    const errs = rec.frames.filter(r => r.phase === 'check' && r.target?.settled && r.aim).map(r => ({ dx: (r.aim.x - r.target.x) * innerWidth, dy: (r.aim.y - r.target.y) * innerHeight }));
    const med = a => { const b = [...a].sort((p, q) => p - q); return b[Math.floor(b.length / 2)] ?? NaN; };
    const err = { px: Math.round(med(errs.map(e => Math.hypot(e.dx, e.dy)))), x: Math.round(med(errs.map(e => Math.abs(e.dx)))), y: Math.round(med(errs.map(e => Math.abs(e.dy)))) };
    rec.model = model && { ...model, keys: GAZE_KEYS };
    rec.error = err;
    if (model) try { localStorage.setItem(GAZE_MODEL_KEY, JSON.stringify({ ...model, keys: GAZE_KEYS, error: err, date: new Date().toISOString(), screen: rec.screen })); } catch { /* private mode */ }
    let saved = '';
    try { const r = await fetch('/__gaze-data', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(rec) }); if (r.ok) saved = ` Запись сохранена: ${(await r.json()).file}.`; } catch { /* no dev server: only the model is kept */ }
    window.motionGazeLab = rec;
    begin('free');
    say('Готово', `${model ? `Промах прицела: в среднем ${err.px} пикселей (по горизонтали ${err.x}, по вертикали ${err.y}).` : 'Подобрать прицел не вышло — попробуй ещё раз, глядя точно в центр точки.'} Теперь посмотри свободно по экрану — прицел идёт за взглядом.${saved}`,
      [['Ещё раз', () => location.reload()], ['В игру', () => { location.search = '?games=1&game=gaze'; }]]);
  }
  requestAnimationFrame(loop);
}
