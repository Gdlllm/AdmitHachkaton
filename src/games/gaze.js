/** Where on the screen the player is looking, from a webcam: the head's turn
 * and tilt plus the pupils' place between the eye corners. Every feature is
 * measured in the mirrored picture the player sees, with a known direction:
 * it grows when the player looks to the screen's right (x) or down (y). The
 * calibration (look at a few dots) fits an offset and non-negative gains, so
 * the aim can never come out mirrored. A 1€ filter calms the jitter.
 *
 *   const gaze = createGaze();
 *   gaze.features(frame)                 → { hx, ix, bx, hy, iy, by, lo } or null (no face / eyes shut)
 *   gaze.nudge(dx, dy)                   while the player surely looks at something: drift the aim towards it
 *   gaze.addSample(features, { x, y })   during calibration (x, y in 0..1 of the screen)
 *   gaze.solve()                         → true when the fit is usable
 *   gaze.point(frame, t)                 → { x, y } in 0..1 of the screen, smoothed, or null
 */
import { createOneEuro } from './controls.js';

// Face mesh indices (MediaPipe Face Landmarker, 478 points).
const EYES = [
  { outer: 33, inner: 133, top: 159, bottom: 145, iris: 468 },
  { outer: 263, inner: 362, top: 386, bottom: 374, iris: 473 },
];
const NOSE = 1, CHEEKS = [234, 454], LIPS = [13, 14];
const blink = (frame, name) => frame?.face?.blendshapes?.find?.(c => c.categoryName === name)?.score ?? 0;

export function gazeFeatures(frame) {
  const p = frame?.face?.landmarks;
  if (!p || p.length < 478) return null;
  if (blink(frame, 'eyeBlinkLeft') > .55 && blink(frame, 'eyeBlinkRight') > .55) return null;   // eyes shut: keep the last aim
  const [l, r] = CHEEKS.map(i => p[i]), n = p[NOSE];
  const faceW = Math.hypot(r.x - l.x, r.y - l.y) || 1;
  // Head turn: the nose against the middle of the cheeks. The camera picture is not
  // mirrored, the screen is: minus makes "towards the screen's right" positive.
  const hx = -(n.x - (l.x + r.x) / 2) / faceW;
  // Head tilt: where the nose sits between the eye line and the mouth (grows when looking down).
  const eyeY = (p[33].y + p[263].y) / 2, mouthY = (p[LIPS[0]].y + p[LIPS[1]].y) / 2;
  const hy = (n.y - eyeY) / Math.max(1e-3, mouthY - eyeY);
  // Pupils against the eye corners, in eye widths (mirrored like the head). The corners stay
  // put while the eye turns; the lids do not (they follow the gaze), so up–down is measured
  // from the corner line, and the lid opening is its own cue: it narrows looking down.
  let ix = 0, iy = 0, lo = 0;
  for (const e of EYES) {
    const o = p[e.outer], i = p[e.inner], c = p[e.iris], w = Math.hypot(o.x - i.x, o.y - i.y) || 1;
    ix += -(c.x - (o.x + i.x) / 2) / w;
    iy += (c.y - (o.y + i.y) / 2) / w;
    lo += -(p[e.bottom].y - p[e.top].y) / w;
  }
  // The face model's own eye-direction scores: they account for the lids, which hide the
  // pupils when the player looks down at a screen below the camera. Names follow the
  // player's left and right; to the screen's right is the right eye out, the left eye in.
  const b = name => blink(frame, name);
  const bx = (b('eyeLookOutRight') + b('eyeLookInLeft') - b('eyeLookInRight') - b('eyeLookOutLeft')) / 2;
  const by = (b('eyeLookDownLeft') + b('eyeLookDownRight') - b('eyeLookUpLeft') - b('eyeLookUpRight')) / 2;
  return { hx, ix: ix / 2, bx, hy, iy: iy / 2, by, lo: lo / 2 };
}
const X = ['hx', 'ix', 'bx'], Y = ['hy', 'iy', 'by', 'lo'];

/** Least squares for target ≈ c0 + Σ ck·featk, with every ck ≥ 0 (features that
 * point the wrong way for this player are dropped, never flipped). */
function fitPositive(rows, targets, ridge = null) {
  const n = rows[0]?.length ?? 0;
  const solve = cols => {
    const k = cols.length + 1, A = Array.from({ length: k }, () => Array(k).fill(0)), y = Array(k).fill(0);
    rows.forEach((r, s) => {
      const v = [1, ...cols.map(c => r[c])];
      for (let i = 0; i < k; i++) { y[i] += v[i] * targets[s]; for (let j = 0; j < k; j++) A[i][j] += v[i] * v[j]; }
    });
    for (let i = 1; i < k; i++) A[i][i] += (ridge ? ridge[cols[i - 1]] : 1e-6) * rows.length;
    // Gaussian elimination.
    for (let c = 0; c < k; c++) {
      let piv = c; for (let r = c + 1; r < k; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      [A[c], A[piv]] = [A[piv], A[c]]; [y[c], y[piv]] = [y[piv], y[c]];
      if (Math.abs(A[c][c]) < 1e-12) return null;
      for (let r = 0; r < k; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let j = c; j < k; j++) A[r][j] -= f * A[c][j]; y[r] -= f * y[c]; }
    }
    return y.map((v, i) => v / A[i][i]);
  };
  let cols = [...Array(n).keys()];
  for (let round = 0; round <= n; round++) {
    const c = solve(cols);
    if (!c) { if (!cols.length) return null; cols = cols.slice(0, -1); continue; }
    const bad = cols.filter((_, i) => c[i + 1] < 0);
    if (!bad.length) { const out = Array(n + 1).fill(0); out[0] = c[0]; cols.forEach((col, i) => { out[col + 1] = c[i + 1]; }); return out; }
    cols = cols.filter(col => !bad.includes(col));
  }
  return null;
}

/** Fit the aim from calibration samples [{ f, target, group }]: each feature is
 * penalised by the jitter it would add to the aim (its noise while the eyes rest
 * on one dot, times its gain), so a feature that barely moves with the gaze but
 * shakes a lot (a still head, a lid that does not narrow) gets a gain near zero
 * instead of a huge one. Gains stay non-negative. → { x: [c0, ...], y: [c0, ...] } or null. */
export function fitGaze(samples, { lambda = 1.5 } = {}) {
  const groups = new Map();
  for (const s of samples) { const g = s.group ?? `${s.target.x},${s.target.y}`; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(s.f); }
  // Noise of each feature: its spread around the mean while looking at one dot, pooled over the dots.
  const noise = Object.fromEntries([...X, ...Y].map(k => {
    let ss = 0, n = 0;
    for (const fs of groups.values()) { if (fs.length < 3) continue; const m = fs.reduce((a, f) => a + f[k], 0) / fs.length; for (const f of fs) { ss += (f[k] - m) ** 2; n++; } }
    return [k, n > 1 ? Math.sqrt(ss / n) : 1e-3];
  }));
  const axis = (keys, t) => fitPositive(samples.map(s => keys.map(k => s.f[k])), samples.map(s => s.target[t]), keys.map(k => lambda * noise[k] ** 2));
  const x = axis(X, 'x'), y = axis(Y, 'y');
  return x && y && [...x, ...y].every(Number.isFinite) ? { x, y, noise } : null;
}
export const GAZE_KEYS = { x: X, y: Y };

// ---- the calibrated aim: ridge regression on expanded features ----------------
// Chosen on a recorded calibration (ml/gaze/analyze.py, select.py): across the screen
// the pupils, the face model's in/out scores and a little head turn, with all their
// pairwise products; up and down the up/down scores and the pupil height, plus products
// with the across signals — the "looking down" score also rises when looking to the
// side, and without those terms the aim sank on one side of the screen.
const BASE = ['ix', 'bx', 'hx', 'iy', 'by'];
const PAIRS = BASE.flatMap((a, i) => BASE.slice(i).map(b => `${a}*${b}`));
const Y_CORE = ['iy', 'by', 'bx', 'ix', 'bx*by', 'bx*iy', 'ix*iy', 'bx*bx'];
export const TERMS = { x: [...BASE, ...PAIRS], y: [...Y_CORE, ...Y_CORE.map(k => `${k}^2`)] };
const clampTo = (v, [lo, hi], margin = .25) => { const m = (hi - lo) * margin; return Math.min(hi + m, Math.max(lo - m, v)); };
export function expand(f, range = null) {
  // Outside what the calibration saw, products run wild: hold each signal within that range (plus a margin).
  if (range) { f = { ...f }; for (const k of BASE) f[k] = clampTo(f[k], range[k]); }
  const e = { ...f };
  for (const k of PAIRS) { const [a, b] = k.split('*'); e[k] = f[a] * f[b]; }
  for (const k of Y_CORE) e[`${k}^2`] = e[k] * e[k];
  return e;
}
function solveLinear(A, y) {
  const k = y.length; A = A.map(r => [...r]); y = [...y];
  for (let c = 0; c < k; c++) {
    let piv = c; for (let r = c + 1; r < k; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]]; [y[c], y[piv]] = [y[piv], y[c]];
    if (Math.abs(A[c][c]) < 1e-12) return null;
    for (let r = 0; r < k; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let j = c; j < k; j++) A[r][j] -= f * A[c][j]; y[r] -= f * y[c]; }
  }
  return y.map((v, i) => v / A[i][i]);
}
/** Ridge regression on standardised terms; samples [{ e (expanded features), t (target), w }]. */
function ridgeAxis(samples, terms, lambda) {
  const n = terms.length, W = samples.reduce((a, s) => a + s.w, 0);
  const mu = terms.map(k => samples.reduce((a, s) => a + s.w * s.e[k], 0) / W);
  const sd = terms.map((k, i) => Math.sqrt(samples.reduce((a, s) => a + s.w * (s.e[k] - mu[i]) ** 2, 0) / W) + 1e-9);
  const ym = samples.reduce((a, s) => a + s.w * s.t, 0) / W;
  const A = Array.from({ length: n }, () => Array(n).fill(0)), b = Array(n).fill(0);
  for (const s of samples) {
    const z = terms.map((k, i) => (s.e[k] - mu[i]) / sd[i]);
    for (let i = 0; i < n; i++) { b[i] += s.w * z[i] * (s.t - ym); for (let j = 0; j < n; j++) A[i][j] += s.w * z[i] * z[j]; }
  }
  for (let i = 0; i < n; i++) A[i][i] += lambda * W;
  const c = solveLinear(A, b);
  return c && { terms, mu, sd, c, b: ym };
}
const predictAxis = (m, e) => m.c.reduce((a, c, i) => a + c * (e[m.terms[i]] - m.mu[i]) / m.sd[i], m.b);
const corr = (a, b) => {
  const ma = a.reduce((s, v) => s + v, 0) / a.length, mb = b.reduce((s, v) => s + v, 0) / b.length;
  let sab = 0, sa = 0, sb = 0; a.forEach((v, i) => { sab += (v - ma) * (b[i] - mb); sa += (v - ma) ** 2; sb += (b[i] - mb) ** 2; });
  return sab / Math.sqrt(sa * sb + 1e-18);
};
/** Fit the aim from calibration samples [{ f, target, weight? }] (fixed dots and a gliding
 * dot, the glide's target taken a moment back to follow the eyes' lag). Returns a model for
 * createGaze().load(), or null when the fit does not really follow the gaze. */
export function trainGaze(samples, { lambda = .01 } = {}) {
  const good = samples.filter(s => s.f && s.target);
  // Enough of the screen covered (a gliding dot, or many dots), or the plain fit is safer.
  const spots = new Set(good.map(s => `${Math.round(s.target.x * 10)},${Math.round(s.target.y * 10)}`));
  if (good.length < 60 || spots.size < 20) return null;
  const range = Object.fromEntries([...BASE, 'hy'].map(k => { const v = good.map(s => s.f[k]); return [k, [Math.min(...v), Math.max(...v)]]; }));
  const rows = good.map(s => ({ e: expand(s.f, range), tx: s.target.x, ty: s.target.y, w: s.weight ?? 1 }));
  const x = ridgeAxis(rows.map(r => ({ e: r.e, t: r.tx, w: r.w })), TERMS.x, lambda);
  const y = ridgeAxis(rows.map(r => ({ e: r.e, t: r.ty, w: r.w })), TERMS.y, lambda);
  if (!x || !y) return null;
  // It must follow the dot, the right way round, over the whole calibration.
  const px = rows.map(r => predictAxis(x, r.e)), py = rows.map(r => predictAxis(y, r.e));
  if (corr(px, rows.map(r => r.tx)) < .8 || corr(py, rows.map(r => r.ty)) < .7) return null;
  return { v: 2, x, y, range };
}

/** Online correction while playing: whenever the player surely looks at something (a rock
 * held in the sights), the miss (target − aim) is kept, filed under the part of the screen
 * the aim was in (a 3 × 3 grid). The aim is shifted by the median miss of its part, blended
 * smoothly between parts, falling back to the median of all recent misses where a part has
 * no data yet. Only shifts: the aim is too noisy up and down to refit its scale on the fly
 * (tried on a recorded game — it squashed the aim and doubled the error). A recorded game
 * also showed the miss differs across the screen (right side about twice the left), which a
 * single shift cannot follow. */
// Tuned on a recorded 200 s game (ml/gaze/play_review.mjs): a fast shift over the last ~40
// locked frames plus slow zones kept the aim within 80 px of the burned rock 68 % of the time,
// against 32 % for one slow shift.
export function createAdapter({ keep = 40, perCell = 80, limit = .25, grid = 3, zones = true } = {}) {
  // A fast shift from the latest misses follows the drift of the pose; on top of it, a slow
  // per-zone remainder (the miss minus the shift of that moment) follows where on the screen
  // the aim is off more or less.
  const all = [], cells = Array.from({ length: grid * grid }, () => []);
  let global = { x: 0, y: 0 }, rest = Array(grid * grid).fill(null);
  const median = a => { const b = [...a].sort((p, q) => p - q); return b[Math.floor(b.length / 2)]; };
  const clamp = v => Math.max(-limit, Math.min(limit, v));
  const cellOf = p => Math.min(grid - 1, Math.max(0, Math.floor(p.y * grid))) * grid + Math.min(grid - 1, Math.max(0, Math.floor(p.x * grid)));
  const restAt = aim => {
    if (!zones) return { x: 0, y: 0 };
    const gx = Math.min(grid - 1, Math.max(0, aim.x * grid - .5)), gy = Math.min(grid - 1, Math.max(0, aim.y * grid - .5));
    const x0 = Math.floor(gx), y0 = Math.floor(gy), x1 = Math.min(grid - 1, x0 + 1), y1 = Math.min(grid - 1, y0 + 1), fx = gx - x0, fy = gy - y0;
    const at = (cx, cy) => rest[cy * grid + cx] ?? { x: 0, y: 0 };
    const mix = (a, b, f) => ({ x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f });
    return mix(mix(at(x0, y0), at(x1, y0), fx), mix(at(x0, y1), at(x1, y1), fx), fy);
  };
  return {
    get fix() { return { global, zones: rest }; },
    learn(aim, _head, target) {
      const miss = { x: target.x - aim.x, y: target.y - aim.y };
      all.push(miss); if (all.length > keep) all.shift();
      if (all.length >= 20) global = { x: clamp(median(all.map(m => m.x))), y: clamp(median(all.map(m => m.y))) };
      const cell = cells[cellOf(aim)]; cell.push({ x: miss.x - global.x, y: miss.y - global.y }); if (cell.length > perCell) cell.shift();
      rest = cells.map(c => c.length >= 20 ? { x: clamp(median(c.map(m => m.x)) * .7), y: clamp(median(c.map(m => m.y)) * .7) } : null);
    },
    apply(aim) { const r = restAt(aim); return { x: aim.x + global.x + r.x, y: aim.y + global.y + r.y }; },
  };
}

/** Recalibration during play from confirmed labels: pairs (aim, where the player surely
 * looked) — frames of a rock burned until it exploded, matched by movement. The fit goes the
 * right way round: aim = a + b·true per axis (the aim carries the noise, so regressing the aim
 * on the truth does not squash the scale, unlike the other way round), with Huber weights
 * against outliers, recent labels weighing more, and a prior at "no change" that fades as
 * labels come in. The correction is its inverse: true = (aim − a) / b. */
export function createRecalibrator({ prior = 30, halfLife = 20, keep = 3000, minB = .5, maxB = 2 } = {}) {
  let labels = [], fix = { x: [0, 1], y: [0, 1] };
  const fitAxis = (key, now) => {
    let a = 0, b = 1;
    for (let round = 0; round < 3; round++) {
      // Residual scale for the Huber weights (median absolute residual).
      const res = labels.map(l => l.aim[key] - (a + b * l.target[key]));
      const mad = [...res.map(Math.abs)].sort((p, q) => p - q)[Math.floor(res.length / 2)] || .05;
      let S = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
      const add = (x, y, w) => { S += w; Sx += w * x; Sy += w * y; Sxx += w * x * x; Sxy += w * x * y; };
      labels.forEach((l, i) => {
        const r = Math.abs(res[i]) / (1.5 * mad + 1e-6), huber = r <= 1 ? 1 : 1 / r;
        add(l.target[key], l.aim[key], huber * Math.pow(.5, (now - l.t) / halfLife));
      });
      for (const x of [.15, .5, .85]) add(x, x, prior / 3 * 60 / (60 + S));   // "no change" until labels say otherwise; fades as they pile up
      const det = S * Sxx - Sx * Sx;
      if (Math.abs(det) < 1e-12) break;
      b = Math.max(minB, Math.min(maxB, (S * Sxy - Sx * Sy) / det)); a = (Sy - b * Sx) / S;
    }
    return [a, b];
  };
  return {
    get fix() { return fix; },
    get count() { return labels.length; },
    /** pairs: [{ aim: {x, y}, target: {x, y} }] at game time `now` (seconds). */
    add(pairs, now) {
      for (const p of pairs) if (p.aim && p.target) labels.push({ ...p, t: now });
      if (labels.length > keep) labels = labels.slice(-keep);
      if (labels.length >= 10) fix = { x: fitAxis('x', now), y: fitAxis('y', now) };
    },
    apply(aim) { return { x: (aim.x - fix.x[0]) / fix.x[1], y: (aim.y - fix.y[0]) / fix.y[1] }; },
    reset() { labels = []; fix = { x: [0, 1], y: [0, 1] }; },
  };
}

// Screen widths/heights per unit of head turn/tilt beyond the calibrated range (from a recorded session).
const HEAD_PRIOR = { x: 1.4, y: 2 };

/** Does the face now sit roughly as it did during this model's calibration? (else calibrate again) */
export function samePose(model, fs) {
  if (model?.v !== 2 || !fs.length) return false;
  const avg = k => fs.reduce((a, f) => a + f[k], 0) / fs.length;
  return ['hx', 'hy'].every(k => { const [lo, hi] = model.range[k], span = Math.max(hi - lo, .02); return avg(k) > lo - span && avg(k) < hi + span; });
}

// Gains before (or without) a calibration: turning the head a little, or the eyes, sweeps the screen.
const DEFAULT_GAIN = { x: [2.2, 2.6, .5], y: [2.6, 3, .6, 2] };

// The in-play correction (createAdapter) is off: pulling the aim onto the rock held in the
// sights made it jump away when the player looked at another one ("it keeps knocking the aim
// off"). The calibration before every game does the job instead.
export function createGaze({ minCutoff = .6, beta = .004, adapt = false } = {}) {
  let samples = [], model = null, lastT = null, last = null, bias = { x: 0, y: 0 }, lastF = null, adapter = createAdapter(), lastCentred = null;
  const recal = createRecalibrator();
  // Up and down the webcam gaze is noisier: smoothed harder.
  const sx = createOneEuro({ minCutoff, beta, dCutoff: 1 }), sy = createOneEuro({ minCutoff: minCutoff * .6, beta, dCutoff: 1 });
  // Without a fit: default gains around the face's resting position (taken from the middle dot when there is one).
  const fallback = () => {
    const mid = samples.filter(s => Math.abs(s.target.x - .5) < .05 && Math.abs(s.target.y - .5) < .05).map(s => s.f);
    const avg = k => mid.length ? mid.reduce((a, f) => a + f[k], 0) / mid.length : ({ hy: .45, lo: -.3 })[k] ?? 0;
    const axis = (keys, gains) => [.5 - keys.reduce((a, k, i) => a + gains[i] * avg(k), 0), ...gains];
    return { x: axis(X, DEFAULT_GAIN.x), y: axis(Y, DEFAULT_GAIN.y) };
  };
  let rough = fallback();
  const apply = (c, f, keys) => keys.reduce((a, key, i) => a + c[i + 1] * f[key], c[0]);
  const raw = f => {
    const k = model ?? rough;
    if (k.v === 2) {
      const e = expand(f, k.range);
      // A head turned further than during calibration: what the model cannot know, a plain rule
      // adds — the gaze goes where the head turns (eyes relative to the head are already in the model).
      const over = key => k.range[key] ? f[key] - clampTo(f[key], k.range[key]) : 0;
      return { x: predictAxis(k.x, e) + HEAD_PRIOR.x * over('hx'), y: predictAxis(k.y, e) + HEAD_PRIOR.y * over('hy') };
    }
    return { x: apply(k.x, f, X), y: apply(k.y, f, Y) };
  };
  const centred = f => { const r = raw(f); return { x: r.x + bias.x, y: r.y + bias.y }; };
  const map = f => recal.apply(adapt ? adapter.apply(centred(f)) : centred(f));
  return {
    features: gazeFeatures,
    addSample(f, target) { if (f) samples.push({ f, target }); },
    clear() { samples = []; model = null; rough = fallback(); bias = { x: 0, y: 0 }; adapter = createAdapter(); recal.reset(); },
    /** The recalibration during play (labels from rocks burned down, see createRecalibrator). */
    recal,
    /** The last aim before the recalibration (what the labels are paired with). */
    get lastCentred() { return lastCentred; },
    /** The player surely looks at this point now (a rock held in the sights): learn the miss. */
    learn(target) { if (adapt && lastF) adapter.learn(centred(lastF), lastF, target); },
    get shift() { return adapter.fix; },
    get bias() { return bias; },
    get model() { return model; },
    get calibrated() { return Boolean(model); },
    /** Use a model fitted elsewhere (the calibration page); then only the centre is checked. */
    load(saved) {
      const ok = saved?.v === 2 ? saved.x?.terms?.join() === TERMS.x.join() && saved.y?.terms?.join() === TERMS.y.join() : saved?.x?.length === X.length + 1 && saved?.y?.length === Y.length + 1;
      if (ok) { model = saved.v === 2 ? saved : { x: saved.x, y: saved.y }; bias = { x: 0, y: 0 }; }
      return Boolean(ok);
    },
    /** Shift the aim so that these features (taken looking at a known point) land on it. */
    recenter(fs, target = { x: .5, y: .5 }) {
      if (!fs.length) return;
      const avg = key => fs.reduce((a, f) => a + f[key], 0) / fs.length, f = Object.fromEntries([...X, ...Y].map(key => [key, avg(key)]));
      const r = raw(f); bias = { x: target.x - r.x, y: target.y - r.y };
    },
    /** The aim for one set of features, without smoothing (replays, tests). */
    aimOf: f => map(f),
    /** What the calibration saw and fitted (for tuning with a real player). */
    debug() { return { samples: samples.map(s => ({ target: s.target, f: s.f })), model, rough, bias, keys: { x: X, y: Y } }; },
    solve() {
      rough = fallback(); bias = { x: 0, y: 0 };
      if (samples.length < 12) return false;
      // The expanded model first; the plain one (never mirrored) if that does not hold up.
      const best = trainGaze(samples);
      if (best) { model = best; return true; }
      // One robust row per dot: the median of each feature (blinks and glances do not count).
      const groups = new Map();
      for (const s of samples) { const key = `${s.target.x},${s.target.y}`; if (!groups.has(key)) groups.set(key, { target: s.target, fs: [] }); groups.get(key).fs.push(s.f); }
      const median = v => { const a = [...v].sort((p, q) => p - q); return a[Math.floor(a.length / 2)]; };
      const rows = [...groups.values()].filter(g => g.fs.length >= 4).map(g => ({ target: g.target, f: Object.fromEntries([...X, ...Y].map(k => [k, median(g.fs.map(f => f[k]))])) }));
      if (rows.length < 4) return false;
      const fitted = fitGaze(samples);
      if (!fitted) return false;
      const { x, y } = fitted;
      // The fit must actually move the aim: looking left and right should span the screen.
      const xs = rows.map(s => apply(x, s.f, X)), ys = rows.map(s => apply(y, s.f, Y));
      if (Math.max(...xs) - Math.min(...xs) < .3 || Math.max(...ys) - Math.min(...ys) < .25) return false;
      model = { x, y };
      return true;
    },
    /** Smoothed gaze point on the screen (0..1), held for a moment through blinks. */
    point(frame, t, width = innerWidth, height = innerHeight) {
      const f = gazeFeatures(frame);
      if (!f) return last && lastT !== null && t - lastT < 400 ? last : null;
      lastF = f; lastCentred = centred(f);
      if (lastT !== null && t === lastT) return last;                  // the same camera frame again
      const raw = map(f), dt = lastT === null ? 1 / 30 : Math.min(.1, Math.max(.005, (t - lastT) / 1000));
      lastT = t;
      const x = sx.filter(Math.min(1.05, Math.max(-.05, raw.x)) * width, dt) / width, y = sy.filter(Math.min(1.05, Math.max(-.05, raw.y)) * height, dt) / height;
      last = { x: Math.min(1, Math.max(0, x)), y: Math.min(1, Math.max(0, y)) };
      return last;
    },
  };
}
