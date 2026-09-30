/** Exercises: how a repetition looks (one number that goes down and back up,
 * or up and back down), when it counts, and what is wrong with it. Every
 * mistake has a concrete hint and the body parts to highlight.
 *
 * A definition:
 *   signal(m)        → number | null from measureBody() (null: cannot tell)
 *   direction        'down' (the number falls during the rep: knee angle) or 'up'
 *   start / target / finish   thresholds: leaving start begins a rep, reaching
 *                    target is a full rep, getting back past finish ends it
 *   needs            setup: fullBody, view ('front' | 'side' | null)
 *   rules            { code, text, parts, when: 'during' | 'extreme' | 'rep', check }
 *                    'during' runs every frame of the rep (must hold 250 ms),
 *                    'extreme' on the frame deepest into the rep, 'rep' on the finished rep.
 */
const ok = a => a && a.value !== null && a.confidence >= .5;
const mean = (...values) => { const v = values.filter(x => x !== null && Number.isFinite(x)); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
const both = pair => [pair.left, pair.right].filter(ok).map(a => a.value);
const lowest = pair => { const v = both(pair); return v.length ? Math.min(...v) : null; };

export const EXERCISES = {
  squat: {
    title: 'Приседания', direction: 'down', start: 150, target: 100, finish: 160,
    needs: { fullBody: true },
    signal: m => { const v = both(m.angles.knee); return v.length ? mean(...v) : null; },
    rules: [
      { code: 'squat-depth', when: 'rep', parts: ['leftHip', 'rightHip', 'leftKnee', 'rightKnee'],
        text: 'Сядь глубже: опускай таз, пока бёдра не станут параллельны полу',
        check: (rep, def) => rep.extreme > def.target },
      { code: 'knees-in', when: 'during', parts: ['leftKnee', 'rightKnee'],
        text: 'Колени заваливаются внутрь: разводи их в стороны, по линии носков',
        check: m => (ok(m.angles.kneeDrift.left) && m.angles.kneeDrift.left.value < -12) || (ok(m.angles.kneeDrift.right) && m.angles.kneeDrift.right.value < -12)
          || (m.measures.kneeToAnkleWidth !== null && m.measures.kneeToAnkleWidth < .75 && lowest(m.angles.knee) < 130) },
      { code: 'knees-past-toes', when: 'extreme', parts: ['leftKnee', 'rightKnee', 'leftFoot', 'rightFoot'],
        text: 'Колени выходят за носки: сядь глубже назад, будто на стул, вес на пятки',
        check: m => m.measures.kneePastToe !== null && m.measures.kneePastToe > .1 },
      { code: 'lean-forward', when: 'extreme', parts: ['back', 'torso'],
        text: 'Сильно наклоняешься вперёд: держи грудь выше и спину прямой, отводи таз назад',
        check: m => m.angles.trunk && m.angles.trunk.confidence >= .5 && m.angles.trunk.forward > 50 },
      { code: 'uneven', when: 'during', parts: ['leftKnee', 'rightKnee', 'hips'],
        text: 'Садишься на одну ногу: распредели вес поровну на обе',
        check: m => m.measures.kneeAsymmetry !== null && Math.abs(m.measures.kneeAsymmetry) > 22 && lowest(m.angles.knee) < 135 },
      { code: 'too-fast', when: 'rep', parts: ['leftLeg', 'rightLeg'],
        text: 'Слишком быстро: опускайся плавно, примерно за секунду',
        check: rep => rep.downMs < 450 },
    ],
  },
  lunge: {
    title: 'Выпады', direction: 'down', start: 145, target: 105, finish: 160,
    needs: { fullBody: true },
    signal: m => lowest(m.angles.knee),
    rules: [
      { code: 'lunge-depth', when: 'rep', parts: ['leftKnee', 'rightKnee'],
        text: 'Опустись ниже: заднее колено почти до пола, оба колена под прямым углом',
        check: (rep, def) => rep.extreme > def.target },
      { code: 'torso-lean', when: 'extreme', parts: ['back', 'torso'],
        text: 'Держи корпус вертикально, не наклоняйся вперёд',
        check: m => m.angles.trunk && m.angles.trunk.confidence >= .5 && Math.abs(m.angles.trunk.forward) > 30 },
      { code: 'knee-past-toes', when: 'extreme', parts: ['leftKnee', 'rightKnee', 'leftFoot', 'rightFoot'],
        text: 'Колено уходит за носок: сделай шаг шире',
        check: m => m.measures.frontKneePastToe !== null && m.measures.frontKneePastToe > .06 },
    ],
  },
  jumpingJack: {
    title: 'Прыжки «звёздочка»', direction: 'up', start: .35, target: .8, finish: .2,
    needs: { fullBody: true, view: 'front' },
    // 0 closed … 1 open: arms overhead and feet wide.
    signal: m => {
      const arms = mean(...both(m.angles.shoulder)), stance = m.measures.stanceToHipWidth;
      if (arms === null || stance === null) return null;
      return Math.min(Math.max(0, Math.min(1, (arms - 30) / 120)), Math.max(0, Math.min(1, (stance - 1.1) / .8)));
    },
    rules: [
      { code: 'arms-low', when: 'extreme', parts: ['leftArm', 'rightArm'],
        text: 'Поднимай руки выше: ладони встречаются над головой',
        check: m => { const a = both(m.angles.shoulder); return a.length > 0 && Math.min(...a) < 145; } },
      { code: 'legs-narrow', when: 'extreme', parts: ['leftLeg', 'rightLeg'],
        text: 'Шире ноги в прыжке: стопы дальше плеч',
        check: m => m.measures.stanceToHipWidth !== null && m.measures.stanceToHipWidth < 1.7 },
    ],
  },
  highKnees: {
    title: 'Колени к груди', direction: 'down', start: 150, target: 110, finish: 160,
    needs: { fullBody: true },
    signal: m => lowest(m.angles.hip),
    rules: [
      { code: 'knee-low', when: 'rep', parts: ['leftHip', 'rightHip', 'leftKnee', 'rightKnee'],
        text: 'Выше колено: поднимай бедро до уровня пояса',
        check: (rep, def) => rep.extreme > def.target },
      { code: 'lean-back', when: 'extreme', parts: ['back', 'torso'],
        text: 'Не заваливайся назад: корпус прямо, колено тянется к груди',
        check: m => m.angles.trunk && m.angles.trunk.confidence >= .5 && m.angles.trunk.forward < -15 },
    ],
  },
  armRaise: {
    title: 'Руки вверх', direction: 'up', start: 95, target: 150, finish: 60,
    needs: { fullBody: false },
    signal: m => lowest(m.angles.shoulder),
    rules: [
      { code: 'arms-not-overhead', when: 'rep', parts: ['leftArm', 'rightArm'],
        text: 'Поднимай руки выше: прямо над головой',
        check: (rep, def) => rep.extreme < def.target },
      { code: 'elbows-bent', when: 'extreme', parts: ['leftElbow', 'rightElbow'],
        text: 'Выпрями руки в локтях',
        check: m => { const e = both(m.angles.elbow); return e.length > 0 && Math.min(...e) < 145; } },
      { code: 'arms-uneven', when: 'extreme', parts: ['leftArm', 'rightArm'],
        text: 'Одна рука ниже другой: поднимай обе одинаково',
        check: m => ok(m.angles.shoulder.left) && ok(m.angles.shoulder.right) && Math.abs(m.angles.shoulder.left.value - m.angles.shoulder.right.value) > 25 },
    ],
  },
};

/** Counts repetitions of one exercise and collects their mistakes.
 * update(measure, t) → { phase, signal, rep | null (finished this frame), active: [{code,text,parts}] }
 * Start/finish adapt to how this person stands at rest (a knee that reads
 * 158° standing still ends its squats); the target (depth) stays absolute. */
export function createRepCounter(definition, { minRepMs = 350, lostAfterMs = 1500, holdMs = 250, restMs = 3000 } = {}) {
  const sign = definition.direction === 'down' ? 1 : -1;       // compare as a falling signal
  const below = (a, b) => sign * a < sign * b;
  const rest = [];                                              // { t, v } while not in a rep
  let def = definition;
  let phase = 'ready', rep = null, lastSeen = null, count = 0, good = 0;
  function adapt(t, v) {
    rest.push({ t, v });
    while (rest.length && t - rest[0].t > restMs) rest.shift();
    if (rest.length < 5) return;
    // Resting level: the 80th percentile towards "rest" (straight legs, arms down).
    const sorted = rest.map(r => sign * r.v).sort((a, b) => a - b);
    const level = sign * sorted[Math.floor(sorted.length * .8)];
    const gap = Math.abs(definition.finish - definition.start);
    const finish = sign > 0 ? Math.min(definition.finish, level - 6) : Math.max(definition.finish, level + 6);
    const start = sign > 0 ? Math.min(definition.start, finish - gap) : Math.max(definition.start, finish + gap);
    // Never let the rep start beyond the depth it has to reach.
    def = below(start, definition.target) ? definition : { ...definition, start, finish };
  }
  const pending = new Map();     // 'during' rule → first time seen in this rep
  const tally = {};

  function finish(t) {
    const done = { index: count + 1, extreme: rep.extreme, full: sign * rep.extreme <= sign * def.target,
      durationMs: t - rep.startT, downMs: rep.extremeT - rep.startT, upMs: t - rep.extremeT, errors: [] };
    for (const rule of def.rules) {
      if (rule.when === 'rep' && rule.check(done, def)) rep.errors.set(rule.code, rule);
    }
    done.errors = [...rep.errors.values()].map(({ code, text, parts }) => ({ code, text, parts }));
    for (const e of done.errors) tally[e.code] = (tally[e.code] ?? 0) + 1;
    count++; if (!done.errors.length) good++;
    rep = null; pending.clear();
    return done;
  }

  return {
    get count() { return count; },
    get good() { return good; },
    get tally() { return { ...tally }; },
    update(m, t) {
      const signal = m ? def.signal(m) : null;
      const out = { phase, signal, rep: null, active: [] };
      if (signal === null) {
        if (rep && lastSeen !== null && t - lastSeen > lostAfterMs) { rep = null; phase = 'ready'; pending.clear(); }
        out.phase = phase; return out;
      }
      lastSeen = t;
      if (!rep && phase === 'ready') adapt(t, signal);
      if (!rep) {
        if (below(signal, def.start)) { rep = { startT: t, extreme: signal, extremeT: t, errors: new Map(), extremeMeasure: m }; phase = 'going'; }
        else phase = 'ready';
      }
      if (rep) {
        if (below(signal, rep.extreme)) { rep.extreme = signal; rep.extremeT = t; rep.extremeMeasure = m; }
        phase = !below(def.target, signal) ? 'bottom' : t - rep.extremeT > 150 ? 'returning' : 'going';
        for (const rule of def.rules) {
          if (rule.when !== 'during') continue;
          if (rule.check(m, def)) {
            if (!pending.has(rule.code)) pending.set(rule.code, t);
            if (t - pending.get(rule.code) >= holdMs) { rep.errors.set(rule.code, rule); out.active.push({ code: rule.code, text: rule.text, parts: rule.parts }); }
          } else pending.delete(rule.code);
        }
        if (below(def.finish, signal)) {
          // Back at the start: judge the deepest frame, then close the rep.
          for (const rule of def.rules) if (rule.when === 'extreme' && rule.check(rep.extremeMeasure, def)) rep.errors.set(rule.code, rule);
          if (t - rep.startT >= minRepMs && below(rep.extreme, def.start)) out.rep = finish(t);
          else { rep = null; pending.clear(); }
          phase = 'ready';
        }
      }
      out.phase = phase;
      return out;
    },
    reset() { phase = 'ready'; rep = null; lastSeen = null; count = 0; good = 0; pending.clear(); rest.length = 0; def = definition; for (const k of Object.keys(tally)) delete tally[k]; },
  };
}
