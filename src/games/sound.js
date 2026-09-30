/** Game sounds through Web Audio.
 * - One-shots from the Kenney audio packs (CC0): crash, countdown, hints, menu.
 * - The engine is a real recording: a dyno pull of a turbo car rising steadily
 *   from ~2500 to ~7500 rpm ("Import car revs on Chassis Dyno", Freesound, CC0).
 *   Short overlapping grains are played from the spot of the recording where the
 *   engine was at the current rpm, so the pitch and the timbre both follow the
 *   revs without stretching one loop. A gear change lets the revs fall back
 *   smoothly, the way a real gearbox does; off throttle the sound goes duller.
 *   Nitro whooshes only while it is on; braking screeches the tyres (a recorded
 *   squeal, CC0), louder the faster the car goes.
 * Browsers start audio only after a click or key press on the page; until then
 * everything is silent and `needsUnlock` is true. */
import { createMenuMusic } from './music.js';

const FILES = ['nitro', 'brake.wav', 'space-explode1', 'space-explode2', 'space-boom', 'space-shield', 'space-zap', 'space-hum', 'space-hit', 'crash', 'crash2', 'tick', 'go', 'hint', 'select', 'finish', 'engine-sweep.m4a',
  // Table tennis (Freesound, CC0): paddle hits, table and floor bounces, a whoosh, the crowd.
  ...['pp-hit1', 'pp-hit2', 'pp-hit3', 'pp-table1', 'pp-table2', 'pp-table3', 'pp-floor', 'whoosh', 'club-loop'].map(n => `${n}.wav`),
  ...['crowd-loop', 'applause', 'cheer'].map(n => `${n}.m4a`)];

// Where in engine-sweep the engine plays each pitch: [seconds, Hz], measured once.
const SWEEP = [[0.275, 153.4], [0.355, 159.9], [2.164, 166.6], [4.625, 173.6], [7.296, 180.9], [8.379, 188.5], [8.773, 196.4], [9.377, 204.7],
  [9.6, 213.3], [10.033, 222.3], [10.38, 231.6], [10.759, 241.4], [11.019, 251.5], [11.3, 262.1], [11.524, 273.1], [11.762, 284.6], [11.899, 296.6],
  [12.107, 309.0], [12.339, 322.0], [12.542, 335.6], [12.772, 349.7], [13.121, 364.4], [13.427, 379.7], [13.775, 395.7]];
const RPM_LOW = 2400, RPM_HIGH = 7600, GRAIN = .09, HOP = GRAIN / 2;

/** rpm → { at: seconds into the sweep, rate: playback rate }. */
function sweepAt(rpm) {
  const lo = SWEEP[0][1], hi = SWEEP[SWEEP.length - 1][1];
  const u = Math.min(1, Math.max(0, Math.log(rpm / RPM_LOW) / Math.log(RPM_HIGH / RPM_LOW)));
  const hz = lo * (hi / lo) ** u;
  let i = 1; while (i < SWEEP.length - 1 && SWEEP[i][1] < hz) i++;
  const [t0, f0] = SWEEP[i - 1], [t1, f1] = SWEEP[i];
  return { at: t0 + (t1 - t0) * Math.min(1, Math.max(0, (hz - f0) / (f1 - f0))), rate: rpm < RPM_LOW ? Math.max(.55, rpm / RPM_LOW) : 1 };
}

function noiseBuffer(ctx, seconds = 2) {
  const buffer = ctx.createBuffer(1, ctx.sampleRate * seconds, ctx.sampleRate), data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  return buffer;
}

export function createSound(base = '/games/sounds') {
  const Context = window.AudioContext || window.webkitAudioContext;
  const ctx = Context ? new Context() : null;
  const buffers = {};
  let master = null, engine = null;
  if (ctx) { master = ctx.createGain(); master.gain.value = .8; master.connect(ctx.destination); }
  const loaded = ctx ? Promise.all(FILES.map(file => fetch(`${base}/${file.includes('.') ? file : `${file}.ogg`}`).then(r => r.arrayBuffer()).then(b => ctx.decodeAudioData(b))
    .then(buffer => { buffers[file.split('/').pop().split('.')[0]] = buffer; }).catch(() => {}))) : Promise.resolve();
  const unlock = () => ctx?.resume();
  addEventListener('pointerdown', unlock); addEventListener('keydown', unlock);
  const running = () => ctx && ctx.state === 'running';

  /** A one-shot; pan −1 (left) .. 1 (right). */
  function play(name, { volume = 1, rate = 1, pan = 0 } = {}) {
    if (!running() || !buffers[name]) return;
    const src = ctx.createBufferSource(), gain = ctx.createGain();
    src.buffer = buffers[name]; src.playbackRate.value = rate; gain.gain.value = volume;
    let node = src.connect(gain);
    if (pan && ctx.createStereoPanner) { const p = ctx.createStereoPanner(); p.pan.value = Math.max(-1, Math.min(1, pan)); node = node.connect(p); }
    node.connect(master); src.start();
  }

  /** A looping bed (crowd murmur); returns a handle to fade or stop it. */
  function loop(name, { volume = .3 } = {}) {
    let src = null, gain = null, stopped = false;
    const start = () => {
      if (stopped || src || !running() || !buffers[name]) return;
      src = ctx.createBufferSource(); src.buffer = buffers[name]; src.loop = true;
      gain = ctx.createGain(); gain.gain.value = 0; gain.gain.setTargetAtTime(volume, ctx.currentTime, .5);
      src.connect(gain).connect(master); src.start();
    };
    start();
    return {
      tick: start,                                   // starts once audio is unlocked and loaded
      stop() { stopped = true; if (src) { gain.gain.setTargetAtTime(0, ctx.currentTime, .3); src.stop(ctx.currentTime + 1.5); src = null; } },
    };
  }

  function buildEngine() {
    const out = ctx.createGain(); out.gain.value = 0;
    const tone = ctx.createBiquadFilter(); tone.type = 'lowpass'; tone.frequency.value = 3000; tone.Q.value = .5;
    const body = ctx.createBiquadFilter(); body.type = 'lowshelf'; body.frequency.value = 200; body.gain.value = 4;
    out.connect(body).connect(tone).connect(master);
    // Nitro hiss: band-passed noise.
    const hissSrc = ctx.createBufferSource(); hissSrc.buffer = noiseBuffer(ctx); hissSrc.loop = true;
    const hissBand = ctx.createBiquadFilter(); hissBand.type = 'bandpass'; hissBand.frequency.value = 1800; hissBand.Q.value = .8;
    const hiss = ctx.createGain(); hiss.gain.value = 0;
    hissSrc.connect(hissBand).connect(hiss).connect(master); hissSrc.start();
    // Tyre screech: a recorded squeal looping quietly, its level set by the brake.
    const squeal = ctx.createGain(); squeal.gain.value = 0;
    const soften = ctx.createBiquadFilter(); soften.type = 'lowpass'; soften.frequency.value = 2400;
    squeal.connect(soften).connect(master);
    const e = { out, tone, hiss, hissBand, squeal, screech: null, nitroSrc: null, nitroOn: false, target: null, rpm: null, last: null, next: ctx.currentTime + .05, on: true };
    // Grains are scheduled a little ahead of the audio clock; the timer only tops the queue up.
    e.timer = setInterval(() => {
      if (!e.on || e.target === null || !buffers['engine-sweep']) return;
      const now = ctx.currentTime;
      if (e.next < now) e.next = now + .01;
      while (e.next < now + .12) { grain(e, e.next); e.next += HOP; }
    }, 25);
    return e;
  }

  /** One grain at audio time t: the revs glide towards the target (quick up, softer down). */
  function grain(e, t) {
    const dt = e.last === null ? HOP : t - e.last; e.last = t;
    if (e.rpm === null) e.rpm = e.target;
    const tau = e.target > e.rpm ? .07 : .13;
    e.rpm += (e.target - e.rpm) * (1 - Math.exp(-dt / tau));
    const { at, rate } = sweepAt(e.rpm);
    const buffer = buffers['engine-sweep'];
    const offset = Math.min(buffer.duration - GRAIN * 2, Math.max(0, at + (Math.random() - .5) * .03));
    const src = ctx.createBufferSource(), g = ctx.createGain();
    src.buffer = buffer; src.playbackRate.value = rate;
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(1, t + GRAIN / 2); g.gain.linearRampToValueAtTime(0, t + GRAIN);
    src.connect(g).connect(e.out); src.start(t, offset, GRAIN * rate + .01); src.stop(t + GRAIN + .02);
  }

  /** The nitro whoosh starts with the boost and fades out the moment it ends. */
  function nitroSample(e, on) {
    if (on === e.nitroOn) return;
    e.nitroOn = on;
    const now = ctx.currentTime;
    if (on && buffers.nitro) {
      const src = ctx.createBufferSource(), g = ctx.createGain();
      src.buffer = buffers.nitro; g.gain.setValueAtTime(0, now); g.gain.linearRampToValueAtTime(.7, now + .04);
      src.connect(g).connect(master); src.start(); e.nitroSrc = { src, g };
    } else if (!on && e.nitroSrc) {
      const { src, g } = e.nitroSrc; e.nitroSrc = null;
      g.gain.cancelScheduledValues(now); g.gain.setValueAtTime(g.gain.value, now); g.gain.setTargetAtTime(0, now, .06); src.stop(now + .4);
    }
  }

  /** rpm, throttle 0..1, nitro, brake 0..1 (how hard the tyres scrub); null stops everything. */
  function engineAt(rpm, { throttle = 1, nitro = false, brake = 0 } = {}) {
    if (!running()) return;
    const now = ctx.currentTime;
    if (rpm === null) {
      if (engine) {
        for (const g of [engine.out.gain, engine.hiss.gain, engine.squeal.gain]) g.setTargetAtTime(0, now, .08);
        engine.target = null; engine.rpm = null; engine.last = null; nitroSample(engine, false);
      }
      return;
    }
    engine ??= buildEngine();
    engine.target = rpm;
    if (!engine.dip) engine.out.gain.setTargetAtTime(.55 + throttle * .35 + (nitro ? .1 : 0), now, .08);
    engine.tone.frequency.setTargetAtTime(throttle > .5 ? (nitro ? 3600 : 3000) : 1300, now, throttle > .5 ? .05 : .12);
    engine.hiss.gain.setTargetAtTime(nitro ? .07 : 0, now, .08);
    engine.hissBand.frequency.setTargetAtTime(nitro ? 3200 : 1500, now, .3);
    nitroSample(engine, nitro);
    if (!engine.screech && buffers.brake) {
      engine.screech = ctx.createBufferSource(); engine.screech.buffer = buffers.brake; engine.screech.loop = true;
      engine.screech.connect(engine.squeal); engine.screech.start();
    }
    const scrub = brake * .35;
    engine.squeal.gain.setTargetAtTime(scrub, now, scrub > engine.squeal.gain.value ? .04 : .09);
    if (engine.screech) engine.screech.playbackRate.setTargetAtTime(.88 + brake * .2, now, .1);
  }

  /** Gear change: the throttle lifts for a moment and the revs fall to the next gear on their own. */
  function shift() {
    if (!running() || !engine) return;
    const now = ctx.currentTime, g = engine.out.gain;
    g.cancelScheduledValues(now); g.setValueAtTime(g.value, now);
    g.linearRampToValueAtTime(g.value * .6, now + .06); g.linearRampToValueAtTime(g.value, now + .25);
    engine.dip = true; setTimeout(() => { if (engine) engine.dip = false; }, 260);
  }

  // Menu: music while it is open, a blip when the hand lands on a card, a rising
  // tone while the selection fills, a bright chord and a swoosh when it picks.
  let music = null, wantMusic = false, charge = null;
  function menuMusic(on) {
    wantMusic = on;
    if (!ctx) return;
    if (on && running()) { music ??= createMenuMusic(ctx, master); music.start(); } else music?.stop();
  }
  function tone(freq, t, { type = 'sine', len = .25, level = .12, to = null } = {}) {
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = type; o.frequency.setValueAtTime(freq, t); if (to) o.frequency.exponentialRampToValueAtTime(to, t + len);
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(level, t + .01); g.gain.exponentialRampToValueAtTime(.0005, t + len);
    o.connect(g).connect(master); o.start(t); o.stop(t + len + .05);
  }
  const ui = {
    hover() { if (running()) tone(1320, ctx.currentTime, { len: .09, level: .06, type: 'triangle' }); },
    /** progress 0..1 of the dwell selection; 0 or null ends the tone. */
    charge(progress) {
      if (!running()) return;
      const now = ctx.currentTime;
      if (!progress) { if (charge) { charge.g.gain.setTargetAtTime(0, now, .03); charge.o.stop(now + .2); charge = null; } return; }
      if (!charge) {
        const o = ctx.createOscillator(), g = ctx.createGain(); o.type = 'triangle'; g.gain.value = 0;
        o.connect(g).connect(master); o.start(); charge = { o, g };
      }
      charge.o.frequency.setTargetAtTime(330 * 2 ** (progress * 1.5), now, .03);
      charge.g.gain.setTargetAtTime(.025 + progress * .03, now, .05);
    },
    confirm() {
      if (!running()) return;
      ui.charge(0);
      const t = ctx.currentTime;
      [[659.3, 0], [987.8, .07], [1318.5, .14]].forEach(([f, d]) => tone(f, t + d, { len: .45, level: .09, type: 'triangle' }));
      tone(220, t, { len: .35, level: .12, to: 880 });
      play('select', { volume: .6 });
    },
  };

  // Space laser: a buzzing hum while the beams burn, rising with the heat of the rock.
  let laserNodes = null;
  function laser(on, heat = 0) {
    if (!running()) return;
    const now = ctx.currentTime;
    if (!laserNodes && on) {
      // Two stages: `g` switches the beam on and off; `buzz` flutters 30 times a second.
      // (The flutter must not feed `g`: added to a gain of 0 it would never go silent.)
      const g = ctx.createGain(); g.gain.value = 0;
      const buzz = ctx.createGain(); buzz.gain.value = .65;
      const band = ctx.createBiquadFilter(); band.type = 'bandpass'; band.frequency.value = 1400; band.Q.value = 1.2;
      const a = ctx.createOscillator(), b = ctx.createOscillator(), lfo = ctx.createOscillator(), depth = ctx.createGain();
      a.type = 'sawtooth'; b.type = 'square'; b.detune.value = 9; lfo.frequency.value = 31; depth.gain.value = .35;
      lfo.connect(depth).connect(buzz.gain);
      a.connect(band); b.connect(band); band.connect(buzz).connect(g).connect(master);
      a.start(); b.start(); lfo.start();
      laserNodes = { g, a, b, band };
    }
    if (!laserNodes) return;
    const { g, a, b, band } = laserNodes;
    g.gain.setTargetAtTime(on ? .09 : 0, now, on ? .02 : .05);
    a.frequency.setTargetAtTime(150 + heat * 220, now, .05); b.frequency.setTargetAtTime(300 + heat * 440, now, .05);
    band.frequency.setTargetAtTime(1200 + heat * 1800, now, .08);
  }

  return { loaded, play, loop, engineAt, shift, menuMusic, ui, laser,
    /** Called every frame: starts the music once the browser lets audio play. */
    tick() { if (wantMusic && running() && !music?.playing) menuMusic(true); }, get needsUnlock() { return Boolean(ctx) && ctx.state !== 'running'; }, unlock };
}
