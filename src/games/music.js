/** Background music for the game centre menu, written in code: a calm
 * synthwave loop at 96 BPM in A minor (Am – F – C – G, one bar each). Warm
 * detuned pads, a round bass, a plucked arpeggio through a delay, a soft kick,
 * clap and hats. Notes are scheduled a little ahead of the audio clock.
 *
 *   const music = createMenuMusic(ctx, destination);
 *   music.start(); music.stop();
 */
const BPM = 96, BEAT = 60 / BPM, BAR = BEAT * 4;
// Chords as MIDI notes (root in the bass, three tones for the pad).
const CHORDS = [
  { bass: 45, pad: [57, 60, 64] },       // Am
  { bass: 41, pad: [53, 57, 60] },       // F
  { bass: 48, pad: [55, 60, 64] },       // C
  { bass: 43, pad: [55, 59, 62] },       // G
];
const ARP = [0, 1, 2, 1, 0, 2, 1, 2];     // chord tones for the eight 8ths of a bar
const hz = midi => 440 * 2 ** ((midi - 69) / 12);

export function createMenuMusic(ctx, destination) {
  const out = ctx.createGain(); out.gain.value = 0;
  const warm = ctx.createBiquadFilter(); warm.type = 'lowpass'; warm.frequency.value = 5200;
  out.connect(warm).connect(destination);
  // A dotted-8th echo for the arpeggio.
  const delay = ctx.createDelay(1), feedback = ctx.createGain(), wet = ctx.createGain();
  delay.delayTime.value = BEAT * .75; feedback.gain.value = .38; wet.gain.value = .45;
  delay.connect(feedback).connect(delay); delay.connect(wet).connect(out);
  let noise = null;
  {
    const b = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate), d = b.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    noise = b;
  }

  function pad(chord, t) {
    for (const note of chord.pad) for (const detune of [-7, 6]) {
      const o = ctx.createOscillator(), f = ctx.createBiquadFilter(), g = ctx.createGain();
      o.type = 'sawtooth'; o.frequency.value = hz(note); o.detune.value = detune;
      f.type = 'lowpass'; f.frequency.setValueAtTime(500, t); f.frequency.linearRampToValueAtTime(1300, t + BAR * .5); f.frequency.linearRampToValueAtTime(700, t + BAR);
      g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(.022, t + .5); g.gain.setValueAtTime(.022, t + BAR - .3); g.gain.linearRampToValueAtTime(0, t + BAR + .25);
      o.connect(f).connect(g).connect(out); o.start(t); o.stop(t + BAR + .3);
    }
  }
  function bass(note, t, len) {
    const o = ctx.createOscillator(), g = ctx.createGain(), f = ctx.createBiquadFilter();
    o.type = 'triangle'; o.frequency.value = hz(note); f.type = 'lowpass'; f.frequency.value = 400;
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(.16, t + .02); g.gain.exponentialRampToValueAtTime(.001, t + len);
    o.connect(f).connect(g).connect(out); o.start(t); o.stop(t + len + .05);
  }
  function pluck(note, t) {
    const o = ctx.createOscillator(), g = ctx.createGain(), f = ctx.createBiquadFilter();
    o.type = 'square'; o.frequency.value = hz(note); f.type = 'lowpass'; f.frequency.setValueAtTime(2600, t); f.frequency.exponentialRampToValueAtTime(500, t + .25);
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(.035, t + .005); g.gain.exponentialRampToValueAtTime(.001, t + .3);
    o.connect(f).connect(g); g.connect(out); g.connect(delay); o.start(t); o.stop(t + .35);
  }
  function kick(t) {
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.setValueAtTime(110, t); o.frequency.exponentialRampToValueAtTime(42, t + .18);
    g.gain.setValueAtTime(.22, t); g.gain.exponentialRampToValueAtTime(.001, t + .35);
    o.connect(g).connect(out); o.start(t); o.stop(t + .4);
  }
  function hiss(t, { len = .04, level = .02, freq = 8000, q = .8 } = {}) {
    const src = ctx.createBufferSource(), f = ctx.createBiquadFilter(), g = ctx.createGain();
    src.buffer = noise; f.type = 'bandpass'; f.frequency.value = freq; f.Q.value = q;
    g.gain.setValueAtTime(level, t); g.gain.exponentialRampToValueAtTime(.0005, t + len);
    src.connect(f).connect(g).connect(out); src.start(t, Math.random() * .5); src.stop(t + len + .02);
  }

  let timer = null, nextBar = 0, bar = 0;
  function scheduleBar(t) {
    const chord = CHORDS[bar % CHORDS.length], lap = Math.floor(bar / CHORDS.length);
    pad(chord, t);
    bass(chord.bass, t, BEAT * 1.4); bass(chord.bass, t + BEAT * 1.5, BEAT * .45); bass(chord.bass, t + BEAT * 2, BEAT * 1.4); bass(chord.bass + 12, t + BEAT * 3.5, BEAT * .45);
    // The arpeggio comes in on the second pass, the drums on the third, so the loop builds up.
    if (lap >= 1) ARP.forEach((k, i) => pluck(chord.pad[k] + 12, t + i * BEAT / 2));
    if (lap >= 2) for (let b = 0; b < 4; b++) {
      if (b % 2 === 0) kick(t + b * BEAT); else hiss(t + b * BEAT, { len: .16, level: .05, freq: 1500, q: .6 });
      hiss(t + b * BEAT + BEAT / 2, {});
    }
    bar++;
  }
  return {
    get playing() { return timer !== null; },
    /** Schedule `bars` bars right away, without a timer (offline rendering, previews). */
    schedule(bars) { out.gain.value = .9; nextBar = ctx.currentTime + .05; bar = 0; for (let i = 0; i < bars; i++) { scheduleBar(nextBar); nextBar += BAR; } },
    start() {
      if (timer || ctx.state !== 'running') return;
      const now = ctx.currentTime;
      out.gain.cancelScheduledValues(now); out.gain.setValueAtTime(out.gain.value, now); out.gain.linearRampToValueAtTime(.9, now + 2);
      nextBar = now + .1; bar = 0;
      timer = setInterval(() => { while (nextBar < ctx.currentTime + .3) { scheduleBar(nextBar); nextBar += BAR; } }, 60);
    },
    stop() {
      if (!timer) return;
      clearInterval(timer); timer = null;
      const now = ctx.currentTime;
      out.gain.cancelScheduledValues(now); out.gain.setValueAtTime(out.gain.value, now); out.gain.linearRampToValueAtTime(0, now + .6);
    },
  };
}
