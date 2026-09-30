/** Spoken coaching: counts repetitions and says the short part of new hints
 * ("Сядь глубже"), with the browser's own speech synthesis (no network, no
 * files). Some browsers only speak after the page has had one click or tap. */
const NUMBERS = ['ноль', 'раз', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять'];

export function createVoice({ synth = globalThis.speechSynthesis, Utterance = globalThis.SpeechSynthesisUtterance, lang = 'ru-RU',
  hintCooldownMs = 6000, gapMs = 1200, now = () => performance.now() } = {}) {
  if (!synth || !Utterance) return null;
  let enabled = true, lastSpoke = -Infinity, voice;
  const lastHint = new Map();
  const pick = () => voice !== undefined ? voice : (voice = synth.getVoices?.().find(v => v.lang?.toLowerCase().startsWith('ru')) ?? null);
  function say(text, { interrupt = false } = {}) {
    if (!enabled || !text) return false;
    if (interrupt) synth.cancel();
    const u = new Utterance(text);
    u.lang = lang; u.rate = 1.1;
    const v = pick(); if (v) u.voice = v;
    synth.speak(u); lastSpoke = now();
    return true;
  }
  return {
    get enabled() { return enabled; },
    set enabled(value) { enabled = Boolean(value); if (!enabled) synth.cancel(); },
    say,
    /** Feed frame.motion.events. */
    events(events) {
      const t = now();
      for (const e of events ?? []) {
        if (e.type === 'rep') say(e.rep.index < NUMBERS.length ? NUMBERS[e.rep.index] : String(e.rep.index), { interrupt: true });
        else if (e.type === 'hint' && t - (lastHint.get(e.hint.code) ?? -Infinity) > hintCooldownMs && t - lastSpoke > gapMs) {
          // The short command before the colon: "Колени выходят за носки".
          if (say(e.hint.text.split(':')[0])) lastHint.set(e.hint.code, t);
        }
      }
    },
  };
}
