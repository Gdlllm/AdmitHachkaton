/** Developer overlay for ?hud=1: live angles, facing, framing, gestures, the
 * exercise state and the current hint. Plain DOM text, no layout of its own
 * beyond a corner panel; not part of the product UI. */
export function createHud(parent) {
  const panel = document.createElement('pre');
  panel.style.cssText = 'position:fixed;left:12px;top:12px;z-index:9;margin:0;padding:10px 12px;max-width:min(420px,calc(100vw - 24px));'
    + 'background:rgba(6,18,20,.78);color:#dffbf6;font:12px/1.45 ui-monospace,Menlo,monospace;border-radius:8px;pointer-events:none;white-space:pre-wrap';
  const hint = document.createElement('div');
  hint.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:9;max-width:min(720px,calc(100vw - 32px));'
    + 'padding:12px 18px;border-radius:12px;background:rgba(255,77,94,.92);color:#fff;font:600 18px/1.35 system-ui,sans-serif;text-align:center;display:none';
  parent.append(panel, hint);
  const SURE = .5;                          // the bar the exercise rules use
  const num = (v, digits = 0) => typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—';
  // Joints MediaPipe only guesses (out of frame, hidden) show as a dash, not as a number.
  const deg = a => a && a.confidence >= SURE ? num(a.value) : '—';
  const lr = pair => pair ? `${deg(pair.left)} / ${deg(pair.right)}` : '—';
  return {
    update(frame) {
      const m = frame?.motion;
      if (!m) { panel.textContent = 'нет данных'; hint.style.display = 'none'; return; }
      const a = m.angles, lines = [];
      if (m.exercise) {
        const signal = m.exercise.signal;
        lines.push(`${m.exercise.title}: ${m.exercise.reps} повт. (чистых ${m.exercise.goodReps})  фаза ${m.exercise.phase}  сигнал ${num(signal, Math.abs(signal ?? 0) < 10 ? 2 : 0)}`);
      }
      if (a) {
        const trunk = a.trunk?.confidence >= SURE ? a.trunk : null;
        lines.push(`колени Л/П   ${lr(a.knee)}°   таз ${lr(a.hip)}°`);
        lines.push(`локти Л/П    ${lr(a.elbow)}°   руки ${lr(a.shoulder)}°`);
        lines.push(`колено внутрь/наружу ${lr(a.kneeDrift)}°   корпус вперёд ${num(trunk?.forward)}° вбок ${num(trunk?.side)}°`);
        lines.push(m.framing?.fullBody
          ? `колени/стопы ${num(m.measures.kneeToAnkleWidth, 2)}  стойка/таз ${num(m.measures.stanceToHipWidth, 2)}  колено за носком ${num(m.measures.kneePastToe, 2)} м`
          : 'ноги: не в кадре');
      }
      if (m.facing) lines.push(`поворот ${m.facing.view ?? '—'} ${num(m.facing.yaw)}° (${m.facing.source ?? '—'})  лицо ${m.facing.face ? 'да' : 'нет'}`);
      if (m.framing) lines.push(`кадр: ${m.framing.fullBody ? 'всё тело' : 'не всё тело'}  высота ${num(m.framing.bodyHeight, 2)}  центр ${num(m.framing.centreX, 2)}`);
      const on = Object.entries(m.gestures ?? {}).filter(([, v]) => v).map(([k]) => k);
      lines.push(`жесты: ${on.join(', ') || '—'}${m.pointer ? `  палец ${num(m.pointer.x, 2)},${num(m.pointer.y, 2)}${m.pointer.pinch ? ' щипок' : ''}` : ''}`);
      panel.textContent = lines.join('\n');
      hint.textContent = m.hint?.text ?? '';
      hint.style.display = m.hint ? 'block' : 'none';
    },
  };
}
