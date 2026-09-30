/** Hands-free buttons: hover a target with the index fingertip and hold it
 * there (dwellMs) or pinch thumb and index to select it. Targets are screen
 * rectangles in the same 0..1 coordinates as frame.motion.pointer (the video
 * as the user sees it, mirrored).
 *
 *   const buttons = createDwellSelector([{ id: 'start', x: .4, y: .7, w: .2, h: .12 }]);
 *   const { hover, progress, selected } = buttons.update(frame.motion.pointer, frame.timestamp);
 */
export function createDwellSelector(targets, { dwellMs = 1000, graceMs = 250, repeatAfterMs = 1500 } = {}) {
  let list = targets.slice(), hover = null, since = 0, lostAt = null, lastSelect = -Infinity, pinched = false;
  const hit = p => p ? list.find(r => p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h) ?? null : null;
  return {
    setTargets(next) { list = next.slice(); if (hover && !list.some(r => r.id === hover.id)) hover = null; },
    update(pointer, t) {
      const target = hit(pointer);
      // A finger that briefly leaves the button (tracking jitter) keeps its progress.
      if (target && target.id === hover?.id) lostAt = null;
      else if (target) { hover = target; since = t; lostAt = null; }
      else if (hover) { lostAt ??= t; if (t - lostAt > graceMs) hover = null; }
      const progress = hover ? Math.min(1, (t - since) / dwellMs) : 0;
      const pinch = Boolean(pointer?.pinch && target);
      let selected = null;
      if (hover && t - lastSelect > repeatAfterMs && ((progress >= 1 && !lostAt) || (pinch && !pinched))) {
        selected = hover.id; lastSelect = t; since = t;
      }
      pinched = Boolean(pointer?.pinch);
      return { hover: hover?.id ?? null, progress, selected };
    },
  };
}
