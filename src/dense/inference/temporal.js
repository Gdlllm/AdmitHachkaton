/** Temporal state for the optional body surface. Timestamps are milliseconds
 * from one monotonic clock; a clock that goes backwards starts over. */

/**
 * On: enough of a body (see body-gate.js) is seen for `armMs`. Stays on while
 * the shoulders are tracked and enough was seen within `graceMs`, so an elbow
 * hidden for a moment or a turn does not blink the surface. Off: shoulders lost
 * for `holdMs`, or too little of the body visible for longer than `graceMs`.
 */
export class SurfaceActivation {
  constructor({ armMs = 150, holdMs = 450, graceMs = 1200 } = {}) {
    Object.assign(this, { armMs, holdMs, graceMs });
    this.reset();
  }

  reset() {
    this.state = 'idle'; this.active = false;
    this.enoughSince = null; this.lastEnough = null; this.lastTracked = null; this.last = null;
  }

  update({ enough = false, tracked = false } = {}, timestamp) {
    if (!Number.isFinite(timestamp)) throw new TypeError('A finite timestamp is required');
    let changed = null;
    if (this.last !== null && timestamp < this.last) {
      const wasActive = this.active;
      this.reset(); this.last = timestamp;
      return { state: this.state, active: false, changed: wasActive ? 'off' : null };
    }
    this.last = timestamp;
    if (!this.active) {
      if (!enough) { this.enoughSince = null; this.state = 'idle'; }
      else {
        this.enoughSince ??= timestamp;
        this.state = 'arming';
        if (timestamp - this.enoughSince >= this.armMs) {
          this.active = true; this.state = 'active'; changed = 'on';
          this.lastEnough = this.lastTracked = timestamp;
        }
      }
      return { state: this.state, active: this.active, changed };
    }
    if (enough) this.lastEnough = timestamp;
    if (enough || tracked) this.lastTracked = timestamp;
    if (timestamp - this.lastTracked > this.holdMs || timestamp - this.lastEnough > this.graceMs) {
      this.reset(); this.last = timestamp; changed = 'off';
    }
    return { state: this.state, active: this.active, changed };
  }
}

const alpha = (cutoff, dt) => 1 / (1 + 1 / (2 * Math.PI * Math.max(cutoff, 1e-6) * dt));

/** Element-wise 1 Euro filter (Casiez et al. 2012) for model parameter vectors.
 * Defaults follow the InstantHMR demo smoother; time is in seconds. */
export class OneEuroVector {
  constructor(size, { minCutoff = 1, beta = 0, derivativeCutoff = 1 } = {}) {
    if (!Number.isInteger(size) || size < 1) throw new RangeError('Vector size must be a positive integer');
    Object.assign(this, { size, minCutoff, beta, derivativeCutoff });
    this.reset();
  }

  reset() { this.value = null; this.derivative = null; this.time = null; }

  update(values, time) {
    if (values?.length !== this.size) throw new TypeError(`Expected ${this.size} values`);
    if (!this.value || !Number.isFinite(this.time) || !(time > this.time)) {
      this.value = Float64Array.from(values); this.derivative = new Float64Array(this.size); this.time = time;
      return Float32Array.from(values);
    }
    const dt = Math.min(1, time - this.time), derivativeAlpha = alpha(this.derivativeCutoff, dt);
    const output = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      this.derivative[i] += derivativeAlpha * ((values[i] - this.value[i]) / dt - this.derivative[i]);
      this.value[i] += alpha(this.minCutoff + this.beta * Math.abs(this.derivative[i]), dt) * (values[i] - this.value[i]);
      output[i] = this.value[i];
    }
    this.time = time;
    return output;
  }
}
