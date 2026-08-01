/* handTracker.js — MediaPipe Hands wrapper with EMA smoothing and adaptive thresholds */

const dist2D = (p1, p2) => Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);

// Exponential Moving Average helper
function ema(current, previous, alpha) {
  return alpha * current + (1 - alpha) * previous;
}

export class HandTracker {
  constructor(frameW = 1280, frameH = 720) {
    this.W = frameW;
    this.H = frameH;

    // ── Smoothing ────────────────────────────────────────────────────────────
    // Lower = smoother but more lag. 0.25 is a good balance for game control.
    this.SMOOTH_ALPHA = 0.25;

    // ── Pinch Detection ──────────────────────────────────────────────────────
    // Normalized (0-1) threshold — adapts automatically to hand distance.
    // ~0.06 units in normalized coords ≈ fingers just touching regardless of depth.
    this.PINCH_THRESHOLD_NORM = 0.07;

    // Require this many consecutive frames to confirm a pinch (debounce)
    this.PINCH_DEBOUNCE_FRAMES = 3;
    this.pinchConsecutive = 0;
    this.releaseConsecutive = 0;
    this.pinchActive = false;    // debounced pinch state

    // ── Z-Push Detection ─────────────────────────────────────────────────────
    this.Z_CLICK_THRESHOLD_M = 0.012;
    this.Z_CLICK_XY_MAX_PX   = 35;
    this.zHistoryLen          = 10;
    this.zCooldownFrames      = 25;

    // State
    this.zHistory  = [];
    this.zCooldown = 0;
    this.zStartXY  = null;

    // ── Smoothed positions (subpixel) ────────────────────────────────────────
    this._smoothIX = -1;  // -1 = uninitialised
    this._smoothIY = -1;
    this._smoothPX = -1;
    this._smoothPY = -1;

    // ── ToF Sensor State ─────────────────────────────────────────────────────
    this.tofActive    = false;
    this.tofSimulated = false;

    // ── Finger Lock State ───────────────────────────────────────────────────
    this._lockedPos  = null;
    this._lostFrames = 0;

    // ── Output ────────────────────────────────────────────────────────────────
    this.gesture = this._emptyGesture();
    this.hands   = null;
  }

  toggleTofSimulation() {
    this.tofSimulated = !this.tofSimulated;
    return this.tofSimulated;
  }

  sampleTofDepth(zVal) {
    if (this.tofActive || this.tofSimulated) {
      const zM = Number(Math.max(0.15, 0.45 + zVal * 0.6).toFixed(3));
      const label = this.tofActive ? "ToF IR Hardware" : "ToF Hardware (Simulated)";
      return { active: true, depthM: zM, source: label };
    }
    return { active: false, depthM: 0, source: "RGB MediaPipe Estimate" };
  }

  _emptyGesture() {
    return {
      hand_visible:      false,
      index_pos:         [0, 0],
      pinch_pos:         [0, 0],
      is_pinching:       false,
      click_just_fired:  false,
      is_index_isolated: false,
      zDelta:            0,
      xyDrift:           0,
      tof_active:        false,
      tof_z_m:           0,
      depth_source:      "RGB MediaPipe Estimate",
    };
  }

  /**
   * Initialize MediaPipe Hands.
   * @param {Function} onReadyCallback
   */
  init(onReadyCallback) {
    const MP_Hands = window.Hands;
    if (!MP_Hands) {
      console.error('[ERROR] MediaPipe Hands not loaded from CDN.');
      return;
    }

    this.hands = new MP_Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    this.hands.setOptions({
      maxNumHands:            2,
      modelComplexity:        1,
      minDetectionConfidence: 0.75,
      minTrackingConfidence:  0.75,   // raised from 0.6 → fewer jittery re-acquisitions
    });

    this.hands.onResults((results) => this._processResults(results));

    this.hands.initialize()
      .then(() => {
        console.log('[INFO] MediaPipe Hands initialized.');
        if (onReadyCallback) onReadyCallback();
      })
      .catch((err) => {
        console.error('[ERROR] MediaPipe Hands init failed:', err);
      });
  }

  async process(videoElement) {
    if (this.hands) await this.hands.send({ image: videoElement });
  }

  // ── Internal result processor ─────────────────────────────────────────────

  _processResults(results) {
    const g = this._emptyGesture();

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const lm = this._selectLockedHand(results.multiHandLandmarks);
      if (lm) {

      // ── Raw normalized coords (no flooring — keep subpixel) ───────────────
      const normX = (id) => (1 - lm[id].x) * this.W;   // mirror horizontally
      const normY = (id) =>      lm[id].y  * this.H;

      const rawIX = normX(8);  // index tip
      const rawIY = normY(8);
      const rawTX = normX(4);  // thumb tip
      const rawTY = normY(4);

      // ── EMA smoothing on landmark positions ───────────────────────────────
      // Bootstrap on first frame
      if (this._smoothIX < 0) {
        this._smoothIX = rawIX;
        this._smoothIY = rawIY;
        this._smoothPX = (rawIX + rawTX) / 2;
        this._smoothPY = (rawIY + rawTY) / 2;
      }

      // ── 2. Z-Push / Click Detection ───────────────────────────────────────
      const zVal = lm[8].z;
      const { fired, delta, drift } = this._detectZClick(zVal, [Math.round(this._smoothIX), Math.round(this._smoothIY)]);
      g.click_just_fired = fired;
      g.zDelta           = delta;
      g.xyDrift          = drift;

      // Heavy dampening during forward Z-push to prevent lateral cursor drift
      const currentAlpha = delta > 0.015 ? 0.02 : this.SMOOTH_ALPHA;

      this._smoothIX = ema(rawIX, this._smoothIX, currentAlpha);
      this._smoothIY = ema(rawIY, this._smoothIY, currentAlpha);

      const rawPX = (rawIX + rawTX) / 2;
      const rawPY = (rawIY + rawTY) / 2;
      this._smoothPX = ema(rawPX, this._smoothPX, currentAlpha);
      this._smoothPY = ema(rawPY, this._smoothPY, currentAlpha);

      // Output as rounded integers for pixel consumption, but smoothed first
      g.hand_visible = true;
      g.index_pos    = [Math.round(this._smoothIX), Math.round(this._smoothIY)];
      g.pinch_pos    = [Math.round(this._smoothPX), Math.round(this._smoothPY)];

      // ── 1. Pinch Detection (normalized + debounced) ───────────────────────
      const pinchDistNorm = Math.sqrt(
        (lm[4].x - lm[8].x) ** 2 +
        (lm[4].y - lm[8].y) ** 2 +
        (lm[4].z - lm[8].z) ** 2 * 0.5   // z has lower weight (noisier)
      );

      const rawPinching = pinchDistNorm < this.PINCH_THRESHOLD_NORM;

      // Debounce: require N consecutive frames before toggling state
      if (rawPinching) {
        this.pinchConsecutive++;
        this.releaseConsecutive = 0;
        if (this.pinchConsecutive >= this.PINCH_DEBOUNCE_FRAMES) {
          this.pinchActive = true;
        }
      } else {
        this.releaseConsecutive++;
        this.pinchConsecutive = 0;
        if (this.releaseConsecutive >= this.PINCH_DEBOUNCE_FRAMES) {
          this.pinchActive = false;
        }
      }

      g.is_pinching = this.pinchActive;

      // ── 3. Index Finger Isolation (curl-based) ────────────────────────────
      const dist3D = (a, b) => Math.sqrt(
        (lm[a].x - lm[b].x) ** 2 +
        (lm[a].y - lm[b].y) ** 2 +
        (lm[a].z - lm[b].z) ** 2
      );

      // Palm reference: landmark 9 (middle finger MCP) — stable anchor
      const isExtended = (tip, pip, mcp) =>
        dist3D(tip, 9) > dist3D(mcp, 9) * 0.85;

      const indexExt  = isExtended(8,  6,  5);
      const middleExt = isExtended(12, 10, 9);
      const ringExt   = isExtended(16, 14, 13);
      const pinkyExt  = isExtended(20, 18, 17);

      // Relaxed index isolation: allow middle finger co-extension (natural tendon attachment)
      g.is_index_isolated = indexExt && !(ringExt || pinkyExt);

      // ToF Depth sampling
      const tofRes = this.sampleTofDepth(lm[8].z);
      g.tof_active   = tofRes.active;
      g.tof_z_m      = tofRes.depthM;
      g.depth_source = tofRes.source;

      } else {
        this._lostFrames++;
        if (this._lostFrames > 12) {
          this._smoothIX = -1;
          this._smoothIY = -1;
          this._lockedPos = null;
        }
      }
    } else {
      // Hand lost — reset all smoothing and detection state
      this._lostFrames++;
      if (this._lostFrames > 12) {
        this._smoothIX      = -1;
        this._smoothIY      = -1;
        this._smoothPX      = -1;
        this._smoothPY      = -1;
        this.pinchConsecutive   = 0;
        this.releaseConsecutive = 0;
        this.pinchActive    = false;
        this.zHistory       = [];
        this.zStartXY       = null;
        this._lockedPos     = null;
      }
    }

    this.gesture = g;
  }

  _selectLockedHand(multiHandLandmarks) {
    const MAX_LOCK_DIST_PX = 250.0;
    const candidates = multiHandLandmarks.map((lm) => {
      const ix = (1 - lm[8].x) * this.W;
      const iy = lm[8].y * this.H;
      return { ix, iy, lm };
    });

    if (candidates.length === 0) return null;

    if (!this._lockedPos) {
      this._lockedPos = [candidates[0].ix, candidates[0].iy];
      this._lostFrames = 0;
      return candidates[0].lm;
    }

    const [lx, ly] = this._lockedPos;
    let bestLm = null;
    let bestDist = Infinity;
    let bestPos = null;

    for (const c of candidates) {
      const dist = Math.hypot(c.ix - lx, c.iy - ly);
      if (dist < bestDist) {
        bestDist = dist;
        bestLm = c.lm;
        bestPos = [c.ix, c.iy];
      }
    }

    if (bestDist <= MAX_LOCK_DIST_PX) {
      this._lockedPos = bestPos;
      this._lostFrames = 0;
      return bestLm;
    }

    return null;
  }

  // ── Z-Push click algorithm ────────────────────────────────────────────────

  _detectZClick(zNow, xyNow) {
    const result = { fired: false, delta: 0, drift: 0 };

    if (this.zCooldown > 0) {
      this.zCooldown--;
      this.zHistory.push(zNow);
      if (this.zHistory.length > this.zHistoryLen) this.zHistory.shift();
      return result;
    }

    this.zHistory.push(zNow);
    if (this.zHistory.length > this.zHistoryLen) this.zHistory.shift();
    if (this.zHistory.length < this.zHistoryLen)  return result;

    const zBaseline = this.zHistory[0];
    const deltaZ    = zBaseline - zNow;  // positive = finger moving toward camera
    result.delta    = deltaZ;

    if (this.zStartXY !== null) {
      result.drift = dist2D(this.zStartXY, xyNow);
    }

    if (deltaZ >= this.Z_CLICK_THRESHOLD_M) {
      if (this.zStartXY === null) {
        this.zStartXY = xyNow;
        result.drift  = 0;
      } else {
        result.drift  = dist2D(this.zStartXY, xyNow);
      }

      if (result.drift < this.Z_CLICK_XY_MAX_PX) {
        // Confirmed push-click
        this.zHistory  = [];
        this.zStartXY  = null;
        this.zCooldown = this.zCooldownFrames;
        result.fired   = true;
      } else {
        // Too much lateral drift — not a clean push
        this.zHistory = [];
        this.zStartXY = null;
      }
    } else {
      this.zStartXY = xyNow;
    }

    return result;
  }
}
