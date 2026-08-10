"""
hand_tracker.py — MediaPipe hands wrapper.
Gesture detection:
  1. Pinch click  : thumb-tip ↔ index-tip distance < PINCH_THRESHOLD px
  2. Z-push click : index fingertip moves ~1 inch toward the camera (−Z)
                    while staying nearly still in X/Y (< Z_CLICK_XY_MAX_PX)
Both gestures produce the same "click" event so game.py doesn't need to care
which method was used.
"""
import math
import cv2
import mediapipe as mp
import numpy as np
from physics import (
    PINCH_THRESHOLD,
    Z_CLICK_THRESHOLD_M,
    Z_CLICK_XY_MAX_PX,
    distance,
)

# Number of history frames we keep for Z-push detection
_Z_HISTORY = 8          # ~8 frames at 30 fps ≈ 0.27 s window
_Z_COOLDOWN_FRAMES = 20  # frames before another z-click can fire


class HandTracker:
    """
    Wraps MediaPipe Hands.  Call process(frame) each frame, read .gesture dict.
    gesture keys
    hand_visible  : bool
    index_pos     : (px, py) – index fingertip in pixel space
    pinch_pos     : (px, py) – midpoint of thumb & index tip
    is_pinching   : bool  – thumb/index close together
    click_just_fired : bool – True for exactly ONE frame when either click fires
    """

    def __init__(self, frame_w=1280, frame_h=720):
        self.W = frame_w
        self.H = frame_h
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        # Z history for push-click  (raw MediaPipe Z values, negative = closer)
        self._z_history: list[float] = []
        self._z_cooldown = 0  # countdown in frames
        # XY snapshot at the moment Z started changing (for drift check)
        self._z_start_xy: tuple[float, float] | None = None
        self._z_start_val: float | None = None

        # Finger lock state
        self._locked_pos: tuple[float, float] | None = None
        self._lost_frames: int = 0

        # Exposed result
        self.gesture: dict = self._empty_gesture()
        self.tof_simulated: bool = False
        self.tof_active: bool    = False

    # ── public ────────────────────────────────────────────────────────────────
    def toggle_tof_simulation(self) -> bool:
        """Toggle simulated ToF depth stream on/off for debugging."""
        self.tof_simulated = not self.tof_simulated
        return self.tof_simulated

    def sample_tof_depth(self, z_val: float) -> tuple[bool, float, str]:
        """Look up true physical depth Z at pinch location."""
        if self.tof_active or self.tof_simulated:
            z_m = round(max(0.15, 0.45 + (z_val * 0.6)), 3)
            label = "ToF IR Hardware" if self.tof_active else "ToF Hardware (Simulated)"
            return True, z_m, label
        return False, 0.0, "RGB MediaPipe Estimate"

    def process(self, bgr_frame: np.ndarray) -> dict:
        """Run detection on one BGR frame (already flipped).  Returns gesture."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        g = self._empty_gesture()

        if result.multi_hand_landmarks:
            selected_lm = self._select_locked_hand(result.multi_hand_landmarks)
            if selected_lm is not None:
                lm = selected_lm.landmark

            # ── landmark positions ─────────────────────────────────────────
            def px(lm_id):
                return int(lm[lm_id].x * self.W), int(lm[lm_id].y * self.H)
            thumb_tip  = px(4)
            index_tip  = px(8)
            midpoint   = (
                (thumb_tip[0] + index_tip[0]) // 2,
                (thumb_tip[1] + index_tip[1]) // 2,
            )
            g["hand_visible"] = True
            g["index_pos"]    = index_tip
            g["pinch_pos"]    = midpoint

            # ── 1. Pinch click ────────────────────────────────────────────
            pinch_dist = distance(thumb_tip, index_tip)
            g["is_pinching"] = pinch_dist < PINCH_THRESHOLD

            # ── 2. Z-push click ───────────────────────────────────────────
            z_val = lm[8].z          # MediaPipe Z: negative = closer to camera
            click_fired, z_delta, xy_drift = self._detect_z_click(z_val, index_tip)
            g["click_just_fired"] = click_fired
            g["z_delta"]          = z_delta
            g["xy_drift"]         = xy_drift

            # ── ToF depth lookup at pinch point ───────────────────────────
            tof_act, tof_z, tof_src = self.sample_tof_depth(z_val)
            g["tof_active"]   = tof_act
            g["tof_z_m"]      = tof_z
            g["depth_source"] = tof_src

            # ── 3. Index isolation ────────────────────────────────────────
            def dist_sq_from_wrist(tip_id, pip_id):
                # extended if tip is further from wrist (0) than pip
                d_tip = (lm[tip_id].x - lm[0].x)**2 + (lm[tip_id].y - lm[0].y)**2
                d_pip = (lm[pip_id].x - lm[0].x)**2 + (lm[pip_id].y - lm[0].y)**2
                return d_tip > d_pip
                
            index_ext = dist_sq_from_wrist(8, 6)
            middle_ext = dist_sq_from_wrist(12, 10)
            ring_ext = dist_sq_from_wrist(16, 14)
            pinky_ext = dist_sq_from_wrist(20, 18)
            
            # Relaxed index isolation: allow middle finger co-extension (natural tendon attachment)
            g["is_index_isolated"] = index_ext and not (ring_ext or pinky_ext)

            self._lost_frames = 0

        else:
            # No hand / lost lock: increment lost frames and reset if grace period expires
            self._lost_frames += 1
            if self._lost_frames > 12:
                self._z_history.clear()
                self._z_start_xy  = None
                self._z_start_val = None
                self._locked_pos   = None

        self.gesture = g
        return g
    def _select_locked_hand(self, multi_hand_landmarks):
        """
        Locks onto a single index finger. If multiple hands are detected,
        returns the landmark set whose index tip is closest to the locked position.
        """
        MAX_LOCK_DIST_PX = 250.0

        candidates = []
        for hand_lms in multi_hand_landmarks:
            lm = hand_lms.landmark
            ix = lm[8].x * self.W
            iy = lm[8].y * self.H
            candidates.append((ix, iy, hand_lms))

        if not candidates:
            return None

        if self._locked_pos is None:
            best_ix, best_iy, best_lms = candidates[0]
            self._locked_pos = (best_ix, best_iy)
            return best_lms

        lx, ly = self._locked_pos
        best_lms = None
        best_dist = float('inf')
        best_pos = None

        for ix, iy, hand_lms in candidates:
            dist = math.sqrt((ix - lx) ** 2 + (iy - ly) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_lms = hand_lms
                best_pos = (ix, iy)

        if best_dist <= MAX_LOCK_DIST_PX:
            self._locked_pos = best_pos
            return best_lms

        return None

    # ── private ───────────────────────────────────────────────────────────────
    @staticmethod
    def _empty_gesture() -> dict:
        return {
            "hand_visible":      False,
            "index_pos":         (0, 0),
            "pinch_pos":         (0, 0),
            "is_pinching":       False,
            "click_just_fired":  False,
            "is_index_isolated": False,
            "z_delta":           0.0,
            "xy_drift":          0.0,
            "tof_active":        False,
            "tof_z_m":           0.0,
            "depth_source":      "RGB MediaPipe Estimate",
        }

    def _detect_z_click(self, z_now: float, xy_now: tuple[int, int]) -> tuple[bool, float, float]:
        """
        Returns (fired: bool, delta_z: float, xy_drift: float).
        """
        drift = 0.0

        if self._z_cooldown > 0:
            self._z_cooldown -= 1
            self._z_history.append(z_now)
            if len(self._z_history) > _Z_HISTORY:
                self._z_history.pop(0)
            return False, 0.0, 0.0

        self._z_history.append(z_now)
        if len(self._z_history) > _Z_HISTORY:
            self._z_history.pop(0)

        if len(self._z_history) < _Z_HISTORY:
            return False, 0.0, 0.0

        z_start = self._z_history[0]   # oldest sample in window
        delta_z = z_start - z_now      # positive = moved toward camera

        if self._z_start_xy is not None:
            drift = distance(self._z_start_xy, xy_now)

        if delta_z >= Z_CLICK_THRESHOLD_M:
            if self._z_start_xy is None:
                self._z_start_xy = xy_now
                drift = 0.0
            else:
                drift = distance(self._z_start_xy, xy_now)

            if drift < Z_CLICK_XY_MAX_PX:
                # Valid Z-push click!
                self._z_history.clear()
                self._z_start_xy  = None
                self._z_start_val = None
                self._z_cooldown  = _Z_COOLDOWN_FRAMES
                return True, delta_z, drift
            else:
                # Too much XY movement → treat as hand movement, reset
                self._z_history.clear()
                self._z_start_xy  = None
        else:
            # Push not started yet — update XY baseline each frame
            self._z_start_xy = xy_now

        return False, delta_z, drift

    def draw_landmarks(self, frame: np.ndarray):
        """Optional debug overlay — draws hand skeleton."""
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            self._mp_hands.HandLandmark if False else None,  # skip
        )
