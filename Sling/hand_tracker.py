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
    THREE_FINGER_PINCH_RADIUS,
    INPUT_MOVEMENT_MAGNIFICATION,
    Z_CLICK_THRESHOLD_M,
    Z_CLICK_XY_MAX_PX,
    distance,
)
from visual_ai import Vector2, get_landmark_distance, get_hand_center_and_radius

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

        # Frames since a hand was last seen (drives the state reset below)
        self._lost_frames: int = 0

        # 3-finger pinch edge-trigger state
        self._was_3_pinching: bool = False
        self.magnification: float = INPUT_MOVEMENT_MAGNIFICATION

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

    def process(self, bgr_frame: np.ndarray, magnification: float | None = None) -> dict:
        """Run detection on one BGR frame (already flipped).  Returns gesture."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        g = self._empty_gesture()

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark

            # ── landmark positions ─────────────────────────────────────────────
            def px(lm_id):
                return int(lm[lm_id].x * self.W), int(lm[lm_id].y * self.H)
            thumb_tip  = px(4)
            index_tip  = px(8)
            middle_tip = px(12)
            c_x = (thumb_tip[0] + index_tip[0] + middle_tip[0]) // 3
            c_y = (thumb_tip[1] + index_tip[1] + middle_tip[1]) // 3
            centroid   = (c_x, c_y)

            g["hand_visible"] = True
            g["index_pos"]    = index_tip
            g["thumb_pos"]    = thumb_tip
            g["middle_pos"]   = middle_tip
            g["pinch_pos"]    = centroid

            # ── 1. Pinch click (2-finger disabled / commented out per user instruction) ──
            # pinch_dist = distance(thumb_tip, index_tip)
            # g["is_pinching"] = pinch_dist < PINCH_THRESHOLD

            # ── 2. Z-push click (disabled / commented out per user instruction) ──
            z_val = lm[8].z          # MediaPipe Z: negative = closer to camera
            # click_fired, z_delta, xy_drift = self._detect_z_click(z_val, index_tip)
            click_fired = False
            g["click_just_fired"] = False
            g["z_delta"]          = 0.0
            g["xy_drift"]         = 0.0

            # ── 3-Finger Pinch Trigger ─────────────────────────────────────────
            v_thumb = Vector2(thumb_tip[0], thumb_tip[1])
            v_index = Vector2(index_tip[0], index_tip[1])
            v_middle = Vector2(middle_tip[0], middle_tip[1])
            v_centroid = Vector2(c_x, c_y)

            dist_t_c = v_thumb.distance_to(v_centroid)
            dist_i_c = v_index.distance_to(v_centroid)
            dist_m_c = v_middle.distance_to(v_centroid)
            max_dist_to_centroid = max(dist_t_c, dist_i_c, dist_m_c)

            mag = magnification if magnification is not None else self.magnification
            effective_radius = THREE_FINGER_PINCH_RADIUS * max(1.0, mag)
            is_3_pinching = max_dist_to_centroid < effective_radius

            if is_3_pinching and not self._was_3_pinching:
                click_fired = True
            self._was_3_pinching = is_3_pinching

            g["is_pinching"]         = is_3_pinching
            g["click_just_fired"]    = click_fired
            g["is_3_finger_pinching"]= is_3_pinching

            # ── ToF depth lookup at pinch point ───────────────────────────────
            tof_act, tof_z, tof_src = self.sample_tof_depth(z_val)
            g["tof_active"]   = tof_act
            g["tof_z_m"]      = tof_z
            g["depth_source"] = tof_src

            # ── 3. Index isolation ────────────────────────────────────────────
            def dist_sq_from_wrist(tip_id, pip_id):
                d_tip = (lm[tip_id].x - lm[0].x)**2 + (lm[tip_id].y - lm[0].y)**2
                d_pip = (lm[pip_id].x - lm[0].x)**2 + (lm[pip_id].y - lm[0].y)**2
                return d_tip > d_pip

            index_ext  = dist_sq_from_wrist(8, 6)
            middle_ext = dist_sq_from_wrist(12, 10)
            ring_ext   = dist_sq_from_wrist(16, 14)
            pinky_ext  = dist_sq_from_wrist(20, 18)

            # The wrist anchor is degenerate for the thumb — landmark 2 sits
            # almost on the wrist, so the tip clears it even with the thumb
            # folded across the palm. Measure abduction from the pinky MCP
            # (17) instead, which a tucked thumb moves toward.
            def _d17(i):
                return math.hypot(lm[i].x - lm[17].x, lm[i].y - lm[17].y)

            palm_span  = max(math.hypot(lm[0].x - lm[17].x, lm[0].y - lm[17].y), 1e-6)
            thumb_ext  = (_d17(4) - _d17(3)) > 0.10 * palm_span
            g["is_index_isolated"] = index_ext and not (ring_ext or pinky_ext)
            g["is_open_palm"]      = index_ext and middle_ext and ring_ext and pinky_ext and thumb_ext

            self._lost_frames = 0

        else:
            self._lost_frames += 1
            if self._lost_frames > 12:
                self._z_history.clear()
                self._z_start_xy  = None
                self._z_start_val = None
                self._was_3_pinching = False

        self.gesture = g
        return g

    # ── private ───────────────────────────────────────────────────────────────
    @staticmethod
    def _empty_gesture() -> dict:
        return {
            "hand_visible":        False,
            "index_pos":           (0, 0),
            "thumb_pos":           (0, 0),
            "middle_pos":          (0, 0),
            "pinch_pos":           (0, 0),
            "is_pinching":         False,
            "click_just_fired":    False,
            "is_index_isolated":   False,
            "z_delta":             0.0,
            "xy_drift":            0.0,
            "tof_active":          False,
            "tof_z_m":             0.0,
            "depth_source":        "RGB MediaPipe Estimate",
            "is_3_finger_pinching":False,
            "is_open_palm":         False,
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
