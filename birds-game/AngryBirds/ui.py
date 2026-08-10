"""
ui.py — HUD, bird carousel, trajectory preview, click-mode indicator,
         score display, level indicator, FPS counter.
"""

import cv2
import numpy as np
import math
from bird import Bird, BIRD_ORDER, COLOURS, RADII


# ── Colours ───────────────────────────────────────────────────────────────────
HUD_TEXT   = (240, 240, 240)
SHADOW_COL = (30, 30, 30)
TRAJ_COL   = (220, 220, 220)
OVERLAY_ALPHA = 0.45


def _text(frame, txt, pos, scale=0.7, col=HUD_TEXT, thickness=1):
    """Draws text with a drop-shadow for legibility."""
    x, y = pos
    cv2.putText(frame, txt, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, SHADOW_COL, thickness+1, cv2.LINE_AA)
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, col, thickness, cv2.LINE_AA)


def draw_rect_alpha(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                    color: tuple[int, int, int], alpha: float):
    """Draws a semi-transparent filled rectangle using ROI slicing to avoid full-frame hardcopies."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = frame[y1:y2, x1:x2]
    rect = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)


def draw_ground(frame: np.ndarray, floor_y: int = 660):
    """Draw semi-transparent sky gradient + ground strip."""
    h, w = frame.shape[:2]
    # Sky blue tint at top (alpha blended via ROI slice, no frame.copy())
    draw_rect_alpha(frame, 0, 0, w, floor_y, (200, 160, 80), 0.18)
    # Ground
    cv2.rectangle(frame, (0, floor_y), (w, h), (40, 130, 60), -1)
    cv2.rectangle(frame, (0, floor_y), (w, floor_y+6), (30, 100, 45), -1)


def draw_carousel(frame: np.ndarray, bird_types: list, selected_idx: int):
    """
    Draw the bird selection carousel at the top centre.

    Parameters
    ----------
    bird_types   : ordered list of bird kind strings still available
    selected_idx : index of the currently highlighted bird
    """
    h, w = frame.shape[:2]
    cx   = w // 2
    cy   = 90
    spacing = 120

    # Background panel (alpha blended via ROI slice)
    panel_w = spacing * len(bird_types) + 80
    panel_h = 140
    px      = cx - panel_w // 2
    draw_rect_alpha(frame, px, cy - 60, px + panel_w, cy + panel_h - 60, (20, 20, 20), 0.6)
    cv2.rectangle(frame, (px, cy-60), (px+panel_w, cy+panel_h-60), (80, 80, 80), 2)

    total = len(bird_types)
    for i, kind in enumerate(bird_types):
        bx = cx + (i - total // 2) * spacing
        is_sel = (i == selected_idx)
        scale  = 1.5 if is_sel else 0.9

        if is_sel:
            # Glow ring
            cv2.circle(frame, (bx, cy), int(RADII[kind]*scale) + 12,
                       (60, 200, 255), 2)
            cv2.circle(frame, (bx, cy), int(RADII[kind]*scale) + 8,
                       (30, 120, 180), 2)

        # Draw miniature bird
        tmp = Bird(kind, bx, cy)
        tmp.draw(frame, scale=scale)

        # Label
        label_col = (0, 255, 200) if is_sel else (160, 160, 160)
        _text(frame, kind, (bx - 20, cy + int(RADII[kind]*scale) + 20),
              scale=0.55 if is_sel else 0.45, col=label_col)

    # Instruction
    _text(frame, "3-FINGER LOCK: Keep apart to lock -> Pinch 3 to select/fire",
          (cx - 260, cy + 95), scale=0.55, col=(180, 220, 255))


def draw_trajectory(frame: np.ndarray,
                    start_x: float, start_y: float,
                    vx: float, vy: float,
                    gravity: float = 0.15, n_dots: int = 28):
    """Draw dotted parabolic trajectory preview from launch position."""
    x, y   = start_x, start_y
    cvx, cvy = vx, vy
    for i in range(n_dots):
        cvy += gravity
        x   += cvx
        y   += cvy
        alpha = 1.0 - i / n_dots          # fade out
        r_dot = max(2, int(5 * alpha))
        bright = int(255 * alpha)
        col = (bright, bright, bright)
        cv2.circle(frame, (int(x), int(y)), r_dot, col, -1)


def draw_3finger_lock_overlay(frame: np.ndarray, gesture: dict):
    """
    Renders step-by-step lock-in animations for Thumb, Index, and Middle fingers,
    connected triangle mesh when locked, and centroid pinch trigger ripple.
    """
    if not gesture or not gesture.get("hand_visible", False):
        return

    thumb_pos = gesture.get("thumb_pos", (0, 0))
    index_pos = gesture.get("index_pos", (0, 0))
    middle_pos = gesture.get("middle_pos", (0, 0))
    pinch_pos = gesture.get("pinch_pos", (0, 0))
    progress = gesture.get("lock_progress", 0.0)
    is_locked = gesture.get("three_finger_locked", False)
    locked_fingers = gesture.get("locked_fingers", (False, False, False))
    is_pinching = gesture.get("is_3_finger_pinching", False)
    click_fired = gesture.get("click_just_fired", False)

    # Finger definitions: (name, position, color, progress_range_start, progress_range_end, is_locked)
    fingers = [
        ("THUMB", thumb_pos, (0, 215, 255), 0.0, 0.33, locked_fingers[0]),
        ("INDEX", index_pos, (255, 255, 0), 0.33, 0.66, locked_fingers[1]),
        ("MIDDLE", middle_pos, (255, 0, 255), 0.66, 1.0, locked_fingers[2]),
    ]

    for name, pos, col, start_p, end_p, is_f_locked in fingers:
        x, y = pos
        if x <= 0 or y <= 0:
            continue

        # Outer base ring
        cv2.circle(frame, (x, y), 22, (50, 50, 50), 2)

        if is_f_locked:
            # Fully locked: glowing solid badge
            cv2.circle(frame, (x, y), 22, col, -1)
            cv2.circle(frame, (x, y), 25, (255, 255, 255), 2)
            _text(frame, "LOCKED", (x - 22, y - 28), scale=0.45, col=col, thickness=2)
        else:
            # Partial locking animation arc
            f_ratio = max(0.0, min(1.0, (progress - start_p) / (end_p - start_p)))
            angle = int(f_ratio * 360)
            if angle > 0:
                cv2.ellipse(frame, (x, y), (22, 22), 0, -90, -90 + angle, col, 4)
            cv2.circle(frame, (x, y), 6, col, -1)
            if f_ratio > 0:
                _text(frame, f"{name} LOCKING...", (x - 35, y - 28), scale=0.42, col=(200, 240, 255), thickness=1)

    # Connected triangle mesh & lock field when 3 fingers are locked
    if is_locked and thumb_pos != (0, 0) and index_pos != (0, 0) and middle_pos != (0, 0):
        pts = np.array([thumb_pos, index_pos, middle_pos], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw centroid locking core
        cx, cy = pinch_pos
        cv2.circle(frame, (cx, cy), 14, (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 120), -1)

    # Pinch / Fire ripple animation at centroid
    if is_pinching or click_fired:
        cx, cy = pinch_pos
        cv2.circle(frame, (cx, cy), 40, (0, 80, 255), 4)
        cv2.circle(frame, (cx, cy), 60, (0, 220, 255), 2)
        _text(frame, "3-PINCH TRIGGER!", (cx - 65, cy - 45), scale=0.6, col=(0, 255, 255), thickness=2)


def draw_hud(
    frame: np.ndarray,
    state: str,
    birds_left: list,
    click_mode: str,
    z_debug: float,
    xy_drift: float,
    click_fired: bool,
    score: int,
    level_idx: int,
    fps: int,
    tof_active: bool = False,
    tof_z_m: float = 0.0,
    depth_source: str = "RGB MediaPipe Estimate",
    gesture: dict | None = None,
    magnification: float = 2.0,
):
    """Render full head-up display overlay (modifies frame in-place)."""
    h, w = frame.shape[:2]

    # Draw 3-finger lock animation directly on canvas if gesture is provided
    if gesture:
        draw_3finger_lock_overlay(frame, gesture)

    # Top-left info box
    _text(frame, f"SCORE: {score}", (20, 40), scale=0.9, col=(0, 255, 200), thickness=2)
    _text(frame, f"LEVEL: {level_idx + 1}", (20, 70), scale=0.6, col=(200, 200, 200))
    _text(frame, f"FPS: {fps}", (20, 95), scale=0.5, col=(150, 150, 150))
    _text(frame, f"MAGNIFICATION: {magnification:.1f}x", (20, 118), scale=0.5, col=(0, 220, 255))

    # Birds remaining icons
    _text(frame, "BIRDS LEFT:", (20, h - 50), scale=0.5, col=(180, 180, 180))
    for i, kind in enumerate(birds_left):
        tmp = Bird(kind, 100 + i * 55, h - 30)
        tmp.draw(frame, scale=0.7)

    # ── 3-Finger Lock & Gesture Debug Panel — top-right (below camera box)
    box_w, box_h = 250, 145
    box_x, box_y = w - box_w - 12, 187   # top-right, directly below camera preview

    draw_rect_alpha(frame, box_x, box_y, box_x + box_w, box_y + box_h, (15, 18, 28), 0.75)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (60, 90, 130), 1)

    # Title
    _text(frame, "3-FINGER LOCK SYSTEM", (box_x + 8, box_y + 18), scale=0.48, col=(0, 220, 255), thickness=2)

    # Lock progress parameters from gesture dict
    progress = gesture.get("lock_progress", 0.0) if gesture else 0.0
    is_locked = gesture.get("three_finger_locked", False) if gesture else False
    locked_fingers = gesture.get("locked_fingers", (False, False, False)) if gesture else (False, False, False)

    # Progress bar
    bar_x, bar_y = box_x + 8, box_y + 30
    bar_w, bar_h = 234, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 35, 45), -1)
    bar_col = (0, 255, 120) if is_locked else (0, 200, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), bar_col, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 120, 150), 1)

    # Finger lock badges status
    t_icon = "THUMB 🔒" if locked_fingers[0] else "THUMB ⏳"
    i_icon = "INDEX 🔒" if locked_fingers[1] else "INDEX ⏳"
    m_icon = "MID 🔒" if locked_fingers[2] else "MID ⏳"

    t_col = (0, 255, 120) if locked_fingers[0] else (160, 160, 160)
    i_col = (0, 255, 120) if locked_fingers[1] else (160, 160, 160)
    m_col = (0, 255, 120) if locked_fingers[2] else (160, 160, 160)

    _text(frame, t_icon, (box_x + 8, box_y + 65), scale=0.40, col=t_col, thickness=1)
    _text(frame, i_icon, (box_x + 88, box_y + 65), scale=0.40, col=i_col, thickness=1)
    _text(frame, m_icon, (box_x + 168, box_y + 65), scale=0.40, col=m_col, thickness=1)

    # Overall Status indicator
    if click_fired:
        st_txt, st_col = "ACTION FIRED! 💥", (0, 80, 255)
    elif is_locked:
        st_txt, st_col = "LOCKED! PINCH 3 TO FIRE", (0, 255, 120)
    elif progress > 0.0:
        st_txt, st_col = f"LOCKING ({int(progress*100)}%)...", (0, 200, 255)
    else:
        st_txt, st_col = "KEEP 3 FINGERS APART", (140, 140, 150)

    _text(frame, f"Status: {st_txt}", (box_x + 8, box_y + 95), scale=0.44, col=st_col, thickness=2)
    _text(frame, "Z-Push & 2-Pinch: Disabled", (box_x + 8, box_y + 122), scale=0.38, col=(120, 140, 160))

    # ── Key hints — bottom-right ──────────────────────────────────────────
    hints = {
        "SELECTION": "Lock 3 fingers apart -> Pinch 3 to select bird",
        "ARMED":     "Pinch 3 to aim & pull  |  Release pinch to fire",
        "FLIGHT":    "Bird in flight...",
    }
    hint = hints.get(state, "")
    _text(frame, hint, (w - 420, h - 20), scale=0.55, col=(200, 200, 200))

    # Level-switch hint
    _text(frame, "1/2/3: Switch Level  |  +/-: Magnification  |  R: Restart",
          (w - 480, h - 44), scale=0.45, col=(160, 160, 160))


def draw_done_overlay(frame: np.ndarray, score: int = 0):
    """Full-screen semi-transparent 'Done' screen with final score."""
    h, w = frame.shape[:2]
    draw_rect_alpha(frame, 0, 0, w, h, (10, 10, 10), 0.65)
    _text(frame, "ALL BIRDS USED!", (w//2 - 200, h//2 - 50),
          scale=1.8, col=(0, 220, 255), thickness=3)
    _text(frame, f"Final Score: {score}", (w//2 - 140, h//2 + 10),
          scale=1.2, col=(0, 255, 200), thickness=2)
    _text(frame, "Press  R  to restart  |  1/2/3  to change level",
          (w//2 - 260, h//2 + 60),
          scale=0.8, col=(220, 220, 220), thickness=2)
