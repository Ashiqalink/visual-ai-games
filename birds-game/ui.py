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


def draw_ground(frame: np.ndarray, floor_y: int = 660):
    """Draw semi-transparent sky gradient + ground strip."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    # Sky blue tint at top
    cv2.rectangle(overlay, (0, 0), (w, floor_y), (200, 160, 80), -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
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

    # Background panel
    panel_w = spacing * len(bird_types) + 80
    panel_h = 140
    px      = cx - panel_w // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, cy-60), (px+panel_w, cy+panel_h-60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
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
    _text(frame, "PINCH or Z-PUSH on bird to select",
          (cx - 180, cy + 95), scale=0.55, col=(180, 220, 255))


def draw_trajectory(frame: np.ndarray,
                    start_x: float, start_y: float,
                    vx: float, vy: float,
                    gravity: float = 0.45, n_dots: int = 28):
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


def draw_hud(frame: np.ndarray, state: str, birds_left: list,
             click_mode: str = "PINCH", z_debug: float = 0.0,
             xy_drift: float = 0.0, click_fired: bool = False,
             score: int = 0, level_idx: int = 0, fps: int = 0,
             tof_active: bool = False, tof_z_m: float = 0.0,
             depth_source: str = "RGB MediaPipe Estimate"):
    """
    Parameters
    ----------
    state        : current game state string
    birds_left   : list of bird kind strings remaining
    click_mode   : 'PINCH' or 'Z-PUSH' — which was last used
    z_debug      : current Z delta (for debug bar)
    xy_drift     : lateral XY movement since Z push started (px)
    click_fired  : whether Z-click just fired this frame
    score        : current score
    level_idx    : current level index (0-based)
    fps          : current frames per second
    tof_active   : bool indicating if hardware/simulated ToF sensor is active
    tof_z_m      : true physical depth measurement in meters
    depth_source : string label of active depth measurement source
    """
    h, w = frame.shape[:2]

    # State label top-left
    state_cols = {
        "SELECTION": (0, 220, 255),
        "ARMED":     (0, 255, 120),
        "FLIGHT":    (0, 180, 255),
    }
    scol = state_cols.get(state, HUD_TEXT)
    _text(frame, f"State: {state}", (20, 36), scale=0.7, col=scol, thickness=2)

    # ── Score display — top centre ────────────────────────────────────────
    score_text = f"SCORE: {score}"
    _text(frame, score_text, (w // 2 - 80, 36), scale=0.8,
          col=(0, 255, 255), thickness=2)

    # ── Level indicator — below score ─────────────────────────────────────
    level_names = ["Easy", "Medium", "Hard"]
    lname = level_names[min(level_idx, len(level_names) - 1)]
    level_text = f"Level {level_idx + 1}: {lname}"
    _text(frame, level_text, (w // 2 - 70, 62), scale=0.55,
          col=(200, 220, 255))

    # ── FPS — top-left, below state ───────────────────────────────────────
    if fps > 0:
        fps_col = (0, 200, 0) if fps >= 25 else (0, 140, 255)
        _text(frame, f"FPS: {fps}", (20, 62), scale=0.5, col=fps_col)

    # Birds remaining bottom-left
    _text(frame, "Birds:", (20, h - 50), scale=0.6)
    for i, kind in enumerate(birds_left):
        tmp = Bird(kind, 100 + i * 55, h - 30)
        tmp.draw(frame, scale=0.7)

    # ── Z-Push & ToF Debug Panel — top-right ──────────────────────────────
    box_w, box_h = 240, 145
    box_x, box_y = w - box_w - 260, 12   # left of camera box

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (15, 18, 28), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (60, 90, 130), 1)

    # Title
    _text(frame, "GESTURE & ToF DEBUG", (box_x + 8, box_y + 18), scale=0.48, col=(0, 220, 255), thickness=2)

    # ── ToF Sensor Badge ──────────────────────────────────────────────────
    if tof_active:
        tof_badge_col = (0, 255, 180)   # Neon Cyan/Green
        tof_badge_txt = f"ToF SENSOR: ACTIVE ({tof_z_m:.2f}m)"
    else:
        tof_badge_col = (0, 180, 255)   # Amber / Cyan-grey
        tof_badge_txt = "ToF SENSOR: INACTIVE (RGB Est)"

    cv2.rectangle(frame, (box_x + 8, box_y + 24), (box_x + box_w - 8, box_y + 40), (25, 35, 50), -1)
    cv2.rectangle(frame, (box_x + 8, box_y + 24), (box_x + box_w - 8, box_y + 40), tof_badge_col, 1)
    _text(frame, tof_badge_txt, (box_x + 12, box_y + 36), scale=0.40, col=tof_badge_col, thickness=1)

    from physics import Z_CLICK_THRESHOLD_M, Z_CLICK_XY_MAX_PX
    thresh = Z_CLICK_THRESHOLD_M
    ratio  = min(1.0, max(0.0, z_debug / max(thresh, 1e-6)))

    # Progress bar
    bar_x, bar_y = box_x + 8, box_y + 48
    bar_w, bar_h = 224, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 35, 45), -1)
    bar_col = (0, 255, 120) if ratio >= 1.0 else ((0, 180, 255) if ratio > 0.3 else (0, 120, 200))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * ratio), bar_y + bar_h), bar_col, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 120, 150), 1)

    # Threshold marker line
    cv2.line(frame, (bar_x + int(bar_w * 0.98), bar_y - 2), (bar_x + int(bar_w * 0.98), bar_y + bar_h + 2), (255, 255, 255), 2)

    # Numeric metrics
    z_txt = f"Z-Delta: {z_debug:+.3f}m / {thresh:.3f}m"
    _text(frame, z_txt, (box_x + 8, box_y + 78), scale=0.42, col=(230, 230, 230))

    drift_col = (0, 255, 120) if xy_drift < Z_CLICK_XY_MAX_PX else (0, 0, 255)
    drift_txt = f"XY Drift: {xy_drift:.0f}px / {Z_CLICK_XY_MAX_PX}px"
    _text(frame, drift_txt, (box_x + 8, box_y + 96), scale=0.42, col=drift_col)

    # Depth Source readout
    _text(frame, f"Src: {depth_source}", (box_x + 8, box_y + 114), scale=0.38, col=(180, 200, 220))

    # Status indicator
    if click_fired:
        st_txt, st_col = "FIRE! (CLICK)", (0, 80, 255)
    elif ratio >= 1.0:
        st_txt, st_col = "PUSH CONFIRMED", (0, 255, 120)
    elif z_debug > 0.015:
        st_txt, st_col = "PUSHING...", (0, 200, 255)
    else:
        st_txt, st_col = "IDLE", (140, 140, 150)

    _text(frame, f"Status: {st_txt}", (box_x + 8, box_y + 134), scale=0.44, col=st_col, thickness=2)

    # ── Key hints — bottom-right ──────────────────────────────────────────
    hints = {
        "SELECTION": "Pinch or Z-push to select bird",
        "ARMED":     "Pinch & drag to pull  |  Release to launch",
        "FLIGHT":    "Bird in flight...",
    }
    hint = hints.get(state, "")
    _text(frame, hint, (w - 420, h - 20), scale=0.55, col=(200, 200, 200))

    # Level-switch hint
    _text(frame, "1/2/3: Switch Level  |  R: Restart",
          (w - 380, h - 44), scale=0.45, col=(160, 160, 160))


def draw_done_overlay(frame: np.ndarray, score: int = 0):
    """Full-screen semi-transparent 'Done' screen with final score."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    _text(frame, "ALL BIRDS USED!", (w//2 - 200, h//2 - 50),
          scale=1.8, col=(0, 220, 255), thickness=3)
    _text(frame, f"Final Score: {score}", (w//2 - 140, h//2 + 10),
          scale=1.2, col=(0, 255, 200), thickness=2)
    _text(frame, "Press  R  to restart  |  1/2/3  to change level",
          (w//2 - 260, h//2 + 60),
          scale=0.8, col=(220, 220, 220), thickness=2)
