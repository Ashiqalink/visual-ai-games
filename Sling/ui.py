"""
ui.py — HUD, bird carousel, trajectory preview, click-mode indicator,
         score display, level indicator, FPS counter.
"""

import cv2
import numpy as np
import math
import sys
import os
import time

# Ensure visual ai game engine imports are accessible
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'visual ai game engine', 'src'))
# ...and the games root, for the shared how-to-play card. main.py already puts
# it on the path; doing it here too means ui.py imports standalone as well.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from visual_ai import Renderer3D, Mesh3D, Transform3D, Camera3D, Material, predict_projectile_trajectory
from visual_ai.imaging import blit_ellipse_alpha

from instructions import draw_card

from bird import Bird
from config import (
    GRAVITY, AIR_DRAG,
    FLOOR_Y,
    BIRD_ORDER, 
    BIRD_COLOURS as COLOURS, 
    BIRD_RADII as RADII,
    HUD_TEXT_COL as HUD_TEXT,
    SHADOW_COL,
    TRAJ_COL,
    OVERLAY_ALPHA,
    CAROUSEL_Y,
    CAROUSEL_SPACING,
    CAROUSEL_PANEL_H,
    LEVEL_NAMES,
    LIGHTING_SETTINGS,
    # Pinch and fire used to be drawn on the canvas by main.py. They are panel
    # rows now, and keep their original colours so the meaning carries over.
    CURSOR_PINCH_COL,
    CURSOR_FIRE_COL,
)

#: The engine sets ``click_just_fired`` for exactly one frame. At 60fps that is
#: 16ms of lit text — too short to read, and the on-canvas "FIRE!" flash that
#: used to cover for it is gone. Latch the panel row instead.
_FIRE_FLASH_SECONDS = 0.6
_fire_flash_until = 0.0

# Global 3D Renderer instance for UI showcase
_ui_cam3d = Camera3D(fov=60.0, position=(0.0, 0.0, 500.0), screen_width=1280.0, screen_height=720.0)
_ui_renderer3d = Renderer3D(camera=_ui_cam3d)
_carousel_3d_mesh = Mesh3D.create_sphere(radius=28.0, rings=10, sectors=14)
_trophy_3d_mesh = Mesh3D.create_pyramid(width=45.0, height=60.0)
_trophy_angle_3d = 0.0




# ── Ground shadow helper ──────────────────────────────────────────────────────

def draw_ground_shadow(frame: np.ndarray, cx: int, cy_bottom: int,
                       shadow_w: int, shadow_h: int,
                       min_alpha: float = 0.08,
                       max_range: int = 160,
                       range_divisor: float = 160.0,
                       light_divisor: float = 120.0,
                       center_y_offset: int = 4):
    """Draw a ground-projected drop shadow ellipse under an object.

    Parameters
    ----------
    cx : int
        Horizontal centre of the source object (px).
    cy_bottom : int
        Bottom edge of the source object (px), used for shadow distance.
    shadow_w, shadow_h : int
        Base ellipse half-widths before distance attenuation.
    min_alpha : float
        Floor opacity so shadows remain visible at max range.
    max_range : int
        Maximum shadow_offset_y before the shadow is culled.
    range_divisor : float
        Divisor for the distance attenuation formula.
    light_divisor : float
        Divisor for the horizontal light-shift parallax.
    center_y_offset : int
        Vertical offset from FLOOR_Y for the shadow centre.
    """
    if not LIGHTING_SETTINGS.get("SHADOWS_ENABLED", True):
        return
    shadow_offset_y = int(FLOOR_Y - cy_bottom)
    if not (0 <= shadow_offset_y < max_range):
        return

    base_opacity = LIGHTING_SETTINGS.get("SHADOW_OPACITY", 0.45)
    shadow_alpha = max(min_alpha,
                       base_opacity * (1.0 - shadow_offset_y / range_divisor))

    angle_rad = math.radians(LIGHTING_SETTINGS.get("LIGHT_ANGLE", 45.0))
    light_shift_x = int(
        math.cos(angle_rad)
        * LIGHTING_SETTINGS.get("SHADOW_OFFSET_X", 15.0)
        * (1.0 + shadow_offset_y / light_divisor)
    )

    shadow_col = LIGHTING_SETTINGS.get("SHADOW_COLOR", (10, 15, 20))
    shadow_center = (cx + light_shift_x, int(FLOOR_Y - center_y_offset))
    blit_ellipse_alpha(frame, shadow_center,
                       (max(2, shadow_w), max(2, shadow_h)),
                       shadow_col, shadow_alpha)


def _text(frame, txt, pos, scale=0.7, col=HUD_TEXT, thickness=1):
    """Draws text with a drop-shadow for legibility."""
    x, y = pos
    cv2.putText(frame, txt, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, SHADOW_COL, thickness+1, cv2.LINE_AA)
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, col, thickness, cv2.LINE_AA)


def _star_points(cx: float, cy: float, r_outer: float) -> np.ndarray:
    """Ten alternating outer/inner vertices of a five-pointed star, apex up."""
    r_inner = r_outer * 0.382                      # classic pentagram ratio
    return np.array(
        [(cx + (r_outer if i % 2 == 0 else r_inner) * math.cos(math.radians(-90 + i * 36)),
          cy + (r_outer if i % 2 == 0 else r_inner) * math.sin(math.radians(-90 + i * 36)))
         for i in range(10)],
        dtype=np.int32,
    )


def _draw_star(frame: np.ndarray, cx: int, cy: int, r: int,
               col: tuple[int, int, int], filled: bool = True, thickness: int = 2):
    """
    A five-pointed star drawn as a polygon, with the same drop-shadow as `_text`.

    Polygons rather than a glyph because cv2.putText only speaks Hershey, which
    is ASCII-only: the '*'/'o' star characters this used to draw rendered as '?'
    boxes, so the win screen's whole 1/2/3 rating was invisible.
    """
    pts = _star_points(cx, cy, r)
    shadow = _star_points(cx + 2, cy + 2, r)
    if filled:
        cv2.fillPoly(frame, [shadow], SHADOW_COL, cv2.LINE_AA)
        cv2.fillPoly(frame, [pts], col, cv2.LINE_AA)
    else:
        cv2.polylines(frame, [shadow], True, SHADOW_COL, thickness + 1, cv2.LINE_AA)
        cv2.polylines(frame, [pts], True, col, thickness, cv2.LINE_AA)


def draw_rect_alpha(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                    color: tuple[int, int, int], alpha: float):
    """Draws a semi-transparent filled rectangle using ROI slicing to avoid full-frame hardcopies."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = frame[y1:y2, x1:x2]
    rect = np.empty_like(sub)
    rect[:] = color
    cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)


_bg_cache: np.ndarray | None = None


def _build_background(w: int, h: int, floor_y: int) -> np.ndarray:
    """Paint the full scene backdrop once: gradient sky, sun, clouds, layered
    hills and a shaded ground strip. Every per-frame cost the old tint pass
    paid is now a single memcpy in draw_ground."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky — deep blue at the top easing into a warm pale horizon.
    t = np.linspace(0.0, 1.0, floor_y, dtype=np.float32)[:, None]
    top = np.array([150, 96, 38], np.float32)     # BGR deep sky blue
    hor = np.array([212, 202, 156], np.float32)   # pale warm horizon
    bg[:floor_y] = (top[None, :] * (1 - t) + hor[None, :] * t)[:, None, :].astype(np.uint8)

    # Sun with a soft glow, upper right (away from the slingshot on the left).
    sun_x, sun_y = int(w * 0.82), int(floor_y * 0.20)
    for r, a in ((70, 0.10), (48, 0.16), (30, 0.30)):
        ov = bg.copy()
        cv2.circle(ov, (sun_x, sun_y), r, (200, 235, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(ov, a, bg, 1 - a, 0, bg)
    cv2.circle(bg, (sun_x, sun_y), 20, (210, 245, 255), -1, cv2.LINE_AA)

    # Clouds — flat stylised puffs, low alpha so they stay behind the action.
    ov = bg.copy()
    for cx0, cy0, s in ((int(w*0.16), int(floor_y*0.18), 1.0),
                        (int(w*0.52), int(floor_y*0.10), 0.8),
                        (int(w*0.68), int(floor_y*0.30), 0.65)):
        for dx, dy, rw, rh in ((-40, 4, 42, 16), (0, -8, 52, 22), (44, 4, 38, 15)):
            cv2.ellipse(ov, (cx0 + int(dx*s), cy0 + int(dy*s)),
                        (int(rw*s), int(rh*s)), 0, 0, 360, (250, 250, 245), -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.45, bg, 0.55, 0, bg)

    # Two hill layers on the horizon — far one hazier, near one greener.
    xs = np.arange(w)
    far_y = (floor_y - 58 - 26 * np.sin(xs / 210.0 + 0.8)
             - 12 * np.sin(xs / 66.0 + 2.1)).astype(np.int32)
    near_y = (floor_y - 26 - 18 * np.sin(xs / 150.0 + 3.4)
              - 8 * np.sin(xs / 48.0)).astype(np.int32)
    for ys, col in ((far_y, (150, 140, 96)), (near_y, (96, 122, 66))):
        pts = np.vstack([np.column_stack([xs, ys]),
                         [[w - 1, floor_y], [0, floor_y]]]).astype(np.int32)
        cv2.fillPoly(bg, [pts], col)

    # Ground — vertical gradient with a bright grass lip at the floor line.
    gt = np.linspace(0.0, 1.0, h - floor_y, dtype=np.float32)[:, None]
    g_top = np.array([64, 142, 74], np.float32)
    g_bot = np.array([34, 84, 44], np.float32)
    bg[floor_y:] = (g_top[None, :] * (1 - gt) + g_bot[None, :] * gt)[:, None, :].astype(np.uint8)
    cv2.rectangle(bg, (0, floor_y), (w, floor_y + 4), (80, 178, 96), -1)
    # Sparse deterministic grass tufts so the strip is not a flat band.
    for gx in range(12, w, 46):
        gy = floor_y + 14 + (gx * 7) % 26
        cv2.line(bg, (gx, gy), (gx - 3, gy - 6), (52, 118, 60), 1, cv2.LINE_AA)
        cv2.line(bg, (gx, gy), (gx + 3, gy - 7), (52, 118, 60), 1, cv2.LINE_AA)
    return bg


def draw_ground(frame: np.ndarray, floor_y: int = 660):
    """Blit the cached painted backdrop (sky, hills, ground) onto the frame."""
    global _bg_cache
    h, w = frame.shape[:2]
    if _bg_cache is None or _bg_cache.shape[0] != h or _bg_cache.shape[1] != w:
        _bg_cache = _build_background(w, h, floor_y)
    np.copyto(frame, _bg_cache)


def draw_carousel(frame: np.ndarray, bird_types: list, selected_idx: int, rot_angle_3d: float = 0.0):
    """
    Draw the bird selection carousel at the top centre with 3D visual_ai element showcase.

    Parameters
    ----------
    bird_types   : ordered list of bird kind strings still available
    selected_idx : index of the currently highlighted bird
    rot_angle_3d : rotation angle for visual_ai 3D element rendering
    """
    h, w = frame.shape[:2]
    cx   = w // 2
    cy   = CAROUSEL_Y
    spacing = CAROUSEL_SPACING

    # Ensure 3D camera screen width/height match current frame size
    _ui_cam3d.screen_width = float(w)
    _ui_cam3d.screen_height = float(h)
    _ui_cam3d.focal_length = (float(w) / 2.0) / math.tan(math.radians(_ui_cam3d.fov / 2.0))
    _ui_cam3d.position[2] = _ui_cam3d.focal_length

    # Background panel (alpha blended via ROI slice)
    panel_w = spacing * len(bird_types) + 80
    panel_h = CAROUSEL_PANEL_H
    px      = cx - panel_w // 2
    draw_rect_alpha(frame, px, cy - 60, px + panel_w, cy + panel_h - 60, (28, 22, 16), 0.55)
    cv2.rectangle(frame, (px, cy-60), (px+panel_w, cy+panel_h-60), (140, 110, 70), 1, cv2.LINE_AA)

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

            # Update 3D showcase rotation on the selected bird
            pass


        # Draw miniature bird 2D details
        tmp = Bird(kind, bx, cy)
        tmp.draw(frame, scale=scale)

        # Label. Kinds are lowercase because they double as asset filenames;
        # title-case them for display.
        label_col = (0, 255, 200) if is_sel else (160, 160, 160)
        _text(frame, kind.title(), (bx - 20, cy + int(RADII[kind]*scale) + 20),
              scale=0.55 if is_sel else 0.45, col=label_col)

    # Instruction — matches the actual control scheme (fist grab / open fire).
    _text(frame, "Point at a bird, close a FIST to pick it up",
          (cx - 175, cy + 95), scale=0.55, col=(180, 220, 255))



def draw_trajectory(frame: np.ndarray,
                    start_x: float, start_y: float,
                    vx: float, vy: float,
                    gravity: float = GRAVITY, air_drag: float = AIR_DRAG, n_dots: int = 40,
                    mass: float = 1.0):
    """Draw dotted parabolic trajectory preview from launch position using visual_ai trajectory prediction."""
    # Step duration (dt=1.0 per frame step to match game engine step)
    trajectory_points = predict_projectile_trajectory(
        start_pos=(start_x, start_y),
        initial_vel=(vx, vy),
        gravity=gravity,
        time_step=1.0,
        num_steps=n_dots,
    )

    for i, (x, y) in enumerate(trajectory_points):
        alpha = 1.0 - (i / len(trajectory_points)) * 0.7  # minimum 30% opacity
        r_dot = max(3, int(6 * alpha))
        bright = int(255 * alpha)
        col = (bright, bright, bright)
        cv2.circle(frame, (int(x), int(y)), r_dot, col, -1)


def draw_settle_ring(frame: np.ndarray,
                     x: float, y: float,
                     progress: float,
                     radius: int = 22,
                     timeout_progress: float = 0.0):
    """
    Feedback for the READY settle phase, drawn on the hand.

    The player has the bird but no aim origin yet, and nothing else on screen
    says so — the bird sits on the fork exactly as it does when idle. This ring
    is the whole affordance: the dashed circle is the stillness tolerance, the
    arc filling clockwise is how close the anchor is to locking, and the thin
    outer arc is the timeout that locks it regardless.
    """
    cx, cy = int(x), int(y)
    progress = max(0.0, min(1.0, progress))

    # Tolerance circle — the box you have to stay inside, drawn as dashes so it
    # reads as a boundary rather than as another progress indicator.
    for a in range(0, 360, 24):
        p1 = (int(cx + radius * math.cos(math.radians(a))),
              int(cy + radius * math.sin(math.radians(a))))
        p2 = (int(cx + radius * math.cos(math.radians(a + 12))),
              int(cy + radius * math.sin(math.radians(a + 12))))
        cv2.line(frame, p1, p2, (120, 130, 145), 1, cv2.LINE_AA)

    # Settle arc — green as it approaches lock.
    arc_r = radius + 10
    col = (int(60 + 40 * progress), int(140 + 115 * progress), int(255 - 135 * progress))
    if progress > 0.0:
        cv2.ellipse(frame, (cx, cy), (arc_r, arc_r), -90, 0, int(360 * progress),
                    col, 3, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), arc_r, (45, 50, 62), 1, cv2.LINE_AA)

    # Auto-lock timeout — outer, dimmer, so it never competes with the settle arc.
    if timeout_progress > 0.02:
        cv2.ellipse(frame, (cx, cy), (arc_r + 7, arc_r + 7), -90,
                    0, int(360 * min(1.0, timeout_progress)),
                    (90, 120, 160), 1, cv2.LINE_AA)

    label = "LOCKING..." if progress > 0.5 else "HOLD STILL TO AIM"
    _text(frame, label, (cx - 70, cy - arc_r - 14), scale=0.5,
          col=col, thickness=2)


# Sign -> (label, BGR colour). Kept here so the overlay and the HUD panel
# below cannot drift apart on naming or colour.
SIGN_STYLE = {
    "fist":      ("GRAB",    (0, 140, 255)),
    "open_palm": ("RELEASE", (0, 255, 120)),
    "point":     ("POINT",   (255, 220, 0)),
    "peace":     ("PEACE",   (255, 0, 255)),
    "unknown":   ("...",     (150, 150, 160)),
}


def draw_hand_sign_overlay(frame: np.ndarray, gesture: dict):
    """
    Draws one small fingertip marker per tracked finger, and nothing else.

    The badge disc and the sign label that used to sit on the pinch midpoint
    are gone: they covered the bird, the band and the trajectory preview at the
    one moment the player needs to see all three. Everything they said — the
    sign, the pinch, the fire — is now read off the HAND SIGN CONTROL panel in
    ``draw_hud``, which lives outside the play area. The markers stay because
    they are the only on-canvas confirmation that tracking is following the
    right fingers; their colour still encodes the current sign.
    """
    if not gesture or not gesture.get("hand_visible", False):
        return

    sign = gesture.get("hand_sign", "unknown")
    _, col = SIGN_STYLE.get(sign, SIGN_STYLE["unknown"])

    for pos in (gesture.get("thumb_pos"), gesture.get("index_pos"), gesture.get("middle_pos")):
        if not pos:
            continue
        x, y = int(pos[0]), int(pos[1])
        if x <= 0 or y <= 0:
            continue
        cv2.circle(frame, (x, y), 5, col, -1)
        cv2.circle(frame, (x, y), 8, (30, 30, 40), 1)


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
    """Render full head-up display overlay (modifies frame in-place).

    Deliberately minimal: one score card top-left, one slim hand-state strip
    under the camera preview, birds-left bottom-left, and a single hint line.
    The debug telemetry (finger pips, pinch/fire lamps, Z/drift, jitter) that
    used to fill the right panel is gone — the tracking lamp, the sign and the
    status line carry the same message.
    """
    h, w = frame.shape[:2]

    # ── Score card — top-left ─────────────────────────────────────────────
    draw_rect_alpha(frame, 14, 14, 236, 92, (24, 20, 14), 0.55)
    cv2.rectangle(frame, (14, 14), (236, 92), (140, 110, 70), 1, cv2.LINE_AA)
    _text(frame, "SCORE", (24, 34), scale=0.45, col=(170, 190, 200))
    _text(frame, f"{score:,}", (24, 66), scale=1.0, col=(0, 255, 200), thickness=2)
    lvl_name = LEVEL_NAMES[level_idx] if 0 <= level_idx < len(LEVEL_NAMES) else str(level_idx + 1)
    _text(frame, f"Level {level_idx + 1} - {lvl_name}", (24, 85), scale=0.45, col=(200, 200, 200))
    _text(frame, f"{fps} fps", (176, 34), scale=0.4, col=(130, 140, 150))

    # ── Birds remaining — bottom-left ─────────────────────────────────────
    n = max(1, len(birds_left))
    draw_rect_alpha(frame, 14, h - 72, 70 + n * 52, h - 12, (24, 20, 14), 0.5)
    _text(frame, "BIRDS", (24, h - 54), scale=0.42, col=(170, 190, 200))
    for i, kind in enumerate(birds_left):
        tmp = Bird(kind, 96 + i * 52, h - 34)
        tmp.draw(frame, scale=0.65)

    # ── Hand-state strip — top-right, below camera preview ────────────────
    hand_visible = gesture.get("hand_visible", False) if gesture else False
    sign = gesture.get("hand_sign", "unknown") if gesture else "unknown"
    smoothing_on = gesture.get("smoothing_enabled", True) if gesture else True
    sign_label, sign_col = SIGN_STYLE.get(sign, SIGN_STYLE["unknown"])

    box_w, box_h = 250, 78
    box_x, box_y = w - box_w - 12, 187
    draw_rect_alpha(frame, box_x, box_y, box_x + box_w, box_y + box_h, (15, 18, 28), 0.65)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (60, 90, 130), 1, cv2.LINE_AA)

    # Tracking lamp + current sign on one row. Losing the hand is the single
    # most confusing failure, so it keeps the brightest slot.
    lamp_col = (0, 255, 120) if hand_visible else (60, 60, 200)
    cv2.circle(frame, (box_x + 14, box_y + 16), 5, lamp_col, -1, cv2.LINE_AA)
    _text(frame, "HAND" if hand_visible else "NO HAND", (box_x + 26, box_y + 21),
          scale=0.42, col=lamp_col, thickness=1)
    _text(frame, sign_label, (box_x + 130, box_y + 22), scale=0.55, col=sign_col, thickness=2)

    # Status line — what to do next in the current phase.
    if state == "READY":
        # READY reverses the signs: the fist carries the bird, opening it puts
        # the bird back rather than firing — "open to fire" would be wrong here.
        st_txt, st_col = ("HOLD STILL to lock aim" if sign == "fist"
                          else "Re-close fist to keep the bird"), (0, 200, 255)
    elif sign == "fist":
        st_txt, st_col = "HOLDING - open to fire", (0, 140, 255)
    elif sign == "open_palm":
        st_txt, st_col = "HAND OPEN - ready", (0, 255, 120)
    else:
        st_txt, st_col = "Make a fist to grab", (150, 150, 160)
    _text(frame, st_txt, (box_x + 8, box_y + 46), scale=0.44, col=st_col, thickness=2)

    # Phase + smoothing state, tiny. Without the phase, a fist that does
    # nothing is indistinguishable from a fist that was not recognised.
    _text(frame, f"Phase: {state}", (box_x + 8, box_y + 66), scale=0.4,
          col=(0, 200, 235), thickness=1)
    if not smoothing_on:
        _text(frame, "raw (K)", (box_x + 150, box_y + 66), scale=0.38, col=(0, 180, 255))
    _text(frame, f"{magnification:.1f}x", (box_x + 205, box_y + 66), scale=0.38,
          col=(150, 170, 185))

    # ── Hints — bottom-right, one phase hint + one key line ───────────────
    hints = {
        "SELECTION": "Make a FIST over a bird to grab it",
        "READY":     "Keep the fist, move anywhere, HOLD STILL to lock aim",
        "ARMED":     "Move fist to pull  |  OPEN HAND to fire",
        "FLIGHT":    "Bird in flight...",
    }
    hint = hints.get(state, "")
    if hint:
        _text(frame, hint, (w - 440, h - 40), scale=0.55, col=(210, 210, 210))
    _text(frame, "H help   1/2/3 level   R restart   +/- aim gain",
          (w - 380, h - 16), scale=0.42, col=(160, 160, 160))


# ── How to play ───────────────────────────────────────────────────────────────
#
# The per-phase hints in draw_hud say what to do *now*; this says what the game
# is. Sling's control is three phases deep — grab, lock an origin, pull — and
# none of it is guessable from a screen showing a bird on a fork, so the game
# opens on the card and `H` brings it back.

INSTRUCTIONS_TITLE = "Sling — fist to grab, open hand to fire"
INSTRUCTIONS_GOAL = (
    "Sling birds at the blocks to bring the structure down. You get the birds "
    "on the shelf and no more, so a wasted shot is a wasted bird."
)
INSTRUCTIONS_CONTROLS = (
    ("Show one hand",
     "the ring on screen follows your index fingertip, and the panel on the "
     "right says which sign the tracker thinks you are making"),
    ("1 · Close a FIST",
     "over the carousel of birds at the top to pick one up"),
    ("2 · Keep the fist",
     "and bring your hand wherever you want to shoot from — nothing is being "
     "pulled yet"),
    ("Hold still a moment",
     "the ring on your hand fills; when it locks, that spot is the aim origin"),
    ("3 · Move the fist",
     "to draw the band back — the dotted arc is where the bird will go, and "
     "pulling further is a harder shot"),
    ("OPEN your hand",
     "to fire. Opening it after only a short pull puts the bird back on the "
     "shelf instead of wasting it"),
)
INSTRUCTIONS_KEYS = (
    ("H", "show this card again"),
    ("1 / 2 / 3", "level: Easy, Medium, Hard"),
    ("R", "restart the level"),
    ("+ / -", "aim magnification — how far the band pulls per cm of hand"),
    ("K", "landmark smoothing on / off"),
    ("L", "ToF depth stabilizer on / off (hold still 3 s)"),
    ("X", "cancel a calibration"),
    ("Q / ESC", "quit"),
)


def draw_instructions_card(frame: np.ndarray):
    """The how-to-play card, drawn over the live game."""
    draw_card(frame, INSTRUCTIONS_TITLE, INSTRUCTIONS_GOAL,
              INSTRUCTIONS_CONTROLS, INSTRUCTIONS_KEYS)


def draw_done_overlay(frame: np.ndarray, score: int = 0, won: bool = False, stars: int = 0, bonus: int = 0, rot_angle_3d: float = 0.0):
    """Full-screen semi-transparent 'Done' screen with final score, stars, and 3D victory trophy."""
    h, w = frame.shape[:2]
    draw_rect_alpha(frame, 0, 0, w, h, (10, 10, 10), 0.75)
    
    if won:
        # Render 3D Gold Trophy Pyramid
        _ui_cam3d.screen_width = float(w)
        _ui_cam3d.screen_height = float(h)
        _ui_cam3d.focal_length = (float(w) / 2.0) / math.tan(math.radians(_ui_cam3d.fov / 2.0))
        _ui_cam3d.position[2] = _ui_cam3d.focal_length

        mat_gold = Material(base_color=(1.0, 0.84, 0.0, 1.0), opacity=0.95)
        rel_x = 0.0
        rel_y = (h / 2.0) - (h / 2.0 - 140)
        t_trophy = Transform3D(x=rel_x, y=rel_y, z=0.0, rx=5.0, ry=rot_angle_3d, rz=0.0, sx=1.2, sy=1.2, sz=1.2)
        _ui_renderer3d.render_mesh(frame, _trophy_3d_mesh, t_trophy, material=mat_gold)

        _text(frame, "LEVEL CLEARED!", (w//2 - 200, h//2 - 90),
              scale=1.8, col=(0, 255, 100), thickness=3)
        
        # Draw Stars — filled for earned, outlined for not.
        star_col = (0, 215, 255)
        star_r, star_gap = 26, 74
        for i in range(3):
            _draw_star(frame, w//2 + (i - 1) * star_gap, h//2 - 18, star_r,
                       star_col, filled=(i < max(0, min(3, stars))))
        
        if bonus > 0:
            _text(frame, f"Unused Birds Bonus: +{bonus}", (w//2 - 160, h//2 + 40), scale=0.8, col=(180, 220, 255), thickness=2)
    else:
        _text(frame, "ALL BIRDS USED!", (w//2 - 200, h//2 - 50),
              scale=1.8, col=(0, 100, 255), thickness=3)
        
    _text(frame, f"Final Score: {score}", (w//2 - 140, h//2 + 90),
          scale=1.2, col=(0, 255, 200), thickness=2)
    _text(frame, "Press  R  to restart  |  1/2/3  to change level",
          (w//2 - 260, h//2 + 140),
          scale=0.8, col=(220, 220, 220), thickness=2)
