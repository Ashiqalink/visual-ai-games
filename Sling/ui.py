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

from instructions import draw_card

from bird import Bird
from config import (
    GRAVITY, AIR_DRAG,
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
    rect = np.empty_like(sub)
    rect[:] = color
    cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)


def draw_ground(frame: np.ndarray, floor_y: int = 660):
    """Draw semi-transparent sky gradient + ground strip."""
    h, w = frame.shape[:2]
    # Sky blue tint at top (alpha blended via ROI slice, no frame.copy())
    draw_rect_alpha(frame, 0, 0, w, floor_y, (200, 160, 80), 0.18)
    # Ground
    cv2.rectangle(frame, (0, floor_y), (w, h), (40, 130, 60), -1)
    cv2.rectangle(frame, (0, floor_y), (w, floor_y+6), (30, 100, 45), -1)


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

    # Instruction
    _text(frame, "3-FINGER LOCK: Keep apart to lock -> Pinch 3 to select/fire | 3D Engine Active",
          (cx - 290, cy + 95), scale=0.52, col=(180, 220, 255))



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
    """Render full head-up display overlay (modifies frame in-place)."""
    h, w = frame.shape[:2]

    # Fingertip markers on the canvas — the only on-hand drawing left
    if gesture:
        draw_hand_sign_overlay(frame, gesture)

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

    # ── Hand Sign & Gesture Panel — top-right (below camera box) ──────────
    #
    # This panel is now the only place the input state is legible: the cursor
    # on the canvas is a bare ring, so anything the player needs to reason
    # about — tracking, sign, pinch, fire, phase — has to be readable here.
    box_w, box_h = 250, 232
    box_x, box_y = w - box_w - 12, 187   # top-right, directly below camera preview

    draw_rect_alpha(frame, box_x, box_y, box_x + box_w, box_y + box_h, (15, 18, 28), 0.75)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (60, 90, 130), 1)

    hand_visible = gesture.get("hand_visible", False) if gesture else False
    sign = gesture.get("hand_sign", "unknown") if gesture else "unknown"
    fingers_ext = gesture.get("fingers_extended", (False,) * 5) if gesture else (False,) * 5
    smoothing_on = gesture.get("smoothing_enabled", True) if gesture else True
    is_pinching = gesture.get("is_pinching", False) if gesture else False
    sign_label, sign_col = SIGN_STYLE.get(sign, SIGN_STYLE["unknown"])

    # Title, with a tracking lamp on the right edge. Losing the hand is the
    # single most confusing failure — the cursor simply stops — so it gets the
    # most prominent slot in the panel.
    _text(frame, "HAND SIGN CONTROL", (box_x + 8, box_y + 18), scale=0.48, col=(0, 220, 255), thickness=2)
    lamp_col = (0, 255, 120) if hand_visible else (60, 60, 200)
    cv2.circle(frame, (box_x + box_w - 16, box_y + 13), 5, lamp_col, -1)
    _text(frame, "TRACKING" if hand_visible else "NO HAND",
          (box_x + box_w - 78, box_y + 34), scale=0.36, col=lamp_col, thickness=1)

    # Current sign
    _text(frame, f"Sign: {sign_label}", (box_x + 8, box_y + 44), scale=0.55, col=sign_col, thickness=2)

    # Per-finger extension pips — the raw signal behind the classification, so a
    # misread sign can be traced to the finger responsible.
    pip_names = ("T", "I", "M", "R", "P")
    for i, (nm, ext) in enumerate(zip(pip_names, fingers_ext)):
        px = box_x + 12 + i * 30
        py = box_y + 70
        col_p = (0, 255, 120) if ext else (70, 70, 85)
        cv2.circle(frame, (px, py), 9, col_p, -1 if ext else 2)
        _text(frame, nm, (px - 4, py + 22), scale=0.38, col=(170, 170, 180), thickness=1)

    # ── Pinch and fire lamps — what main.py used to draw on the hand ──────
    global _fire_flash_until
    now = time.time()
    if click_fired:
        _fire_flash_until = now + _FIRE_FLASH_SECONDS
    firing = now < _fire_flash_until

    pinch_col = CURSOR_PINCH_COL if is_pinching else (70, 70, 85)
    cv2.circle(frame, (box_x + 14, box_y + 112), 6, pinch_col, -1 if is_pinching else 2)
    _text(frame, "PINCH" if is_pinching else "pinch", (box_x + 26, box_y + 117),
          scale=0.42, col=pinch_col, thickness=2 if is_pinching else 1)

    fire_col = CURSOR_FIRE_COL if firing else (70, 70, 85)
    cv2.circle(frame, (box_x + 120, box_y + 112), 6, fire_col, -1 if firing else 2)
    _text(frame, "FIRED" if firing else "fire", (box_x + 132, box_y + 117),
          scale=0.42, col=fire_col, thickness=2 if firing else 1)

    # Status line — always the instruction for what to do next. The fire event
    # itself is the lamp above; repeating it here would cost the one row that
    # tells the player what to do with their hand.
    if state == "READY":
        # READY reverses the meaning of both signs: the fist is carrying the bird
        # rather than drawing it, and opening the hand puts it back instead of
        # firing. Saying "open to fire" here would be actively wrong.
        st_txt, st_col = ("HOLD STILL to lock aim" if sign == "fist"
                          else "Re-close fist to keep the bird"), (0, 200, 255)
    elif sign == "fist":
        st_txt, st_col = "HOLDING - open to fire", (0, 140, 255)
    elif sign == "open_palm":
        st_txt, st_col = "HAND OPEN - ready", (0, 255, 120)
    else:
        st_txt, st_col = "Make a fist to grab", (140, 140, 150)

    _text(frame, st_txt, (box_x + 8, box_y + 140), scale=0.44, col=st_col, thickness=2)

    # Phase + click mode: which of SELECTION / ARMED / FLIGHT the game thinks
    # it is in. Without it, a fist that does nothing is indistinguishable from
    # a fist that was not recognised.
    _text(frame, f"Phase: {state}", (box_x + 8, box_y + 160), scale=0.42,
          col=(0, 220, 255), thickness=1)
    if click_mode:
        _text(frame, f"Mode: {click_mode}", (box_x + 130, box_y + 160), scale=0.38,
              col=(150, 160, 175), thickness=1)

    # Raw push / drift, previously computed and passed in but never displayed.
    _text(frame, f"Z push: {z_debug:+.3f}   Drift: {xy_drift:.1f}px",
          (box_x + 8, box_y + 178), scale=0.38, col=(170, 180, 195), thickness=1)

    # Depth, when a ToF source is driving it
    depth_txt = (f"Depth: {tof_z_m:.2f}m  [{depth_source}]" if tof_active
                 else f"Depth: {depth_source}")
    _text(frame, depth_txt[:38], (box_x + 8, box_y + 194), scale=0.34,
          col=(0, 220, 200) if tof_active else (120, 130, 145), thickness=1)

    # Jitter + smoothing state
    jitter = gesture.get("jitter", {}) if gesture else {}
    raw_std = jitter.get("raw_jitter_std", 0.0)
    smooth_std = jitter.get("smoothed_jitter_std", 0.0)
    red_pct = jitter.get("jitter_reduction_pct", 0.0)

    j_text = f"Jitter: {smooth_std:.1f}px (Raw: {raw_std:.1f}px | -{red_pct:.0f}%)"
    _text(frame, j_text, (box_x + 8, box_y + 210), scale=0.38, col=(0, 240, 200), thickness=1)

    sm_txt = "Smoothing: ON" if smoothing_on else "Smoothing: OFF (raw)"
    sm_col = (120, 140, 160) if smoothing_on else (0, 180, 255)
    _text(frame, sm_txt, (box_x + 8, box_y + 226), scale=0.34, col=sm_col)

    # ── Key hints — bottom-right ──────────────────────────────────────────
    hints = {
        "SELECTION": "Make a FIST over a bird to grab it",
        "READY":     "Keep the fist | Bring your hand somewhere comfortable | HOLD STILL to lock aim",
        "ARMED":     "Move fist to pull | OPEN HAND to fire | Short pull = put the bird back",
        "FLIGHT":    "Bird in flight...",
    }
    hint = hints.get(state, "")
    _text(frame, hint, (w - 460, h - 20), scale=0.55, col=(200, 200, 200))

    # Level-switch hint
    _text(frame, "H: How to play  |  1/2/3: Level  |  +/-: Magnification  |  R: Restart  |  K: Smoothing  |  L: ToF Stab",
          (w - 700, h - 44), scale=0.45, col=(160, 160, 160))


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
        
        # Draw Stars
        star_str = "★" * stars + "☆" * (3 - stars)
        _text(frame, star_str, (w//2 - 120, h//2 - 10), scale=2.5, col=(0, 215, 255), thickness=4)
        
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
