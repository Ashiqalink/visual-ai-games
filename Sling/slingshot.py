"""
slingshot.py — Slingshot rendering: organic wooden Y-frame, leather bindings,
               volumetric elastic bands, bird pouch, and grounded base pedestal.

Visual improvements over the plain line-based version:
  - Sculpted tapered wooden handle & curved fork arms with bark shading and
    wood-grain highlights
  - Criss-cross leather / twine cord lashings at the Y-junction
  - Brass ferrule collar bands with rivet dots at the fork tips
  - 3-pass catenary elastic bands: dark shadow edge → vivid tension core →
    glossy specular sheen line; colour and thickness scale with pull distance
  - Leather bird-pouch cup drawn behind the bird when armed or on the sling
  - Organic dirt-mound base pedestal with a soft ground shadow
  - Snap-back vibration animation on release (unchanged API)
"""

import cv2
import numpy as np
import math
from visual_ai import lerp, Vector2
from config import (
    SLING_X, SLING_Y,
    FORK_SPREAD_X, FORK_RISE_Y, HANDLE_DROP_Y,
    SLING_WOOD_COLOR, SLING_WOOD_DARK,
    SLING_ELASTIC_NEAR, SLING_ELASTIC_FAR,
    SLING_SNAP_DURATION,
)

# ── Geometry ───────────────────────────────────────────────────────────────────
FORK_LEFT  = (SLING_X - FORK_SPREAD_X, SLING_Y - FORK_RISE_Y)
FORK_RIGHT = (SLING_X + FORK_SPREAD_X, SLING_Y - FORK_RISE_Y)
HANDLE_BOT = (SLING_X, SLING_Y + HANDLE_DROP_Y)

# ── Colours (all BGR) ─────────────────────────────────────────────────────────
# Wood tones
_W_BARK     = (28,  72, 130)   # darkest — bark edges
_W_BASE     = (42, 108, 175)   # mid warm brown
_W_GRAIN    = (60, 140, 205)   # lighter grain lines
_W_SUN      = (90, 175, 235)   # directional sun highlight
# Leather / twine
_L_DARK     = (20,  38,  68)   # deep shadow leather
_L_MID      = (35,  62, 110)   # cord wrap mid tone
_L_LIGHT    = (55, 100, 160)   # cord highlight
# Metal (ferrule bands on fork tips)
_M_DARK     = (45,  55,  65)
_M_BASE     = (80,  90, 100)
_M_SHINE    = (170, 185, 195)
# Ground / shadow
_SHADOW_COL = (10,  15,  22)
_DIRT_COL   = (38,  68,  88)
_GRASS_COL  = (52, 118,  60)
# Elastic
ELASTIC_COL_NEAR = SLING_ELASTIC_NEAR
ELASTIC_COL_FAR  = SLING_ELASTIC_FAR

# ── Snap-back state ────────────────────────────────────────────────────────────
_snap_timer: int = 0
_snap_from: tuple = (SLING_X, SLING_Y)
_SNAP_DURATION: int = SLING_SNAP_DURATION


def trigger_snap(bird_pos: tuple):
    """Call when transitioning ARMED → FLIGHT to start snap-back."""
    global _snap_timer, _snap_from
    _snap_timer = _SNAP_DURATION
    _snap_from = (int(bird_pos[0]), int(bird_pos[1]))


def tick():
    """Advance snap-back animation by one frame. Call once per game update."""
    global _snap_timer
    if _snap_timer > 0:
        _snap_timer -= 1


def _snap_pos():
    """Current snap-back elastic endpoint (or None when idle)."""
    if _snap_timer <= 0:
        return None
    t   = 1.0 - _snap_timer / _SNAP_DURATION          # 0 → 1 over duration
    osc = math.sin(t * math.pi * 3.0) * (1 - t) * 0.4
    mix = min(1.0, t + osc)
    x   = _snap_from[0] + mix * (SLING_X - _snap_from[0])
    y   = _snap_from[1] + mix * (SLING_Y - _snap_from[1])
    return (int(x), int(y))


# ── Internal colour helpers ────────────────────────────────────────────────────

def _elastic_color(pull_dist: float) -> tuple:
    t = min(1.0, pull_dist / 150.0)
    return tuple(int(lerp(n, f, t)) for n, f in zip(ELASTIC_COL_NEAR, ELASTIC_COL_FAR))


def _elastic_thickness(pull_dist: float) -> int:
    t = min(1.0, pull_dist / 150.0)
    return int(lerp(4.0, 9.0, t))


# ── Catenary curve ─────────────────────────────────────────────────────────────

def _catenary_points(start: tuple, end: tuple, max_sag: float = 20.0,
                     n: int = 14) -> np.ndarray:
    dist = Vector2(start[0], start[1]).distance_to(Vector2(end[0], end[1]))
    sag  = max_sag * max(0.0, 1.0 - dist / 180.0)
    pts  = []
    for i in range(n + 1):
        t = i / n
        x = lerp(start[0], end[0], t)
        y = lerp(start[1], end[1], t)
        y += sag * 4.0 * t * (1.0 - t)
        pts.append([int(x), int(y)])
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


# ── Elastic band — 3-pass volumetric draw ─────────────────────────────────────

def _draw_elastic(frame: np.ndarray, start: tuple, end: tuple,
                  pull_dist: float = 0.0):
    e_col  = _elastic_color(pull_dist)
    thick  = _elastic_thickness(pull_dist)
    points = _catenary_points(start, (int(end[0]), int(end[1])))

    # Pass 1: dark shadow / rim (wider)
    shadow = tuple(max(0, c - 60) for c in e_col)
    cv2.polylines(frame, [points], False, shadow, thick + 4, cv2.LINE_AA)

    # Pass 2: vivid core
    cv2.polylines(frame, [points], False, e_col, thick, cv2.LINE_AA)

    # Pass 3: thin specular sheen on the top edge
    sheen = tuple(min(255, c + 80) for c in e_col)
    cv2.polylines(frame, [points], False, sheen, max(1, thick - 4), cv2.LINE_AA)


# ── Bird pouch (leather cup) ───────────────────────────────────────────────────

def _draw_pouch(frame: np.ndarray, bird_pos: tuple, pull_dist: float = 0.0):
    """Draw a small leather cup behind the bird position."""
    px, py = int(bird_pos[0]), int(bird_pos[1])
    # Size scales slightly with pull to look like it's being stretched
    r = int(lerp(10, 14, min(1.0, pull_dist / 150.0)))

    # Shadow layer
    cv2.circle(frame, (px, py + 1), r + 3, _L_DARK, -1, cv2.LINE_AA)
    # Leather base
    cv2.circle(frame, (px, py), r + 2, _L_MID, -1, cv2.LINE_AA)
    # Centre highlight
    cv2.circle(frame, (px - r//4, py - r//4), max(2, r//2), _L_LIGHT, -1, cv2.LINE_AA)
    # Stitching border
    cv2.circle(frame, (px, py), r + 2, _L_DARK, 1, cv2.LINE_AA)


# ── Ground pedestal ────────────────────────────────────────────────────────────

def _draw_pedestal(frame: np.ndarray):
    """Draw an organic dirt-mound base and ground shadow beneath the handle."""
    bx, by = SLING_X, HANDLE_BOT[1]

    # Ground shadow ellipse (blended overlay)
    cv2.ellipse(frame, (bx, by + 8), (34, 8), 0, 0, 360, _SHADOW_COL, -1, cv2.LINE_AA)

    # Dirt mound (stacked ellipses for organic look)
    cv2.ellipse(frame, (bx, by + 2), (22, 10), 0, 0, 360, _DIRT_COL, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (bx, by + 4), (28, 7),  0, 0, 360, _DIRT_COL, -1, cv2.LINE_AA)

    # Grass lip
    cv2.ellipse(frame, (bx, by + 1), (26, 6),  0, 180, 360, _GRASS_COL, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (bx, by + 1), (26, 6),  0, 180, 360, (40, 90, 48), 1, cv2.LINE_AA)


# ── Wooden structure ───────────────────────────────────────────────────────────

def _draw_structure(frame: np.ndarray):
    """Draw the sculpted wooden handle and fork arms."""

    sx, sy  = SLING_X, SLING_Y
    fl      = FORK_LEFT
    fr      = FORK_RIGHT
    hb      = HANDLE_BOT
    jx, jy  = sx, sy   # Y-junction (fork centre)

    # ── Handle — tapering from fork base down to the grip ─────────────────
    # Widths: 14px at top (Y-junction), 10px at grip bottom
    _draw_tapered_line(frame, (jx, jy), hb, 14, 10,
                       _W_BASE, _W_BARK, _W_SUN, _W_GRAIN)

    # ── Left fork arm ─────────────────────────────────────────────────────
    _draw_tapered_line(frame, (jx, jy), fl, 12, 8,
                       _W_BASE, _W_BARK, _W_SUN, _W_GRAIN)

    # ── Right fork arm ────────────────────────────────────────────────────
    _draw_tapered_line(frame, (jx, jy), fr, 12, 8,
                       _W_BASE, _W_BARK, _W_SUN, _W_GRAIN)

    # ── Y-junction lashing (leather twine wrap) ───────────────────────────
    _draw_lashing(frame, jx, jy)

    # ── Fork tip ferrules (metal bands) ──────────────────────────────────
    _draw_ferrule(frame, fl)
    _draw_ferrule(frame, fr)


def _draw_tapered_line(frame: np.ndarray,
                       p0: tuple, p1: tuple,
                       w0: int, w1: int,
                       base_col, bark_col, sun_col, grain_col):
    """Draw a tapered 'beam' from p0 (w0 wide) to p1 (w1 wide) using a stack
    of cross-section segments, giving it a 3-D carved look."""
    n = 8  # number of segments
    for i in range(n):
        t0 = i / n
        t1 = (i + 1) / n
        x0 = int(lerp(p0[0], p1[0], t0))
        y0 = int(lerp(p0[1], p1[1], t0))
        x1 = int(lerp(p0[0], p1[0], t1))
        y1 = int(lerp(p0[1], p1[1], t1))
        w  = int(lerp(w0, w1, (t0 + t1) / 2))

        # Dark bark border (widest)
        cv2.line(frame, (x0, y0), (x1, y1), bark_col, w + 4, cv2.LINE_AA)
        # Base wood fill
        cv2.line(frame, (x0, y0), (x1, y1), base_col, w,     cv2.LINE_AA)
        # Sun-side highlight (left side of stroke)
        cv2.line(frame, (x0 - 2, y0 - 1), (x1 - 2, y1 - 1), sun_col,
                 max(1, w // 3), cv2.LINE_AA)
        # Grain mark (subtle lighter stripe)
        cv2.line(frame, (x0 + 1, y0 + 1), (x1 + 1, y1 + 1), grain_col,
                 max(1, w // 5), cv2.LINE_AA)


def _draw_lashing(frame: np.ndarray, cx: int, cy: int):
    """Criss-cross leather cord wrap at the Y-junction."""
    r = 11  # radius of the lashing region
    # Dark backing disc
    cv2.circle(frame, (cx, cy), r + 3, _L_DARK, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r,     _L_MID,  -1, cv2.LINE_AA)
    # Diagonal cord wraps
    for dy in range(-r + 2, r, 4):
        x0 = cx - r + abs(dy)
        x1 = cx + r - abs(dy)
        col = _L_LIGHT if abs(dy) % 8 < 4 else _L_MID
        cv2.line(frame, (x0, cy + dy), (x1, cy + dy), col, 1, cv2.LINE_AA)
    # Outer rim
    cv2.circle(frame, (cx, cy), r + 3, _L_DARK, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r,     _L_DARK, 1, cv2.LINE_AA)


def _draw_ferrule(frame: np.ndarray, tip: tuple):
    """Brass / metal collar band at a fork tip, with a couple of rivet dots."""
    fx, fy = tip
    # Outer ring (dark metal shadow)
    cv2.circle(frame, (fx, fy), 11, _M_DARK,  -1, cv2.LINE_AA)
    # Metal band
    cv2.circle(frame, (fx, fy),  9, _M_BASE,  -1, cv2.LINE_AA)
    # Shine arc (top-left quadrant)
    cv2.ellipse(frame, (fx - 2, fy - 2), (5, 3), 30, 200, 340, _M_SHINE, 2, cv2.LINE_AA)
    # Two tiny rivet dots
    for angle_deg in (60, 240):
        a = math.radians(angle_deg)
        rx, ry = int(fx + 6 * math.cos(a)), int(fy + 6 * math.sin(a))
        cv2.circle(frame, (rx, ry), 2, _M_DARK,  -1, cv2.LINE_AA)
        cv2.circle(frame, (rx - 1, ry - 1), 1, _M_SHINE, -1, cv2.LINE_AA)


# ── Public API ─────────────────────────────────────────────────────────────────

def draw_back(frame: np.ndarray, bird_pos: tuple | None = None,
              pull_dist: float = 0.0):
    """Draw BACK elastic (left fork → pouch/bird) + pedestal base.
    Call BEFORE drawing the bird."""
    _draw_pedestal(frame)
    target = bird_pos if bird_pos is not None else _snap_pos()
    if target is not None:
        _draw_pouch(frame, target, pull_dist)
        _draw_elastic(frame, FORK_LEFT, target, pull_dist)


def draw_front(frame: np.ndarray, bird_pos: tuple | None = None,
               pull_dist: float = 0.0):
    """Draw FRONT elastic (right fork → pouch/bird) + wooden structure.
    Call AFTER drawing the bird."""
    target = bird_pos if bird_pos is not None else _snap_pos()
    if target is not None:
        _draw_elastic(frame, FORK_RIGHT, target, pull_dist)
    _draw_structure(frame)


def draw(frame: np.ndarray, bird_pos: tuple | None = None,
         pull_dist: float = 0.0):
    """Convenience: draws everything in one call (backward compatible).

    For proper depth layering use draw_back → bird.draw → draw_front.
    """
    draw_back(frame, bird_pos, pull_dist)
    draw_front(frame, bird_pos, pull_dist)
