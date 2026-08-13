
"""
slingshot.py — Slingshot rendering: handle, fork arms, elastic bands.

Improvements:
  - Catenary curve for elastic bands (slight sag that reduces when stretched)
  - Elastic thickness scales with pull distance
  - Snap-back vibration animation on release
  - Depth-layered draw calls (back elastic → bird → front elastic → structure)
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

# Slingshot anchor (fork centre, where the bird sits)
FORK_LEFT  = (SLING_X - FORK_SPREAD_X, SLING_Y - FORK_RISE_Y)
FORK_RIGHT = (SLING_X + FORK_SPREAD_X, SLING_Y - FORK_RISE_Y)
HANDLE_BOT = (SLING_X, SLING_Y + HANDLE_DROP_Y)

WOOD_COL  = SLING_WOOD_COLOR
WOOD_DARK = SLING_WOOD_DARK
ELASTIC_COL_NEAR = SLING_ELASTIC_NEAR
ELASTIC_COL_FAR  = SLING_ELASTIC_FAR

# ── Module-level snap-back state ──────────────────────────────────────────────
_snap_timer: int = 0
_snap_from: tuple = (SLING_X, SLING_Y)
_SNAP_DURATION: int = SLING_SNAP_DURATION


def trigger_snap(bird_pos: tuple):
    """Call when transitioning from ARMED → FLIGHT to start snap-back."""
    global _snap_timer, _snap_from
    _snap_timer = _SNAP_DURATION
    _snap_from = (int(bird_pos[0]), int(bird_pos[1]))


def tick():
    """Advance snap-back animation by one frame.  Call once per game update."""
    global _snap_timer
    if _snap_timer > 0:
        _snap_timer -= 1


def _snap_pos():
    """Current snap-back elastic endpoint (or None when idle)."""
    if _snap_timer <= 0:
        return None
    t = 1.0 - _snap_timer / _SNAP_DURATION          # 0 → 1 over duration
    osc = math.sin(t * math.pi * 3.0) * (1 - t) * 0.4   # damped oscillation
    mix = min(1.0, t + osc)
    x = _snap_from[0] + mix * (SLING_X - _snap_from[0])
    y = _snap_from[1] + mix * (SLING_Y - _snap_from[1])
    return (int(x), int(y))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _elastic_color(pull_dist: float):
    """Blend elastic colour using visual_ai.lerp."""
    t = min(1.0, pull_dist / 150.0)
    r = int(lerp(ELASTIC_COL_FAR[2], ELASTIC_COL_NEAR[2], t))
    g = int(lerp(ELASTIC_COL_FAR[1], ELASTIC_COL_NEAR[1], t))
    b = int(lerp(ELASTIC_COL_FAR[0], ELASTIC_COL_NEAR[0], t))
    return (b, g, r)


def _elastic_thickness(pull_dist: float) -> int:
    """Thickness grows from 3 → 7 px with pull distance using visual_ai.lerp."""
    t = min(1.0, pull_dist / 150.0)
    return int(lerp(3.0, 7.0, t))


def _catenary_points(start: tuple, end: tuple, max_sag: float = 18.0,
                     n: int = 10) -> np.ndarray:
    """Generate a parabolic-sag curve approximating a catenary using visual_ai Vector2."""
    dist = Vector2(start[0], start[1]).distance_to(Vector2(end[0], end[1]))
    # Sag decreases as band is stretched taut
    sag = max_sag * max(0.0, 1.0 - dist / 200.0)
    pts = []
    for i in range(n + 1):
        t = i / n
        x = lerp(start[0], end[0], t)
        y = lerp(start[1], end[1], t)
        y += sag * 4.0 * t * (1.0 - t)          # parabolic sag
        pts.append([int(x), int(y)])
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _draw_elastic(frame: np.ndarray, start: tuple, end: tuple,
                  pull_dist: float = 0.0):
    """Draw a single elastic band as a catenary curve."""
    e_col  = _elastic_color(pull_dist)
    thick  = _elastic_thickness(pull_dist)
    points = _catenary_points(start, (int(end[0]), int(end[1])))
    cv2.polylines(frame, [points], False, e_col, thick)


def _draw_structure(frame: np.ndarray):
    """Draw the wooden handle and fork arms with PBR wood specular highlights."""
    # Handle (thick vertical)
    cv2.line(frame, (SLING_X, SLING_Y), HANDLE_BOT, WOOD_COL, 14)
    cv2.line(frame, (SLING_X - 3, SLING_Y), (HANDLE_BOT[0] - 3, HANDLE_BOT[1]), (220, 160, 90), 2)
    cv2.line(frame, (SLING_X, SLING_Y), HANDLE_BOT, WOOD_DARK, 3)

    # Fork arms
    cv2.line(frame, (SLING_X, SLING_Y), FORK_LEFT,  WOOD_COL, 12)
    cv2.line(frame, (SLING_X, SLING_Y), FORK_RIGHT, WOOD_COL, 12)
    cv2.line(frame, (SLING_X - 2, SLING_Y), (FORK_LEFT[0] - 2, FORK_LEFT[1]), (220, 160, 90), 2)
    cv2.line(frame, (SLING_X + 2, SLING_Y), (FORK_RIGHT[0] + 2, FORK_RIGHT[1]), (220, 160, 90), 2)
    # inner dark stripe for depth
    cv2.line(frame, (SLING_X, SLING_Y), FORK_LEFT,  WOOD_DARK, 3)
    cv2.line(frame, (SLING_X, SLING_Y), FORK_RIGHT, WOOD_DARK, 3)
    # Fork tips (small circles)
    cv2.circle(frame, FORK_LEFT,  8, WOOD_COL, -1)
    cv2.circle(frame, FORK_LEFT,  3, (240, 180, 110), -1)
    cv2.circle(frame, FORK_RIGHT, 8, WOOD_COL, -1)
    cv2.circle(frame, FORK_RIGHT, 3, (240, 180, 110), -1)


# ── Public API ────────────────────────────────────────────────────────────────

def draw_back(frame: np.ndarray, bird_pos: tuple | None = None,
              pull_dist: float = 0.0):
    """Draw the BACK elastic (left fork → bird).  Call BEFORE drawing the bird."""
    target = bird_pos if bird_pos is not None else _snap_pos()
    if target is not None:
        _draw_elastic(frame, FORK_LEFT, target, pull_dist)


def draw_front(frame: np.ndarray, bird_pos: tuple | None = None,
               pull_dist: float = 0.0):
    """Draw the FRONT elastic (right fork → bird) + wooden structure.
    Call AFTER drawing the bird."""
    target = bird_pos if bird_pos is not None else _snap_pos()
    if target is not None:
        _draw_elastic(frame, FORK_RIGHT, target, pull_dist)
    _draw_structure(frame)


def draw(frame: np.ndarray, bird_pos: tuple | None = None,
         pull_dist: float = 0.0):
    """Convenience: draws everything in one call (backward compatible).

    For proper depth layering, prefer calling draw_back → bird.draw → draw_front.
    """
    draw_back(frame, bird_pos, pull_dist)
    draw_front(frame, bird_pos, pull_dist)
