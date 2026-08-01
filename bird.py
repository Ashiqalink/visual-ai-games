"""
bird.py — Bird class (5 types, drawn with OpenCV primitives).

Improvements over original:
  - Per-type mass values
  - Flight trail (fading circles behind the bird)
  - Impact pop ring on collision
  - Speed lines during fast flight
  - Floor bounce + linger timer (bird rolls after landing)
  - Air drag during flight
"""

import cv2
import numpy as np
import math
from physics import GRAVITY, FLOOR_Y, AIR_DRAG, BIRD_BOUNCE, BIRD_LINGER, magnitude

# Bird type IDs
RED   = "Red"
CHUCK = "Chuck"
BOMB  = "Bomb"
BLUES = "Blues"
WHITE = "White"

BIRD_ORDER = [RED, CHUCK, BOMB, BLUES, WHITE]

# Base radius for each type (px)
RADII = {RED: 28, CHUCK: 26, BOMB: 30, BLUES: 22, WHITE: 26}

# Per-type mass (affects damage dealt + knockback received)
MASSES = {RED: 1.0, CHUCK: 0.7, BOMB: 1.5, BLUES: 0.5, WHITE: 1.0}

# BGR colour palette
COLOURS = {
    RED:   (50,  50,  200),
    CHUCK: (0,  215,  255),
    BOMB:  (40,  40,   40),
    BLUES: (220, 80,   50),
    WHITE: (240, 240, 240),
}

# Trail length (number of past positions stored)
_TRAIL_LEN = 12


class Bird:
    def __init__(self, kind: str, x: float, y: float):
        self.kind    = kind
        self.x       = float(x)
        self.y       = float(y)
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.radius  = RADII[kind]
        self.mass    = MASSES[kind]
        self.active  = True
        self.launched = False

        # ── New state ─────────────────────────────────────────────────────
        self.trail: list[tuple[float, float]] = []   # past positions for trail effect
        self.impact_timer: int = 0                    # countdown for impact pop ring
        self.grounded: bool   = False                 # True after first floor/block impact
        self.linger_timer: int = 0                    # frames until deactivation after grounding
        self._hit_blocks: set  = set()                # ids of blocks already hit (multi-hit)

    # ── Physics ───────────────────────────────────────────────────────────────

    def update(self):
        if not self.launched or not self.active:
            return

        # Gravity
        self.vy += GRAVITY

        # Air drag (subtle deceleration)
        self.vx *= AIR_DRAG
        self.vy *= AIR_DRAG

        # Move
        self.x += self.vx
        self.y += self.vy

        # Store trail position
        self.trail.append((self.x, self.y))
        if len(self.trail) > _TRAIL_LEN:
            self.trail.pop(0)

        # Floor collision / bounce
        if self.y + self.radius >= FLOOR_Y:
            self.y = FLOOR_Y - self.radius
            if abs(self.vy) > 1.5:
                self.vy = -self.vy * BIRD_BOUNCE
            else:
                self.vy = 0.0
            self.vx *= 0.85          # ground friction

            if not self.grounded:
                self.grounded = True
                self.linger_timer = BIRD_LINGER
                self.impact_timer = 15   # pop ring on first ground hit

        # Linger countdown (after grounding, bird rolls then deactivates)
        if self.grounded:
            self.linger_timer -= 1
            speed = magnitude(self.vx, self.vy)
            if self.linger_timer <= 0 or speed < 0.3:
                self.active = False

        # Impact animation countdown
        if self.impact_timer > 0:
            self.impact_timer -= 1

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, scale: float = 1.0):
        cx, cy = int(self.x), int(self.y)
        r = int(self.radius * scale)
        col = COLOURS[self.kind]

        # 1. Trail (behind bird)
        if self.launched and len(self.trail) > 1:
            self._draw_trail(frame)

        # 2. Bird body (existing artwork)
        if self.kind == RED:
            self._draw_red(frame, cx, cy, r, col)
        elif self.kind == CHUCK:
            self._draw_chuck(frame, cx, cy, r, col)
        elif self.kind == BOMB:
            self._draw_bomb(frame, cx, cy, r, col)
        elif self.kind == BLUES:
            self._draw_blues(frame, cx, cy, r, col)
        elif self.kind == WHITE:
            self._draw_white(frame, cx, cy, r, col)

        # 3. Speed lines (during fast flight only)
        if self.launched and not self.grounded and scale >= 1.0:
            self._draw_speed_lines(frame, cx, cy, r)

        # 4. Impact pop ring
        if self.impact_timer > 0 and scale >= 1.0:
            self._draw_impact_pop(frame, cx, cy, r)

    # ── New visual effects ────────────────────────────────────────────────────

    def _draw_trail(self, frame: np.ndarray):
        """Fading circles along past positions."""
        n = len(self.trail)
        col = COLOURS[self.kind]
        for i, (tx, ty) in enumerate(self.trail):
            alpha = (i + 1) / (n + 1)
            r = max(2, int(self.radius * 0.25 * alpha))
            faded = (
                max(0, min(255, int(col[0] * alpha * 0.5))),
                max(0, min(255, int(col[1] * alpha * 0.5))),
                max(0, min(255, int(col[2] * alpha * 0.5))),
            )
            cv2.circle(frame, (int(tx), int(ty)), r, faded, -1)

    def _draw_speed_lines(self, frame: np.ndarray, cx: int, cy: int, r: int):
        """Short lines behind the bird during fast flight."""
        speed = magnitude(self.vx, self.vy)
        if speed < 3.0:
            return
        angle = math.atan2(self.vy, self.vx)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        for i in range(3):
            offset = r + 5 + i * 7
            lx = cx - int(cos_a * offset)
            ly = cy - int(sin_a * offset)
            length = 8 + i * 4
            ex = lx - int(cos_a * length)
            ey = ly - int(sin_a * length)
            bright = max(0, int(180 * (1.0 - i * 0.3)))
            cv2.line(frame, (lx, ly), (ex, ey), (bright, bright, bright), 1)

    def _draw_impact_pop(self, frame: np.ndarray, cx: int, cy: int, r: int):
        """Expanding ring that fades out over 15 frames."""
        progress = 1.0 - self.impact_timer / 15.0
        pop_r = int(r + progress * 20)
        alpha_val = max(0, 1.0 - progress)
        col_base = COLOURS[self.kind]
        col = (
            min(255, int(col_base[0] * alpha_val) + 50),
            min(255, int(col_base[1] * alpha_val) + 50),
            min(255, int(col_base[2] * alpha_val) + 50),
        )
        thick = max(1, 3 - int(progress * 3))
        cv2.circle(frame, (cx, cy), pop_r, col, thick)

    # ── Individual drawers (unchanged artwork) ────────────────────────────────

    @staticmethod
    def _draw_red(f, cx, cy, r, col):
        # Body
        cv2.circle(f, (cx, cy), r, col, -1)
        cv2.circle(f, (cx, cy), r, (20, 20, 120), 2)
        # Angry brows
        cv2.line(f, (cx - r//2, cy - r//3), (cx - 2, cy - r//2 + 2), (20,20,20), 2)
        cv2.line(f, (cx + r//2, cy - r//3), (cx + 2, cy - r//2 + 2), (20,20,20), 2)
        # Eyes
        cv2.circle(f, (cx - r//4, cy - r//6), max(2, r//5), (255,255,255), -1)
        cv2.circle(f, (cx + r//4, cy - r//6), max(2, r//5), (255,255,255), -1)
        cv2.circle(f, (cx - r//4, cy - r//6), max(1, r//9), (0,0,0), -1)
        cv2.circle(f, (cx + r//4, cy - r//6), max(1, r//9), (0,0,0), -1)
        # Beak
        pts = np.array([[cx-r//5, cy+r//8], [cx+r//5, cy+r//8], [cx, cy+r//2]], np.int32)
        cv2.fillPoly(f, [pts], (0, 180, 255))

    @staticmethod
    def _draw_chuck(f, cx, cy, r, col):
        # Triangular-ish (ellipse tilted)
        cv2.ellipse(f, (cx, cy), (r, int(r*1.15)), -15, 0, 360, col, -1)
        cv2.ellipse(f, (cx, cy), (r, int(r*1.15)), -15, 0, 360, (0,160,200), 2)
        # Angry brows
        cv2.line(f, (cx-r//2, cy-r//3), (cx-2, cy-r//2), (20,20,20), 2)
        cv2.line(f, (cx+r//2, cy-r//3), (cx+2, cy-r//2), (20,20,20), 2)
        # Eyes
        cv2.circle(f, (cx-r//4, cy-r//8), max(2,r//5), (255,255,255), -1)
        cv2.circle(f, (cx+r//4, cy-r//8), max(2,r//5), (255,255,255), -1)
        cv2.circle(f, (cx-r//4, cy-r//8), max(1,r//9), (0,0,0), -1)
        cv2.circle(f, (cx+r//4, cy-r//8), max(1,r//9), (0,0,0), -1)
        # Beak
        pts = np.array([[cx-r//5, cy+r//6],[cx+r//5, cy+r//6],[cx, cy+r//2]], np.int32)
        cv2.fillPoly(f, [pts], (0,180,255))

    @staticmethod
    def _draw_bomb(f, cx, cy, r, col):
        cv2.circle(f, (cx, cy), r, col, -1)
        cv2.circle(f, (cx, cy), r, (10,10,10), 2)
        # Fuse
        cv2.line(f, (cx, cy-r), (cx+r//3, cy-r-r//2), (80,80,80), 2)
        cv2.circle(f, (cx+r//3, cy-r-r//2), 3, (0,120,255), -1)
        # Red angry eyes
        cv2.circle(f, (cx-r//4, cy-r//6), max(2,r//5), (0,0,200), -1)
        cv2.circle(f, (cx+r//4, cy-r//6), max(2,r//5), (0,0,200), -1)
        cv2.circle(f, (cx-r//4, cy-r//6), max(1,r//9), (0,0,0), -1)
        cv2.circle(f, (cx+r//4, cy-r//6), max(1,r//9), (0,0,0), -1)
        # Angry brows
        cv2.line(f, (cx-r//2, cy-r//3), (cx-2, cy-r//2+2), (180,180,180), 2)
        cv2.line(f, (cx+r//2, cy-r//3), (cx+2, cy-r//2+2), (180,180,180), 2)

    @staticmethod
    def _draw_blues(f, cx, cy, r, col):
        cv2.circle(f, (cx, cy), r, col, -1)
        cv2.circle(f, (cx, cy), r, (160,50,30), 2)
        # Wide innocent eyes
        cv2.circle(f, (cx-r//3, cy-r//5), max(3,r//3), (255,255,255), -1)
        cv2.circle(f, (cx+r//3, cy-r//5), max(3,r//3), (255,255,255), -1)
        cv2.circle(f, (cx-r//3, cy-r//5), max(1,r//7), (0,0,0), -1)
        cv2.circle(f, (cx+r//3, cy-r//5), max(1,r//7), (0,0,0), -1)
        # Small beak
        pts = np.array([[cx-r//6, cy+r//6],[cx+r//6, cy+r//6],[cx, cy+r//3]], np.int32)
        cv2.fillPoly(f, [pts], (0,160,220))

    @staticmethod
    def _draw_white(f, cx, cy, r, col):
        # Slightly oval (taller)
        cv2.ellipse(f, (cx, cy), (r, int(r*1.2)), 0, 0, 360, col, -1)
        cv2.ellipse(f, (cx, cy), (r, int(r*1.2)), 0, 0, 360, (180,180,180), 2)
        # Eyes
        cv2.circle(f, (cx-r//4, cy-r//5), max(2,r//5), (0,0,0), -1)
        cv2.circle(f, (cx+r//4, cy-r//5), max(2,r//5), (0,0,0), -1)
        # Beak
        pts = np.array([[cx-r//5, cy+r//6],[cx+r//5, cy+r//6],[cx, cy+r//2]], np.int32)
        cv2.fillPoly(f, [pts], (0,180,255))
