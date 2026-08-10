"""
block.py — Block with material system (wood/stone/ice), health, gravity,
           floor bounce, angular rotation, crack overlays, conditional health bar.
"""

import cv2
import numpy as np
import math
import random
from physics import GRAVITY, FLOOR_Y, RESTITUTION, DAMAGE_FACTOR, BLOCK_HEALTH
from config import BLOCK_VX_TRANSFER, BLOCK_VY_TRANSFER, DEBRIS_FADE_FRAMES

# ── Material definitions ──────────────────────────────────────────────────────
# BGR colour values; health/density per material
MATERIALS = {
    "wood": {
        "health":  50,
        "density": 1.0,
        "color":   (34, 100, 172),     # warm brown
        "dark":    (20,  65, 120),
        "grain":   (55, 130, 195),
    },
    "stone": {
        "health":  100,
        "density": 1.8,
        "color":   (130, 130, 135),    # neutral grey
        "dark":    (90,   90,  95),
        "grain":   (155, 155, 160),
    },
    "ice": {
        "health":  25,
        "density": 0.6,
        "color":   (240, 220, 180),    # light icy cyan
        "dark":    (200, 180, 140),
        "grain":   (250, 235, 200),
    },
}


class Block:
    def __init__(self, x: int, y: int, w: int = 60, h: int = 60,
                 material: str = "wood"):
        # rect = top-left corner
        self.rect = [float(x), float(y), float(w), float(h)]
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.angle: float = 0.0          # degrees, for visual tilt
        self.angular_vel: float = 0.0
        self.active: bool = True
        self.on_ground: bool = False
        self.is_static: bool = False
        
        # Debris lifespan (ported from visual_ai)
        self.is_debris: bool = False
        self.lifespan: int = -1

        # Material system
        self.material = material
        mat = MATERIALS.get(material, MATERIALS["wood"])
        self.health: float     = mat["health"]
        self.max_health: float = mat["health"]
        self.density: float    = mat["density"]

    # ── Convenience ──────────────────────────────────────────────────────────
    @property
    def cx(self): return self.rect[0] + self.rect[2] / 2
    @property
    def cy(self): return self.rect[1] + self.rect[3] / 2
    @property
    def left(self):   return self.rect[0]
    @property
    def right(self):  return self.rect[0] + self.rect[2]
    @property
    def top(self):    return self.rect[1]
    @property
    def bottom(self): return self.rect[1] + self.rect[3]

    # ── Physics update ───────────────────────────────────────────────────────
    def update(self):
        if not self.active:
            return
        if self.health <= 0:
            self.active = False
            return
        if self.is_static:
            return

        # Debris lifespan logic
        if self.is_debris and self.lifespan > 0:
            self.lifespan -= 1
            if self.lifespan <= 0:
                self.active = False
                return

        self.vy += GRAVITY
        self.rect[0] += self.vx
        self.rect[1] += self.vy
        self.angle    += self.angular_vel
        self.angular_vel *= 0.90
        if abs(self.angular_vel) < 0.01:
            self.angular_vel = 0.0

        # Floor collision
        # Removed since floor is disabled in testing

    def apply_impulse(self, bird_vx: float, bird_vy: float, bird_mass: float):
        """Bird hit → damage + physical push."""
        speed  = math.sqrt(bird_vx**2 + bird_vy**2)
        force  = bird_mass * speed
        self.health -= force * DAMAGE_FACTOR
        if not self.is_static:
            self.vx     += bird_vx * BLOCK_VX_TRANSFER
            self.vy     += bird_vy * BLOCK_VY_TRANSFER
            self.angular_vel += random.uniform(-2.5, 2.5) * max(0.5, speed / 10.0)
            self.on_ground = False

    # ── Drawing ──────────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray):
        if not self.active:
            return

        x, y, w, h = [int(v) for v in self.rect]
        cx, cy = x + w // 2, y + h // 2

        # Build a rotated rectangle patch and paste it
        angle_rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        # Corner offsets from centre
        corners_local = [
            (-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)
        ]
        corners = np.array([
            [cx + dx*cos_a - dy*sin_a, cy + dx*sin_a + dy*cos_a]
            for dx, dy in corners_local
        ], dtype=np.int32)

        # Material colours
        mat = MATERIALS.get(self.material, MATERIALS["wood"])
        fill_col  = mat["color"]
        dark_col  = mat["dark"]
        grain_col = mat["grain"]

        # Fade out debris before it despawns
        if self.is_debris and self.lifespan > 0 and self.lifespan < DEBRIS_FADE_FRAMES:
            alpha = self.lifespan / 30.0
            # Blend with background (18, 18, 28) for simple fade effect
            bg = (18, 18, 28)
            fill_col = tuple(int(c * alpha + b * (1 - alpha)) for c, b in zip(fill_col, bg))
            dark_col = tuple(int(c * alpha + b * (1 - alpha)) for c, b in zip(dark_col, bg))
            grain_col = tuple(int(c * alpha + b * (1 - alpha)) for c, b in zip(grain_col, bg))

        # Fill
        cv2.fillPoly(frame, [corners], fill_col)

        # Grain lines (horizontal in local space, rotated)
        for gy in [-h//4, 0, h//4]:
            p1 = np.array([cx + (-w//2)*cos_a - gy*sin_a,
                           cy + (-w//2)*sin_a + gy*cos_a], dtype=np.int32)
            p2 = np.array([cx + (w//2)*cos_a  - gy*sin_a,
                           cy + (w//2)*sin_a  + gy*cos_a], dtype=np.int32)
            cv2.line(frame, tuple(p1), tuple(p2), grain_col, 1)

        # Border
        cv2.polylines(frame, [corners], True, dark_col, 2)

        # Crack overlays (drawn on top of the block face)
        self._draw_cracks(frame, cx, cy, w, h, cos_a, sin_a)

        # Health bar — only shown when damaged
        if self.health < self.max_health:
            self._draw_health_bar(frame, x, y, w)

    # ── Crack overlays ───────────────────────────────────────────────────────

    def _draw_cracks(self, frame, cx, cy, w, h, cos_a, sin_a):
        """Draw deterministic crack lines when health drops below thresholds."""
        hp_ratio = self.health / self.max_health
        if hp_ratio >= 0.7:
            return

        # Crack colour depends on material
        if self.material == "ice":
            crack_col = (180, 160, 120)
        elif self.material == "stone":
            crack_col = (60, 60, 60)
        else:
            crack_col = (15, 50, 90)

        def _rot(lx, ly):
            """Rotate a local offset into world space."""
            return (int(cx + lx * cos_a - ly * sin_a),
                    int(cy + lx * sin_a + ly * cos_a))

        # Minor cracks (health < 70%)
        cv2.line(frame, _rot(-w//4, -h//4), _rot(w//6,  h//3),  crack_col, 1)
        cv2.line(frame, _rot( w//5, -h//3), _rot(-w//6, h//4),  crack_col, 1)

        if hp_ratio < 0.4:
            # Major cracks (health < 40%)
            cv2.line(frame, _rot(-w//3, 0),      _rot(w//3, -h//5), crack_col, 2)
            cv2.line(frame, _rot(0, -h//3),      _rot(w//5,  h//3), crack_col, 2)
            cv2.line(frame, _rot(-w//5,  h//4),  _rot(w//4, -h//6), crack_col, 1)

    # ── Health bar ───────────────────────────────────────────────────────────

    def _draw_health_bar(self, frame, x, y, w):
        """Small bar above the block, only visible when damaged."""
        bar_w = w
        bar_h = 6
        bx = x
        by = y - 10
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 40), -1)
        hp_ratio = max(0.0, self.health / self.max_health)
        hp_col = (0, int(200 * hp_ratio), int(200 * (1 - hp_ratio)))
        cv2.rectangle(frame, (bx, by),
                      (bx + int(bar_w * hp_ratio), by + bar_h), hp_col, -1)
