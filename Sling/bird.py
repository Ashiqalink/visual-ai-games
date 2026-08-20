"""
bird.py — Bird: physics, flight effects, and sprite compositing.

Artwork is no longer drawn here. Each creature is a `CreatureSpec` in config.py
which the engine rasterises, or a PNG in assets/ when one has been authored.
See the sprite section at the bottom of this file.

  - Per-type mass values
  - Flight trail (fading circles behind the bird)
  - Impact pop ring on collision
  - Speed lines during fast flight
  - Floor bounce + linger timer (bird rolls after landing)
  - Air drag during flight
"""

import cv2
import numpy as np
import hashlib
import math
import collections
import os
from physics import GRAVITY, FLOOR_Y, AIR_DRAG, BIRD_BOUNCE, BIRD_LINGER, magnitude, bird_hits_ground
from config import (
    TRAIL_LEN, SPEED_LINE_THRESHOLD, IMPACT_POP_DURATION, BIRD_IDLE_SPEED,
    BIRD_ORDER, BIRD_SPECS,
    BIRD_RADII as RADII, BIRD_MASSES as MASSES, BIRD_COLOURS as COLOURS,
    LIGHTING_SETTINGS
)
from visual_ai import Vector2, render_creature, rgb_to_bgr, load_image
from visual_ai.imaging import blit_sprite, blit_ellipse_alpha

# Trail length (number of past positions stored)
_TRAIL_LEN = TRAIL_LEN

_bird_images_2d = {}
_bird_images_side_2d = {}


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
        self.on_slingshot = False

        # ── New state ─────────────────────────────────────────────────────
        self.trail = collections.deque(maxlen=_TRAIL_LEN) # O(1) past positions for trail effect
        self.impact_timer: int = 0                    # countdown for impact pop ring
        self.grounded: bool   = False                 # True after first floor/block impact
        self.linger_timer: int = 0                    # frames until deactivation after grounding
        self._hit_blocks: set  = set()                # ids of blocks already hit (multi-hit)
        self.rot_z: float = 0.0                       # 3D rotational angle

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

        # Roll rotation angle update
        self.rot_z += self.vx * 2.0

        # Store trail position
        self.trail.append((self.x, self.y))

        # Floor collision
        if bird_hits_ground(self):
            self.y = FLOOR_Y - self.radius
            self.vy = -self.vy * BIRD_BOUNCE
            self.vx *= 0.8
            if not self.grounded:
                self.grounded = True
                self.linger_timer = BIRD_LINGER

        # Linger countdown (after grounding, bird rolls then deactivates)
        if self.grounded:
            self.linger_timer -= 1
            speed = magnitude(self.vx, self.vy)
            if self.linger_timer <= 0 or speed < BIRD_IDLE_SPEED:
                self.active = False

        # Impact animation countdown
        if self.impact_timer > 0:
            self.impact_timer -= 1

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, scale: float = 1.0):
        cx, cy = int(self.x), int(self.y)
        r = int(self.radius * scale)
        col = COLOURS[self.kind]

        # 0. Ground Drop Shadow under bird
        from ui import draw_ground_shadow  # local: ui imports bird, avoid circular import
        shadow_scale_x = LIGHTING_SETTINGS.get("SHADOW_SCALE_X", 1.25)
        shadow_scale_y = LIGHTING_SETTINGS.get("SHADOW_SCALE_Y", 0.35)
        shadow_w = int(r * shadow_scale_x * (1.2 - 0.3 * max(0, min(1, (FLOOR_Y - self.y) / 160.0))))
        shadow_h = int(r * shadow_scale_y * (1.0 - 0.4 * max(0, min(1, (FLOOR_Y - self.y) / 160.0))))
        draw_ground_shadow(frame, cx, int(self.y), shadow_w, shadow_h)

        # 1. Trail (behind bird)
        if self.launched and len(self.trail) > 1:
            self._draw_trail(frame)

        # 3. Bird Base Body (2D Image or Fallback Silhouette)
        img = _bird_images_side_2d.get(self.kind) if self.on_slingshot else _bird_images_2d.get(self.kind)
        if img is None:
            img = _bird_images_2d.get(self.kind)
        if img is not None and img.shape[2] == 4:
            # Scale, rotate, clip and alpha-blend — all of it in the engine, so
            # this and block.py share one implementation instead of two.
            blit_sprite(frame, img, cx, cy, size=int(r * 2.2),
                        angle=-math.degrees(self.rot_z * 0.1))
        else:
            # Last-resort silhouette. The sprite path above cannot normally
            # fail — the engine renders a spec when no PNG exists — so this
            # only shows up if a kind is missing from BIRD_SPECS entirely.
            cv2.circle(frame, (cx, cy), r, col, -1)
            cv2.circle(frame, (cx, cy), r, (20, 20, 20), max(1, r // 12))
            Bird._draw_finish(frame, cx, cy, r, col)

        # 3. Speed lines (during fast flight only)
        if self.launched and not self.grounded and scale >= 1.0:
            self._draw_speed_lines(frame, cx, cy, r)

        # 4. Impact pop ring
        if self.impact_timer > 0 and scale >= 1.0:
            self._draw_impact_pop(frame, cx, cy, r)

    @staticmethod
    def _draw_finish(frame: np.ndarray, cx: int, cy: int, r: int, col):
        """Small illustration passes shared by all birds: wing, rim and shine."""
        if r < 5:
            return
        # A tucked wing adds depth without covering the face.
        wing_col = tuple(max(0, int(c * 0.68)) for c in col)
        wing = (cx - int(r * .47), cy + int(r * .18))
        cv2.ellipse(frame, wing, (max(3, int(r*.35)), max(3, int(r*.22))), 28, 10, 190, wing_col, -1, cv2.LINE_AA)
        cv2.ellipse(frame, wing, (max(3, int(r*.35)), max(3, int(r*.22))), 28, 10, 190, (20, 20, 20), 1, cv2.LINE_AA)
        # A restrained upper-left sheen makes the simple OpenCV artwork feel
        # round while preserving the crisp, game-like style.
        shine = tuple(min(255, int(c + (255-c)*.35)) for c in col)
        cv2.ellipse(frame, (cx - int(r*.22), cy - int(r*.35)),
                    (max(2, int(r*.25)), max(1, int(r*.10))), -28, 185, 340,
                    shine, max(1, r//12), cv2.LINE_AA)

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
        if speed < SPEED_LINE_THRESHOLD:
            return
        vel_dir = Vector2(self.vx, self.vy).normalize()
        cos_a, sin_a = vel_dir.x, vel_dir.y
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
        progress = 1.0 - self.impact_timer / IMPACT_POP_DURATION
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


# ── Sprite artwork ────────────────────────────────────────────────────────────
#
# This used to be five hundred lines of per-character OpenCV and Pillow drawing
# code — a `_draw_red`, a `_draw_chuck`, a supersampled Pillow variant of each
# for the side view, and a pile of shared helpers underneath them. All of it is
# gone. A creature is now a `CreatureSpec` declared in config.py, and the engine
# rasterises it. Adding a bird is a spec; it is not a new drawing function.
#
# Pre-rendered PNGs in assets/ still take priority when they exist, so artwork
# authored with `spritesmith` — or drawn by hand, or cut out of a generated
# image — drops in without touching any code.

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

#: Rasterisation size. Sprites are scaled down per-bird at draw time, so this
#: only needs to exceed the largest on-screen radius; 192 leaves headroom for
#: the carousel, which draws them larger than gameplay does.
SPRITE_SIZE = 192


#: Rasterised sprites live here between runs. `render_creature` costs ~37 ms per
#: view, and ten of them (five kinds x two views) put ~0.4 s of pure CPU in front
#: of the first frame every single launch. The output is deterministic, so it is
#: cached to disk and only re-rendered when the inputs change.
SPRITE_CACHE_DIR = os.path.join(ASSET_DIR, ".sprite-cache")


def _sprite_cache_key(kind: str, view: str) -> str:
    """
    Cache filename for one creature view, keyed by everything that changes it.

    The spec's repr plus the raster size go through a hash, so editing a
    creature in config.py — or bumping SPRITE_SIZE — lands on a different file
    and the stale sprite is simply never read again. No manual cache-busting.
    """
    payload = f"{kind}|{view}|{SPRITE_SIZE}|{BIRD_SPECS[kind]!r}".encode("utf-8")
    return f"{kind}_{view}_{hashlib.sha1(payload).hexdigest()[:12]}.png"


def _sprite(kind: str, view: str) -> np.ndarray:
    """
    One creature sprite as BGRA, from disk if authored, from the engine if not.

    The engine works in RGB like most of the world; this file composites onto
    OpenCV's BGR frames, hence the swap on the way out.
    """
    suffix = "" if view == "front" else "_side"
    path = os.path.join(ASSET_DIR, f"{kind}{suffix}.png")
    if os.path.isfile(path):
        # load_image normalises to RGBA, so a PNG saved without an alpha
        # channel still arrives with one and the blit path stays uniform.
        return rgb_to_bgr(load_image(path))

    # No authored art — fall back to the engine's rasteriser, via the cache.
    cached = os.path.join(SPRITE_CACHE_DIR, _sprite_cache_key(kind, view))
    if os.path.isfile(cached):
        try:
            img = cv2.imread(cached, cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                return img
        except Exception:
            pass          # unreadable/half-written cache entry — re-render below

    sprite = rgb_to_bgr(render_creature(BIRD_SPECS[kind], view=view, size=SPRITE_SIZE))
    try:
        os.makedirs(SPRITE_CACHE_DIR, exist_ok=True)
        # Write to a temp name and rename, so a crash mid-write cannot leave a
        # truncated PNG that the next launch would load as the bird. The temp
        # name keeps the .png extension — imwrite picks its encoder from the
        # extension and silently writes nothing for an unknown one.
        tmp = cached + ".tmp.png"
        if cv2.imwrite(tmp, sprite):
            os.replace(tmp, cached)
    except Exception:
        pass              # read-only checkout: slower start, still correct
    return sprite


def _load_bird_images():
    if _bird_images_2d:
        return
    for kind in BIRD_ORDER:
        _bird_images_2d[kind] = _sprite(kind, "front")
        _bird_images_side_2d[kind] = _sprite(kind, "side")


_load_bird_images()
