"""
bird_pygame.py — the birds, drawn by pygame.

This is the renderer half of ``bird.py`` ported off OpenCV. Everything the
OpenCV path draws is here: ground shadow, flight trail, the rotated creature
sprite, speed lines and the impact pop ring. Nothing else moved — physics still
lives in ``Bird.update()``, and this module never touches it.

There is no ``import cv2`` at module scope, and no ``import bird`` either: the
draw functions take any object with the fields a bird has (``kind``, ``x``,
``y``, ``vx``, ``vy``, ``radius``, ``trail``, ``rot_z``, ``launched``,
``grounded``, ``impact_timer``, ``on_slingshot``), so a pygame front-end can
pull in this file without dragging OpenCV — and the engine's sprites — along
with it. The demo at the bottom imports ``Bird`` inside ``main()`` for exactly
that reason.

Two things differ from the OpenCV version by necessity:

  * Colours. ``config.BIRD_COLOURS`` is BGR because it was authored against
    OpenCV frames; ``rgb()`` flips it on the way in.
  * Sprites. ``render_creature`` already returns RGBA in pygame's byte order,
    so the ``rgb_to_bgr`` swap ``bird.py`` performs is simply not needed here.
    Rotation and scaling are one ``rotozoom`` call instead of a ``warpAffine``.

Run it
------
    "D:/visual/visual ai games/Sling/.venv/Scripts/python.exe" bird_pygame.py

    SPACE — launch another bird      R — clear the field
    Q / ESC — quit
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

from engine_bootstrap import ensure_engine  # noqa: E402

ensure_engine()

from config import (  # noqa: E402
    BIRD_ORDER, BIRD_SPECS,
    BIRD_COLOURS as COLOURS,
    IMPACT_POP_DURATION, SPEED_LINE_THRESHOLD,
    LIGHTING_SETTINGS,
)
from physics import FLOOR_Y  # noqa: E402
from visual_ai import render_creature  # noqa: E402

#: Rasterisation size, matching ``bird.SPRITE_SIZE`` — sprites are scaled down
#: per-bird at draw time, so this only has to exceed the largest on-screen size.
SPRITE_SIZE = 192

ASSET_DIR = os.path.join(BASE_DIR, "assets")

_sprites: dict[tuple[str, str], pygame.Surface] = {}


def rgb(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """config.py stores colours BGR, for OpenCV. pygame wants them the other way."""
    return (bgr[2], bgr[1], bgr[0])


# ── Sprites ───────────────────────────────────────────────────────────────────

def _surface_from_rgba(array: np.ndarray) -> pygame.Surface:
    """RGBA ndarray -> surface. ``frombuffer`` needs tightly packed rows."""
    height, width = array.shape[:2]
    surface = pygame.image.frombuffer(
        np.ascontiguousarray(array).tobytes(), (width, height), "RGBA")
    # convert_alpha needs a display; without one the unconverted surface still
    # blits correctly, just slower — which is fine for offscreen use and tests.
    return surface.convert_alpha() if pygame.display.get_init() else surface


def sprite(kind: str, view: str = "front") -> pygame.Surface:
    """
    One creature sprite as a pygame surface, from disk if authored, from the
    engine if not — the same precedence ``bird._sprite`` uses, so both renderers
    show the same artwork.
    """
    key = (kind, view)
    cached = _sprites.get(key)
    if cached is not None:
        return cached

    suffix = "" if view == "front" else "_side"
    path = os.path.join(ASSET_DIR, f"{kind}{suffix}.png")
    if os.path.isfile(path):
        surface = pygame.image.load(path)
        if pygame.display.get_init():
            surface = surface.convert_alpha()
    else:
        surface = _surface_from_rgba(
            render_creature(BIRD_SPECS[kind], view=view, size=SPRITE_SIZE))

    _sprites[key] = surface
    return surface


def preload_sprites() -> None:
    """Rasterise every bird up front, so the first launch does not stutter."""
    for kind in BIRD_ORDER:
        sprite(kind, "front")
        sprite(kind, "side")


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_bird(screen: pygame.Surface, bird, scale: float = 1.0) -> None:
    """
    Draw one bird, effects and all. Layered exactly as ``Bird.draw`` layers it:
    shadow, trail, body, speed lines, impact ring.
    """
    cx, cy = int(bird.x), int(bird.y)
    r = max(1, int(bird.radius * scale))

    if LIGHTING_SETTINGS.get("SHADOWS_ENABLED", True):
        _draw_shadow(screen, bird, r)

    if bird.launched and len(bird.trail) > 1:
        _draw_trail(screen, bird)

    view = "side" if getattr(bird, "on_slingshot", False) else "front"
    _draw_body(screen, bird, cx, cy, r, view)

    if bird.launched and not bird.grounded and scale >= 1.0:
        _draw_speed_lines(screen, bird, cx, cy, r)

    if bird.impact_timer > 0 and scale >= 1.0:
        _draw_impact_pop(screen, bird, cx, cy, r)


def _draw_body(screen: pygame.Surface, bird, cx: int, cy: int,
               r: int, view: str) -> None:
    """
    The creature itself: one ``rotozoom``, one blit.

    ``bird.py`` feeds ``blit_sprite`` a pixel size of ``r * 2.2`` and an angle
    of ``-degrees(rot_z * 0.1)``; both are reproduced here so a bird rolling
    along the floor looks the same under either renderer.
    """
    base = sprite(bird.kind, view)
    target = r * 2.2
    zoom = target / base.get_width()
    angle = -math.degrees(bird.rot_z * 0.1)
    # rotozoom rotates counter-clockwise, OpenCV's rotation matrix clockwise —
    # hence the extra negation on top of bird.py's own.
    surface = pygame.transform.rotozoom(base, -angle, zoom)
    screen.blit(surface, surface.get_rect(center=(cx, cy)))


def _draw_shadow(screen: pygame.Surface, bird, r: int) -> None:
    """Translucent oval on the floor, shrinking and fading with altitude."""
    height = int(FLOOR_Y - bird.y)
    if not 0 < height < 160:
        return

    fade = height / 160.0
    base_opacity = LIGHTING_SETTINGS.get("SHADOW_OPACITY", 0.45)
    alpha = int(255 * max(0.08, base_opacity * (1.0 - fade)))

    angle_rad = math.radians(LIGHTING_SETTINGS.get("LIGHT_ANGLE", 45.0))
    shift_x = int(math.cos(angle_rad)
                  * LIGHTING_SETTINGS.get("SHADOW_OFFSET_X", 15.0)
                  * (1.0 + height / 120.0))

    w = max(2, int(r * LIGHTING_SETTINGS.get("SHADOW_SCALE_X", 1.25)
                   * (1.2 - 0.3 * fade)))
    h = max(2, int(r * LIGHTING_SETTINGS.get("SHADOW_SCALE_Y", 0.35)
                   * (1.0 - 0.4 * fade)))

    colour = rgb(LIGHTING_SETTINGS.get("SHADOW_COLOR", (10, 15, 20)))
    layer = pygame.Surface((w * 2, h * 2), pygame.SRCALPHA)
    pygame.draw.ellipse(layer, (*colour, alpha), layer.get_rect())
    screen.blit(layer, layer.get_rect(
        center=(int(bird.x) + shift_x, int(FLOOR_Y - 4))))


def _draw_trail(screen: pygame.Surface, bird) -> None:
    """Fading dots along past positions. Alpha here, colour scaling there —
    OpenCV had no alpha channel to work with, so it dimmed towards black."""
    trail = bird.trail
    n = len(trail)
    colour = rgb(COLOURS[bird.kind])
    for i, (tx, ty) in enumerate(trail):
        weight = (i + 1) / (n + 1)
        r = max(2, int(bird.radius * 0.25 * weight))
        dot = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(dot, (*colour, int(200 * weight)), (r, r), r)
        screen.blit(dot, (int(tx) - r, int(ty) - r))


def _draw_speed_lines(screen: pygame.Surface, bird, cx: int, cy: int,
                      r: int) -> None:
    """Three short streaks trailing the velocity vector during fast flight."""
    speed = math.hypot(bird.vx, bird.vy)
    if speed < SPEED_LINE_THRESHOLD:
        return
    dx, dy = bird.vx / speed, bird.vy / speed
    for i in range(3):
        offset = r + 5 + i * 7
        sx, sy = cx - dx * offset, cy - dy * offset
        length = 8 + i * 4
        ex, ey = sx - dx * length, sy - dy * length
        bright = max(0, int(180 * (1.0 - i * 0.3)))
        pygame.draw.line(screen, (bright, bright, bright),
                         (int(sx), int(sy)), (int(ex), int(ey)))


def _draw_impact_pop(screen: pygame.Surface, bird, cx: int, cy: int,
                     r: int) -> None:
    """Ring that expands and fades over IMPACT_POP_DURATION frames."""
    progress = 1.0 - bird.impact_timer / IMPACT_POP_DURATION
    pop_r = int(r + progress * 20)
    remaining = max(0.0, 1.0 - progress)
    base = rgb(COLOURS[bird.kind])
    colour = tuple(min(255, int(c * remaining) + 50) for c in base)
    thickness = max(1, 3 - int(progress * 3))
    pygame.draw.circle(screen, colour, (cx, cy), pop_r, thickness)


# ── Demo ──────────────────────────────────────────────────────────────────────
#
# A camera-free harness: real Bird physics, pygame pixels. Useful for eyeballing
# the port against the OpenCV build without standing in front of a webcam.

def main() -> int:
    # Imported here, not at module scope: bird.py pulls in cv2 and rasterises
    # every sprite at import time. The renderer above needs neither.
    from bird import Bird
    from config import FRAME_W, FRAME_H, BG_COLOR
    import random

    pygame.init()
    pygame.display.set_caption("Birds — pygame renderer")
    screen = pygame.display.set_mode((FRAME_W, FRAME_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 15)
    preload_sprites()

    birds: list = []

    def launch() -> None:
        kind = random.choice(BIRD_ORDER)
        b = Bird(kind, 140, FLOOR_Y - 220)
        b.launched = True
        b.vx = random.uniform(9.0, 15.0)
        b.vy = random.uniform(-11.0, -6.0)
        birds.append(b)

    launch()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    launch()
                elif event.key == pygame.K_r:
                    birds.clear()

        for b in birds:
            b.update()
            # The game gives impact_timer its value on collision; there are no
            # blocks here, so the floor bounce stands in for one.
            if b.grounded and b.impact_timer == 0 and b.vy < -1.0:
                b.impact_timer = IMPACT_POP_DURATION
        birds[:] = [b for b in birds if b.active and b.x < FRAME_W + 200]

        screen.fill(rgb(BG_COLOR))
        pygame.draw.line(screen, (46, 58, 74),
                         (0, FLOOR_Y), (FRAME_W, FLOOR_Y), 2)
        for b in birds:
            draw_bird(screen, b)

        hud = (f"{len(birds)} airborne   {clock.get_fps():4.1f} fps   "
               f"SPACE launch   R clear   Q quit")
        screen.blit(font.render(hud, True, (150, 170, 190)), (16, 16))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
