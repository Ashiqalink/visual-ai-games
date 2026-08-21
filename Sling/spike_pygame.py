"""
spike_pygame.py — Renderer-swap spike.

Proves one thing: ``visual_ai`` is renderer-agnostic. The vision engine runs
here completely unmodified — same ``VisionPipeline``, same bounded queue, same
payload dict — while every pixel is drawn by pygame instead of OpenCV.

There is no ``import cv2`` in this file. That is the point.

What it demonstrates, in the order it matters
---------------------------------------------
1. The queue boundary holds. ``VisionPipeline`` neither knows nor cares that
   OpenCV is gone from the consumer side.
2. Render rate decouples from vision rate. The HUD reports both: vision ticks
   at the pipeline's ~30 Hz while the window redraws at a vsynced 60.
3. Real sprites. The creature comes from ``visual_ai.render_creature`` as RGBA
   and is blitted with per-pixel alpha — no hand-rolled ROI blending.
4. Free transforms. The bird rotates and scales via ``rotozoom`` each frame,
   which in the OpenCV build costs a ``warpAffine`` or a pre-rotated cache.
5. Real typography. TTF text, not Hershey vector fonts.
6. The camera frame still crosses the boundary — the payload's BGR ndarray is
   converted to a pygame surface for the preview box.

Run it
------
    "D:/visual/visual ai games/Sling/.venv/Scripts/python.exe" spike_pygame.py

    Move your index finger  — the bird follows
    Pinch (thumb + index)   — the bird flashes and grows
    Q or ESC                — quit
"""

from __future__ import annotations

import os
import queue
import sys
import time
from collections import deque

import numpy as np
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

from engine_bootstrap import ensure_engine

ensure_engine()

from config import (  # noqa: E402
    BG_COLOR,
    BIRD_ORDER,
    BIRD_SPECS,
    CAM_H,
    CAM_MARGIN,
    CAM_W,
    FRAME_H,
    FRAME_W,
    SMOOTH_ALPHA,
)
from visual_ai import (  # noqa: E402
    CPP_ENGINE_AVAILABLE,
    VisionPipeline,
    render_creature,
)

# config.BG_COLOR is BGR, because everything downstream of it was an OpenCV
# canvas. pygame wants RGB, so this is the first of the small conversions a
# full port would push back into config.py.
BG_RGB = tuple(reversed(BG_COLOR))

ACCENT = (80, 200, 255)
ACCENT_DIM = (40, 100, 128)
PINCH_COL = (255, 190, 60)
TEXT_COL = (225, 235, 245)
TEXT_DIM = (120, 140, 160)


def bgr_to_surface(frame: np.ndarray) -> pygame.Surface:
    """
    Convert a BGR ndarray from the vision payload into a pygame surface.

    ``frombuffer`` needs tightly packed rows, and the reversed-stride view
    produced by the BGR->RGB flip is not contiguous, hence the explicit copy.
    """
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    height, width = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (width, height), "RGB")


def load_bird_sprite() -> pygame.Surface:
    """
    The game's own artwork, straight from the engine.

    ``render_creature`` returns RGBA in exactly the layout pygame's
    ``frombuffer`` wants, so the sprite crosses from engine to renderer with no
    conversion at all — which is the same point the rest of this file makes,
    made once more for artwork rather than tracking.
    """
    sprite = render_creature(BIRD_SPECS[BIRD_ORDER[0]], view="side", size=160)
    height, width = sprite.shape[:2]
    surface = pygame.image.frombuffer(
        np.ascontiguousarray(sprite).tobytes(), (width, height), "RGBA")
    return surface.convert_alpha()


class RateMeter:
    """Rolling events-per-second over a short window."""

    def __init__(self, window: float = 1.0):
        self.window = window
        self.stamps: deque[float] = deque()

    def tick(self, now: float) -> None:
        self.stamps.append(now)
        cutoff = now - self.window
        while self.stamps and self.stamps[0] < cutoff:
            self.stamps.popleft()

    @property
    def rate(self) -> float:
        return len(self.stamps) / self.window


def main() -> int:
    print(f"[spike] visual_ai engine: "
          f"{'C++ core' if CPP_ENGINE_AVAILABLE else 'Python fallback'}")
    print(f"[spike] pygame-ce {pygame.version.ver}")

    # ── Vision: byte-for-byte the same setup main.py uses ─────────────────
    ai_queue: queue.Queue = queue.Queue(maxsize=1)
    pipeline = VisionPipeline(
        result_queue=ai_queue,
        width=FRAME_W,
        height=FRAME_H,
        camera_index=0,
        smooth_alpha=SMOOTH_ALPHA,
    )
    pipeline.start()

    # ── Render ────────────────────────────────────────────────────────────
    pygame.init()
    pygame.display.set_caption("Renderer spike — visual_ai on pygame")
    try:
        screen = pygame.display.set_mode((FRAME_W, FRAME_H), vsync=1)
        vsync = "on"
    except pygame.error:
        # Some drivers refuse a vsynced window; the spike still proves its
        # point without it, so fall back rather than dying here.
        screen = pygame.display.set_mode((FRAME_W, FRAME_H))
        vsync = "unavailable"

    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("Segoe UI", 26, bold=True)
    font = pygame.font.SysFont("Segoe UI", 17)
    font_small = pygame.font.SysFont("Consolas", 14)

    bird = load_bird_sprite()

    cam_surface: pygame.Surface | None = None
    gesture: dict = {}
    vision_rate = RateMeter()
    render_rate = RateMeter()
    payloads_seen = 0
    angle = 0.0
    pinch_pulse = 0.0
    trail: deque[tuple[int, int]] = deque(maxlen=18)
    started = time.time()

    running = True
    while running:
        now = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        # ── Drain the queue exactly as main.py does ───────────────────────
        latest = None
        while True:
            try:
                latest = ai_queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            payloads_seen += 1
            vision_rate.tick(now)
            gesture = latest
            frame = latest.get("frame")
            if frame is not None:
                # Downscale before converting: the preview is 240x160, so
                # converting the full 1280x720 would waste most of the work.
                small = frame[::max(1, frame.shape[0] // CAM_H),
                              ::max(1, frame.shape[1] // CAM_W)]
                cam_surface = pygame.transform.smoothscale(
                    bgr_to_surface(small), (CAM_W, CAM_H))

        render_rate.tick(now)

        hand_visible = bool(gesture.get("hand_visible"))
        is_pinching = bool(gesture.get("is_pinching"))
        ix, iy = gesture.get("index_pos", (FRAME_W // 2, FRAME_H // 2))

        if hand_visible:
            trail.append((int(ix), int(iy)))
        elif trail:
            trail.popleft()

        # Rotation and pulse advance on the RENDER clock, not the vision
        # clock — which is the whole point of decoupling them. The bird
        # stays smooth at 60fps even though tracking arrives at 30Hz.
        angle = (angle + 90.0 * clock.get_time() / 1000.0) % 360.0
        if gesture.get("click_just_fired"):
            pinch_pulse = 1.0
        pinch_pulse = max(0.0, pinch_pulse - clock.get_time() / 400.0)

        # ── Draw ──────────────────────────────────────────────────────────
        screen.fill(BG_RGB)

        for i, (tx, ty) in enumerate(trail):
            weight = (i + 1) / len(trail)
            radius = int(3 + 9 * weight)
            dot = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(dot, (*ACCENT, int(90 * weight)),
                               (radius, radius), radius)
            screen.blit(dot, (tx - radius, ty - radius))

        if hand_visible:
            scale = 0.75 + (0.25 if is_pinching else 0.0) + 0.35 * pinch_pulse
            sprite = pygame.transform.rotozoom(bird, angle, scale)
            rect = sprite.get_rect(center=(int(ix), int(iy)))
            screen.blit(sprite, rect)

            ring = PINCH_COL if is_pinching else ACCENT
            pygame.draw.circle(screen, ring, (int(ix), int(iy)),
                               int(rect.width * 0.62), 3)
            if pinch_pulse > 0:
                pygame.draw.circle(screen, PINCH_COL, (int(ix), int(iy)),
                                   int(rect.width * 0.62 + 40 * (1 - pinch_pulse)),
                                   max(1, int(4 * pinch_pulse)))
        else:
            prompt = font_big.render("show a hand to the camera", True, TEXT_DIM)
            screen.blit(prompt, prompt.get_rect(center=(FRAME_W // 2, FRAME_H // 2)))

        # ── HUD: the numbers this spike exists to produce ─────────────────
        panel = pygame.Surface((330, 150), pygame.SRCALPHA)
        panel.fill((8, 12, 20, 205))
        pygame.draw.rect(panel, ACCENT_DIM, panel.get_rect(), 1)
        screen.blit(panel, (16, 16))

        v_rate = vision_rate.rate
        r_rate = render_rate.rate
        lines = [
            (f"render   {r_rate:5.1f} fps", TEXT_COL),
            (f"vision   {v_rate:5.1f} Hz", TEXT_COL),
            (f"ratio    {r_rate / v_rate:4.2f}x decoupled" if v_rate else
             "ratio    waiting", ACCENT),
            (f"payloads {payloads_seen}", TEXT_DIM),
            (f"vsync    {vsync}", TEXT_DIM),
            (f"engine   {'C++ core' if CPP_ENGINE_AVAILABLE else 'py fallback'}",
             TEXT_DIM),
        ]
        for i, (text, colour) in enumerate(lines):
            screen.blit(font_small.render(text, True, colour), (30, 28 + i * 20))

        title = font.render("visual_ai  ->  pygame   (zero cv2 in this file)",
                            True, ACCENT)
        screen.blit(title, (16, 180))

        # What to do with your hands. The spike has no goal to explain, but it
        # still opens on a window that does nothing until a hand appears.
        screen.blit(font_small.render(
            "move your index finger — the bird follows   ·   "
            "pinch thumb + index — it flashes and grows   ·   Q quits",
            True, TEXT_DIM), (16, 204))

        state = "PINCH" if is_pinching else ("TRACKING" if hand_visible else "NO HAND")
        state_col = PINCH_COL if is_pinching else (
            ACCENT if hand_visible else TEXT_DIM)
        state_surf = font_big.render(state, True, state_col)
        screen.blit(state_surf, (16, FRAME_H - 52))

        sign = gesture.get("hand_sign", "unknown")
        screen.blit(font_small.render(f"sign: {sign}   z: "
                                      f"{gesture.get('z_delta', 0.0):+.3f}",
                                      True, TEXT_DIM), (16, FRAME_H - 20))

        if cam_surface is not None:
            cx = FRAME_W - CAM_W - CAM_MARGIN
            cy = CAM_MARGIN
            screen.blit(cam_surface, (cx, cy))
            pygame.draw.rect(screen, ACCENT, (cx - 2, cy - 2, CAM_W + 4, CAM_H + 4), 2)
            screen.blit(font_small.render("CAMERA", True, ACCENT_DIM),
                        (cx, cy + CAM_H + 6))

        pygame.display.flip()
        clock.tick(60)

    pipeline.stop()
    pygame.quit()

    elapsed = time.time() - started
    print(f"\n[spike] ran {elapsed:.1f}s — {payloads_seen} vision payloads "
          f"({payloads_seen / elapsed:.1f} Hz avg)")
    print("[spike] VisionPipeline was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
