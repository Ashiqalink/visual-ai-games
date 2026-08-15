"""
depth_pong.py — your whole silhouette is the paddle. Keep the ball alive.

What this is testing
--------------------
Every other consumer of ToF in this repo samples exactly one pixel patch, at the
fingertip. The sensor produces a whole frame and the pipeline throws all of it
away. This game uses the occupancy grid instead: any cell nearer than a
threshold is solid, and the ball bounces off it — so the input is a shape, not a
point.

That makes it the throughput test. `depth_grid` is the only payload key whose
size is not O(1), and every game in this repo drains a `maxsize=1` queue, so the
question worth answering is what carrying it costs. `--grid` resizes it, and the
report includes the measured payload size and the number of frames the consumer
dropped — a game consuming slower than the pipeline produces shows up there and
in no fps counter.

Without ToF hardware the grid is synthesised by the engine from the tracked hands
and face. The gameplay is therefore real but the depths are not measured; the
throughput numbers are real either way, which is the point of the exercise.

Scope: game side, on the engine's new opt-in `depth_grid`.

    python depth_pong.py                            play it
    python depth_pong.py --headless 900             scripted run
    python depth_pong.py --headless 900 --grid 64x48
"""

from __future__ import annotations

import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import cv2
import numpy as np

import labkit
from labkit import LabGame, draw_panel, draw_text

#: Cells nearer than this are solid.
SOLID_M = 1.2
BALL_SPEED = 460.0
BALL_RADIUS = 13


class DepthPong(LabGame):
    title = "Depth Pong — silhouette as paddle"
    width, height = 1000, 750
    wants_depth_grid = True
    wants_max_hands = 2

    goal = ("Your whole silhouette is the paddle. Keep the ball off the bottom "
            "of the screen — every bounce off you is a rally.")
    controls = (
        ("Lean toward the camera",
         "anything nearer than the depth threshold turns green and becomes "
         "solid — that green shape is your paddle"),
        ("Move under the ball",
         "it bounces off the slope of the shape it hits, so an angled hand "
         "sends it sideways and a flat one sends it straight up"),
        ("Use both hands",
         "two hands give you a wider paddle than one"),
        ("Let it fall off the bottom",
         "counts as a drop and the ball respawns at the top"),
    )

    def add_args(self, parser):
        parser.add_argument("--grid", metavar="WxH", default=None,
                            help="depth grid resolution, e.g. 16x12, 32x24, 64x48, 160x120")
        parser.add_argument("--solid", type=float, default=SOLID_M,
                            help="cells nearer than this many metres are solid")

    def configure(self):
        spec = getattr(self.args, "grid", None)
        if spec:
            try:
                cols, rows = (int(v) for v in spec.lower().split("x"))
            except ValueError:
                raise SystemExit(f"--grid wants WxH, got {spec!r}")
            self.grid_size = (cols, rows)
        self.solid_m = float(getattr(self.args, "solid", SOLID_M) or SOLID_M)

    def tune_pipeline(self, pipeline):
        pipeline.emit_depth_grid = True
        if self.grid_size is not None:
            # ScriptedPipeline forwards to the template source; VisionPipeline
            # carries the attribute itself.
            target = getattr(pipeline, "_template_source", pipeline)
            target.depth_grid_size = self.grid_size

    def __init__(self):
        self.grid_size = None
        self.solid_m = SOLID_M
        self.grid = None
        self.mask = None

        angle = math.radians(35.0)
        self.ball = [self.width * 0.5, self.height * 0.35]
        self.vel = [math.cos(angle) * BALL_SPEED, math.sin(angle) * BALL_SPEED]
        self.score = 0
        self.drops = 0
        self.flash = 0.0
        self._bounce_cooldown = 0.0

        # ── measurement ───────────────────────────────────────────────────────
        self.frames = 0
        self.grid_frames = 0
        self.grid_bytes = 0
        self.grid_shape = None
        self.solid_fraction: list[float] = []
        self.mask_ms: list[float] = []

    def update(self, payload, dt):
        import time as _time
        self.frames += 1
        self.flash = max(0.0, self.flash - dt * 3.0)
        self._bounce_cooldown = max(0.0, self._bounce_cooldown - dt)

        grid = payload.get("depth_grid")
        if grid is not None:
            self.grid_frames += 1
            self.grid = grid
            if self.grid_shape is None:
                self.grid_shape = list(grid.shape)
                self.grid_bytes = int(grid.nbytes)
            t0 = _time.perf_counter()
            # "Nearer than the threshold and actually returning" — a zero cell
            # is no return, not a surface at 0 m, and treating it as solid would
            # make every sensor dropout a wall.
            self.mask = (grid > 0.0) & (grid < self.solid_m)
            self.mask_ms.append((_time.perf_counter() - t0) * 1000.0)
            self.solid_fraction.append(float(self.mask.mean()))

        self._step_ball(dt)

    def _step_ball(self, dt):
        self.ball[0] += self.vel[0] * dt
        self.ball[1] += self.vel[1] * dt

        if self.ball[0] < BALL_RADIUS:
            self.ball[0], self.vel[0] = BALL_RADIUS, abs(self.vel[0])
        elif self.ball[0] > self.width - BALL_RADIUS:
            self.ball[0], self.vel[0] = self.width - BALL_RADIUS, -abs(self.vel[0])
        if self.ball[1] < BALL_RADIUS:
            self.ball[1], self.vel[1] = BALL_RADIUS, abs(self.vel[1])

        if self.ball[1] > self.height + 60:
            self.drops += 1
            self.ball = [self.width * 0.5, self.height * 0.3]
            self.vel = [BALL_SPEED * 0.7, BALL_SPEED * 0.7]
            return

        if self.mask is None or not self.mask.any() or self._bounce_cooldown > 0.0:
            return

        rows, cols = self.mask.shape
        cx = int(self.ball[0] * cols / self.width)
        cy = int(self.ball[1] * rows / self.height)
        if not (0 <= cx < cols and 0 <= cy < rows) or not self.mask[cy, cx]:
            return

        # Bounce off the local surface normal, approximated from the solid cells
        # around the hit. A flat "reverse vy" would make the silhouette shape
        # irrelevant, which is the one thing this game is for.
        y0, y1 = max(0, cy - 1), min(rows, cy + 2)
        x0, x1 = max(0, cx - 1), min(cols, cx + 2)
        patch = self.mask[y0:y1, x0:x1].astype(np.float32)
        gy, gx = np.gradient(patch) if patch.size >= 4 else (np.zeros(1), np.zeros(1))
        nx, ny = -float(np.sum(gx)), -float(np.sum(gy))
        norm = math.hypot(nx, ny)
        if norm < 1e-4:
            nx, ny = 0.0, -1.0
        else:
            nx, ny = nx / norm, ny / norm

        dot = self.vel[0] * nx + self.vel[1] * ny
        if dot < 0:
            self.vel[0] -= 2.0 * dot * nx
            self.vel[1] -= 2.0 * dot * ny
        else:
            self.vel[1] = -abs(self.vel[1])

        speed = math.hypot(*self.vel) or 1.0
        scale = min(1.35 * BALL_SPEED, speed * 1.02) / speed
        self.vel[0] *= scale
        self.vel[1] *= scale
        self._bounce_cooldown = 0.08
        self.score += 1
        self.flash = 1.0
        self.ball[1] -= 6.0

    def render(self, canvas):
        canvas[:] = (14, 14, 20)

        if self.grid is not None:
            # Depth as a colour field, then the solid cells picked out on top.
            g = np.clip(self.grid, 0.0, 2.5) / 2.5
            field = (np.stack([1.0 - g, g * 0.6, g], axis=-1) * 90).astype(np.uint8)
            canvas[:] = cv2.resize(field, (self.width, self.height),
                                   interpolation=cv2.INTER_LINEAR)
            if self.mask is not None and self.mask.any():
                solid = cv2.resize(self.mask.astype(np.uint8) * 255,
                                   (self.width, self.height),
                                   interpolation=cv2.INTER_NEAREST)
                tint = np.zeros_like(canvas)
                tint[:] = (150, 220, 120)
                canvas[:] = np.where(solid[..., None] > 0,
                                     cv2.addWeighted(canvas, 0.45, tint, 0.55, 0),
                                     canvas)

        cv2.circle(canvas, (int(self.ball[0]), int(self.ball[1])),
                   int(BALL_RADIUS + 6 * self.flash), (255, 245, 220), -1)
        cv2.circle(canvas, (int(self.ball[0]), int(self.ball[1])),
                   int(BALL_RADIUS + 6 * self.flash), (90, 160, 255), 2)

        draw_panel(canvas, 12, 12, 420, 116)
        draw_text(canvas, f"rallies {self.score}   drops {self.drops}", 26, 44,
                  0.75, (235, 240, 245))
        if self.grid_shape:
            draw_text(canvas,
                      f"grid {self.grid_shape[1]}x{self.grid_shape[0]}"
                      f"  {self.grid_bytes / 1024.0:.1f} KiB/frame",
                      26, 74, 0.55, (150, 230, 200), 1)
        if self.solid_fraction:
            draw_text(canvas, f"solid {self.solid_fraction[-1] * 100:5.1f}% of cells"
                              f"   threshold {self.solid_m:.2f} m",
                      26, 100, 0.55, (200, 200, 220), 1)
        if self.grid is None:
            draw_text(canvas, "no depth grid in payload", 26, 122, 0.55,
                      (120, 120, 220), 1)

    def metrics(self):
        out = {
            "frames": self.frames,
            "grid_frames": self.grid_frames,
            "grid_shape": self.grid_shape,
            "grid_bytes_per_frame": self.grid_bytes,
            "grid_kib_per_second_at_30fps": (
                round(self.grid_bytes * 30.0 / 1024.0, 1) if self.grid_bytes else 0),
            "rallies": self.score,
            "drops": self.drops,
            "solid_threshold_m": self.solid_m,
        }
        if self.solid_fraction:
            out["solid_cells_mean_pct"] = round(
                100.0 * statistics.fmean(self.solid_fraction), 2)
        if self.mask_ms:
            ordered = sorted(self.mask_ms)
            out["mask_build_mean_ms"] = round(statistics.fmean(ordered), 4)
            out["mask_build_p95_ms"] = round(ordered[int(len(ordered) * 0.95)], 4)
        return out


def _landmarks(t, i):
    """Two hands sweeping as paddles, at a depth that keeps them solid."""
    left = labkit.hand_landmarks(cx=0.30 + 0.22 * math.sin(t * 1.1), cy=0.72,
                                 sign="open_palm", scale=0.14, z=-0.2)
    right = labkit.hand_landmarks(cx=0.70 + 0.22 * math.sin(t * 0.8 + 1.0), cy=0.66,
                                  sign="open_palm", scale=0.14, z=-0.2)
    return [left, right]


DepthPong.headless_landmarks = staticmethod(_landmarks)


if __name__ == "__main__":
    raise SystemExit(labkit.run(DepthPong()))
