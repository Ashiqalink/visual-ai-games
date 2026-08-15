"""
duckhunt.py — ducks dive at your FACE; you swat them with your index finger.

What this is testing
--------------------
`target_x` / `target_y` have been emitted on every payload since the pipeline was
written and no game has ever read them. This one makes the face the thing under
attack, so face tracking is load-bearing rather than decorative, and it answers
the question that follows immediately: what does MediaPipe FaceDetection do when
a hand crosses the face?

That is the measurement. Every frame the game records whether a face was found
and how far the swatting hand is from it, then splits the face-loss rate into
"hand near the face" and "hand away from the face". If the two rates match, hand
occlusion is a non-issue and other games can put the hand over the face freely.
If they diverge, no game should ever require it.

Scope: game side. It reads the existing face keys plus the new `face_box`; the
engine change it depends on is only that the box is now carried.

    python duckhunt.py                 play it
    python duckhunt.py --headless 900  scripted run, prints the JSON report
"""

from __future__ import annotations

import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import cv2
import numpy as np

import labkit
from labkit import LabGame, draw_bar, draw_panel, draw_text

#: A hand this close to the face centre counts as "over the face" for the
#: occlusion split. Scaled by the face box when one is available.
NEAR_FACE_PX = 180.0
SWAT_RADIUS = 70.0
DUCK_SPEED = (110.0, 210.0)
SPAWN_EVERY = 1.1


class Duck:
    __slots__ = ("x", "y", "vx", "vy", "phase", "alive", "hue", "speed")

    def __init__(self, x, y, speed, hue):
        self.x, self.y = x, y
        self.vx = self.vy = 0.0
        self.phase = random.uniform(0, math.tau)
        self.alive = True
        self.hue = hue
        self.speed = speed

    def steer(self, tx, ty, dt):
        dx, dy = tx - self.x, ty - self.y
        dist = max(1e-3, math.hypot(dx, dy))
        self.vx, self.vy = dx / dist * self.speed, dy / dist * self.speed
        # Ducks do not fly in straight lines; the wobble also keeps the swat
        # from being a pure distance-holding exercise.
        self.phase += dt * 6.0
        self.x += (self.vx + math.cos(self.phase) * 60.0) * dt
        self.y += (self.vy + math.sin(self.phase) * 45.0) * dt


class DuckHunt(LabGame):
    title = "Duck Hunt — face as the target"
    width, height = 960, 720
    wants_max_hands = 1

    def __init__(self):
        self.ducks: list[Duck] = []
        self.score = 0
        self.lives = 5
        self.spawn_timer = 0.0
        self.face = (self.width / 2.0, self.height / 2.0)
        self.face_box = (0, 0, 0, 0)
        self.hand = None
        self.swat_flash = 0.0
        self.rng = random.Random(7)

        # ── measurement ───────────────────────────────────────────────────────
        self.frames = 0
        self.face_frames = 0
        self.face_centres: list[tuple[float, float]] = []
        self.loss_near = [0, 0]      # [lost, total] while the hand is over the face
        self.loss_far = [0, 0]
        self.face_jumps: list[float] = []
        self._prev_face = None
        self._prev_visible = False
        self.face_dropouts = 0

    # ── input ─────────────────────────────────────────────────────────────────
    def update(self, payload, dt):
        self.frames += 1
        visible = bool(payload.get("face_visible"))
        self.face_box = payload.get("face_box", (0, 0, 0, 0))
        fx, fy = float(payload["target_x"]), float(payload["target_y"])

        hand_visible = bool(payload.get("hand_visible"))
        self.hand = tuple(payload["index_pos"]) if hand_visible else None

        # Occlusion split. The bucket is chosen by where the hand is *this*
        # frame, so a face lost while the hand sits over it lands in "near".
        near = False
        if hand_visible:
            radius = NEAR_FACE_PX
            if self.face_box[2] > 0:
                radius = max(NEAR_FACE_PX, self.face_box[2] * 1.1)
            near = math.hypot(self.hand[0] - fx, self.hand[1] - fy) < radius
        bucket = self.loss_near if near else self.loss_far
        bucket[1] += 1
        if not visible:
            bucket[0] += 1

        if visible:
            self.face_frames += 1
            self.face_centres.append((fx, fy))
            if self._prev_face is not None and self._prev_visible:
                self.face_jumps.append(math.dist(self._prev_face, (fx, fy)))
            self._prev_face = (fx, fy)
            self.face = (fx, fy)
        elif self._prev_visible:
            self.face_dropouts += 1
        self._prev_visible = visible

        # ── gameplay ─────────────────────────────────────────────────────────
        self.spawn_timer -= dt
        if self.spawn_timer <= 0.0 and len(self.ducks) < 9:
            self.spawn_timer = SPAWN_EVERY
            self._spawn()

        self.swat_flash = max(0.0, self.swat_flash - dt * 3.0)

        for duck in self.ducks:
            duck.steer(*self.face, dt)
            if math.dist((duck.x, duck.y), self.face) < 55.0:
                duck.alive = False
                self.lives -= 1
            elif self.hand and math.dist((duck.x, duck.y), self.hand) < SWAT_RADIUS:
                duck.alive = False
                self.score += 1
                self.swat_flash = 1.0
        self.ducks = [d for d in self.ducks if d.alive]

        if self.lives <= 0:
            self.lives = 5
            self.score = 0
            self.ducks.clear()

    def _spawn(self):
        edge = self.rng.randrange(4)
        w, h = self.width, self.height
        x, y = {0: (self.rng.uniform(0, w), -30.0),
                1: (self.rng.uniform(0, w), h + 30.0),
                2: (-30.0, self.rng.uniform(0, h)),
                3: (w + 30.0, self.rng.uniform(0, h))}[edge]
        self.ducks.append(Duck(x, y, self.rng.uniform(*DUCK_SPEED),
                               self.rng.randrange(0, 180)))

    # ── render ────────────────────────────────────────────────────────────────
    def render(self, canvas):
        canvas[:] = (24, 16, 12)
        cv2.circle(canvas, (int(self.face[0]), int(self.face[1])), 46, (40, 90, 140), -1)
        cv2.circle(canvas, (int(self.face[0]), int(self.face[1])), 46, (90, 170, 230), 2)
        if self.face_box[2] > 0:
            x, y, w, h = self.face_box
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (70, 120, 90), 1)
        draw_text(canvas, "YOU", self.face[0] - 28, self.face[1] + 8, 0.7, (200, 230, 255), 2)

        for duck in self.ducks:
            colour = tuple(int(c) for c in cv2.cvtColor(
                np.uint8([[[duck.hue, 200, 235]]]), cv2.COLOR_HSV2BGR)[0][0])
            cx, cy = int(duck.x), int(duck.y)
            cv2.ellipse(canvas, (cx, cy), (22, 14), 0, 0, 360, colour, -1)
            flap = math.sin(duck.phase * 2.0) * 12
            cv2.ellipse(canvas, (cx, cy - 4), (16, 6), flap, 0, 360, (245, 245, 245), 2)
            cv2.circle(canvas, (cx + 16, cy - 6), 5, (250, 250, 250), -1)

        if self.hand:
            radius = int(SWAT_RADIUS * (1.0 + 0.25 * self.swat_flash))
            cv2.circle(canvas, self.hand, radius, (120, 255, 190), 2)
            cv2.circle(canvas, self.hand, 6, (120, 255, 190), -1)

        draw_panel(canvas, 12, 12, 330, 96)
        draw_text(canvas, f"score {self.score}", 26, 44, 0.85, (150, 255, 190))
        draw_text(canvas, f"lives {self.lives}", 190, 44, 0.85, (150, 190, 255))
        pct = 100.0 * self.face_frames / max(1, self.frames)
        draw_text(canvas, f"face seen {pct:5.1f}%", 26, 76, 0.6,
                  (120, 220, 120) if pct > 90 else (90, 160, 240), 1)
        draw_bar(canvas, 26, 86, 300, 8, pct / 100.0)

    def metrics(self):
        def rate(bucket):
            return round(100.0 * bucket[0] / bucket[1], 2) if bucket[1] else None
        centres = self.face_centres
        jitter = None
        if len(centres) > 3:
            jitter = round(statistics.pstdev([c[0] for c in centres]), 2)
        return {
            "frames": self.frames,
            "face_visible_pct": round(100.0 * self.face_frames / max(1, self.frames), 2),
            "face_dropout_events": self.face_dropouts,
            "face_centre_x_stdev_px": jitter,
            "face_frame_to_frame_jump_p95_px": (
                round(sorted(self.face_jumps)[int(len(self.face_jumps) * 0.95)], 2)
                if len(self.face_jumps) > 20 else None),
            "face_loss_pct_hand_near_face": rate(self.loss_near),
            "face_loss_pct_hand_away": rate(self.loss_far),
            "frames_hand_near_face": self.loss_near[1],
            "score": self.score,
        }


def _script(t, i):
    """
    Headless input: a face that drifts gently while the hand sweeps back and
    forth across it, so the occlusion buckets both fill up.
    """
    w, h = 960, 720
    fx = w / 2 + math.sin(t * 0.7) * 90
    fy = h / 2 + math.cos(t * 0.5) * 50
    # The hand crosses the face twice per cycle.
    hx = w / 2 + math.sin(t * 1.9) * 320
    hy = h / 2 + math.cos(t * 1.3) * 120
    return {
        "target_x": fx, "target_y": fy,
        "face_visible": True,
        "face_box": (int(fx - 60), int(fy - 75), 120, 150),
        "face_count": 1,
        "hand_visible": True,
        "index_pos": (int(hx), int(hy)),
        "pinch_pos": (int(hx), int(hy)),
    }


DuckHunt.headless_script = staticmethod(_script)


if __name__ == "__main__":
    raise SystemExit(labkit.run(DuckHunt()))
