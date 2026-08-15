"""
cradle.py — a string stretched between both hands. Span it across the targets.

What this is testing
--------------------
This is the game the payload change exists for. `max_num_hands=2` has been set on
the detector since the pipeline was written, and until now the second hand was
discarded one line later — so nothing in either repo has ever run two-handed.

The control is the *relationship* between the hands rather than either hand's
position: the string is the segment between the two palms, and a ring is cleared
by passing the string through it. That makes slot stability load-bearing in a way
a two-cursor game would not — if the hands swap slots, the string is unchanged,
but if the slots swap their filter state the string jumps.

Measured: how often both hands are actually tracked, and how far a slot's
reported position moves between frames. That second one is the swap detector —
a slot handed the wrong hand teleports across the gap between them, whereas
counting which *side* of the frame a slot is on would just count the hands
crossing the midline, which they are supposed to do.

Scope: game side, on top of the engine's new `hands` / `hand_left` / `hand_right`.

    python cradle.py                  play it
    python cradle.py --headless 900   scripted run, hands crossing repeatedly
"""

from __future__ import annotations

import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import cv2

import labkit
from labkit import LabGame, draw_panel, draw_text

RING_RADIUS = 46
RING_LIFETIME = 6.0
MIN_SPAN = 120.0        # hands closer than this cannot thread a ring


def _segment_point_distance(ax, ay, bx, by, px, py):
    """Distance from point P to segment AB — the string-through-ring test."""
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-6:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class Ring:
    __slots__ = ("x", "y", "age", "cleared")

    def __init__(self, x, y):
        self.x, self.y, self.age, self.cleared = x, y, 0.0, False


class Cradle(LabGame):
    title = "Cat's Cradle — two-handed string"
    width, height = 1120, 760
    wants_max_hands = 2

    def __init__(self):
        self.rings: list[Ring] = []
        self.spawn_timer = 0.0
        self.score = 0
        self.missed = 0
        self.rng = random.Random(3)
        self.hands: list[dict] = []
        self.a = None
        self.b = None
        self.flash = 0.0

        # ── measurement ───────────────────────────────────────────────────────
        self.frames = 0
        self.two_hand_frames = 0
        self.one_hand_frames = 0
        self.spans: list[float] = []
        self.slot_speeds: dict[int, list[float]] = {0: [], 1: []}
        # A slot that swaps to the other hand teleports across the gap between
        # them. Counting *sides* instead would just count the hands crossing the
        # midline, which they are supposed to do.
        self.slot_jumps: dict[int, list[float]] = {0: [], 1: []}
        self._slot_last_pos: dict[int, tuple[int, int]] = {}
        self.handedness_seen: dict[str, int] = {}

    def update(self, payload, dt):
        self.frames += 1
        self.flash = max(0.0, self.flash - dt * 3.0)
        self.hands = list(payload.get("hands", ()))

        for entry in self.hands:
            label = entry.get("handedness", "unknown")
            self.handedness_seen[label] = self.handedness_seen.get(label, 0) + 1
            slot = entry.get("slot", 0)
            if slot in self.slot_speeds:
                self.slot_speeds[slot].append(float(entry.get("index_speed", 0.0)))

            pos = tuple(entry.get("pinch_pos", (0, 0)))
            if slot in self._slot_last_pos:
                self.slot_jumps.setdefault(slot, []).append(
                    math.dist(pos, self._slot_last_pos[slot]))
            self._slot_last_pos[slot] = pos

        # A slot that vanished this frame must not be diffed against a stale
        # position next time it appears — that reads as a swap when it is just a
        # re-acquire.
        present = {e.get("slot", 0) for e in self.hands}
        for slot in list(self._slot_last_pos):
            if slot not in present:
                del self._slot_last_pos[slot]

        if len(self.hands) >= 2:
            self.two_hand_frames += 1
            self.a = tuple(self.hands[0]["pinch_pos"])
            self.b = tuple(self.hands[1]["pinch_pos"])
            self.spans.append(math.dist(self.a, self.b))
        elif len(self.hands) == 1:
            self.one_hand_frames += 1
            self.a = tuple(self.hands[0]["pinch_pos"])
            self.b = None
        else:
            self.a = self.b = None

        self.spawn_timer -= dt
        if self.spawn_timer <= 0.0 and len(self.rings) < 5:
            self.spawn_timer = 1.6
            self.rings.append(Ring(self.rng.uniform(140, self.width - 140),
                                   self.rng.uniform(140, self.height - 160)))

        span_ok = self.a and self.b and math.dist(self.a, self.b) >= MIN_SPAN
        for ring in self.rings:
            ring.age += dt
            if not ring.cleared and span_ok:
                d = _segment_point_distance(*self.a, *self.b, ring.x, ring.y)
                if d < RING_RADIUS * 0.55:
                    ring.cleared = True
                    self.score += 1
                    self.flash = 1.0
        for ring in self.rings:
            if ring.age > RING_LIFETIME and not ring.cleared:
                self.missed += 1
        self.rings = [r for r in self.rings
                      if not r.cleared and r.age <= RING_LIFETIME]

    def render(self, canvas):
        canvas[:] = (16, 18, 26)

        for ring in self.rings:
            fade = max(0.0, 1.0 - ring.age / RING_LIFETIME)
            colour = (int(90 + 120 * fade), int(160 * fade + 40), 230)
            cv2.circle(canvas, (int(ring.x), int(ring.y)), RING_RADIUS, colour, 4)
            cv2.circle(canvas, (int(ring.x), int(ring.y)), int(RING_RADIUS * 0.55),
                       (60, 60, 80), 1)

        if self.a and self.b:
            span = math.dist(self.a, self.b)
            taut = span >= MIN_SPAN
            colour = (120, 240, 200) if taut else (90, 90, 200)
            # A slack string sags; a taut one is straight. Purely cosmetic, but
            # it makes the MIN_SPAN rule legible without a HUD readout.
            sag = max(0.0, (MIN_SPAN - span)) * 0.6
            mid = ((self.a[0] + self.b[0]) // 2,
                   int((self.a[1] + self.b[1]) / 2 + sag))
            cv2.line(canvas, self.a, mid, colour, 3, cv2.LINE_AA)
            cv2.line(canvas, mid, self.b, colour, 3, cv2.LINE_AA)

        for i, entry in enumerate(self.hands):
            pos = tuple(entry["pinch_pos"])
            slot = entry.get("slot", i)
            colour = (255, 190, 120) if slot == 0 else (180, 140, 255)
            cv2.circle(canvas, pos, 26, colour, 3)
            cv2.circle(canvas, pos, 5, colour, -1)
            draw_text(canvas, f"slot {slot} · {entry.get('handedness', '?')}",
                      pos[0] - 60, pos[1] - 38, 0.5, colour, 1)

        draw_panel(canvas, 12, 12, 400, 108)
        draw_text(canvas, f"score {self.score}   missed {self.missed}", 26, 44,
                  0.75, (230, 235, 245))
        pct = 100.0 * self.two_hand_frames / max(1, self.frames)
        draw_text(canvas, f"two hands tracked {pct:5.1f}%", 26, 76, 0.55,
                  (140, 230, 170) if pct > 80 else (140, 160, 240), 1)
        jumps = sum(1 for js in self.slot_jumps.values() for j in js if j > 120.0)
        draw_text(canvas, f"slot jumps >120px  {jumps}", 26, 102, 0.55,
                  (255, 180, 150), 1)

        if len(self.hands) < 2:
            draw_text(canvas, "show BOTH hands", self.width // 2 - 150,
                      self.height - 40, 0.9, (150, 170, 220), 2)

    def metrics(self):
        out = {
            "frames": self.frames,
            "two_hand_pct": round(100.0 * self.two_hand_frames / max(1, self.frames), 2),
            "one_hand_pct": round(100.0 * self.one_hand_frames / max(1, self.frames), 2),
            "slot_position_jumps_over_120px": sum(
                1 for js in self.slot_jumps.values() for j in js if j > 120.0),
            "handedness_frames": self.handedness_seen,
            "score": self.score,
            "missed": self.missed,
        }
        if self.spans:
            out["span_mean_px"] = round(statistics.fmean(self.spans), 1)
            out["span_min_px"] = round(min(self.spans), 1)
            out["span_max_px"] = round(max(self.spans), 1)
        for slot, speeds in self.slot_speeds.items():
            if speeds:
                out[f"slot{slot}_speed_mean_px_s"] = round(statistics.fmean(speeds), 1)
                out[f"slot{slot}_frames"] = len(speeds)
        for slot, jumps in self.slot_jumps.items():
            if jumps:
                ordered = sorted(jumps)
                out[f"slot{slot}_jump_p95_px"] = round(ordered[int(len(ordered) * 0.95)], 1)
                out[f"slot{slot}_jump_max_px"] = round(ordered[-1], 1)
        return out


def _landmarks(t, i):
    """
    Two hands sweeping past each other and crossing over twice per cycle — the
    exact motion that makes MediaPipe's list order unreliable, and therefore the
    motion that would scramble per-hand filter state if slots followed that
    order. One hand also drops out of frame periodically so the re-acquire path
    is covered.
    """
    left_x = 0.5 + 0.30 * math.sin(t * 0.8)
    right_x = 0.5 - 0.30 * math.sin(t * 0.8)
    left = labkit.hand_landmarks(cx=left_x, cy=0.55, sign="fist", scale=0.11)
    right = labkit.hand_landmarks(cx=right_x, cy=0.62, sign="fist", scale=0.11)

    # Roughly one second in every six with only one hand visible, so the
    # re-acquire path is covered rather than assumed.
    if (t % 6.0) > 5.0:
        return [left]

    # Hand the pair back in a deliberately unstable order. MediaPipe makes no
    # promise about this ordering, and a game that survives it only because the
    # test always fed the same order has not been tested.
    if int(t * 3.0) % 2:
        return [right, left]
    return [left, right]


Cradle.headless_landmarks = staticmethod(_landmarks)


if __name__ == "__main__":
    raise SystemExit(labkit.run(Cradle()))
