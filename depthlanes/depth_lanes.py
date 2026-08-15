"""
depth_lanes.py — a rhythm game whose three lanes are three ToF distance bands.

What this is testing
--------------------
Rhythm is a merciless latency oracle. Everything else built on ToF here (punchy,
flappy) reacts to a *change* in depth, which hides steady-state error: you can be
150 ms late and still feel fine as long as the push registers. This game instead
asks you to be in the right absolute depth band on the beat, and scores the
signed timing error, so latency shows up as a number with a direction.

The second thing it measures is band flicker. `ToFStabilizer` exists to gate
depth noise; the way that failure looks in a game is the lane indicator
chattering between two bands while the hand is objectively still. The game counts
every band change and separately counts the ones that reverse within 100 ms —
those are flicker, not intent, and no amount of "it looks smoother" replaces the
count.

Run it with the stabilizer off and on (`L` calibrates, `X` cancels) and compare
the two reports.

Scope: game side.

    python depth_lanes.py                 play it
    python depth_lanes.py --headless 900  scripted run, prints the JSON report
"""

from __future__ import annotations

import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import cv2

import labkit
from labkit import LabGame, draw_panel, draw_text

#: Depth band edges in metres. The simulated ToF source sits around 0.45 m, so
#: the bands are placed either side of that rather than at pretty round numbers.
BAND_EDGES = (0.38, 0.52)
LANE_NAMES = ("NEAR", "MID", "FAR")
LANE_COLOURS = ((110, 120, 255), (120, 240, 200), (255, 190, 110))

BPM = 100.0
BEAT = 60.0 / BPM
HIT_WINDOW = 0.120          # +/- seconds counted as a hit
APPROACH = 2.0              # seconds a note is visible before its beat
FLICKER_WINDOW = 0.100      # a band change reversed within this is flicker


def band_of(depth_m: float) -> int:
    if depth_m < BAND_EDGES[0]:
        return 0
    if depth_m < BAND_EDGES[1]:
        return 1
    return 2


class Note:
    __slots__ = ("lane", "beat_time", "judged", "result", "error", "best")

    def __init__(self, lane, beat_time):
        self.lane, self.beat_time = lane, beat_time
        self.judged = False
        self.result = None
        self.error = None
        #: Smallest |t - beat_time| at which the player was in this lane, or
        #: None if they never were. Judging on the FIRST in-window frame instead
        #: scores the moment the window opened, not the moment the player
        #: arrived, which pins every error to the window edge and reports a
        #: bimodal +/-window distribution that says nothing about latency.
        self.best = None


class DepthLanes(LabGame):
    title = "Depth Lanes — ToF rhythm"
    width, height = 960, 720
    wants_max_hands = 1

    def add_args(self, parser):
        parser.add_argument("--stabilize", action="store_true",
                            help="calibrate the ToF stabilizer at frame 20, while "
                                 "the scripted hand is still — the correct procedure")
        parser.add_argument("--stabilize-at", type=int, metavar="FRAME",
                            help="calibrate at an arbitrary frame. Point this at a "
                                 "frame where the hand is moving to see what "
                                 "calibrating against motion costs")

    def configure(self):
        if getattr(self.args, "stabilize_at", None) is not None:
            self.stabilize_at_frame = self.args.stabilize_at
        elif getattr(self.args, "stabilize", False):
            self.stabilize_at_frame = 20

    def __init__(self):
        self.time = 0.0
        self.notes = [Note((i * 2 + (i // 3)) % 3, 2.0 + i * BEAT) for i in range(400)]
        self.next_note = 0
        self.depth = 0.45
        self.band = 1
        self.hand_visible = False
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.stab_state = "inactive"
        self.stab_progress = 0.0
        self.flash = 0.0

        # ── measurement ───────────────────────────────────────────────────────
        self.errors_ms: list[float] = []
        self.hits = 0
        self.misses = 0
        self.band_changes = 0
        self.flicker_changes = 0
        self._last_band_change_t = -99.0
        self._band_before_last_change = 1
        self.depth_samples: list[float] = []

    def update(self, payload, dt):
        self.time += dt
        self.hand_visible = bool(payload.get("hand_visible"))
        self.stab_state = payload.get("stabilizer_state", "inactive")
        self.stab_progress = float(payload.get("stabilizer_progress", 0.0))

        if self.hand_visible:
            self.depth = float(payload.get("tof_z_m", 0.45))
            self.depth_samples.append(self.depth)
            new_band = band_of(self.depth)
            if new_band != self.band:
                self.band_changes += 1
                # A change that undoes the previous one within the flicker
                # window is chatter, not a lane switch the player asked for.
                if (self.time - self._last_band_change_t) < FLICKER_WINDOW \
                        and new_band == self._band_before_last_change:
                    self.flicker_changes += 1
                self._band_before_last_change = self.band
                self._last_band_change_t = self.time
                self.band = new_band

        self.flash = max(0.0, self.flash - dt * 4.0)
        self._judge()

    def _judge(self):
        """
        Track how close to the beat the player got into each note's lane, and
        settle the note once its window closes.

        Settling at window close rather than on first contact is what makes the
        reported error mean "how far off the beat were you"; it costs nothing in
        feel, because the visual feedback still fires the moment you land.
        """
        for note in self.notes[self.next_note:]:
            if note.beat_time - self.time > HIT_WINDOW:
                break                      # notes are in time order
            delta = self.time - note.beat_time
            if not note.judged and abs(delta) <= HIT_WINDOW \
                    and self.hand_visible and self.band == note.lane:
                if note.best is None or abs(delta) < abs(note.best):
                    note.best = delta
                self.flash = 1.0
            if not note.judged and delta > HIT_WINDOW:
                note.judged = True
                if note.best is None:
                    note.result = "miss"
                    self.misses += 1
                    self.combo = 0
                else:
                    note.result, note.error = "hit", note.best
                    self.errors_ms.append(note.best * 1000.0)
                    self.hits += 1
                    self.combo += 1
                    self.best_combo = max(self.best_combo, self.combo)
                    self.score += 100 + self.combo * 5

        while self.next_note < len(self.notes) and self.notes[self.next_note].judged:
            self.next_note += 1

    def render(self, canvas):
        canvas[:] = (16, 12, 20)
        lane_h = self.height // 3
        judge_x = 200

        for lane in range(3):
            y0 = lane * lane_h
            active = (lane == self.band and self.hand_visible)
            shade = 42 if active else 22
            cv2.rectangle(canvas, (0, y0), (self.width, y0 + lane_h - 4),
                          tuple(int(c * shade / 255) for c in LANE_COLOURS[lane]), -1)
            draw_text(canvas, LANE_NAMES[lane], 20, y0 + 34, 0.7,
                      LANE_COLOURS[lane] if active else (110, 110, 120), 2)
            cv2.line(canvas, (judge_x, y0), (judge_x, y0 + lane_h - 4),
                     (200, 200, 210), 2)
            if active:
                cv2.circle(canvas, (judge_x, y0 + lane_h // 2), 26,
                           LANE_COLOURS[lane], 3)

        px_per_s = (self.width - judge_x) / APPROACH
        for note in self.notes[self.next_note:self.next_note + 40]:
            dt_to_beat = note.beat_time - self.time
            if dt_to_beat > APPROACH or note.judged:
                continue
            x = judge_x + dt_to_beat * px_per_s
            y = note.lane * lane_h + lane_h // 2
            cv2.circle(canvas, (int(x), int(y)), 20, LANE_COLOURS[note.lane], -1)
            cv2.circle(canvas, (int(x), int(y)), 20, (250, 250, 250), 2)

        draw_panel(canvas, self.width - 300, 12, 288, 148)
        draw_text(canvas, f"score {self.score}", self.width - 286, 44, 0.75, (240, 240, 250))
        draw_text(canvas, f"combo {self.combo}  best {self.best_combo}",
                  self.width - 286, 72, 0.55, (180, 220, 255), 1)
        draw_text(canvas, f"depth {self.depth:.3f} m", self.width - 286, 100, 0.55,
                  (150, 240, 190), 1)
        if self.errors_ms:
            draw_text(canvas, f"mean err {statistics.fmean(self.errors_ms):+6.1f} ms",
                      self.width - 286, 126, 0.55, (255, 210, 140), 1)
        draw_text(canvas,
                  f"flicker {self.flicker_changes}/{self.band_changes}",
                  self.width - 286, 150, 0.5, (255, 160, 160), 1)

        badge = {"sampling": (f"CALIBRATING {self.stab_progress * 100:.0f}%", (0, 190, 255)),
                 "active": ("STABILIZER ON", (110, 230, 130))}.get(
                     self.stab_state, ("STABILIZER OFF  [L] calibrate", (110, 110, 200)))
        draw_text(canvas, badge[0], 20, self.height - 20, 0.55, badge[1], 1)

    def metrics(self):
        errs = self.errors_ms
        out = {
            "notes_judged": self.hits + self.misses,
            "hits": self.hits,
            "misses": self.misses,
            "best_combo": self.best_combo,
            "band_changes": self.band_changes,
            "flicker_changes": self.flicker_changes,
            "flicker_pct": (round(100.0 * self.flicker_changes / self.band_changes, 2)
                            if self.band_changes else None),
            "stabilizer_state": self.stab_state,
        }
        if errs:
            ordered = sorted(abs(e) for e in errs)
            out.update({
                "timing_mean_ms": round(statistics.fmean(errs), 2),
                "timing_abs_mean_ms": round(statistics.fmean(ordered), 2),
                "timing_abs_p95_ms": round(ordered[int(len(ordered) * 0.95)], 2),
                "timing_stdev_ms": round(statistics.pstdev(errs), 2) if len(errs) > 1 else 0.0,
                "late_pct": round(100.0 * sum(1 for e in errs if e > 0) / len(errs), 1),
            })
        if self.depth_samples:
            out["depth_stdev_m"] = round(statistics.pstdev(self.depth_samples), 5)
        return out


#: Lane centres in metres, and how long a real hand takes to travel between
#: bands. A player who is trying does not teleport; 90 ms is a brisk but
#: plausible reposition, and the point of the headless run is to see how much
#: latency the signal chain adds on top of that.
LANE_DEPTH = (0.32, 0.45, 0.60)
MOVE_TIME = 0.090
DEPTH_NOISE_M = 0.010


def _lane_for_beat(n):
    return (n * 2 + (n // 3)) % 3


def _landmarks(t, i):
    """
    Headless input: a player who starts moving toward the next lane exactly on
    the beat and takes MOVE_TIME to get there. Anything the report shows beyond
    MOVE_TIME is latency the signal chain added, not the hand being slow.

    Simulated ToF maps MediaPipe Z as 0.45 + z*0.6, so target depths are driven
    backwards through that.
    """
    first_beat = 2.0
    beat_index = int((t - first_beat) / BEAT) if t >= first_beat else -1
    if beat_index < 0:
        depth = LANE_DEPTH[_lane_for_beat(0)]
    else:
        since = t - (first_beat + beat_index * BEAT)
        prev = LANE_DEPTH[_lane_for_beat(max(0, beat_index - 1))]
        nxt = LANE_DEPTH[_lane_for_beat(beat_index)]
        u = min(1.0, since / MOVE_TIME)
        u = u * u * (3.0 - 2.0 * u)              # smoothstep — hands ease in/out
        depth = prev + (nxt - prev) * u

    # Real ToF is noisy, and the band edges are exactly where that noise turns
    # into a gameplay event.
    depth += math.sin(t * 37.0) * DEPTH_NOISE_M
    return labkit.hand_landmarks(cx=0.5, cy=0.55, sign="point",
                                 z=(depth - 0.45) / 0.6)


DepthLanes.headless_landmarks = staticmethod(_landmarks)


if __name__ == "__main__":
    raise SystemExit(labkit.run(DepthLanes()))
