"""
conductor.py — conduct a metronome. Beats are the reversals of your baton hand.

What this is testing
--------------------
A conductor's downbeat is a *direction reversal at speed*, which is the worst
case for the One-Euro filter: high velocity, then an instant flip. Everything
else in the repo tests the filter at rest (jitter) or in steady motion (aiming).
Nothing has tested it where it is hardest.

This game detects a beat as the frame the horizontal velocity changes sign, and
does it twice in parallel — once from `index_velocity` (smoothed) and once from
`index_velocity_raw` (unsmoothed). Both are scored against the same metronome.
The difference between the two reported latencies is the phase lag One-Euro is
costing, measured rather than asserted, and it is exactly the number you need to
decide whether `filter_beta` is set right for fast input.

`--beta` re-tunes the live filter so a run can be repeated across values.

Scope: game side, reading the engine's new motion keys.

    python conductor.py                       play it
    python conductor.py --headless 900        scripted run
    python conductor.py --headless 900 --beta 0.05
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

BPM = 92.0
BEAT = 60.0 / BPM
#: A reversal only counts if the hand was actually moving. Below this the sign
#: of the velocity is noise, and every trembling frame would be a downbeat.
MIN_SPEED = 90.0
#: Ignore reversals this soon after the last one — a real baton does not
#: change direction twice in 100 ms.
REFRACTORY = 0.12
GRADE = ((0.045, "PERFECT", (120, 240, 160)),
         (0.090, "GOOD", (150, 220, 255)),
         (0.160, "LATE", (120, 170, 255)))


class ReversalDetector:
    """Fires when the sign of vx flips while the hand is moving with intent."""

    def __init__(self, name):
        self.name = name
        self.prev_vx = 0.0
        self.last_fire = -99.0
        self.beats: list[float] = []

    def feed(self, vx, speed, t):
        fired = False
        if (speed >= MIN_SPEED and self.prev_vx != 0.0
                and (vx < 0.0) != (self.prev_vx < 0.0)
                and (t - self.last_fire) >= REFRACTORY):
            fired = True
            self.last_fire = t
            self.beats.append(t)
        if abs(vx) > 1e-6:
            self.prev_vx = vx
        return fired


class Conductor(LabGame):
    title = "Conductor — beat by velocity reversal"
    width, height = 960, 720
    wants_max_hands = 1

    goal = ("Wave one hand side to side like a conductor's baton. You do not "
            "tap or click anything: the moment your hand STOPS going one way "
            "and starts going the other — the turn at the end of a sweep — is "
            "your beat. Make those turns land on the metronome.")
    controls = (
        ("Point one finger",
         "the ring is the tip of your baton; the arrow shows which way it is "
         "travelling right now"),
        ("Sweep left, then right, then left",
         "one continuous side-to-side wave — never a flick and a return to "
         "the middle"),
        ("A beat = the turn, not the sweep",
         "nothing is scored while you are moving across; the beat fires on "
         "the single frame your direction flips"),
        ("Copy the dot at the top",
         "the metronome dot bounces between the two brackets. Turn when it "
         "turns and you are on the beat"),
        ("Swing with intent",
         f"a turn only counts above {MIN_SPEED:.0f} px/s — a drifting hand "
         f"reverses on noise, not on purpose"),
        ("Turn on the pulse",
         "you are graded PERFECT / GOOD / LATE against the nearest beat"),
    )

    def add_args(self, parser):
        parser.add_argument("--beta", type=float, default=None,
                            help="override the One-Euro speed coupling")
        parser.add_argument("--bpm", type=float, default=BPM)

    def configure(self):
        self.beat = 60.0 / float(getattr(self.args, "bpm", None) or BPM)

    def tune_pipeline(self, pipeline):
        beta = getattr(self.args, "beta", None)
        if beta is not None:
            pipeline.set_filter_tuning(beta=beta)

    def __init__(self):
        self.beat = BEAT
        self.time = 0.0
        self.smooth = ReversalDetector("smoothed")
        self.raw = ReversalDetector("raw")
        self.trail: list[tuple[int, int]] = []
        self.pos = (self.width // 2, self.height // 2)
        self.speed = 0.0
        self.vx = 0.0
        self.last_grade = ""
        self.last_grade_colour = (200, 200, 200)
        self.last_err_ms = 0.0
        self.flash = 0.0
        self.score = 0
        self.hand_visible = False

        # ── measurement ───────────────────────────────────────────────────────
        self.errors = {"smoothed": [], "raw": []}
        self.frames = 0
        self.speeds: list[float] = []

    def _nearest_beat_error(self, t):
        """Signed seconds from t to the closest metronome beat."""
        n = round(t / self.beat)
        return t - n * self.beat

    def update(self, payload, dt):
        self.frames += 1
        self.time += dt
        self.hand_visible = bool(payload.get("hand_visible"))
        self.flash = max(0.0, self.flash - dt * 4.0)

        if not self.hand_visible:
            return

        self.pos = tuple(payload["index_pos"])
        self.trail.append(self.pos)
        if len(self.trail) > 45:
            self.trail.pop(0)

        vx_s, _ = payload.get("index_velocity", (0.0, 0.0))
        vx_r, vy_r = payload.get("index_velocity_raw", (0.0, 0.0))
        self.vx = vx_s
        self.speed = float(payload.get("index_speed", 0.0))
        self.speeds.append(self.speed)
        raw_speed = math.hypot(vx_r, vy_r)

        if self.smooth.feed(vx_s, self.speed, self.time):
            err = self._nearest_beat_error(self.time)
            self.errors["smoothed"].append(err * 1000.0)
            self.last_grade, self.last_grade_colour = self._grade(err)
            self.last_err_ms = err * 1000.0
            self.flash = 1.0
            if abs(err) <= GRADE[-1][0]:
                self.score += 1

        if self.raw.feed(vx_r, raw_speed, self.time):
            self.errors["raw"].append(self._nearest_beat_error(self.time) * 1000.0)

    @staticmethod
    def _grade(err):
        for limit, label, colour in GRADE:
            if abs(err) <= limit:
                return label, colour
        return "MISS", (90, 90, 220)

    def render(self, canvas):
        canvas[:] = (18, 14, 22)

        # Metronome. The dot sweeps end-to-end over *two* beats, so it reverses
        # exactly once per beat — the dot's turn is the beat, which is the whole
        # thing the player has to copy. (A one-beat sweep turns twice per beat
        # and teaches the wrong rhythm.)
        left, right = 80, self.width - 80
        span = right - left
        sweep = (self.time % (2.0 * self.beat)) / (2.0 * self.beat)
        tri = abs(1.0 - 2.0 * sweep)                 # 1 at the ends, 0 mid-sweep
        bx = int(left + span * tri)
        phase = (self.time % self.beat) / self.beat  # 0 on every beat
        pulse = max(0.0, 1.0 - phase * 6.0)
        cv2.line(canvas, (left, 90), (right, 90), (60, 60, 74), 3)
        # Brackets marking where the dot turns — i.e. where a beat happens.
        for ex in (left, right):
            cv2.line(canvas, (ex, 66), (ex, 114), (110, 150, 190), 2)
        cv2.circle(canvas, (bx, 90), int(14 + 16 * pulse),
                   (110, 200, 255) if pulse > 0.4 else (70, 110, 150), -1)
        draw_text(canvas, f"{60.0 / self.beat:.0f} BPM", left, 56, 0.7, (170, 200, 230), 2)
        draw_text(canvas, "TURN", left - 30, 138, 0.5, (110, 150, 190), 1)
        draw_text(canvas, "TURN", right - 30, 138, 0.5, (110, 150, 190), 1)

        # Which way to swing next, and how close the turn is.
        going_left = sweep < 0.5           # dot is travelling towards `left`
        next_turn = "TURN AT " + ("LEFT" if going_left else "RIGHT")
        near = max(0.0, 1.0 - abs(1.0 - phase) * 5.0)   # ramps up into the beat
        colour = (150, 235, 255) if near > 0.5 else (110, 120, 140)
        (tw, _), _ = cv2.getTextSize(next_turn, labkit.FONT, 0.9, 2)
        draw_text(canvas, next_turn, (self.width - tw) // 2, 180, 0.9, colour, 2)

        for i, (x, y) in enumerate(self.trail):
            a = (i + 1) / len(self.trail)
            cv2.circle(canvas, (x, y), int(3 + 9 * a),
                       (int(70 + 120 * a), int(120 + 110 * a), int(200 * a)), -1)

        if self.hand_visible:
            radius = int(20 + 22 * self.flash)
            cv2.circle(canvas, self.pos, radius, (140, 230, 255), 3)
            # Velocity vector — the thing the beat detector is actually reading.
            cv2.arrowedLine(canvas, self.pos,
                            (int(self.pos[0] + self.vx * 0.12), self.pos[1]),
                            (255, 200, 130), 3, tipLength=0.3)
            moving = ("moving LEFT" if self.vx < -MIN_SPEED else
                      "moving RIGHT" if self.vx > MIN_SPEED else
                      "too slow to count")
            draw_text(canvas, moving, self.pos[0] - 60, self.pos[1] - 34, 0.5,
                      (255, 200, 130) if abs(self.vx) > MIN_SPEED
                      else (140, 140, 150), 1)
        else:
            draw_text(canvas, "show one hand to the camera",
                      self.width // 2 - 150, self.height // 2, 0.7,
                      (180, 180, 195), 2)

        # First-run coach: the rule people miss is that the *turn* is the beat.
        if not self.errors["smoothed"]:
            draw_panel(canvas, self.width // 2 - 260, 210, 520, 56)
            draw_text(canvas, "Wave your hand side to side, wide and steady.",
                      self.width // 2 - 244, 232, 0.55, (215, 220, 235), 1)
            draw_text(canvas, "The beat is the TURN at each end - not the sweep.",
                      self.width // 2 - 244, 254, 0.55, (150, 235, 255), 1)

        if self.last_grade:
            (tw, _), _ = cv2.getTextSize(self.last_grade, labkit.FONT, 1.6, 4)
            draw_text(canvas, self.last_grade, (self.width - tw) // 2,
                      self.height - 90, 1.6, self.last_grade_colour, 4)
            detail = ("%+.0f ms  (%s)" % (self.last_err_ms,
                      "early" if self.last_err_ms < 0 else "late"))
            (dw, _), _ = cv2.getTextSize(detail, labkit.FONT, 0.6, 2)
            draw_text(canvas, detail, (self.width - dw) // 2,
                      self.height - 108, 0.6, self.last_grade_colour, 1)

        draw_panel(canvas, 12, self.height - 66, 430, 54)
        s_err = self.errors["smoothed"]
        r_err = self.errors["raw"]
        draw_text(canvas,
                  f"beats  smoothed {len(s_err):3d}   raw {len(r_err):3d}   score {self.score}",
                  24, self.height - 44, 0.55, (200, 210, 225), 1)
        if s_err and r_err:
            draw_text(canvas,
                      f"mean err  smoothed {statistics.fmean(s_err):+6.1f} ms"
                      f"   raw {statistics.fmean(r_err):+6.1f} ms",
                      24, self.height - 22, 0.55, (255, 205, 140), 1)

        draw_text(canvas, f"speed {self.speed:6.0f} px/s", self.width - 260, 56,
                  0.6, (150, 240, 190), 1)

    def metrics(self):
        out = {"frames": self.frames, "score": self.score,
               "bpm": round(60.0 / self.beat, 1)}
        if self.speeds:
            out["speed_max_px_s"] = round(max(self.speeds), 1)
            out["speed_mean_px_s"] = round(statistics.fmean(self.speeds), 1)
        for name, errs in self.errors.items():
            if not errs:
                out[f"{name}_beats"] = 0
                continue
            out[f"{name}_beats"] = len(errs)
            out[f"{name}_mean_err_ms"] = round(statistics.fmean(errs), 2)
            out[f"{name}_abs_mean_err_ms"] = round(
                statistics.fmean([abs(e) for e in errs]), 2)
            out[f"{name}_stdev_ms"] = (round(statistics.pstdev(errs), 2)
                                       if len(errs) > 1 else 0.0)
        if self.errors["smoothed"] and self.errors["raw"]:
            out["filter_lag_ms"] = round(
                statistics.fmean(self.errors["smoothed"])
                - statistics.fmean(self.errors["raw"]), 2)
        return out


# ── Headless script ───────────────────────────────────────────────────────────

SWEEP_PX = 0.30          # normalised amplitude either side of centre


def _landmarks(t, i):
    """
    A baton sweeping left-right in perfect time: one reversal per beat, exactly
    on the beat. A triangle wave rather than a sine, because a sine's velocity
    goes to zero at the turn and the reversal frame becomes ambiguous — the
    detector would be scored on where the noise floor sits rather than on the
    filter.
    """
    period = 2.0 * BEAT                      # left-right-left is two beats
    u = (t % period) / period
    tri = 4.0 * abs(u - 0.5) - 1.0           # -1 .. 1 .. -1, reversing on beats
    return labkit.hand_landmarks(cx=0.5 + SWEEP_PX * tri, cy=0.55, sign="point")


Conductor.headless_landmarks = staticmethod(_landmarks)


def main():
    game = Conductor()
    code = labkit.run(game)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
