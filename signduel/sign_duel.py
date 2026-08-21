"""
sign_duel.py — Simon says with hand signs, on a shrinking timer.

What this is testing
--------------------
`_SIGN_DEBOUNCE = 3` is what makes `hand_sign` stable, and it is also the single
thing that will make this game feel sluggish. Nothing in the repo has ever put a
number on it: Sling uses fist/open-palm where a few frames of lag are invisible
against a slingshot pull, so the cost has never been paid anywhere it shows.

The headless run measures it directly. The script changes the *landmarks* at a
known frame and the game records the frame on which the reported sign actually
changed, so the reported latency includes the debounce, the classifier, and any
ambiguous shapes the hand passes through on the way. It also counts how often a
transition detours through a wrong sign — a hand opening from a fist genuinely
passes point and peace, and whether those leak out is the interesting part.

Scope: game side. Nothing here changes the engine; it reads `hand_sign` and
`fingers_extended` as they already are.

    python sign_duel.py                  play it
    python sign_duel.py --headless 900   scripted run, prints the JSON report
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import cv2

import labkit
from labkit import LabGame, draw_bar, draw_panel, draw_text

SIGNS = ("fist", "open_palm", "point", "peace")
SIGN_LABEL = {"fist": "FIST", "open_palm": "OPEN PALM", "point": "POINT",
              "peace": "PEACE"}
SIGN_COLOR = {"fist": (120, 130, 255), "open_palm": (130, 240, 190),
               "point": (255, 200, 120), "peace": (230, 150, 255)}

START_TIME = 3.0
MIN_TIME = 0.9
SPEEDUP = 0.93          # per successful round
HOLD_FRAMES = 2         # frames the correct sign must persist to count


class SignDuel(LabGame):
    title = "Sign Duel — hand-sign reaction"
    width, height = 960, 720
    wants_max_hands = 1

    goal = ("Make the hand sign named in the middle of the screen before the "
            "bar runs out. Every round you get right, the next one is faster.")
    controls = (
        ("FIST", "curl all five fingers"),
        ("OPEN PALM", "all five fingers spread"),
        ("POINT", "index finger only"),
        ("PEACE", "index and middle finger"),
        ("Hold it steady",
         f"the sign has to survive {HOLD_FRAMES} frames to score — flicking "
         f"through it does not count"),
        ("Watch the pips",
         "T I M R P light up per finger, so a misread sign shows you which "
         "finger the tracker disagrees about"),
    )

    def __init__(self):
        self.round = 0
        self.score = 0
        self.lives = 3
        self.target = SIGNS[0]
        self.limit = START_TIME
        self.remaining = START_TIME
        self.hold = 0
        self.sign = "unknown"
        self.fingers = (False,) * 5
        self.flash = 0.0
        self.flash_ok = True
        self.rng = __import__("random").Random(11)
        self._advance()

        # ── measurement ───────────────────────────────────────────────────────
        self.frames = 0
        self.response_ms: list[float] = []
        self._round_started_frame = 0
        self.sign_history: list[str] = []
        self.transitions = 0
        self.unknown_frames = 0

        # Debounce measurement. Only populated when the harness publishes the
        # ground-truth shape (`_script_sign`); a windowed run has no such thing
        # and simply leaves these empty.
        self.debounce_frames: list[int] = []
        self.detour_signs: dict[str, int] = {}
        self._truth = None
        self._prev_truth = None
        self._truth_changed_frame = None
        self._detours_this_change: set[str] = set()

    def _measure_debounce(self, payload):
        """
        Time the gap between the hand changing shape and `hand_sign` reporting
        it, and record any sign the report detours through on the way.
        """
        truth = payload.get("_script_sign")
        if truth is None:
            return

        if truth != self._truth:
            self._prev_truth, self._truth = self._truth, truth
            self._truth_changed_frame = self.frames
            self._detours_this_change = set()
        elif self._truth_changed_frame is not None:
            if self.sign == truth:
                self.debounce_frames.append(self.frames - self._truth_changed_frame)
                for detour in self._detours_this_change:
                    self.detour_signs[detour] = self.detour_signs.get(detour, 0) + 1
                self._truth_changed_frame = None
            elif self.sign not in ("unknown", self._prev_truth):
                # The sign we were just holding is not a detour — it is the
                # debounce doing its job. A *third* sign appearing is the thing
                # worth counting, because that one leaks into gameplay.
                self._detours_this_change.add(self.sign)

    def _advance(self):
        self.round += 1
        choices = [s for s in SIGNS if s != self.target]
        self.target = self.rng.choice(choices)
        self.limit = max(MIN_TIME, self.limit * SPEEDUP)
        self.remaining = self.limit
        self.hold = 0

    def update(self, payload, dt):
        self.frames += 1
        new_sign = payload.get("hand_sign", "unknown")
        self.fingers = payload.get("fingers_extended", (False,) * 5)

        if new_sign == "unknown":
            self.unknown_frames += 1
        if new_sign != self.sign:
            self.sign_history.append(new_sign)
            self.transitions += 1
        self.sign = new_sign

        self._measure_debounce(payload)

        self.remaining -= dt
        self.flash = max(0.0, self.flash - dt * 3.0)

        if self.sign == self.target:
            self.hold += 1
            if self.hold >= HOLD_FRAMES:
                self.score += 1
                self.flash, self.flash_ok = 1.0, True
                self.response_ms.append(
                    (self.frames - self._round_started_frame) * dt * 1000.0)
                self._round_started_frame = self.frames
                self._advance()
        else:
            self.hold = 0

        if self.remaining <= 0.0:
            self.lives -= 1
            self.flash, self.flash_ok = 1.0, False
            self._round_started_frame = self.frames
            self._advance()
            if self.lives <= 0:
                self.lives, self.score, self.limit = 3, 0, START_TIME

    def render(self, canvas):
        tint = 26
        if self.flash > 0.05:
            tint = int(26 + 40 * self.flash)
        canvas[:] = ((tint, tint // 2, tint // 2) if not self.flash_ok
                     else (tint // 2, tint, tint // 2))

        color = SIGN_COLOR[self.target]
        draw_text(canvas, "SHOW", self.width // 2 - 62, 130, 1.0, (200, 200, 210), 2)
        label = SIGN_LABEL[self.target]
        scale = 2.6
        (tw, _), _ = cv2.getTextSize(label, labkit.FONT, scale, 5)
        draw_text(canvas, label, (self.width - tw) // 2, 250, scale, color, 5)

        draw_bar(canvas, 140, 300, self.width - 280, 22,
                 self.remaining / max(1e-3, self.limit),
                 color if self.remaining > self.limit * 0.3 else (90, 90, 240))

        # Live hand readout — the five extension flags are what the classifier
        # sees, so showing them makes a stuck sign diagnosable in place.
        draw_panel(canvas, 140, 360, self.width - 280, 150)
        detected = SIGN_LABEL.get(self.sign, "—")
        ok = self.sign == self.target
        draw_text(canvas, f"detected: {detected}", 170, 404, 0.9,
                  (140, 240, 160) if ok else (200, 200, 210), 2)
        for i, (name, up) in enumerate(zip("T I M R P", self.fingers)):
            x = 170 + i * 62
            cv2.circle(canvas, (x + 12, 452), 16, (90, 220, 130) if up else (70, 70, 82), -1)
            draw_text(canvas, "TIMRP"[i], x + 5, 458, 0.5, (20, 20, 24), 2, shadow=False)
        draw_text(canvas, f"hold {self.hold}/{HOLD_FRAMES}", 170, 494, 0.5,
                  (170, 170, 190), 1)

        draw_panel(canvas, 12, 12, 300, 60)
        draw_text(canvas, f"round {self.round}   score {self.score}", 26, 40, 0.7,
                  (230, 230, 240))
        draw_text(canvas, "♥" * max(0, self.lives), 26, 64, 0.7, (120, 130, 255))

        if self.response_ms:
            draw_text(canvas, f"mean response {statistics.fmean(self.response_ms):.0f} ms",
                      self.width - 330, 40, 0.6, (255, 210, 140), 1)

    def metrics(self):
        out = {
            "frames": self.frames,
            "rounds": self.round,
            "score": self.score,
            "sign_transitions": self.transitions,
            "unknown_frames": self.unknown_frames,
            "unknown_pct": round(100.0 * self.unknown_frames / max(1, self.frames), 2),
        }
        if self.response_ms:
            ordered = sorted(self.response_ms)
            out.update({
                "round_response_mean_ms": round(statistics.fmean(ordered), 1),
                "round_response_median_ms": round(ordered[len(ordered) // 2], 1),
            })
        if self.debounce_frames:
            ordered = sorted(self.debounce_frames)
            out.update({
                "sign_latency_samples": len(ordered),
                "sign_latency_mean_frames": round(statistics.fmean(ordered), 2),
                "sign_latency_max_frames": ordered[-1],
                "sign_latency_mean_ms_at_30fps": round(
                    statistics.fmean(ordered) * 1000.0 / 30.0, 1),
                "transitions_detouring_through_a_wrong_sign": self.detour_signs,
            })
        return out


# ── Headless script ───────────────────────────────────────────────────────────

SCRIPT_PERIOD = 0.80


def _scripted_sign(t):
    """Which sign the scripted hand is *shaped* as right now."""
    return SIGNS[int(t / SCRIPT_PERIOD) % len(SIGNS)]


def _landmarks(t, i):
    """
    Cycle through the four signs, changing shape instantly. An instant change is
    deliberate: it makes the measured latency the pure cost of the classifier
    and the debounce, with no hand-travel time mixed in.
    """
    return labkit.hand_landmarks(cx=0.5, cy=0.6, sign=_scripted_sign(t))


def _script(t, i):
    """
    Publish the ground truth alongside the payload so the game can measure the
    gap between the hand changing shape and `hand_sign` admitting it. The game's
    own round-response figure cannot do this: the scripted hand cycles on its own
    schedule rather than obeying the prompt, so that number measures luck.
    """
    return {"_script_sign": _scripted_sign(t)}


SignDuel.headless_landmarks = staticmethod(_landmarks)
SignDuel.headless_script = staticmethod(_script)


if __name__ == "__main__":
    raise SystemExit(labkit.run(SignDuel()))
