"""
labkit.py — shared scaffolding for the seven limit-test games.

These seven exist to push a specific part of the SDK until it complains, which
only works if the complaint is measurable. Every one of them therefore runs in
two modes off the same code:

    python <game>.py                 windowed, real camera, a person plays it
    python <game>.py --headless 600  no window, scripted input, prints JSON

A windowed run opens on a how-to-play card built from the game's own `goal`,
`controls` and `keys` — these are gesture games, and a player who does not know
which shape their hand should be in has no way to discover it. Any key starts,
`H` brings it back, `--no-instructions` skips it. Headless runs never show it.

It also carries a `HAND TRACKING` badge in the bottom-right corner for the same
reason: a game that ignores you and a game that cannot see your hand look the
same from the player's chair, so the driver says which one it is — including
when there is no camera at all and the pipeline is quietly simulating.

The headless mode is the point. It drives the game from `ScriptedPipeline`,
which speaks the same queue contract as `VisionPipeline` but emits payloads from
a deterministic motion script, so a run is repeatable and a regression shows up
as a number rather than as "it felt worse". Frame times are collected either way.

Sling, flappy and punchy do not use any of this — they predate it and are left
alone.

Scope: this is game-side only. Nothing here belongs in the engine SDK; it is the
harness the games are measured with, not part of the tracking contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import statistics
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine_bootstrap import ensure_engine  # noqa: E402

ensure_engine()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from visual_ai import VisionPipeline  # noqa: E402

# The how-to-play card lives outside the harness so that Sling, flappy and
# punchy — which are no part of it — can show the same one.
from instructions import draw_card, draw_hint, wrap  # noqa: E402,F401

# ── Drawing ───────────────────────────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_text(img, text, x, y, size=0.7, color=(235, 235, 235), thickness=2,
              shadow=True):
    """Text with a dark outline so it stays legible over a camera feed."""
    if shadow:
        cv2.putText(img, text, (int(x), int(y)), FONT, size, (0, 0, 0),
                    thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (int(x), int(y)), FONT, size, color, thickness, cv2.LINE_AA)


def draw_panel(img, x, y, w, h, alpha=0.55, color=(18, 14, 24)):
    """Translucent backing plate. Cheap, and keeps HUDs readable over anything."""
    x, y = max(0, int(x)), max(0, int(y))
    w, h = int(w), int(h)
    region = img[y:y + h, x:x + w]
    if region.size == 0:
        return
    plate = np.full(region.shape, color, dtype=np.uint8)
    img[y:y + h, x:x + w] = cv2.addWeighted(region, 1.0 - alpha, plate, alpha, 0)


def draw_bar(img, x, y, w, h, fraction, color=(90, 220, 120), bg=(60, 60, 70)):
    fraction = max(0.0, min(1.0, float(fraction)))
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), bg, -1)
    if fraction > 0:
        cv2.rectangle(img, (int(x), int(y)),
                      (int(x + w * fraction), int(y + h)), color, -1)
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (140, 140, 150), 1)


def vgradient(w, h, top=(38, 20, 16), bottom=(10, 8, 14)):
    """Vertical gradient background, built once and copied per frame."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / max(1, h - 1)
        canvas[y, :] = [int(top[c] + (bottom[c] - top[c]) * t) for c in range(3)]
    return canvas


# ── Instructions card ─────────────────────────────────────────────────────────

#: Keys the driver handles for every game, so no game has to list them itself.
DRIVER_KEYS = (
    ("H", "show this card again"),
    ("K", "landmark smoothing on / off"),
    ("L", "calibrate ToF depth (hold still 3 s)"),
    ("X", "cancel a calibration"),
    ("Q / ESC", "quit"),
)




# ── Tracking status ───────────────────────────────────────────────────────────

class TrackingStatus:
    """
    Whether hand tracking is actually live, tracked frame by frame.

    A gesture game that stops responding looks identical whether the hand left
    the frame, the lighting killed the detector, or the camera never opened and
    the pipeline quietly fell back to simulated mode. The player has no way to
    tell those apart from inside the game, so the driver says which it is.

    Three things are distinguished, because the fix for each is different:

    * no camera            — nothing to track; the pipeline is simulating
    * camera, no hand      — put a hand in frame
    * hand seen this frame — tracking is live

    The state is read off one clock — time since a payload last reported a hand
    — rather than off the last payload's hand count, which covers two cases at
    once. MediaPipe misses the odd frame at the edge of the view, and a badge
    that strobed on every one of those would be noise, so a hand counts as held
    for `HOLD` seconds after it was last seen. And the render loop spins faster
    than the 30 fps pipeline, so most frames drain an empty queue; reading the
    last payload's count instead would leave the badge stuck on ON if the
    pipeline thread stalled or died, which is precisely the failure the badge
    exists to show.
    """

    HOLD = 0.35             # seconds a hand stays "on" after its last sighting
    LOST_GRACE = 1.5        # seconds of no hand before "lost" becomes "off"

    #: state -> (label, BGR color)
    _BADGE = {
        "start": ("STARTING", (150, 150, 160)),
        "sim":   ("NO CAMERA", (215, 175, 120)),
        "on":    ("ON", (110, 235, 130)),
        "lost":  ("LOST", (90, 200, 250)),
        "off":   ("OFF", (110, 110, 235)),
    }

    def __init__(self):
        self.saw_payload = False
        self.live_camera = False
        self.hand_count = 0
        # Starts at infinity, not zero: a hand that has never been seen must not
        # spend the first HOLD seconds of the session reading as tracked.
        self.since_seen = math.inf
        self.frames = 0
        self.seen_frames = 0

    def update(self, pipeline, payload, dt: float) -> None:
        """
        Fold one frame in. `payload` may be None when the queue had nothing new,
        which is not the same as "no hand" — the clock still advances, but the
        frame is not counted either way.
        """
        # Read the camera flag off the pipeline rather than off construction:
        # VisionPipeline only learns whether the camera opened once its thread
        # runs, so asking too early reports a failure that never happened.
        self.live_camera = bool(getattr(pipeline, "camera_available", False))
        self.since_seen += dt
        if payload is None:
            return

        self.saw_payload = True
        count = payload.get("hand_count")
        if count is None:
            count = 1 if payload.get("hand_visible") else 0
        self.hand_count = int(count)
        self.frames += 1
        if self.hand_count > 0:
            self.seen_frames += 1
            self.since_seen = 0.0

    @property
    def state(self) -> str:
        if not self.saw_payload:
            return "start"
        if not self.live_camera:
            return "sim"
        if self.since_seen < self.HOLD:
            return "on"
        return "lost" if self.since_seen < self.LOST_GRACE else "off"

    @property
    def tracked_fraction(self) -> float:
        return self.seen_frames / self.frames if self.frames else 0.0

    def badge(self) -> tuple[str, tuple[int, int, int]]:
        text, color = self._BADGE[self.state]
        if self.state == "on" and self.hand_count > 1:
            text = f"{text} x{self.hand_count}"
        return text, color


def draw_tracking_badge(img, status: TrackingStatus, margin: int = 14) -> None:
    """
    Bottom-right badge reading `HAND TRACKING <state>`.

    Bottom-right because it is the one corner none of the seven games draw in —
    the driver has no way to ask a game where its HUD is, so it has to pick a
    spot that is free in all of them.
    """
    label, color = status.badge()
    head = "HAND TRACKING"
    (hw, _), _ = cv2.getTextSize(head, FONT, 0.45, 1)
    (lw, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)

    dot_w, gap = 20, 10
    w = dot_w + hw + gap + lw + 16
    h = 28
    x = img.shape[1] - margin - w
    y = img.shape[0] - margin - h

    draw_panel(img, x, y, w, h, alpha=0.62, color=(16, 14, 22))
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                  tuple(int(c * 0.55) for c in color), 1)

    cy = int(y + h / 2)
    # Hollow while nothing is being tracked, so the state is readable at a
    # glance and without relying on color alone.
    cv2.circle(img, (int(x + 12), cy), 5, color,
               -1 if status.state == "on" else 1, cv2.LINE_AA)
    draw_text(img, head, x + dot_w + 4, cy + 5, 0.45, (185, 190, 200), 1)
    draw_text(img, label, x + dot_w + 4 + hw + gap, cy + 5, 0.5, color, 1)


# ── Frame timing ──────────────────────────────────────────────────────────────

class FrameClock:
    """
    Per-frame wall-clock timing plus a percentile summary.

    Mean fps hides exactly the thing that ruins a gesture game — the occasional
    80 ms frame — so p95 and worst are reported alongside it.
    """

    def __init__(self, keep: int = 20000):
        self.samples: list[float] = []
        self.keep = keep
        self._last = time.perf_counter()
        self.dt = 1.0 / 60.0

    def tick(self, record: bool = True) -> float:
        """
        Advance the clock. `record=False` still advances it but keeps the frame
        out of the summary — frames spent behind the instructions card are not
        frames of gameplay, and letting them into the percentiles would make the
        reported p95 a measure of how long the player read for.
        """
        now = time.perf_counter()
        self.dt = min(0.25, now - self._last)   # cap so a stall cannot teleport physics
        self._last = now
        if record and len(self.samples) < self.keep:
            self.samples.append(self.dt)
        return self.dt

    @property
    def fps(self) -> float:
        return 1.0 / self.dt if self.dt > 1e-6 else 0.0

    def summary(self) -> dict:
        if not self.samples:
            return {"frames": 0}
        ms = sorted(s * 1000.0 for s in self.samples)
        return {
            "frames": len(ms),
            "mean_ms": round(statistics.fmean(ms), 3),
            "median_ms": round(ms[len(ms) // 2], 3),
            "p95_ms": round(ms[min(len(ms) - 1, int(len(ms) * 0.95))], 3),
            "worst_ms": round(ms[-1], 3),
            "mean_fps": round(1000.0 / statistics.fmean(ms), 1),
        }


# ── Queue draining ────────────────────────────────────────────────────────────

def drain(q: queue.Queue):
    """
    Take the freshest payload and discard the rest, per the queue contract.

    Returns (payload_or_None, dropped_count). The drop count is worth keeping:
    a game that steadily drops frames is consuming slower than the pipeline
    produces, which no fps counter on the render side will tell you.
    """
    latest, dropped = None, 0
    while True:
        try:
            latest = q.get_nowait()
            dropped += 1
        except queue.Empty:
            break
    return latest, max(0, dropped - 1)


# ── Scripted (headless) input ─────────────────────────────────────────────────

class _LM:
    """Stand-in for a MediaPipe NormalizedLandmark."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _LandmarkSet:
    """Stand-in for a MediaPipe NormalizedLandmarkList."""

    __slots__ = ("landmark",)

    def __init__(self, landmarks):
        self.landmark = landmarks


#: Finger directions from the wrist, fanned out. (thumb handled separately)
_FINGER_DIRS = {
    "index":  (-0.25, -1.0),
    "middle": (0.00, -1.0),
    "ring":   (0.25, -1.0),
    "pinky":  (0.50, -1.0),
}
_FINGER_IDS = {                     # (mcp, pip, dip, tip)
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}
#: Which fingers are extended for each named sign (index, middle, ring, pinky)
_SIGN_SHAPES = {
    "fist":      (False, False, False, False, False),
    "point":     (False, True,  False, False, False),
    "peace":     (False, True,  True,  False, False),
    "open_palm": (True,  True,  True,  True,  True),
}


def hand_landmarks(cx=0.5, cy=0.6, sign="open_palm", scale=0.13, z=-0.02):
    """
    21 synthetic MediaPipe landmarks in normalised coords, shaped to classify as
    `sign`. Fingertips sit further from the wrist than their own PIP when
    extended and nearer when curled, which is the test the engine actually
    applies — so these drive the real classifier rather than faking its output.
    """
    thumb_ext, *four = _SIGN_SHAPES.get(sign, _SIGN_SHAPES["open_palm"])
    pts = [(cx, cy)] * 21
    s = scale

    r_pip, r_tip_ext, r_tip_curl = 1.1 * s, 1.8 * s, 0.75 * s
    for (name, extended) in zip(("index", "middle", "ring", "pinky"), four):
        dx, dy = _FINGER_DIRS[name]
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        mcp, pip, dip, tip = _FINGER_IDS[name]
        r_tip = r_tip_ext if extended else r_tip_curl
        pts[mcp] = (cx + ux * 0.55 * s, cy + uy * 0.55 * s)
        pts[pip] = (cx + ux * r_pip, cy + uy * r_pip)
        pts[dip] = (cx + ux * (r_pip + r_tip) * 0.5, cy + uy * (r_pip + r_tip) * 0.5)
        pts[tip] = (cx + ux * r_tip, cy + uy * r_tip)

    # Pinky MCP doubles as the palm-span anchor and the thumb-abduction
    # reference, so it is placed explicitly rather than left on the fan.
    pts[17] = (cx + 0.5 * s, cy - 0.5 * s)

    # Thumb: extension is measured as abduction away from the pinky MCP, not as
    # reach from the wrist — a folded thumb still clears its own MCP.
    tdx, tdy = -1.0, -0.4
    tnorm = math.hypot(tdx, tdy)
    tux, tuy = tdx / tnorm, tdy / tnorm
    pts[1] = (cx + tux * 0.35 * s, cy + tuy * 0.35 * s)
    pts[2] = (cx + tux * 0.65 * s, cy + tuy * 0.65 * s)
    pts[3] = (cx + tux * 0.90 * s, cy + tuy * 0.90 * s)
    pts[4] = ((cx + tux * 1.55 * s, cy + tuy * 1.55 * s) if thumb_ext
              else (cx + 0.10 * s, cy - 0.30 * s))

    pts[0] = (cx, cy)
    return [_LM(x, y, z) for (x, y) in pts]


class ScriptedPipeline:
    """
    A drop-in stand-in for VisionPipeline that emits scripted payloads.

    The payload template comes from a real (never started) VisionPipeline via
    `_empty_payload`, so the schema is the engine's own rather than a copy that
    silently rots. A script callable receives (t_seconds, frame_index) and
    returns a dict of keys to override.

    `landmark_script` is the stronger option: it returns synthetic landmarks
    which are pushed through the engine's real `_extract_gesture`, so the sign
    debounce, the One-Euro filters and the velocity difference are all genuinely
    exercised instead of being written over by the script.
    """

    def __init__(self, result_queue, width=960, height=720, script=None,
                 fps=30.0, emit_depth_grid=False, max_hands=2,
                 landmark_script=None):
        self.result_queue = result_queue
        self.width, self.height = width, height
        self.script = script or (lambda t, i: {})
        self.landmark_script = landmark_script
        self.fps = fps
        self.running = False
        self.frame_index = 0
        self.script_time = 0.0

        self._template_source = VisionPipeline(
            result_queue=queue.Queue(maxsize=1),
            width=width, height=height, camera_index=999, max_hands=max_hands,
        )
        self._template_source.tof_simulated = True
        # The stabilizer's calibration window is in seconds. Headless runs step
        # frames far faster than real time, so leaving it on the wall clock
        # makes a 1 s window cover hundreds of scripted frames � it would
        # calibrate its "hold still" baseline against whatever the script does
        # for the next 15 simulated seconds.
        self._template_source.tof_stabilizer.clock = lambda: self.script_time
        self._template_source.emit_depth_grid = emit_depth_grid
        self.tof_simulated = True
        self.emit_depth_grid = emit_depth_grid
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)

    # VisionPipeline API surface the games touch
    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def begin_stabilization(self, duration=3.0):
        self._template_source.begin_stabilization(duration)

    def cancel_stabilization(self):
        self._template_source.cancel_stabilization()

    def disable_stabilization(self):
        self._template_source.disable_stabilization()

    def toggle_smoothing(self):
        return self._template_source.toggle_smoothing()

    def set_filter_tuning(self, **kwargs):
        self._template_source.set_filter_tuning(**kwargs)

    def set_smoothing_enabled(self, enabled):
        self._template_source.set_smoothing_enabled(enabled)

    def step(self):
        """Produce exactly one payload. The driver calls this, not a thread."""
        t = self.script_time = self.frame_index / self.fps
        payload = self._template_source._empty_payload(
            self.width / 2.0, self.height / 2.0, self._frame)

        if self.landmark_script is not None:
            sets = self.landmark_script(t, self.frame_index)
            if sets:
                # Accept a single hand or a list of hands.
                if hasattr(sets[0], "x"):
                    sets = [sets]
                # Route through the engine's own slot matching rather than
                # trusting list order. Assigning by index here would quietly
                # bypass the exact mechanism a two-handed game needs tested.
                wrapped = [_LandmarkSet(lm) for lm in sets]
                assignment = self._template_source._assign_slots(wrapped)
                hands = []
                for mp_index, slot in sorted(assignment.items(), key=lambda kv: kv[1]):
                    entry = self._template_source._extract_gesture(
                        wrapped[mp_index].landmark, slot=slot, now=1000.0 + t)
                    entry["slot"] = slot
                    entry["handedness"] = "Left" if slot == 0 else "Right"
                    entry["handedness_score"] = 0.9
                    hands.append(entry)
                for slot in range(self._template_source.max_hands):
                    if slot not in set(assignment.values()):
                        gs = self._template_source._gs_slots[slot]
                        gs.lost_frames += 1
                        if gs.lost_frames > 12:
                            gs.reset()
                            self._template_source._slot_anchor[slot] = None
                payload.update(hands[0])
                payload["hands"] = tuple(hands)
                payload["hand_count"] = len(hands)
                payload["hand_left"] = hands[0] if hands else None
                payload["hand_right"] = hands[1] if len(hands) > 1 else None
                if self.emit_depth_grid:
                    payload["depth_grid"] = self._template_source._build_depth_grid(
                        hands, payload.get("face_box"))

        payload.update(self.script(t, self.frame_index) or {})

        # The stabilization clock is advanced from the capture loop in the real
        # pipeline, not from gesture extraction — a calibration started with no
        # hand in view would otherwise never finish. Mirror that here.
        self._template_source.tof_stabilizer.tick()

        # A scripted hand has to look like a real one to consumers that read
        # `hands` rather than the flat keys.
        if payload.get("hand_visible") and not payload.get("hands"):
            entry = dict(payload)
            entry.pop("frame", None)
            entry.pop("hands", None)
            payload["hands"] = (entry,)
            payload["hand_count"] = 1

        self.frame_index += 1
        if self.result_queue.full():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
        self.result_queue.put_nowait(payload)
        return payload


# ── Scripted motion helpers ───────────────────────────────────────────────────

def sine(t, period, low, high, phase=0.0):
    """Smooth oscillation between low and high with the given period (s)."""
    u = 0.5 * (1.0 + math.sin(2.0 * math.pi * t / period + phase))
    return low + (high - low) * u


def square(t, period, low, high):
    return high if (t % period) < period * 0.5 else low


def sawtooth(t, period, low, high):
    return low + (high - low) * ((t % period) / period)


# ── Game protocol + driver ────────────────────────────────────────────────────

class LabGame:
    """
    What the driver expects. Subclasses implement the three hooks.

    Splitting update from render is what makes headless mode meaningful: the
    headless run exercises all the game logic and skips only the pixels, so a
    crash in scoring or state transitions is caught without a display.
    """

    title = "Lab Game"
    width, height = 960, 720
    #: One sentence: what the player is trying to do. Shown on the card the game
    #: opens with, so every game explains itself without a README.
    goal = ""
    #: (input, what it does) pairs — the actual play instructions.
    controls: tuple[tuple[str, str], ...] = ()
    #: Game-specific keys. The driver's own keys (H/K/L/X/Q) are appended for
    #: every game, so do not repeat them here.
    keys: tuple[tuple[str, str], ...] = ()
    #: What the pipeline needs turned on for this game.
    wants_depth_grid = False
    wants_max_hands = 2
    #: Scripted input for headless runs: fn(t, frame_index) -> payload overrides
    headless_script = None
    #: Stronger scripted input: fn(t, frame_index) -> 21 landmarks (or a list of
    #: hands), pushed through the engine's real gesture extraction.
    headless_landmarks = None
    #: Frame at which the driver kicks off a ToF calibration, so a run with the
    #: stabilizer active can be compared against one without it.
    stabilize_at_frame = None
    #: Parsed CLI args, set by `run()` before the game starts.
    args = None

    def add_args(self, parser):
        """Hook for game-specific CLI flags. Called before parsing."""

    def update(self, payload: dict, dt: float) -> None:
        raise NotImplementedError

    def render(self, canvas: np.ndarray) -> None:
        raise NotImplementedError

    def on_key(self, key: int) -> bool:
        """Return False to quit."""
        return True

    def configure(self) -> None:
        """Called once after `args` is set, before anything is constructed."""

    def tune_pipeline(self, pipeline) -> None:
        """Called once with the live pipeline, before the first frame."""

    def metrics(self) -> dict:
        """Whatever this game measured. Merged into the headless JSON report."""
        return {}


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--headless", type=int, metavar="N", default=0,
                        help="run N frames off scripted input with no window, "
                             "then print a JSON report")
    parser.add_argument("--metrics", metavar="PATH",
                        help="also write the JSON report to PATH")
    parser.add_argument("--camera", type=int, default=None,
                        help="webcam index (the launcher's --camera also works)")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--no-camera", action="store_true",
                        help="windowed, but driven by the scripted input")
    parser.add_argument("--no-instructions", action="store_true",
                        help="skip the how-to-play card the game opens with")
    return parser


def run(game: LabGame, argv=None) -> int:
    """Drive a LabGame either windowed or headless."""
    parser = common_parser(game.title)
    game.add_args(parser)
    args = parser.parse_args(argv)
    game.args = args
    game.configure()
    if args.width:
        game.width = args.width
    if args.height:
        game.height = args.height

    if args.headless:
        return _run_headless(game, args)
    return _run_windowed(game, args)


def _run_headless(game: LabGame, args) -> int:
    q = queue.Queue(maxsize=1)
    pipeline = ScriptedPipeline(
        q, width=game.width, height=game.height,
        script=game.headless_script,
        landmark_script=game.headless_landmarks,
        emit_depth_grid=game.wants_depth_grid,
        max_hands=game.wants_max_hands,
    )
    game.tune_pipeline(pipeline)
    pipeline.start()

    clock = FrameClock()
    canvas = np.zeros((game.height, game.width, 3), dtype=np.uint8)
    dt = 1.0 / 30.0
    errors: list[str] = []

    # Render still runs headless. The point is to catch a crash in the draw
    # path, and to time it — a game that updates in 1 ms and renders in 40 is
    # not a tracking problem, and only a timed render pass distinguishes them.
    render_ms: list[float] = []

    for frame_no in range(args.headless):
        if game.stabilize_at_frame is not None and frame_no == game.stabilize_at_frame:
            pipeline.begin_stabilization(1.0)
        pipeline.step()
        payload, _dropped = drain(q)
        t0 = time.perf_counter()
        try:
            if payload is not None:
                game.update(payload, dt)
            canvas[:] = 0
            game.render(canvas)
        except Exception as exc:                     # noqa: BLE001 — report, keep going
            errors.append(f"{type(exc).__name__}: {exc}")
            if len(errors) > 5:
                break
        render_ms.append((time.perf_counter() - t0) * 1000.0)
        clock.samples.append(time.perf_counter() - t0)

    pipeline.stop()

    report = {
        "game": game.title,
        "frames": args.headless,
        "resolution": [game.width, game.height],
        "errors": errors,
        "ok": not errors,
        "update_render": clock.summary(),
        "stabilizer_requested": game.stabilize_at_frame is not None,
        "metrics": game.metrics(),
    }
    text = json.dumps(report, indent=2, default=_jsonable)
    print(text)
    if args.metrics:
        with open(args.metrics, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0 if not errors else 1


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _run_windowed(game: LabGame, args) -> int:
    q = queue.Queue(maxsize=1)

    if args.no_camera:
        pipeline = ScriptedPipeline(
            q, width=game.width, height=game.height, script=game.headless_script,
            landmark_script=game.headless_landmarks,
            emit_depth_grid=game.wants_depth_grid, max_hands=game.wants_max_hands)
    else:
        kwargs = dict(result_queue=q, width=game.width, height=game.height,
                      max_hands=game.wants_max_hands)
        if args.camera is not None:
            kwargs["camera_index"] = args.camera
        pipeline = VisionPipeline(**kwargs)
        pipeline.tof_simulated = True
        pipeline.emit_depth_grid = game.wants_depth_grid
    game.tune_pipeline(pipeline)
    pipeline.start()

    window = game.title
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, game.width, game.height)

    clock = FrameClock()
    canvas = np.zeros((game.height, game.width, 3), dtype=np.uint8)
    tracking = TrackingStatus()
    dropped_total = 0
    running = True

    # The card is up before the first frame of play. The pipeline runs behind it
    # regardless, so the camera has warmed up and the hand is already tracked by
    # the time the player dismisses it.
    showing_help = not args.no_instructions

    try:
        while running:
            if isinstance(pipeline, ScriptedPipeline):
                pipeline.step()
            payload, dropped = drain(q)
            dropped_total += dropped
            dt = clock.tick(record=not showing_help)
            # Updated behind the card too: the badge is how the player checks
            # the camera found their hand before play starts, which is exactly
            # when the card is still up.
            tracking.update(pipeline, payload, dt)

            if payload is not None and not showing_help:
                game.update(payload, dt)
            canvas[:] = 0
            game.render(canvas)

            draw_text(canvas, f"{clock.fps:5.1f} fps", game.width - 150, 30,
                      0.6, (150, 200, 255), 1)
            draw_tracking_badge(canvas, tracking)
            if showing_help:
                draw_card(canvas, game.title, game.goal, game.controls,
                          tuple(game.keys) + DRIVER_KEYS)
            else:
                draw_hint(canvas)
            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                running = False
            elif showing_help:
                # Any key starts the game. Reading the card is not gameplay, so
                # the clock is re-based before the first live frame rather than
                # handing the game a dt measured from when the card went up.
                if key != 255:
                    showing_help = False
                    clock.tick(record=False)
            elif key == ord('h'):
                showing_help = True
            elif key == ord('k'):
                pipeline.toggle_smoothing()
            elif key == ord('l'):
                pipeline.begin_stabilization(3.0)
            elif key == ord('x'):
                pipeline.cancel_stabilization()
            elif key != 255:
                running = game.on_key(key)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    summary = clock.summary()
    summary["dropped_frames"] = dropped_total
    # The same number the badge was showing, over the whole session — a run that
    # felt unresponsive is usually one that tracked a hand 40% of the time.
    summary["hand_tracked_pct"] = round(tracking.tracked_fraction * 100.0, 1)
    print(json.dumps({"game": game.title, "session": summary,
                      "metrics": game.metrics()}, indent=2, default=_jsonable))
    return 0
