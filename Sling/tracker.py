"""Offline run tracker for Sling.

The same instrument flappy carries, pointed at the thing Sling is actually made
of: the aim, and what happens to it in the last few frames before the hand
opens. Everything here is local and stays local. It writes JSON Lines to
`Sling/data/runs-YYYY-MM-DD.jsonl` on this machine, sends nothing anywhere,
opens no socket, and imports nothing outside the standard library. The data
directory is gitignored, so a run log cannot be committed by accident.

It is also **off by default on every machine**. Tracking is a development
instrument for the machine doing the tuning, not something that should follow a
copy of the game to whoever else runs it, so it starts only where someone opted
in -- SLING_TRACKING=1, or the marker file that enable_here() writes into the
(gitignored) data directory.

What it records is numbers only -- no camera frames, no landmarks, no images:

    per run     level, score, how it ended, wall duration, frame count
    per frame   hand position, where the band is being held, which state the
                machine is in, whether the hand was tracked, grip openness,
                frame time in ms
    per shot    the anchor the pull was measured from, the hand position at
                release, the pull vector, the launch velocity, what caused the
                fire (an opened hand, a lost hand, or the screen edge), how
                long READY took to settle -- and the last few hundred
                milliseconds of hand and band positions leading up to the shot
    per hit     which shot it belonged to and what it scored

The last of those is the point. Sling's long-standing complaint is that the aim
drifts as the hand opens -- the shot leaves lower than where it was pointed --
and that is not something a player can measure from the chair. The pre-release
window makes it a number: `tracker_report.py` reads the band positions back and
prints how far the aim angle moved over the 100 ms before the bird left.

Frames are kept in memory and written once, at the end of a run, so nothing
does file IO inside the game loop.
"""

import io
import os
import json
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# One JSONL line per run. Every frame is kept: the release-drift measurement
# reads the frames immediately before a shot, and there are only a handful of
# them -- a stride would throw away exactly the samples the log exists for.
FRAME_STRIDE = 1

# Belt and braces on top of the stride, so a forgotten session cannot grow a
# file without bound. 20000 frames is about 5.5 minutes of unbroken play.
MAX_FRAMES = 20000

# How much of the run-up to a shot is stored with it. 24 frames is ~400 ms at
# 60 fps, comfortably longer than the ~100 ms window the drift is measured over
# so the report has room either side of it.
PRE_RELEASE_FRAMES = 24

# Opt-in marker. Tracking runs only on a machine that has this file, or that
# sets SLING_TRACKING=1. The file lives inside the gitignored data directory,
# so it cannot be committed and cannot travel with a copy of the game.
OPT_IN_MARKER = os.path.join(DATA_DIR, "tracking-enabled")

# The state machine as one small integer per frame rather than a repeated
# string. Unknown states fall back to -1 rather than raising: a diagnostic must
# not be the thing that crashes the game when someone adds a state.
STATE_CODES = {"SELECTION": 0, "READY": 1, "ARMED": 2, "FLIGHT": 3,
               "DONE": 4, "WIN": 5}
STATE_NAMES = {v: k for k, v in STATE_CODES.items()}


def tracking_enabled():
    """True only where run tracking has been deliberately switched on.

    Off by default, everywhere. `SLING_TRACKING=1` turns it on for one run;
    creating the marker file turns it on for this machine (see enable_here()).
    An explicit SLING_TRACKING=0 wins over the marker, so a single run can
    always be kept out of the log.
    """
    env = os.environ.get("SLING_TRACKING", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    return os.path.exists(OPT_IN_MARKER)


def enable_here(note=""):
    """Switch tracking on for this machine by writing the marker."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with io.open(OPT_IN_MARKER, "w", encoding="utf-8") as fh:
        fh.write("Run tracking is on for this machine.\n"
                 "Delete this file to turn it off. It is gitignored and never "
                 "travels with the game.\n" + (note and note + "\n"))
    return OPT_IN_MARKER


def disable_here():
    """Switch tracking off for this machine. Existing logs are left alone."""
    if os.path.exists(OPT_IN_MARKER):
        os.remove(OPT_IN_MARKER)
        return True
    return False


class NullTracker:
    """What `Game` holds until a real tracker is handed to it.

    game.py calls into the tracker from the middle of the state machine, and
    the alternative to this class is an `if self.tracker` around every call
    site -- five guards protecting a diagnostic. This keeps the game code
    reading as if tracking were always on, which is also what stops a call
    being added without one.
    """

    enabled = False
    frames = ()

    def frame(self, *a, **k):
        pass

    def shot(self, *a, **k):
        pass

    def hit(self, *a, **k):
        pass

    def flight_ended(self, *a, **k):
        pass

    def end(self, *a, **k):
        return None


class RunTracker:
    """Collects one run and appends it to today's log when it ends.

    A run is one attempt at one level: it starts when the help card is
    dismissed and ends on WIN, on DONE (birds exhausted), on restart, on a
    level switch, or on quit. `end()` is idempotent, so quitting on the DONE
    screen writes one line, not two.
    """

    def __init__(self, level_idx, settings, enabled=True, started_at=None):
        self.enabled = enabled
        self.level_idx = level_idx
        self.settings = dict(settings)
        self.started_at = started_at if started_at is not None else time.time()
        self._t0 = time.perf_counter()
        self._frame_index = 0
        # [t_ms, hand_x, hand_y, band_x, band_y, state, visible, openness, ms]
        self.frames = []
        self.shots = []
        self.score = 0
        self.written = False
        # Rolling window of the frames the report needs to see a release in
        # slow motion. Kept separately from `frames` so a shot carries its own
        # run-up even once MAX_FRAMES has stopped the main series growing.
        self._recent = []

    # -- collection --------------------------------------------------------

    def _now_ms(self):
        return round((time.perf_counter() - self._t0) * 1000.0, 1)

    def frame(self, hand_pos, band_pos, state, hand_visible, openness,
              frame_ms):
        if not self.enabled:
            return
        row = [
            self._now_ms(),
            round(float(hand_pos[0]), 1),
            round(float(hand_pos[1]), 1),
            round(float(band_pos[0]), 1),
            round(float(band_pos[1]), 1),
            STATE_CODES.get(state, -1),
            1 if hand_visible else 0,
            round(float(openness), 3),
            round(float(frame_ms), 2),
        ]
        self._recent.append(row)
        if len(self._recent) > PRE_RELEASE_FRAMES:
            self._recent.pop(0)
        i = self._frame_index
        self._frame_index += 1
        if i % FRAME_STRIDE or len(self.frames) >= MAX_FRAMES:
            return
        self.frames.append(row)

    def shot(self, cause, bird_kind, anchor, hand, band, velocity, pull_dist,
             gain, ready_ms, settle_forced):
        """Logged the instant ARMED becomes FLIGHT.

        `cause` is 'open' (the hand opened, the normal path), 'lost' (the hand
        left the tracker mid-pull and the pull was fired rather than stranded)
        or 'edge' (the pull left the frame). Those three are worth separating:
        only the first is a shot the player timed, and a log dominated by the
        other two is describing a tracking problem, not an aiming one.
        """
        if not self.enabled:
            return
        self.shots.append({
            "index": len(self.shots),
            "t_ms": self._now_ms(),
            "cause": cause,
            "bird": bird_kind,
            "anchor": [round(float(anchor[0]), 1), round(float(anchor[1]), 1)],
            "hand": [round(float(hand[0]), 1), round(float(hand[1]), 1)],
            "band": [round(float(band[0]), 1), round(float(band[1]), 1)],
            "velocity": [round(float(velocity[0]), 3),
                         round(float(velocity[1]), 3)],
            "pull_dist": round(float(pull_dist), 1),
            "gain": round(float(gain), 2),
            "ready_ms": round(float(ready_ms), 1),
            "settle_forced": bool(settle_forced),
            "points": 0,
            "hits": 0,
            # The run-up, so a release can be replayed frame by frame.
            "pre": list(self._recent),
        })

    def hit(self, points):
        """A block hit during the current flight."""
        if not self.enabled or not self.shots:
            return
        self.shots[-1]["points"] += int(points)
        self.shots[-1]["hits"] += 1

    def flight_ended(self, score):
        """FLIGHT is over. `score` is the running total, used only to keep the
        run's score current if the game ends without a further event."""
        if not self.enabled:
            return
        self.score = int(score)

    # -- writing -----------------------------------------------------------

    def end(self, cause, score=None):
        """Close the run and append it. `cause` is 'win', 'done', 'restart',
        'switch' (level changed mid-run) or 'quit'."""
        if self.written or not self.enabled:
            return None
        self.written = True
        if score is not None:
            self.score = int(score)
        record = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                        time.localtime(self.started_at)),
            "level": self.level_idx,
            "settings": self.settings,
            "score": self.score,
            "cause": cause,
            "duration_s": round(time.perf_counter() - self._t0, 2),
            "frame_stride": FRAME_STRIDE,
            "frames": self.frames,
            "shots": self.shots,
        }
        # A run with nothing in it (card dismissed and immediately quit) is
        # noise in the report -- drop it rather than logging an empty shell.
        if len(self.frames) < 10:
            return None
        try:
            return _append(record)
        except OSError as exc:
            # Diagnostics must never take the game down with them.
            print(f"[tracker] could not write run log: {exc}")
            return None


def _append(record):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(
        DATA_DIR,
        "runs-%s.jsonl" % time.strftime("%Y-%m-%d", time.localtime()))
    with io.open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return path


def load_runs(paths=None):
    """Read run records back. Used by tracker_report.py."""
    if paths is None:
        if not os.path.isdir(DATA_DIR):
            return []
        paths = [os.path.join(DATA_DIR, n) for n in sorted(os.listdir(DATA_DIR))
                 if n.startswith("runs-") and n.endswith(".jsonl")]
    runs = []
    for path in paths:
        with io.open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    runs.append(json.loads(line))
                except ValueError:
                    # A run interrupted mid-write leaves a partial last line.
                    print(f"[tracker] skipping unreadable line "
                          f"{os.path.basename(path)}:{line_no}")
    return runs
