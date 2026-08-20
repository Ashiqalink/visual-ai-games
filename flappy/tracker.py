"""Offline run tracker for ToF Flappy.

Everything here is local and stays local. It writes JSON Lines to
`flappy/data/runs-YYYY-MM-DD.jsonl` on this machine, sends nothing anywhere,
opens no socket, and imports nothing outside the standard library. The data
directory is gitignored, so a run log cannot be committed by accident.

It is also **off by default on every machine**. Tracking is a development
instrument for the machine doing the tuning, not something that should follow
a copy of the game to whoever else runs it, so it starts only where someone
opted in -- FLAPPY_TRACKING=1, or the marker file that enable_here() writes
into the (gitignored) data directory.

What it records is numbers only -- no camera frames, no landmarks, no images:

    per run     difficulty, score, how it ended, wall duration, frame count
    per frame   target y (where the fingertip says the bird should be),
                bird y (where the bird actually is), whether the hand was
                tracked, frame time in ms
    per pipe    clearance -- how far off the gap centre the bird was as it
                crossed -- plus the gap and the gap centre it crossed

The point is diagnosis. The per-frame target/bird pair is what makes response
measurable after the fact: cross-correlating the two gives the follow lag in
milliseconds, and their difference gives the tracking error the player is
actually fighting. `tracker_report.py` reads these files back and does that.

Frames are kept in memory and written once, at the end of a run, so nothing
does file IO inside the game loop.
"""

import os
import sys
import time

# runlog.py lives at the repo root. tracker.py is imported both by the
# game (which has already put the root on sys.path) and by
# tracker_report.py / test_tracker.py, which have not, so it puts the
# root there itself rather than depending on who imported it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import runlog
from runlog import RunStore

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# One JSONL line per run. Frame series can be downsampled by this factor to
# shrink the file, but it is 1 on purpose: follow lag is measured by cross-
# correlating target against bird, so the sample interval is the resolution of
# the measurement. At 60 fps a stride of 2 would put the floor at 33 ms, which
# is the same order as the lag itself. A minute of play is ~400 KB.
FRAME_STRIDE = 1

# Belt and braces on top of the stride, so a forgotten session cannot grow a
# file without bound. 20000 frames is about 5.5 minutes of unbroken play.
MAX_FRAMES = 20000


# Opt-in marker. Tracking runs only on a machine that has this file, or that
# sets FLAPPY_TRACKING=1. The file lives inside the gitignored data directory,
# so it cannot be committed and cannot travel with a copy of the game.
OPT_IN_MARKER = runlog.marker_path(DATA_DIR)

_ENV_VAR = "FLAPPY_TRACKING"


# The store is built per call rather than once at import, so that patching
# DATA_DIR or OPT_IN_MARKER at runtime redirects the files too - which is how
# test_tracker.py keeps its synthetic runs out of the player's real log.
def _store():
    return RunStore(DATA_DIR, _ENV_VAR, marker=OPT_IN_MARKER)


# Thin wrappers, kept as module-level names because the game, the report tool
# and the tests all import them from here.
def tracking_enabled():
    """True only where run tracking has been deliberately switched on."""
    return _store().enabled()


def enable_here(note=""):
    """Switch tracking on for this machine by writing the marker."""
    return _store().enable_here(note)


def disable_here():
    """Switch tracking off for this machine. Existing logs are left alone."""
    return _store().disable_here()


def load_runs(paths=None):
    """Read run records back. Used by tracker_report.py."""
    return _store().load(paths)


class RunTracker:
    """Collects one run and appends it to today's log when it ends.

    A run is one life: it starts when the help card is dismissed or the game
    is restarted, and ends on a crash or on quit. `end()` is idempotent, so
    quitting mid-run and quitting after a crash both write exactly one line.
    """

    def __init__(self, diff, enabled=True, started_at=None):
        self.enabled = enabled
        self.diff = diff
        self.started_at = started_at if started_at is not None else time.time()
        self._t0 = time.perf_counter()
        self._frame_index = 0
        self.frames = []      # [t_ms, target_y, bird_y, hand_visible, frame_ms]
        self.pipes = []       # one entry per pipe crossed
        self.score = 0
        self.written = False

    # -- collection --------------------------------------------------------

    def frame(self, target_y, bird_y, hand_visible, frame_ms):
        if not self.enabled:
            return
        i = self._frame_index
        self._frame_index += 1
        if i % FRAME_STRIDE or len(self.frames) >= MAX_FRAMES:
            return
        self.frames.append([
            round((time.perf_counter() - self._t0) * 1000.0, 1),
            round(float(target_y), 1),
            round(float(bird_y), 1),
            1 if hand_visible else 0,
            round(float(frame_ms), 2),
        ])

    def pipe_cleared(self, index, bird_y, gap_centre, gap):
        """Logged as the bird passes a pipe. Clearance is signed: positive
        means the bird crossed below the gap centre."""
        if not self.enabled:
            return
        self.score = index
        self.pipes.append({
            "index": index,
            "clearance": round(float(bird_y - gap_centre), 1),
            "gap": gap,
            "gap_centre": round(float(gap_centre), 1),
            "t_ms": round((time.perf_counter() - self._t0) * 1000.0, 1),
        })

    # -- writing -----------------------------------------------------------

    def end(self, cause, score=None):
        """Close the run and append it. `cause` is 'pipe', 'ceiling',
        'floor', 'quit', or 'switch' (difficulty changed mid-run)."""
        if self.written or not self.enabled:
            return None
        self.written = True
        if score is not None:
            self.score = score
        record = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                        time.localtime(self.started_at)),
            "difficulty": self.diff["name"],
            "settings": {k: self.diff[k] for k in
                         ("speed", "gap", "spacing", "follow_tau",
                          "ramp_pipes")},
            "score": self.score,
            "cause": cause,
            "duration_s": round(time.perf_counter() - self._t0, 2),
            "frame_stride": FRAME_STRIDE,
            "frames": self.frames,
            "pipes": self.pipes,
        }
        # A run with nothing in it (card dismissed and immediately quit) is
        # noise in the report -- drop it rather than logging an empty shell.
        if len(self.frames) < 10:
            return None
        try:
            return _store().append(record)
        except OSError as exc:
            # Diagnostics must never take the game down with them.
            print(f"[tracker] could not write run log: {exc}")
            return None
