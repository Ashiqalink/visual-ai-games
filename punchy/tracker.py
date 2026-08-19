"""Offline run tracker for ToF Punch.

The same instrument flappy carries, pointed at the thing punchy is actually
made of: depth. Everything here is local and stays local. It writes JSON Lines
to `punchy/data/runs-YYYY-MM-DD.jsonl` on this machine, sends nothing anywhere,
opens no socket, and imports nothing outside the standard library. The data
directory is gitignored, so a run log cannot be committed by accident.

It is also **off by default on every machine**. Tracking is a development
instrument for the machine doing the tuning, not something that should follow a
copy of the game to whoever else runs it, so it starts only where someone opted
in -- PUNCHY_TRACKING=1, or the marker file that enable_here() writes into the
(gitignored) data directory.

What it records is numbers only -- no camera frames, no landmarks, no images:

    per run     score, misses, how it ended, wall duration, frame count
    per frame   ToF depth in metres, whether the hand was tracked, whether the
                stabilizer was calibrating/active/off, frame time in ms
    per punch   the depth drop that triggered it, the baseline it was measured
                against, and how long the target had been alive
    per target  hit or missed, and how long it lived

The point is diagnosis, and punchy has one question worth asking above the
others: is `Z_PUNCH_THRESHOLD` in the right place? Too high and real punches do
nothing; too low and depth noise fires the game by itself. That is answerable
only with both halves of the picture -- how big the drops that *did* fire were,
and how big the depth wobble is when nobody is punching. `tracker_report.py`
reads these files back and puts those two numbers next to each other.

Frames are kept in memory and written once, at the end of a run, so nothing
does file IO inside the game loop.
"""

import io
import os
import json
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# One JSONL line per run. Every frame is kept: the noise floor is measured from
# the frame-to-frame depth series, so skipping frames would flatter it. Punchy
# runs at ~60 fps and stores five small numbers per frame -- a minute is ~250 KB.
FRAME_STRIDE = 1

# Belt and braces on top of the stride, so a forgotten session cannot grow a
# file without bound. 20000 frames is about 5.5 minutes of unbroken play.
MAX_FRAMES = 20000

# Opt-in marker. Tracking runs only on a machine that has this file, or that
# sets PUNCHY_TRACKING=1. The file lives inside the gitignored data directory,
# so it cannot be committed and cannot travel with a copy of the game.
OPT_IN_MARKER = os.path.join(DATA_DIR, "tracking-enabled")

# Stabilizer state as one small integer per frame rather than a repeated string.
STAB_CODES = {"inactive": 0, "sampling": 1, "active": 2}
STAB_NAMES = {v: k for k, v in STAB_CODES.items()}


def tracking_enabled():
    """True only where run tracking has been deliberately switched on.

    Off by default, everywhere. `PUNCHY_TRACKING=1` turns it on for one run;
    creating the marker file turns it on for this machine (see enable_here()).
    An explicit PUNCHY_TRACKING=0 wins over the marker, so a single run can
    always be kept out of the log.
    """
    env = os.environ.get("PUNCHY_TRACKING", "").strip().lower()
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


class RunTracker:
    """Collects one run and appends it to today's log when it ends.

    A run is one sitting: it starts when the help card is dismissed and ends on
    quit. Punchy has no lives and no game over, so unlike flappy nothing ends a
    run early -- `end()` is still idempotent, which is what makes it safe to
    call from both the quit path and any later death path.
    """

    def __init__(self, settings, enabled=True, started_at=None):
        self.enabled = enabled
        self.settings = dict(settings)
        self.started_at = started_at if started_at is not None else time.time()
        self._t0 = time.perf_counter()
        self._frame_index = 0
        self.frames = []      # [t_ms, tof_z, hand_visible, frame_ms, stab_code]
        self.punches = []     # one entry per landed punch
        self.targets = []     # one entry per target that resolved
        self.score = 0
        self.misses = 0
        self.written = False

    # -- collection --------------------------------------------------------

    def _now_ms(self):
        return round((time.perf_counter() - self._t0) * 1000.0, 1)

    def frame(self, tof_z, hand_visible, frame_ms, stab_state="inactive"):
        if not self.enabled:
            return
        i = self._frame_index
        self._frame_index += 1
        if i % FRAME_STRIDE or len(self.frames) >= MAX_FRAMES:
            return
        self.frames.append([
            self._now_ms(),
            round(float(tof_z), 4),
            1 if hand_visible else 0,
            round(float(frame_ms), 2),
            STAB_CODES.get(stab_state, 0),
        ])

    def punch(self, delta_z, baseline_z, target_age_ms, threshold):
        """Logged the frame a punch is detected. `delta_z` is how far the hand
        came forward against `baseline_z`; the margin over `threshold` is what
        says whether the trigger sits where the punches actually are."""
        if not self.enabled:
            return
        self.punches.append({
            "t_ms": self._now_ms(),
            "delta_z": round(float(delta_z), 4),
            "baseline_z": round(float(baseline_z), 4),
            "threshold": round(float(threshold), 4),
            "target_age_ms": round(float(target_age_ms), 1),
        })

    def target_resolved(self, outcome, age_ms):
        """`outcome` is 'hit' or 'miss'. Age is how long the target lived."""
        if not self.enabled:
            return
        if outcome == "hit":
            self.score += 1
        else:
            self.misses += 1
        self.targets.append({
            "t_ms": self._now_ms(),
            "outcome": outcome,
            "age_ms": round(float(age_ms), 1),
        })

    # -- writing -----------------------------------------------------------

    def end(self, cause, score=None, misses=None):
        """Close the run and append it. `cause` is 'quit' or 'stopped'."""
        if self.written or not self.enabled:
            return None
        self.written = True
        if score is not None:
            self.score = score
        if misses is not None:
            self.misses = misses
        record = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                        time.localtime(self.started_at)),
            "settings": self.settings,
            "score": self.score,
            "misses": self.misses,
            "cause": cause,
            "duration_s": round(time.perf_counter() - self._t0, 2),
            "frame_stride": FRAME_STRIDE,
            "frames": self.frames,
            "punches": self.punches,
            "targets": self.targets,
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
