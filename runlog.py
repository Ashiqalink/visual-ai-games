"""
runlog.py - where a game's offline run logs live, and whether it may write them.

Sling, flappy and punchy each carry a `tracker.py` recording a run to JSON
Lines under `<game>/data/`, and each had grown its own byte-identical copy of
the parts that are not about the game at all: the opt-in gate, the marker file,
the dated log path, and reading the lines back.

`RunStore` is those parts and nothing else. It never sees a record's contents -
it takes a dict and writes `json.dumps(record)` - so what each game logs, and
the schema its report reads, stay entirely that game's business. That is
deliberate rather than incidental: Sling's log is the instrument for an open
investigation into aim drift on release, and a shared base that touched record
shape could invalidate every run already on disk.

The `RunTracker` classes themselves are NOT shared. They differ in every
method - what a frame is, what an event is, what ends a run - and the only
common ground is four lines of bookkeeping in `__init__`. A base class over
that would buy nothing and put the record-building code one indirection away
from the game it describes.

Everything here is local and stays local: standard library only, no socket, no
network. `data/` is gitignored in every game, so a run log cannot be committed
by accident and the opt-in marker cannot travel with a copy of the game.
"""

from __future__ import annotations

import io
import json
import os
import time


def marker_path(data_dir: str) -> str:
    """Where the opt-in marker sits for a given data directory."""
    return os.path.join(data_dir, "tracking-enabled")


class RunStore:
    """
    One game's run-log directory, plus the opt-in that guards it.

    `env_var` is the per-game environment override - `SLING_TRACKING`,
    `FLAPPY_TRACKING`, `PUNCHY_TRACKING`. `data_dir` is the game's own
    `data/`, which is where the marker file lives too, so that turning
    tracking on for a machine cannot be committed.

    Cheap to construct, and meant to be: each `tracker.py` builds one per call
    from its own module-level `DATA_DIR` / `OPT_IN_MARKER`, so redirecting
    those at runtime still redirects everything. That is exactly how each
    `test_tracker.py` keeps its synthetic runs out of the player's real log.
    """

    def __init__(self, data_dir: str, env_var: str, marker: str | None = None):
        self.data_dir = data_dir
        self.env_var = env_var
        self.marker = marker_path(data_dir) if marker is None else marker

    # -- opt-in ------------------------------------------------------------

    def enabled(self) -> bool:
        """
        True only where run tracking has been deliberately switched on.

        Off by default, everywhere. Tracking is a development instrument for
        the machine doing the tuning, not something that should follow a copy
        of the game to whoever else runs it.

        `<GAME>_TRACKING=1` turns it on for one run; creating the marker file
        turns it on for this machine. An explicit `<GAME>_TRACKING=0` wins over
        the marker, so a single run can always be kept out of the log.
        """
        env = os.environ.get(self.env_var, "").strip().lower()
        if env in ("0", "false", "no", "off"):
            return False
        if env in ("1", "true", "yes", "on"):
            return True
        return os.path.exists(self.marker)

    def enable_here(self, note: str = "") -> str:
        """Switch tracking on for this machine by writing the marker."""
        os.makedirs(self.data_dir, exist_ok=True)
        with io.open(self.marker, "w", encoding="utf-8") as fh:
            fh.write("Run tracking is on for this machine.\n"
                     "Delete this file to turn it off. It is gitignored and never "
                     "travels with the game.\n" + (note and note + "\n"))
        return self.marker

    def disable_here(self) -> bool:
        """Switch tracking off for this machine. Existing logs are left alone."""
        if os.path.exists(self.marker):
            os.remove(self.marker)
            return True
        return False

    # -- files -------------------------------------------------------------

    def append(self, record: dict) -> str:
        """
        Append one run to today's log and return the path it went to.

        The record is written exactly as handed over. Callers own their schema.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(
            self.data_dir,
            "runs-%s.jsonl" % time.strftime("%Y-%m-%d", time.localtime()))
        with io.open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return path

    def load(self, paths=None) -> list:
        """Read run records back, oldest log first. Used by the report tools."""
        if paths is None:
            if not os.path.isdir(self.data_dir):
                return []
            paths = [os.path.join(self.data_dir, n)
                     for n in sorted(os.listdir(self.data_dir))
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
