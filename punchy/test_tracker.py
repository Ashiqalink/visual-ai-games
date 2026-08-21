"""End-to-end check of the offline run tracker. Standard library only:

    python test_tracker.py

Ground truth here is a synthetic depth stream with a known noise amplitude and
punches of a known size, so the two numbers the report exists to compare can
both be checked against what went in: the measured noise floor must land near
the amplitude that was injected, and the trigger margins must match the punches
that were thrown. A run with punches far above the noise must report headroom
well over 1; a run where they overlap must report it under 1, because that is
the case the report is there to catch.
"""
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tracker
import tracker_report

# Never touch the real punchy/data: the point of that directory is that it
# holds the user's own play history.
SCRATCH = tempfile.mkdtemp(prefix="punchy-tracker-test-")
tracker.DATA_DIR = SCRATCH
tracker_report.DATA_DIR = SCRATCH

DT = 1 / 60.0
MS = DT * 1000
THRESHOLD = 0.02
Z_REST = 0.45


class FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


clock = FakeClock()
tracker.time.perf_counter = clock

SETTINGS = {"threshold": THRESHOLD, "z_history": 10, "cooldown_frames": 15,
            "target_frames": 100, "target_budget_ms": 1600.0,
            "tof_simulated": True}


def synth(noise, punch_depth, frames=1200, punch_every=150, seed=0,
          dropout_at=None):
    """A depth stream that sits at Z_REST with `noise` metres of wobble, with a
    `punch_depth` lunge toward the camera every `punch_every` frames.

    The punch is driven through the same detector the game runs -- furthest of
    the last nine samples against the newest -- so what gets logged is what the
    game would have logged, not an idealised version of it.
    """
    rnd = random.Random(seed)
    tr = tracker.RunTracker(SETTINGS, enabled=True)
    history, cooldown, spawn_ms = [], 0, 0.0
    for i in range(frames):
        clock.t += DT
        now_ms = i * MS
        phase = i % punch_every
        # A punch is four frames of travel in and eight of pull-back.
        if 0 <= phase < 4:
            lunge = punch_depth * (phase + 1) / 4.0
        elif phase < 12:
            lunge = punch_depth * (12 - phase) / 8.0
        else:
            lunge = 0.0
        z = Z_REST - lunge + rnd.gauss(0, noise)
        visible = not (dropout_at and dropout_at <= i < dropout_at + 40)

        if visible:
            history.append(z)
            if len(history) > 10:
                history.pop(0)
        else:
            history = []

        tr.frame(z if visible else Z_REST, visible, MS,
                 "active" if i > 200 else "inactive")

        if len(history) >= 3 and cooldown == 0:
            baseline = max(history[:-1])
            delta = baseline - history[-1]
            if delta >= THRESHOLD:
                tr.punch(delta, baseline, now_ms - spawn_ms, THRESHOLD)
                tr.target_resolved("hit", now_ms - spawn_ms)
                spawn_ms = now_ms
                cooldown = 15
                history = []
        if cooldown:
            cooldown -= 1
        if now_ms - spawn_ms > 1600.0:
            tr.target_resolved("miss", now_ms - spawn_ms)
            spawn_ms = now_ms
    tr.end("quit")
    return tr


print("clean stream: punches must sit clear of the noise")
synth(noise=0.0015, punch_depth=0.06, seed=1)
run = tracker.load_runs()[-1]
p99, worst, n_quiet = tracker_report.noise_floor(run)
margins = tracker_report.punch_margins(run)
print(f"  punches {len(margins)}  weakest margin {min(margins):+.4f}  "
      f"noise p99 {p99:.4f} worst {worst:.4f} over {n_quiet} quiet frames")
assert len(margins) == 8, "one punch per cycle, no double-fires"
assert min(margins) > 0, "a logged punch cleared the threshold by definition"
# Injected sigma is 0.0015; the floor is a max-over-nine of a Gaussian, so it
# lands around 4-6 sigma. Loose bounds -- the check is the order of magnitude.
assert 0.003 < p99 < 0.015, f"noise floor {p99} nowhere near the 0.0015 injected"
weakest_punch = min(margins) + THRESHOLD
assert weakest_punch / p99 > 2.0, "clean stream must show real headroom"
assert n_quiet > 400, "most of a quiet stream should be measurable"

for f in os.listdir(SCRATCH):
    os.remove(os.path.join(SCRATCH, f))

print("\nnoisy stream: the floor must reach the trigger and headroom collapse")
synth(noise=0.006, punch_depth=0.025, seed=2)
run = tracker.load_runs()[-1]
p99, worst, _ = tracker_report.noise_floor(run)
margins = tracker_report.punch_margins(run)
weakest_punch = min(margins) + THRESHOLD
print(f"  weakest punch {weakest_punch:.4f}  noise p99 {p99:.4f}  "
      f"headroom {weakest_punch / p99:.2f}x")
assert p99 > THRESHOLD * 0.5, "6 mm of wobble must show up against a 20 mm trigger"
assert weakest_punch / p99 < 1.5, "overlapping punch and noise must read as thin"

for f in os.listdir(SCRATCH):
    os.remove(os.path.join(SCRATCH, f))

print("\ndropouts and reaction")
synth(noise=0.0015, punch_depth=0.06, seed=3, dropout_at=400)
run = tracker.load_runs()[-1]
miss, longest = tracker_report.dropouts(run)
react, react_p95 = tracker_report.reaction(run)
fm, fp95 = tracker_report.frame_ms(run)
shares = tracker_report.stabilizer_share(run)
print(f"  no hand {miss * 100:4.1f}%  longest {longest:.0f}ms  "
      f"react {react:.0f}/{react_p95:.0f}ms  frame {fm:.1f}/{fp95:.1f}ms  "
      f"stab {shares}")
assert abs(miss - 40 / 1200) < 0.005
assert abs(longest - 39 * MS) < 2
assert react is not None and react > 0
assert abs(fm - MS) < 0.05
assert abs(shares["active"] - 999 / 1200) < 0.01
# No punch is credited to a frame with no hand -- the detector never saw one.
assert all(p["delta_z"] >= p["threshold"] for p in run["punches"])

print("\n--- report ---")
tracker_report.report(tracker.load_runs(), show_runs=True)

for f in os.listdir(SCRATCH):
    os.remove(os.path.join(SCRATCH, f))

print("core bookkeeping")
tr = tracker.RunTracker(SETTINGS, enabled=True)
for _ in range(50):
    clock.t += DT
    tr.frame(Z_REST, True, MS)
tr.target_resolved("hit", 300.0)
tr.target_resolved("miss", 1600.0)
assert (tr.score, tr.misses) == (1, 1), "outcomes are counted as they resolve"
tr.end("quit")
tr.end("quit")                         # idempotent: no second line
off = tracker.RunTracker(SETTINGS, enabled=False)
for _ in range(50):
    off.frame(Z_REST, True, MS)
off.target_resolved("hit", 100.0)
assert off.frames == [] and off.score == 0
off.end("quit")
short = tracker.RunTracker(SETTINGS, enabled=True)
short.frame(Z_REST, True, MS)
short.end("quit")                      # too short: dropped, not logged
n = len(tracker.load_runs())
print(f"runs on disk: {n} (expected 1)")
assert n == 1
# A run with no punches at all must not divide by anything it does not have.
tracker_report.report(tracker.load_runs())
print("core checks OK")

# ── Opt-in ───────────────────────────────────────────────────────────────────
# Tracking is a development instrument for one machine, not something that
# should follow a copy of the game to whoever else runs it.

marker = os.path.join(SCRATCH, "tracking-enabled")
tracker.OPT_IN_MARKER = marker
os.environ.pop("PUNCHY_TRACKING", None)
assert tracker.tracking_enabled() is False, "must be off by default everywhere"

os.environ["PUNCHY_TRACKING"] = "1"
assert tracker.tracking_enabled() is True
os.environ["PUNCHY_TRACKING"] = "0"
assert tracker.tracking_enabled() is False
del os.environ["PUNCHY_TRACKING"]

tracker.DATA_DIR = SCRATCH
tracker.enable_here()
assert os.path.exists(marker)
assert tracker.tracking_enabled() is True, "the marker opts this machine in"

os.environ["PUNCHY_TRACKING"] = "0"      # one run kept out of the log
assert tracker.tracking_enabled() is False, "an explicit off beats the marker"
del os.environ["PUNCHY_TRACKING"]

assert tracker.disable_here() is True
assert tracker.tracking_enabled() is False
assert tracker.disable_here() is False, "disabling twice is not an error"
print("opt-in: off by default, marker and env both respected")
shutil.rmtree(SCRATCH, ignore_errors=True)
print("OK")
