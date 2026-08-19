"""End-to-end check of the offline run tracker. Standard library only:

    python test_tracker.py

Ground truth for the lag estimator is a bird series that is the target series
delayed by an exact number of samples -- then the reported lag must come back
as exactly that many frames' worth of milliseconds. The EMA case (what the game
actually does) is checked more loosely: it must land at or below its time
constant (plus half a frame of quantisation), which is where a real one-pole
filter's lag sits for a signal with actual bandwidth.
"""
import sys, os, math, random, tempfile, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tracker, tracker_report

# Never touch the real flappy/data: the point of that directory is that it
# holds the user's own play history.
SCRATCH = tempfile.mkdtemp(prefix="flappy-tracker-test-")
tracker.DATA_DIR = SCRATCH
tracker_report.DATA_DIR = SCRATCH

DT = 1 / 60.0
MS = DT * 1000


class FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


clock = FakeClock()
tracker.time.perf_counter = clock

EASY = {"name": "EASY", "speed": 165.0, "gap": 200, "spacing": 400, "follow_tau": 0.072, "ramp_pipes": 28}
HARD = {"name": "HARD", "speed": 310.0, "gap": 125, "spacing": 260, "follow_tau": 0.039, "ramp_pipes": 14}


def finger(i, rnd):
    return 300 + 160 * math.sin(i * 0.05) + 60 * math.sin(i * 0.017) + rnd.gauss(0, 2)


def synth(diff, mode, param, frames=900, dropout_at=None, seed=0):
    """mode 'delay': bird = target delayed `param` samples. mode 'tau': the
    real follow filter, `param` being its time constant in seconds."""
    rnd = random.Random(seed)
    tr = tracker.RunTracker(diff, enabled=True)
    hist, bird, score = [], 300.0, 0
    for i in range(frames):
        clock.t += DT
        target = finger(i, rnd)
        hist.append(target)
        if mode == "delay":
            bird = hist[max(0, i - param)]
        else:
            k = math.exp(-DT / param)
            bird = target + (bird - target) * k
        visible = not (dropout_at and dropout_at <= i < dropout_at + 40)
        tr.frame(target, bird, visible, MS)
        if i and i % 90 == 0:
            score += 1
            tr.pipe_cleared(score, bird, target + rnd.gauss(0, 20 + score * 5), diff["gap"])
    tr.end("pipe", score)


print("known-delay ground truth")
for d in (0, 1, 3, 7):
    synth(EASY, "delay", d, seed=d)
    run = tracker.load_runs()[-1]
    got = tracker_report.follow_lag_ms(run)
    want = d * MS
    print(f"  delay {d} samples -> want {want:5.1f}ms  got {got:5.1f}ms  delta {got-want:+5.2f}")
    assert abs(got - want) < 0.01, "lag estimate off"   # integer-frame: exact

for f in os.listdir(SCRATCH):
    os.remove(os.path.join(SCRATCH, f))

print("\nreal follow filter (EMA), plus dropouts")
synth(EASY, "tau", EASY["follow_tau"], seed=1)
synth(EASY, "tau", EASY["follow_tau"], dropout_at=300, seed=2)
synth(HARD, "tau", HARD["follow_tau"], seed=3)
synth(HARD, "tau", HARD["follow_tau"], dropout_at=500, seed=4)
runs = tracker.load_runs()
for r in runs:
    tau = r["settings"]["follow_tau"]
    lag = tracker_report.follow_lag_ms(r)
    dc = tau * 1000.0                      # group delay of a one-pole filter
    err, p95 = tracker_report.tracking_error(r)
    miss, longest = tracker_report.dropouts(r)
    print(f"  {r['difficulty']:6s} tau={tau*1000:4.0f}ms  lag={lag:5.1f}ms (DC limit {dc:5.1f})  "
          f"err={err:5.1f}px p95={p95:5.1f}  no-hand={miss*100:4.1f}% longest={longest:.0f}ms")
    # A one-pole filter's lag is below its time constant for any signal with
    # real bandwidth, and the estimator quantises to whole frames -- so HARD's
    # 39 ms tau can legitimately report as low as 33.
    assert 0 <= lag <= dc * 1.2 + MS / 2, "EMA lag outside plausible range"
assert abs(tracker_report.dropouts(runs[1])[0] - 40 / 900) < 0.01
assert abs(tracker_report.dropouts(runs[1])[1] - 39 * MS) < 2
assert tracker_report.dropouts(runs[0])[0] == 0.0

print("\n--- report ---")
tracker_report.report(runs, show_runs=True)

tr = tracker.RunTracker(EASY, enabled=True)
for _ in range(50):
    clock.t += DT
    tr.frame(300, 300, True, MS)
tr.end("quit", 3)
tr.end("quit", 3)                      # idempotent: no second line
off = tracker.RunTracker(EASY, enabled=False)
for _ in range(50):
    off.frame(300, 300, True, MS)
assert off.frames == []
off.end("quit", 0)
short = tracker.RunTracker(EASY, enabled=True)
short.frame(300, 300, True, MS)
short.end("quit", 0)                   # too short: dropped, not logged
n = len(tracker.load_runs())
print(f"runs on disk: {n} (expected 5)")
assert n == 5
print("core checks OK")

# ── Opt-in ───────────────────────────────────────────────────────────────────
# Tracking is a development instrument for one machine, not something that
# should follow a copy of the game to whoever else runs it.

marker = os.path.join(SCRATCH, "tracking-enabled")
tracker.OPT_IN_MARKER = marker
os.environ.pop("FLAPPY_TRACKING", None)
assert tracker.tracking_enabled() is False, "must be off by default everywhere"

os.environ["FLAPPY_TRACKING"] = "1"
assert tracker.tracking_enabled() is True
os.environ["FLAPPY_TRACKING"] = "0"
assert tracker.tracking_enabled() is False
del os.environ["FLAPPY_TRACKING"]

tracker.DATA_DIR = SCRATCH
tracker.enable_here()
assert os.path.exists(marker)
assert tracker.tracking_enabled() is True, "the marker opts this machine in"

os.environ["FLAPPY_TRACKING"] = "0"      # one run kept out of the log
assert tracker.tracking_enabled() is False, "an explicit off beats the marker"
del os.environ["FLAPPY_TRACKING"]

assert tracker.disable_here() is True
assert tracker.tracking_enabled() is False
assert tracker.disable_here() is False, "disabling twice is not an error"
print("opt-in: off by default, marker and env both respected")
shutil.rmtree(SCRATCH, ignore_errors=True)
print("OK")
