"""End-to-end check of the offline run tracker. Standard library only:

    python test_tracker.py

Ground truth is a synthetic aiming series with a release built into it: the
hand pulls the band down and back, then, over the last few frames, the aim is
rotated by a known number of degrees before the shot goes. The report has to
come back with that number and with its sign the right way round -- an aim that
falls before release must read positive, because "the bird goes lower than I
pointed" is the complaint this whole log exists to test, and a sign error would
turn the answer inside out without anyone noticing.

The band-lag estimator is checked the way flappy checks its own: against a band
series that is the hand series delayed by an exact number of samples.
"""
import sys, os, math, tempfile, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tracker, tracker_report

# Never touch the real Sling/data: the point of that directory is that it holds
# the user's own play history.
SCRATCH = tempfile.mkdtemp(prefix="sling-tracker-test-")
tracker.DATA_DIR = SCRATCH
tracker_report.DATA_DIR = SCRATCH

DT = 1 / 60.0
MS = DT * 1000
SLING = [260.0, 430.0]
GAIN = 1.5
SETTINGS = {"sling": SLING, "gain": GAIN, "min_fire_pull": 40,
            "max_pull": 140, "smooth_alpha": 0.2,
            "grip_release_openness": 0.35, "grip_release_frames": 2,
            "settle_frames": 8, "settle_radius": 18}
ARMED = tracker.STATE_CODES["ARMED"]


class FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


clock = FakeClock()
tracker.time.perf_counter = clock


def _band(angle_deg, pull):
    """Where the band sits for an aim of `angle_deg` at `pull` px.

    Inverse of tracker_report._aim: the launch direction is band -> slingshot,
    so a band at angle A sits on the opposite side of the slingshot from where
    the bird will go.
    """
    a = math.radians(angle_deg)
    return SLING[0] - pull * math.cos(a), SLING[1] + pull * math.sin(a)


def synth(level, shots, drift_deg, pull_drift=0.0, aim_deg=25.0, pull=110.0,
          lag_samples=0, dropout_at=None, frames_per_shot=120):
    """One run of `shots` shots, each ending with a `drift_deg` rotation of the
    aim and a `pull_drift` px change of the pull over the last 100 ms."""
    tr = tracker.RunTracker(level, SETTINGS, enabled=True)
    hand_hist = []
    score = 0
    for s in range(shots):
        for i in range(frames_per_shot):
            clock.t += DT
            # The release is the last 100 ms, which at 60 fps is the six
            # samples the report reaches back over: five run-up frames and the
            # shot itself. The sample it anchors on -- one frame before those,
            # `togo == 5` -- is deliberately still at the base aim, so the
            # answer the report should give is exactly `drift_deg`.
            togo = frames_per_shot - 1 - i
            if togo <= 4:
                f = (5 - togo) / 6.0
                angle = aim_deg - drift_deg * f      # aim falling = angle down
                length = pull + pull_drift * f
            elif togo <= 11:
                angle, length = aim_deg, pull        # steady before the release
            else:
                angle = aim_deg + 3.0 * math.sin(i * 0.09)
                length = pull + 4.0 * math.sin(i * 0.05)
            bx, by = _band(angle, length)
            # The hand is the band mapped back through the gain; the band is
            # optionally held back by a few samples, so the lag estimator has a
            # known answer to find.
            hx = (bx - SLING[0]) / GAIN + 400.0
            hy = (by - SLING[1]) / GAIN + 300.0
            hand_hist.append((bx, by))
            bx, by = hand_hist[max(0, len(hand_hist) - 1 - lag_samples)]
            visible = not (dropout_at and dropout_at <= i < dropout_at + 20)
            tr.frame((hx, hy), (bx, by), "ARMED", visible,
                     0.1 if togo > 6 else 0.5, MS)
        # The shot itself, one frame past the end of the run-up.
        clock.t += DT
        bx, by = _band(aim_deg - drift_deg, pull + pull_drift)
        tr.shot("open", "red", (400.0, 300.0),
                ((bx - SLING[0]) / GAIN + 400.0, (by - SLING[1]) / GAIN + 300.0),
                (bx, by), (6.0, -4.0), pull, GAIN, 320.0, s == 0)
        tr.hit(500)
        score += 500
        tr.flight_ended(score)
    tr.end("done", score)
    return tr


print("release drift: a known rotation must come back with the right sign")
for want in (0.0, 4.0, -3.0):
    for f in os.listdir(SCRATCH):
        os.remove(os.path.join(SCRATCH, f))
    synth(0, 3, drift_deg=want)
    run = tracker.load_runs()[-1]
    got = [tracker_report.release_drift(run, s)[0] for s in run["shots"]]
    print(f"  aim fell {want:+5.1f} deg -> reported "
          f"{', '.join('%+.2f' % g for g in got)}")
    assert all(abs(g - want) < 0.05 for g in got), "drift or its sign is wrong"

print("\npull drift: a collapsing pull must read negative")
for f in os.listdir(SCRATCH):
    os.remove(os.path.join(SCRATCH, f))
synth(1, 2, drift_deg=0.0, pull_drift=-18.0)
run = tracker.load_runs()[-1]
pulls = [tracker_report.release_drift(run, s)[1] for s in run["shots"]]
print(f"  pull -18 px -> reported {', '.join('%+.2f' % p for p in pulls)}")
assert all(abs(p + 18.0) < 0.5 for p in pulls)

print("\nband lag: known delay in samples")
for delay in (0, 2, 5):
    for f in os.listdir(SCRATCH):
        os.remove(os.path.join(SCRATCH, f))
    synth(0, 2, drift_deg=0.0, lag_samples=delay)
    run = tracker.load_runs()[-1]
    got = tracker_report.band_lag_ms(run)
    want = delay * MS
    print(f"  delay {delay} samples -> want {want:5.1f}ms  got {got:5.1f}ms")
    # Integer-frame estimate, so the only slack is the 0.1 ms the timestamps
    # are stored to -- which is what the per-sample interval is derived from.
    assert abs(got - want) < 0.05 * (delay + 1), "lag estimate off"

print("\nband error, dropouts, frame time")
for f in os.listdir(SCRATCH):
    os.remove(os.path.join(SCRATCH, f))
synth(2, 2, drift_deg=2.0, dropout_at=40)
run = tracker.load_runs()[-1]
err, p95 = tracker_report.band_error(run)
miss, longest = tracker_report.dropouts(run)
fm, fp95 = tracker_report.frame_ms(run)
print(f"  band err {err:.3f}/{p95:.3f}px  no hand {miss * 100:4.1f}%  "
      f"longest {longest:.0f}ms  frame {fm:.1f}/{fp95:.1f}ms")
# The synthetic hand IS the band mapped through the gain, so with no delay the
# reconstruction has to come back to within the 0.1 px the log stores positions
# to. Anything larger is an arithmetic slip in the report, not the data.
assert err < 0.2, "ideal-band reconstruction disagrees with the gain mapping"

# A pull past MAX_PULL is the band running out of travel, not the tracking
# missing: the game holds the band at the limit, and so must the ideal it is
# compared against. Without the clamp a hard pull reads as hundreds of pixels
# of error -- which is exactly what a first drive of the real game showed.
for f in os.listdir(SCRATCH):
    os.remove(os.path.join(SCRATCH, f))
synth(0, 1, drift_deg=0.0, pull=SETTINGS["max_pull"])   # sits on the limit
run = tracker.load_runs()[-1]
far = dict(run)
far["settings"] = dict(SETTINGS, max_pull=SETTINGS["max_pull"] / 2)
clamped, _ = tracker_report.band_error(far)
print(f"  pull held at half the limit -> err {clamped:.1f}px "
      f"(the travel it lost, not tracking error)")
assert clamped > 10, "the clamp must actually be applied"
assert tracker_report.band_error(run)[0] < 0.2, "at the limit, nothing clamps"
assert abs(miss - 2 * 20 / 240) < 0.01
assert abs(longest - 19 * MS) < 2
assert abs(fm - MS) < 0.05

print("\n--- report ---")
for f in os.listdir(SCRATCH):
    os.remove(os.path.join(SCRATCH, f))
synth(0, 3, drift_deg=3.5)
synth(0, 2, drift_deg=2.0, dropout_at=50)
synth(2, 4, drift_deg=1.0, pull_drift=-9.0)
tracker_report.report(tracker.load_runs(), show_runs=True, show_shots=True)

print("core bookkeeping")
for f in os.listdir(SCRATCH):
    os.remove(os.path.join(SCRATCH, f))
tr = tracker.RunTracker(0, SETTINGS, enabled=True)
for _ in range(50):
    clock.t += DT
    tr.frame((400, 300), SLING, "SELECTION", True, 0.1, MS)
tr.shot("edge", "red", (400, 300), (400, 300), SLING, (1, -1), 60, GAIN, 100, False)
tr.hit(500)
tr.hit(100)
assert tr.shots[-1] == dict(tr.shots[-1], points=600, hits=2)
# The run-up carried on a shot is capped, so a long hold cannot grow the record.
assert len(tr.shots[-1]["pre"]) == tracker.PRE_RELEASE_FRAMES
tr.end("quit", 600)
tr.end("quit", 600)                    # idempotent: no second line
off = tracker.RunTracker(0, SETTINGS, enabled=False)
for _ in range(50):
    off.frame((400, 300), SLING, "ARMED", True, 0.1, MS)
off.shot("open", "red", (400, 300), (400, 300), SLING, (1, -1), 60, GAIN, 100, False)
assert off.frames == [] and off.shots == []
off.end("quit", 0)
short = tracker.RunTracker(0, SETTINGS, enabled=True)
short.frame((400, 300), SLING, "ARMED", True, 0.1, MS)
short.end("quit", 0)                   # too short: dropped, not logged
n = len(tracker.load_runs())
print(f"runs on disk: {n} (expected 1)")
assert n == 1
# A run with no ARMED frames must not fall over on any of the aim metrics.
run = tracker.load_runs()[0]
assert tracker_report.band_lag_ms(run) is None
assert tracker_report.band_error(run) == (None, None)
assert tracker_report.release_drift(run, run["shots"][0]) == (None, None)
tracker_report.report(tracker.load_runs())
# NullTracker is what game.py holds until main.py hands over a real one; every
# call site has to be safe against it or tracking-off crashes the game.
null = tracker.NullTracker()
null.frame((0, 0), (0, 0), "ARMED", True, 0.0, 0.0)
null.shot("open", "red", (0, 0), (0, 0), (0, 0), (0, 0), 0, 1, 0, False)
null.hit(1)
null.flight_ended(0)
assert null.end("quit", 0) is None
print("core checks OK")

# ── Opt-in ───────────────────────────────────────────────────────────────────
# Tracking is a development instrument for one machine, not something that
# should follow a copy of the game to whoever else runs it.

marker = os.path.join(SCRATCH, "tracking-enabled")
tracker.OPT_IN_MARKER = marker
os.environ.pop("SLING_TRACKING", None)
assert tracker.tracking_enabled() is False, "must be off by default everywhere"

os.environ["SLING_TRACKING"] = "1"
assert tracker.tracking_enabled() is True
os.environ["SLING_TRACKING"] = "0"
assert tracker.tracking_enabled() is False
del os.environ["SLING_TRACKING"]

tracker.DATA_DIR = SCRATCH
tracker.enable_here()
assert os.path.exists(marker)
assert tracker.tracking_enabled() is True, "the marker opts this machine in"

os.environ["SLING_TRACKING"] = "0"       # one run kept out of the log
assert tracker.tracking_enabled() is False, "an explicit off beats the marker"
del os.environ["SLING_TRACKING"]

assert tracker.disable_here() is True
assert tracker.tracking_enabled() is False
assert tracker.disable_here() is False, "disabling twice is not an error"
print("opt-in: off by default, marker and env both respected")
shutil.rmtree(SCRATCH, ignore_errors=True)
print("OK")
