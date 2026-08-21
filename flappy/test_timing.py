"""Motion must not depend on frame rate. Standard library plus the game:

    python test_timing.py

Two claims are checked. First, that the same wall-clock second of play moves
the pipes and the bird identically whether the loop runs at 20, 30, 60 or 240
fps -- the property the old per-frame constants did not have. Second, that the
converted numbers still reproduce the feel they replaced: at the ~32.5 fps the
loop actually ran at, the new per-second speeds and time constants land within
a couple of percent of the old per-frame ones.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flappy as F

MEASURED_FPS = 32.5          # what the logs showed before the loop was paced
OLD_SPEED = {"EASY": 5.0, "MEDIUM": 7.0, "HARD": 9.5}        # px per frame
OLD_FOLLOW = {"EASY": 0.35, "MEDIUM": 0.45, "HARD": 0.55}    # EMA per frame

SPAN = 0.1   # seconds of simulated play per comparison

print("frame-rate independence: 100 ms of play at four loop rates")
for diff in F.DIFFICULTIES:
    results = []
    for fps in (20, 30, 60, 240):
        dt = 1.0 / fps
        # Bird chasing a target it never reaches, and one pipe travelling.
        # 100 ms -- a whole number of frames at every rate tested, and short
        # enough that the bird is still mid-convergence. Sampling after it has
        # converged would match at every frame rate whether or not the maths
        # is right.
        bird, target, x = 0.0, 100.0, 1000.0
        for _ in range(int(SPAN * fps)):
            bird = F.approach(bird, target, diff["follow_tau"], dt)
            x -= diff["speed"] * dt
        results.append((fps, bird, x))
        assert 70.0 < bird < 95.0, "sample point should be mid-convergence"
    birds = [r[1] for r in results]
    xs = [r[2] for r in results]
    print(f"  {diff['name']:6s} bird {[f'{b:.6f}' for b in birds]}  "
          f"pipe x {[f'{x:.3f}' for x in xs]}")
    assert max(birds) - min(birds) < 1e-9, "bird position depends on frame rate"
    assert max(xs) - min(xs) < 1e-6, "pipe position depends on frame rate"

print("\nfeel preserved: new per-second numbers vs the old per-frame ones "
      f"at {MEASURED_FPS} fps")
dt = 1.0 / MEASURED_FPS
for diff in F.DIFFICULTIES:
    name = diff["name"]
    old_px_s = OLD_SPEED[name] * MEASURED_FPS
    speed_err = abs(diff["speed"] - old_px_s) / old_px_s

    # Fraction of the remaining distance covered in one frame at that rate.
    new_alpha = 1.0 - math.exp(-dt / diff["follow_tau"])
    follow_err = abs(new_alpha - OLD_FOLLOW[name]) / OLD_FOLLOW[name]

    print(f"  {name:6s} speed {old_px_s:6.1f} -> {diff['speed']:6.1f} px/s "
          f"({speed_err*100:4.1f}%)   follow {OLD_FOLLOW[name]:.3f} -> "
          f"{new_alpha:.3f} per frame ({follow_err*100:4.1f}%)")
    assert speed_err < 0.02, "scroll speed drifted from the tuning it replaced"
    assert follow_err < 0.02, "follow drifted from the tuning it replaced"

print("\nstall clamp")
assert F.MAX_FRAME_DT <= 0.2
travel = F.DIFFICULTIES[2]["speed"] * F.MAX_FRAME_DT
print(f"  worst single-frame pipe travel {travel:.1f} px, "
      f"pipe width {F.PIPE_WIDTH} px, bird diameter {F.BIRD_RADIUS * 2} px")
# A pipe must not clear the bird's own width in one clamped frame, or a stall
# could carry it past the collision test entirely.
assert travel < F.PIPE_WIDTH, "a stalled frame can tunnel a pipe through the bird"

print("\nOK")
