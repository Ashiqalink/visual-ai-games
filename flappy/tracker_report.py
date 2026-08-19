"""Read the local ToF Flappy run logs back and diagnose them.

Offline like the tracker itself: reads `flappy/data/*.jsonl` from this machine
and prints to your terminal. Standard library only, so any interpreter runs it.

    python tracker_report.py                 # everything logged so far
    python tracker_report.py --last 10       # the 10 most recent runs
    python tracker_report.py --difficulty HARD
    python tracker_report.py --runs          # one line per run as well

The numbers worth reading first:

  follow lag      how many milliseconds the bird trails the fingertip. This is
                  the cross-correlation peak between the target series and the
                  bird series, so it measures the whole chain -- camera,
                  MediaPipe, one-euro smoothing, and the game's own follow
                  factor -- not just the last of those. Resolution is one
                  frame, ~17 ms at 60 fps.
  tracking error  mean |bird - target| in pixels while the hand is visible.
                  Lag shows up here too; what is left after lag is jitter.
  dropouts        share of frames with no hand, and the longest single gap.
                  A high number here explains deaths that felt unfair.
  clearance       how far off the gap centre the bird crossed each pipe, and
                  how that grows with pipe index -- the ramp working, or the
                  player running out of reach.
"""

import os
import sys
import math
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracker import load_runs, DATA_DIR

MAX_LAG_FRAMES = 30   # logged frames, i.e. 30 * stride real frames


def _series(run):
    """(times_ms, target, bird, visible, frame_ms) for one run."""
    frames = run.get("frames") or []
    cols = list(zip(*frames)) if frames else [[], [], [], [], []]
    return cols


def _corr(bc, tc, lag, energy):
    """Correlation of bird against target shifted by `lag` samples.

    Positive lag = the bird is reproducing where the finger was `lag` samples
    ago. Negative lags are evaluated too, not because the bird can lead the
    finger, but so a true lag of under one frame still has a sample on each
    side of the peak for the parabola below to fit.
    """
    acc = 0.0
    if lag >= 0:
        for i in range(lag, len(bc)):
            acc += bc[i] * tc[i - lag]
    else:
        for i in range(0, len(bc) + lag):
            acc += bc[i] * tc[i - lag]
    return acc / energy


def follow_lag_ms(run):
    """Cross-correlation peak between target and bird, in milliseconds.

    Both series are mean-centred first, so this measures how far the bird's
    *movement* trails the fingertip's, not any constant offset between them.
    Returns None when a run holds too little movement to say anything.
    """
    t_ms, target, bird, visible, _ = _series(run)
    if len(target) < 60:
        return None
    n = len(target)
    tm = statistics.fmean(target)
    bm = statistics.fmean(bird)
    tc = [v - tm for v in target]
    bc = [v - bm for v in bird]
    energy = math.sqrt(sum(v * v for v in tc) * sum(v * v for v in bc))
    if energy < 1e-6:
        return None

    lags = range(-2, min(MAX_LAG_FRAMES, n // 4))
    corrs = [_corr(bc, tc, lag, energy) for lag in lags]
    best = max(range(len(corrs)), key=lambda i: corrs[i])

    if len(t_ms) < 2:
        return None
    per_sample = (t_ms[-1] - t_ms[0]) / (len(t_ms) - 1)
    # Reported to the nearest whole frame -- about 17 ms at 60 fps. Fitting a
    # parabola through the peak for a sub-frame figure was tried and dropped:
    # tracking noise correlates only at the exact lag, so the peak sits as a
    # spike on a broad base and the fit came back biased by a third of a frame
    # even on a series with an exactly known delay. A coarse honest number
    # beats a precise wrong one, and the summary averages over runs anyway.
    return max(0.0, float(lags[best])) * per_sample


def tracking_error(run):
    """Mean and 95th-percentile |bird - target| over tracked frames."""
    _, target, bird, visible, _ = _series(run)
    errs = sorted(abs(b - t) for t, b, v in zip(target, bird, visible) if v)
    if not errs:
        return None, None
    p95 = errs[min(len(errs) - 1, int(0.95 * len(errs)))]
    return statistics.fmean(errs), p95


def dropouts(run):
    """(share of frames with no hand, longest gap in ms)."""
    t_ms, _, _, visible, _ = _series(run)
    if not visible:
        return 0.0, 0.0
    share = 1.0 - (sum(visible) / float(len(visible)))
    longest = run_len = 0
    start = None
    for i, v in enumerate(visible):
        if not v:
            if start is None:
                start = i
            run_len = t_ms[i] - t_ms[start]
            longest = max(longest, run_len)
        else:
            start = None
    return share, longest


def frame_ms(run):
    _, _, _, _, fms = _series(run)
    if not fms:
        return None, None
    ordered = sorted(fms)
    return (statistics.fmean(fms),
            ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))])


def clearance_by_stage(runs):
    """Mean |clearance| bucketed by pipe index, across the given runs."""
    buckets = {}
    for run in runs:
        for pipe in run.get("pipes") or []:
            key = min(4, pipe["index"] // 5)
            buckets.setdefault(key, []).append(abs(pipe["clearance"]))
    return buckets


def _fmt(value, spec="6.1f", none="   n/a"):
    return none if value is None else format(value, spec)


def report(runs, show_runs=False):
    if not runs:
        print(f"No runs logged yet. Play a round -- logs land in {DATA_DIR}")
        return

    print(f"\n{len(runs)} run(s) from {DATA_DIR}\n")

    by_diff = {}
    for run in runs:
        by_diff.setdefault(run["difficulty"], []).append(run)

    print("PER DIFFICULTY")
    print(f"  {'level':7s} {'runs':>4s} {'best':>5s} {'mean':>6s} "
          f"{'sec':>6s}  ended")
    for name in ("EASY", "MEDIUM", "HARD"):
        group = by_diff.get(name)
        if not group:
            continue
        scores = [r["score"] for r in group]
        causes = {}
        for r in group:
            causes[r["cause"]] = causes.get(r["cause"], 0) + 1
        cause_str = ", ".join(f"{k} {v}" for k, v in
                              sorted(causes.items(), key=lambda kv: -kv[1]))
        print(f"  {name:7s} {len(group):4d} {max(scores):5d} "
              f"{statistics.fmean(scores):6.1f} "
              f"{statistics.fmean([r['duration_s'] for r in group]):6.1f}  "
              f"{cause_str}")

    print("\nRESPONSE  (per difficulty, averaged over runs)")
    print(f"  {'level':7s} {'lag ms':>7s} {'err px':>7s} {'p95 px':>7s} "
          f"{'no hand':>8s} {'max gap':>8s} {'frame ms':>9s} {'p95':>6s}")
    for name in ("EASY", "MEDIUM", "HARD"):
        group = by_diff.get(name)
        if not group:
            continue

        def avg(fn):
            vals = [v for v in (fn(r) for r in group) if v is not None]
            return statistics.fmean(vals) if vals else None

        lag = avg(follow_lag_ms)
        err = avg(lambda r: tracking_error(r)[0])
        p95 = avg(lambda r: tracking_error(r)[1])
        miss = avg(lambda r: dropouts(r)[0])
        gap = avg(lambda r: dropouts(r)[1])
        fm = avg(lambda r: frame_ms(r)[0])
        fp95 = avg(lambda r: frame_ms(r)[1])
        print(f"  {name:7s} {_fmt(lag, '7.0f'):>7s} {_fmt(err, '7.1f'):>7s} "
              f"{_fmt(p95, '7.1f'):>7s} "
              f"{('%7.1f%%' % (miss * 100)) if miss is not None else '    n/a':>8s} "
              f"{_fmt(gap, '8.0f'):>8s} {_fmt(fm, '9.1f'):>9s} "
              f"{_fmt(fp95, '6.1f'):>6s}")

    print("\nCLEARANCE BY STAGE  (mean |bird - gap centre| at the crossing)")
    print("  a rising line is the gap-centre ramp biting; a flat one means the"
          "\n  ramp is not what is ending runs")
    for name in ("EASY", "MEDIUM", "HARD"):
        group = by_diff.get(name)
        if not group:
            continue
        buckets = clearance_by_stage(group)
        cells = []
        for key in range(5):
            vals = buckets.get(key)
            label = f"{key * 5}-{key * 5 + 4}"
            cells.append(f"{label}: {statistics.fmean(vals):5.1f}" if vals
                         else f"{label}:   -- ")
        print(f"  {name:7s} " + "  ".join(cells))

    if show_runs:
        print("\nRUNS")
        for run in runs:
            lag = follow_lag_ms(run)
            err, _ = tracking_error(run)
            miss, _ = dropouts(run)
            print(f"  {run['started_at']}  {run['difficulty']:6s} "
                  f"score {run['score']:3d}  {run['duration_s']:6.1f}s  "
                  f"{run['cause']:7s}  lag {_fmt(lag, '4.0f')}ms  "
                  f"err {_fmt(err, '5.1f')}px  no hand {miss * 100:4.1f}%")
    print()


def main(argv):
    show_runs = "--runs" in argv
    runs = load_runs()

    if "--difficulty" in argv:
        want = argv[argv.index("--difficulty") + 1].upper()
        runs = [r for r in runs if r["difficulty"] == want]
    if "--last" in argv:
        runs = runs[-int(argv[argv.index("--last") + 1]):]

    report(runs, show_runs=show_runs)


if __name__ == "__main__":
    main(sys.argv[1:])
