"""Read this machine's ToF Punch run logs back and diagnose them.

Offline like the tracker itself: reads `punchy/data/*.jsonl` from this machine
and prints to your terminal. Standard library only, so any interpreter runs it.

    python tracker_report.py                 # everything logged so far
    python tracker_report.py --last 10       # the 10 most recent runs
    python tracker_report.py --runs          # one line per run as well
    python tracker_report.py --enable        # turn tracking on for THIS machine
    python tracker_report.py --disable       # and off again

Tracking is off by default on every machine. It records only where someone
opted in, and what it records never leaves the disk it was written on.

The numbers worth reading first:

  trigger margin  punchy fires when the depth drop against the last few frames
                  clears Z_PUNCH_THRESHOLD. This is how much the punches that
                  actually landed cleared it by. A small margin on the weakest
                  punch means real punches are being missed.
  noise floor     the same drop measured over stretches where nobody was
                  punching -- i.e. how big a drop the depth stream produces on
                  its own. It is the other half of the same question: the
                  threshold has to sit above this, or the game punches itself.
                  Read the two together; either alone says nothing.
  headroom        weakest landed punch divided by the noise floor. Under 1 the
                  two overlap and no threshold can separate them; the fix is
                  cleaner depth (calibrate the stabilizer with S), not tuning.
  reaction        how long a target had been alive when it was hit, against the
                  time it was given. Near the budget means the ring is too
                  tight, not that the player is slow.
  dropouts        share of frames with no hand, and the longest single gap. A
                  punch thrown during a gap never happened as far as the game
                  is concerned, which is what makes those misses feel unfair.
"""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracker import DATA_DIR, STAB_NAMES, disable_here, enable_here, load_runs, tracking_enabled

# The game keeps ten frames of depth history and measures each frame against
# the furthest of the previous nine. The noise floor has to be measured the
# same way or it is not comparable with the punches that fired.
WINDOW = 10

# A drop is only "noise" if it is nowhere near a punch. Half a second either
# side of one keeps the wind-up and the pull-back out of the quiet sample --
# both are real hand movement, and counting them would inflate the floor into
# meaninglessness.
QUIET_GUARD_MS = 500.0


def _series(run):
    """(times_ms, tof_z, visible, frame_ms, stab) for one run."""
    frames = run.get("frames") or []
    return list(zip(*frames)) if frames else [[], [], [], [], []]


def _pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def punch_margins(run):
    """How far past the threshold each landed punch went, in metres."""
    return [p["delta_z"] - p["threshold"] for p in (run.get("punches") or [])]


def noise_floor(run):
    """Largest depth drop the stream produces while nobody is punching.

    Computed exactly as the game computes a punch -- current sample against the
    furthest of the previous nine -- but only over frames with a tracked hand
    and no punch within QUIET_GUARD_MS. Returns (p99, worst, samples); p99
    rather than the maximum because one swallowed frame should not define the
    floor, and the maximum is printed beside it anyway.
    """
    t_ms, z, visible, _, _ = _series(run)
    if len(z) < WINDOW * 2:
        return None, None, 0
    punch_times = [p["t_ms"] for p in (run.get("punches") or [])]
    drops = []
    for i in range(WINDOW, len(z)):
        window = z[i - WINDOW:i]
        if not all(visible[i - WINDOW:i + 1]):
            continue
        now = t_ms[i]
        if any(abs(now - pt) <= QUIET_GUARD_MS for pt in punch_times):
            continue
        drops.append(max(window) - z[i])
    if not drops:
        return None, None, 0
    return _pct(drops, 0.99), max(drops), len(drops)


def dropouts(run):
    """(share of frames with no hand, longest gap in ms)."""
    t_ms, _, visible, _, _ = _series(run)
    if not visible:
        return 0.0, 0.0
    share = 1.0 - (sum(visible) / float(len(visible)))
    longest = 0.0
    start = None
    for i, v in enumerate(visible):
        if not v:
            if start is None:
                start = i
            longest = max(longest, t_ms[i] - t_ms[start])
        else:
            start = None
    return share, longest


def frame_ms(run):
    _, _, _, fms, _ = _series(run)
    if not fms:
        return None, None
    return statistics.fmean(fms), _pct(fms, 0.95)


def stabilizer_share(run):
    """Share of frames spent in each stabilizer state, by name."""
    _, _, _, _, stab = _series(run)
    if not stab:
        return {}
    total = float(len(stab))
    out = {}
    for code in set(stab):
        out[STAB_NAMES.get(code, str(code))] = stab.count(code) / total
    return out


def reaction(run):
    """(mean, p95) time-to-hit in ms, over targets that were hit."""
    ages = [t["age_ms"] for t in (run.get("targets") or [])
            if t["outcome"] == "hit"]
    if not ages:
        return None, None
    return statistics.fmean(ages), _pct(ages, 0.95)


def _fmt(value, spec="6.1f", none="   n/a"):
    return none if value is None else format(value, spec)


def _avg(runs, fn):
    vals = [v for v in (fn(r) for r in runs) if v is not None]
    return statistics.fmean(vals) if vals else None


def report(runs, show_runs=False):
    if not runs:
        print(f"No runs logged yet. Play a round -- logs land in {DATA_DIR}")
        return

    print(f"\n{len(runs)} run(s) from {DATA_DIR}\n")

    hits = sum(r["score"] for r in runs)
    misses = sum(r["misses"] for r in runs)
    resolved = hits + misses
    print("SESSIONS")
    print(f"  runs {len(runs)}   hits {hits}   misses {misses}   "
          f"hit rate {(100.0 * hits / resolved) if resolved else 0.0:.0f}%   "
          f"total {sum(r['duration_s'] for r in runs):.0f}s")

    thresholds = sorted({p["threshold"] for r in runs
                         for p in (r.get("punches") or [])})
    margins = [m for r in runs for m in punch_margins(r)]
    floors = [noise_floor(r) for r in runs]
    floor_p99 = [f[0] for f in floors if f[0] is not None]
    floor_max = [f[1] for f in floors if f[1] is not None]
    quiet_n = sum(f[2] for f in floors)

    print("\nTRIGGER  (all depths in metres)")
    if not margins:
        print("  no punches logged yet")
    else:
        thr = thresholds[0] if len(thresholds) == 1 else None
        thr_text = (_fmt(thr, '7.4f') if thr is not None
                    else 'varies: ' + ', '.join(f'{t:.4f}' for t in thresholds))
        print(f"  threshold        {thr_text}")
        print(f"  landed punches   {len(margins)}   "
              f"margin over threshold: weakest {min(margins):+.4f}  "
              f"median {statistics.median(margins):+.4f}  "
              f"strongest {max(margins):+.4f}")
    if floor_p99:
        mean_p99 = statistics.fmean(floor_p99)
        print(f"  noise floor      p99 {mean_p99:.4f}   worst {max(floor_max):.4f}"
              f"   ({quiet_n} quiet frames)")
        if thresholds:
            thr = thresholds[0]
            print(f"  vs threshold     the floor is "
                  f"{100.0 * mean_p99 / thr:.0f}% of the trigger "
                  f"-- above 100% the game can fire itself")
        if margins:
            weakest = min(margins) + (thresholds[0] if thresholds else 0.0)
            head = weakest / mean_p99 if mean_p99 > 1e-9 else None
            note = ("punches and noise overlap -- calibrate depth (S), do not "
                    "retune" if head is not None and head < 1.0
                    else "punches sit clear of the noise")
            print(f"  headroom         {_fmt(head, '7.2f')}x  -- {note}")
    else:
        print("  noise floor      n/a (no quiet stretch long enough to measure)")

    print("\nRESPONSE")
    react, react_p95 = _avg(runs, lambda r: reaction(r)[0]), _avg(runs, lambda r: reaction(r)[1])
    budget = sorted({r["settings"].get("target_budget_ms")
                     for r in runs if r["settings"].get("target_budget_ms")})
    print(f"  time to hit      mean {_fmt(react, '6.0f')} ms   "
          f"p95 {_fmt(react_p95, '6.0f')} ms"
          + (f"   of {budget[0]:.0f} ms allowed" if len(budget) == 1 else ""))
    miss_share = _avg(runs, lambda r: dropouts(r)[0])
    gap = _avg(runs, lambda r: dropouts(r)[1])
    fm, fp95 = _avg(runs, lambda r: frame_ms(r)[0]), _avg(runs, lambda r: frame_ms(r)[1])
    print(f"  no hand          {(miss_share * 100) if miss_share is not None else 0.0:6.1f}%"
          f"          longest gap {_fmt(gap, '6.0f')} ms")
    print(f"  frame time       mean {_fmt(fm, '6.1f')} ms   p95 {_fmt(fp95, '6.1f')} ms")

    shares = {}
    for run in runs:
        for name, share in stabilizer_share(run).items():
            shares.setdefault(name, []).append(share)
    if shares:
        cells = ", ".join(f"{name} {100 * statistics.fmean(v):.0f}%"
                          for name, v in sorted(shares.items()))
        print(f"  stabilizer       {cells}")

    if show_runs:
        print("\nRUNS")
        for run in runs:
            p99, worst, _ = noise_floor(run)
            margins = punch_margins(run)
            miss, _ = dropouts(run)
            print(f"  {run['started_at']}  hits {run['score']:3d}  "
                  f"miss {run['misses']:3d}  {run['duration_s']:6.1f}s  "
                  f"{run['cause']:7s}  weakest margin "
                  f"{_fmt(min(margins) if margins else None, '+7.4f')}  "
                  f"noise p99 {_fmt(p99, '7.4f')}  no hand {miss * 100:4.1f}%")
    print()


def main(argv):
    # Opting in is per machine and explicit. Nothing records anywhere else.
    if "--enable" in argv:
        path = enable_here()
        print(f"\nRun tracking is ON for this machine."
              f"\n  marker  {path}"
              f"\n  logs    {DATA_DIR}"
              f"\nBoth are gitignored and stay on this disk.\n")
        return 0
    if "--disable" in argv:
        removed = disable_here()
        tail = "" if removed else " (it was already off)"
        print(f"\nRun tracking is OFF for this machine.{tail}"
              f"\nExisting logs are untouched.\n")
        return 0

    show_runs = "--runs" in argv
    runs = load_runs()

    if "--last" in argv:
        runs = runs[-int(argv[argv.index("--last") + 1]):]

    state = "ON" if tracking_enabled() else "OFF - turn it on with --enable"
    print(f"\ntracking on this machine: {state}")
    report(runs, show_runs=show_runs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
