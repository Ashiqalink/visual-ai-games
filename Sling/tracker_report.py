"""Read this machine's Sling run logs back and diagnose them.

Offline like the tracker itself: reads `Sling/data/*.jsonl` from this machine
and prints to your terminal. Standard library only, so any interpreter runs it.

    python tracker_report.py                 # everything logged so far
    python tracker_report.py --last 10       # the 10 most recent runs
    python tracker_report.py --level 2       # one level only (0 / 1 / 2)
    python tracker_report.py --runs          # one line per run as well
    python tracker_report.py --shots         # one line per shot as well
    python tracker_report.py --enable        # turn tracking on for THIS machine
    python tracker_report.py --disable       # and off again

Tracking is off by default on every machine. It records only where someone
opted in, and what it records never leaves the disk it was written on.

The numbers worth reading first:

  release drift   how far the aim angle moved over the 100 ms before the bird
                  left, in degrees. This is the long-standing "it fires lower
                  than I aimed" complaint made measurable. Signed and by
                  convention positive means the shot left *below* where it was
                  pointed. A mean near zero with a wide spread is release
                  jitter; a mean well away from zero is a systematic pull, and
                  the sign says which way.
  power drift     the same window, but the pull length rather than its angle.
                  A pull that collapses as the hand opens fires short even when
                  the angle held.
  fire cause      'open' is a shot the player timed. 'lost' means the hand left
                  the tracker mid-pull and the pull was fired rather than
                  stranded; 'edge' means it left the frame. A log full of the
                  last two is describing a tracking problem, not an aim one.
  band lag        how many milliseconds the band trails the hand while aiming.
                  Cross-correlation peak over ARMED frames, so it measures the
                  whole chain -- camera, MediaPipe, one-euro, and the game's
                  own adaptive EMA. Resolution is one frame, ~17 ms at 60 fps.
  band error      mean |band - where the band should be| in pixels, against the
                  anchor and gain the shot was actually pulled with. What is
                  left after lag is jitter the player is fighting.
  settle          how long READY took to lock the aim anchor, and how often it
                  had to be forced by the timeout instead of the hand holding
                  still. A high forced share means READY_SETTLE_RADIUS is
                  tighter than a real hand can hold.
"""

import os
import sys
import math
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracker import (DATA_DIR, STATE_CODES, disable_here, enable_here,
                     load_runs, tracking_enabled)

MAX_LAG_FRAMES = 30   # logged frames, i.e. 30 * stride real frames

# The release window. 100 ms is about six frames at 60 fps -- long enough to
# hold the whole of a hand opening, short enough that ordinary aiming movement
# from before the release does not swamp what the release itself did.
RELEASE_MS = 100.0

LEVEL_NAMES = {0: "EASY", 1: "MEDIUM", 2: "HARD"}
ARMED = STATE_CODES["ARMED"]


def _series(run, state=None):
    """Columns for one run, optionally only frames in one state."""
    frames = run.get("frames") or []
    if state is not None:
        frames = [f for f in frames if f[5] == state]
    return list(zip(*frames)) if frames else [[]] * 9


def _pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


# ── response ────────────────────────────────────────────────────────────────

def _corr(bc, tc, lag, energy):
    """Correlation of band against hand shifted by `lag` samples.

    Positive lag = the band is reproducing where the hand was `lag` samples
    ago. Negative lags are evaluated too, not because the band can lead the
    hand, but so a true lag of under one frame still has a sample on each side
    of the peak.
    """
    acc = 0.0
    if lag >= 0:
        for i in range(lag, len(bc)):
            acc += bc[i] * tc[i - lag]
    else:
        for i in range(0, len(bc) + lag):
            acc += bc[i] * tc[i - lag]
    return acc / energy


def band_lag_ms(run):
    """Cross-correlation peak between hand and band over ARMED frames.

    Both series are mean-centred first, so this measures how far the band's
    *movement* trails the hand's, not the constant offset between them -- the
    band lives at the slingshot and the hand does not. Vertical only: the pull
    is mostly vertical and the two axes share a filter, so one axis with real
    movement in it beats averaging a second axis that may be nearly still.
    Returns None when a run holds too little aiming to say anything.
    """
    cols = _series(run, ARMED)
    t_ms, hand_y, band_y = cols[0], cols[2], cols[4]
    if len(hand_y) < 60:
        return None
    hm, bm = statistics.fmean(hand_y), statistics.fmean(band_y)
    tc = [v - hm for v in hand_y]
    bc = [v - bm for v in band_y]
    energy = math.sqrt(sum(v * v for v in tc) * sum(v * v for v in bc))
    if energy < 1e-6:
        return None
    lags = range(-2, min(MAX_LAG_FRAMES, len(hand_y) // 4))
    corrs = [_corr(bc, tc, lag, energy) for lag in lags]
    best = max(range(len(corrs)), key=lambda i: corrs[i])
    # Median rather than mean interval: ARMED frames are not contiguous -- a
    # run holds one stretch per shot, and the flights between them would
    # otherwise be spread back over the aiming as a longer sample time.
    per_sample = statistics.median(
        [b - a for a, b in zip(t_ms, t_ms[1:])])
    # Reported to the nearest whole frame. A sub-frame parabola fit was tried
    # in flappy and dropped as biased; the same argument applies here.
    return max(0.0, float(lags[best])) * per_sample


def band_error(run):
    """Mean and p95 |band - ideal band| in px, over the logged shot run-ups.

    The ideal band is where the pull maps to with no smoothing at all:
    slingshot + (hand - anchor) * gain, held to MAX_PULL exactly as the game
    holds it -- a pull past the band's limit is the game working, not error.
    Measured inside each shot's `pre` window because that is where the anchor
    and gain that were in force are known; reconstructing them for the rest of
    the run would be guesswork.
    """
    sling = run["settings"].get("sling")
    if not sling:
        return None, None
    max_pull = run["settings"].get("max_pull")
    errs = []
    for shot in run.get("shots") or []:
        ax, ay = shot["anchor"]
        gain = shot["gain"]
        for row in shot.get("pre") or []:
            if row[5] != ARMED or not row[6]:
                continue
            dx = (row[1] - ax) * gain
            dy = (row[2] - ay) * gain
            dist = math.hypot(dx, dy)
            if max_pull and dist > max_pull:
                dx, dy = dx / dist * max_pull, dy / dist * max_pull
            errs.append(math.hypot(row[3] - sling[0] - dx,
                                   row[4] - sling[1] - dy))
    if not errs:
        return None, None
    return statistics.fmean(errs), _pct(errs, 0.95)


def dropouts(run):
    """(share of frames with no hand, longest gap in ms)."""
    cols = _series(run)
    t_ms, visible = cols[0], cols[6]
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
    fms = _series(run)[8]
    if not fms:
        return None, None
    return statistics.fmean(fms), _pct(fms, 0.95)


# ── the release ─────────────────────────────────────────────────────────────

def _aim(sling, band):
    """(angle in degrees, pull length in px) for a band position.

    The launch direction is band -> slingshot, which is what `_launch_velocity`
    uses, so the angle here is the direction the bird will actually leave in.
    Zero is straight along +x; positive is upward on screen, because screen y
    grows downward and this is meant to be read like an aim, not like a matrix.
    """
    dx = sling[0] - band[0]
    dy = sling[1] - band[1]
    return math.degrees(math.atan2(-dy, dx)), math.hypot(dx, dy)


def release_drift(run, shot):
    """(angle drift in degrees, pull drift in px) over the last RELEASE_MS.

    Positive angle drift means the aim rotated *downward* before the shot left
    -- the bird goes lower than where it was pointed, which is the complaint
    this whole log exists to test. Positive pull drift means the pull grew;
    negative means it collapsed as the hand opened, firing short.

    Returns (None, None) when the run-up is too short or has no tracked frames
    in it, which is what a shot fired from a lost hand looks like.
    """
    sling = run["settings"].get("sling")
    pre = [r for r in (shot.get("pre") or []) if r[5] == ARMED]
    if not sling or len(pre) < 3:
        return None, None
    # The endpoint is the shot itself, not the last frame before it: the band
    # position the bird actually left from is recorded on the shot, and it is
    # one frame newer than anything in the run-up.
    end_t, end_band = shot["t_ms"], shot["band"]
    cutoff = end_t - RELEASE_MS
    earlier = [r for r in pre if r[0] <= cutoff]
    start = earlier[-1] if earlier else pre[0]
    if end_t - start[0] < RELEASE_MS * 0.5:
        return None, None
    a0, p0 = _aim(sling, (start[3], start[4]))
    a1, p1 = _aim(sling, end_band)
    # Negated so "the aim fell" reads positive -- see the docstring.
    return -(a1 - a0), p1 - p0


def all_shots(runs):
    return [(run, shot) for run in runs for shot in (run.get("shots") or [])]


# ── printing ────────────────────────────────────────────────────────────────

def _fmt(value, spec="6.1f", none="   n/a"):
    return none if value is None else format(value, spec)


def _avg(runs, fn):
    vals = [v for v in (fn(r) for r in runs) if v is not None]
    return statistics.fmean(vals) if vals else None


def report(runs, show_runs=False, show_shots=False):
    if not runs:
        print(f"No runs logged yet. Play a round -- logs land in {DATA_DIR}")
        return

    print(f"\n{len(runs)} run(s) from {DATA_DIR}\n")

    by_level = {}
    for run in runs:
        by_level.setdefault(run["level"], []).append(run)
    levels = sorted(by_level)

    print("PER LEVEL")
    print(f"  {'level':7s} {'runs':>4s} {'best':>6s} {'mean':>7s} "
          f"{'sec':>6s} {'shots':>6s}  ended")
    for lvl in levels:
        group = by_level[lvl]
        scores = [r["score"] for r in group]
        causes = {}
        for r in group:
            causes[r["cause"]] = causes.get(r["cause"], 0) + 1
        cause_str = ", ".join(f"{k} {v}" for k, v in
                              sorted(causes.items(), key=lambda kv: -kv[1]))
        shots = sum(len(r.get("shots") or []) for r in group)
        print(f"  {LEVEL_NAMES.get(lvl, str(lvl)):7s} {len(group):4d} "
              f"{max(scores):6d} {statistics.fmean(scores):7.0f} "
              f"{statistics.fmean([r['duration_s'] for r in group]):6.1f} "
              f"{shots:6d}  {cause_str}")

    print("\nRELEASE  (the %.0f ms before the bird leaves)" % RELEASE_MS)
    print("  positive angle drift = the aim fell before the shot left;"
          "\n  negative pull drift  = the pull collapsed, so it fires short")
    print(f"  {'cause':7s} {'shots':>5s} {'angle deg':>10s} {'|angle|':>8s} "
          f"{'p95':>6s} {'pull px':>8s} {'pull dist':>9s} {'pts/shot':>9s}")
    pairs = all_shots(runs)
    by_cause = {}
    for run, shot in pairs:
        by_cause.setdefault(shot["cause"], []).append((run, shot))
    for cause in sorted(by_cause, key=lambda c: -len(by_cause[c])):
        group = by_cause[cause]
        drifts = [(a, p) for a, p in
                  (release_drift(r, s) for r, s in group) if a is not None]
        angles = [a for a, _ in drifts]
        pulls = [p for _, p in drifts]
        print(f"  {cause:7s} {len(group):5d} "
              f"{_fmt(statistics.fmean(angles) if angles else None, '10.2f'):>10s} "
              f"{_fmt(statistics.fmean([abs(a) for a in angles]) if angles else None, '8.2f'):>8s} "
              f"{_fmt(_pct([abs(a) for a in angles], 0.95) if angles else None, '6.2f'):>6s} "
              f"{_fmt(statistics.fmean(pulls) if pulls else None, '8.2f'):>8s} "
              f"{statistics.fmean([s['pull_dist'] for _, s in group]):9.1f} "
              f"{statistics.fmean([s['points'] for _, s in group]):9.1f}")
    if pairs:
        measured = sum(1 for r, s in pairs if release_drift(r, s)[0] is not None)
        if measured < len(pairs):
            print(f"  ({len(pairs) - measured} shot(s) had too little tracked "
                  f"run-up to measure -- typically a lost hand)")

    print("\nSETTLE  (READY -> ARMED: how the aim anchor got locked)")
    ready = [s["ready_ms"] for _, s in pairs]
    forced = [s["settle_forced"] for _, s in pairs]
    if ready:
        print(f"  time to lock     mean {statistics.fmean(ready):6.0f} ms   "
              f"p95 {_pct(ready, 0.95):6.0f} ms")
        print(f"  forced by timeout {100.0 * sum(forced) / len(forced):5.0f}%   "
              f"-- a high share means the settle radius is tighter than a hand "
              f"can hold")
    else:
        print("  no shots logged yet")

    print("\nAIM RESPONSE  (per level, averaged over runs)")
    print(f"  {'level':7s} {'lag ms':>7s} {'err px':>7s} {'p95 px':>7s} "
          f"{'no hand':>8s} {'max gap':>8s} {'frame ms':>9s} {'p95':>6s}")
    for lvl in levels:
        group = by_level[lvl]
        lag = _avg(group, band_lag_ms)
        err = _avg(group, lambda r: band_error(r)[0])
        p95 = _avg(group, lambda r: band_error(r)[1])
        miss = _avg(group, lambda r: dropouts(r)[0])
        gap = _avg(group, lambda r: dropouts(r)[1])
        fm = _avg(group, lambda r: frame_ms(r)[0])
        fp95 = _avg(group, lambda r: frame_ms(r)[1])
        print(f"  {LEVEL_NAMES.get(lvl, str(lvl)):7s} {_fmt(lag, '7.0f'):>7s} "
              f"{_fmt(err, '7.1f'):>7s} {_fmt(p95, '7.1f'):>7s} "
              f"{('%7.1f%%' % (miss * 100)) if miss is not None else '    n/a':>8s} "
              f"{_fmt(gap, '8.0f'):>8s} {_fmt(fm, '9.1f'):>9s} "
              f"{_fmt(fp95, '6.1f'):>6s}")

    if show_runs:
        print("\nRUNS")
        for run in runs:
            lag = band_lag_ms(run)
            err, _ = band_error(run)
            miss, _ = dropouts(run)
            print(f"  {run['started_at']}  "
                  f"{LEVEL_NAMES.get(run['level'], run['level']):6s} "
                  f"score {run['score']:5d}  {run['duration_s']:6.1f}s  "
                  f"{run['cause']:7s}  shots {len(run.get('shots') or []):2d}  "
                  f"lag {_fmt(lag, '4.0f')}ms  err {_fmt(err, '5.1f')}px  "
                  f"no hand {miss * 100:4.1f}%")

    if show_shots:
        print("\nSHOTS")
        for run, shot in pairs:
            angle, pull = release_drift(run, shot)
            aim, _ = _aim(run["settings"].get("sling", (0, 0)), shot["band"])
            print(f"  {run['started_at']} #{shot['index']:2d}  "
                  f"{shot['bird']:8s} {shot['cause']:6s}  "
                  f"pull {shot['pull_dist']:5.0f}px  aim {aim:6.1f}deg  "
                  f"drift {_fmt(angle, '6.2f')}deg {_fmt(pull, '6.1f')}px  "
                  f"v ({shot['velocity'][0]:6.2f},{shot['velocity'][1]:6.2f})  "
                  f"{shot['hits']:2d} hit  {shot['points']:5d} pts")
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

    runs = load_runs()

    if "--level" in argv:
        want = int(argv[argv.index("--level") + 1])
        runs = [r for r in runs if r["level"] == want]
    if "--last" in argv:
        runs = runs[-int(argv[argv.index("--last") + 1]):]

    state = "ON" if tracking_enabled() else "OFF - turn it on with --enable"
    print(f"\ntracking on this machine: {state}")
    report(runs, show_runs="--runs" in argv, show_shots="--shots" in argv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
