"""
run_lab_tests.py — run every lab game headless and collect the reports.

This is the regression gate for the seven limit-test games. Each one runs off
scripted input for a fixed number of frames with no window, so the whole suite
is deterministic and needs neither a camera nor a person. A game that throws
anywhere in update or render fails the run and says where.

    python run_lab_tests.py                  every game, default frame counts
    python run_lab_tests.py --frames 300     shorter
    python run_lab_tests.py --only conductor cradle
    python run_lab_tests.py --json out.json  full reports, for diffing runs

Exit code is 0 only if every game reported ok.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#: id -> (entry path, default frames, extra args)
GAMES = {
    "duckhunt":   ("duckhunt/duckhunt.py", 900, []),
    "depthlanes": ("depthlanes/depth_lanes.py", 1200, []),
    "signduel":   ("signduel/sign_duel.py", 900, []),
    "sculptor":   ("sculptor/sculptor.py", 600, []),
    "cradle":     ("cradle/cradle.py", 900, []),
    "conductor":  ("conductor/conductor.py", 900, []),
    "depthpong":  ("depthpong/depth_pong.py", 900, []),
}

#: Extra runs that exist to compare a setting against its default rather than to
#: check that a game works. Kept separate so a plain run stays quick.
VARIANTS = {
    "depthlanes-stabilized": ("depthlanes/depth_lanes.py", 1200, ["--stabilize"]),
    "depthlanes-miscalibrated": ("depthlanes/depth_lanes.py", 1200,
                                 ["--stabilize-at", "300"]),
    "conductor-beta0": ("conductor/conductor.py", 900, ["--beta", "0.0"]),
    "conductor-beta05": ("conductor/conductor.py", 900, ["--beta", "0.05"]),
    "sculptor-soak": ("sculptor/sculptor.py", 1200, ["--soak", "sphere"]),
    "depthpong-160": ("depthpong/depth_pong.py", 600, ["--grid", "160x120"]),
}


def run_one(name, entry, frames, extra, python):
    path = os.path.join(BASE_DIR, entry)
    started = time.perf_counter()
    proc = subprocess.run(
        [python, path, "--headless", str(frames), *extra],
        capture_output=True, text=True, cwd=os.path.dirname(path),
    )
    elapsed = time.perf_counter() - started

    # The engine and MediaPipe both log to stdout, so the report is the last
    # JSON object rather than the whole stream.
    report, parse_error = None, None
    text = proc.stdout
    start = text.find("{")
    while start != -1 and report is None:
        try:
            report = json.loads(text[start:])
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
            start = text.find("{", start + 1)

    if report is None:
        return {"game": name, "ok": False, "wall_s": round(elapsed, 2),
                "error": parse_error or "no JSON report",
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr.strip().splitlines()[-4:]}
    report["wall_s"] = round(elapsed, 2)
    report["returncode"] = proc.returncode
    report["variant_args"] = extra
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=None,
                        help="override every game's frame count")
    parser.add_argument("--only", nargs="+", metavar="ID",
                        help="run only these ids")
    parser.add_argument("--variants", action="store_true",
                        help="also run the comparison variants")
    parser.add_argument("--json", metavar="PATH", help="write all reports here")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    jobs = dict(GAMES)
    if args.variants:
        jobs.update(VARIANTS)
    if args.only:
        missing = [k for k in args.only if k not in jobs]
        if missing:
            raise SystemExit(f"unknown id(s): {', '.join(missing)}")
        jobs = {k: v for k, v in jobs.items() if k in args.only}

    reports = []
    failures = []
    print(f"{'game':<26}{'ok':<5}{'frames':>8}{'mean ms':>10}{'p95 ms':>9}{'wall s':>9}")
    print("-" * 67)
    for name, (entry, frames, extra) in jobs.items():
        report = run_one(name, entry, args.frames or frames, extra, args.python)
        reports.append(report)
        timing = report.get("update_render", {})
        ok = report.get("ok", False)
        if not ok:
            failures.append(name)
        print(f"{name:<26}{'yes' if ok else 'NO':<5}"
              f"{timing.get('frames', 0):>8}"
              f"{timing.get('mean_ms', 0):>10.2f}"
              f"{timing.get('p95_ms', 0):>9.2f}"
              f"{report.get('wall_s', 0):>9.2f}")
        for line in report.get("errors", [])[:3]:
            print(f"    {line}")
        if "error" in report:
            print(f"    {report['error']}")
            for line in report.get("stderr_tail", []):
                print(f"    | {line}")

    print("-" * 67)
    print(f"{len(jobs) - len(failures)}/{len(jobs)} ok"
          + (f"   failed: {', '.join(failures)}" if failures else ""))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2)
        print(f"reports written to {args.json}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
