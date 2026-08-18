"""
play.py - one entry point for every game in this repo.

Each game lives in its own folder, hard-codes its own camera index and
resolution, and expects the engine to be importable. This launcher hides that:
it locates the sibling engine clone, puts it on the child's ``PYTHONPATH``, and
can override the pipeline settings a game hard-codes without editing any game
file.

    python play.py                    interactive menu
    python play.py list               every title, and whether it will run
    python play.py doctor             check Python, deps, engine, and cameras
    python play.py sling              run a game
    python play.py punchy --camera 1  run it against a different webcam
    python play.py sling --width 1920 --height 1080 --smooth 0.3
    python play.py flappy --tof sim   force simulated ToF depth ("sim" | "off")
    python play.py info sling         controls, without launching
    python play.py labtests           the headless regression suite

Overrides work by re-entering this file in ``__exec`` mode in the child
process, which wraps ``VisionPipeline`` before the game's module-level code
runs. Nothing is patched on disk, so please do not "fix" a game by hard-coding
a camera index - add it here instead. ``--tof`` is applied in ``start()``
rather than ``__init__`` because games set ``tof_simulated`` on the instance
after construction and would otherwise win.

Anything after ``--`` is passed straight through to the game, so
``python play.py duckhunt -- --headless 900`` works.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# A cp1252 console cannot encode box-drawing characters, and the resulting
# UnicodeEncodeError would kill the launcher before it printed anything.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent

ENV_CAMERA = "VISUAL_AI_CAMERA"
ENV_WIDTH = "VISUAL_AI_WIDTH"
ENV_HEIGHT = "VISUAL_AI_HEIGHT"
ENV_SMOOTH = "VISUAL_AI_SMOOTH"
ENV_TOF = "VISUAL_AI_TOF"
ENV_ENTRY = "VISUAL_AI_ENTRY"

#: Python versions mediapipe publishes wheels for. Checked up front because the
#: failure otherwise surfaces as an opaque pip resolver error.
MIN_PY = (3, 9)
MAX_PY = (3, 12)


def engine_src() -> Path | None:
    """The engine's src/ directory, or None. Mirrors engine_bootstrap."""
    override = os.environ.get("VISUAL_AI_ENGINE")
    candidates = []
    if override:
        candidates += [Path(override) / "src", Path(override)]
    candidates.append(ROOT.parent / "visual ai game engine" / "src")
    for path in candidates:
        if (path / "visual_ai" / "__init__.py").is_file():
            return path
    return None


# ── Colour ────────────────────────────────────────────────────────────────────

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _paint(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(t): return _paint("2", t)
def bold(t): return _paint("1", t)
def green(t): return _paint("32", t)
def red(t): return _paint("31", t)
def yellow(t): return _paint("33", t)


# ── Registry ──────────────────────────────────────────────────────────────────

@dataclass
class Title:
    id: str
    name: str
    entry: Path
    blurb: str
    controls: list[str] = field(default_factory=list)
    aliases: tuple[str, ...] = ()
    note: str = ""
    headless: bool = False          # supports --headless N

    @property
    def directory(self) -> Path:
        return self.entry.parent

    @property
    def available(self) -> bool:
        return self.entry.is_file()


def _titles() -> list[Title]:
    g = ROOT
    return [
        Title(
            id="sling", name="Sling", entry=g / "Sling" / "main.py",
            blurb="Webcam slingshot. Close a fist to grab, open your hand to fire.",
            controls=["Closed fist - grab the bird, then pull back to aim",
                      "Open hand - release / fire",
                      "1 / 2 / 3 - Easy / Medium / Hard",
                      "H - how-to-play card", "K - landmark smoothing",
                      "L - depth calibration", "X - cancel calibration",
                      "R - restart", "Q or ESC - quit"],
            aliases=("s", "birds"),
        ),
        Title(
            id="flappy", name="ToF Flappy", entry=g / "flappy" / "tof_flappy.py",
            blurb="Flap by raising your index finger. Runs on simulated ToF depth.",
            controls=["Index finger height - flap", "H - how-to-play card",
                      "K - landmark smoothing", "L - depth calibration",
                      "Q or ESC - quit"],
            aliases=("f",),
        ),
        Title(
            id="punchy", name="ToF Z-Punch", entry=g / "punchy" / "tof_punch.py",
            blurb="Punch targets by driving your fist at the camera.",
            controls=["Punch toward the camera - hit the target",
                      "H - how-to-play card", "Q or ESC - quit"],
            aliases=("punch", "p"),
        ),
        Title(
            id="avatarcatch", name="Avatar Catch",
            entry=g / "avatarcatch" / "avatar_catch.py",
            blurb="Photograph yourself once, then catch falling shapes with the cutout.",
            controls=["SPACE - take your photo, then hold still for the countdown",
                      "Move hand left/right - move your avatar",
                      "L - toggle the exposure fix on the same capture",
                      "R - discard the photo and start over", "Q or ESC - quit"],
            aliases=("avatar", "catch"),
            note="Portrait matting needs onnxruntime; without it you get an uncut crop.",
        ),
        # The seven lab games. Each pushes one part of the SDK until it
        # complains, and each also runs headless off scripted input.
        Title(
            id="duckhunt", name="Duck Hunt", entry=g / "duckhunt" / "duckhunt.py",
            blurb="Ducks dive at your face; swat them with a finger. Tests face tracking.",
            controls=["Move your index finger - swat", "Q or ESC - quit"],
            aliases=("ducks", "dh"), headless=True,
        ),
        Title(
            id="depthlanes", name="Depth Lanes",
            entry=g / "depthlanes" / "depth_lanes.py",
            blurb="Rhythm game on three ToF distance bands. Scores timing error in ms.",
            controls=["Hold your hand at the right depth on the beat",
                      "L - calibrate the stabilizer (hold still)", "X - cancel",
                      "Q or ESC - quit"],
            aliases=("lanes", "dl"), headless=True,
        ),
        Title(
            id="signduel", name="Sign Duel", entry=g / "signduel" / "sign_duel.py",
            blurb="Simon says with hand signs on a shrinking timer. Measures sign latency.",
            controls=["Make the sign shown - fist / open palm / point / peace",
                      "Q or ESC - quit"],
            aliases=("duel", "signs"), headless=True,
        ),
        Title(
            id="sculptor", name="Sculptor", entry=g / "sculptor" / "sculptor.py",
            blurb="Shape a mesh with grip and depth. A soak test for the 3D renderer.",
            controls=["Move hand - orbit", "Grip - scale", "Depth - extrude",
                      "SPACE or 1-8 - change mesh", "Q or ESC - quit"],
            aliases=("sculpt",), headless=True,
        ),
        Title(
            id="cradle", name="Cat's Cradle", entry=g / "cradle" / "cradle.py",
            blurb="Two-handed: a string between your palms threads the rings.",
            controls=["Show BOTH hands",
                      "Move them apart to pull the string taut", "Q or ESC - quit"],
            aliases=("string",), headless=True,
            note="The only two-handed title; needs both hands in frame.",
        ),
        Title(
            id="conductor", name="Conductor", entry=g / "conductor" / "conductor.py",
            blurb="Conduct a metronome. Beats are your hand's direction reversals.",
            controls=["Sweep your hand left and right in time",
                      "--beta A - retune the One-Euro speed coupling",
                      "Q or ESC - quit"],
            aliases=("beat", "baton"), headless=True,
        ),
        Title(
            id="depthpong", name="Depth Pong",
            entry=g / "depthpong" / "depth_pong.py",
            blurb="Your whole silhouette is the paddle, off the ToF occupancy grid.",
            controls=["Move into the ball's path",
                      "--grid WxH - depth grid resolution", "Q or ESC - quit"],
            aliases=("dpong",), headless=True,
        ),
        Title(
            id="labtests", name="Lab regression suite",
            entry=g / "run_lab_tests.py",
            blurb="Run all seven lab games headless and print the report table.",
            controls=["--frames N - shorter runs", "--only <id> ...",
                      "--json out.json - full reports"],
            aliases=("tests", "lab"),
        ),
    ]


def find_title(query: str) -> Title | None:
    query = query.lower()
    for t in _titles():
        if query == t.id or query in t.aliases:
            return t
    matches = [t for t in _titles() if t.id.startswith(query)]
    return matches[0] if len(matches) == 1 else None


# ── Environment checks ────────────────────────────────────────────────────────

def check_python() -> list[str]:
    problems = []
    v = sys.version_info[:2]
    if v < MIN_PY or v > MAX_PY:
        problems.append(
            "Python {}.{} is outside the supported {}.{}-{}.{} range. mediapipe "
            "publishes no wheels beyond {}.{}, so hand tracking cannot install "
            "here.".format(v[0], v[1], *MIN_PY, *MAX_PY, *MAX_PY)
        )
    return problems


def check_engine() -> list[str]:
    if engine_src() is None:
        return [
            "The engine was not found. These games import visual_ai from a "
            "sibling clone:\n"
            '    git clone https://github.com/Ashiqalink/visual-ai-game-engine.git '
            '"{}"\n'
            "or point VISUAL_AI_ENGINE at it if it lives elsewhere."
            .format(ROOT.parent / "visual ai game engine")
        ]
    return []


def check_packages() -> list[str]:
    missing = []
    for name in ("cv2", "numpy", "mediapipe"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        return ["Missing required packages: {}. Install them with:\n"
                "    pip install -r requirements.txt".format(", ".join(missing))]
    return []


def preflight() -> list[str]:
    """Everything that stops any game from running at all."""
    return check_python() + check_engine() + check_packages()


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list(args) -> int:
    print()
    for t in _titles():
        mark = green("ok ") if t.available else red("missing")
        extra = dim("  headless") if t.headless else ""
        print(f"  {mark}  {bold(t.id):<24} {t.blurb}{extra}")
    print()
    for problem in preflight():
        print(yellow("  ! ") + problem.replace("\n", "\n    "))
        print()
    return 0


def cmd_info(args) -> int:
    t = find_title(args.title)
    if t is None:
        print(red(f"  unknown title: {args.title}"))
        return 1
    show_controls(t)
    print(f"  {dim('entry   ')} {t.entry}")
    print(f"  {dim('engine  ')} {engine_src() or red('not found')}\n")
    return 0


def show_controls(t: Title) -> None:
    print(f"\n  {bold(t.name)}  {dim(t.blurb)}")
    if t.note:
        print(f"  {yellow('note')} {t.note}")
    for line in t.controls:
        print(f"    {dim('-')} {line}")


def cmd_doctor(args) -> int:
    print(f"\n  {bold('python')}   {sys.version.split()[0]}  {dim(sys.executable)}")

    src = engine_src()
    print(f"  {bold('engine')}   {src if src else red('not found')}")

    if src:
        sys.path.insert(0, str(src))
    for name in ("cv2", "numpy", "mediapipe", "psutil", "pygame", "onnxruntime"):
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "?")
            print(f"  {green('ok  '):<9} {name} {dim(version)}")
        except ImportError:
            required = name in ("cv2", "numpy", "mediapipe")
            tag = red("MISSING") if required else yellow("absent ")
            why = "required" if required else "optional"
            print(f"  {tag:<9} {name} {dim('(' + why + ')')}")

    try:
        import visual_ai
        engine = "C++ engine_core" if visual_ai.CPP_ENGINE_AVAILABLE \
            else "Python fallback (slower, same physics)"
        print(f"  {green('ok  '):<9} visual_ai {dim(engine)}")
    except Exception as exc:
        print(f"  {red('FAIL'):<9} visual_ai {dim(f'{type(exc).__name__}: {exc}')}")

    print(f"\n  {bold('cameras')}")
    try:
        import cv2
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
        found = False
        for index in range(3):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"    {green('ok')}  index {index}  {w}x{h}")
                    found = True
            cap.release()
        if not found:
            print(f"    {yellow('none found')} - games fall back to simulated input")
    except ImportError:
        print(f"    {red('cannot probe')} - opencv-python is not installed")

    problems = preflight()
    print()
    for problem in problems:
        print(yellow("  ! ") + problem.replace("\n", "\n    ") + "\n")
    if not problems:
        print(green("  everything needed is in place.\n"))
    return 1 if problems else 0


def cmd_menu(args) -> int:
    titles = [t for t in _titles() if t.available]
    while True:
        print(f"\n  {bold('Visual AI games')}\n")
        for i, t in enumerate(titles, 1):
            print(f"    {bold(str(i)):>3}. {t.name:<22} {dim(t.blurb)}")
        print(f"    {bold('d'):>3}. doctor")
        print(f"    {bold('q'):>3}. quit\n")
        try:
            choice = input("  pick: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice in ("q", "quit", "exit"):
            return 0
        if choice == "d":
            cmd_doctor(args)
            continue
        title = None
        if choice.isdigit() and 1 <= int(choice) <= len(titles):
            title = titles[int(choice) - 1]
        else:
            title = find_title(choice)
        if title is None:
            print(red("  no such title"))
            continue
        run_title(title, args)


# ── Launching ─────────────────────────────────────────────────────────────────

def build_env(args) -> dict:
    env = os.environ.copy()
    src = engine_src()
    if src:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(src) + (os.pathsep + existing if existing else "")
    for attr, key in (("camera", ENV_CAMERA), ("width", ENV_WIDTH),
                      ("height", ENV_HEIGHT), ("smooth", ENV_SMOOTH),
                      ("tof", ENV_TOF)):
        value = getattr(args, attr, None)
        if value is not None:
            env[key] = str(value)
    return env


def run_title(title: Title, args) -> int:
    if not title.available:
        print(red(f"  {title.id} is not installed at {title.entry}"))
        return 1

    problems = preflight()
    if problems:
        print()
        for problem in problems:
            print(red("  ! ") + problem.replace("\n", "\n    ") + "\n")
        return 1

    env = build_env(args)
    env[ENV_ENTRY] = str(title.entry)
    extra = list(getattr(args, "extra", []) or [])

    show_controls(title)
    print(f"\n  {dim('python  ')} {sys.executable}")
    print(f"  {dim('engine  ')} {engine_src()}")
    overrides = [f"{k.replace('VISUAL_AI_', '').lower()}={env[k]}"
                 for k in (ENV_CAMERA, ENV_WIDTH, ENV_HEIGHT, ENV_SMOOTH, ENV_TOF)
                 if k in env]
    if overrides:
        print(f"  {dim('override')} {', '.join(overrides)}")
    print(f"  {dim('cwd     ')} {title.directory}\n")
    # Without this the banner sits in a buffered pipe and lands after the
    # game's own output.
    sys.stdout.flush()

    command = [sys.executable, str(Path(__file__).resolve()), "__exec", *extra]
    started = time.time()
    try:
        code = subprocess.call(command, cwd=str(title.directory), env=env)
    except KeyboardInterrupt:
        code = 130

    elapsed = time.time() - started
    verdict = green("exited cleanly") if code == 0 else red(f"exited with code {code}")
    print(f"\n  {title.name} {verdict} after {elapsed:,.1f}s\n")
    return code


# ── Child process ─────────────────────────────────────────────────────────────

def exec_entry(extra_argv: list[str]) -> int:
    """Apply the overrides, then run the game as ``python <entry>`` would."""
    import runpy

    entry = Path(os.environ[ENV_ENTRY])
    src = engine_src()
    if src:
        sys.path.insert(0, str(src))
    sys.path.insert(0, str(entry.parent))
    sys.path.insert(0, str(ROOT))       # engine_bootstrap, labkit, instructions

    _patch_pipeline()

    sys.argv = [str(entry), *extra_argv]
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except KeyboardInterrupt:
        print("\n  interrupted")
        return 130
    return 0


def _patch_pipeline() -> None:
    """
    Apply --camera / --width / --height / --smooth / --tof.

    Constructor kwargs are rewritten on ``__init__``; the ToF flags are applied
    in ``start()`` instead, because games set ``tof_simulated`` on the instance
    after constructing it and would otherwise win.
    """
    camera = os.environ.get(ENV_CAMERA)
    width = os.environ.get(ENV_WIDTH)
    height = os.environ.get(ENV_HEIGHT)
    smooth = os.environ.get(ENV_SMOOTH)
    tof = os.environ.get(ENV_TOF)
    if not any((camera, width, height, smooth, tof)):
        return

    try:
        from visual_ai import pipeline as pipeline_module
    except Exception as exc:
        print(f"  [play] could not import the engine to apply overrides: {exc}")
        return

    cls = pipeline_module.VisionPipeline
    original_init = cls.__init__
    original_start = cls.start

    def patched_init(self, *a, **kw):
        if camera is not None:
            kw["camera_index"] = int(camera)
        if width:
            kw["width"] = int(width)
        if height:
            kw["height"] = int(height)
        if smooth is not None:
            kw["smooth_alpha"] = float(smooth)
        original_init(self, *a, **kw)

    def patched_start(self, *a, **kw):
        if tof == "sim":
            self.tof_simulated = True
            self.disable_camera = False
        elif tof == "off":
            self.tof_simulated = False
        return original_start(self, *a, **kw)

    cls.__init__ = patched_init
    cls.start = patched_start


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str]) -> int:
    # __exec is the internal child-process mode, not a user-facing command.
    if argv and argv[0] == "__exec":
        return exec_entry(argv[1:])

    parser = argparse.ArgumentParser(
        prog="play", description="Launcher for the Visual AI games.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `python play.py list` to see every title. Any flag this "
               "launcher does not own is passed straight to the game, so "
               "`python play.py duckhunt --headless 900` works.",
    )
    parser.add_argument("title", nargs="?", help="game id, or 'list' / 'doctor' / 'info'")
    parser.add_argument("rest", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument("--camera", type=int, help="webcam index")
    parser.add_argument("--width", type=int, help="capture width")
    parser.add_argument("--height", type=int, help="capture height")
    parser.add_argument("--smooth", type=float,
                        help="One-Euro resting smoothness (was smooth_alpha)")
    parser.add_argument("--tof", choices=("sim", "off"), help="simulated ToF depth")

    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1:]
    else:
        extra = []

    # Split the launcher's own flags off the front by hand rather than leaning
    # on parse_known_args, which returns the leftovers regrouped: positionals
    # and unknown flags come back in separate lists, so `duckhunt --headless
    # 120` reassembled as `120 --headless` and the game rejected it. A manual
    # scan is the only way to keep the game's argv in the order it was typed.
    mine, passthrough = [], []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("--camera", "--width", "--height", "--smooth", "--tof"):
            mine += argv[index:index + 2]
            index += 2
            continue
        if token.split("=", 1)[0] in ("--camera", "--width", "--height",
                                      "--smooth", "--tof"):
            mine.append(token)
            index += 1
            continue
        if token in ("-h", "--help") and not passthrough:
            parser.print_help()
            return 0
        passthrough.append(token)
        index += 1

    # The title is the first bare word; everything after it belongs to the game.
    title_arg = passthrough[0] if passthrough else None
    args = parser.parse_args(mine)
    args.title = title_arg
    args.rest = passthrough[1:]
    args.extra = extra

    if args.title in (None, "menu"):
        return cmd_menu(args)
    if args.title == "list":
        return cmd_list(args)
    if args.title == "doctor":
        return cmd_doctor(args)
    if args.title == "info":
        if not args.rest:
            print(red("  info needs a title, e.g. `python play.py info sling`"))
            return 1
        args.title = args.rest[0]
        return cmd_info(args)

    title = find_title(args.title)
    if title is None:
        print(red(f"\n  unknown title: {args.title}"))
        print(dim("  run `python play.py list` to see them all\n"))
        return 1
    # A bare `play duckhunt --headless 900` should work without the `--`.
    args.extra = extra + list(args.rest)
    return run_title(title, args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
