"""
play.py - one entry point for every game, demo, playground, and tool.

Each game lives in its own folder, hard-codes its own camera index and
resolution, and expects the engine to be importable. This launcher hides that:
it finds a working interpreter, locates the sibling engine clone, puts it on
the child's ``PYTHONPATH``, and can override the pipeline settings a game
hard-codes without editing any game file.

    python play.py                    interactive menu (loops until you quit)
    python play.py list               every title, and whether it will run
    python play.py list --json        the same, machine-readable
    python play.py info sling         controls, without launching
    python play.py doctor             check Python, deps, engine, and cameras
    python play.py sling              run a game
    python play.py punchy --camera 1  run it against a different webcam
    python play.py sling --width 1920 --height 1080 --smooth 0.3
    python play.py flappy --tof sim   force simulated ToF depth ("sim" | "off")
    python play.py labtests           the headless regression suite

Overrides work by re-entering this file in ``__exec`` mode in the child
process, which wraps ``VisionPipeline`` before the game's module-level code
runs. Nothing is patched on disk, so please do not "fix" a game by hard-coding
a camera index - add it here instead. ``--tof`` is applied in ``start()``
rather than ``__init__`` because games set ``tof_simulated`` on the instance
after construction and would otherwise win.

Anything after ``--`` is passed straight through to the game, so
``python play.py duckhunt -- --headless 900`` works.

This is the only launcher implementation. Titles that live in the *engine*
clone - the demos, the playgrounds, the benches - are not registered here,
because this repo has to run standalone and frozen without that clone. The
workspace launcher at ``D:\\visual\\play.py`` contributes them through
``register()`` before calling ``main()``.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import os
import shlex
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

#: True inside the PyInstaller bundle. The frozen build ships the engine and
#: every game inside the executable, so there is no sibling clone to find and
#: no second interpreter to spawn - see engine_src(), interpreter_for() and
#: run_title().
FROZEN = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

#: Where the games and the engine live. In a bundle that is the unpacked
#: temporary directory PyInstaller extracts to, not the location of the .exe.
ROOT = Path(sys._MEIPASS) if FROZEN else Path(__file__).resolve().parent

#: The container both clones sit in. Only meaningful in a source checkout;
#: used to find the workspace-level .venv when a title has none of its own.
WORKSPACE = ROOT.parent

ENV_CAMERA = "VISUAL_AI_CAMERA"
ENV_WIDTH = "VISUAL_AI_WIDTH"
ENV_HEIGHT = "VISUAL_AI_HEIGHT"
ENV_SMOOTH = "VISUAL_AI_SMOOTH"
ENV_TOF = "VISUAL_AI_TOF"
ENV_ENTRY = "VISUAL_AI_ENTRY"

#: The supported interpreter range, checked up front because the failure
#: otherwise surfaces as an opaque error. The ceiling is mediapipe's (no
#: wheels for 3.13+); the floor is the engine's, which uses PEP-604 unions in
#: signatures evaluated at def time and so raises TypeError on import under
#: 3.9. Matches the engine's requires-python.
MIN_PY = (3, 10)
MAX_PY = (3, 12)

KIND_ORDER = ("game", "demo", "playground", "tool")
KIND_HEADINGS = {
    "game": "games",
    "demo": "engine demos",
    "playground": "playgrounds - poke at the SDK live",
    "tool": "tools",
}


def engine_src() -> Path | None:
    """The engine's src/ directory, or None. Mirrors engine_bootstrap."""
    if FROZEN:
        # The engine is bundled, so visual_ai imports without any path work.
        # Report the bundle root rather than None: callers use this to decide
        # whether the engine was found at all, and in a bundle it always is.
        return ROOT
    override = os.environ.get("VISUAL_AI_ENGINE")
    candidates = []
    if override:
        candidates += [Path(override) / "src", Path(override)]
    candidates.append(WORKSPACE / "visual ai game engine" / "src")
    for path in candidates:
        if (path / "visual_ai" / "__init__.py").is_file():
            return path
    return None


# ── Color ────────────────────────────────────────────────────────────────────

def _ansi() -> bool:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR") is not None:
        return False
    if os.name == "nt":
        try:
            import ctypes

            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


_COLOR = _ansi()


def _paint(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(t: str) -> str: return _paint("2", t)
def bold(t: str) -> str: return _paint("1", t)
def green(t: str) -> str: return _paint("32", t)
def red(t: str) -> str: return _paint("31", t)
def yellow(t: str) -> str: return _paint("33", t)
def cyan(t: str) -> str: return _paint("36", t)


def banner(text: str) -> None:
    print(f"\n{cyan('-' * 66)}\n  {bold(text)}\n{cyan('-' * 66)}")


# ── Registry ──────────────────────────────────────────────────────────────────

@dataclass
class Title:
    """One runnable thing: a game, an engine demo, a playground, or a tool."""

    id: str
    name: str
    kind: str
    entry: Path
    blurb: str
    #: Working directory for the child. Defaults to the entry's own folder,
    #: which is right for every game; engine titles run from their repo root
    #: while their entry sits in examples/ or tools/, so they set it.
    directory: Path | None = None
    controls: list[str] = field(default_factory=list)
    aliases: tuple[str, ...] = ()
    note: str = ""                  # caveat shown before launch
    headless: bool = False          # supports --headless N

    def __post_init__(self) -> None:
        if self.directory is None:
            self.directory = self.entry.parent

    @property
    def available(self) -> bool:
        return self.entry.is_file()


def _titles() -> list[Title]:
    g = ROOT
    return [
        Title(
            id="sling", name="Sling", kind="game", entry=g / "Sling" / "main.py",
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
            id="flappy", name="Flappy", kind="game", entry=g / "flappy" / "flappy.py",
            blurb="Flap by raising your index finger. Pure RGB tracking, no depth sensor.",
            controls=["Index finger height - flap", "1 / 2 / 3 - easy / medium / hard",
                      "H - how-to-play card", "K - landmark smoothing",
                      "T - local run tracking (off unless opted in)",
                      "R - restart", "Q or ESC - quit"],
            aliases=("f",),
        ),
        Title(
            id="punchy", name="ToF Z-Punch", kind="game",
            entry=g / "punchy" / "tof_punch.py",
            blurb="Punch targets by driving your fist at the camera.",
            controls=["Punch toward the camera - hit the target",
                      "H - how-to-play card", "Q or ESC - quit"],
            aliases=("punch", "p"),
        ),
        Title(
            id="avatarcatch", name="Avatar Catch", kind="game",
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
        # complains, and each also runs headless off scripted input - which is
        # what run_lab_tests.py drives as the regression suite.
        Title(
            id="duckhunt", name="Duck Hunt", kind="game",
            entry=g / "duckhunt" / "duckhunt.py",
            blurb="Ducks dive at your face; swat them with a finger. Tests face tracking.",
            controls=["Move your index finger - swat", "Q or ESC - quit"],
            aliases=("ducks", "dh"), headless=True,
        ),
        Title(
            id="depthlanes", name="Depth Lanes", kind="game",
            entry=g / "depthlanes" / "depth_lanes.py",
            blurb="Rhythm game on three ToF distance bands. Scores timing error in ms.",
            controls=["Hold your hand at the right depth on the beat",
                      "L - calibrate the stabilizer (hold still)", "X - cancel",
                      "Q or ESC - quit"],
            aliases=("lanes", "dl"), headless=True,
        ),
        Title(
            id="signduel", name="Sign Duel", kind="game",
            entry=g / "signduel" / "sign_duel.py",
            blurb="Simon says with hand signs on a shrinking timer. Measures sign latency.",
            controls=["Make the sign shown - fist / open palm / point / peace",
                      "Q or ESC - quit"],
            aliases=("duel", "signs"), headless=True,
        ),
        Title(
            id="sculptor", name="Sculptor", kind="game",
            entry=g / "sculptor" / "sculptor.py",
            blurb="Shape a mesh with grip and depth. A soak test for the 3D renderer.",
            controls=["Move hand - orbit", "Grip - scale", "Depth - extrude",
                      "SPACE or 1-8 - change mesh", "Q or ESC - quit"],
            aliases=("sculpt",), headless=True,
        ),
        Title(
            id="cradle", name="Cat's Cradle", kind="game",
            entry=g / "cradle" / "cradle.py",
            blurb="Two-handed: a string between your palms threads the rings.",
            controls=["Show BOTH hands",
                      "Move them apart to pull the string taut", "Q or ESC - quit"],
            aliases=("string",), headless=True,
            note="The only two-handed title; needs both hands in frame.",
        ),
        Title(
            id="conductor", name="Conductor", kind="game",
            entry=g / "conductor" / "conductor.py",
            blurb="Conduct a metronome. Beats are your hand's direction reversals.",
            controls=["Sweep your hand left and right in time",
                      "--beta A - retune the One-Euro speed coupling",
                      "Q or ESC - quit"],
            aliases=("beat", "baton", "metronome"), headless=True,
        ),
        Title(
            id="depthpong", name="Depth Pong", kind="game",
            entry=g / "depthpong" / "depth_pong.py",
            blurb="Your whole silhouette is the paddle, off the ToF occupancy grid.",
            controls=["Move into the ball's path",
                      "--grid WxH - depth grid resolution", "Q or ESC - quit"],
            aliases=("dpong",), headless=True,
        ),
        # ── Demos ────────────────────────────────────────────────────────────
        Title(
            id="spike", name="Sling - pygame renderer spike", kind="demo",
            entry=g / "Sling" / "spike_pygame.py",
            blurb="Same vision pipeline, drawn by pygame instead of OpenCV.",
            controls=["Move index finger - the bird follows",
                      "Pinch - flash and grow", "Q or ESC - quit"],
            aliases=("pygame",),
        ),
        Title(
            # Not "birds": that is already an alias of sling, and the first
            # match wins in find_title(), so this would never be reachable.
            id="birds-pygame", name="Birds - pygame renderer", kind="demo",
            entry=g / "Sling" / "bird_pygame.py",
            blurb="Sling's birds and their flight effects, drawn by pygame. No camera needed.",
            controls=["SPACE - launch another bird", "R - clear the field",
                      "Q or ESC - quit"],
            aliases=("bp",),
        ),
        # ── Tools ────────────────────────────────────────────────────────────
        Title(
            id="labtests", name="Lab regression suite", kind="tool",
            entry=g / "run_lab_tests.py",
            blurb="Run all seven lab games headless and print the report table.",
            controls=["--frames N - shorter runs", "--only <id> ...",
                      "--json out.json - full reports"],
            aliases=("tests", "lab"),
        ),
        Title(
            id="diagnose", name="Pipeline diagnostics", kind="tool",
            entry=g / "Sling" / "diagnose_pipeline.py",
            blurb="Dump raw VisionPipeline payloads - use when a gesture will not fire.",
            controls=["Ctrl+C - stop"],
            aliases=("diag",),
        ),
        Title(
            id="flappy-report", name="Flappy run diagnostics", kind="tool",
            entry=g / "flappy" / "tracker_report.py",
            blurb="Read back this machine's flappy run logs: follow lag, tracking error, dropouts.",
            controls=["play flappy-report --runs - one line per run",
                      "play flappy-report --last 10 - the 10 most recent",
                      "play flappy-report --difficulty HARD"],
            aliases=("fr", "flappy-stats"),
            note="Reads flappy/data/ on this machine only; nothing is collected elsewhere.",
        ),
        Title(
            id="sling-report", name="Sling run diagnostics", kind="tool",
            entry=g / "Sling" / "tracker_report.py",
            blurb="Read back this machine's Sling run logs: release drift, band lag, fire causes.",
            controls=["play sling-report --shots - one line per shot",
                      "play sling-report --runs - one line per run",
                      "play sling-report --level 2 - one level only"],
            aliases=("sr", "sling-stats"),
            note="Reads Sling/data/ on this machine only; nothing is collected elsewhere.",
        ),
        Title(
            id="punchy-report", name="Punchy run diagnostics", kind="tool",
            entry=g / "punchy" / "tracker_report.py",
            blurb="Read back this machine's punchy run logs: trigger margin "
                  "against the depth noise floor.",
            controls=["play punchy-report --runs - one line per run",
                      "play punchy-report --last 10 - the 10 most recent"],
            aliases=("pr", "punchy-stats"),
            note="Reads punchy/data/ on this machine only; nothing is collected elsewhere.",
        ),
    ]


#: Titles contributed by a wrapping launcher through register(). The workspace
#: launcher puts the engine clone's demos, playgrounds and benches here; this
#: repo cannot own them, because it must still run from a standalone clone -
#: and inside the frozen bundle - where that clone does not exist.
_EXTRA: list[Title] = []
_REGISTRY: list[Title] | None = None


def register(titles: list[Title]) -> None:
    """Add titles that live outside this repo. Call before main()."""
    global _REGISTRY
    _EXTRA.extend(titles)
    _REGISTRY = None


def registry() -> list[Title]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _titles() + _EXTRA
    return _REGISTRY


def find_title(query: str) -> Title | None:
    query = query.strip().lower()
    titles = registry()
    for t in titles:
        if query == t.id or query in t.aliases:
            return t
    # Accept a path to a game file, not just an id. run_lab_tests.py launches
    # each game as `sys.executable <path> --headless N`, which in a frozen
    # build makes sys.executable this launcher and hands it a path where it
    # expects a title - so the whole suite reported 0/7 inside the bundle.
    if query.endswith(".py"):
        name = Path(query).name
        for t in titles:
            if t.entry.name.lower() == name:
                return t
    matches = [t for t in titles if t.id.startswith(query)]
    return matches[0] if len(matches) == 1 else None


def suggest(token: str) -> list[str]:
    """Close-match ids for an unknown token, best first."""
    titles = registry()
    names = [t.id for t in titles] + [a for t in titles for a in t.aliases]
    hits = difflib.get_close_matches(token.lower(), names, n=3, cutoff=0.5)
    # map aliases back to their canonical id, dedup preserving order
    out: list[str] = []
    for hit in hits:
        title = find_title(hit)
        if title and title.id not in out:
            out.append(title.id)
    return out


def _unknown_title(token: str) -> int:
    print(red(f"  unknown title: {token}"))
    close = suggest(token)
    if close:
        print(f"  {dim('did you mean: ' + ', '.join(close) + '?')}")
    print(f"  {dim('known ids: ' + ', '.join(t.id for t in registry()))}")
    return 1


# ── Interpreter discovery ─────────────────────────────────────────────────────

def _venv_python(venv: Path) -> Path | None:
    exe = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return exe if exe.is_file() else None


def _has_cv2(python: Path) -> bool:
    """Cheap check: is OpenCV installed next to this interpreter?"""
    site = python.parent.parent / ("Lib/site-packages" if os.name == "nt" else "lib")
    if (site / "cv2").exists():
        return True
    return any(p.name == "cv2" for p in site.glob("*/cv2")) if site.exists() else False


def interpreter_for(title: Title) -> tuple[Path, str]:
    """
    Pick the interpreter for a title: its own venv first, then this repo's,
    then the workspace's, then whatever is running this script.

    Returns ``(python_path, where_it_came_from)``.
    """
    if FROZEN:
        # There is one interpreter in a bundle and it is this one.
        return Path(sys.executable), "bundled"

    candidates: list[tuple[Path, str]] = []
    for venv_dir, label in (
        (title.directory / ".venv", f"{title.directory.name}/.venv"),
        (ROOT / ".venv", f"{ROOT.name}/.venv"),
        (WORKSPACE / ".venv", ".venv"),
    ):
        exe = _venv_python(venv_dir)
        if exe:
            candidates.append((exe, label))

    for exe, label in candidates:
        if _has_cv2(exe):
            return exe, label
    if candidates:
        return candidates[0]
    return Path(sys.executable), "current interpreter"


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
            .format(WORKSPACE / "visual ai game engine")
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


# ── Commands: list / info ─────────────────────────────────────────────────────

def show_controls(t: Title) -> None:
    print(f"\n  {bold(t.name)}  {dim('[' + t.kind + ']')}")
    print(f"  {dim(t.blurb)}")
    if t.note:
        print(f"  {yellow('note:')} {dim(t.note)}")
    for line in t.controls:
        print(f"    {dim('-')} {line}")


def cmd_list(args) -> int:
    if getattr(args, "json", False):
        import json

        rows = [{"id": t.id, "name": t.name, "kind": t.kind,
                 "available": t.available, "aliases": list(t.aliases),
                 "headless": t.headless, "entry": str(t.entry), "blurb": t.blurb}
                for t in registry()]
        print(json.dumps(rows, indent=2))
        return 0

    banner("Visual AI - installed titles")
    for kind in KIND_ORDER:
        rows = [t for t in registry() if t.kind == kind]
        if not rows:
            continue
        print(f"\n  {bold(KIND_HEADINGS[kind])}")
        for t in rows:
            status = green("ready") if t.available else red("missing")
            alias = f"  {dim('(' + ', '.join(t.aliases) + ')')}" if t.aliases else ""
            extra = dim("  headless") if t.headless else ""
            print(f"    {bold(t.id.ljust(16))} {status.ljust(16)} {t.name}{alias}{extra}")
            print(f"    {' ' * 16} {dim(t.blurb)}")
    print(f"\n  {dim('play <id>  -  play info <id>  -  play <id> --camera 1  -  play doctor')}\n")
    for problem in preflight():
        print(yellow("  ! ") + problem.replace("\n", "\n    ") + "\n")
    return 0


def cmd_info(args) -> int:
    token = args.title
    t = find_title(token) if token else None
    if t is None:
        return _unknown_title(token or "")
    show_controls(t)
    python, source = interpreter_for(t)
    status = green("ready") if t.available else red("missing")
    print(f"\n  {dim('status  ')} {status}")
    print(f"  {dim('entry   ')} {t.entry}")
    print(f"  {dim('python  ')} {python}  {dim('(' + source + ')')}")
    print(f"  {dim('engine  ')} {engine_src() or red('not found')}\n")
    return 0


# ── Doctor ────────────────────────────────────────────────────────────────────

_PROBE = r"""
import json, sys
out = {"python": sys.version.split()[0], "packages": {}, "cameras": [],
       "engine": None, "depth": [], "luma": None}
for name in ("cv2", "mediapipe", "numpy", "psutil", "pygame", "onnxruntime"):
    try:
        __import__(name); out["packages"][name] = True
    except Exception:
        out["packages"][name] = False
try:
    import visual_ai
    out["engine"] = visual_ai.__file__
    out["cpp_engine"] = bool(getattr(visual_ai, "CPP_ENGINE_AVAILABLE", False))
except Exception as exc:
    out["engine_error"] = f"{type(exc).__name__}: {exc}"
if PROBE_CAMERAS:
    try:
        import cv2
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
        for index in range(3):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    out["cameras"].append([index, int(frame.shape[1]), int(frame.shape[0])])
            cap.release()
    except Exception as exc:
        out["camera_error"] = f"{type(exc).__name__}: {exc}"

    # Scene brightness on the first working camera. MediaPipe stops finding
    # hands well before a room looks dark to a person, so a number here is
    # worth more than "the camera opened".
    try:
        import cv2
        from visual_ai.low_light import measure_luma
        if out["cameras"]:
            index = out["cameras"][0][0]
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
            cap = cv2.VideoCapture(index, backend)
            for _ in range(10):          # let auto-exposure settle
                cap.read()
            ok, frame = cap.read()
            if ok and frame is not None:
                out["luma"] = round(float(measure_luma(frame)), 1)
            cap.release()
    except Exception as exc:
        out["luma_error"] = f"{type(exc).__name__}: {exc}"

# Depth backends. Cheap and safe: each opens, takes one frame, and closes.
try:
    from visual_ai.depth_source import probe_depth_sources
    out["depth"] = [[name, ok, detail] for name, ok, detail in probe_depth_sources()]
except Exception as exc:
    out["depth_error"] = f"{type(exc).__name__}: {exc}"
print("PLAY_JSON" + json.dumps(out))
"""


def _probe(python: Path, probe_cameras: bool) -> dict:
    """Run the environment probe under one interpreter and parse its report."""
    import json

    source = f"PROBE_CAMERAS = {probe_cameras}\n" + _PROBE
    try:
        if FROZEN:
            # A frozen executable cannot re-spawn itself as an interpreter:
            # there is no play.py on disk and sys.executable is the bundle.
            # Run the probe here instead - the bundle is the only environment
            # there is, so there is nothing else it could be reporting on.
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exec(compile(source, "<probe>", "exec"), {})
            raw = buffer.getvalue()
        else:
            env = os.environ.copy()
            src = engine_src()
            if src:
                env["PYTHONPATH"] = str(src)
            raw = subprocess.run([str(python), "-c", source], capture_output=True,
                                 text=True, timeout=90, env=env).stdout
        return json.loads(raw.split("PLAY_JSON", 1)[1]) if "PLAY_JSON" in raw else {}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def cmd_doctor(args) -> int:
    banner("Visual AI - environment check")

    src = engine_src()
    print(f"\n  {'engine source'.ljust(16)} {green(str(src)) if src else red('not found')}")

    checked: dict[str, dict] = {}
    for t in registry():
        if not t.available:
            continue
        python, source = interpreter_for(t)
        if str(python) in checked:
            continue
        info = _probe(python, not getattr(args, "no_cameras", False))
        info["_source"] = source
        checked[str(python)] = info

    for python, info in checked.items():
        print(f"\n  {bold(info.get('_source', python))}")
        print(f"    {dim('path      ')} {python}")
        if "error" in info:
            print(f"    {red('probe failed')} {info['error']}")
            continue
        print(f"    {dim('python    ')} {info.get('python', '?')}")
        packages = " ".join(
            (green(name) if ok else red(name)) for name, ok in info["packages"].items())
        print(f"    {dim('packages  ')} {packages}")
        if info.get("engine"):
            correct = src is not None and str(src) in info["engine"]
            marker = green("current") if correct else yellow("stale copy")
            print(f"    {dim('engine    ')} {info['engine']}  [{marker}]")
            print(f"    {dim('C++ core  ')} "
                  f"{green('available') if info.get('cpp_engine') else dim('python fallback')}")
        else:
            print(f"    {red('engine    ')} {info.get('engine_error', 'not importable')}")
        if not getattr(args, "no_cameras", False):
            if info.get("luma") is not None:
                luma = info["luma"]
                if luma < 30:
                    verdict = red("too dark to track - uncover the lens or add light")
                elif luma < 90:
                    verdict = yellow("dim; the low-light boost will be working")
                else:
                    verdict = green("fine")
                print(f"    {dim('scene     ')} luma {luma}  [{verdict}]")
            for name, ok, detail in info.get("depth", []):
                marker = green("open") if ok else dim("no")
                print(f"    {dim('depth ' + name.ljust(4))} {marker}  {detail}")
            if info.get("cameras"):
                shown = ", ".join(f"index {i} ({w}x{h})" for i, w, h in info["cameras"])
                print(f"    {dim('cameras   ')} {green(shown)}")
            else:
                print(f"    {dim('cameras   ')} "
                      f"{red(info.get('camera_error', 'none opened - is one in use?'))}")

    print(f"\n  {bold('titles')}")
    for t in registry():
        mark = green("ok") if t.available else red("missing")
        print(f"    {mark.ljust(14)} {t.id.ljust(16)} {dim(str(t.entry))}")

    problems = preflight()
    print()
    for problem in problems:
        print(yellow("  ! ") + problem.replace("\n", "\n    ") + "\n")
    if not problems:
        print(green("  everything needed is in place.\n"))
    return 1 if problems else 0


# ── Launching ─────────────────────────────────────────────────────────────────

def build_env(args) -> dict:
    """Child environment: engine on PYTHONPATH plus any pipeline overrides."""
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

    python, source = interpreter_for(title)
    env = build_env(args)
    env[ENV_ENTRY] = str(title.entry)
    extra = list(getattr(args, "extra", []) or [])

    # Invoked by path rather than by id means a machine is calling: that is
    # how run_lab_tests.py launches each game, and it parses stdout as JSON.
    # The banner below would corrupt that, which is why the suite reported
    # 0/7 inside the frozen bundle even once the games ran correctly.
    quiet = bool(getattr(args, "quiet", False))

    if not quiet:
        show_controls(title)
        print(f"\n  {dim('python  ')} {python}  {dim('(' + source + ')')}")
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

    started = time.time()
    if FROZEN:
        # A frozen executable cannot re-spawn itself as an interpreter: there
        # is no play.py on disk to hand it, and sys.executable is the bundle.
        # Run the game in this process instead. The overrides still apply -
        # they are read from the environment by _patch_pipeline either way -
        # and the games each guard their entry point with __main__, so runpy
        # reaches the same code the subprocess path would have.
        os.environ.update({k: v for k, v in env.items() if k.startswith("VISUAL_AI_")})
        previous = os.getcwd()
        try:
            os.chdir(str(title.directory))
            code = exec_entry(extra)
        except SystemExit as exc:            # a game calling sys.exit()
            code = exc.code if isinstance(exc.code, int) else 0
        except KeyboardInterrupt:
            code = 130
        finally:
            os.chdir(previous)
    else:
        # Re-enter this file in __exec mode so the child can wrap
        # VisionPipeline before the game's own module-level code runs.
        command = [str(python), str(Path(__file__).resolve()), "__exec", *extra]
        try:
            code = subprocess.call(command, cwd=str(title.directory), env=env)
        except KeyboardInterrupt:
            code = 130

    elapsed = time.time() - started
    if not quiet:
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


# ── Interactive menu ──────────────────────────────────────────────────────────

def _split_choice(line: str) -> list[str]:
    """
    Split a menu line into argv-style tokens, honouring double quotes so
    ``--out "D:\\my assets"`` stays one argument. posix=False keeps Windows
    backslashes literal; the surrounding quotes it leaves behind are stripped.
    """
    try:
        tokens = shlex.split(line, posix=False)
    except ValueError:                     # unbalanced quote - fall back
        tokens = line.split()
    return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t
            for t in tokens]


def _pause() -> None:
    """Hold the screen so a title's output survives before the menu redraws."""
    try:
        input(dim("  Enter to return to the menu... "))
    except (EOFError, KeyboardInterrupt):
        print()


def cmd_menu(args) -> int:
    """Looping menu: pick by number or id; returns here when a title exits."""
    while True:
        titles = [t for t in registry() if t.available]
        banner("Visual AI - pick a title")
        number = 0
        numbered: list[Title] = []
        for kind in KIND_ORDER:
            rows = [t for t in titles if t.kind == kind]
            if not rows:
                continue
            print(f"\n  {bold(KIND_HEADINGS[kind])}")
            for t in rows:
                number += 1
                numbered.append(t)
                print(f"  {bold(str(number).rjust(2))}. {t.id.ljust(14)} {dim(t.blurb)}")
        print(f"\n  {dim('number, id, or a full command (duckhunt --headless 900)')}")
        print(f"  {dim('i <id> info - d doctor - q quit')}\n")

        try:
            # lstrip("\ufeff"): piped input can carry a UTF-8 BOM
            raw = input("  choice: ").lstrip("\ufeff").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not raw:
            return 0
        tokens = _split_choice(raw)
        # a pasted shell command starts with the program name - drop it
        if tokens and tokens[0].lower() == "play":
            tokens = tokens[1:]
        if not tokens:
            continue

        first = tokens[0].lower()
        if first in ("q", "quit", "exit"):
            return 0
        if first in ("d", "doctor") and len(tokens) == 1:
            cmd_doctor(args)
            _pause()
            continue
        if first in ("i", "info") and len(tokens) > 1:
            t = find_title(tokens[1])
            if t:
                show_controls(t)
                print()
            else:
                _unknown_title(tokens[1])
            continue

        # Anything else is a command line: a title or menu number, optionally
        # followed by launcher flags and title arguments, exactly as it would
        # be typed after `play` in a shell.
        line_args = parse_argv(tokens)
        if line_args is None:              # -h / --help printed its own text
            continue
        # menu-level overrides (play --camera 1) still apply unless the line
        # sets its own
        for key in ("camera", "width", "height", "smooth", "tof"):
            if getattr(line_args, key, None) is None:
                setattr(line_args, key, getattr(args, key, None))

        if line_args.title in ("list", "ls"):
            cmd_list(line_args)
            _pause()
            continue
        if line_args.title == "doctor":
            cmd_doctor(line_args)
            _pause()
            continue

        token = line_args.title or ""
        if token.isdigit() and 1 <= int(token) <= len(numbered):
            t = numbered[int(token) - 1]
        else:
            t = find_title(token)
        if t is None:
            _unknown_title(token)
            continue
        run_title(t, line_args)
        # hold before redrawing, or a fast-exiting title's output (a help
        # screen, a crash traceback) vanishes under the menu
        _pause()


# ── CLI ───────────────────────────────────────────────────────────────────────

#: Launcher flags that take a value, and the bare switches. Everything else on
#: the command line belongs to the title.
_VALUE_FLAGS = ("--camera", "--width", "--height", "--smooth", "--tof")
_SWITCHES = ("--json", "--no-cameras")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="play", description="Run any Visual AI game, demo, playground, or tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `play list` to see every title. Any flag this launcher does "
               "not own is passed straight to the game, so `play duckhunt "
               "--headless 900` works.",
    )
    parser.add_argument("title", nargs="?", help="title id, or 'list' / 'doctor' / 'info'")
    parser.add_argument("--camera", type=int, metavar="N", help="webcam index")
    parser.add_argument("--width", type=int, metavar="PX", help="capture width")
    parser.add_argument("--height", type=int, metavar="PX", help="capture height")
    parser.add_argument("--smooth", type=float, metavar="A",
                        help="One-Euro resting smoothness (was smooth_alpha)")
    parser.add_argument("--tof", choices=("sim", "off"), help="simulated ToF depth")
    parser.add_argument("--json", action="store_true", help="list: machine-readable output")
    parser.add_argument("--no-cameras", action="store_true",
                        help="doctor: skip the camera probe (it is the slow part)")
    return parser


def parse_argv(argv: list[str]) -> argparse.Namespace | None:
    """
    Split the launcher's own flags off the front by hand rather than leaning
    on parse_known_args, which returns the leftovers regrouped: positionals
    and unknown flags come back in separate lists, so `duckhunt --headless
    120` reassembled as `120 --headless` and the game rejected it. A manual
    scan is the only way to keep the game's argv in the order it was typed.

    Returns None when argparse printed help instead of parsing.
    """
    parser = build_parser()

    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1:]
    else:
        extra = []

    mine, passthrough = [], []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in _VALUE_FLAGS:
            mine += argv[index:index + 2]
            index += 2
            continue
        if token.split("=", 1)[0] in _VALUE_FLAGS or token in _SWITCHES:
            mine.append(token)
            index += 1
            continue
        if token in ("-h", "--help") and not passthrough:
            parser.print_help()
            return None
        passthrough.append(token)
        index += 1

    # The title is the first bare word; everything after it belongs to the game.
    title_arg = passthrough[0] if passthrough else None
    args = parser.parse_args(mine)
    args.title = title_arg
    args.rest = passthrough[1:]
    args.extra = extra + passthrough[1:]
    # A path means a script is driving us and wants only the game's own
    # output on stdout - see the `quiet` note in run_title().
    args.quiet = bool(title_arg) and title_arg.lower().endswith(".py")
    return args


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # __exec is the internal child-process mode, not a user-facing command.
    if argv and argv[0] == "__exec":
        return exec_entry(argv[1:])

    args = parse_argv(argv)
    if args is None:
        return 0

    if args.title in (None, "menu"):
        return cmd_menu(args)
    if args.title in ("list", "ls"):
        return cmd_list(args)
    if args.title == "doctor":
        return cmd_doctor(args)
    if args.title == "info":
        if not args.rest:
            print(red("  info needs a title, e.g. `play info sling`"))
            return 1
        args.title = args.rest[0]
        return cmd_info(args)

    title = find_title(args.title)
    if title is None:
        return _unknown_title(args.title)
    return run_title(title, args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
