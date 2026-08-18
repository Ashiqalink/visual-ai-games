"""
build_exe.py - freeze the games and the engine into a single executable.

The point is a build you can hand to someone who has no Python, no venv, and
no clone of either repo: they run one file and get the menu. The engine ships
inside the bundle, so there is no sibling folder to find.

    python tools/build_exe.py                  build it (a folder, in ../releases)
    python tools/build_exe.py --onefile        a single .exe instead
    python tools/build_exe.py --slim           drop onnxruntime (much smaller)
    python tools/build_exe.py --output DIR     somewhere else entirely

Nothing is written inside the repo. This is a rare, one-off act and the
bundle is a few hundred megabytes; it defaults to ../releases so a working
tree you use every day stays exactly as it was. Intermediates go to
<output>/.build and are deleted once the build succeeds.

Build it on the platform you are shipping to - PyInstaller does not
cross-compile, so a Windows .exe needs a Windows build.

Two things this has to get right, and which are easy to get wrong:

* mediapipe loads its models (.tflite, .binarypb) from data files inside the
  installed package at runtime, not at import time. PyInstaller's dependency
  analysis follows imports, so it does not see them and they must be collected
  explicitly. Without them hand tracking fails at the first frame with a
  missing-file error rather than at startup.
* The games are bundled as .py data files rather than imported modules,
  because play.py runs them through runpy.run_path with __name__ == "__main__"
  exactly as `python Sling/main.py` would. Keeping them as data keeps that one
  code path shared between the frozen and the source builds.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT.parent / "visual ai game engine"
ENGINE_SRC = ENGINE / "src"

NAME = "visual-ai-games"

#: Everything play.py can launch, plus the modules those games import from the
#: repo root. Bundled as data because they are executed by path, not imported.
GAME_DIRS = [
    "Sling", "flappy", "punchy", "avatarcatch",
    "duckhunt", "depthlanes", "signduel", "sculptor",
    "cradle", "conductor", "depthpong",
]
ROOT_MODULES = ["engine_bootstrap.py", "instructions.py", "labkit.py",
                "run_lab_tests.py"]

#: Pulled in at runtime by name, so the import graph never mentions them.
HIDDEN = [
    "visual_ai", "visual_ai.pipeline", "visual_ai.fallback_engine",
    "visual_ai.noise_filter", "visual_ai.tof_stabilizer",
    "visual_ai.jitter_analyzer", "visual_ai.math_utils",
    "visual_ai.gesture_math", "visual_ai.render3d", "visual_ai.material",
    "visual_ai.spritegen", "visual_ai.imaging", "visual_ai.matting",
    "visual_ai.gesture_mlp",
]

#: Heavy transitive dependencies none of the games touch.
#:
#: matplotlib is deliberately NOT in this list, tempting as it is. mediapipe's
#: __init__ imports its solutions package, which imports drawing_utils, which
#: imports matplotlib - so excluding it makes `import mediapipe` fail outright
#: and every game loses hand tracking. That failure surfaced only as a doctor
#: line reading "MISSING mediapipe" in a bundle that plainly contained it.
#: jax is the single biggest win here: mediapipe declares it for its
#: model_maker and genai converters, which these games never touch, and it
#: drags in a 226 MB jax_common.dll. The solutions API - hands, face, pose -
#: does not import it. Verified by running the bundle afterwards, since an
#: exclusion that breaks `import mediapipe` is exactly what matplotlib did.
EXCLUDE = [
    "jax", "jaxlib", "pandas", "scipy", "torch", "tensorflow", "IPython",
    "notebook", "jupyter", "pytest", "PyQt5", "PyQt6", "PySide2", "PySide6",
    "tkinter",
]


#: Files PyInstaller collects that nothing here loads. These arrive as
#: *binaries* pulled in by mediapipe's hook, not as imported modules, so
#: --exclude-module does not touch them - the only way out is to delete them
#: afterwards. Each entry must be verified by running the pruned bundle.
#:
#: jax_common.dll is 226 MB, an eighth of the whole build. mediapipe declares
#: jax for its model_maker and genai converters; the solutions API the games
#: use never loads it. Checked by running the bundle with the file removed:
#: mediapipe imports and hand tracking still initialises its XNNPACK delegate
#: against a live camera.
PRUNE = ["jax_common.dll"]


def prune(internal: Path) -> None:
    """Delete collected binaries nothing loads, reporting what went."""
    for name in PRUNE:
        path = internal / name
        if path.is_file():
            size = path.stat().st_size
            path.unlink()
            print(f"  pruned {name} ({size / 1_048_576:,.0f} MB)")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # onedir is the default. A --onefile bundle of this size re-extracts
    # ~600 MB to a temp directory on every single launch, which measured at
    # 30 s before the game window even appears. onedir pays that cost once,
    # when the zip is unpacked, and starts in a couple of seconds after.
    parser.add_argument("--onefile", action="store_true",
                        help="a single .exe: easier to send, ~30 s slower every launch")
    parser.add_argument("--slim", action="store_true",
                        help="drop onnxruntime; avatarcatch falls back to an uncut crop")
    # Everything lands outside the repo by default. This build is a rare,
    # one-off act; the ~400 MB bundle and ~40 MB of intermediates have no
    # business sitting in a working tree you use every day, gitignored or
    # not - they slow searches and editor indexing and clutter the folder.
    parser.add_argument("--output", metavar="DIR", default=str(ROOT.parent / "releases"),
                        help="where the build lands (default: ../releases, outside the repo)")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="keep the work directory instead of deleting it after a good build")
    args = parser.parse_args(argv)

    if not ENGINE_SRC.is_dir():
        print(f"error: the engine is not next to this repo (looked in {ENGINE_SRC})",
              file=sys.stderr)
        return 1

    output = Path(args.output).resolve()
    work = output / ".build"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    sep = ";" if sys.platform == "win32" else ":"

    def data(source: Path, dest: str) -> list[str]:
        return ["--add-data", f"{source}{sep}{dest}"]

    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", NAME,
        "--onefile" if args.onefile else "--onedir",
        # A console build on purpose: play.py IS a terminal menu, and a
        # windowed build would give a double-clicking user no visible output
        # at all - including the doctor report when something is wrong.
        "--console",
        str(ROOT / "play.py"),
        # The engine, as an importable package rather than loose data.
        "--paths", str(ENGINE_SRC),
        # Keep PyInstaller's three outputs - the bundle, the work directory
        # and the generated .spec - out of the repo. Left to itself it writes
        # dist/, build/ and a .spec into the current directory.
        "--distpath", str(output),
        "--workpath", str(work),
        "--specpath", str(work),
    ]

    for module in HIDDEN:
        command += ["--hidden-import", module]
    for module in EXCLUDE:
        command += ["--exclude-module", module]
    if args.slim:
        command += ["--exclude-module", "onnxruntime"]

    # mediapipe needs collect-all, not collect-data. Its solution modules are
    # imported by name at runtime, so PyInstaller's analysis finds neither the
    # submodules nor the .tflite/.binarypb models beside them; collect-data
    # alone produced a bundle whose doctor reported "MISSING mediapipe".
    command += ["--collect-all", "mediapipe"]
    # pygame reaches the bundle only through Sling's renderer spikes, which
    # play.py launches by path, so nothing in the import graph mentions it.
    command += ["--collect-all", "pygame"]

    # The engine's own assets: creature sprites and the pose model.
    if (ENGINE / "assets").is_dir():
        command += data(ENGINE / "assets", "assets")

    # The games themselves, from a filtered staging copy.
    #
    # --add-data takes a directory or nothing: it has no exclude option and
    # copies whatever it is pointed at, recursively. Sling/ carries its own
    # 1,064 MB .venv - gitignored, so it shows up in neither git status nor a
    # fresh clone - and pointing --add-data straight at the source tree
    # shipped that entire virtualenv inside the bundle. It was 71% of the
    # build. Stage a copy without the junk and bundle that instead.
    staging = work / "staged-games"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    skip = shutil.ignore_patterns(
        ".venv", "venv", "env", "__pycache__", "*.pyc", "*.pyo",
        ".agents", ".git", "*.log", "*.spec", "build", "dist",
    )
    for name in GAME_DIRS:
        directory = ROOT / name
        if directory.is_dir():
            shutil.copytree(directory, staging / name, ignore=skip)
            command += data(staging / name, name)
    for name in ROOT_MODULES:
        if (ROOT / name).is_file():
            command += data(ROOT / name, ".")

    print("  building", NAME, "(--onefile)" if args.onefile else "(--onedir)")
    print("  this takes a few minutes\n")
    result = subprocess.call(command, cwd=str(ROOT))
    if result != 0:
        return result

    if args.onefile:
        target = output / (NAME + (".exe" if sys.platform == "win32" else ""))
    else:
        target = output / NAME
        prune(target / "_internal")

    if not args.keep_intermediates:
        shutil.rmtree(work, ignore_errors=True)
    if target.exists():
        size = (sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                if target.is_dir() else target.stat().st_size)
        print(f"\n  built {target}  ({size / 1_048_576:,.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
