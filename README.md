# visual ai games

A collection of camera-controlled games built on the
[visual-ai-game-engine](https://github.com/Ashiqalink/visual-ai-game-engine).
You play them with your hands and body in front of a webcam - no controller.

## Quick start

You need **Python 3.9-3.12**. Not 3.13: mediapipe, which does the hand
tracking, publishes no wheels for it and the install will fail. Check with
`python --version` before anything else.

```bash
# 1. Clone both repos side by side. The games import the engine from a
#    sibling folder, so cloning this one alone is not enough.
git clone https://github.com/Ashiqalink/visual-ai-game-engine.git "visual ai game engine"
git clone https://github.com/Ashiqalink/visual-ai-games.git "visual ai games"

# 2. Install the dependencies.
cd "visual ai games"
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt

# 3. Check the machine is ready, then play.
python play.py doctor
python play.py
```

`play.py doctor` reports your Python version, every dependency, whether the
engine was found, which physics engine you got, and which webcams respond. If
something is wrong it says what and how to fix it - run it first.

You do **not** need to install the engine. `engine_bootstrap.ensure_engine()`
finds the sibling clone on `sys.path` at startup. Installing it is optional and
only adds the compiled C++ physics core:

```bash
pip install -e "../visual ai game engine"
```

That build needs a C++ compiler, and **it is fine if you don't have one** - the
install prints why it skipped the extension and carries on. The games then run
on `PythonFallbackEngine`, which has the same API and the same physics, just
slower. `python play.py doctor` tells you which one you ended up with.

If you keep the engine somewhere other than a sibling folder, point at it:

```bash
set VISUAL_AI_ENGINE=C:\path\to\engine     # Windows
export VISUAL_AI_ENGINE=/path/to/engine    # macOS / Linux
```

## Running

`play.py` is the entry point for everything. It finds the engine, puts it on
the child's `PYTHONPATH`, and can override settings a game hard-codes.

```bash
python play.py                    # interactive menu
python play.py list               # every title, and whether it will run
python play.py doctor             # Python, deps, engine, cameras
python play.py sling              # run a game
python play.py info sling         # its controls, without launching
python play.py punchy --camera 1  # a different webcam
python play.py sling --width 1920 --height 1080 --smooth 0.3
python play.py flappy --tof sim   # force simulated ToF depth ("sim" | "off")
python play.py labtests           # the headless regression suite
```

On Windows `play sling` works too (via `play.cmd`); on macOS and Linux it is
`./play sling`. Any flag the launcher does not own goes straight to the game,
so `python play.py duckhunt --headless 900` does what it looks like.

You can still run a game directly if you prefer - `python Sling/main.py` - but
then the launcher's overrides are not available.

## Sending it to someone who has no Python

`tools/build_exe.py` freezes the games *and* the engine into a standalone
build. Whoever you send it to needs no Python, no venv, and no clone of either
repo - they unzip it and double-click.

```bash
pip install pyinstaller
python tools/build_exe.py               # a folder in dist/, ~1.5 GB
python tools/build_exe.py --onefile     # a single .exe instead
python tools/build_exe.py --slim        # drop onnxruntime; smaller
```

Zip `dist/visual-ai-games/` and send that. They run `visual-ai-games.exe` for
the menu, or `visual-ai-games.exe doctor` if anything looks wrong.

Two things worth knowing before you rely on it:

- **PyInstaller does not cross-compile.** A Windows `.exe` must be built on
  Windows, a macOS build on macOS.
- **The folder build is the default on purpose.** `--onefile` re-extracts the
  whole ~1.5 GB bundle to a temp directory on *every* launch, measured at 30
  seconds before anything appears. The folder build pays that once and then
  starts in a few seconds.

## Games

| Game | Input | Entry point |
| --- | --- | --- |
| **Sling** | Hand signs - close a fist to grab the bird, open your hand to fire | `Sling/main.py` |
| **flappy** | Finger tracking - raise and lower your index finger to fly | `flappy/tof_flappy.py` |
| **punchy** | Time-of-Flight depth - punch toward the camera to hit targets | `punchy/tof_punch.py` |
| **avatarcatch** | Your own webcam portrait, cut out and used as the paddle | `avatarcatch/avatar_catch.py` |

Plus seven **lab games** - `duckhunt`, `depthlanes`, `signduel`, `sculptor`,
`cradle`, `conductor`, `depthpong` - which each push one part of the SDK until
it complains and report what they measured, not just what you scored. Each also
runs headless off scripted input, so they double as the regression suite. See
[LAB_GAMES.md](LAB_GAMES.md).

### Controls

**Every game opens on a how-to-play card** — what you are trying to do, what
your hands do, and which keys exist. Any key starts the game, and `H` brings the
card back mid-game. Nothing moves and no gesture is read while it is up, so a
fist held while reading cannot grab a bird. The lab games take
`--no-instructions` to skip it.

Sling reads hand signs rather than pinch gestures. Make a **fist** over a
bird in the top carousel to grab it — aiming starts immediately — move the fist
to pull the slingshot, then **open your hand** to fire. To swap birds, open your
hand and lift it back into the carousel strip.

Both Sling and flappy share two stabilisation keys, because there are two
independent smoothing systems and they are worth feeling separately:

| Key | Effect |
| --- | --- |
| `H` | Show the how-to-play card again. |
| `K` | Toggle One-Euro landmark smoothing. Off = raw positions: maximally responsive, visibly jittery at rest. |
| `L` | Toggle the ToF depth stabilizer, which runs a 3-second hold-still calibration. |
| `X` | Cancel a calibration already in progress. |

Every game needs a webcam, and `Q` or `ESC` quits. With no camera present the
pipeline runs in simulated mode rather than crashing, so a headless machine can
still run the suite.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `ERROR: Could not find a version that satisfies the requirement mediapipe` | Python 3.13+. Make the venv with a 3.12 interpreter. |
| "Cannot start: the visual_ai engine is unavailable" | The engine repo is not a sibling folder. Clone it, or set `VISUAL_AI_ENGINE`. |
| Install warns the C++ extension was not built | Expected without a compiler. Harmless - you get the Python fallback. |
| The window opens but nothing tracks | Another app holds the webcam, or the wrong index. `python play.py doctor` lists what responds; `--camera N` picks one. |
| Tracking is jittery at rest | Press `K`; if depth is the problem press `L` and hold still for 3 seconds. |

## What these games do with your camera

Every game here points a webcam at you, so it is worth being precise about
where those frames go: **nowhere**. They are read from the camera, passed
through MediaPipe in memory, and dropped when the next frame arrives. Nothing
in this repo writes an image or a video to disk, and nothing uploads one —
there is no `imwrite`, no `VideoWriter`, and no analytics or telemetry of any
kind. Closing the window is all it takes to be rid of the data.

`avatarcatch` is the one game that keeps a frame rather than discarding it: it
photographs you once and uses the cutout as your paddle for the session. It
asks first — the countdown does not start until you press `SPACE`, and until
then you get a preview without a capture. The photo lives in memory only,
`R` discards it, and quitting takes it with you.

Two features fetch a model file the first time you use them, and only then:

| Feature | Downloads | From |
| --- | --- | --- |
| `avatarcatch` portrait matting | MODNet weights, 25 MB, pinned to a fixed revision and SHA-256 verified | huggingface.co |
| Person-mask segmentation | MediaPipe selfie segmenter | storage.googleapis.com |

Both land in a per-user cache and are reused afterwards. If you would rather
fetch them yourself, point `VISUAL_AI_MODNET_ONNX` at a local copy of the
weights and no download is attempted. Everything else runs fully offline.

## License

The games in this repo are MIT licensed — download them, play them, fork them,
build on them. See `LICENSE`.

That covers **this repo only**. The engine they import is a separate,
proprietary project: it is not open source, carries no license grant, and
nothing here gives you any right to it. The MIT grant above reaches the game
code and nothing underneath it.

## Repo layout notes

`Sling/` was imported from
[sling](https://github.com/Ashiqalink/sling)
with its full commit history preserved, so that repo remains its upstream.

To pull later changes from upstream into this repo:

```bash
git fetch old-repo main
git merge -s ours --no-commit --allow-unrelated-histories FETCH_HEAD
git read-tree --prefix=Sling/ -u FETCH_HEAD
git commit -m "sync Sling from upstream"
```

where `old-repo` is the remote pointing at `sling.git`.
