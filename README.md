# visual ai games

A collection of camera-controlled games built on the
[visual-ai-game-engine](https://github.com/Ashiqalink/visual-ai-game-engine).
You play them with your hands and body in front of a webcam - no controller.

## Games

| Game | Input | Entry point |
| --- | --- | --- |
| **AngryBirds** | Hand signs - close a fist to grab the bird, open your hand to fire | `AngryBirds/main.py` |
| **flappy** | Finger tracking - raise and lower your index finger to fly | `flappy/tof_flappy.py` |
| **punchy** | Time-of-Flight depth - punch toward the camera to hit targets | `punchy/tof_punch.py` |

### Controls

AngryBirds reads hand signs rather than pinch gestures. Make a **fist** over a
bird in the top carousel to grab it — aiming starts immediately — move the fist
to pull the slingshot, then **open your hand** to fire. To swap birds, open your
hand and lift it back into the carousel strip.

Both AngryBirds and flappy share two stabilisation keys, because there are two
independent smoothing systems and they are worth feeling separately:

| Key | Effect |
| --- | --- |
| `K` | Toggle One-Euro landmark smoothing. Off = raw positions: maximally responsive, visibly jittery at rest. |
| `L` | Toggle the ToF depth stabilizer, which runs a 3-second hold-still calibration. |
| `X` | Cancel a calibration already in progress. |

## Setup

The games import the engine from a **sibling directory**, not from a package
index. Clone both repos next to each other, or nothing will import:

```
your-folder/
├── visual ai game engine/    <- clone this too
└── visual ai games/          <- this repo
```

```bash
git clone https://github.com/Ashiqalink/visual-ai-game-engine.git "visual ai game engine"
git clone https://github.com/Ashiqalink/visual-ai-games.git "visual ai games"

cd "visual ai games"
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r AngryBirds/requirements.txt
```

If you keep the engine somewhere else, point at it with an environment
variable instead:

```bash
set VISUAL_AI_ENGINE=C:\path\to\engine    # Windows
export VISUAL_AI_ENGINE=/path/to/engine   # macOS/Linux
```

Every game calls `ensure_engine()` from `engine_bootstrap.py` at startup. It
locates the engine and, if it can't, exits with a message saying exactly what
to clone and where - rather than a bare `ModuleNotFoundError`.

### About the compiled extension

The engine includes a C++ extension (`engine_core`, built with pybind11). The
binary checked into that repo is built for **CPython 3.12 on Windows x86-64**.
On any other Python version or platform you will need to rebuild it, which
requires a C++ compiler:

```bash
pip install -e "../visual ai game engine"
```

## Running

```bash
python AngryBirds/main.py
python flappy/tof_flappy.py
python punchy/tof_punch.py
```

Each game needs a webcam. Press `q` to quit.

## Repo layout notes

`AngryBirds/` was imported from
[angry-birds-using-opencv](https://github.com/Ashiqalink/angry-birds-using-opencv)
with its full commit history preserved, so that repo remains its upstream.

To pull later changes from upstream into this repo:

```bash
git fetch old-repo main
git merge -s ours --no-commit --allow-unrelated-histories FETCH_HEAD
git read-tree --prefix=AngryBirds/ -u FETCH_HEAD
git commit -m "sync AngryBirds from upstream"
```

where `old-repo` is the remote pointing at `angry-birds-using-opencv.git`.
