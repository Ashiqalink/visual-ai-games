# visual ai games

A collection of camera-controlled games built on the
[visual-ai-game-engine](https://github.com/Ashiqalink/visual-ai-game-engine).
You play them with your hands and body in front of a webcam - no controller.

## Games

| Game | Input | Entry point |
| --- | --- | --- |
| **Sling** | Hand signs - close a fist to grab the bird, open your hand to fire | `Sling/main.py` |
| **flappy** | Finger tracking - raise and lower your index finger to fly | `flappy/tof_flappy.py` |
| **punchy** | Time-of-Flight depth - punch toward the camera to hit targets | `punchy/tof_punch.py` |

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
pip install -r Sling/requirements.txt
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
python Sling/main.py
python flappy/tof_flappy.py
python punchy/tof_punch.py
```

Each game needs a webcam. Press `q` to quit.

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
