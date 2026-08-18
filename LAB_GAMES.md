# Lab games

Seven games that each push one part of the `visual_ai` SDK until it complains,
and report what they measured rather than only what you scored. They live
alongside Sling, flappy and punchy but share none of their code — those three
are untouched.

Each game runs two ways off the same source:

```bash
python duckhunt/duckhunt.py                 # windowed, real camera
python duckhunt/duckhunt.py --headless 900  # scripted input, JSON report
```

or through the launcher, which puts the engine on `PYTHONPATH` for you:

```bash
play duckhunt
play labtests --variants        # the whole headless suite
```

## What each one is for

| game | pushes | the number it produces |
| --- | --- | --- |
| `duckhunt` | face tracking — `target_x/y`, `face_box` | face-loss rate split by whether the hand is over the face |
| `depthlanes` | ToF depth bands + `ToFStabilizer` | signed timing error (ms), band-flicker count |
| `signduel` | `hand_sign` and its 3-frame debounce | sign latency in frames and ms |
| `sculptor` | `render3d` under sustained load | per-mesh render cost, p95, drift across a run |
| `cradle` | two hands, slot stability | two-hand tracking %, per-slot position jumps |
| `conductor` | One-Euro at a fast reversal | filter lag in ms, smoothed vs raw beat detection |
| `depthpong` | the whole `depth_grid` | payload bytes/frame, mask build cost |

## How to play any of them

Each game opens on a card built from its own `goal`, `controls` and `keys` — a
gesture game gives no hint that the hand should be a fist rather than a point,
or that depth matters and position does not, so the game has to say so. Any key
starts, `H` brings the card back, `--no-instructions` skips it. Frames spent
behind the card are kept out of the timing summary and the game does not update,
so reading the rules costs neither a life nor a percentile. Headless runs never
show it.

The card itself is `instructions.py` at the repo root rather than part of this
harness, because Sling, flappy and punchy show the same one and are no part of
the harness.

## Is it tracking?

Every windowed run carries a `HAND TRACKING` badge in the bottom-right corner,
drawn by the driver so all seven get it without a line of their own. A game that
ignores you and a game that cannot see your hand look identical from the
player's chair, and the fix for each is different:

| badge | what it means |
| --- | --- |
| `STARTING` | no payload has arrived yet — the camera is still warming up |
| `NO CAMERA` | the camera never opened; the pipeline is emitting simulated input |
| `ON` / `ON x2` | a hand was seen this frame, and how many |
| `LOST` | no hand for up to 1.5 s |
| `OFF` | no hand beyond that, or none seen at all |

The state hangs off one clock — time since a payload last reported a hand — so a
single dropped MediaPipe detection does not strobe the badge, and a pipeline
thread that stalls or dies flips it off rather than leaving it stuck on `ON`.
The session summary a windowed run prints on exit carries the same figure over
the whole run as `hand_tracked_pct`.

Headless runs draw no badge; they already report tracking through their own
metrics.

## Headless mode

`--headless N` runs N frames with no window and no camera, driving the game from
`labkit.ScriptedPipeline`, then prints a JSON report: frame timings, any
exception raised in update or render, and the game's own metrics.

Two levels of scripting sit behind it. A **payload script** overrides payload
keys directly — enough for face or depth work. A **landmark script** returns
synthetic MediaPipe landmarks that go through the engine's real
`_extract_gesture`, so the sign debounce, the One-Euro filters and the velocity
difference are genuinely exercised rather than written over. `hand_landmarks()`
shapes fingertips against the same wrist-distance test the classifier applies;
all four signs round-trip through it.

Scripted hands are routed through the engine's own `_assign_slots`, so a
two-handed test exercises the real slot matching instead of being handed slots
by list index.

## The suite

```bash
python run_lab_tests.py                    # all seven, exit 0 if all ok
python run_lab_tests.py --variants         # plus the comparison runs
python run_lab_tests.py --json out.json    # full reports, for diffing runs
python run_lab_tests.py --only conductor cradle
```

The variant runs exist to compare a setting against its default rather than to
check that a game works — stabilizer on/off, calibrated while still versus while
moving, `filter_beta` swept, depth grid resized — and they roughly double the
runtime.

## Useful flags

```bash
python depthlanes/depth_lanes.py --headless 1200 --stabilize
python depthlanes/depth_lanes.py --headless 1200 --stabilize-at 300   # calibrate mid-motion
python conductor/conductor.py    --headless 900  --beta 0.05
python sculptor/sculptor.py      --headless 1200 --soak sphere
python depthpong/depth_pong.py   --headless 600  --grid 160x120
```

## Scope

All of this is **game side**. The engine changes these depend on are in the
engine repo on `feature/multi-hand-motion-depth-payload`: the payload now carries
every tracked hand, fingertip velocity, the face box, and an opt-in depth grid.
`labkit.py` is the harness the games are measured with and is not part of the
tracking contract — nothing in it belongs in the SDK.
