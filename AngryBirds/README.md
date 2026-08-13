# Angry Birds — OpenCV Gesture-Controlled Game

> A physics-based Angry Birds clone driven entirely by **hand gestures** via webcam, powered by MediaPipe, OpenCV, and a custom [Visual AI Game Engine SDK](file:///d:/visual%20ai%20game%20engine).

> 🎮 **Design Principle:** This game is supposed to be a laid back and enjoy game, not a fast paced game.


---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Vision / Input** | MediaPipe Hands, OpenCV | Real-time hand tracking, gesture detection (pinch + Z-push click) |
| **Physics Engine** | Custom Python (`physics.py`) | Gravity, AABB collision, impulse resolution, projectile motion |
| **Rendering** | OpenCV drawing primitives (`cv2`) | All game art — birds, blocks, slingshot, HUD, trajectory — rendered frame-by-frame |
| **Game Logic** | Python state machine (`game.py`) | SELECTION → ARMED → FLIGHT → DONE lifecycle |
| **AI SDK** | `visual_ai` (C++ / pybind11 + Python fallback) | Threaded camera pipeline, smoothed gesture queue, physics core |
| **Web Frontend** | Vite + Vanilla JS + HTML Canvas | Browser-based alternative UI with WebRTC hand tracking |

---

## 📁 Project Structure

```
angry-birds-using-opencv/
├── main.py              # Entry point — game loop, camera preview, visual_ai integration
├── game.py              # State machine (SELECTION → ARMED → FLIGHT → DONE), scoring, levels
├── bird.py              # Bird class — 5 types (Red, Chuck, Bomb, Blues, White), trail/impact FX
├── block.py             # Block class — material system (wood/stone/ice), health, crack overlays
├── slingshot.py         # Slingshot rendering — catenary elastics, snap-back animation
├── physics.py           # Constants + helpers (gravity, AABB, impulse, Z-click thresholds)
├── hand_tracker.py      # MediaPipe wrapper — pinch detection + Z-push click algorithm
├── ui.py                # HUD, carousel, trajectory preview, score popups, overlays
├── web/                 # Browser-based alternative (Vite + JS)
│   ├── index.html
│   └── src/
│       ├── main.js
│       ├── game.js
│       ├── gameScene.js
│       ├── bird.js
│       ├── block.js
│       ├── slingshot.js
│       ├── physics.js
│       ├── handTracker.js
│       ├── ui.js
│       └── style.css
└── .agents/skills/      # AI Agent Customization Skills for Antigravity
    └── angry-birds-opencv/
        └── SKILL.md
```

---

## 🧩 Module Breakdown

### `main.py` — Entry Point
- Initialises `VisionPipeline` (threaded camera) and `GameEngine` from `visual_ai` SDK
- Runs the main game loop at ~30+ FPS
- Drains the AI queue each frame for the latest gesture data
- Composites game canvas (dark background) + camera preview (top-right box)
- Renders hand cursor overlay (index ring, pinch dot, Z-click flash)

### `game.py` — Game State Machine
- **States:** `SELECTION` → `ARMED` → `FLIGHT` → `DONE`
- **Levels:** 3 built-in layouts (Easy / Medium / Hard) using wood, stone, ice blocks
- **Scoring:** 500 pts per block, 100 pts per debris; floating `+N` popups
- **Controls:** Carousel selection via hand X-position with debounce; aiming via EMA-smoothed finger tracking; firing via Z-push or edge-exit

### `bird.py` — Bird Entities
- 5 bird types with unique mass, radius, colour, and drawn artwork
- Flight trail (fading circles), speed lines, impact pop ring
- Floor bounce + linger timer for realistic grounding behaviour
- Multi-hit tracking (`_hit_blocks` set) — bird punches through weak blocks

### `block.py` — Destructible Blocks
- **Material system:** wood (medium), stone (tough), ice (fragile)
- Health bar shown only when damaged; crack overlays at 70% and 40% thresholds
- Physics: gravity, floor bounce, angular rotation from impulse
- Debris spawning on destruction (4 quarter-size fragments)

### `slingshot.py` — Slingshot Rendering
- Depth-layered draw order: back elastic → bird → front elastic → wooden structure
- Catenary curve for elastic bands (parabolic sag reduces when stretched)
- Elastic colour and thickness scale with pull distance
- Snap-back vibration animation on bird release

### `physics.py` — Physics Engine
- Constants: gravity, floor Y, restitution, power factor, max pull, air drag
- AABB circle-vs-rect collision detection
- Momentum-based block-block collision resolution with mass from area × density
- Z-click thresholds and pinch distance constants

### `hand_tracker.py` — Gesture Recognition
- MediaPipe Hands wrapper (single hand, 21 landmarks)
- **Pinch click:** thumb-tip ↔ index-tip distance < 30 px
- **Z-push click:** index fingertip moves ~1 inch toward camera within 8-frame window, with XY drift guard (< 30 px)
- Cooldown system (20 frames) to prevent double-fire
- Index finger isolation detection (only index extended, others curled)

### `ui.py` — HUD & Overlays
- Bird selection carousel with glow ring on selected bird
- Dotted parabolic trajectory preview during ARMED state
- Score display, level indicator, FPS counter
- Click-mode indicator (Z-PUSH vs PINCH) with Z-push debug bar
- Semi-transparent "ALL BIRDS USED" end screen with final score

---

## 🎮 Controls

| Input | Action |
|---|---|
| **Move hand left/right** | Scroll bird carousel (SELECTION) / Aim slingshot (ARMED) |
| **Z-push** (push finger toward camera) | Select bird / Fire bird |
| **Pinch** (thumb + index) | Alternative select/fire |
| **Move hand to screen edge** | Fire bird (after hand was in safe zone) |
| **R** | Restart current level |
| **1 / 2 / 3** | Switch to Easy / Medium / Hard level |
| **Q / ESC** | Quit |

---

## 🔗 Visual AI SDK Integration
Warning!! mediapipe currently works only in 3.12 or below so do not upgradee!!

This game depends on the **Visual AI Game Engine SDK** (`visual_ai`) located at `../visual ai game engine/`.

The SDK provides:
- **`VisionPipeline`** — Threaded camera capture + MediaPipe hand detection with EMA smoothing
- **`GameEngine`** — C++ pybind11 physics core (falls back to pure Python if C++ build is unavailable)
- **Thread-safe queue** — Gesture data is pushed to a `queue.Queue(maxsize=1)` so the game loop always gets the freshest frame

---

## 🧪 Running the Game

### Python (Desktop — OpenCV Window)
```bash
pip install opencv-python mediapipe numpy
python main.py
```

### Web (Browser — Vite Dev Server)
```bash
cd web
npm install
npm run dev
```

---

## 📐 Key Constants

| Constant | Value | Location |
|---|---|---|
| `GRAVITY` | 0.45 px/frame² | `physics.py` |
| `FLOOR_Y` | 660 px | `physics.py` |
| `MAX_PULL` | 150 px | `physics.py` |
| `POWER_FACTOR` | 0.18 | `physics.py` |
| `PINCH_THRESHOLD` | 30 px | `physics.py` |
| `Z_CLICK_THRESHOLD_M` | 0.012 (~0.5 inch) | `physics.py` |
| `Z_CLICK_XY_MAX_PX` | 30 px | `physics.py` |
| `AIR_DRAG` | 0.998 | `physics.py` |
| `BIRD_BOUNCE` | 0.35 | `physics.py` |
| `BIRD_LINGER` | 90 frames | `physics.py` |
| `SLING_X / SLING_Y` | 280 / 440 px | `slingshot.py` |
| `FRAME_W / FRAME_H` | 1280 × 720 | `main.py` |
| `smooth_alpha` | 0.40 | `main.py` (pipeline config) |
