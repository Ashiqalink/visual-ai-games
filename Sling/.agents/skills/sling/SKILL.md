---
name: angry-birds-opencv
description: Use when working on, modifying, running, or debugging the Angry Birds OpenCV gesture-controlled game codebase (Python OpenCV/MediaPipe desktop & Vite/JS web client).
---

# Angry Birds OpenCV — Agent Skill

This skill provides architectural guidance, file maps, physics parameters, gesture thresholds, and execution workflows for the **Angry Birds OpenCV** codebase.

---

## 🏗️ Architecture & Component Map

| File | Primary Role | Key Dependencies |
|---|---|---|
| [main.py](file:///d:/angry%20birds%20using%20opencv/main.py) | Entry point, camera preview rendering, AI queue draining, gesture cursor overlay | `visual_ai`, `game.py`, `cv2` |
| [game.py](file:///d:/angry%20birds%20using%20opencv/game.py) | Game loop state machine (`SELECTION` → `ARMED` → `FLIGHT` → `DONE`), level management, scoring | `bird.py`, `block.py`, `slingshot.py`, `ui.py` |
| [bird.py](file:///d:/angry%20birds%20using%20opencv/bird.py) | Bird entity definition (5 types: Red, Chuck, Bomb, Blues, White), trails, bounce & multi-hit logic | `physics.py` |
| [block.py](file:///d:/angry%20birds%20using%20opencv/block.py) | Material-based blocks (wood, stone, ice), damage cracks, rotation, debris spawning | `physics.py` |
| [slingshot.py](file:///d:/angry%20birds%20using%20opencv/slingshot.py) | Slingshot rendering, catenary curve sag calculation, snap-back recoil animation | `cv2`, `numpy` |
| [physics.py](file:///d:/angry%20birds%20using%20opencv/physics.py) | Constants (gravity, floor level, thresholds), AABB collision, momentum transfer | `math` |
| [hand_tracker.py](file:///d:/angry%20birds%20using%20opencv/hand_tracker.py) | MediaPipe hand tracking wrapper, pinch distance, Z-push depth movement detector | `mediapipe`, `cv2` |
| [ui.py](file:///d:/angry%20birds%20using%20opencv/ui.py) | HUD, carousel selection UI, parabolic trajectory projection, score popups | `cv2` |
| [web/](file:///d:/angry%20birds%20using%20opencv/web) | Alternative browser frontend built with Vite, HTML5 Canvas, and WebRTC MediaPipe JS | Vite, JS ES Modules |

---

## ⚙️ Physics & Control Constants

When tweaking gameplay balance or gesture sensitivity, reference and adjust these canonical parameters in [physics.py](file:///d:/angry%20birds%20using%20opencv/physics.py) and [main.py](file:///d:/angry%20birds%20using%20opencv/main.py):

```python
# physics.py
GRAVITY             = 0.45     # Downward acceleration (px/frame²)
FLOOR_Y             = 660      # Ground pixel boundary
RESTITUTION         = 0.25     # Block/floor bounce elasticity
POWER_FACTOR        = 0.18     # Slingshot pull distance to launch velocity ratio
MAX_PULL            = 150      # Maximum allowed slingshot pull radius (px)
PINCH_THRESHOLD     = 30       # Max distance (px) between thumb-tip and index-tip for pinch
Z_CLICK_THRESHOLD_M = 0.025    # Forward movement threshold (~1 inch in MediaPipe Z scale)
Z_CLICK_XY_MAX_PX   = 30       # Max allowed X/Y drift during Z-push
AIR_DRAG            = 0.998    # Per-frame velocity decay
BIRD_BOUNCE          = 0.35     # Bird ground bounce coefficient
BIRD_LINGER          = 90       # Frames bird stays active after impact

# main.py
smooth_alpha        = 0.40     # EMA filter smoothing factor (higher = faster tracking, less lag)
```

---

## 🔄 Game Lifecycle State Machine

1. **`SELECTION`**:
   - Player moves hand left/right to scroll the bird carousel.
   - Triggering a click (Z-push or Pinch) equips the highlighted bird and advances to `ARMED`.
2. **`ARMED`**:
   - Bird is attached to slingshot at `(280, 440)`.
   - Aim vector is determined by hand position relative to slingshot center (constrained to `MAX_PULL`).
   - Parabolic trajectory line is rendered.
   - Click trigger or moving hand to screen edge releases the bird, transitioning to `FLIGHT`.
3. **`FLIGHT`**:
   - Bird moves under physics velocity & gravity.
   - AABB collision checks with blocks and floor are active each frame.
   - When bird grounds or comes to rest, transition to `DONE` (or next bird in carousel).
4. **`DONE`**:
   - Level complete / Game over evaluation.

---

## 🛠️ Verification & Testing Commands

### Python Desktop App
```bash
# Verify OpenCV and MediaPipe dependencies
python main.py
```

### Web App (Vite)
```bash
cd web
npm install
npm run dev
```

---

## 📝 Modification Guidelines

- **Adding a new bird type:** Update [bird.py](file:///d:/angry%20birds%20using%20opencv/bird.py) `Bird` class with type properties (color, mass, radius, special ability) and add it to `CAROUSEL_BIRDS` in [game.py](file:///d:/angry%20birds%20using%20opencv/game.py).
- **Adding a new level:** Define block layouts in [game.py](file:///d:/angry%20birds%20using%20opencv/game.py) `LEVELS` array specifying `(x, y, w, h, material)`.
- **Adjusting gesture sensitivity:** Modify `smooth_alpha` in [main.py](file:///d:/angry%20birds%20using%20opencv/main.py) or `PINCH_THRESHOLD` / `Z_CLICK_THRESHOLD_M` in [physics.py](file:///d:/angry%20birds%20using%20opencv/physics.py).

---

## ⚡ Token-Efficient Editing Strategy (CSS & Minor Tweaks)

To minimize token usage and context load during minor edits (such as CSS styling, UI layout tweaks, or parameter tweaks):

1. **Targeted Reading (`view_file` line ranges):**
   - ALWAYS specify `StartLine` and `EndLine` when reading CSS files (`web/src/style.css`), UI layout files (`web/src/ui.js`, `ui.py`), or configuration files.
   - Do NOT load entire files into memory when modifying localized selectors or variables.

2. **Precise Surgical Edits (`replace_file_content`):**
   - NEVER use `write_to_file` to rewrite full files for CSS or small tweaks.
   - Use `replace_file_content` targeting strictly the lines being modified.

3. **Selector Location via `grep_search`:**
   - Use `grep_search` to directly find CSS selectors (e.g. `.camera-preview`, `#game-canvas`, `.hud-overlay`) or variable definitions instead of browsing files manually.

4. **Concise Tool Responses:**
   - Avoid re-printing entire files or large code snippets in chat outputs; report only the modified lines or exact diff snippet.

---

## 🎯 Target Scope Disambiguation Rule

Whenever asking questions, proposing designs, or suggesting code changes, **ALWAYS explicitly state which repository / scope the change applies to:**

- ⚙️ **Game Engine SDK (`visual_ai`)**: Camera pipeline, MediaPipe hand tracking, EMA smoothing, threaded frame queue, pybind11 C++ physics core (`d:\visual ai game engine`).
- 🎮 **The Game (`angry-birds-opencv`)**: Game loop state machine, level layouts, bird/block entities, slingshot rendering, HUD UI (`d:\angry birds using opencv`).


