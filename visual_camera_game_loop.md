# Angry Birds OpenCV — Visual Camera Game Loop Spec

A design + implementation plan for the core game loop of a gesture-controlled physics-based slingshot game (birds vs. pigs, destructible block structures) using a webcam and AI hand tracking.

---

## 1. Tech Assumptions

- **Rendering**: OpenCV (`cv2.imshow`), 3D overlays via custom `visual_ai` engine
- **Physics**: Python custom lightweight physics or Pymunk for rigid-body collisions
- **Loop driver**: `while` loop bound to webcam frame capture (`cap.read()`)
- **Language**: Python
- **Input**: MediaPipe hand tracking (Index + Thumb pinch for grab, distance for pull)

---

## 2. High-Level Game States

```
MENU → LEVEL_LOAD → AIMING → LAUNCHED → SETTLING → (LEVEL_COMPLETE | LEVEL_FAILED) → NEXT_LEVEL / RETRY
```

| State | Description |
|---|---|
| `MENU` | Title screen overlay on camera feed; use hand to point/select |
| `LEVEL_LOAD` | Spawn slingshot, birds queue, pigs, blocks/terrain on screen |
| `AIMING` | Player pinches fingers to grab bird; hand movement translates to slingshot pull |
| `LAUNCHED` | Fingers released (pinch ends), physics sim running |
| `SETTLING` | All bodies below velocity threshold; check win/lose |
| `LEVEL_COMPLETE` | All pigs destroyed — show star rating, score overlay |
| `LEVEL_FAILED` | No birds left, pigs remain — offer retry |

---

## 3. Core Loop Skeleton

```python
import cv2
import time
from visual_ai import HandTracker

last_time = time.time()
FIXED_DT = 1 / 60.0
accumulator = 0.0

tracker = HandTracker()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    current_time = time.time()
    frame_time = min(current_time - last_time, 0.25)
    last_time = current_time
    accumulator += frame_time

    # 1. AI Vision & Input Processing
    hands = tracker.process(frame)
    handle_gesture_input(hands)

    # 2. Fixed-step physics for determinism
    while accumulator >= FIXED_DT:
        update_physics(FIXED_DT)
        accumulator -= FIXED_DT

    # 3. Game Logic
    update_game_state()
    update_camera_viewport()
    
    # 4. Rendering
    render(frame, accumulator / FIXED_DT)
    
    cv2.imshow("Angry Birds AI", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 4. Subsystems Called Each Frame

### 4.1 `handle_gesture_input(hands)`
- **AIMING state**: 
  - Detect Index + Thumb pinch over the slingshot area.
  - Track hand position relative to anchor, clamp to max pull radius.
  - Compute launch vector `(dx, dy) * power_scale`.
- **On release**: Pinch distance > threshold → set bird velocity, transition to `LAUNCHED`.
- Support secondary gestures (e.g., tap other hand) for special abilities mid-flight.

### 4.2 `update_physics(dt)`
- Integrate velocity/position for dynamic bodies.
- Apply gravity: `v.y += GRAVITY * dt`.
- Perform collision detection and resolution (Pymunk space step).
- Apply damage on collision impacts; trigger debris particles and block states.

### 4.3 `update_game_state()`
```python
if state == AIMING:
    if gesture_released():
        state = LAUNCHED
elif state == LAUNCHED:
    if all_bodies_settled() or bird_out_of_bounds():
        state = SETTLING
elif state == SETTLING:
    if pigs_remaining == 0:
        state = LEVEL_COMPLETE
    elif birds_remaining == 0:
        state = LEVEL_FAILED
    else:
        spawn_next_bird()
        state = AIMING
```

### 4.4 `update_camera_viewport()`
- OpenCV rendering is fixed-window, but logical camera can translate all rendering coordinates to simulate following the bird or panning back.

### 4.5 `render(frame, interpolation)`
- Draw background layers using alpha blending over the camera frame.
- Draw terrain, slingshot, blocks, pigs, and birds.
- Apply 3D perspective transforms via `visual_ai` for blocks/characters.
- Draw trajectory dots while `AIMING`.
- Draw UI/HUD (birds remaining, score, gesture feedback markers).

---

## 5. Entity Data Shape (suggested)

```python
class Entity:
    def __init__(self):
        self.id = 0
        self.type = 'bird' # 'pig' | 'block'
        self.material = 'wood' 
        self.pos = [x, y]
        self.vel = [vx, vy]
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.hp = 100
        self.is_static = False
        self.is_sleeping = False
```

---

## 6. Win/Lose & Scoring
- Same as original: Score = block destruction + pig kill + remaining-bird bonus.
- Save to local JSON file or SQLite for persistent scores.

---

## 7. Performance Notes
- MediaPipe processing can bottleneck the loop. Consider running it in a separate thread or at a lower FPS (e.g., 30 FPS) while running physics and rendering at 60 FPS using interpolation.
- Limit the number of destructible particles rendered via `cv2.circle`/`cv2.fillPoly`.
