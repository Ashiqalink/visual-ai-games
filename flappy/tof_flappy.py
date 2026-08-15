import os
import sys
import time
import queue
import random
import cv2
import numpy as np

# Add visual_ai engine path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, '..'))

from engine_bootstrap import ensure_engine
ensure_engine()

from instructions import draw_card
from visual_ai import VisionPipeline

W, H = 800, 600

TITLE = "ToF Flappy — fly with your fingertip"
GOAL = ("Fly the bird through the gaps in the pipes. Hitting a pipe, the "
        "ceiling or the floor ends the run.")
CONTROLS = (
    ("Hold up one hand",
     "the bird follows the height of your index fingertip — nothing else "
     "on your hand matters"),
    ("Raise your finger", "the bird climbs"),
    ("Lower your finger", "the bird dives"),
    ("Stay in the middle band",
     "only the middle stretch of the camera frame is mapped to the screen, "
     "so you never have to reach to the very edge"),
    ("Lose tracking",
     "the bird drifts back to the centre instead of dropping — a brief "
     "dropout is survivable"),
)
KEYS = (
    ("R", "restart after a crash"),
    ("H", "show this card again"),
    ("K", "landmark smoothing on / off"),
    ("L", "ToF depth stabilizer on / off (hold still 3 s)"),
    ("X", "cancel a calibration"),
    ("Q / ESC", "quit"),
)

# Fingertip control band. Only the middle stretch of the camera frame is mapped
# to the full screen height: the very top and bottom of frame are awkward to
# reach and are where MediaPipe tracking degrades, so excluding them means the
# whole playable range sits inside comfortable, reliable finger travel.
FINGER_TOP    = 0.15 * H
FINGER_BOTTOM = 0.85 * H

# How hard the bird chases the fingertip, per frame. Higher = more responsive
# and twitchier; lower = floatier.
BIRD_FOLLOW = 0.35

def draw_text(img, text, x, y, size=1.0, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = 60
        self.gap_y = random.randint(150, H - 150)
        self.gap_size = 200
        self.passed = False

    def update(self, speed):
        self.x -= speed

    def draw(self, frame):
        # Top pipe
        cv2.rectangle(frame, (int(self.x), 0), (int(self.x + self.width), int(self.gap_y - self.gap_size//2)), (0, 200, 0), -1)
        cv2.rectangle(frame, (int(self.x), 0), (int(self.x + self.width), int(self.gap_y - self.gap_size//2)), (0, 100, 0), 2)
        # Bottom pipe
        cv2.rectangle(frame, (int(self.x), int(self.gap_y + self.gap_size//2)), (int(self.x + self.width), H), (0, 200, 0), -1)
        cv2.rectangle(frame, (int(self.x), int(self.gap_y + self.gap_size//2)), (int(self.x + self.width), H), (0, 100, 0), 2)

    def collides(self, bx, by, br):
        if bx + br > self.x and bx - br < self.x + self.width:
            if by - br < self.gap_y - self.gap_size//2 or by + br > self.gap_y + self.gap_size//2:
                return True
        return False

def draw_tracking_status(canvas, hand_visible, w=800):
    x0 = w - 120
    y0 = 20
    status_col = (0, 255, 0) if hand_visible else (0, 0, 255)
    status_text = "TRACKING" if hand_visible else "NO HAND"
    cv2.circle(canvas, (x0, y0), 8, status_col, -1)
    cv2.putText(canvas, status_text, (x0 + 15, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

def main():
    print("Starting ToF Flappy Bird...")
    ai_queue = queue.Queue(maxsize=1)
    
    # Initialize pipeline with camera disabled
    pipeline = VisionPipeline(
        result_queue=ai_queue,
        width=W,
        height=H,
    )
    # Force ToF simulated true for testing, but keep camera on so we can simulate ToF depth!
    pipeline.tof_simulated = True
    pipeline.disable_camera = False
    pipeline.start()

    cv2.namedWindow("ToF Flappy", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ToF Flappy", W, H)

    bird_x = 200
    bird_y = H // 2
    bird_r = 15
    
    pipes = [Pipe(W + 200), Pipe(W + 600)]
    score = 0
    game_over = False
    
    target_y = float(H // 2)
    cam_frame = None
    hand_visible = False
    tof_stab_on = False

    # The card is up before the first pipe moves. The pipeline keeps running
    # behind it, so the camera has warmed up and the hand is already tracked by
    # the time the player starts.
    showing_help = True

    while True:
        try:
            latest = ai_queue.get_nowait()
        except queue.Empty:
            latest = None

        if latest is not None:
            cam_frame = latest.get("frame")
            hand_visible = latest.get("hand_visible", False)
            if hand_visible:
                # Height follows the index fingertip directly.
                #
                # This used to map ToF depth to height, but with no real sensor
                # the depth came from the simulated branch (0.45 + lm_z * 0.6).
                # MediaPipe's relative Z spans roughly +/-0.1, so it drove only
                # a sliver of the [0.2, 0.5] range it was mapped through --
                # cramped at one end and saturating at the other. Fingertip Y is
                # absolute, uses the full screen, and is far easier to hold
                # steady with one finger.
                finger_y = float(latest.get("index_pos", (0, H // 2))[1])
                target_y = np.interp(finger_y, [FINGER_TOP, FINGER_BOTTOM], [0, H])
            else:
                # Hand lost: let the bird sink toward the middle rather than
                # snapping, so a brief tracking dropout is survivable.
                target_y = target_y * 0.9 + (H / 2.0) * 0.1

        # Smooth bird movement towards target
        if not showing_help:
            bird_y = bird_y * (1.0 - BIRD_FOLLOW) + target_y * BIRD_FOLLOW

        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (40, 30, 20) # Dark background

        if not game_over and not showing_help:
            # Update pipes
            for p in pipes:
                p.update(5.0)
                if not p.passed and p.x + p.width < bird_x:
                    p.passed = True
                    score += 1
            
            # Remove off-screen pipes and add new ones
            if pipes[0].x < -pipes[0].width:
                pipes.pop(0)
                pipes.append(Pipe(pipes[-1].x + 400))

            # Collision
            for p in pipes:
                if p.collides(bird_x, bird_y, bird_r):
                    game_over = True
            if bird_y > H or bird_y < 0:
                game_over = True

        # Draw pipes
        for p in pipes:
            p.draw(frame)

        # Draw bird
        bird_color = (0, 200, 255) if not game_over else (0, 0, 255)
        cv2.circle(frame, (int(bird_x), int(bird_y)), bird_r, bird_color, -1)
        cv2.circle(frame, (int(bird_x), int(bird_y)), bird_r, (0, 100, 255), 2)

        # Draw UI
        draw_text(frame, f"Score: {score}", 20, 40, 1.0)
        smoothing_on = latest.get("smoothing_enabled", True) if latest else True
        draw_text(frame, f"Smoothing: {'ON' if smoothing_on else 'OFF'}", 20, 80, 0.7,
                  (0, 255, 0) if smoothing_on else (0, 180, 255))
        draw_text(frame, "Raise/lower your index finger to fly  |  H: how to play  |  K: smoothing  L: ToF stab",
                  20, H - 20, 0.5, (150, 150, 150))

        if game_over:
            draw_text(frame, "GAME OVER", W//2 - 150, H//2, 2.0, (0, 0, 255), 3)
            draw_text(frame, "Press 'R' to restart", W//2 - 120, H//2 + 50, 0.8)

        draw_tracking_status(frame, hand_visible, W)

        if showing_help:
            draw_card(frame, TITLE, GOAL, CONTROLS, KEYS)

        cv2.imshow("ToF Flappy", frame)

        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord('q')):
            break
        elif showing_help:
            # Any key starts. Nothing else is dispatched this frame — the key
            # that dismisses the card should not also toggle a filter.
            if key != 255:
                showing_help = False
        elif key in (ord('h'), ord('H')):
            showing_help = True
        elif key in (ord('k'), ord('K')):
            on = pipeline.toggle_smoothing()
            print(f"[Stabilisation] Landmark smoothing: {'ON' if on else 'OFF (raw positions)'}")
        elif key in (ord('l'), ord('L')):
            tof_stab_on = not tof_stab_on
            if tof_stab_on:
                pipeline.begin_stabilization(3.0)
                print("[Stabilisation] ToF stabilizer: calibrating — hold still")
            else:
                pipeline.disable_stabilization()
                print("[Stabilisation] ToF stabilizer: OFF")
        elif key in (ord('x'), ord('X')):
            pipeline.cancel_stabilization()
            tof_stab_on = False
        elif key == ord('r') and game_over:
            score = 0
            pipes = [Pipe(W + 200), Pipe(W + 600)]
            bird_y = H // 2
            target_y = float(H // 2)
            game_over = False

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
