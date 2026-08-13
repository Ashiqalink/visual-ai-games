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
sys.path.insert(0, os.path.join(BASE_DIR,'..', '..', 'visual ai game engine', 'src'))

from visual_ai import VisionPipeline

W, H = 800, 600

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
    
    # For smoothing Z
    smooth_z = 0.45
    cam_frame = None
    hand_visible = False

    while True:
        try:
            latest = ai_queue.get_nowait()
        except queue.Empty:
            latest = None

        if latest is not None:
            cam_frame = latest.get("frame")
            hand_visible = latest.get("hand_visible", False)
            # Only update depth if the hand is actually visible!
            if hand_visible:
                tof_z = latest.get("tof_z_m", 0.45)
                # Simple EMA smoothing for the raw depth
                smooth_z = 0.6 * smooth_z + 0.4 * tof_z
            else:
                # If hand is lost, let the bird drop towards the bottom faster
                smooth_z = 0.8 * smooth_z + 0.2 * 0.5

        # Map ToF Z depth (approx 0.2m to 0.5m) to screen height
        # Close to sensor (0.2m) -> Bird high (Y=0)
        # Far from sensor (0.5m) -> Bird low (Y=H)
        target_y = np.interp(smooth_z, [0.2, 0.5], [0, H])
        
        # Smooth bird movement towards target
        bird_y = bird_y * 0.7 + target_y * 0.3

        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (40, 30, 20) # Dark background

        if not game_over:
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
        draw_text(frame, f"ToF Z: {smooth_z:.3f}m", 20, 80, 0.7, (0, 255, 0))
        draw_text(frame, "Control bird height with depth!", 20, H - 20, 0.6, (150, 150, 150))

        if game_over:
            draw_text(frame, "GAME OVER", W//2 - 150, H//2, 2.0, (0, 0, 255), 3)
            draw_text(frame, "Press 'R' to restart", W//2 - 120, H//2 + 50, 0.8)

        draw_tracking_status(frame, hand_visible, W)

        cv2.imshow("ToF Flappy", frame)

        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('r') and game_over:
            score = 0
            pipes = [Pipe(W + 200), Pipe(W + 600)]
            game_over = False

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
