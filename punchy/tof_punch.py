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
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'visual ai game engine', 'src'))

from visual_ai import VisionPipeline

W, H = 800, 600

def draw_text(img, text, x, y, size=1.0, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

class Target:
    def __init__(self):
        self.x = random.randint(100, W - 100)
        self.y = random.randint(100, H - 100)
        self.r = 40
        self.timer = 100  # Frames to live
        self.max_timer = 100

    def draw(self, frame):
        # Draw fading target
        alpha = self.timer / self.max_timer
        color = (0, int(255 * alpha), int(255 * alpha))
        cv2.circle(frame, (self.x, self.y), self.r, color, -1)
        cv2.circle(frame, (self.x, self.y), self.r, (255, 255, 255), 2)
        # Timer ring
        cv2.ellipse(frame, (self.x, self.y), (self.r + 10, self.r + 10), 0, 0, 360 * alpha, (0, 0, 255), 3)

def draw_tracking_status(canvas, hand_visible, w=800):
    x0 = w - 120
    y0 = 20
    status_col = (0, 255, 0) if hand_visible else (0, 0, 255)
    status_text = "TRACKING" if hand_visible else "NO HAND"
    cv2.circle(canvas, (x0, y0), 8, status_col, -1)
    cv2.putText(canvas, status_text, (x0 + 15, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

def main():
    print("Starting ToF Z-Punch Game...")
    ai_queue = queue.Queue(maxsize=1)
    
    pipeline = VisionPipeline(
        result_queue=ai_queue,
        width=W,
        height=H,
    )
    pipeline.tof_simulated = True
    pipeline.disable_camera = False
    pipeline.start()

    cv2.namedWindow("ToF Punch", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ToF Punch", W, H)

    score = 0
    misses = 0
    target = Target()
    
    # Z velocity tracking
    z_history = []
    Z_PUNCH_THRESHOLD = 0.02  # Lowered from 0.08 (simulated depth changes are smaller)
    lost_frames = 0
    
    punch_cooldown = 0
    bg_flash = 0
    
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
            if hand_visible:
                tof_z = latest.get("tof_z_m", 0.45)
                z_history.append(tof_z)
                if len(z_history) > 10:  # 10 frames window
                    z_history.pop(0)
                lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames > 5:
                    z_history.clear()

        # Detect Punch
        punch_detected = False
        if len(z_history) >= 3 and punch_cooldown == 0:
            # Baseline is the furthest Z in recent history (to account for noise)
            z_baseline = max(z_history[:-1])
            current_z = z_history[-1]
            # If current Z is significantly closer than baseline
            delta_z = z_baseline - current_z
            if delta_z >= Z_PUNCH_THRESHOLD:
                punch_detected = True
                punch_cooldown = 15  # prevent multi-hits
                z_history.clear()
        
        if punch_cooldown > 0:
            punch_cooldown -= 1

        frame = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Background flash on punch
        if bg_flash > 0:
            frame[:] = (0, bg_flash, 0)
            bg_flash = max(0, bg_flash - 20)
        else:
            frame[:] = (30, 20, 20)
            
        if punch_detected:
            bg_flash = 100
            if target is not None:
                score += 1
                target = Target()  # Spawn new target immediately

        if target is not None:
            target.timer -= 1
            if target.timer <= 0:
                misses += 1
                target = Target()
            target.draw(frame)

        # UI
        draw_text(frame, f"Score: {score}", 20, 40, 1.2, (0, 255, 0))
        draw_text(frame, f"Misses: {misses}", 20, 80, 1.2, (0, 0, 255))
        
        current_z = z_history[-1] if z_history else 0.45
        draw_text(frame, f"ToF Depth: {current_z:.3f}m", 20, H - 30, 0.7)
        draw_text(frame, "PUNCH FORWARD (rapidly decrease depth) to smash!", 20, H - 60, 0.7, (200, 200, 200))
        
        if punch_cooldown > 0:
            draw_text(frame, "PUNCHED!", W // 2 - 100, H // 2, 2.0, (255, 255, 255), 4)

        draw_tracking_status(frame, hand_visible, W)

        cv2.imshow("ToF Punch", frame)

        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord('q')):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
