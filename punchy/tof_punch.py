import os
import sys
import time
import queue
import random
import cv2
import numpy as np
import math

# Add visual_ai engine path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, '..','..', 'visual ai game engine', 'src'))

from visual_ai import VisionPipeline

W, H = 800, 600

def draw_text(img, text, x, y, size=1.0, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-15, 15)
        self.vy = random.uniform(-15, 15)
        self.life = 1.0
        # Vibrant colors (yellow/orange/red in BGR)
        self.color = (random.randint(0, 50), random.randint(150, 255), random.randint(200, 255))

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.8  # gravity
        self.life -= 0.05

class Target:
    def __init__(self):
        # Fixed position for 1D ToF Punching
        self.x = W // 2
        self.y = H // 2
        self.r = 80
        self.timer = 100  # Frames to live
        self.max_timer = 100
        self.spawn_time = time.time()

    def draw(self, frame):
        # Pulsing effect
        pulse = math.sin((time.time() - self.spawn_time) * 10) * 8
        current_r = int(self.r + pulse)
        
        alpha = self.timer / self.max_timer
        
        # Draw multiple concentric circles for aesthetics
        for i in range(3, 0, -1):
            r = current_r * i // 3
            # Glowing cyan/blue effect in BGR
            color = (int(255 * alpha * (4-i)/3), int(150 * alpha * (4-i)/3), 0)
            cv2.circle(frame, (self.x, self.y), r, color, -1)
            
        cv2.circle(frame, (self.x, self.y), current_r, (255, 255, 255), 2)
        
        # Timer ring
        cv2.ellipse(frame, (self.x, self.y), (current_r + 20, current_r + 20), 0, 0, 360 * alpha, (0, 0, 255), 4)

def draw_tracking_status(canvas, hand_visible, w=800):
    x0 = w - 150
    y0 = 230
    status_col = (0, 255, 0) if hand_visible else (0, 0, 255)
    status_text = "TRACKING" if hand_visible else "NO HAND"
    # Glowing dot
    cv2.circle(canvas, (x0, y0), 10, status_col, -1)
    cv2.circle(canvas, (x0, y0), 15, status_col, 2)
    draw_text(canvas, status_text, x0 + 25, y0 + 5, 0.7, status_col, 2)

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
    particles = []
    
    # Z velocity tracking
    z_history = []
    Z_PUNCH_THRESHOLD = 0.02  # Lowered from 0.08 (simulated depth changes are smaller)
    lost_frames = 0
    
    punch_cooldown = 0
    bg_flash = 0
    
    cam_frame = None
    hand_visible = False

    # Pre-calculate gradient background
    bg_base = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        c = int(40 - (y / H) * 30)
        bg_base[y, :] = (c + 30, c + 10, c)  # Deep blue/purple gradient

    latest_data = None
    while True:
        try:
            new_data = ai_queue.get_nowait()
            latest = new_data
            latest_data = new_data
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

        frame = bg_base.copy()
        
        # Background flash on punch
        if bg_flash > 0:
            flash_overlay = np.full((H, W, 3), (255, 255, 255), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 1.0, flash_overlay, bg_flash/100.0, 0)
            bg_flash = max(0, bg_flash - 10)
            
        if punch_detected:
            bg_flash = 80
            if target is not None:
                score += 1
                for _ in range(40):
                    particles.append(Particle(target.x, target.y))
                target = Target()  # Spawn new target immediately

        if target is not None:
            target.timer -= 1
            if target.timer <= 0:
                misses += 1
                target = Target()
            target.draw(frame)

        # Draw particles
        active_particles = []
        for p in particles:
            p.update()
            if p.life > 0:
                cv2.circle(frame, (int(p.x), int(p.y)), int(p.life * 8), p.color, -1)
                active_particles.append(p)
        particles = active_particles

        # UI
        draw_text(frame, f"Score: {score}", 30, 50, 1.5, (50, 255, 50), 3)
        draw_text(frame, f"Misses: {misses}", 30, 100, 1.2, (50, 50, 255), 2)
        
        current_z = z_history[-1] if z_history else 0.45
        
        # Draw a stylish depth meter
        meter_x = W - 60
        meter_y = H - 250
        meter_h = 200
        meter_w = 20
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (100, 100, 100), 2)
        
        fill_h = int(min(max((0.5 - current_z) / 0.3, 0), 1) * meter_h)
        if fill_h > 0:
            cv2.rectangle(frame, (meter_x, meter_y + meter_h - fill_h), (meter_x + meter_w, meter_y + meter_h), (0, 200, 255), -1)
            
        draw_text(frame, f"Z: {current_z:.2f}m", W - 120, H - 20, 0.7, (200, 255, 255))
        draw_text(frame, "PUNCH FORWARD (decrease depth)!", 30, H - 30, 0.8, (200, 200, 200))
        
        if punch_cooldown > 0:
            draw_text(frame, "BAM!", W // 2 - 80, H // 2 - 120, 3.0, (0, 150, 255), 5)

        draw_tracking_status(frame, hand_visible, W)

        # ── Stabilizer HUD badge ──────────────────────────────────────────────
        stab_state = latest_data.get("stabilizer_state", "inactive") if latest_data else "inactive"
        stab_noise = latest_data.get("stabilizer_noise_amp", 0.0) if latest_data else 0.0
        
        if stab_state == "sampling":
            # Draw prominent warning in-game since camera frame is hidden
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), (15, 10, 30), -1)
            frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)
            draw_text(frame, "!! LID-SHAKE STABILIZATION !!", W//2 - 250, H//2 - 50, 1.2, (0, 140, 255), 3)
            draw_text(frame, "Please do not move or change position", W//2 - 200, H//2 + 10, 0.7, (255, 255, 255), 2)
            
            prog = latest_data.get("stabilizer_progress", 0.0) if latest_data else 0.0
            draw_text(frame, f"Calibrating... {prog*100:.0f}%", W//2 - 120, H//2 + 50, 0.7, (0, 220, 200), 2)

            badge_color = (0, 200, 255)   # amber-orange
            badge_text  = "SAMPLING..."
        elif stab_state == "active":
            badge_color = (0, 220, 80)    # green
            badge_text  = f"STAB ON  noise={stab_noise*1000:.1f}mm"
        else:
            badge_color = (60, 60, 200)   # muted red
            badge_text  = "STAB OFF  [S]=calibrate"

        bx, by = 30, H - 65
        cv2.circle(frame, (bx, by + 5), 8, badge_color, -1)
        draw_text(frame, badge_text, bx + 20, by + 10, 0.6, badge_color, 2)

        cv2.imshow("ToF Punch", frame)

        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('s'):                          # 3-second calibration
            pipeline.begin_stabilization(duration=3.0)
        elif key == ord('S'):                          # 5-second calibration
            pipeline.begin_stabilization(duration=5.0)
        elif key == ord('x'):                          # disable
            pipeline.disable_stabilization()

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
