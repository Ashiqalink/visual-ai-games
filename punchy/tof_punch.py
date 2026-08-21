"""
punchy - smash targets by driving your fist at the camera.

Depth is the whole control: a punch is a sudden drop in the hand's depth
against its recent baseline, so where the hand sits on screen makes no
difference. Runs against simulated depth when no sensor is present. Timers
are wall-clock durations (converted from the old iteration counts at the
nominal 16 ms frame - see the pacing note below), and the render loop is
paced by gameloop.frame_pacer.
"""
import math
import os
import queue
import random
import sys
import time

import cv2
import numpy as np

# Add visual_ai engine path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, '..'))

from engine_bootstrap import ensure_engine

ensure_engine()

from tracker import RunTracker, tracking_enabled
from visual_ai import VisionPipeline

from gameloop import drain, draw_text, frame_pacer
from instructions import draw_card

W, H = 800, 600

# How many frames of depth history the punch detector measures against, and how
# big a drop has to be to count. Named here rather than inline so the tracker
# can log the threshold a run was played at -- a log that does not say which
# threshold produced it cannot answer whether the threshold was right.
Z_HISTORY = 10
Z_PUNCH_THRESHOLD = 0.02  # Lowered from 0.08 (simulated depth changes are smaller)

# Render pacing, and the durations that used to be frame counts.
#
# The loop asked for cv2.waitKey(16) and got ~31 ms: OpenCV's Windows highgui
# pumps on a ~15.9 ms tick, so a 16 ms request lands just past one tick and
# waits out a second. Measured here, waitKey(16) is 30.89 ms/frame (32.4 fps)
# against waitKey(1)'s 15.06 ms. flappy hit this first and carries the same
# note; punchy never got the fix.
#
# Everything below that used to count loop iterations is a duration now. The
# counts were all written against a nominal 16 ms frame -- the tracker settings
# said so outright -- so they are converted at 16 ms, which is what the game was
# designed for. Against the ~31 ms it actually ran at, that makes targets expire
# about twice as fast as they have been: this is a deliberate difficulty change
# back to the intended pace, not a side effect of the pacing fix.
RENDER_HZ = 60.0
RENDER_DT = 1.0 / RENDER_HZ

TARGET_LIFETIME = 1.6      # s, was 100 frames
PUNCH_COOLDOWN = 0.24      # s, was 15 frames
FLASH_DECAY = 0.128        # s for the punch flash to fade out, was 8 frames

WINDOW = "Punchy"      # the OpenCV window name; "ToF" dropped with the depth rename
TITLE = "ToF Punch — hit it with depth"
GOAL = ("Punch toward the camera to smash the target before the red ring "
        "around it runs out.")
CONTROLS = (
    ("Hold one hand up",
     "the meter on the right is how far your hand is from the camera — that "
     "distance is the whole control"),
    ("Punch forward",
     "a quick move toward the camera lands the hit; where your hand sits on "
     "screen makes no difference, so aim is not part of this"),
    ("Pull back between punches",
     "a hit is a sudden drop in depth against the last few frames, so a hand "
     "parked out in front cannot punch again until it comes back"),
    ("Let the ring empty",
     "the target is missed and a fresh one spawns"),
)
KEYS = (
    ("H", "show this card again"),
    ("S", "calibrate the depth stabilizer, 3 s — hold still"),
    ("Shift + S", "longer 5 s calibration"),
    ("X", "turn the stabilizer off"),
    ("Q / ESC", "quit"),
)

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
        self.remaining = TARGET_LIFETIME   # seconds of life left
        self.spawn_time = time.time()

    def age(self, dt: float) -> bool:
        """Spend `dt` seconds of the target's life. True once it has expired."""
        self.remaining -= dt
        return self.remaining <= 0.0

    def draw(self, frame):
        # Pulsing effect
        pulse = math.sin((time.time() - self.spawn_time) * 10) * 8
        current_r = int(self.r + pulse)

        alpha = max(0.0, self.remaining / TARGET_LIFETIME)

        # Draw multiple concentric circles for aesthetics
        for i in range(3, 0, -1):
            r = current_r * i // 3
            # Glowing cyan/blue effect in BGR
            color = (int(255 * alpha * (4-i)/3), int(150 * alpha * (4-i)/3), 0)
            cv2.circle(frame, (self.x, self.y), r, color, -1)
            
        cv2.circle(frame, (self.x, self.y), current_r, (255, 255, 255), 2)
        
        # Timer ring
        cv2.ellipse(frame, (self.x, self.y), (current_r + 20, current_r + 20),
                    0, 0, 360 * alpha, (0, 0, 255), 4)

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
        # Punchy reads ToF depth and hand position only, never a face key,
        # so the face graph was pure cost — 3.1 ms/frame of inference.
        detect_face=False,
    )
    pipeline.tof_simulated = True
    pipeline.disable_camera = False
    pipeline.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, W, H)

    score = 0
    misses = 0
    target = Target()
    particles = []
    
    # Z velocity tracking
    z_history = []
    lost_frames = 0
    
    punch_cooldown = 0.0        # s of hit lockout left
    bg_flash = 0.0              # 0..1 strength of the punch flash
    
    hand_visible = False

    # Pre-calculate gradient background
    bg_base = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        c = int(40 - (y / H) * 30)
        bg_base[y, :] = (c + 30, c + 10, c)  # Deep blue/purple gradient

    # Full-frame effect layers. Both are constant colors, and both used to be
    # rebuilt inside the loop — the flash allocated a fresh white frame on every
    # frame of every punch, and the stabilizer dim copied the whole frame only
    # to paint over every pixel of the copy.
    flash_overlay = np.full((H, W, 3), (255, 255, 255), dtype=np.uint8)
    stab_overlay = np.empty((H, W, 3), dtype=np.uint8)
    stab_overlay[:] = (15, 10, 30)

    latest_data = None

    # Shown before the first target is live. The pipeline runs behind the card,
    # so depth history has already filled by the time the player throws a punch.
    showing_help = True

    # Offline run tracking. Off unless this machine opted in; the log stays in
    # punchy/data and goes nowhere. `tracker_report.py --enable` switches it on.
    tracking_on = tracking_enabled()
    tracker = RunTracker({
        "threshold": Z_PUNCH_THRESHOLD,
        "z_history": Z_HISTORY,
        "cooldown_s": PUNCH_COOLDOWN,
        # The target's life is wall-clock now, so the budget is the real one
        # rather than a frame count read at a nominal frame time.
        "target_budget_ms": TARGET_LIFETIME * 1000.0,
        "tof_simulated": bool(getattr(pipeline, "tof_simulated", False)),
    }, enabled=tracking_on)
    print(f"[tracker] run logging: {'ON (local file)' if tracking_on else 'OFF'}")
    prev_tick = time.perf_counter()
    paced_key = frame_pacer(RENDER_DT)

    while True:
        now = time.perf_counter()
        frame_ms = (now - prev_tick) * 1000.0
        prev_tick = now
        # A frame longer than this is a stall -- a window drag, a GC pause.
        # Ageing the target by the true elapsed time would expire it mid-hitch.
        dt = min(frame_ms / 1000.0, 0.10)

        latest = drain(ai_queue)
        if latest is not None:
            latest_data = latest

        if latest is not None:
            hand_visible = latest.get("hand_visible", False)
            if hand_visible:
                tof_z = latest.get("tof_z_m", 0.45)
                z_history.append(tof_z)
                if len(z_history) > Z_HISTORY:
                    z_history.pop(0)
                lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames > 5:
                    z_history.clear()

        # The depth the meter shows, and the one the tracker logs: the newest
        # sample while the hand is tracked, and the resting default otherwise.
        current_z = z_history[-1] if z_history else 0.45
        stab_state = (latest_data.get("stabilizer_state", "inactive")
                      if latest_data else "inactive")
        if not showing_help:
            tracker.frame(current_z, hand_visible, frame_ms, stab_state)

        # Detect Punch
        punch_detected = False
        punch_delta = punch_baseline = 0.0
        if len(z_history) >= 3 and punch_cooldown <= 0.0 and not showing_help:
            # Baseline is the furthest Z in recent history (to account for noise)
            z_baseline = max(z_history[:-1])
            current_z = z_history[-1]
            # If current Z is significantly closer than baseline
            delta_z = z_baseline - current_z
            if delta_z >= Z_PUNCH_THRESHOLD:
                punch_detected = True
                punch_delta = delta_z
                punch_baseline = z_baseline
                punch_cooldown = PUNCH_COOLDOWN   # prevent multi-hits
                z_history.clear()
        
        if punch_cooldown > 0.0:
            punch_cooldown = max(0.0, punch_cooldown - dt)

        frame = bg_base.copy()
        
        # Background flash on punch
        if bg_flash > 0.0:
            frame = cv2.addWeighted(frame, 1.0, flash_overlay, bg_flash * 0.8, 0)
            bg_flash = max(0.0, bg_flash - dt / FLASH_DECAY)
            
        if punch_detected:
            bg_flash = 1.0
            if target is not None:
                age_ms = (time.time() - target.spawn_time) * 1000.0
                tracker.punch(punch_delta, punch_baseline, age_ms,
                              Z_PUNCH_THRESHOLD)
                tracker.target_resolved("hit", age_ms)
                score += 1
                for _ in range(40):
                    particles.append(Particle(target.x, target.y))
                target = Target()  # Spawn new target immediately

        if target is not None:
            # The target still draws behind the card — a live screen reads as a
            # game waiting to start, a frozen blank one reads as a hang — but it
            # does not age, so nobody is charged a miss for reading the rules.
            if not showing_help:
                if target.age(dt):
                    tracker.target_resolved(
                        "miss", (time.time() - target.spawn_time) * 1000.0)
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
        cv2.rectangle(frame, (meter_x, meter_y),
                      (meter_x + meter_w, meter_y + meter_h), (100, 100, 100), 2)
        
        fill_h = int(min(max((0.5 - current_z) / 0.3, 0), 1) * meter_h)
        if fill_h > 0:
            cv2.rectangle(frame, (meter_x, meter_y + meter_h - fill_h),
                          (meter_x + meter_w, meter_y + meter_h), (0, 200, 255), -1)
            
        draw_text(frame, f"Z: {current_z:.2f}m", W - 120, H - 20, 0.7, (200, 255, 255))
        draw_text(frame, "PUNCH FORWARD (decrease depth)!  |  H: how to play",
                  30, H - 30, 0.7, (200, 200, 200))
        
        if punch_cooldown > 0.0:
            draw_text(frame, "BAM!", W // 2 - 80, H // 2 - 120, 3.0, (0, 150, 255), 5)

        draw_tracking_status(frame, hand_visible, W)

        # ── Stabilizer HUD badge ──────────────────────────────────────────────
        stab_noise = latest_data.get("stabilizer_noise_amp", 0.0) if latest_data else 0.0
        
        if stab_state == "sampling":
            # Draw prominent warning in-game since camera frame is hidden
            frame = cv2.addWeighted(frame, 0.4, stab_overlay, 0.6, 0)
            draw_text(frame, "!! LID-SHAKE STABILIZATION !!",
                      W//2 - 250, H//2 - 50, 1.2, (0, 140, 255), 3)
            draw_text(frame, "Please do not move or change position",
                      W//2 - 200, H//2 + 10, 0.7, (255, 255, 255), 2)
            
            prog = latest_data.get("stabilizer_progress", 0.0) if latest_data else 0.0
            draw_text(frame, f"Calibrating... {prog*100:.0f}%",
                      W//2 - 120, H//2 + 50, 0.7, (0, 220, 200), 2)

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

        if showing_help:
            draw_card(frame, TITLE, GOAL, CONTROLS, KEYS)

        cv2.imshow(WINDOW, frame)

        key = paced_key()
        if key in (27, ord('q')):
            break
        elif showing_help:
            # Any key starts, and nothing else is dispatched this frame — the
            # key that dismisses the card should not also start a calibration.
            if key != 255:
                showing_help = False
                # The run starts here, not at launch: time spent reading the
                # card is not play, and charging it to the first target would
                # put a minute of reading into the time-to-hit figure.
                if not tracker.frames:
                    tracker = RunTracker(tracker.settings, enabled=tracking_on)
                if target is not None:
                    target.spawn_time = time.time()
        elif key in (ord('h'), ord('H')):
            showing_help = True
        elif key == ord('s'):                          # 3-second calibration
            pipeline.begin_stabilization(duration=3.0)
        elif key == ord('S'):                          # 5-second calibration
            pipeline.begin_stabilization(duration=5.0)
        elif key == ord('x'):                          # disable
            pipeline.disable_stabilization()

    tracker.end("quit", score, misses)

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
