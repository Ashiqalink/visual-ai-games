"""
main.py — Entry point.  Runs the webcam game loop.

Powered by visual_ai (VisionPipeline + GameEngine).

Layout
------
  Full window  : game canvas on solid dark background (no camera bleed)
  Top-right box: small camera preview (CAM_W x CAM_H px)

Controls
--------
Gesture – pinch (thumb+index) OR Z-push (move index ~1 inch toward camera)
  Both gestures fire a "click" event.
  During ARMED: move index finger to aim, Z-push / edge-exit to fire.

Keyboard
--------
Q / ESC  : quit
R        : restart
1 / 2 / 3: select level (Easy / Medium / Hard)
"""

import os
import queue
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'visual ai game engine'))

from visual_ai import VisionPipeline, GameEngine, CPP_ENGINE_AVAILABLE
from game import Game

# ── Window / canvas resolution ────────────────────────────────────────────────
FRAME_W = 1280
FRAME_H = 720

# ── Camera preview box (top-right corner) ─────────────────────────────────────
CAM_W      = 240
CAM_H      = 160
CAM_MARGIN = 12      # gap from window edges
CAM_BORDER = 2       # border thickness in pixels


def _paste_cam(canvas: np.ndarray, cam_frame: np.ndarray):
    """Paste a resized, mirrored camera preview into the top-right corner."""
    if cam_frame is None:
        return
    preview = cv2.resize(cam_frame, (CAM_W, CAM_H))
    preview = cv2.flip(preview, 1)          # mirror so it feels natural

    x0 = FRAME_W - CAM_W - CAM_MARGIN
    y0 = CAM_MARGIN
    x1 = x0 + CAM_W
    y1 = y0 + CAM_H

    # Cyan border
    cv2.rectangle(canvas,
                  (x0 - CAM_BORDER, y0 - CAM_BORDER),
                  (x1 + CAM_BORDER, y1 + CAM_BORDER),
                  (80, 200, 255), CAM_BORDER)

    canvas[y0:y1, x0:x1] = preview

    # Small label underneath
    cv2.putText(canvas, "CAMERA", (x0 + 4, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 180, 220), 1, cv2.LINE_AA)


def main():
    print(f"[INFO] visual_ai engine: {'C++ core' if CPP_ENGINE_AVAILABLE else 'Python fallback'}")

    # ── visual_ai setup ───────────────────────────────────────────────────
    # maxsize=1 ensures we always discard stale frames and work on the latest.
    ai_queue = queue.Queue(maxsize=1)
    pipeline = VisionPipeline(
        result_queue=ai_queue,
        width=FRAME_W,
        height=FRAME_H,
        camera_index=0,
        # Adaptive EMA smoothing baseline (0.20 provides smooth aiming without lag).
        smooth_alpha=0.20,
    )
    pipeline.start()

    _engine = GameEngine(float(FRAME_W), float(FRAME_H))
    game    = Game(frame_w=FRAME_W, frame_h=FRAME_H)

    cv2.namedWindow("Angry Birds — Vision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Angry Birds — Vision", FRAME_W, FRAME_H)

    # ── FPS tracking ──────────────────────────────────────────────────────
    prev_time = time.time()
    fps = 0

    cam_frame = None          # raw BGR from pipeline (for the preview box only)
    gesture = {
        "hand_visible":      False,
        "index_pos":         (0, 0),
        "pinch_pos":         (0, 0),
        "is_pinching":       False,
        "click_just_fired":  False,
        "is_index_isolated": False,
        "z_delta":           0.0,
        "xy_drift":          0.0,
    }

    while True:
        # ── Pull latest vision data — drain queue so we never stall ───────
        latest = None
        while True:
            try:
                latest = ai_queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            cam_frame = latest["frame"]        # raw camera frame for preview box
            gesture   = latest                 # full payload IS the gesture dict
            game._z_delta_display  = latest.get("z_delta", 0.0)
            game._xy_drift_display = latest.get("xy_drift", 0.0)

        # Wait for first camera frame before rendering
        if cam_frame is None:
            time.sleep(0.005)
            continue

        # ── FPS calculation ───────────────────────────────────────────────
        now = time.time()
        dt  = max(now - prev_time, 1e-6)
        prev_time = now
        fps = int(1.0 / dt)
        game.fps = fps

        # ── Keyboard ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('t'), ord('T')):
            pipeline.tof_simulated = not getattr(pipeline, "tof_simulated", False)
            print(f"[ToF Debugger] Simulated ToF Depth Stream: {pipeline.tof_simulated}")

        # ── Game logic ────────────────────────────────────────────────────
        game.update(gesture, key)

        # ── Build solid game canvas (no camera bleed-through) ─────────────
        canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 28)            # deep navy-black background

        # ── Draw game elements onto canvas ────────────────────────────────
        game.draw(canvas)

        # ── Hand cursor overlay ───────────────────────────────────────────
        if gesture["hand_visible"]:
            ix, iy = gesture["index_pos"]
            # Pipeline already mirrors X (frame was flipped before MediaPipe)
            # so ix/iy are correct display coordinates — no extra flip needed.
            dix = ix
            diy = iy

            # Index fingertip ring
            cv2.circle(canvas, (dix, diy), 14, (0, 255, 200), 2)
            cv2.circle(canvas, (dix, diy),  5, (0, 255, 200), -1)

            # Pinch midpoint
            if gesture["is_pinching"]:
                px, py = gesture["pinch_pos"]
                dpx = px
                cv2.circle(canvas, (dpx, py), 10, (0, 200, 255), -1)
                cv2.circle(canvas, (dpx, py), 14, (0, 200, 255),  2)

            # Z-click flash
            if gesture["click_just_fired"]:
                cv2.circle(canvas, (dix, diy), 30, (0, 80, 255), 3)
                cv2.putText(canvas, "FIRE!", (dix + 18, diy - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2, cv2.LINE_AA)

        # ── Camera preview — top-right corner ─────────────────────────────
        _paste_cam(canvas, cam_frame)

        cv2.imshow("Angry Birds — Vision", canvas)

        if key in (ord('q'), ord('Q'), 27):
            break

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
