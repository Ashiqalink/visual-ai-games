"""
main.py — Entry point.  Runs the webcam game loop.

Powered by visual_ai (VisionPipeline + GameEngine).

Layout
------
  Full window  : game canvas on solid dark background (no camera bleed)
  Top-right box: small camera preview (CAM_W x CAM_H px)

Controls
--------
Fist / open hand, in three phases:
  SELECTION : hover the carousel strip at the top, close your fist to grab a bird.
  READY     : keep the fist and bring your hand wherever you want to shoot from —
              nothing pulls yet. Hold still and the aim origin locks there.
  ARMED     : move the fist to pull the band, open your hand to fire.
              A pull too short to fire puts the bird back on the shelf.

Keyboard
--------
H        : how-to-play card (shown on open; any key dismisses it)
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, '..'))

from engine_bootstrap import ensure_engine
ensure_engine()

# config.py depends on nothing but os/sys, so the window geometry is available
# before the expensive imports below — see _splash().
from config import (
    FRAME_W, FRAME_H, BG_COLOR,
    CAM_W, CAM_H, CAM_MARGIN, CAM_BORDER, CAM_BORDER_COLOR,
    SMOOTH_ALPHA,
    # CURSOR_PINCH_COL / CURSOR_FIRE_COL are no longer drawn on the canvas —
    # ui.py uses them for the pinch and fire rows of the HUD panel instead.
    CURSOR_INDEX_COL,
)


def _splash(message: str):
    """
    Put the window on screen before the slow imports run.

    `import visual_ai` pulls in MediaPipe, which costs over a second on its own,
    and nothing was drawn until after it finished — so launching Sling looked
    like nothing had happened at all. This paints a title card first. It is a
    one-shot paint, not a loop: highgui gets no events while an import blocks,
    which is fine because there is nothing here to interact with.
    """
    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    canvas[:] = BG_COLOR
    cv2.putText(canvas, "SLING", (FRAME_W // 2 - 110, FRAME_H // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (60, 200, 255), 4, cv2.LINE_AA)
    cv2.putText(canvas, message, (FRAME_W // 2 - 110, FRAME_H // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 190, 225), 1, cv2.LINE_AA)
    cv2.namedWindow("Sling", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sling", FRAME_W, FRAME_H)
    cv2.imshow("Sling", canvas)
    cv2.waitKey(1)


_splash("Loading vision engine...")

import psutil
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from visual_ai import VisionPipeline, CPP_ENGINE_AVAILABLE

_splash("Preparing game...")

from game import Game
import ui

# (Window / canvas constants imported from config.py above, before _splash.)


_last_metrics_time = 0.0
_cached_cpu = 0.0
_cached_gpu_str = "N/A"

def _draw_system_metrics(canvas: np.ndarray):
    """Draw live CPU and GPU usage metrics on HUD (smoothed & throttled updates)."""
    global _last_metrics_time, _cached_cpu, _cached_gpu_str
    now = time.time()
    
    # Update reading every 500ms so numbers don't flicker uncontrollably
    if now - _last_metrics_time > 0.5:
        _last_metrics_time = now
        raw_cpu = psutil.cpu_percent(interval=None)
        # Exponential Moving Average smoothing
        _cached_cpu = 0.7 * _cached_cpu + 0.3 * raw_cpu if _cached_cpu > 0 else raw_cpu
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_val = gpus[0].load * 100
                    _cached_gpu_str = f"{gpu_val:.1f}%"
            except Exception:
                _cached_gpu_str = "N/A"
            
    text = f"CPU: {_cached_cpu:.1f}% | GPU: {_cached_gpu_str}"
    cv2.rectangle(canvas, (10, 10), (310, 42), (0, 0, 0), -1)
    cv2.rectangle(canvas, (10, 10), (310, 42), (50, 200, 255), 1)
    cv2.putText(canvas, text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

_cached_preview: np.ndarray | None = None


def _paste_cam(canvas: np.ndarray, cam_frame: np.ndarray, new_frame: bool = True):
    """
    Paste a resized, mirrored camera preview into the top-right corner.

    ``new_frame`` says whether ``cam_frame`` is one the loop has not pasted yet.
    The render loop runs faster than the camera delivers, so re-scaling the same
    frame on every pass was repeating identical work; the scaled copy is kept and
    reused until a new payload arrives. The caller owns that signal rather than
    this function comparing frames, so a pipeline that ever reuses its capture
    buffer cannot leave the preview silently frozen.
    """
    global _cached_preview
    if cam_frame is None:
        return
    if new_frame or _cached_preview is None:
        _cached_preview = cv2.resize(cam_frame, (CAM_W, CAM_H))
    preview = _cached_preview
    # Frame from VisionPipeline is already mirrored, no second flip needed

    x0 = FRAME_W - CAM_W - CAM_MARGIN
    y0 = CAM_MARGIN
    x1 = x0 + CAM_W
    y1 = y0 + CAM_H

    # Cyan border
    cv2.rectangle(canvas,
                  (x0 - CAM_BORDER, y0 - CAM_BORDER),
                  (x1 + CAM_BORDER, y1 + CAM_BORDER),
                  CAM_BORDER_COLOR, CAM_BORDER)

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
        smooth_alpha=SMOOTH_ALPHA,
        # Sling is driven entirely by hand signs and never reads target_x,
        # face_visible or face_box, so the face graph was 3.1 ms/frame of
        # inference feeding keys nothing here looks at.
        detect_face=False,
    )
    pipeline.start()

    game    = Game(frame_w=FRAME_W, frame_h=FRAME_H)

    cv2.namedWindow("Sling", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sling", FRAME_W, FRAME_H)

    # ── FPS & Timing setup ────────────────────────────────────────────────
    prev_time = time.time()
    fps = 0
    accumulator = 0.0
    FIXED_DT = 1 / 60.0

    # ── Render pacing ─────────────────────────────────────────────────────
    # The loop used to have no governor at all: `cv2.waitKey(1)` let it spin at
    # a few hundred iterations a second while the pipeline delivers 30 payloads
    # a second, so the same unchanged state was cleared, drawn and blitted
    # roughly eight times per camera frame. That is where the laptop's heat was
    # going. Physics runs on its own fixed accumulator below and is unaffected
    # by how often we draw, so capping the draw rate costs nothing visually.
    RENDER_DT = 1 / 60.0
    next_render = time.time()

    def _paced_key() -> int:
        """
        Pump highgui, return the pressed key, and wait out the rest of the
        frame budget. `waitKey` returns as soon as a key arrives, so the longer
        timeout paces idle frames without adding input latency.
        """
        nonlocal next_render
        wait_ms = int((next_render - time.time()) * 1000.0)
        key = cv2.waitKey(max(1, wait_ms)) & 0xFF
        # Re-base rather than accumulate: a frame that overran its budget must
        # not bank credit and let the next few frames run back-to-back.
        next_render = max(time.time(), next_render + RENDER_DT)
        return key

    # ToF stabilizer starts off; L toggles it, X cancels an in-progress run.
    tof_stab_on = False

    # Sling opens on its how-to-play card. The pipeline runs behind it, so the
    # camera has warmed up and the hand is tracked by the time play starts.
    showing_help = True

    cam_frame = None          # raw BGR from pipeline (for the preview box only)
    gesture = {
        "hand_visible":        False,
        "index_pos":           (0, 0),
        "thumb_pos":           (0, 0),
        "middle_pos":          (0, 0),
        "pinch_pos":           (0, 0),
        "is_pinching":         False,
        "click_just_fired":    False,
        "is_index_isolated":   False,
        "z_delta":             0.0,
        "xy_drift":            0.0,
        "hand_sign":           "unknown",
        "fingers_extended":    (False, False, False, False, False),
        "is_fist":             False,
        "is_open_palm":        False,
        "smoothing_enabled":   True,
    }

    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    while True:
        # ── Pull latest vision data — drain queue so we never stall ───────
        latest = None
        while True:
            try:
                latest = ai_queue.get_nowait()
            except queue.Empty:
                break

        cam_is_new = latest is not None
        if latest is not None:
            cam_frame = latest["frame"]        # raw camera frame for preview box
            gesture   = latest                 # full payload IS the gesture dict
            game._z_delta_display  = latest.get("z_delta", 0.0)
            game._xy_drift_display = latest.get("xy_drift", 0.0)

        # The camera takes ~0.75 s to hand over its first frame. Rendering
        # nothing until then left the window blank for the whole of startup,
        # which reads as a hang; the how-to-play card goes up immediately
        # instead, so the wait is spent reading the controls. Keys are ignored
        # until the camera is live — dismissing the card early would drop the
        # player into a game that cannot see their hand yet.
        if cam_frame is None:
            canvas[:] = BG_COLOR
            ui.draw_instructions_card(canvas)
            cv2.putText(canvas, "Starting camera...", (FRAME_W // 2 - 90, FRAME_H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 180, 220), 1, cv2.LINE_AA)
            cv2.imshow("Sling", canvas)
            if _paced_key() in (ord('q'), ord('Q'), 27):
                pipeline.stop()
                cv2.destroyAllWindows()
                return
            prev_time = time.time()   # do not charge the wait to the first frame
            continue

        # ── FPS calculation & Timing ──────────────────────────────────────
        now = time.time()
        dt  = max(now - prev_time, 1e-6)
        # Cap frame time to prevent spiral of death if running slow
        frame_time = min(dt, 0.25)
        prev_time = now
        fps = int(1.0 / dt)
        game.fps = fps
        accumulator += frame_time

        # ── Keyboard (also paces the loop) ────────────────────────────────
        key = _paced_key()

        if showing_help:
            # The card swallows the frame. No key is dispatched — the keypress
            # that dismisses it must not also switch level — and the gesture
            # never reaches the state machine, so a fist held while reading
            # cannot grab a bird before the player has read what a fist does.
            if key != 255 and key not in (ord('q'), ord('Q'), 27):
                showing_help = False
            accumulator = 0.0        # do not bank physics steps for the wait
        else:
            if key in (ord('h'), ord('H')):
                showing_help = True
            if key in (ord('t'), ord('T')):
                pipeline.tof_simulated = not getattr(pipeline, "tof_simulated", False)
                print(f"[ToF Debugger] Simulated ToF Depth Stream: {pipeline.tof_simulated}")
            if key in (ord('c'), ord('C')):
                pipeline.disable_camera = not getattr(pipeline, "disable_camera", False)
                print(f"[Camera Debugger] Camera Disabled: {pipeline.disable_camera}")

            # Two independent stabilisation systems, two keys.
            # K — One-Euro landmark smoothing (X/Y jitter vs responsiveness)
            # L — ToF depth stabilizer (lid-shake calibration)
            if key in (ord('k'), ord('K')):
                on = pipeline.toggle_smoothing()
                print(f"[Stabilisation] Landmark smoothing: {'ON' if on else 'OFF (raw positions)'}")
            if key in (ord('l'), ord('L')):
                tof_stab_on = not tof_stab_on
                if tof_stab_on:
                    pipeline.begin_stabilization(3.0)
                    print("[Stabilisation] ToF stabilizer: calibrating — hold still")
                else:
                    pipeline.disable_stabilization()
                    print("[Stabilisation] ToF stabilizer: OFF")
            # The calibration overlay tells the player "Press X to cancel", but
            # nothing was listening for it, so the overlay could not be dismissed.
            if key in (ord('x'), ord('X')):
                pipeline.cancel_stabilization()
                tof_stab_on = False

            # ── Game logic (Input & State) ────────────────────────────────
            game.update_game_state(gesture, key)
            if hasattr(pipeline, "set_movement_magnification"):
                pipeline.set_movement_magnification(game._AIM_PULL_GAIN)

            # ── Fixed-Step Physics ────────────────────────────────────────
            while accumulator >= FIXED_DT:
                game.update_physics()
                accumulator -= FIXED_DT

        # ── Build solid game canvas (no camera bleed-through) ─────────────
        canvas[:] = BG_COLOR            # deep navy-black background

        # ── Draw game elements onto canvas ────────────────────────────────
        game.draw(canvas)

        # ── Draw live performance HUD (CPU/GPU) ─────────────────────────
        _draw_system_metrics(canvas)

        # ── Hand cursor overlay ───────────────────────────────────────────
        if gesture["hand_visible"]:
            ix, iy = gesture["index_pos"]
            # Pipeline already mirrors X (frame was flipped before MediaPipe)
            # so ix/iy are correct display coordinates — no extra flip needed.
            dix = ix
            diy = iy

            # Index fingertip ring — the whole cursor. The pinch-midpoint disc
            # and the "FIRE!" flash that used to live here sat on top of the
            # bird and the slingshot band at the exact moment the player is
            # aiming. Pinch and fire state now read off the HAND SIGN CONTROL
            # panel in the top-right corner, which is out of the play area.
            cv2.circle(canvas, (dix, diy), 14, CURSOR_INDEX_COL, 2)
            cv2.circle(canvas, (dix, diy),  5, CURSOR_INDEX_COL, -1)

        # ── Camera preview — top-right corner ─────────────────────────────
        _paste_cam(canvas, cam_frame, new_frame=cam_is_new)

        # ── How to play — over everything, including the camera preview ───
        if showing_help:
            ui.draw_instructions_card(canvas)

        cv2.imshow("Sling", canvas)

        if key in (ord('q'), ord('Q'), 27):
            break

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
