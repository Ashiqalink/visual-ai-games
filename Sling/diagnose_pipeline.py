r"""
diagnose_pipeline.py — Run this with the game's venv to check VisionPipeline hand tracking.

Usage:  .\.venv\Scripts\python.exe diagnose_pipeline.py

Note on measurement
-------------------
The previous version created a Queue(maxsize=2) and drained it only *after*
stopping the pipeline. VisionPipeline drops the oldest payload whenever the
queue is full, so that script could never report more than 2 frames no matter
how fast capture actually ran, and it judged hand detection from the last two
frames alone. This version drains continuously while the pipeline runs, so the
frame count is real throughput and the hand statistics cover the whole window.
"""
import os
import queue
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from engine_bootstrap import ensure_engine

ensure_engine()

DURATION = 8.0

print("=== VisionPipeline Diagnostics ===")

# 1. Module-level MediaPipe state
from visual_ai.pipeline import HAS_MEDIAPIPE, mp_face_detection_module, mp_hands_module

print(f"HAS_MEDIAPIPE       : {HAS_MEDIAPIPE}")
print(f"mp_hands_module     : {'loaded' if mp_hands_module else 'MISSING'}")
print(f"mp_face_module      : {'loaded' if mp_face_detection_module else 'MISSING'}")

# 2. Instantiate pipeline and check detectors
from visual_ai.pipeline import VisionPipeline

q = queue.Queue(maxsize=2)
p = VisionPipeline(result_queue=q, width=640, height=480, camera_index=0, noise_duration=0.0)
print(f"_mp_face            : {'ready' if p._mp_face else 'MISSING'}")
print(f"_mp_hands           : {'ready' if p._mp_hands else 'MISSING'}")

# 3. Start and drain continuously
print(f"\nStarting pipeline for {DURATION:.0f} seconds — hold your hand in front of the camera...")
p.start()

frames = 0
hand_frames = 0
sign_counts = {}
first_hand_at = None
started = time.time()

while time.time() - started < DURATION:
    try:
        payload = q.get(timeout=0.5)
    except queue.Empty:
        continue

    frames += 1
    if payload.get("hand_visible"):
        hand_frames += 1
        if first_hand_at is None:
            first_hand_at = time.time() - started
        sign = payload.get("hand_sign")
        if sign:
            sign_counts[sign] = sign_counts.get(sign, 0) + 1

elapsed = time.time() - started
p.stop()

status = p.get_status()
print("\n--- Capture ---")
print(f"Camera available    : {status['camera_available']}")
print(f"Last error          : {status['last_error'] or 'none'}")
print(f"Frames received     : {frames}")
print(f"Throughput          : {frames / elapsed:.1f} fps over {elapsed:.1f}s")

print("\n--- Hand tracking ---")
pct = (100.0 * hand_frames / frames) if frames else 0.0
print(f"Frames with a hand  : {hand_frames} ({pct:.0f}%)")
if first_hand_at is not None:
    print(f"First detection at  : {first_hand_at:.1f}s")
if sign_counts:
    print("Signs seen          : " + ", ".join(
        f"{name}x{count}" for name, count in sorted(sign_counts.items(), key=lambda kv: -kv[1])
    ))

print()
if frames == 0:
    print("[FAIL] No frames at all — the camera never delivered an image.")
elif frames / elapsed < 5.0:
    print(f"[WARN] Only {frames / elapsed:.1f} fps. Capture is running but far too slow to play.")
elif hand_frames == 0:
    print("[WARN] Frames arrived but no hand was ever detected. Check lighting and that")
    print("       your whole hand is inside the frame.")
elif pct < 50.0:
    print(f"[WARN] Hand detected in only {pct:.0f}% of frames — tracking is intermittent.")
else:
    print("[OK] Capture rate and hand tracking both look healthy.")
