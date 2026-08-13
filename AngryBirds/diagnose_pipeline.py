r"""
diagnose_pipeline.py — Run this with the game's venv to check VisionPipeline hand tracking.

Usage:  .\.venv\Scripts\python.exe diagnose_pipeline.py
"""
import os, sys, queue, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'visual ai game engine', 'src'))

print("=== VisionPipeline Diagnostics ===")

# 1. Check module-level MediaPipe state
from visual_ai.pipeline import HAS_MEDIAPIPE, mp_hands_module, mp_face_detection_module
print(f"HAS_MEDIAPIPE       : {HAS_MEDIAPIPE}")
print(f"mp_hands_module     : {mp_hands_module}")
print(f"mp_face_module      : {mp_face_detection_module}")

# 2. Instantiate pipeline and check detectors
from visual_ai.pipeline import VisionPipeline
q = queue.Queue(maxsize=2)
p = VisionPipeline(result_queue=q, width=640, height=480, camera_index=0, noise_duration=0.0)
print(f"\n_mp_face  : {p._mp_face}")
print(f"_mp_hands : {p._mp_hands}")

# 3. Start and collect a few frames
print("\nStarting pipeline for 5 seconds — hold your hand in front of the camera...")
p.start()
time.sleep(5)
p.stop()

frames_received = 0
hand_visible_count = 0

while not q.empty():
    try:
        payload = q.get_nowait()
        frames_received += 1
        if payload.get("hand_visible"):
            hand_visible_count += 1
            print(f"  ✓ Hand detected: index_pos={payload['index_pos']}  pinching={payload['is_pinching']}")
    except queue.Empty:
        break

print(f"\nFrames received : {frames_received}")
print(f"Hand visible    : {hand_visible_count}")
if frames_received > 0 and hand_visible_count == 0:
    print("\n[WARN] hand_visible was NEVER True — MediaPipe Hands is not detecting.")
elif hand_visible_count > 0:
    print("\n[OK] Hand tracking is working!")
else:
    print("\n[WARN] No frames received at all — camera or pipeline issue.")
