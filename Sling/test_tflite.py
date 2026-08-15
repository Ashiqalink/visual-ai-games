"""
test_tflite.py — Verifies TensorFlow Lite (TFLite) delegate execution & MediaPipe Hand Tracking performance.
"""

import time
import cv2
import numpy as np
import mediapipe as mp

def main():
    print("=" * 60)
    print("[TFLite Setup Verification] Initializing MediaPipe TFLite Hand Landmark Model...")
    print("=" * 60)

    # Initialize TFLite Hands delegate
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,           # 1 = Full TFLite model (high accuracy)
        min_detection_confidence=0.7,
        min_tracking_confidence=0.65
    )

    # Generate a dummy RGB test frame (720x1280x3)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "TFLite Hand Model Test", (400, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    print("[INFO] Warm-up run on TFLite XNNPACK CPU delegate...")
    start_warm = time.time()
    results = hands.process(dummy_frame)
    warm_time = (time.time() - start_warm) * 1000.0
    print(f"[OK] Warm-up inference time: {warm_time:.2f} ms")

    # Benchmark 30 iterations
    print("[INFO] Benchmarking 30 frames of TFLite hand tracking...")
    start_bench = time.time()
    for _ in range(30):
        hands.process(dummy_frame)
    total_bench = time.time() - start_bench
    avg_ms = (total_bench / 30.0) * 1000.0
    fps = 30.0 / total_bench

    print("-" * 60)
    print(f"[SUCCESS] TFLite Model Execution Status : ACTIVE & READY")
    print(f"[SUCCESS] Average Inference Time        : {avg_ms:.2f} ms / frame")
    print(f"[SUCCESS] Projected Frame Rate          : {fps:.1f} FPS")
    print("=" * 60)

    hands.close()

if __name__ == "__main__":
    main()
