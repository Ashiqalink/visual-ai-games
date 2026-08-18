"""
Avatar Catch — test harness for "capture your avatar once, play with it".

Not a real game, a testbed: on launch the player holds still for a short
countdown, one frame is grabbed from the live pipeline, matted to a
transparent cutout, and that cutout becomes the paddle sprite for the rest of
the session. Everything downstream of capture (movement, collisions, score)
is placeholder catch-the-falling-shapes gameplay — the point being tested is
the capture -> matte -> reuse pipeline, not this game design.

Matting backend is swappable via MATTE_BACKEND below:
    "modnet" — portrait matting via visual_ai.matting (MODNet on onnxruntime).
               Best edge on hair, and the reason this harness exists. Weights
               (25 MB) download themselves on first capture; that plus model
               load makes the very first capture take a few seconds, and later
               ones ~0.3s. Falls back to a plain center-crop (no matte) if
               onnxruntime is missing or the download fails, so the capture
               flow still runs.
    "rembg"  — already a dependency of visual_ai.imaging, general-purpose
               (not portrait-specific), works out of the box.

This only makes sense as a one-shot: it runs once at the CAPTURE state, not
per frame, so a slower matting model (MODNet) is fine here in a way it would
not be inside VisionPipeline's live thread. For per-frame masking the SDK has
VisionPipeline.emit_person_mask instead.
"""

import os
import sys
import time
import queue
import random

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, '..'))

from engine_bootstrap import ensure_engine
ensure_engine()

from visual_ai import VisionPipeline, cut_out_person
from visual_ai.imaging import bgr_to_rgb, pad_to, remove_background, resize as resize_rgba

W, H = 800, 600
TITLE = "Avatar Catch (test harness)"

MATTE_BACKEND = "modnet"  # "modnet" | "rembg" — modnet needs weights, see module docstring

HOLD_STILL_SECONDS = 3.0
AVATAR_SIZE = 140

CATCH_SPEED_MIN = 4
CATCH_SPEED_MAX = 9
SPAWN_EVERY = 45  # frames


def _center_crop_rgba(bgr_frame: np.ndarray) -> np.ndarray:
    """No-matte fallback: an opaque square crop, used when a backend is unavailable."""
    h, w = bgr_frame.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = bgr_frame[y0:y0 + side, x0:x0 + side]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])


def capture_avatar(bgr_frame: np.ndarray) -> np.ndarray:
    """One still frame in, an AVATAR_SIZE x AVATAR_SIZE RGBA cutout out."""
    rgb = bgr_to_rgb(bgr_frame)
    try:
        if MATTE_BACKEND == "modnet":
            rgba = cut_out_person(rgb)
        else:
            rgba = remove_background(rgb, model="isnet-general-use")
    except RuntimeError as exc:
        print(f"[avatar_catch] {MATTE_BACKEND} matting unavailable ({exc}); "
              f"falling back to a plain center crop with no matte.")
        # Already square and opaque, so there is nothing to crop to.
        return resize_rgba(_center_crop_rgba(bgr_frame), AVATAR_SIZE, AVATAR_SIZE)

    # pad_to trims to what the matte actually kept and fits that on a square
    # canvas, so the player is not squashed by the frame's 4:3 aspect. It
    # resizes alpha-aware too — resampling straight alpha would drag
    # background colour into the soft hair edges MODNet is here to get right.
    return pad_to(rgba, AVATAR_SIZE, fit=1.0)


def draw_text(img, text, x, y, size=1.0, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)


def blit_rgba(dst_bgr, rgba, cx, cy):
    """Alpha-composite an RGBA sprite onto a BGR frame, centered at (cx, cy)."""
    h, w = rgba.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1 = x0 + w, y0 + h

    dst_x0, dst_y0 = max(x0, 0), max(y0, 0)
    dst_x1, dst_y1 = min(x1, dst_bgr.shape[1]), min(y1, dst_bgr.shape[0])
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return

    src = rgba[dst_y0 - y0:dst_y1 - y0, dst_x0 - x0:dst_x1 - x0]
    region = dst_bgr[dst_y0:dst_y1, dst_x0:dst_x1]

    alpha = (src[..., 3:4].astype(np.float32) / 255.0)
    src_bgr = cv2.cvtColor(src[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    region[:] = (src_bgr * alpha + region.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


class FallingShape:
    def __init__(self):
        self.x = random.randint(30, W - 30)
        self.y = -20
        self.r = random.randint(12, 20)
        self.speed = random.uniform(CATCH_SPEED_MIN, CATCH_SPEED_MAX)
        self.color = random.choice([(60, 200, 255), (120, 255, 120), (255, 160, 60)])

    def update(self):
        self.y += self.speed

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.r, self.color, -1)


def main():
    ai_queue = queue.Queue(maxsize=1)
    pipeline = VisionPipeline(result_queue=ai_queue, width=W, height=H)
    pipeline.start()

    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITLE, W, H)

    STATE_CAPTURE, STATE_PLAYING, STATE_GAMEOVER = "capture", "playing", "gameover"
    state = STATE_CAPTURE
    hold_start = None
    avatar = None  # RGBA, set once capture completes

    paddle_x = W / 2.0
    hand_visible = False
    shapes = []
    score = 0
    lives = 3
    frame_count = 0

    while True:
        try:
            latest = ai_queue.get_nowait()
        except queue.Empty:
            latest = None

        cam_frame = latest.get("frame") if latest else None
        if latest is not None:
            hand_visible = latest.get("hand_visible", False)
            if hand_visible:
                index_x = float(latest.get("index_pos", (W // 2, 0))[0])
                paddle_x = paddle_x * 0.7 + index_x * 0.3

        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (30, 25, 20)

        if state == STATE_CAPTURE:
            if cam_frame is not None:
                preview = cv2.resize(cam_frame, (W, H))
                canvas[:] = preview

            if hand_visible and hold_start is None:
                hold_start = time.time()
            elif not hand_visible:
                hold_start = None

            draw_text(canvas, "Hold a hand up to frame yourself, then hold still",
                      W // 2 - 320, 40, 0.8)

            if hold_start is not None:
                elapsed = time.time() - hold_start
                remaining = max(0.0, HOLD_STILL_SECONDS - elapsed)
                draw_text(canvas, f"Capturing in {remaining:0.1f}s", W // 2 - 120, H // 2, 1.2, (0, 255, 255))
                if elapsed >= HOLD_STILL_SECONDS and cam_frame is not None:
                    avatar = capture_avatar(cam_frame)
                    state = STATE_PLAYING
            else:
                draw_text(canvas, "Waiting for hand tracking...", W // 2 - 160, H // 2, 0.9, (0, 180, 255))

        elif state == STATE_PLAYING:
            frame_count += 1
            if frame_count % SPAWN_EVERY == 0:
                shapes.append(FallingShape())

            paddle_y = H - 80
            for s in list(shapes):
                s.update()
                s.draw(canvas)
                if s.y > paddle_y - 10:
                    if abs(s.x - paddle_x) < AVATAR_SIZE / 2:
                        score += 1
                    else:
                        lives -= 1
                    shapes.remove(s)
                elif s.y > H + 30:
                    shapes.remove(s)

            blit_rgba(canvas, avatar, paddle_x, paddle_y)

            draw_text(canvas, f"Score: {score}", 20, 40)
            draw_text(canvas, f"Lives: {lives}", 20, 70)
            draw_text(canvas, "Move your hand left/right to catch shapes with your avatar",
                      20, H - 20, 0.5, (150, 150, 150))

            if lives <= 0:
                state = STATE_GAMEOVER

        elif state == STATE_GAMEOVER:
            blit_rgba(canvas, avatar, W // 2, H // 2 - 60)
            draw_text(canvas, "GAME OVER", W // 2 - 130, H // 2 + 80, 1.4, (0, 0, 255), 3)
            draw_text(canvas, f"Final score: {score}", W // 2 - 100, H // 2 + 120, 0.9)
            draw_text(canvas, "Press 'R' to recapture and play again", W // 2 - 190, H // 2 + 150, 0.7)

        cv2.imshow(TITLE, canvas)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('r'):
            state = STATE_CAPTURE
            hold_start = None
            avatar = None
            shapes = []
            score = 0
            lives = 3
            frame_count = 0

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
