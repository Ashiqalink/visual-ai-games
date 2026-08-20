"""
Avatar Catch — test harness for "capture your avatar once, play with it".

Not a real game, a testbed: on launch the pipeline's face detector frames the
player, who holds still for a short countdown, then one frame is grabbed from
the live pipeline, matted to a transparent cutout, and that cutout becomes the
paddle sprite for the rest of the session. Gameplay after that is driven by
the hand, but the capture is not — see main() for why the face runs it. Everything downstream of capture (movement, collisions, score)
is placeholder catch-the-falling-shapes gameplay — the point being tested is
the capture -> matte -> reuse pipeline, not this game design.

What happens to the photo: nothing leaves this process. The capture lives in
memory as `capture_rgba` and the sprite built from it, both dropped on R and
gone when the window closes — there is no imwrite anywhere in this file and no
upload. Capture is on SPACE rather than on a timer that fires the moment a face
appears, because this points a camera at whoever is in front of it and a
countdown nobody started is not consent. The one thing that does reach the
network is the MODNet weight download on first use, and only then.

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
from visual_ai.imaging import (bgr_to_rgb, clipped_fraction, normalize_lighting,
                               pad_to, remove_background, resize as resize_rgba)

W, H = 800, 600
TITLE = "Avatar Catch (test harness)"

MATTE_BACKEND = "modnet"  # "modnet" | "rembg" — modnet needs weights, see module docstring

HOLD_STILL_SECONDS = 3.0
STILL_RADIUS = 45  # px of face drift that restarts the countdown
LIGHTING_STRENGTH = 0.85  # 0 disables the exposure fix, 1 applies it in full
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


def matte_frame(bgr_frame: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    One still frame in, a full-resolution (RGBA cutout, matte worked) out.

    Kept separate from build_sprite so the harness can re-render the same shot
    with different lighting settings without paying for the matte again —
    comparing treatments on one capture is the entire point of a testbed, and
    re-matting per toggle would cost 0.3s each time.
    """
    rgb = bgr_to_rgb(bgr_frame)
    try:
        if MATTE_BACKEND == "modnet":
            return cut_out_person(rgb), True
        return remove_background(rgb, model="isnet-general-use"), True
    except RuntimeError as exc:
        print(f"[avatar_catch] {MATTE_BACKEND} matting unavailable ({exc}); "
              f"falling back to a plain center crop with no matte.")
        return _center_crop_rgba(bgr_frame), False


def build_sprite(rgba: np.ndarray, matted: bool,
                 lighting: bool = True) -> tuple[np.ndarray, str]:
    """
    A matted capture in, an (AVATAR_SIZE x AVATAR_SIZE sprite, HUD label) out.

    The label says which backend actually ran. That matters more than it
    looks: when the fallback fires, the sprite is an uncut opaque rectangle,
    and a rectangle is exactly what a working matte of someone filling the
    frame could plausibly look like at a glance. Reporting it only on stdout —
    which nobody reads while a game window has focus — makes a missing
    dependency read as a broken feature.
    """
    if not matted:
        # Already square and opaque, so there is nothing to crop or light.
        return (resize_rgba(rgba, AVATAR_SIZE, AVATAR_SIZE),
                f"{MATTE_BACKEND} unavailable - uncut crop (see console)")

    alpha = rgba[..., 3]
    blown = clipped_fraction(rgba, mask=alpha)
    if lighting:
        # Fix the lighting *after* matting, masked to the cutout. Webcams
        # meter for the whole scene, so a window behind the player buys a face
        # that is half blown out; correcting before the cut would spend the
        # correction on the background, and the background is what we are
        # throwing away.
        rgba = normalize_lighting(rgba, strength=LIGHTING_STRENGTH, mask=alpha)

    # pad_to trims to what the matte actually kept and fits that on a square
    # canvas, so the player is not squashed by the frame's 4:3 aspect. It
    # resizes alpha-aware too — resampling straight alpha would drag
    # background colour into the soft hair edges MODNet is here to get right.
    sprite = pad_to(rgba, AVATAR_SIZE, fit=1.0)
    kept = float(sprite[..., 3].mean()) / 255.0
    note = (f"{MATTE_BACKEND} matte - {kept:.0%} kept, "
            f"lighting {'on' if lighting else 'off'} (L)")
    if blown > 0.02:
        # Clipped pixels recorded nothing, so this is the one part no
        # correction can touch. Say so rather than leaving the player
        # wondering why the bright patch stayed bright.
        note += f", {blown:.0%} blown out - dim the light behind you"
    return sprite, note


def capture_avatar(bgr_frame: np.ndarray,
                   lighting: bool = True) -> tuple[np.ndarray, str]:
    """Matte and render in one call, for callers that will not re-render."""
    rgba, matted = matte_frame(bgr_frame)
    return build_sprite(rgba, matted, lighting)


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
    consented = False  # SPACE arms the countdown; nothing is grabbed before it
    hold_start = None
    hold_anchor = None  # face centre the countdown started at, in canvas px
    avatar = None  # RGBA sprite, set once capture completes
    capture_rgba = None  # the full-res matte behind it, kept for re-rendering
    matted = False
    lighting = True  # L toggles the exposure fix against the same capture
    matte_note = ""  # which backend produced `avatar`, shown on the HUD

    paddle_x = W / 2.0
    cam_frame = None
    hand_visible = False
    face_visible = False
    face_box = (0, 0, 0, 0)
    shapes = []
    score = 0
    lives = 3
    frame_count = 0

    while True:
        # Drain to the freshest payload, per the queue contract. A single
        # get_nowait() leaves the game acting on a stale frame every time the
        # render loop runs slower than the pipeline produces.
        latest = None
        while True:
            try:
                latest = ai_queue.get_nowait()
            except queue.Empty:
                break

        # The queue only refills at camera rate, so most loop iterations drain
        # nothing. Everything below reads the last known frame and face rather
        # than treating an empty queue as "no face", which would otherwise
        # restart the countdown on roughly every other iteration.
        if latest is not None:
            cam_frame = latest.get("frame", cam_frame)
            hand_visible = latest.get("hand_visible", False)
            face_visible = latest.get("face_visible", False)
            face_box = latest.get("face_box", (0, 0, 0, 0))
            if hand_visible:
                index_x = float(latest.get("index_pos", (W // 2, 0))[0])
                paddle_x = paddle_x * 0.7 + index_x * 0.3

        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (30, 25, 20)

        if state == STATE_CAPTURE:
            if cam_frame is not None:
                preview = cv2.resize(cam_frame, (W, H))
                canvas[:] = preview

            # The countdown tracks the face rather than a raised hand. A hand
            # only proved someone was there; a face is what the capture is
            # actually of, so this both frames the shot and gives "hold still"
            # something to measure — drifting more than STILL_RADIUS from
            # where the countdown started restarts it from the new position.
            face_center = None
            if face_visible and face_box[2] > 0 and cam_frame is not None:
                scale_x, scale_y = W / cam_frame.shape[1], H / cam_frame.shape[0]
                face_center = ((face_box[0] + face_box[2] / 2.0) * scale_x,
                               (face_box[1] + face_box[3] / 2.0) * scale_y)
                cv2.rectangle(canvas,
                              (int(face_box[0] * scale_x), int(face_box[1] * scale_y)),
                              (int((face_box[0] + face_box[2]) * scale_x),
                               int((face_box[1] + face_box[3]) * scale_y)),
                              (0, 255, 255), 2)

            # SPACE gates the countdown, and the countdown is the only thing
            # that reaches for a frame. The face box above is drawn either way
            # so the player can frame themselves *before* deciding — showing a
            # preview is not the same as taking a photo, and only one of the
            # two needs asking.
            if not consented or face_center is None:
                hold_start = None
                hold_anchor = None
            elif hold_start is None or np.hypot(face_center[0] - hold_anchor[0],
                                                face_center[1] - hold_anchor[1]) > STILL_RADIUS:
                hold_start = time.time()
                hold_anchor = face_center

            if not consented:
                draw_text(canvas, "Press SPACE to take your photo", W // 2 - 250, 40, 0.9)
                draw_text(canvas, "Q to quit without one", W // 2 - 130, 70, 0.6, (180, 180, 180), 1)
            else:
                draw_text(canvas, "Look at the camera and hold still",
                          W // 2 - 240, 40, 0.8)

            if hold_start is not None:
                elapsed = time.time() - hold_start
                remaining = max(0.0, HOLD_STILL_SECONDS - elapsed)
                draw_text(canvas, f"Capturing in {remaining:0.1f}s", W // 2 - 120, H // 2, 1.2, (0, 255, 255))
                if elapsed >= HOLD_STILL_SECONDS and cam_frame is not None:
                    # matte_frame blocks, and on the very first run it blocks
                    # for seconds while 25 MB of MODNet weights download. Paint
                    # the reason before handing over the thread, or the window
                    # just freezes at "Capturing in 0.0s" with nothing to read.
                    draw_text(canvas, "Cutting you out...", W // 2 - 130, H // 2 + 60, 0.9, (0, 255, 255))
                    draw_text(canvas, "first run also downloads 25 MB of matting weights",
                              W // 2 - 250, H // 2 + 90, 0.55, (150, 200, 255), 1)
                    cv2.imshow(TITLE, canvas)
                    cv2.waitKey(1)

                    capture_rgba, matted = matte_frame(cam_frame)
                    avatar, matte_note = build_sprite(capture_rgba, matted, lighting)
                    state = STATE_PLAYING
            elif consented:
                draw_text(canvas, "Waiting for a face...", W // 2 - 140, H // 2, 0.9, (0, 180, 255))

            draw_text(canvas, "Your photo stays on this machine - never saved to disk, never uploaded",
                      20, H - 20, 0.5, (150, 220, 150), 1)

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
            draw_text(canvas, matte_note, 20, 96, 0.5, (140, 220, 255), 1)
            draw_text(canvas, "Move your hand left/right to catch shapes with your avatar",
                      20, H - 20, 0.5, (150, 150, 150))

            if lives <= 0:
                state = STATE_GAMEOVER

        elif state == STATE_GAMEOVER:
            blit_rgba(canvas, avatar, W // 2, H // 2 - 60)
            draw_text(canvas, "GAME OVER", W // 2 - 130, H // 2 + 80, 1.4, (0, 0, 255), 3)
            draw_text(canvas, f"Final score: {score}", W // 2 - 100, H // 2 + 120, 0.9)
            draw_text(canvas, "Press 'R' to discard this photo and start over",
                      W // 2 - 220, H // 2 + 150, 0.7)

        cv2.imshow(TITLE, canvas)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' ') and state == STATE_CAPTURE:
            consented = True
        elif key == ord('l') and capture_rgba is not None:
            # Re-render the shot already taken; the matte is the expensive
            # part and it has not changed, so this is instant.
            lighting = not lighting
            avatar, matte_note = build_sprite(capture_rgba, matted, lighting)
        elif key == ord('r'):
            # Dropping both the sprite and the matte behind it is what makes
            # "recapture" also mean "discard the old photo" — the previous
            # shot is unreachable from here on, not just unused.
            state = STATE_CAPTURE
            consented = False
            hold_start = None
            hold_anchor = None
            avatar = None
            capture_rgba = None
            matte_note = ""
            shapes = []
            score = 0
            lives = 3
            frame_count = 0

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
