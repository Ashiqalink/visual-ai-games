"""
gameloop.py - the per-frame plumbing the standalone OpenCV games share.

flappy, punchy and avatarcatch are not lab games: they do not subclass
`LabGame`, do not run headless off scripted input, and do not want labkit's
report machinery. What they do want is the same three things labkit solves
internally, and each had grown its own byte-identical copy:

  drain        take the freshest payload, per the queue contract
  draw_text    outlined text, legible over a camera feed
  frame_pacer  a waitKey that also paces the loop at a fixed rate

This is deliberately small. It is a place for what those games genuinely
share, not a second labkit - anything that needs a game loop, a scripted
input mode and a JSON report should be a `LabGame` instead.

`labkit` keeps its own `drain` and `draw_text` on purpose. Its drain reports
how many payloads were dropped, which its run reports read, and its text is
antialiased and more heavily outlined. Converging them would change what the
lab games render and what their reports say, to remove a duplication that is
only skin deep.
"""

from __future__ import annotations

import queue
import time

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Render budget for a 60 fps loop, in seconds.
RENDER_DT = 1.0 / 60.0


def drain(q: queue.Queue):
    """
    Take the freshest payload and discard the rest; None if there was none.

    The queue contract is `maxsize=1` plus drain-until-Empty. A single
    `get_nowait()` leaves the game acting on a stale frame every time the
    render loop runs slower than the pipeline produces - which is most of
    them, since the pipeline is a camera and the render loop is not.
    """
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            return latest


def draw_text(img, text, x, y, size=1.0, color=(255, 255, 255), thickness=2):
    """Text with a dark outline, so it stays readable over a camera feed."""
    cv2.putText(img, text, (int(x), int(y)), FONT, size, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (int(x), int(y)), FONT, size, color, thickness)


def frame_pacer(render_dt: float = RENDER_DT):
    """
    Return a `key()` callable that pumps highgui and paces the loop.

    Call it once per frame in place of `cv2.waitKey`. It returns the key that
    was pressed, masked to a byte, and spends whatever is left of the frame's
    budget waiting for one.

    `waitKey` returns as soon as a key arrives, so a long timeout paces idle
    frames without adding any input latency. Never ask it for exactly the
    remaining milliseconds: OpenCV's tick is ~15.9 ms and rounding up over it
    costs a whole extra tick - which is what had punchy running at 32 fps
    against a 60 fps budget.
    """
    next_render = time.perf_counter()

    def key() -> int:
        nonlocal next_render
        wait_ms = int((next_render - time.perf_counter()) * 1000.0)
        pressed = cv2.waitKey(max(1, wait_ms)) & 0xFF
        # Re-base rather than accumulate: a frame that overran its budget must
        # not bank credit and let the next few run back-to-back.
        next_render = max(time.perf_counter(), next_render + render_dt)
        return pressed

    return key
