import os
import sys
import math
import time
import queue
import random
import cv2
import numpy as np

# Add visual_ai engine path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, '..'))

from engine_bootstrap import ensure_engine
ensure_engine()

from instructions import draw_card
from tracker import RunTracker, tracking_enabled
from visual_ai import VisionPipeline

W, H = 800, 600

TITLE = "Flappy — fly with your fingertip"
GOAL = ("Fly the bird through the gaps in the pipes. Hitting a pipe, the "
        "ceiling or the floor ends the run. The gaps start near the middle "
        "and drift further apart the longer you survive.")
CONTROLS = (
    ("Hold up one hand",
     "the bird follows the height of your index fingertip — nothing else "
     "on your hand matters"),
    ("Raise your finger", "the bird climbs"),
    ("Lower your finger", "the bird dives"),
    ("Stay in the middle band",
     "only the middle stretch of the camera frame is mapped to the screen, "
     "so you never have to reach to the very edge"),
    ("Lose tracking",
     "the bird drifts back to the centre instead of dropping — a brief "
     "dropout is survivable"),
)
KEYS = (
    ("R", "restart after a crash"),
    ("1 / 2 / 3", "easy / medium / hard — restarts the run at that difficulty"),
    ("T", "run tracking for this session — off unless this machine opted in; "
          "the log stays on this disk"),
    ("H", "show this card again"),
    ("K", "landmark smoothing on / off"),
    ("Q / ESC", "quit"),
)

# Fingertip control band. Only the middle stretch of the camera frame is mapped
# to the full screen height: the very top and bottom of frame are awkward to
# reach and are where MediaPipe tracking degrades, so excluding them means the
# whole playable range sits inside comfortable, reliable finger travel.
CONTROL_BAND_TOP    = 0.15 * H
CONTROL_BAND_BOTTOM = 0.85 * H

# Course geometry, in the vocabulary used throughout this file:
#
#   gap          vertical opening you fly through          (per difficulty)
#   gap centre   where that opening sits on screen         (see GapCourse)
#   gap margin   keep-out band at ceiling and floor
#   spacing      distance between consecutive pipes        (per difficulty)
#   pipe width   thickness of one pillar
#   lane         the fixed column the bird flies in
#   head start   clearance before the first pipe of a run
#   spawn lead   how far past the right edge pipes are kept stocked
#   scroll speed pixels per SECOND the pipes travel        (per difficulty)
#   pipe rate    scroll speed / spacing -- pipes per second
#   follow tau   how quickly the bird converges on the fingertip, as a time
#                constant in seconds                       (per difficulty)
# Render pacing. Everything below is expressed per second and scaled by the
# real elapsed time, so a preset means the same thing whatever the loop rate.
#
# It did not used to: motion was per frame, and the loop ran at ~32 fps rather
# than the 60 it was written for, so every difficulty ran at about half its
# nominal pace and would have sped up on faster hardware. The cause was
# cv2.waitKey(16): OpenCV's Windows highgui pumps on a ~15.9 ms tick, so a
# 16 ms request lands just past one tick and waits out a second -- 31 ms per
# frame, of which only 1.6 ms was drawing. waitKey(1) costs one tick, and the
# pacer below waits out the rest of the budget itself.
RENDER_HZ = 60.0
RENDER_DT = 1.0 / RENDER_HZ

# A frame longer than this is a stall -- a window drag, a GC pause. Advancing
# the world by the true elapsed time would teleport the pipes through the bird.
MAX_FRAME_DT = 0.10

# How fast the bird drifts back to mid-screen while the hand is lost, as a
# time constant. Applied per frame of wall time rather than per payload,
# because payloads arrive at camera rate and frames do not.
DROPOUT_TAU = 0.30

GAP_MARGIN  = 50
PIPE_WIDTH  = 60
LANE_X      = 200
BIRD_RADIUS = 15
HEAD_START  = 200

# Difficulty presets. EASY is the game exactly as it was tuned before the
# presets existed, so nothing about the default run changed.
#
# Each step up raises scroll speed and narrows both the gap and the spacing,
# which is what actually tests tracking: less screen time per pipe and a
# smaller target to hold the fingertip on. "follow_tau" shortens with it -- the
# bird converges on the fingertip faster to reach the next gap in time, at the
# cost of passing more of the raw tracking jitter through to the bird.
#
# "ramp_pipes" is how many pipes a run takes to reach its full vertical range
# (see GapCourse). It shortens with difficulty, so HARD not only ends harder,
# it gets there sooner.
#
# The speeds and time constants are the per-frame numbers (5.0/7.0/9.5 px and
# 0.35/0.45/0.55) converted at the ~32.5 fps the loop actually ran at, not at
# the 60 fps they were written for. That keeps every level feeling exactly as
# it did while the frame coupling is removed -- fixing the loop and changing
# the difficulty in one step would have left neither verifiable. They are
# honest px/s now, so raise them deliberately if HARD wants to be harder.
DIFFICULTIES = (
    {"name": "EASY",   "speed": 165.0, "gap": 200, "spacing": 400,
     "follow_tau": 0.072, "ramp_pipes": 28},
    {"name": "MEDIUM", "speed": 230.0, "gap": 160, "spacing": 320,
     "follow_tau": 0.051, "ramp_pipes": 20},
    {"name": "HARD",   "speed": 310.0, "gap": 125, "spacing": 260,
     "follow_tau": 0.039, "ramp_pipes": 14},
)
DEFAULT_DIFFICULTY = 0

def draw_text(img, text, x, y, size=1.0, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

class Pipe:
    def __init__(self, x, gap, gap_centre):
        self.x = x
        self.width = PIPE_WIDTH
        self.gap = gap
        self.gap_centre = gap_centre
        self.passed = False

    def update(self, scroll_speed, dt):
        self.x -= scroll_speed * dt

    def draw(self, frame):
        top = int(self.gap_centre - self.gap // 2)
        bottom = int(self.gap_centre + self.gap // 2)
        x0, x1 = int(self.x), int(self.x + self.width)
        # Top pipe
        cv2.rectangle(frame, (x0, 0), (x1, top), (0, 200, 0), -1)
        cv2.rectangle(frame, (x0, 0), (x1, top), (0, 100, 0), 2)
        # Bottom pipe
        cv2.rectangle(frame, (x0, bottom), (x1, H), (0, 200, 0), -1)
        cv2.rectangle(frame, (x0, bottom), (x1, H), (0, 100, 0), 2)

    def collides(self, bx, by, bird_radius):
        if bx + bird_radius > self.x and bx - bird_radius < self.x + self.width:
            if (by - bird_radius < self.gap_centre - self.gap // 2
                    or by + bird_radius > self.gap_centre + self.gap // 2):
                return True
        return False


def approach(current, target, tau, dt):
    """Move `current` toward `target` with time constant `tau`.

    The frame-rate-independent form of the old per-frame EMA: the fraction
    covered per frame is derived from how long the frame actually took, so the
    bird converges at the same rate in wall-clock terms whether the loop is
    running at 30 fps or 120.
    """
    if tau <= 0.0:
        return target
    return target + (current - target) * math.exp(-dt / tau)


def _smoothstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


class GapCourse:
    """Places gap centres for one run, and owns the pipes on screen.

    The vertical travel between consecutive gap centres is what makes a run
    feel hard -- more so than the gap itself. So it ramps: the first pipes of
    a run sit near screen centre with only a short hop between them, and the
    reach grows to nearly the whole playable band over `ramp_pipes` pipes.

    Three things keep the ramp from reading as a script:

    * The reach is a *ceiling*, not the step. Each actual step is drawn from
      REACH_JITTER x reach, so an early pipe can still hop further than its
      neighbour and a late one can sit almost still.
    * Direction is drawn per pipe, never alternated. The weighting is just the
      room left on each side, so it is an even coin toss at mid-screen and
      leans back toward the middle near the ceiling or floor -- a mean-
      reverting walk that visits up and down in no guessable order and never
      has to be yanked off an edge.
    * A step that would overshoot the playable band reflects off it rather
      than clamping. Clamping would stack gap centres flat against the margins
      and give the boundary away.

    Measured over 200 HARD runs, consecutive gap centres change direction on
    65% of pipes early and 78% late -- not the near-perfect zigzag (89%) a
    full-band reach produced, which is why REACH_FULL stops at 0.88 and the
    jitter floor is low.

    The band is [gap margin + gap/2, H - gap margin - gap/2], so the whole
    opening is always on screen whatever the difficulty's gap.
    """

    REACH_START  = 0.16   # opening reach, as a fraction of the band
    REACH_FULL   = 0.88   # reach once the ramp is done
    REACH_JITTER = (0.25, 1.0)   # jitter floor -- see the note above

    def __init__(self, diff):
        self.diff = diff
        half = diff["gap"] // 2
        self.lo = GAP_MARGIN + half
        self.hi = H - GAP_MARGIN - half
        self.band = self.hi - self.lo
        # A run opens dead centre: the player's finger starts mid-band, and
        # the first pipe should not cost them a scramble before they have
        # found the mapping.
        self.centre = (self.lo + self.hi) / 2.0
        self.spawned = 0
        self.pipes = []
        self.refill()

    def _next_gap_centre(self):
        if self.spawned == 0:
            self.spawned = 1
            return self.centre

        t = self.spawned / float(self.diff["ramp_pipes"])
        reach = self.band * (self.REACH_START
                             + (self.REACH_FULL - self.REACH_START) * _smoothstep(t))
        step = reach * random.uniform(*self.REACH_JITTER)

        # Direction by remaining room: 50/50 mid-band, biased inward near an
        # edge. No alternation, so the pattern is not learnable.
        room_up = self.centre - self.lo
        room_down = self.hi - self.centre
        if random.uniform(0.0, room_up + room_down) < room_down:
            nxt = self.centre + step
        else:
            nxt = self.centre - step

        # Reflect off the band instead of clamping to it.
        if nxt > self.hi:
            nxt = self.hi - (nxt - self.hi)
        elif nxt < self.lo:
            nxt = self.lo + (self.lo - nxt)
        self.centre = min(self.hi, max(self.lo, nxt))

        self.spawned += 1
        return self.centre

    def refill(self):
        """Stock pipes one spawn lead (a full spacing) past the right edge.

        Filling by distance rather than keeping a fixed-length list is what
        makes the tighter spacings read as tighter -- a fixed pair left HARD's
        260 px spacing covering only a third of the screen.
        """
        spacing = self.diff["spacing"]
        while not self.pipes or self.pipes[-1].x < W + spacing:
            x = self.pipes[-1].x + spacing if self.pipes else W + HEAD_START
            self.pipes.append(Pipe(x, self.diff["gap"], self._next_gap_centre()))

    def scroll(self, dt):
        """Advance every pipe, retire the ones off the left edge, restock."""
        for pipe in self.pipes:
            pipe.update(self.diff["speed"], dt)
        while self.pipes and self.pipes[0].x < -self.pipes[0].width:
            self.pipes.pop(0)
        self.refill()


def draw_tracking_status(canvas, hand_visible, w=800):
    x0 = w - 120
    y0 = 20
    status_col = (0, 255, 0) if hand_visible else (0, 0, 255)
    status_text = "TRACKING" if hand_visible else "NO HAND"
    cv2.circle(canvas, (x0, y0), 8, status_col, -1)
    cv2.putText(canvas, status_text, (x0 + 15, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

def main():
    print("Starting Flappy...")
    ai_queue = queue.Queue(maxsize=1)

    # No depth of any kind: this game steers off the index fingertip's Y in
    # the RGB frame and reads no depth key at all. It used to switch the
    # pipeline's simulated-depth flag on and offer a depth-stabilizer key,
    # both of which gated a Z axis nothing here consumes -- controls that did
    # nothing, under a name that promised a sensor the machine has not got.
    pipeline = VisionPipeline(
        result_queue=ai_queue,
        width=W,
        height=H,
        # Flappy steers off index-finger height alone and reads no face key,
        # so the face graph was pure cost — 3.1 ms/frame of inference.
        detect_face=False,
    )
    pipeline.disable_camera = False
    pipeline.start()

    cv2.namedWindow("Flappy", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Flappy", W, H)

    bird_y = H // 2

    diff_idx = DEFAULT_DIFFICULTY
    diff = DIFFICULTIES[diff_idx]
    course = GapCourse(diff)
    score = 0

    # Offline diagnostics, off unless this machine opted in: it is an
    # instrument for tuning here, not something that should follow a copy of
    # the game to whoever else runs it. `tracker_report.py --enable` switches
    # it on for this machine; T toggles the current session either way.
    # Records numbers only -- no frames, no landmarks -- to flappy/data/.
    tracking_on = tracking_enabled()
    tracker = RunTracker(diff, enabled=tracking_on)
    last_frame_t = time.perf_counter()
    next_render = time.perf_counter()

    def paced_key():
        """Pump highgui, return the key, and wait out the rest of the budget.

        waitKey returns as soon as a key arrives, so the long timeout paces
        idle frames without adding input latency. Never ask for exactly the
        remaining milliseconds: OpenCV's tick is ~15.9 ms and rounding up over
        it costs a whole extra tick, which is what made this loop run at 32 fps
        instead of 60.
        """
        nonlocal next_render
        wait_ms = int((next_render - time.perf_counter()) * 1000.0)
        key = cv2.waitKey(max(1, wait_ms)) & 0xFF
        # Re-base rather than accumulate: a frame that overran its budget must
        # not bank credit and let the next few run back-to-back.
        next_render = max(time.perf_counter(), next_render + RENDER_DT)
        return key
    game_over = False

    target_y = float(H // 2)
    cam_frame = None
    hand_visible = False
    smoothing_on = True

    # The card is up before the first pipe moves. The pipeline keeps running
    # behind it, so the camera has warmed up and the hand is already tracked by
    # the time the player starts.
    showing_help = True

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

        if latest is not None:
            cam_frame = latest.get("frame")
            hand_visible = latest.get("hand_visible", False)
            if hand_visible:
                # Height follows the index fingertip directly.
                #
                # This used to map ToF depth to height, but with no real sensor
                # the depth came from the simulated branch (0.45 + lm_z * 0.6).
                # MediaPipe's relative Z spans roughly +/-0.1, so it drove only
                # a sliver of the [0.2, 0.5] range it was mapped through --
                # cramped at one end and saturating at the other. Fingertip Y is
                # absolute, uses the full screen, and is far easier to hold
                # steady with one finger.
                finger_y = float(latest.get("index_pos", (0, H // 2))[1])
                target_y = np.interp(finger_y, [CONTROL_BAND_TOP, CONTROL_BAND_BOTTOM],
                                    [0, H])

        now = time.perf_counter()
        frame_ms = (now - last_frame_t) * 1000.0
        last_frame_t = now
        dt = min(frame_ms / 1000.0, MAX_FRAME_DT)

        if not hand_visible:
            # Hand lost: let the bird drift toward the middle rather than
            # snapping, so a brief tracking dropout is survivable. On wall
            # time, not per payload -- payloads stop arriving in exactly the
            # case this handles.
            target_y = approach(target_y, H / 2.0, DROPOUT_TAU, dt)

        # Smooth bird movement towards target
        if not showing_help:
            bird_y = approach(bird_y, target_y, diff["follow_tau"], dt)
            if not game_over:
                tracker.frame(target_y, bird_y, hand_visible, frame_ms)

        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (40, 30, 20) # Dark background

        if not game_over and not showing_help:
            course.scroll(dt)
            for pipe in course.pipes:
                if not pipe.passed and pipe.x + pipe.width < LANE_X:
                    pipe.passed = True
                    score += 1
                    tracker.pipe_cleared(score, bird_y, pipe.gap_centre,
                                         pipe.gap)

            # Collision
            for pipe in course.pipes:
                if pipe.collides(LANE_X, bird_y, BIRD_RADIUS):
                    game_over = True
                    tracker.end("pipe", score)
            if bird_y > H or bird_y < 0:
                game_over = True
                tracker.end("floor" if bird_y > H else "ceiling", score)

        # Draw pipes
        for pipe in course.pipes:
            pipe.draw(frame)

        # Draw bird
        bird_color = (0, 200, 255) if not game_over else (0, 0, 255)
        cv2.circle(frame, (LANE_X, int(bird_y)), BIRD_RADIUS, bird_color, -1)
        cv2.circle(frame, (LANE_X, int(bird_y)), BIRD_RADIUS, (0, 100, 255), 2)

        # Draw UI
        draw_text(frame, f"Score: {score}", 20, 40, 1.0)
        draw_text(frame, f"Difficulty: {diff['name']}", 20, 110, 0.7, (0, 220, 255))
        # An optional feature that fails quietly is a feature that reads as
        # broken, so the recording state is on the HUD rather than on stdout.
        draw_text(frame, "REC (local only)" if tracking_on else "REC off", 20, 145, 0.6,
                  (0, 200, 120) if tracking_on else (120, 120, 120))
        if latest is not None:
            smoothing_on = latest.get("smoothing_enabled", True)
        draw_text(frame, f"Smoothing: {'ON' if smoothing_on else 'OFF'}", 20, 80, 0.7,
                  (0, 255, 0) if smoothing_on else (0, 180, 255))
        draw_text(frame, "Raise/lower your index finger to fly  |  1/2/3: difficulty  |  T: tracking  |  H: how to play  |  K: smoothing",
                  20, H - 20, 0.5, (150, 150, 150))

        if game_over:
            draw_text(frame, "GAME OVER", W//2 - 150, H//2, 2.0, (0, 0, 255), 3)
            draw_text(frame, "Press 'R' to restart", W//2 - 120, H//2 + 50, 0.8)

        draw_tracking_status(frame, hand_visible, W)

        if showing_help:
            draw_card(frame, TITLE, GOAL, CONTROLS, KEYS)

        cv2.imshow("Flappy", frame)

        key = paced_key()
        if key in (27, ord('q')):
            break
        elif showing_help:
            # Any key starts. Nothing else is dispatched this frame — the key
            # that dismisses the card should not also toggle a filter.
            if key != 255:
                showing_help = False
                # Start the clock when play starts, not when the window opens,
                # so time spent reading the card is not logged as run duration.
                # Only if nothing has been recorded yet -- pressing H mid-run
                # must not throw that run's samples away.
                if not tracker.frames:
                    tracker = RunTracker(diff, enabled=tracking_on)
        elif key in (ord('h'), ord('H')):
            showing_help = True
        elif key in (ord('k'), ord('K')):
            on = pipeline.toggle_smoothing()
            print(f"[Stabilisation] Landmark smoothing: {'ON' if on else 'OFF (raw positions)'}")
        elif key in (ord('t'), ord('T')):
            tracking_on = not tracking_on
            if not tracking_on:
                tracker.end("switch", score)
            tracker = RunTracker(diff, enabled=tracking_on)
            print(f"[tracker] run logging: {'ON (local file)' if tracking_on else 'OFF'}")
        elif key in (ord('1'), ord('2'), ord('3')):
            # Changing difficulty restarts the run: the pipes already on screen
            # were laid out for the old spacing and gap, so keeping them would
            # mix two difficulties into one score.
            diff_idx = key - ord('1')
            diff = DIFFICULTIES[diff_idx]
            tracker.end("switch", score)
            tracker = RunTracker(diff, enabled=tracking_on)
            score = 0
            course = GapCourse(diff)
            bird_y = H // 2
            target_y = float(H // 2)
            game_over = False
        elif key == ord('r') and game_over:
            tracker = RunTracker(diff, enabled=tracking_on)
            score = 0
            course = GapCourse(diff)
            bird_y = H // 2
            target_y = float(H // 2)
            game_over = False

    # Quitting mid-run still writes it; end() is idempotent, so a run that
    # already ended in a crash is not logged twice.
    tracker.end("quit", score)

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
