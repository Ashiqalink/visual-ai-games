"""
game.py — Game state machine.

States
------
SELECTION  : carousel, choose bird via pinch OR z-push click
ARMED      : pinch to drag slingshot, release to fire
FLIGHT     : bird in flight, collision detection running
DONE       : all birds used

Improvements over original:
  - Scoring system (500/block, 100/debris)
  - Floating score popups
  - Multi-block hits (bird pushes through weak blocks)
  - 3 level layouts (Easy / Medium / Hard)
  - Level selection via 1 / 2 / 3 keys
  - Slingshot depth-layered draw order
  - Slingshot snap-back animation on release
"""

import cv2
import math
import random
import numpy as np
from bird import Bird, BIRD_ORDER, COLOURS
from block import Block
from slingshot import SLING_X, SLING_Y, FORK_LEFT, FORK_RIGHT
from physics import (
    POWER_FACTOR, MAX_PULL, FLOOR_Y, BIRD_LINGER,
    bird_hits_block, distance, magnitude,
    resolve_block_collision,
)
import slingshot
import ui
from config import (
    PTS_BLOCK, PTS_DEBRIS,
    SCORE_POPUP_LIFETIME, SCORE_POPUP_RISE,
    ARMED_GRACE_FRAMES, MIN_FIRE_PULL, EDGE_MARGIN,
    AIM_PULL_GAIN, INPUT_MOVEMENT_MAGNIFICATION, AIM_EMA_HOLD, AIM_EMA_MIN, AIM_EMA_MAX,
    AIM_EMA_JITTER_LO, AIM_EMA_JITTER_HI,
    SEL_DEBOUNCE_FRAMES, Z_PUSH_DETECT_THRESH,
    DEBRIS_LIFESPAN, DEBRIS_VEL_SPREAD, DEBRIS_VY_KICK,
    DEBRIS_CARRY_FACTOR, DEBRIS_HEALTH,
    PLATFORM_W, PLATFORM_H, PLATFORM_HEALTH,
    BIRD_STOP_SPEED,
)

# ── Score constants ────────────────────────────────────────────────────────────────
# (imported from config above)


# ── Score popup ───────────────────────────────────────────────────────────────
class ScorePopup:
    """Floating "+N" text that rises and fades out."""
    def __init__(self, x: float, y: float, points: int):
        self.x = float(x)
        self.y = float(y)
        self.points = points
        self.timer = SCORE_POPUP_LIFETIME   # frames to live

    def update(self) -> bool:
        """Returns True while still alive."""
        self.y -= SCORE_POPUP_RISE          # float upward
        self.timer -= 1
        return self.timer > 0

    def draw(self, frame: np.ndarray):
        if self.timer <= 0:
            return
        alpha = max(0.0, self.timer / SCORE_POPUP_LIFETIME)
        col = (0, int(255 * alpha), int(200 * alpha))
        text = f"+{self.points}"
        x, y = int(self.x), int(self.y)
        # Drop shadow + bright text
        cv2.putText(frame, text, (x + 1, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)


# ── Level layouts ─────────────────────────────────────────────────────────────
LEVEL_NAMES = ["Easy", "Medium", "Hard"]

def _level_easy() -> list[Block]:
    """Simple wooden tower — good for learning controls."""
    blocks: list[Block] = []
    BW, BH = 30, 60
    TH = 20

    # Three pillars
    for gx in [900, 960, 1020]:
        blocks.append(Block(gx, FLOOR_Y - BH, BW, BH, "wood"))

    # Plank on top
    blocks.append(Block(900, FLOOR_Y - BH - TH, 150, TH, "wood"))

    # Single block on top
    blocks.append(Block(950, FLOOR_Y - BH * 2 - TH, BW, BH, "wood"))

    return blocks


def _level_medium() -> list[Block]:
    """Pyramid with mixed wood + ice."""
    blocks: list[Block] = []
    BW, BH = 30, 60
    TH = 20

    # Ground row — outer wood, inner ice
    blocks.append(Block(820, FLOOR_Y - BH, BW, BH, "wood"))
    blocks.append(Block(900, FLOOR_Y - BH, BW, BH, "ice"))
    blocks.append(Block(980, FLOOR_Y - BH, BW, BH, "ice"))
    blocks.append(Block(1060, FLOOR_Y - BH, BW, BH, "wood"))

    # Ground plank
    blocks.append(Block(820, FLOOR_Y - BH - TH, 280, TH, "wood"))

    # Mid row
    blocks.append(Block(860, FLOOR_Y - BH * 2 - TH, BW, BH, "wood"))
    blocks.append(Block(980, FLOOR_Y - BH * 2 - TH, BW, BH, "wood"))

    # Mid plank
    blocks.append(Block(860, FLOOR_Y - BH * 2 - TH * 2, 180, TH, "wood"))

    # Top — ice (fragile crown)
    blocks.append(Block(920, FLOOR_Y - BH * 3 - TH * 2, BW, BH, "ice"))

    return blocks


def _level_hard() -> list[Block]:
    """Stone fortress — two towers with a bridge, mixed materials."""
    blocks: list[Block] = []
    BW, BH = 30, 60
    TH = 20

    # ── Left tower ────────────────────────────────────────────────────────
    # Stone base
    blocks.append(Block(780, FLOOR_Y - BH, BW, BH, "stone"))
    blocks.append(Block(840, FLOOR_Y - BH, BW, BH, "stone"))
    # Base plank
    blocks.append(Block(780, FLOOR_Y - BH - TH, 90, TH, "wood"))
    # Wood pillar
    blocks.append(Block(800, FLOOR_Y - BH * 2 - TH, BW, BH, "wood"))
    # Ice cap
    blocks.append(Block(800, FLOOR_Y - BH * 3 - TH, BW, BH, "ice"))

    # ── Right tower ───────────────────────────────────────────────────────
    blocks.append(Block(1000, FLOOR_Y - BH, BW, BH, "stone"))
    blocks.append(Block(1060, FLOOR_Y - BH, BW, BH, "stone"))
    blocks.append(Block(1000, FLOOR_Y - BH - TH, 90, TH, "wood"))
    blocks.append(Block(1020, FLOOR_Y - BH * 2 - TH, BW, BH, "wood"))
    blocks.append(Block(1020, FLOOR_Y - BH * 3 - TH, BW, BH, "ice"))

    # ── Bridge between towers ─────────────────────────────────────────────
    blocks.append(Block(870, FLOOR_Y - BH * 2 - TH, 130, TH, "wood"))
    blocks.append(Block(900, FLOOR_Y - BH * 3 - TH, BW, BH, "ice"))
    blocks.append(Block(960, FLOOR_Y - BH * 3 - TH, BW, BH, "ice"))

    # ── Outer buttresses ──────────────────────────────────────────────────
    blocks.append(Block(740, FLOOR_Y - BH, BW, BH, "wood"))
    blocks.append(Block(1100, FLOOR_Y - BH, BW, BH, "wood"))

    return blocks


_LEVELS = [_level_easy, _level_medium, _level_hard]


def _build_level(level_idx: int) -> list[Block]:
    idx = max(0, min(level_idx, len(_LEVELS) - 1))
    blocks = _LEVELS[idx]()
    
    # Move blocks to center (X) and same level as slingshot (Y)
    # Old blocks were centered around ~960, new center is 640
    X_OFF = -320
    # Old base was FLOOR_Y (660), new base is SLING_Y (440)
    Y_OFF = SLING_Y - FLOOR_Y
    
    for b in blocks:
        b.rect[0] += X_OFF
        b.rect[1] += Y_OFF

    # Create a static platform for them to rest on (so they don't fall to FLOOR_Y)
    plat_w = PLATFORM_W
    plat_h = PLATFORM_H
    plat_x = 640 - plat_w / 2
    plat_y = SLING_Y
    platform = Block(plat_x, plat_y, plat_w, plat_h, "wood")
    platform.is_static = True
    platform.health = PLATFORM_HEALTH
    platform.max_health = PLATFORM_HEALTH
    blocks.append(platform)

    return blocks


# ══════════════════════════════════════════════════════════════════════════════

class Game:
    def __init__(self, frame_w: int = 1280, frame_h: int = 720):
        self.W = frame_w
        self.H = frame_h
        self.level_idx: int = 0
        self.score: int = 0
        self.fps: int = 0                              # set by main.py each frame
        self.reset()

    # ── Public ────────────────────────────────────────────────────────────────

    def reset(self):
        self.state         = "SELECTION"
        self.bird_queue    = list(BIRD_ORDER)           # kinds left to launch
        self.selected_idx  = 0                          # carousel index
        self.current_bird: Bird | None = None
        self.blocks        = _build_level(self.level_idx)
        self.pull_pos      = (SLING_X, SLING_Y)         # where bird is during ARMED
        self._last_click_mode = "Z-PUSH"
        self._z_delta_display  = 0.0                    # for HUD debug bar
        self._xy_drift_display = 0.0                    # for HUD debug lateral drift
        self._armed_timer: int = 0                      # grace period timer on entering ARMED state
        self._last_click_fired: bool = False            # pass to HUD
        # Smoothed index-finger position for aiming in ARMED state
        self._smoothed_ix  = float(SLING_X)
        self._smoothed_iy  = float(SLING_Y)
        self._aim_anchor_x = float(SLING_X)
        self._aim_anchor_y = float(SLING_Y)
        self._AIM_PULL_GAIN = AIM_PULL_GAIN             # Pull sensitivity gain
        self._SMOOTH        = AIM_EMA_MIN               # EMA factor (base, overridden adaptively)
        # Carousel debounce: require N stable frames before committing selection
        self._sel_candidate: int = 0                   # index being hovered
        self._sel_stable_frames: int = 0               # consecutive frames on same index
        self._SEL_DEBOUNCE: int = SEL_DEBOUNCE_FRAMES  # frames needed to confirm selection
        # Edge-fire guard: only allow edge-exit firing after hand was inside frame
        self._armed_inside: bool = False
        # 3-Finger Pinch Aiming state flags
        self._is_aiming: bool = False
        self._was_unpinched_after_armed: bool = False
        # Scoring
        self.score_popups: list[ScorePopup] = []

    def update(self, gesture: dict, key: int) -> np.ndarray | None:
        """
        Called every frame.  Returns None (frame modified in-place by caller).

        gesture keys used
        -----------------
        hand_visible, index_pos, pinch_pos, is_pinching, click_just_fired
        """
        # Store gesture & ToF depth telemetry
        self._last_gesture = gesture
        self._tof_active   = gesture.get("tof_active", False)
        self._tof_z_m      = gesture.get("tof_z_m", 0.0)
        self._depth_source = gesture.get("depth_source", "RGB MediaPipe Estimate")

        # ── Key handling ──────────────────────────────────────────────────
        if key == ord('r') or key == ord('R'):
            self.reset()
            return

        # Level selection: 1 / 2 / 3
        if key in (ord('1'), ord('2'), ord('3')):
            self.level_idx = key - ord('1')
            self.score = 0
            self.reset()
            return

        # Magnification controls: + / = to increase, - / _ to decrease
        if key in (ord('+'), ord('=')):
            self._AIM_PULL_GAIN = round(min(5.0, self._AIM_PULL_GAIN + 0.5), 1)
        elif key in (ord('-'), ord('_')):
            self._AIM_PULL_GAIN = round(max(0.5, self._AIM_PULL_GAIN - 0.5), 1)

        # ── Slingshot animation tick ──────────────────────────────────────
        slingshot.tick()

        # ── State machine ─────────────────────────────────────────────────
        if self.state == "SELECTION":
            self._update_selection(gesture)
        elif self.state == "ARMED":
            self._update_armed(gesture)
        elif self.state == "FLIGHT":
            self._update_flight(gesture)

        # ── Always update blocks ──────────────────────────────────────────
        new_blocks: list[Block] = []
        for b in self.blocks:
            was_active = b.active
            b.update()
            if was_active and not b.active:
                # Block just broke — score + debris
                is_big = b.rect[2] > 20 and b.rect[3] > 20
                pts = PTS_BLOCK if is_big else PTS_DEBRIS
                self.score += pts
                self.score_popups.append(ScorePopup(b.cx, b.cy, pts))

                if is_big:
                    hw = b.rect[2] / 2
                    hh = b.rect[3] / 2
                    for i in range(2):
                        for j in range(2):
                            debris = Block(b.rect[0] + i * hw,
                                           b.rect[1] + j * hh,
                                           hw, hh,
                                           material=b.material)
                            debris.vx = (random.random() * 2.0 - 1.0) * DEBRIS_VEL_SPREAD + b.vx * DEBRIS_CARRY_FACTOR
                            debris.vy = (random.random() * 2.0 - 1.0) * DEBRIS_VEL_SPREAD + b.vy * DEBRIS_CARRY_FACTOR - DEBRIS_VY_KICK
                            debris.health = DEBRIS_HEALTH
                            debris.is_debris = True
                            debris.lifespan = DEBRIS_LIFESPAN
                            new_blocks.append(debris)

        if new_blocks:
            self.blocks.extend(new_blocks)

        # ── Resolve block-block collisions ────────────────────────────────
        for i in range(len(self.blocks)):
            for j in range(i + 1, len(self.blocks)):
                b1 = self.blocks[i]
                b2 = self.blocks[j]
                if b1.active and b2.active:
                    resolve_block_collision(b1, b2)

        # ── Update score popups ───────────────────────────────────────────
        self.score_popups = [p for p in self.score_popups if p.update()]

    def draw(self, frame: np.ndarray):
        """Render everything onto frame (modifies in-place)."""

        # Blocks
        for b in self.blocks:
            b.draw(frame)

        if self.state == "SELECTION":
            slingshot.draw(frame, bird_pos=None)
            ui.draw_carousel(frame, self.bird_queue, self.selected_idx)

        elif self.state == "ARMED":
            bird = self.current_bird
            pull_d = distance((bird.x, bird.y), (SLING_X, SLING_Y))
            bp = (bird.x, bird.y)

            # Depth-layered: back elastic → bird → front elastic + structure
            slingshot.draw_back(frame, bird_pos=bp, pull_dist=pull_d)
            bird.draw(frame)
            slingshot.draw_front(frame, bird_pos=bp, pull_dist=pull_d)

            # Trajectory preview (only drawn while actively aiming with pinch)
            if self._is_aiming and pull_d > 5:
                vx, vy = self._launch_velocity()
                ui.draw_trajectory(frame, bird.x, bird.y, vx, vy)

        elif self.state == "FLIGHT":
            slingshot.draw(frame, bird_pos=None)
            if self.current_bird and self.current_bird.active:
                self.current_bird.draw(frame)

        elif self.state == "DONE":
            slingshot.draw(frame, bird_pos=None)
            ui.draw_done_overlay(frame, score=self.score)

        # Score popups (floating +N text)
        for popup in self.score_popups:
            popup.draw(frame)

        # HUD always on top
        ui.draw_hud(
            frame,
            state=self.state,
            birds_left=self.bird_queue,
            click_mode=self._last_click_mode,
            z_debug=self._z_delta_display,
            xy_drift=self._xy_drift_display,
            click_fired=self._last_click_fired,
            score=self.score,
            level_idx=self.level_idx,
            fps=self.fps,
            tof_active=getattr(self, "_tof_active", False),
            tof_z_m=getattr(self, "_tof_z_m", 0.0),
            depth_source=getattr(self, "_depth_source", "RGB MediaPipe Estimate"),
            gesture=getattr(self, "_last_gesture", None),
            magnification=self._AIM_PULL_GAIN,
        )

    # ── State handlers ────────────────────────────────────────────────────────

    def _update_selection(self, g: dict):
        if not g["hand_visible"]:
            return

        # Scroll carousel with X position.
        # Pipeline already mirrors the frame before MediaPipe, so raw index_pos
        # is already in display-space — no extra flip needed.
        raw_ix, raw_iy = g.get("pinch_pos", g["index_pos"])
        ix, iy = float(raw_ix), float(raw_iy)

        total = len(self.bird_queue)
        if total == 0:
            self.state = "DONE"
            return

        # Carousel area parameters (top-center panel: y <= 220)
        spacing = 120
        panel_w = spacing * total + 80
        panel_x = self.W // 2 - panel_w // 2

        # Selecting bird can only be chosen once cursor goes into the carousel area
        in_selection_area = (iy <= 220)

        if in_selection_area:
            rel_x = (ix - panel_x) / panel_w
            candidate = int(rel_x * total)
            candidate = max(0, min(total - 1, candidate))

            # Debounce: commit only after _SEL_DEBOUNCE consecutive frames on same bird
            if candidate == self._sel_candidate:
                self._sel_stable_frames += 1
            else:
                self._sel_candidate = candidate
                self._sel_stable_frames = 1            # reset counter on change

            if self._sel_stable_frames >= self._SEL_DEBOUNCE:
                self.selected_idx = self._sel_candidate

        # Detect 1st pinch trigger: 3-finger pinch selection (ONLY allowed inside selection area)
        is_pinching = g.get("is_pinching", False) or g.get("is_3_finger_pinching", False)
        click_trigger = (g.get("click_just_fired", False) or is_pinching) and in_selection_area

        if click_trigger:
            self._last_click_mode = "3-FINGER PINCH"
            self._last_click_fired = True
            if total > 0:
                kind = self.bird_queue[self.selected_idx]
                self.current_bird = Bird(kind, SLING_X, SLING_Y)
                self.current_bird.x = float(SLING_X)
                self.current_bird.y = float(SLING_Y)
                # Anchor relative aiming to current hand position
                if g.get("hand_visible", False):
                    anc_x, anc_y = g.get("pinch_pos", g["index_pos"])
                else:
                    anc_x, anc_y = float(SLING_X), float(SLING_Y)
                self._aim_anchor_x = float(anc_x)
                self._aim_anchor_y = float(anc_y)
                self._smoothed_ix  = float(anc_x)
                self._smoothed_iy  = float(anc_y)
                self._armed_inside = False         # reset edge-fire guard
                self._armed_timer = 0              # immediate readiness
                self._is_aiming = False            # bird stays idle until 2nd pinch
                self._was_unpinched_after_armed = False # require release before 2nd pinch
                self.state = "ARMED"
        else:
            self._last_click_fired = False

    def _update_armed(self, g: dict):
        """
        Pinch Slingshot Controls:
        1. On selection, bird stays IDLE in slingshot at (SLING_X, SLING_Y).
        2. Moving hand without pinching does NOT pull the slingshot.
        3. Pinching (3-finger pinch) a 2nd time activates AIMING & locks the anchor.
        4. Moving hand while pinching pulls the slingshot rubber band.
        5. Releasing the pinch launches the bird (if pull >= MIN_FIRE_PULL).
        """
        bird = self.current_bird
        if bird is None:
            return

        margin = EDGE_MARGIN      # px from edge — larger = less accidental edge-fires

        if g.get("click_just_fired", False):
            self._last_click_fired = True
        else:
            self._last_click_fired = False

        if self._armed_timer > 0:
            self._armed_timer -= 1

        is_pinching = g.get("is_pinching", False) or g.get("is_3_finger_pinching", False)

        # 1. Require user to unpinch at least once after entering ARMED before aiming can begin
        if not self._was_unpinched_after_armed:
            if not is_pinching:
                self._was_unpinched_after_armed = True
            bird.x = float(SLING_X)
            bird.y = float(SLING_Y)
            return

        # 2. Handle UNPINCHED state
        if not is_pinching:
            if self._is_aiming:
                # User was aiming and just RELEASED the pinch -> FIRE!
                rel_dx = (self._smoothed_ix - self._aim_anchor_x) * self._AIM_PULL_GAIN
                rel_dy = (self._smoothed_iy - self._aim_anchor_y) * self._AIM_PULL_GAIN
                dist = math.sqrt(rel_dx * rel_dx + rel_dy * rel_dy)

                if dist >= MIN_FIRE_PULL:
                    self._fire_bird(bird)
                    self._is_aiming = False
                    return
                else:
                    # Insufficient pull distance: cancel aiming, return to idle
                    self._is_aiming = False
                    bird.x = float(SLING_X)
                    bird.y = float(SLING_Y)
            else:
                # Stay idle in slingshot
                bird.x = float(SLING_X)
                bird.y = float(SLING_Y)
            return

        # 3. Handle PINCHED state (actively aiming)
        if g.get("hand_visible", False):
            raw_ix, raw_iy = g.get("pinch_pos", g["index_pos"])
            ix, iy = float(raw_ix), float(raw_iy)
        else:
            ix, iy = self._smoothed_ix, self._smoothed_iy

        # Lock anchor on first frame of 2nd pinch
        if not self._is_aiming:
            self._is_aiming = True
            self._aim_anchor_x = ix
            self._aim_anchor_y = iy
            self._smoothed_ix  = ix
            self._smoothed_iy  = iy

        # Mark hand as "inside" once it's away from all edges
        if (margin < ix < self.W - margin and margin < iy < self.H - margin):
            self._armed_inside = True

        at_edge = (ix < margin or ix > self.W - margin
                   or iy < margin or iy > self.H - margin)

        # ── Adaptive EMA smooth ──
        d_dist = math.sqrt((ix - self._smoothed_ix) ** 2 + (iy - self._smoothed_iy) ** 2)
        scale = max(0.0, min(1.0, (d_dist - AIM_EMA_JITTER_LO) / (AIM_EMA_JITTER_HI - AIM_EMA_JITTER_LO)))
        alpha = AIM_EMA_MIN + scale * (AIM_EMA_MAX - AIM_EMA_MIN)

        self._smoothed_ix = alpha * ix + (1 - alpha) * self._smoothed_ix
        self._smoothed_iy = alpha * iy + (1 - alpha) * self._smoothed_iy

        # ── Map relative hand displacement from anchor with gain multiplier ───
        rel_dx = (self._smoothed_ix - self._aim_anchor_x) * self._AIM_PULL_GAIN
        rel_dy = (self._smoothed_iy - self._aim_anchor_y) * self._AIM_PULL_GAIN

        dist = math.sqrt(rel_dx * rel_dx + rel_dy * rel_dy)
        dx = rel_dx
        dy = rel_dy

        if dist > MAX_PULL:
            dx = dx / dist * MAX_PULL
            dy = dy / dist * MAX_PULL

        bird.x = SLING_X + dx
        bird.y = SLING_Y + dy

        # Edge-exit fire fallback while aiming
        if self._armed_timer <= 0 and dist >= MIN_FIRE_PULL:
            if at_edge and self._armed_inside:
                self._fire_bird(bird)
                self._is_aiming = False
                return

    def _fire_bird(self, bird: Bird):
        """Transition from ARMED → FLIGHT."""
        vx, vy = self._launch_velocity()
        bird.vx = vx
        bird.vy = vy
        bird.launched = True

        # Remove by kind so the correct entry is popped regardless of whether
        # selected_idx drifted since the bird was chosen.
        try:
            self.bird_queue.remove(bird.kind)
        except ValueError:
            pass
        self.selected_idx = max(0, min(self.selected_idx, len(self.bird_queue) - 1))

        # Trigger slingshot snap-back animation
        slingshot.trigger_snap((bird.x, bird.y))

        self.state = "FLIGHT"

    def _update_flight(self, g: dict):
        bird = self.current_bird
        if bird is None:
            self._next_bird()
            return

        bird.update()

        # ── Multi-block collision ─────────────────────────────────────────
        for blk in self.blocks:
            if not blk.active:
                continue
            if id(blk) in bird._hit_blocks:
                continue
            if not bird_hits_block(bird, blk):
                continue

            # Hit!
            bird._hit_blocks.add(id(blk))
            blk.apply_impulse(bird.vx, bird.vy, bird.mass)
            bird.impact_timer = 15

            # Slow bird based on block toughness (ice easy, stone hard)
            block_resist = getattr(blk, "density", 1.0) * 0.3
            factor = max(0.1, 1.0 - block_resist)
            bird.vx *= factor
            bird.vy *= factor

            # If bird is almost stopped after hit, start grounding
            if magnitude(bird.vx, bird.vy) < BIRD_STOP_SPEED and not bird.grounded:
                bird.grounded = True
                bird.linger_timer = BIRD_LINGER // 2

        # ── Bird deactivated or off-screen → next bird ────────────────────
        if (not bird.active
                or bird.y > self.H + 50
                or bird.x > self.W + 100
                or bird.x < -100):
            self._next_bird()

    def _next_bird(self):
        self.current_bird = None
        if self.bird_queue:
            self.state = "SELECTION"
        else:
            self.state = "DONE"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _launch_velocity(self) -> tuple[float, float]:
        """Compute launch vx/vy from pull position vs slingshot anchor."""
        bird = self.current_bird
        dx = SLING_X - bird.x
        dy = SLING_Y - bird.y
        return dx * POWER_FACTOR, dy * POWER_FACTOR
