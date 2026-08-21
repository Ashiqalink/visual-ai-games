"""
game.py — Game state machine.

States
------
SELECTION  : carousel, choose bird via pinch OR z-push click
READY      : bird grabbed, anchor not yet locked — move the fist anywhere
             comfortable and hold still to lock the aim origin
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

import math
import random
import time

import cv2
import numpy as np
import slingshot
import ui
from bird import Bird
from block import Block, Target
from config import (
    AIM_EMA_JITTER_HI,
    AIM_EMA_JITTER_LO,
    AIM_EMA_MAX,
    AIM_EMA_MIN,
    AIM_PULL_GAIN,
    BH,
    BIRD_DAMAGE,
    BIRD_LINGER,
    BIRD_ORDER,
    BIRD_STOP_SPEED,
    BLOCK_RESIST_MAX,
    BW,
    CAROUSEL_SELECTION_MAX_Y,
    CAROUSEL_SPACING,
    DEBRIS_CARRY_FACTOR,
    DEBRIS_HEALTH,
    DEBRIS_LIFESPAN,
    DEBRIS_VEL_SPREAD,
    DEBRIS_VY_KICK,
    EDGE_MARGIN,
    FLOOR_Y,
    GRAB_RELEASE_FRAMES,
    GRIP_RELEASE_FRAMES,
    GRIP_RELEASE_GATE,
    GRIP_RELEASE_OPENNESS,
    GRIP_RELEASE_RATE,
    LEVEL_X_OFF,
    LOST_HAND_FIRE_FRAMES,
    MAX_DEBRIS,
    MAX_PULL,
    MIN_FIRE_PULL,
    PLATFORM_H,
    PLATFORM_HEALTH,
    PLATFORM_W,
    POWER_FACTOR,
    PTS_BLOCK,
    PTS_DEBRIS,
    PTS_TARGET,
    PUNCH_THROUGH_RETAIN,
    READY_LOST_CANCEL_FRAMES,
    READY_MAX_FRAMES,
    READY_SETTLE_FRAMES,
    READY_SETTLE_RADIUS,
    SCORE_POPUP_LIFETIME,
    SCORE_POPUP_RISE,
    SEL_DEBOUNCE_FRAMES,
    TH,
)
from physics import (
    bird_hits_block,
    distance,
    magnitude,
    resolve_bird_block_collision,
    resolve_block_collision,
)
from slingshot import SLING_X, SLING_Y
from tracker import NullTracker

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

def _level_easy() -> list[Block]:
    """Simple wooden tower — good for learning controls."""
    blocks: list[Block] = []

    # Three pillars
    for gx in [900, 960, 1020]:
        blocks.append(Block(gx, FLOOR_Y - BH, BW, BH, "wood"))

    # Plank on top
    blocks.append(Block(900, FLOOR_Y - BH - TH, 150, TH, "wood"))

    # Single block on top
    blocks.append(Block(950, FLOOR_Y - BH * 2 - TH, BW, BH, "wood"))

    # Target
    blocks.append(Target(950, FLOOR_Y - BH * 2 - TH - 30, radius=20))

    return blocks


def _level_medium() -> list[Block]:
    """The Ice Barracks — a two-storey hut standing on ice legs.

    The point of the layout is that nothing here has to be broken one block at
    a time. Every load-bearing member is ice (8 hp, shatters on a graze); the
    wood planks are floors and ceilings, and the two ground-floor pigs sit
    *under* the upper floor. Take out either pair of ice legs and the storey
    above drops on them.

    The stone slab on the left face is the only hard part: it shields a
    straight-line shot at the legs, so the fun opening is a lofted arc onto the
    roof or a low skimmer under the overhang.
    """
    blocks: list[Block] = []

    # ── Ground floor: ice legs, stone shield on the exposed left face ──────
    blocks.append(Block(820, FLOOR_Y - BH,   BW, BH, "stone"))  # shield
    blocks.append(Block(875, FLOOR_Y - BH,   BW, BH, "ice"))
    blocks.append(Block(970, FLOOR_Y - BH,   BW, BH, "ice"))
    blocks.append(Block(1015, FLOOR_Y - BH,  BW, BH, "wood"))

    # First floor
    blocks.append(Block(815, FLOOR_Y - BH - TH, 225, TH, "wood"))

    # Two pigs sheltering on the ground floor, in the bay between the ice legs
    blocks.append(Target(903, FLOOR_Y - 2 * 15,  radius=15))
    blocks.append(Target(937, FLOOR_Y - 2 * 15,  radius=15))

    # ── Upper storey: wood walls, ice crossbeam in the middle ─────────────
    y2 = FLOOR_Y - BH - TH
    blocks.append(Block(825, y2 - BH,  BW, BH, "wood"))
    blocks.append(Block(915, y2 - BH,  BW, BH, "ice"))
    blocks.append(Block(1010, y2 - BH, BW, BH, "wood"))

    # Roof
    blocks.append(Block(815, y2 - BH - TH, 225, TH, "wood"))

    # ── Crown: a pig on the roof, flanked by ice so a hit skitters ─────────
    y3 = y2 - BH - TH
    blocks.append(Block(860, y3 - BH, BW, BH, "ice"))
    blocks.append(Block(970, y3 - BH, BW, BH, "ice"))
    blocks.append(Target(905, y3 - 2 * 18, radius=18))

    return blocks


def _level_hard() -> list[Block]:
    """The Keep — a stone shell on a structure that is anything but stone.

    The old version was three solid stone columns: ten 44 hp blocks with no
    weak point and nothing to collapse, which is why it read as impossible.
    Stone is now armour, not structure. The front battlement soaks the obvious
    flat shot, the great hall's roof is carried on ice pillars, and the tower
    stands on wood. Every pig is under or on top of something that can come
    down, so the level is won by collapsing it, not by chewing through it.

    Three ways in, roughly in order of difficulty:
      · loft over the battlement onto the hall roof and crush the pigs inside
      · a fast flat shot that clips the ice pillars through the front gap
      · knock the tower over sideways into the hall
    """
    blocks: list[Block] = []

    # ── Front battlement (x = 770): stone, deliberately unbreakable-ish ────
    bx = 770
    blocks.append(Block(bx, FLOOR_Y - BH,     BW, BH, "stone"))
    blocks.append(Block(bx, FLOOR_Y - 2 * BH, BW, BH, "stone"))
    blocks.append(Block(bx, FLOOR_Y - 3 * BH, BW, BH, "wood"))   # softer merlon

    # ── Great hall (x = 850..1030) ────────────────────────────────────────
    # Ice pillars carry the whole hall. This is the level's weak point.
    for px in (855, 935, 1010):
        blocks.append(Block(px, FLOOR_Y - BH, BW, BH, "ice"))

    hall_floor_y = FLOOR_Y - BH - TH
    blocks.append(Block(845, hall_floor_y, 190, TH, "wood"))

    # Two pigs on the hall floor, boxed in by stone side walls
    blocks.append(Target(895, hall_floor_y - 2 * 16, radius=16))
    blocks.append(Target(965, hall_floor_y - 2 * 16, radius=16))

    blocks.append(Block(845, hall_floor_y - BH,  BW, BH, "stone"))
    blocks.append(Block(1015, hall_floor_y - BH, BW, BH, "stone"))

    # Hall roof — heavy stone slab, so dropping it is lethal to what is under it
    hall_roof_y = hall_floor_y - BH - TH
    blocks.append(Block(845, hall_roof_y, 190, TH, "stone"))

    # King pig on the roof, with an ice screen in front of him
    blocks.append(Block(880, hall_roof_y - BH, BW, BH, "ice"))
    blocks.append(Target(925, hall_roof_y - 2 * 20, radius=20))

    # ── Rear tower (x = 1080): wood legs, stone cap, pig on top ───────────
    tx = 1080
    blocks.append(Block(tx, FLOOR_Y - BH,     BW, BH, "wood"))
    blocks.append(Block(tx, FLOOR_Y - 2 * BH, BW, BH, "wood"))
    blocks.append(Block(tx, FLOOR_Y - 3 * BH, BW, BH, "stone"))
    blocks.append(Target(tx - 5, FLOOR_Y - 3 * BH - 2 * 15, radius=15))

    return blocks


_LEVELS = [_level_easy, _level_medium, _level_hard]


def _build_level(level_idx: int) -> list[Block]:
    idx = max(0, min(level_idx, len(_LEVELS) - 1))
    blocks = _LEVELS[idx]()

    plat_h = PLATFORM_H

    # Compute how wide the level's footprint is before the X offset is applied,
    # then size the platform to cover it with a small margin on each side.
    if blocks:
        min_x = min(b.rect[0] for b in blocks)
        max_x = max(b.rect[0] + b.rect[2] for b in blocks)
        plat_w = max(PLATFORM_W, int(max_x - min_x) + 40)
        plat_x = min_x - 20
    else:
        plat_w = PLATFORM_W
        plat_x = 920 - plat_w / 2

    plat_y = FLOOR_Y - plat_h

    # Move blocks horizontally by LEVEL_X_OFF, and shift up to rest on the platform
    for b in blocks:
        b.rect[0] += LEVEL_X_OFF
        b.rect[1] -= plat_h

    # Create a static platform at floor level, also shifted by LEVEL_X_OFF
    platform = Block(plat_x + LEVEL_X_OFF, plat_y, plat_w, plat_h, "wood")
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
        # Offline run tracking, off unless main.py hands over a real one. It is
        # set here rather than in reset() so a restart does not throw away the
        # tracker main.py is holding; main.py rotates it itself when a run ends.
        self.tracker = NullTracker()
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
        # Grab hysteresis: consecutive frames the hand has not read as a fist
        # (fallback path only — see _still_gripping)
        self._release_frames: int = 0
        # Grip release: consecutive frames the hand has read as opening, and the
        # previous frame's openness so a snap release can be caught by its rate.
        self._open_frames: int = 0
        self._prev_openness: float = 0.0
        # Consecutive frames the tracker has seen no hand at all while aiming
        self._lost_frames: int = 0
        # 3-Finger Pinch Aiming state flags
        self._is_aiming: bool = False
        self._was_unpinched_after_armed: bool = False
        # READY phase: hand travels freely while the anchor stays unlocked
        self._ready_frames: int = 0                    # frames spent in READY
        self._settle_frames: int = 0                   # consecutive near-still frames
        self._settle_ref_x: float = float(SLING_X)     # centre of the stillness test
        self._settle_ref_y: float = float(SLING_Y)
        self._settle_pos: tuple[float, float] | None = None   # where to draw the ring
        self._settle_forced: bool = False              # locked by timeout, not stillness
        self._ready_t0: float = 0.0                    # when READY began (tracker)
        self._settle_ms: float = 0.0                   # how long the lock took
        self._shake_frames: int = 0
        self._shake_intensity: int = 0
        self._final_stars: int = 0
        self._final_bonus: int = 0
        self.rot_angle_3d: float = 0.0
        # Scoring
        self.score_popups: list[ScorePopup] = []

    def update_game_state(self, gesture: dict, key: int):
        """
        Called once per frame to handle inputs, camera telemetrics, and aiming.
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

        # ── State machine (Input-driven) ──────────────────────────────────
        if self.state == "SELECTION":
            self._update_selection(gesture)
        elif self.state == "READY":
            self._update_ready(gesture)
        elif self.state == "ARMED":
            self._update_armed(gesture)

    def update_physics(self):
        """
        Called at a fixed timestep to update dynamic bodies, animations, and collisions.
        """
        # Increment 3D rotation angle for visual_ai elements
        self.rot_angle_3d = (self.rot_angle_3d + 2.5) % 360.0

        # ── Slingshot animation tick ──────────────────────────────────────
        slingshot.tick()

        if self.state == "FLIGHT":
            self._update_flight(self._last_gesture)

        # ── Always update blocks ──────────────────────────────────────────
        new_blocks: list[Block] = []
        for b in self.blocks:
            was_active = b.active
            b.update()
            if was_active and not b.active:
                # Block just broke — score + debris
                is_target = getattr(b, "is_target", False) or isinstance(b, Target)
                if is_target:
                    pts = PTS_TARGET
                else:
                    is_big = b.rect[2] > 20 and b.rect[3] > 20
                    pts = PTS_BLOCK if is_big else PTS_DEBRIS
                self.score += pts
                self.tracker.hit(pts)
                self.score_popups.append(ScorePopup(b.cx, b.cy, pts))

                if not is_target and is_big:
                    hw = b.rect[2] / 2
                    hh = b.rect[3] / 2
                    for i in range(2):
                        for j in range(2):
                            debris = Block(b.rect[0] + i * hw,
                                           b.rect[1] + j * hh,
                                           hw, hh,
                                           material=b.material)
                            debris.vx = ((random.random() * 2.0 - 1.0) * DEBRIS_VEL_SPREAD
                                         + b.vx * DEBRIS_CARRY_FACTOR)
                            debris.vy = ((random.random() * 2.0 - 1.0) * DEBRIS_VEL_SPREAD
                                         + b.vy * DEBRIS_CARRY_FACTOR - DEBRIS_VY_KICK)
                            debris.health = DEBRIS_HEALTH
                            debris.is_debris = True
                            debris.lifespan = DEBRIS_LIFESPAN
                            new_blocks.append(debris)

        if new_blocks:
            self.blocks.extend(new_blocks)

        # ── Enforce Debris Cap ────────────────────────────────────────────
        debris_blocks = [b for b in self.blocks if getattr(b, "is_debris", False) and b.active]
        if len(debris_blocks) > MAX_DEBRIS:
            # Sort by lifespan (lowest lifespan dies first) to preserve newer debris
            debris_blocks.sort(key=lambda b: b.lifespan)
            to_remove = len(debris_blocks) - MAX_DEBRIS
            for b in debris_blocks[:to_remove]:
                b.active = False

        # ── GC: Remove dead blocks ────────────────────────────────────────
        self.blocks = [b for b in self.blocks if b.active]

        # ── Resolve block-block collisions ────────────────────────────────
        for i in range(len(self.blocks)):
            for j in range(i + 1, len(self.blocks)):
                b1 = self.blocks[i]
                b2 = self.blocks[j]
                if b1.active and b2.active:
                    resolve_block_collision(b1, b2)

        # ── Update score popups ───────────────────────────────────────────
        self.score_popups = [p for p in self.score_popups if p.update()]

        # ── Check win condition ───────────────────────────────────────────
        if self.state in ("SELECTION", "READY", "ARMED", "FLIGHT"):
            # If no target pigs remain
            has_targets = any(getattr(b, "is_target", False) for b in self.blocks)
            if not has_targets:
                # Calculate bonus and stars
                from config import PTS_UNUSED_BIRD, STAR_1_SCORE, STAR_2_SCORE, STAR_3_SCORE
                self._final_bonus = len(self.bird_queue) * PTS_UNUSED_BIRD
                if (self.current_bird and self.current_bird.active
                        and not self.current_bird.launched):
                    self._final_bonus += PTS_UNUSED_BIRD
                self.score += self._final_bonus

                if self.score >= STAR_3_SCORE:
                    self._final_stars = 3
                elif self.score >= STAR_2_SCORE:
                    self._final_stars = 2
                elif self.score >= STAR_1_SCORE:
                    self._final_stars = 1
                else:
                    self._final_stars = 0

                self.state = "WIN"

    def draw(self, frame: np.ndarray):
        """Render everything onto frame (modifies in-place)."""

        # Paint the scene before any world objects.  The desktop entry point
        # intentionally starts from a plain canvas so camera pixels never leak
        # into the game; without this pass the slingshot and level geometry
        # appear to float on a nearly black background.
        ui.draw_ground(frame, FLOOR_Y)

        shake_x, shake_y = 0, 0
        if self._shake_frames > 0:
            shake_x = random.randint(-self._shake_intensity, self._shake_intensity)
            shake_y = random.randint(-self._shake_intensity, self._shake_intensity)
            self._shake_frames -= 1

        # Blocks
        for b in self.blocks:
            if shake_x or shake_y:
                b.rect[0] += shake_x
                b.rect[1] += shake_y
            b.draw(frame)
            if shake_x or shake_y:
                b.rect[0] -= shake_x
                b.rect[1] -= shake_y

        if self.state == "SELECTION":
            slingshot.draw(frame, bird_pos=None)
            ui.draw_carousel(frame, self.bird_queue, self.selected_idx,
                             rot_angle_3d=self.rot_angle_3d)

        elif self.state == "READY":
            # Bird waits on the sling with a slack band — nothing is pulled yet,
            # so it draws exactly like the idle slingshot with the bird added.
            bird = self.current_bird
            bp = (bird.x + shake_x, bird.y + shake_y)
            slingshot.draw_back(frame, bird_pos=bp, pull_dist=0.0)
            bird.draw(frame)
            slingshot.draw_front(frame, bird_pos=bp, pull_dist=0.0)

            if self._settle_pos is not None:
                progress = min(1.0, self._settle_frames / READY_SETTLE_FRAMES)
                ui.draw_settle_ring(frame, self._settle_pos[0], self._settle_pos[1],
                                    progress=progress,
                                    radius=READY_SETTLE_RADIUS,
                                    timeout_progress=min(
                                        1.0, self._ready_frames / READY_MAX_FRAMES))

        elif self.state == "ARMED":
            bird = self.current_bird
            pull_d = distance((bird.x, bird.y), (SLING_X, SLING_Y))
            bp = (bird.x + shake_x, bird.y + shake_y)

            # Depth-layered: back elastic → bird → front elastic + structure
            slingshot.draw_back(frame, bird_pos=bp, pull_dist=pull_d)
            if shake_x or shake_y:
                bird.x += shake_x
                bird.y += shake_y
                bird.draw(frame)
                bird.x -= shake_x
                bird.y -= shake_y
            else:
                bird.draw(frame)
            slingshot.draw_front(frame, bird_pos=bp, pull_dist=pull_d)

            # Trajectory preview (only drawn while actively aiming with pinch)
            if self._is_aiming and pull_d > 5:
                vx, vy = self._launch_velocity()
                ui.draw_trajectory(frame, bird.x, bird.y, vx, vy, mass=bird.mass)

        elif self.state == "FLIGHT":
            slingshot.draw(frame, bird_pos=None)
            if self.current_bird and self.current_bird.active:
                if shake_x or shake_y:
                    self.current_bird.x += shake_x
                    self.current_bird.y += shake_y
                    self.current_bird.draw(frame)
                    self.current_bird.x -= shake_x
                    self.current_bird.y -= shake_y
                else:
                    self.current_bird.draw(frame)

        elif self.state in ("DONE", "WIN"):
            slingshot.draw(frame, bird_pos=None)
            ui.draw_done_overlay(frame, score=self.score, won=(self.state == "WIN"),
                                 stars=self._final_stars, bonus=self._final_bonus,
                                 rot_angle_3d=self.rot_angle_3d)

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

        # Carousel area parameters (top-center panel: y <= CAROUSEL_SELECTION_MAX_Y)
        spacing = CAROUSEL_SPACING
        panel_w = spacing * total + 80
        panel_x = self.W // 2 - panel_w // 2

        # Selecting bird can only be chosen once cursor goes into the carousel area
        in_selection_area = (iy <= CAROUSEL_SELECTION_MAX_Y)

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

        # Grab trigger: a closed fist inside the selection area picks the bird up.
        click_trigger = g.get("is_fist", False) and in_selection_area

        if click_trigger:
            self._last_click_mode = "FIST GRAB"
            self._last_click_fired = True
            if total > 0:
                kind = self.bird_queue[self.selected_idx]
                self.current_bird = Bird(kind, SLING_X, SLING_Y)
                self.current_bird.on_slingshot = True
                self.current_bird.x = float(SLING_X)
                self.current_bird.y = float(SLING_Y)
                self._release_frames = 0           # fresh grab hysteresis
                self._lost_frames = 0              # fresh dropout counter
                # Seed the grip signal from this frame so the first openness
                # reading is not compared against a stale one and read as a
                # release the instant the bird is picked up.
                self._open_frames = 0
                self._prev_openness = float(g.get("grip_openness", 0.0))
                # Grab-and-hold, but not grab-and-aim. Selection only happens up
                # in the carousel strip, so anchoring the aim here would pin
                # every pull to a raised arm. READY lets the fist carry the bird
                # down to wherever the player actually wants to aim from; the
                # anchor locks when they stop moving.
                self._is_aiming = False
                self._was_unpinched_after_armed = True
                self._enter_ready(g)
        else:
            self._last_click_fired = False

    def _enter_ready(self, g: dict):
        """SELECTION → READY. The bird is held but the aim origin is not fixed."""
        if g.get("hand_visible", False):
            hx, hy = g.get("pinch_pos", g["index_pos"])
        else:
            hx, hy = float(SLING_X), float(SLING_Y)
        self._ready_frames  = 0
        self._settle_frames = 0
        self._settle_ref_x  = float(hx)
        self._settle_ref_y  = float(hy)
        self._settle_pos    = (float(hx), float(hy))
        self._settle_forced = False
        self._ready_t0      = time.perf_counter()
        self.state = "READY"

    def _lock_anchor(self, x: float, y: float, forced: bool = False):
        """READY → ARMED. Whatever position the hand settled in becomes the
        origin every later pull is measured from."""
        self._aim_anchor_x = float(x)
        self._aim_anchor_y = float(y)
        self._smoothed_ix  = float(x)
        self._smoothed_iy  = float(y)
        self._armed_inside = False         # reset edge-fire guard
        self._armed_timer  = 0             # immediate readiness
        self._release_frames = 0
        self._open_frames    = 0
        self._lost_frames    = 0
        self._is_aiming      = True
        self._settle_forced  = forced
        self._settle_ms      = (time.perf_counter() - self._ready_t0) * 1000.0
        self.state = "ARMED"

    def _still_gripping(self, g: dict) -> bool:
        """
        Is the hand still holding the bird?

        Read from `grip_openness`, the engine's continuous undebounced grip
        signal, so the hold ends within a frame or two of the fingers actually
        moving. The old test — `is_fist` plus GRAB_RELEASE_FRAMES of hysteresis
        — took around a third of a second to notice an open hand, and the aim
        kept tracking the fingertip centroid throughout, which is what dragged
        the shot off target. See config.py for the thresholds.

        A frame with no hand is not evidence either way: the grip counters hold
        where they are, so a tracking dropout freezes the pull rather than
        releasing it. Callers handle a hand that stays gone via _lost_frames.
        """
        if not g.get("hand_visible", False):
            return True

        openness = g.get("grip_openness")
        if openness is None:
            # Engine predates the continuous signal — fall back to the sign.
            if g.get("is_fist", False):
                self._release_frames = 0
            else:
                self._release_frames += 1
            return self._release_frames < GRAB_RELEASE_FRAMES

        opening_rate = openness - self._prev_openness
        self._prev_openness = openness

        if openness >= GRIP_RELEASE_OPENNESS:
            self._open_frames += 1
        else:
            self._open_frames = 0

        # A hand thrown open clears both tests on the same frame it moves.
        if openness >= GRIP_RELEASE_GATE and opening_rate >= GRIP_RELEASE_RATE:
            return False

        return self._open_frames < GRIP_RELEASE_FRAMES

    def _update_ready(self, g: dict):
        """
        Settle phase between grabbing a bird and aiming it.

        The fist keeps holding the bird, but hand movement does *not* pull the
        band — the player is free to bring their arm down from the carousel to
        wherever they actually want to shoot from. The anchor locks the moment
        the hand holds still (READY_SETTLE_FRAMES frames inside a small radius),
        so any resting position works: high, low, seated, off to one side.

        Opening the hand here puts the bird back rather than firing it: there is
        no pull to launch, and an open hand in READY is the natural "never mind".
        """
        bird = self.current_bird
        if bird is None:
            self.state = "SELECTION"
            return

        # Nothing is pulled yet — the bird rides the fork.
        bird.x = float(SLING_X)
        bird.y = float(SLING_Y)

        self._ready_frames += 1

        if not self._still_gripping(g):
            # Let go before settling → bird goes back on the shelf.
            self.current_bird = None
            self._settle_pos = None
            self.state = "SELECTION"
            return

        if not g.get("hand_visible", False):
            self._lost_frames += 1
            if self._lost_frames >= READY_LOST_CANCEL_FRAMES:
                self.current_bird = None
                self._settle_pos = None
                self.state = "SELECTION"
            return
        self._lost_frames = 0

        raw_ix, raw_iy = g.get("pinch_pos", g["index_pos"])
        ix, iy = float(raw_ix), float(raw_iy)
        self._settle_pos = (ix, iy)

        # Stillness test: stay within READY_SETTLE_RADIUS of the reference point
        # and the streak grows; break out of it and the reference moves to the
        # new position. A slow deliberate drift still counts as settling, which
        # is what a player easing their arm into place actually looks like.
        if distance((ix, iy), (self._settle_ref_x, self._settle_ref_y)) <= READY_SETTLE_RADIUS:
            self._settle_frames += 1
        else:
            self._settle_ref_x = ix
            self._settle_ref_y = iy
            self._settle_frames = 1

        if self._settle_frames >= READY_SETTLE_FRAMES:
            self._lock_anchor(ix, iy, forced=False)
        elif self._ready_frames >= READY_MAX_FRAMES:
            # Never strand the player in a phase they may not know they are in.
            self._lock_anchor(ix, iy, forced=True)

    def _update_armed(self, g: dict):
        """
        Fist-grab slingshot controls, entered from READY with the anchor already
        locked wherever the player settled:
        1. Moving the fist pulls the slingshot rubber band, measured from that
           anchor rather than from the carousel where the bird was picked up.
        2. Opening the hand launches the bird (if pull >= MIN_FIRE_PULL); a
           shorter pull returns the bird to the carousel, and the next grab runs
           a fresh READY settle so the anchor can be repositioned.
        3. Losing the hand mid-pull holds the aim for a few frames, then fires —
           a release close to the camera drops off the tracker before its open
           frames can be counted.
        """
        bird = self.current_bird
        if bird is None:
            return

        margin = EDGE_MARGIN      # px from edge — larger = less accidental edge-fires

        if g.get("click_just_fired", False):
            self._last_click_fired = True
        if self._armed_timer > 0:
            self._armed_timer -= 1

        # Closed fist = holding the bird; the hand starting to open fires it.
        is_grabbing = self._still_gripping(g)

        # A hand that stays gone is a different story. Releasing near the camera
        # pushes the fingers out of MediaPipe's reach before the open frames can
        # be counted, so waiting for them would strand the bird on the band.
        # Fire the pull we already have instead.
        if g.get("hand_visible", False):
            self._lost_frames = 0
        elif self._is_aiming:
            self._lost_frames += 1
            if (self._lost_frames >= LOST_HAND_FIRE_FRAMES
                    and self._pull_distance() >= MIN_FIRE_PULL):
                self._fire_bird(bird, cause="lost")
                self._is_aiming = False
                return

        # 1. Handle OPEN-HAND state
        if not is_grabbing:
            # Opening the hand always ends the hold. A pull worth firing becomes
            # a shot — this beats any change-of-mind reading on purpose, since a
            # pull that ends high on the screen is still a shot.
            if self._pull_distance() >= MIN_FIRE_PULL:
                self._fire_bird(bird, cause="open")
                self._is_aiming = False
                return

            # Too short to fire: the player let go without committing, so put
            # the bird back and let them choose again. There is no idle-in-ARMED
            # state any more — re-grabbing runs the READY settle from scratch,
            # which is also how the aim origin gets repositioned.
            self._is_aiming = False
            self.current_bird = None
            bird.x = float(SLING_X)
            bird.y = float(SLING_Y)
            self.state = "SELECTION"
            return

        # 2. Handle CLOSED-FIST state (actively aiming)
        if g.get("hand_visible", False):
            raw_ix, raw_iy = g.get("pinch_pos", g["index_pos"])
            ix, iy = float(raw_ix), float(raw_iy)
        else:
            ix, iy = self._smoothed_ix, self._smoothed_iy

        # The anchor is set once, by READY. Re-anchoring here would silently undo
        # the position the player just settled into.

        # Mark hand as "inside" once it's away from all edges
        if (margin < ix < self.W - margin and margin < iy < self.H - margin):
            self._armed_inside = True

        at_edge = (ix < margin or ix > self.W - margin
                   or iy < margin or iy > self.H - margin)

        # ── Adaptive EMA smooth ──
        d_dist = math.sqrt((ix - self._smoothed_ix) ** 2 + (iy - self._smoothed_iy) ** 2)
        scale = max(0.0, min(
            1.0, (d_dist - AIM_EMA_JITTER_LO) / (AIM_EMA_JITTER_HI - AIM_EMA_JITTER_LO)))
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
                self._fire_bird(bird, cause="edge")
                self._is_aiming = False
                return

    def _fire_bird(self, bird: Bird, cause: str = "open"):
        """Transition from ARMED → FLIGHT.

        `cause` says which of the three exits fired: the hand opening ('open'),
        the hand dropping off the tracker mid-pull ('lost'), or the pull
        leaving the frame ('edge'). It is passed straight to the tracker and
        changes nothing about the shot — the three are separated because only
        the first is a release the player timed.
        """
        vx, vy = self._launch_velocity()
        bird.vx = vx
        bird.vy = vy
        bird.launched = True

        self.tracker.shot(
            cause=cause,
            bird_kind=bird.kind,
            anchor=(self._aim_anchor_x, self._aim_anchor_y),
            hand=(self._smoothed_ix, self._smoothed_iy),
            band=(bird.x, bird.y),
            velocity=(vx, vy),
            pull_dist=self._pull_distance(),
            gain=self._AIM_PULL_GAIN,
            ready_ms=self._settle_ms,
            settle_forced=self._settle_forced,
        )

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
            if not bird_hits_block(bird, blk):
                continue

            if id(blk) not in bird._hit_blocks:
                # Hit!
                bird._hit_blocks.add(id(blk))
                shattered = blk.apply_impulse(bird.vx, bird.vy, bird.mass,
                                              BIRD_DAMAGE.get(bird.kind, 1.0))
                bird.impact_timer = 15

                # A block that survived the hit stops the bird in proportion to
                # how heavy it is; one that shattered barely slows it, so a
                # powerful shot carries through a stack instead of dying on its
                # front face.
                if shattered:
                    factor = PUNCH_THROUGH_RETAIN
                else:
                    block_resist = min(BLOCK_RESIST_MAX,
                                       getattr(blk, "density", 1.0) * 0.3)
                    factor = max(0.1, 1.0 - block_resist)
                bird.vx *= factor
                bird.vy *= factor

                # Screen shake on hard impact
                impact_speed = magnitude(bird.vx, bird.vy)
                if impact_speed > 3.0:
                    self._shake_frames = 10
                    self._shake_intensity = int(impact_speed)

                # If bird is almost stopped after hit, start grounding
                if magnitude(bird.vx, bird.vy) < BIRD_STOP_SPEED and not bird.grounded:
                    bird.grounded = True
                    bird.linger_timer = BIRD_LINGER // 2

            # A destroyed block offers no surface — the bird punches through.
            if blk.health <= 0:
                continue

            # Push the bird back out and bounce it, same as the floor does,
            # so it can never come to rest inside a surviving block.
            if resolve_bird_block_collision(bird, blk) and not bird.grounded:
                bird.grounded = True
                bird.linger_timer = BIRD_LINGER

        # ── Bird deactivated or off-screen → next bird ────────────────────
        if (not bird.active
                or bird.y > self.H + 50
                or bird.x > self.W + 100
                or bird.x < -100):
            self._next_bird()

    def _next_bird(self):
        # The flight is over, so whatever the shot scored is now final.
        self.tracker.flight_ended(self.score)
        self.current_bird = None
        if self.bird_queue:
            self.state = "SELECTION"
        else:
            self.state = "DONE"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pull_distance(self) -> float:
        """How far the smoothed hand has travelled from the aim anchor, in
        on-screen pull pixels (i.e. after the gain multiplier)."""
        rel_dx = (self._smoothed_ix - self._aim_anchor_x) * self._AIM_PULL_GAIN
        rel_dy = (self._smoothed_iy - self._aim_anchor_y) * self._AIM_PULL_GAIN
        return math.sqrt(rel_dx * rel_dx + rel_dy * rel_dy)

    def _launch_velocity(self) -> tuple[float, float]:
        """Launch vx/vy from pull vs anchor, power scaled for heavier birds."""
        bird = self.current_bird
        dx = SLING_X - bird.x
        dy = SLING_Y - bird.y
        # Larger/heavier birds (e.g. Bomb) need extra spring tension power factor
        size_mass_power = POWER_FACTOR * (bird.mass ** 0.85) * ((bird.radius / 26.0) ** 0.5)
        return (dx * size_mass_power) / bird.mass, (dy * size_mass_power) / bird.mass
