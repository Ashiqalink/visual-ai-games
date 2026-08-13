"""
config.py -- Central configuration for Angry Birds OpenCV.

All magic numbers live here.  Edit this file to tune gameplay, visuals,
and gesture sensitivity without touching any other source file.
"""

#  WINDOW / CANVAS
# =============================================================================
FRAME_W = 1280          # canvas width  (px)
FRAME_H = 720           # canvas height (px)
BG_COLOR = (12, 18, 28) # deep navy-black background (BGR)

# Camera preview box (top-right corner)
CAM_W            = 240          # preview width  (px)
CAM_H            = 160          # preview height (px)
CAM_MARGIN       = 12           # gap from window edges (px)
CAM_BORDER       = 2            # border thickness (px)
CAM_BORDER_COLOR = (80, 200, 255)  # cyan border (BGR)

#  PHYSICS
# =============================================================================
GRAVITY        = 0.072 # downward acceleration (px / frame^2) - scaled for cinematic slower motion
ENGINE_GRAVITY = 500.0 # gravity for the visual_ai C++ engine core object
ENGINE_RADIUS  = 28.0  # radius for the visual_ai C++ engine core object
FLOOR_Y        = 650    # ground line (pixel row) -- used by level builders
RESTITUTION    = 0.35   # block / floor bounce dampening
POWER_FACTOR   = 0.06   # pull-distance to launch speed ratio - scaled for slower flight
MAX_PULL       = 170    # maximum slingshot pull radius (px)
AIR_DRAG       = 0.992  # per-frame velocity multiplier (< 1 = gentle deceleration)
BIRD_BOUNCE    = 0.35   # bird floor-bounce restitution coefficient
BIRD_LINGER    = 90     # frames bird stays active after first ground/block impact
MIN_DAMAGE_VEL = 1.08   # minimum impact speed to deal block-block damage
DAMAGE_FACTOR  = 4.17   # scales bird speed into block health reduction (compensated for slower speed)
BLOCK_HEALTH   = 30     # default block health (overridden per material in block.py)

# Block-block collision dampening (used inside resolve_block_collision)
BLOCK_COLL_X_DAMP  = 0.4   # X velocity dampening after elastic exchange
BLOCK_COLL_Y_DAMP  = 0.5   # Y velocity dampening after elastic exchange
BLOCK_COLL_FRIC    = 0.7   # friction on X velocity when resting
BLOCK_COLL_Y_STACK = 0.1   # vy dampening when a block is resting on another

#  GESTURE / INPUT
# =============================================================================
SMOOTH_ALPHA         = 0.20  # VisionPipeline EMA baseline (higher = faster / more jitter)
PINCH_THRESHOLD      = 30    # thumb-to-index distance for pinch click (px)
THREE_FINGER_PINCH_RADIUS = 45.0 # base centroid distance (px) for 3-finger pinch trigger
Z_CLICK_THRESHOLD_M  = 0.012 # forward movement threshold (~0.5 inch in MediaPipe Z units)
Z_CLICK_XY_MAX_PX    = 30    # max X/Y drift allowed during a Z-push click

#  SLINGSHOT
# =============================================================================
SLING_X = 200           # fork centre X (px)
SLING_Y = 550           # fork centre Y (px)
FORK_SPREAD_X  = 20     # horizontal offset from centre to each fork tip (px)
FORK_RISE_Y    = 25     # how far fork tips sit above fork centre (px)
HANDLE_DROP_Y  = 100    # how far handle base sits below fork centre (px)

SLING_WOOD_COLOR    = (30, 100, 160)  # BGR dark brown
SLING_WOOD_DARK     = (15,  60, 100)  # BGR darker accent
SLING_ELASTIC_NEAR  = (0, 140, 255)   # BGR colour when relaxed (orange)
SLING_ELASTIC_FAR   = (0,  40, 220)   # BGR colour when fully stretched (red)
SLING_SNAP_DURATION = 8               # frames for snap-back animation

#  BIRD VISUALS
# =============================================================================
TRAIL_LEN             = 12   # past positions stored for the flight trail
SPEED_LINE_THRESHOLD  = 3.0  # min speed (px/frame) to draw speed lines
IMPACT_POP_DURATION   = 15   # frames the impact-ring animation lasts

#  GAME LOGIC
# =============================================================================
PTS_BLOCK       = 500   # score for destroying a main block
PTS_DEBRIS      = 100   # score for destroying a debris piece
PTS_PIG         = 5000  # score for destroying a pig
PTS_UNUSED_BIRD = 10000 # bonus for remaining birds

PIG_HEALTH      = 40    # hit points for a pig

# Star rating thresholds (per level basis could be refined, using globals for simplicity)
STAR_1_SCORE    = 10000
STAR_2_SCORE    = 25000
STAR_3_SCORE    = 40000

SCORE_POPUP_LIFETIME = 40   # frames a floating score popup lives
SCORE_POPUP_RISE     = 1.5  # px / frame the popup floats upward

ARMED_GRACE_FRAMES = 60     # grace-period frames after equipping a bird (no firing)
MIN_FIRE_PULL      = 15     # min pull distance (px) before firing is allowed
EDGE_MARGIN        = 40     # px from window edge that triggers edge-exit fire

INPUT_MOVEMENT_MAGNIFICATION = 2.0 # input (x, y) movement magnification factor (e.g. 2.0x double movement)
AIM_PULL_GAIN       = 2.0   # physical hand move -> on-screen pull multiplier
AIM_EMA_HOLD        = 0.02  # EMA alpha while Z-pushing (freezes aim angle)
AIM_EMA_MIN         = 0.08  # EMA alpha for fine micro-adjustments
AIM_EMA_MAX         = 0.22  # EMA alpha for fast drags
AIM_EMA_JITTER_LO   = 2.0   # distance below which alpha stays at AIM_EMA_MIN
AIM_EMA_JITTER_HI   = 20.0  # distance above which alpha reaches AIM_EMA_MAX

SEL_DEBOUNCE_FRAMES  = 6     # stable frames before carousel commits selection
Z_PUSH_DETECT_THRESH = 0.008 # z_delta value above which we consider "Z pushing"

# Debris spawning
MAX_DEBRIS           = 20   # cap to preserve FPS
DEBRIS_LIFESPAN      = 90   # frames debris pieces live before fading (~3 s at 30 fps)
DEBRIS_VEL_SPREAD    = 10.0 # max random velocity spread for debris (px/frame)
DEBRIS_VY_KICK       = 2.0  # upward kick added to debris vy on spawn
DEBRIS_CARRY_FACTOR  = 0.5  # fraction of parent block velocity inherited by debris
DEBRIS_HEALTH        = 50   # health of freshly spawned debris pieces
DEBRIS_FADE_FRAMES   = 30   # frames before despawn during which debris fades out

# Platform (static level base blocks rest on)
PLATFORM_W      = 280
PLATFORM_H      = 12
PLATFORM_HEALTH = 100  # effectively indestructible

# Block impulse transfer ratios (applied when a bird hits a block)
BLOCK_VX_TRANSFER = 0.4   # fraction of bird vx added to block
BLOCK_VY_TRANSFER = 0.3   # fraction of bird vy added to block

# Bird stop thresholds
BIRD_STOP_SPEED = 1.2   # speed below which a hit bird starts grounding (scaled for slower flight)
BIRD_IDLE_SPEED = 0.3   # grounded speed below which the bird deactivates

#  BIRD PROPERTIES
# =============================================================================
RED   = "Red"
CHUCK = "Chuck"
BOMB  = "Bomb"
BLUES = "Blues"
WHITE = "White"

BIRD_ORDER = [RED, CHUCK, BOMB, BLUES, WHITE]

BIRD_RADII = {RED: 28, CHUCK: 26, BOMB: 30, BLUES: 22, WHITE: 26}

BIRD_MASSES = {RED: 1.0, CHUCK: 0.7, BOMB: 1.1, BLUES: 0.5, WHITE: 1.0}

BIRD_COLOURS = {
    RED:   (50,  50,  200),
    CHUCK: (0,  215,  255),
    BOMB:  (40,  40,   40),
    BLUES: (220, 80,   50),
    WHITE: (240, 240, 240),
}

import os
import sys

# Ensure visual_ai path is accessible for Material and ShaderType imports
_ENGINE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'visual ai game engine', 'src'))
if _ENGINE_SRC not in sys.path:
    sys.path.insert(0, _ENGINE_SRC)
_ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'visual ai game engine'))
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

try:
    from visual_ai import Material, ShaderType
except ImportError:
    from dataclasses import dataclass
    from enum import Enum
    class ShaderType(Enum):
        PBR_STANDARD = "PBR_Standard"
        TRANSPARENT = "Transparent"
        UNLIT = "Unlit"
    @dataclass
    class Material:
        name: str = "DefaultMaterial"
        shader_type: ShaderType = ShaderType.PBR_STANDARD
        base_color: tuple = (1.0, 1.0, 1.0, 1.0)
        normal_map: str = ""
        roughness: float = 0.5
        metallic: float = 0.0
        emission: tuple = (0.0, 0.0, 0.0)
        opacity: float = 1.0
        def to_shader_uniforms(self):
            return {
                "u_BaseColor": list(self.base_color),
                "u_Roughness": self.roughness,
                "u_Metallic": self.metallic,
                "u_Emission": list(self.emission),
                "u_Opacity": self.opacity,
                "u_ShaderType": self.shader_type.value,
            }

#  BLOCK MATERIALS
# =============================================================================
# BGR colour values; health/density per material, and visual_ai Material instance
BLOCK_MATERIALS = {
    "wood": {
        "health":  30,
        "density": 1.0,
        "restitution": 0.35,
        "friction": 0.5,
        "angular_drag": 0.94,
        "color":   (34, 100, 172),     # warm brown (BGR)
        "dark":    (20,  65, 120),
        "grain":   (55, 130, 195),
        "material": Material(
            name="Wood",
            shader_type=ShaderType.PBR_STANDARD,
            base_color=(172/255.0, 100/255.0, 34/255.0, 1.0),
            roughness=0.55,
            metallic=0.05,
        )
    },
    "stone": {
        "health":  80,
        "density": 1.8,
        "restitution": 0.05,
        "friction": 0.85,
        "angular_drag": 0.88,
        "color":   (130, 130, 135),    # neutral grey (BGR)
        "dark":    (90,   90,  95),
        "grain":   (155, 155, 160),
        "material": Material(
            name="Stone",
            shader_type=ShaderType.PBR_STANDARD,
            base_color=(135/255.0, 130/255.0, 130/255.0, 1.0),
            roughness=0.85,
            metallic=0.25,
        )
    },
    "ice": {
        "health":  15,
        "density": 0.6,
        "restitution": 0.15,
        "friction": 0.1,
        "angular_drag": 0.98,
        "color":   (240, 220, 180),    # light icy cyan (BGR)
        "dark":    (200, 180, 140),
        "grain":   (250, 235, 200),
        "material": Material(
            name="Ice",
            shader_type=ShaderType.TRANSPARENT,
            base_color=(180/255.0, 220/255.0, 240/255.0, 0.65),
            roughness=0.08,
            metallic=0.10,
            opacity=0.65,
        )
    },
}

#  CHARACTER 3D MATERIALS & MESH CONFIG
# =============================================================================
# Defined within angry-birds-opencv game configuration using visual_ai Material class
CHARACTER_3D_MATERIALS = {
    "red": Material(
        name="RedBirdMat",
        shader_type=ShaderType.PBR_STANDARD,
        base_color=(0.92, 0.15, 0.15, 1.0),
        roughness=0.35,
        metallic=0.05,
    ),
    "chuck": Material(
        name="ChuckBirdMat",
        shader_type=ShaderType.PBR_STANDARD,
        base_color=(0.98, 0.82, 0.12, 1.0),
        roughness=0.30,
        metallic=0.10,
    ),
    "bomb": Material(
        name="BombBirdMat",
        shader_type=ShaderType.PBR_STANDARD,
        base_color=(0.18, 0.18, 0.22, 1.0),
        roughness=0.40,
        metallic=0.30,
    ),
    "blues": Material(
        name="BluesBirdMat",
        shader_type=ShaderType.PBR_STANDARD,
        base_color=(0.20, 0.65, 0.95, 1.0),
        roughness=0.25,
        metallic=0.0,
    ),
    "white": Material(
        name="WhiteBirdMat",
        shader_type=ShaderType.PBR_STANDARD,
        base_color=(0.95, 0.95, 0.92, 1.0),
        roughness=0.20,
        metallic=0.0,
    ),
    "pig": Material(
        name="PigGreenMat",
        shader_type=ShaderType.PBR_STANDARD,
        base_color=(0.16, 0.78, 0.32, 1.0),
        roughness=0.35,
        metallic=0.0,
    ),
}


#  LIGHTING & SHADOW SETTINGS
# =============================================================================
LIGHTING_SETTINGS = {
    "ENABLED": True,              # Global light & shadow master toggle
    "LIGHT_ANGLE": 45.0,          # Directional light source angle in degrees (top-right light source)
    "LIGHT_INTENSITY": 0.85,      # Primary light intensity (0.0 to 1.0)
    "AMBIENT_INTENSITY": 0.45,    # Base ambient fill light (0.0 to 1.0)
    "SHADOWS_ENABLED": True,      # Dynamic shadow rendering toggle
    "SHADOW_OPACITY": 0.45,       # Ground shadow opacity (0.0 translucent to 1.0 dark)
    "SHADOW_OFFSET_X": 15.0,      # Dynamic X offset shift for projected shadow
    "SHADOW_OFFSET_Y": 8.0,       # Dynamic Y offset shift for projected shadow
    "SHADOW_SCALE_X": 1.25,       # Horizontal stretch factor for ground shadows
    "SHADOW_SCALE_Y": 0.35,       # Vertical flattening factor for ground shadows
    "SHADOW_COLOR": (10, 15, 20), # Deep ambient shadow color (BGR)
}

#  UI COLORS & SETTINGS
# =============================================================================
HUD_TEXT_COL = (240, 240, 240)
SHADOW_COL   = (30, 30, 30)
TRAJ_COL     = (220, 220, 220)
OVERLAY_ALPHA = 0.45

#  CAROUSEL UI
# =============================================================================
CAROUSEL_Y = 90
CAROUSEL_SPACING = 120
CAROUSEL_PANEL_H = 140
CAROUSEL_SELECTION_MAX_Y = 220

#  CURSOR SETTINGS
# =============================================================================
CURSOR_INDEX_COL = (0, 255, 200)
CURSOR_PINCH_COL = (0, 200, 255)
CURSOR_FIRE_COL  = (0, 80, 255)

#  LEVEL DIMENSIONS
# =============================================================================
BW, BH = 20, 40     # block width, height
TH = 14             # block thickness (planks)
LEVEL_X_OFF = -250  # Level X offset
LEVEL_NAMES = ["Easy", "Medium", "Hard"]
