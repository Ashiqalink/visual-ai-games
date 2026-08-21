"""
config.py -- Central configuration for Sling.

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
MIN_DAMAGE_VEL = 2.5   # minimum impact speed to deal block-block damage (raised from 1.08 to stop idle-settle damage)
# Damage budget. Peak launch speed is MAX_PULL * POWER_FACTOR ~= 10.2 px/frame,
# so a full-power Ruby hit lands mass*speed*DAMAGE_FACTOR ~= 82 damage: enough
# to punch a stone block (44) and carry on. At 4.17 it was 42 — one point short
# of a single stone block at maximum pull, which made the stone levels
# unwinnable rather than hard. Tune this together with the material healths
# below; they only mean anything relative to each other.
DAMAGE_FACTOR  = 8.0    # scales bird momentum into block health reduction
BLOCK_HEALTH   = 22     # default block health (overridden per material in block.py)

# A bird that destroys a block should carry on through the gap it just made
# instead of being slowed as if the block had held. Without this a shot died on
# the first block of any stack and structures could only be chipped, never
# broken open.
PUNCH_THROUGH_RETAIN = 0.8   # velocity kept after shattering a block
BLOCK_RESIST_MAX     = 0.62  # cap on the slowdown a *surviving* block applies

# Collapse damage. Structure that falls on itself is the main source of destruction
# once the bird has stopped, so this is what makes a good shot cascade.
# `BLOCK_IMPACT_DAMAGE` scales the excess impact speed; the mass ratio on top of
# it means a stone slab landing on ice does real harm while ice landing on stone
# mostly just breaks the ice.
BLOCK_IMPACT_DAMAGE  = 6.0   # per (px/frame) of impact speed above MIN_DAMAGE_VEL
BLOCK_IMPACT_MASS_K  = 2.0   # equal masses -> 1.0x, heavy-onto-light -> up to 2.0x

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
SLING_Y = 530           # fork centre Y (px)  — raised slightly for better composition
FORK_SPREAD_X  = 28     # horizontal offset from centre to each fork tip (px)
FORK_RISE_Y    = 35     # how far fork tips sit above fork centre (px)
HANDLE_DROP_Y  = 115    # how far handle base sits below fork centre (px)

SLING_WOOD_COLOR    = (42, 108, 175)  # BGR warm medium brown
SLING_WOOD_DARK     = (18,  55,  95)  # BGR dark bark shadow
SLING_ELASTIC_NEAR  = (0,  140, 255)  # BGR orange (relaxed)
SLING_ELASTIC_FAR   = (0,   40, 220)  # BGR deep red (fully stretched)
SLING_SNAP_DURATION = 10              # frames for snap-back animation

#  BIRD VISUALS
# =============================================================================
TRAIL_LEN             = 12   # past positions stored for the flight trail
SPEED_LINE_THRESHOLD  = 3.0  # min speed (px/frame) to draw speed lines
IMPACT_POP_DURATION   = 15   # frames the impact-ring animation lasts

#  GAME LOGIC
# =============================================================================
PTS_BLOCK       = 500   # score for destroying a main block
PTS_DEBRIS      = 100   # score for destroying a debris piece
PTS_TARGET      = 5000  # score for destroying a target
PTS_UNUSED_BIRD = 10000 # bonus for remaining birds

TARGET_HEALTH   = 22    # hit points for a target (one solid graze kills)

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
# Frames the hand must read as *not* a fist before the grab is treated as
# released. The tracker briefly mis-reads a moving fist as a partly open hand,
# and without this every such flicker fired or cancelled the shot.
GRAB_RELEASE_FRAMES  = 5

# ── Letting go ───────────────────────────────────────────────────────────────
# `is_fist` is far too slow to end a hold. The engine debounces the hand sign
# over three same-sign frames, and an opening hand cycles through several signs
# on the way out — each one restarting that count — so it can keep reporting
# "fist" for ten frames or more after the fingers start to move. Stack
# GRAB_RELEASE_FRAMES on top and the bird left the band roughly a third of a
# second after the player let go, all of it spent still tracking an aim point
# that the splaying fingers were dragging upward.
#
# Release now thresholds `grip_openness` instead: a continuous, undebounced
# 0.0 (curled fist) → 1.0 (fingers straight) signal from the engine.
# A fist in motion sits near 0.05, so 0.34 leaves a wide noise margin.
GRIP_RELEASE_OPENNESS = 0.34  # above this the hand is opening, not just moving
GRIP_RELEASE_FRAMES   = 2     # consecutive frames above it that end the hold
# Snap-release shortcut: a hand thrown open crosses the whole range in one or
# two frames, so a large single-frame jump fires without waiting for the count.
GRIP_RELEASE_GATE     = 0.20  # minimum openness for the rate test to apply
GRIP_RELEASE_RATE     = 0.18  # openness gained in one frame that means "opening"
# Frames the hand may stay untracked mid-pull before the shot fires anyway. A
# hand released close to the camera drops off MediaPipe before its open frames
# can be counted, and without this the bird stays stuck on the band.
LOST_HAND_FIRE_FRAMES = 8
Z_PUSH_DETECT_THRESH = 0.008 # z_delta value above which we consider "Z pushing"

# ── READY phase (grab -> settle -> aim) ──────────────────────────────────────
# Selecting a bird requires reaching into the carousel strip near the top of the
# frame. Locking the aim anchor there forced the whole pull to happen with the
# arm raised, and left far more pull room below the anchor than above it. The
# READY phase breaks that link: the fist holds the bird while the hand travels
# anywhere it likes, and the anchor only locks once the hand stops moving.
READY_SETTLE_FRAMES  = 12   # consecutive near-still frames that lock the anchor
READY_SETTLE_RADIUS  = 22   # px the hand may wander and still count as "still"
# Never trap the player in READY. A hand that keeps wandering — camera noise, or
# simply a player who does not realise they need to hold still — locks anyway.
READY_MAX_FRAMES     = 150
# Losing the hand in READY cannot fire anything (there is no pull yet), so a long
# dropout just puts the bird back rather than guessing at an aim.
READY_LOST_CANCEL_FRAMES = 30

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
# Kinds are lowercase because they double as asset filenames — assets/ruby.png,
# assets/ruby_side.png — and a case-sensitive filesystem is unforgiving about
# the difference. Display code title-cases them.
RUBY  = "ruby"
AMBER = "amber"
SLATE = "slate"
AZURE = "azure"
IVORY = "ivory"

#: The thing you are trying to hit. Beakless and tailless so it reads as a
#: different species from the birds, and deliberately plain beyond that — a
#: snout with two nostrils is the single most recognisable trait of the obvious
#: inspiration, so there isn't one. Its spec is TARGET_SPEC, further down.
TARGET = "target"

BIRD_ORDER = [RUBY, AMBER, SLATE, AZURE, IVORY]

BIRD_RADII = {RUBY: 28, AMBER: 26, SLATE: 30, AZURE: 22, IVORY: 26}

BIRD_MASSES = {RUBY: 1.0, AMBER: 0.7, SLATE: 1.1, AZURE: 0.5, IVORY: 1.0}

# Damage multiplier applied on top of mass x speed when a bird hits a block.
# Gives each bird a distinct role: Slate is the wrecker, Azure the light
# precision bird that relies on speed rather than punch.
BIRD_DAMAGE = {RUBY: 1.0, AMBER: 1.35, SLATE: 1.7, AZURE: 0.9, IVORY: 1.2}

# BGR, to match the OpenCV frames these are drawn onto. Kept in step with the
# RGB values in BIRD_SPECS below — these tint the trail, impact ring and
# speed lines, so a mismatch shows up as effects that clash with the sprite.
BIRD_COLORS = {
    RUBY:  (58,  74,  226),
    AMBER: (52,  190, 240),
    SLATE: (96,  82,  74),
    AZURE: (214, 148, 72),
    IVORY: (232, 238, 238),
}

#: Which 3D primitive stands in for each kind in the depth-tilt pass.
BIRD_MESH_SHAPES = {
    RUBY: "sphere", AMBER: "pyramid", SLATE: "sphere",
    AZURE: "sphere", IVORY: "capsule",
}

# The engine is NOT put on sys.path here: the old inserts pointed inside this
# repo at a "visual ai game engine" directory that does not exist (the clone is
# a sibling of the repo, one level further up) and so never did anything.
# main.py's engine_bootstrap is what actually resolves visual_ai; the fallback
# below covers standalone imports of this module without it.
try:
    from visual_ai import CreatureSpec, Material, ShaderType
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

    # A stand-in so this module still imports without the engine. Nothing can
    # render from it — bird.py needs the real renderer — but tools that only
    # want tuning constants out of config.py keep working.
    @dataclass
    class CreatureSpec:
        name: str = "bird"
        body: str = "round"
        colour: tuple = (226, 74, 58)
        belly: tuple | None = None
        outline: tuple = (28, 28, 34)
        outline_width: float = 0.055
        beak: tuple = (247, 181, 56)
        beak_size: float = 0.19
        eye_size: float = 0.095
        eye_colour: tuple = (255, 255, 255)
        pupil: tuple = (28, 28, 34)
        gloss: float = 0.22
        tail: float = 0.0
        scale: float = 1.0

#  BLOCK MATERIALS
# =============================================================================
# BGR color values; health/density per material, and visual_ai Material instance
BLOCK_MATERIALS = {
    "wood": {
        "health":  22,
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
        "health":  44,
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
        "health":  8,
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
# Defined using the visual_ai Material class. Keys are bird kinds.
CHARACTER_3D_MATERIALS = {
    RUBY: Material(
        name="RubyMat",
        shader_type=ShaderType.PBR_STANDARD,
        base_color=(0.89, 0.29, 0.23, 1.0),
        roughness=0.35,
        metallic=0.05,
    ),
    AMBER: Material(
        name="AmberMat",
        shader_type=ShaderType.PBR_STANDARD,
        base_color=(0.94, 0.75, 0.20, 1.0),
        roughness=0.30,
        metallic=0.10,
    ),
    SLATE: Material(
        name="SlateMat",
        shader_type=ShaderType.PBR_STANDARD,
        base_color=(0.29, 0.32, 0.38, 1.0),
        roughness=0.40,
        metallic=0.30,
    ),
    AZURE: Material(
        name="AzureMat",
        shader_type=ShaderType.PBR_STANDARD,
        base_color=(0.28, 0.58, 0.84, 1.0),
        roughness=0.25,
        metallic=0.0,
    ),
    IVORY: Material(
        name="IvoryMat",
        shader_type=ShaderType.PBR_STANDARD,
        base_color=(0.93, 0.93, 0.91, 1.0),
        roughness=0.20,
        metallic=0.0,
    ),
    TARGET: Material(
        name="TargetMat",
        shader_type=ShaderType.PBR_STANDARD,
        base_color=(0.34, 0.70, 0.38, 1.0),
        roughness=0.35,
        metallic=0.0,
    ),
}


#  CHARACTER ARTWORK
# =============================================================================
# The cast, as data. Each entry is everything the engine needs to draw that
# creature from both angles — there is no per-character drawing code anywhere
# in this game. Change a color here and the sprite changes; add an entry and
# a new bird exists.
#
# Deliberately plain designs: dot eyes, a small beak, no brows and no crests.
# Expressive brows and head crests are what make a cartoon bird recognisable as
# somebody else's cartoon bird.
#
# To author artwork instead of generating it, render a spec to PNGs and drop
# them in assets/ — bird.py prefers files over generated sprites:
#     python "../../visual ai game engine/tools/spritesmith.py" cast --out assets
BIRD_SPECS = {
    RUBY: CreatureSpec(
        name=RUBY, body="round", colour=(226, 74, 58),
        belly=(246, 214, 196), beak=(247, 181, 56), scale=1.0, tail=0.20,
    ),
    AMBER: CreatureSpec(
        name=AMBER, body="wedge", colour=(240, 190, 52),
        belly=(250, 228, 160), beak=(232, 128, 44), scale=0.94,
        eye_size=0.085, tail=0.16,
    ),
    SLATE: CreatureSpec(
        name=SLATE, body="round", colour=(74, 82, 96),
        belly=(120, 130, 148), beak=(196, 168, 92), scale=1.06,
        gloss=0.28, tail=0.18,
    ),
    AZURE: CreatureSpec(
        name=AZURE, body="round", colour=(72, 148, 214),
        belly=(198, 226, 246), beak=(240, 176, 72), scale=0.80,
        eye_size=0.105, tail=0.24,
    ),
    IVORY: CreatureSpec(
        name=IVORY, body="oval", colour=(238, 238, 232),
        belly=(255, 255, 252), outline=(96, 96, 104),
        beak=(244, 168, 60), scale=0.96, gloss=0.15, tail=0.20,
    ),
}

TARGET_SPEC = CreatureSpec(
    name=TARGET, body="round", colour=(86, 178, 96),
    belly=(150, 214, 150), outline=(28, 44, 30),
    beak_size=0.0, tail=0.0, eye_size=0.13, gloss=0.26, scale=1.0,
)


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
HUD_TEXT_COLOR = (240, 240, 240)
SHADOW_COLOR   = (30, 30, 30)
TRAJ_COLOR     = (220, 220, 220)
OVERLAY_ALPHA = 0.45

#  CAROUSEL UI
# =============================================================================
CAROUSEL_Y = 90
CAROUSEL_SPACING = 120
CAROUSEL_PANEL_H = 140
CAROUSEL_SELECTION_MAX_Y = 220

#  CURSOR SETTINGS
# =============================================================================
CURSOR_INDEX_COLOR = (0, 255, 200)
CURSOR_PINCH_COLOR = (0, 200, 255)
CURSOR_FIRE_COLOR  = (0, 80, 255)

#  LEVEL DIMENSIONS
# =============================================================================
BW, BH = 20, 40     # block width, height
TH = 14             # block thickness (planks)
LEVEL_X_OFF = -250  # Level X offset
LEVEL_NAMES = ["Easy", "Medium", "Hard"]
