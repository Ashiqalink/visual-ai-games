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
GRAVITY        = 0.2   # downward acceleration (px / frame^2)
FLOOR_Y        = 660    # ground line (pixel row) -- used by level builders
RESTITUTION    = 0.25   # block / floor bounce dampening
POWER_FACTOR   = 0.10   # pull-distance to launch speed ratio
MAX_PULL       = 170    # maximum slingshot pull radius (px)
AIR_DRAG       = 0.998  # per-frame velocity multiplier (< 1 = gentle deceleration)
BIRD_BOUNCE    = 0.35   # bird floor-bounce restitution coefficient
BIRD_LINGER    = 90     # frames bird stays active after first ground/block impact
MIN_DAMAGE_VEL = 2.0    # minimum impact speed to deal block-block damage
DAMAGE_FACTOR  = 2.5    # scales bird speed into block health reduction
BLOCK_HEALTH   = 50     # default block health (overridden per material in block.py)

# Block-block collision dampening (used inside resolve_block_collision)
BLOCK_COLL_X_DAMP  = 0.8   # X velocity dampening after elastic exchange
BLOCK_COLL_Y_DAMP  = 0.5   # Y velocity dampening after elastic exchange
BLOCK_COLL_FRIC    = 0.7   # friction on X velocity when resting
BLOCK_COLL_Y_STACK = 0.1   # vy dampening when a block is resting on another

#  GESTURE / INPUT
# =============================================================================
SMOOTH_ALPHA         = 0.20  # VisionPipeline EMA baseline (higher = faster / more jitter)
PINCH_THRESHOLD      = 30    # thumb-to-index distance for pinch click (px)
THREE_FINGER_PINCH_RADIUS = 55.0 # base centroid distance (px) for 3-finger pinch trigger
Z_CLICK_THRESHOLD_M  = 0.012 # forward movement threshold (~0.5 inch in MediaPipe Z units)
Z_CLICK_XY_MAX_PX    = 30    # max X/Y drift allowed during a Z-push click

#  SLINGSHOT
# =============================================================================
SLING_X = 280           # fork centre X (px)
SLING_Y = 440           # fork centre Y (px)
FORK_SPREAD_X  = 30     # horizontal offset from centre to each fork tip (px)
FORK_RISE_Y    = 40     # how far fork tips sit above fork centre (px)
HANDLE_DROP_Y  = 140    # how far handle base sits below fork centre (px)

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
PTS_BLOCK  = 500   # score for destroying a main block
PTS_DEBRIS = 100   # score for destroying a debris piece

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
DEBRIS_LIFESPAN      = 90   # frames debris pieces live before fading (~3 s at 30 fps)
DEBRIS_VEL_SPREAD    = 10.0 # max random velocity spread for debris (px/frame)
DEBRIS_VY_KICK       = 2.0  # upward kick added to debris vy on spawn
DEBRIS_CARRY_FACTOR  = 0.5  # fraction of parent block velocity inherited by debris
DEBRIS_HEALTH        = 50   # health of freshly spawned debris pieces
DEBRIS_FADE_FRAMES   = 30   # frames before despawn during which debris fades out

# Platform (static level base blocks rest on)
PLATFORM_W      = 400
PLATFORM_H      = 30
PLATFORM_HEALTH = 100  # effectively indestructible

# Block impulse transfer ratios (applied when a bird hits a block)
BLOCK_VX_TRANSFER = 0.4   # fraction of bird vx added to block
BLOCK_VY_TRANSFER = 0.3   # fraction of bird vy added to block

# Bird stop thresholds
BIRD_STOP_SPEED = 2.0   # speed below which a hit bird starts grounding
BIRD_IDLE_SPEED = 0.3   # grounded speed below which the bird deactivates
