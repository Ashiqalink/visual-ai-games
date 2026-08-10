"""
physics.py — Physics constants, gravity, AABB collision, impulse helpers.

All tunable values are imported from config.py.
"""

import math
import random
from config import (
    GRAVITY, FLOOR_Y, RESTITUTION, POWER_FACTOR, MAX_PULL,
    AIR_DRAG, BIRD_BOUNCE, BIRD_LINGER, MIN_DAMAGE_VEL,
    DAMAGE_FACTOR, BLOCK_HEALTH,
    PINCH_THRESHOLD, THREE_FINGER_PINCH_RADIUS, INPUT_MOVEMENT_MAGNIFICATION,
    Z_CLICK_THRESHOLD_M, Z_CLICK_XY_MAX_PX,
    BLOCK_COLL_X_DAMP, BLOCK_COLL_Y_DAMP, BLOCK_COLL_FRIC, BLOCK_COLL_Y_STACK,
)


# ── AABB collision ─────────────────────────────────────────────────────────────
def bird_hits_block(bird, block) -> bool:
    """Axis-aligned bounding-box test: circle (bird) vs rectangle (block)."""
    r = bird.radius
    bx, by, bw, bh = block.rect
    nearest_x = max(bx, min(bird.x, bx + bw))
    nearest_y = max(by, min(bird.y, by + bh))
    dx = bird.x - nearest_x
    dy = bird.y - nearest_y
    return (dx * dx + dy * dy) < (r * r)


def bird_hits_ground(bird) -> bool:
    """Check if bird has reached the ground level."""
    return False


# ── Vector helpers ─────────────────────────────────────────────────────────────
def magnitude(vx, vy) -> float:
    return math.sqrt(vx * vx + vy * vy)


def distance(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def resolve_block_collision(b1, b2):
    """Momentum-based AABB collision resolution between two blocks.

    Uses each block's density (derived from material) to compute mass
    from area (w × h × density).  Velocity exchange follows 1-D elastic
    collision equations with dampening.
    """
    if not (b1.left < b2.right and b1.right > b2.left and
            b1.top < b2.bottom and b1.bottom > b2.top):
        return

    overlap_x = min(b1.right - b2.left, b2.right - b1.left)
    overlap_y = min(b1.bottom - b2.top, b2.bottom - b1.top)

    is_b1_static = getattr(b1, "is_static", False)
    is_b2_static = getattr(b2, "is_static", False)
    if is_b1_static and is_b2_static:
        return

    # Pre-collision relative velocity for impact damage calculation
    rel_vx = b1.vx - b2.vx
    rel_vy = b1.vy - b2.vy
    impact_speed = math.sqrt(rel_vx ** 2 + rel_vy ** 2)

    # Mass from area × density
    m1 = b1.rect[2] * b1.rect[3] * getattr(b1, "density", 1.0)
    m2 = b2.rect[2] * b2.rect[3] * getattr(b2, "density", 1.0)
    
    if is_b1_static:
        r1, r2 = 0.0, 1.0
        m1 = 1e9
    elif is_b2_static:
        r1, r2 = 1.0, 0.0
        m2 = 1e9

    total_mass = m1 + m2
    if total_mass < 0.001: return

    if not is_b1_static and not is_b2_static:
        r1 = m2 / total_mass
        r2 = m1 / total_mass

    if overlap_x < overlap_y:
        # ── Resolve along X ──────────────────────────────────────────
        if b1.cx < b2.cx:
            b1.rect[0] -= overlap_x * r1
            b2.rect[0] += overlap_x * r2
        else:
            b1.rect[0] += overlap_x * r1
            b2.rect[0] -= overlap_x * r2

        # 1-D elastic collision velocity exchange + dampening
        new_v1 = ((m1 - m2) / total_mass) * b1.vx + (2 * m2 / total_mass) * b2.vx
        new_v2 = ((m2 - m1) / total_mass) * b2.vx + (2 * m1 / total_mass) * b1.vx
        b1.vx = new_v1 * BLOCK_COLL_X_DAMP
        b2.vx = new_v2 * BLOCK_COLL_X_DAMP
    else:
        # ── Resolve along Y ──────────────────────────────────────────
        if b1.cy < b2.cy:
            b1.rect[1] -= overlap_y * r1
            b2.rect[1] += overlap_y * r2
            if b1.vy > 0:
                b1.vy *= BLOCK_COLL_Y_STACK
        else:
            b1.rect[1] += overlap_y * r1
            b2.rect[1] -= overlap_y * r2
            if b2.vy > 0:
                b2.vy *= BLOCK_COLL_Y_STACK

        new_v1 = ((m1 - m2) / total_mass) * b1.vy + (2 * m2 / total_mass) * b2.vy
        new_v2 = ((m2 - m1) / total_mass) * b2.vy + (2 * m1 / total_mass) * b1.vy
        b1.vy = new_v1 * BLOCK_COLL_Y_DAMP
        b2.vy = new_v2 * BLOCK_COLL_Y_DAMP

        # Friction when resting
        b1.vx *= BLOCK_COLL_FRIC
        b2.vx *= BLOCK_COLL_FRIC

    # Impact damage (only above minimum velocity) using pre-collision impact_speed
    if impact_speed > MIN_DAMAGE_VEL:
        damage = (impact_speed - MIN_DAMAGE_VEL) * 2.5
        if not is_b1_static:
            b1.health -= damage
            b1.angular_vel += random.uniform(-1.5, 1.5) * (impact_speed / 5.0)
        if not is_b2_static:
            b2.health -= damage
            b2.angular_vel += random.uniform(-1.5, 1.5) * (impact_speed / 5.0)
