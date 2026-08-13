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
    Z_CLICK_THRESHOLD_M, Z_CLICK_XY_MAX_PX,
    BLOCK_COLL_Y_STACK, BLOCK_MATERIALS,
)


from visual_ai import Vector2, intersect_circle_aabb

# ── AABB collision ─────────────────────────────────────────────────────────────
def bird_hits_block(bird, block) -> bool:
    """Axis-aligned bounding-box test: circle (bird) vs rectangle (block) using visual_ai math."""
    bx, by, bw, bh = block.rect
    return intersect_circle_aabb((bird.x, bird.y, bird.radius), (bx, by, bw, bh))


def bird_hits_ground(bird) -> bool:
    """Check if bird has reached the ground level."""
    return (bird.y + bird.radius) >= FLOOR_Y


# ── Vector helpers ─────────────────────────────────────────────────────────────
def magnitude(vx, vy) -> float:
    return Vector2(vx, vy).length()


def distance(p1, p2) -> float:
    return Vector2(p1[0], p1[1]).distance_to(Vector2(p2[0], p2[1]))


def _legacy_resolve_block_collision(b1, b2):
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

    mat1 = BLOCK_MATERIALS.get(getattr(b1, "material", "wood"), BLOCK_MATERIALS["wood"])
    mat2 = BLOCK_MATERIALS.get(getattr(b2, "material", "wood"), BLOCK_MATERIALS["wood"])
    
    # Average material properties for collision
    coll_restitution = (mat1.get("restitution", 0.35) + mat2.get("restitution", 0.35)) / 2.0
    coll_friction = (mat1.get("friction", 0.5) + mat2.get("friction", 0.5)) / 2.0
    
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
        b1.vx = new_v1 * coll_restitution
        b2.vx = new_v2 * coll_restitution
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
        b1.vy = new_v1 * coll_restitution
        b2.vy = new_v2 * coll_restitution

        # Friction when resting
        b1.vx *= coll_friction
        b2.vx *= coll_friction

    # Impact damage (only above minimum velocity) using pre-collision impact_speed
    if impact_speed > MIN_DAMAGE_VEL:
        damage = (impact_speed - MIN_DAMAGE_VEL) * 2.5
        if not is_b1_static:
            if getattr(b1, "is_pig", False):
                b1.health = 0
            else:
                b1.health -= damage
            b1.angular_vel += random.uniform(-1.5, 1.5) * (impact_speed / 5.0)
        if not is_b2_static:
            if getattr(b2, "is_pig", False):
                b2.health = 0
            else:
                b2.health -= damage
            b2.angular_vel += random.uniform(-1.5, 1.5) * (impact_speed / 5.0)


def resolve_block_collision(b1, b2):
    """Resolve an AABB collision using stable mass-aware impulses.

    Unlike a velocity swap, this keeps static platforms fixed, prevents stacked
    blocks from gaining energy, and lets falling structures transfer impact
    damage in a predictable way.
    """
    if not (b1.left < b2.right and b1.right > b2.left and
            b1.top < b2.bottom and b1.bottom > b2.top):
        return

    static1 = getattr(b1, "is_static", False)
    static2 = getattr(b2, "is_static", False)
    if static1 and static2:
        return

    m1 = max(1.0, b1.rect[2] * b1.rect[3] * getattr(b1, "density", 1.0))
    m2 = max(1.0, b2.rect[2] * b2.rect[3] * getattr(b2, "density", 1.0))
    inv1 = 0.0 if static1 else 1.0 / m1
    inv2 = 0.0 if static2 else 1.0 / m2
    inv_total = inv1 + inv2
    if inv_total == 0.0:
        return

    overlap_x = min(b1.right - b2.left, b2.right - b1.left)
    overlap_y = min(b1.bottom - b2.top, b2.bottom - b1.top)
    if overlap_x < overlap_y:
        nx = -1.0 if b1.cx < b2.cx else 1.0
        ny = 0.0
        penetration = overlap_x
    else:
        nx = 0.0
        ny = -1.0 if b1.cy < b2.cy else 1.0
        penetration = overlap_y

    # Remove penetration first. A small allowance suppresses resting jitter.
    correction = max(0.0, penetration - 0.25) / inv_total
    if not static1:
        b1.rect[0] += nx * correction * inv1
        b1.rect[1] += ny * correction * inv1
    if not static2:
        b2.rect[0] -= nx * correction * inv2
        b2.rect[1] -= ny * correction * inv2

    rvx, rvy = b1.vx - b2.vx, b1.vy - b2.vy
    normal_speed = rvx * nx + rvy * ny
    impact_speed = max(0.0, -normal_speed)
    if normal_speed < 0.0:
        mat1 = BLOCK_MATERIALS.get(getattr(b1, "material", "wood"), BLOCK_MATERIALS["wood"])
        mat2 = BLOCK_MATERIALS.get(getattr(b2, "material", "wood"), BLOCK_MATERIALS["wood"])
        restitution = min(0.45, (mat1.get("restitution", 0.25) + mat2.get("restitution", 0.25)) / 2.0)
        impulse = -(1.0 + restitution) * normal_speed / inv_total
        ix, iy = impulse * nx, impulse * ny
        if not static1:
            b1.vx += ix * inv1
            b1.vy += iy * inv1
        if not static2:
            b2.vx -= ix * inv2
            b2.vy -= iy * inv2

        # Friction damps sideways sliding only after a real contact.
        friction = (mat1.get("friction", 0.5) + mat2.get("friction", 0.5)) / 2.0
        tangent_damping = max(0.0, 1.0 - friction * 0.22)
        if nx == 0.0:
            if not static1: b1.vx *= tangent_damping
            if not static2: b2.vx *= tangent_damping
        else:
            if not static1: b1.vy *= tangent_damping
            if not static2: b2.vy *= tangent_damping

    if impact_speed > MIN_DAMAGE_VEL:
        damage = (impact_speed - MIN_DAMAGE_VEL) * 2.5
        if not static1:
            if getattr(b1, "is_pig", False):
                b1.health = 0
            else:
                b1.health -= damage
        if not static2:
            if getattr(b2, "is_pig", False):
                b2.health = 0
            else:
                b2.health -= damage
