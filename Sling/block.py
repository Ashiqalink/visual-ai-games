"""
block.py — Block with material system (wood/stone/ice), health, gravity,
           floor bounce, angular rotation, and progressive fracture rendering.
"""

import sys
import os
import cv2
import numpy as np
import math
import random

# Ensure visual ai game engine imports are accessible
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'visual ai game engine', 'src'))
from visual_ai import (
    Renderer3D, Mesh3D, Transform3D, Camera3D, Material,
    render_creature, rgb_to_bgr, load_image,
)
from visual_ai.imaging import blit_sprite, blit_ellipse_alpha

from physics import GRAVITY, FLOOR_Y, RESTITUTION, DAMAGE_FACTOR, BLOCK_HEALTH
from config import (
    BLOCK_VX_TRANSFER, BLOCK_VY_TRANSFER, DEBRIS_FADE_FRAMES,
    BLOCK_MATERIALS as MATERIALS, LIGHTING_SETTINGS,
    TARGET, TARGET_SPEC,
)

# Global 3D Renderer instance for Block 3D depth rendering
_block_cam3d = Camera3D(fov=60.0, position=(0.0, 0.0, 500.0), screen_width=1280.0, screen_height=720.0)
_block_renderer3d = Renderer3D(
    camera=_block_cam3d,
    light_angle_deg=LIGHTING_SETTINGS.get("LIGHT_ANGLE", 45.0),
    ambient_intensity=LIGHTING_SETTINGS.get("AMBIENT_INTENSITY", 0.45),
    light_intensity=LIGHTING_SETTINGS.get("LIGHT_INTENSITY", 0.85),
)
_block_cube_mesh = Mesh3D.create_cube(size=1.0)  # normalized unit cube mesh

# Target artwork, rendered once. Same rule as the birds: an authored PNG in
# assets/ wins, otherwise the engine rasterises the spec.
_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_target_sprite_cache: list = []


def _target_sprite() -> np.ndarray:
    """The target as BGRA, built on first use and reused thereafter."""
    if not _target_sprite_cache:
        path = os.path.join(_ASSET_DIR, f"{TARGET}.png")
        sprite = (load_image(path) if os.path.isfile(path)
                  else render_creature(TARGET_SPEC, view="front", size=192))
        _target_sprite_cache.append(rgb_to_bgr(sprite))
    return _target_sprite_cache[0]



class Block:
    def __init__(self, x: int, y: int, w: int = 60, h: int = 60,
                 material: str = "wood"):
        # rect = top-left corner
        self.rect = [float(x), float(y), float(w), float(h)]
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.angle: float = 0.0          # degrees, for visual tilt
        self.angular_vel: float = 0.0
        self.active: bool = True
        self.on_ground: bool = False
        self.is_static: bool = False
        
        # Debris lifespan (ported from visual_ai)
        self.is_debris: bool = False
        self.lifespan: int = -1

        # Material system
        self.material = material
        mat = MATERIALS.get(material, MATERIALS["wood"])
        self.health: float     = mat["health"]
        self.max_health: float = mat["health"]
        self.density: float    = mat["density"]

    # ── Convenience ──────────────────────────────────────────────────────────
    @property
    def cx(self): return self.rect[0] + self.rect[2] / 2
    @property
    def cy(self): return self.rect[1] + self.rect[3] / 2
    @property
    def left(self):   return self.rect[0]
    @property
    def right(self):  return self.rect[0] + self.rect[2]
    @property
    def top(self):    return self.rect[1]
    @property
    def bottom(self): return self.rect[1] + self.rect[3]

    # ── Physics update ───────────────────────────────────────────────────────
    def update(self):
        if not self.active:
            return
        if self.health <= 0:
            self.active = False
            return
        if self.is_static:
            return

        # Debris lifespan logic
        if self.is_debris and self.lifespan > 0:
            self.lifespan -= 1
            if self.lifespan <= 0:
                self.active = False
                return

        self.vy += GRAVITY
        self.rect[0] += self.vx
        self.rect[1] += self.vy
        # Keep level pieces aligned.  Impact feedback is represented by visible
        # material damage rather than distracting rigid-body rotation.
        self.angle = 0.0
        self.angular_vel = 0.0
        mat = MATERIALS.get(self.material, MATERIALS["wood"])

        # Settle toppled blocks and fragments on the floor instead of letting
        # them fall out of the scene.
        if self.bottom >= FLOOR_Y:
            self.rect[1] = FLOOR_Y - self.rect[3]
            if self.vy > 1.2:
                self.vy = -self.vy * RESTITUTION
            else:
                self.vy = 0.0
                self.on_ground = True
            floor_friction = mat.get("friction", 0.5)
            self.vx *= max(0.0, 1.0 - floor_friction * 0.18)

    def apply_impulse(self, bird_vx: float, bird_vy: float, bird_mass: float):
        """Bird hit → damage + physical push."""
        speed  = math.sqrt(bird_vx**2 + bird_vy**2)
        force  = bird_mass * speed
        self.health -= force * DAMAGE_FACTOR
        if not self.is_static:
            self.vx     += bird_vx * BLOCK_VX_TRANSFER
            self.vy     += bird_vy * BLOCK_VY_TRANSFER
            self.on_ground = False

    def _get_damaged_polygon(self, cx: float, cy: float, w: float, h: float, cos_a: float, sin_a: float) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]]]:
        """
        Compute polygonal contour with missing bite chunks/corner cutouts when block is hit.
        Returns:
            corners: Nx2 numpy array of integer pixel coordinates for fillPoly/polylines.
            inner_cut_edges: list of (p1, p2) segment tuples representing exposed inner core borders.
        """
        hp_ratio = max(0.0, self.health / self.max_health)
        half_w, half_h = w / 2.0, h / 2.0

        rot_mat = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

        if hp_ratio >= 0.80:
            poly_local = np.array([
                [-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]
            ])
            corners = (poly_local @ rot_mat + [cx, cy]).astype(np.int32)
            return corners, []

        seed = int(self.rect[0] * 13 + self.rect[1] * 37) % 4

        tl = np.array([-half_w, -half_h])
        tr = np.array([half_w, -half_h])
        br = np.array([half_w, half_h])
        bl = np.array([-half_w, half_h])

        depth_factor = 0.22 if hp_ratio >= 0.50 else (0.32 if hp_ratio >= 0.25 else 0.42)
        nw = half_w * depth_factor
        nh = half_h * depth_factor

        local_points = []
        inner_cut_edges_local = []

        if hp_ratio < 0.25:
            p_tl1 = np.array([-half_w + nw, -half_h])
            p_tl2 = np.array([-half_w, -half_h + nh])
            local_points.extend([p_tl1, p_tl2])
            inner_cut_edges_local.append((p_tl1, p_tl2))
        else:
            local_points.append(tl)

        p_tr1 = np.array([half_w - nw, -half_h])
        p_tr2 = np.array([half_w - nw * 0.3, -half_h + nh * 0.7])
        p_tr3 = np.array([half_w, -half_h + nh])
        local_points.extend([p_tr1, p_tr2, p_tr3])
        inner_cut_edges_local.append((p_tr1, p_tr2))
        inner_cut_edges_local.append((p_tr2, p_tr3))

        if hp_ratio < 0.50:
            p_br1 = np.array([half_w, half_h - nh])
            p_br2 = np.array([half_w - nw, half_h])
            local_points.extend([p_br1, p_br2])
            inner_cut_edges_local.append((p_br1, p_br2))
        else:
            local_points.append(br)

        if hp_ratio < 0.25 and seed % 2 == 0:
            p_bl1 = np.array([-half_w + nw, half_h])
            p_bl2 = np.array([-half_w, half_h - nh])
            local_points.extend([p_bl1, p_bl2])
            inner_cut_edges_local.append((p_bl1, p_bl2))
        else:
            local_points.append(bl)

        corners_screen = (np.array(local_points) @ rot_mat + [cx, cy]).astype(np.int32)

        inner_cut_edges_screen = []
        for p1, p2 in inner_cut_edges_local:
            sp1 = tuple((p1 @ rot_mat + [cx, cy]).astype(np.int32))
            sp2 = tuple((p2 @ rot_mat + [cx, cy]).astype(np.int32))
            inner_cut_edges_screen.append((sp1, sp2))

        return corners_screen, inner_cut_edges_screen

    # ── Drawing ──────────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, render_3d: bool = False):
        if not self.active:
            return

        x, y, w, h = [int(v) for v in self.rect]
        cx, cy = x + w // 2, y + h // 2
        frame_h, frame_w = frame.shape[:2]

        # Ground Drop Shadow projection for blocks & platform
        if LIGHTING_SETTINGS.get("SHADOWS_ENABLED", True) and not self.is_debris:
            shadow_offset_y = int(FLOOR_Y - (y + h))
            if 0 <= shadow_offset_y < 180:
                base_opacity = LIGHTING_SETTINGS.get("SHADOW_OPACITY", 0.45)
                shadow_alpha = max(0.06, base_opacity * (1.0 - shadow_offset_y / 180.0))
                
                angle_rad = math.radians(LIGHTING_SETTINGS.get("LIGHT_ANGLE", 45.0))
                light_shift_x = int(math.cos(angle_rad) * LIGHTING_SETTINGS.get("SHADOW_OFFSET_X", 15.0) * (1.0 + shadow_offset_y / 140.0))
                
                scale_x = LIGHTING_SETTINGS.get("SHADOW_SCALE_X", 1.25)
                scale_y = LIGHTING_SETTINGS.get("SHADOW_SCALE_Y", 0.35)
                
                shadow_w = int(w * 0.5 * scale_x)
                shadow_h = max(3, int(h * 0.25 * scale_y))
                shadow_center = (cx + light_shift_x, int(FLOOR_Y - 3))
                shadow_col = LIGHTING_SETTINGS.get("SHADOW_COLOR", (10, 15, 20))
                
                # ROI-blended in the engine — see blit_ellipse_alpha. The
                # frame.copy() this replaces was one of the three largest costs
                # in game.draw.
                blit_ellipse_alpha(frame, shadow_center,
                                   (max(4, shadow_w), max(2, shadow_h)),
                                   shadow_col, shadow_alpha)

        angle_rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        if render_3d and not self.is_debris:
            _block_cam3d.screen_width = float(frame_w)
            _block_cam3d.screen_height = float(frame_h)
            _block_cam3d.focal_length = (float(frame_w) / 2.0) / math.tan(math.radians(_block_cam3d.fov / 2.0))
            _block_cam3d.position[2] = _block_cam3d.focal_length

            mat_info = MATERIALS.get(self.material, MATERIALS["wood"])
            bgr_col = mat_info["color"]
            rgb_float = (bgr_col[2] / 255.0, bgr_col[1] / 255.0, bgr_col[0] / 255.0, 1.0)
            
            vmat = mat_info.get("material")
            opacity = vmat.opacity if vmat else 1.0
            block_mat_3d = Material(base_color=rgb_float, opacity=opacity)

            # Map screen center (cx, cy) to camera coordinate space
            rel_x = float(cx) - (frame_w / 2.0)
            rel_y = (frame_h / 2.0) - float(cy)

            # Draw 3D cube with low, subtle 3D factor (tilt angles rx=3.0, ry=3.0, subtle z-extrusion)
            t3d = Transform3D(
                x=rel_x, y=rel_y, z=0.0,
                rx=3.0, ry=3.0, rz=self.angle,
                sx=float(w), sy=float(h), sz=float(min(w, h)) * 0.15
            )
            _block_renderer3d.render_mesh(frame, _block_cube_mesh, t3d, material=block_mat_3d)

            # Draw crack overlays and health bar on top of 3D block
            angle_rad = math.radians(self.angle)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            self._draw_damage(frame, cx, cy, w, h, cos_a, sin_a)
            return

        # Dynamic polygonal contour (corners disappear as block gets damaged)
        corners, inner_cut_edges = self._get_damaged_polygon(cx, cy, w, h, cos_a, sin_a)

        # Material colours & PBR uniforms
        mat = MATERIALS.get(self.material, MATERIALS["wood"])
        fill_col  = mat["color"]
        dark_col  = mat["dark"]
        grain_col = mat["grain"]
        vmat = mat.get("material")
        uniforms = vmat.to_shader_uniforms() if vmat else {}
        roughness = uniforms.get("u_Roughness", 0.5)
        metallic  = uniforms.get("u_Metallic", 0.0)
        shader_type = uniforms.get("u_ShaderType", "PBR_Standard")

        # Fade out debris before it despawns
        if self.is_debris and self.lifespan > 0 and self.lifespan < DEBRIS_FADE_FRAMES:
            alpha = self.lifespan / 30.0
            bg = (18, 18, 28)
            fill_col = tuple(int(c * alpha + b * (1 - alpha)) for c, b in zip(fill_col, bg))
            dark_col = tuple(int(c * alpha + b * (1 - alpha)) for c, b in zip(dark_col, bg))
            grain_col = tuple(int(c * alpha + b * (1 - alpha)) for c, b in zip(grain_col, bg))

        # Base Fill (damaged polygon silhouette)
        cv2.fillPoly(frame, [corners], fill_col)

        # Material texture.  Keep the marks deterministic so a damaged block
        # retains its visual identity from frame to frame (no shimmer).
        if self.material == "wood":
            for gy in [-h//3, -h//7, h//6, h//3]:
                p1 = np.array([cx + (-w//2 + 4)*cos_a - gy*sin_a,
                               cy + (-w//2 + 4)*sin_a + gy*cos_a], dtype=np.int32)
                p2 = np.array([cx + (w//2 - 4)*cos_a  - gy*sin_a,
                               cy + (w//2 - 4)*sin_a  + gy*cos_a], dtype=np.int32)
                cv2.line(frame, tuple(p1), tuple(p2), grain_col, 1, cv2.LINE_AA)
            for lx, ly in [(-w*.27, -h*.18), (w*.23, h*.22)]:
                px, py = int(cx + lx*cos_a - ly*sin_a), int(cy + lx*sin_a + ly*cos_a)
                cv2.ellipse(frame, (px, py), (max(2, w//12), max(1, h//18)), int(-math.degrees(self.angle)), 0, 360, dark_col, 1, cv2.LINE_AA)
        elif self.material == "stone":
            for lx, ly, rr in [(-.28, -.23, .08), (.24, -.12, .06), (-.08, .25, .07), (.33, .29, .04)]:
                px, py = int(cx + lx*w*cos_a - ly*h*sin_a), int(cy + lx*w*sin_a + ly*h*cos_a)
                cv2.circle(frame, (px, py), max(1, int(min(w, h)*rr)), grain_col, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), max(1, int(min(w, h)*rr)), dark_col, 1, cv2.LINE_AA)
        elif self.material == "ice":
            for t in (-.35, 0.0, .35):
                p1 = (int(cx + (-w*.38)*cos_a - (h*t)*sin_a), int(cy + (-w*.38)*sin_a + (h*t)*cos_a))
                p2 = (int(cx + (w*.32)*cos_a - (h*(t+.24))*sin_a), int(cy + (w*.32)*sin_a + (h*(t+.24))*cos_a))
                cv2.line(frame, p1, p2, grain_col, 1, cv2.LINE_AA)

        # PBR Shading & Specular Highlights
        if shader_type == "Transparent": # Ice / Glass
            # Top-left glass refraction glint
            glint_p1 = (int(cx - (w/2.5)*cos_a + (h/2.5)*sin_a), int(cy - (w/2.5)*sin_a - (h/2.5)*cos_a))
            glint_p2 = (int(cx + (w/4.0)*cos_a + (h/2.5)*sin_a), int(cy + (w/4.0)*sin_a - (h/2.5)*cos_a))
            cv2.line(frame, glint_p1, glint_p2, (255, 250, 240), 2)
            # Inner glass refraction diagonal
            refr_p1 = (int(cx - (w/3.0)*cos_a - (h/4.0)*sin_a), int(cy - (w/3.0)*sin_a + (h/4.0)*cos_a))
            refr_p2 = (int(cx + (w/3.0)*cos_a + (h/4.0)*sin_a), int(cy + (w/3.0)*sin_a - (h/4.0)*cos_a))
            cv2.line(frame, refr_p1, refr_p2, (255, 240, 220), 1)
        else: # Standard PBR (Wood / Stone)
            # Directional specular highlight (top edge)
            spec_intensity = int(120 * (1.0 - roughness) + 80 * metallic)
            spec_col = (min(255, fill_col[0] + spec_intensity),
                        min(255, fill_col[1] + spec_intensity),
                        min(255, fill_col[2] + spec_intensity))
            top_p1 = corners[0]
            top_p2 = corners[1]
            cv2.line(frame, tuple(top_p1), tuple(top_p2), spec_col, max(1, int(2 - roughness * 2)))

            # Ambient shadow bevel (bottom & right edges)
            bot_p1 = corners[2]
            bot_p2 = corners[3]
            cv2.line(frame, tuple(bot_p1), tuple(bot_p2), dark_col, 2)

        # Border
        cv2.polylines(frame, [corners], True, dark_col, 2)

        # Render exposed inner material borders where chunks disappeared
        for p1, p2 in inner_cut_edges:
            cv2.line(frame, p1, p2, dark_col, 3, cv2.LINE_AA)
            cv2.line(frame, p1, p2, (40, 40, 40), 1, cv2.LINE_AA)

        # Crack overlays (drawn on top of the block face)
        self._draw_damage(frame, cx, cy, w, h, cos_a, sin_a)

        # Health bar — only shown when damaged

    # ── Crack overlays ───────────────────────────────────────────────────────

    # ── Crack overlays & Edge Chipping ────────────────────────────────────────

    def _draw_damage(self, frame: np.ndarray, cx: float, cy: float, w: float, h: float, cos_a: float, sin_a: float):
        """Show cumulative damage through material-tailored branching crack networks & corner/edge chips."""
        hp_ratio = max(0.0, self.health / self.max_health)
        if hp_ratio >= 0.85:
            return

        def _rot(lx: float, ly: float) -> tuple[int, int]:
            """Rotate local block coordinates to frame pixel coordinates."""
            return (int(cx + lx * cos_a - ly * sin_a),
                    int(cy + lx * sin_a + ly * cos_a))

        # Crack colors and highlight styles by material
        if self.material == "ice":
            crack_dark = (160, 120, 40)    # Deep cyan blue groove
            crack_light = (255, 245, 220)  # Bright icy refraction line
            chip_col = (110, 170, 190)
        elif self.material == "stone":
            crack_dark = (30, 30, 30)      # Charcoal deep fissure
            crack_light = (140, 140, 140)  # Mineral edge glint
            chip_col = (70, 70, 70)
        else: # Wood
            crack_dark = (15, 35, 70)      # Dark timber split
            crack_light = (90, 150, 210)   # Light wood core exposed
            chip_col = (80, 50, 25)

        # ── Tier 1: Light Surface Fractures (< 85% HP) ──
        if self.material == "wood":
            # Fibrous grain splits along horizontal wood axis
            branches = [
                [(-w*0.35, -h*0.2), (-w*0.1, -h*0.18), (w*0.15, -h*0.22), (w*0.38, -h*0.2)],
                [(-w*0.25, h*0.15), (0.0, h*0.18), (w*0.2, h*0.14)]
            ]
        elif self.material == "stone":
            # Jagged diagonal fractures
            branches = [
                [(-w*0.3, -h*0.35), (-w*0.15, -h*0.1), (w*0.05, h*0.1), (w*0.25, h*0.3)],
                [(-w*0.1, -h*0.1), (w*0.15, -h*0.25)]
            ]
        else: # Ice
            # Webbed spiderweb radial cracks
            branches = [
                [(0.0, 0.0), (-w*0.3, -h*0.3)],
                [(0.0, 0.0), (w*0.35, -h*0.2)],
                [(0.0, 0.0), (-w*0.2, h*0.35)],
                [(-w*0.15, -h*0.15), (-w*0.32, 0.0)],
                [(w*0.17, -h*0.1), (w*0.3, h*0.2)]
            ]

        # Draw crack shadow pass then highlight pass for 3D groove depth
        for path in branches:
            pts = np.array([_rot(px, py) for px, py in path], dtype=np.int32)
            cv2.polylines(frame, [pts], False, crack_dark, 2, cv2.LINE_AA)
            cv2.polylines(frame, [pts], False, crack_light, 1, cv2.LINE_AA)

        # ── Tier 2: Medium Structural Cracks & Corner Chipping (< 60% HP) ──
        if hp_ratio < 0.6:
            # Top-left corner chip
            chip_size = max(3.0, min(w, h) * 0.22)
            corner_chip = np.array([
                _rot(-w*0.5, -h*0.5),
                _rot(-w*0.5 + chip_size, -h*0.5),
                _rot(-w*0.5, -h*0.5 + chip_size * 0.8)
            ], dtype=np.int32)
            cv2.fillConvexPoly(frame, corner_chip, chip_col)
            cv2.polylines(frame, [corner_chip], True, crack_dark, 1, cv2.LINE_AA)

            # Secondary deep impact crack branch
            sec_branch = [
                _rot(w*0.3, -h*0.4), _rot(w*0.1, -h*0.05),
                _rot(-w*0.15, h*0.2), _rot(-w*0.35, h*0.38)
            ]
            pts_sec = np.array(sec_branch, dtype=np.int32)
            cv2.polylines(frame, [pts_sec], False, crack_dark, 3, cv2.LINE_AA)
            cv2.polylines(frame, [pts_sec], False, crack_light, 1, cv2.LINE_AA)

            # A recessed impact bite reads more clearly than only drawing a
            # crack: dark rim first, then the exposed material beneath it.
            bite_x, bite_y = _rot(w*0.34, -h*0.22)
            bite_r = max(3, int(min(w, h) * (0.07 + (0.6 - hp_ratio) * 0.14)))
            cv2.circle(frame, (bite_x, bite_y), bite_r + 2, crack_dark, -1, cv2.LINE_AA)
            cv2.circle(frame, (bite_x - 1, bite_y - 1), bite_r, chip_col, -1, cv2.LINE_AA)
            cv2.circle(frame, (bite_x - bite_r//3, bite_y - bite_r//3), max(1, bite_r//3), crack_light, -1, cv2.LINE_AA)

        # ── Tier 3: Severe Shatter Fractures & Dual Edge Notches (< 30% HP) ──
        if hp_ratio < 0.3:
            # Bottom-right edge chip notch
            chip_size_br = max(4.0, min(w, h) * 0.28)
            br_chip = np.array([
                _rot(w*0.5, h*0.5),
                _rot(w*0.5 - chip_size_br, h*0.5),
                _rot(w*0.5, h*0.5 - chip_size_br * 0.85)
            ], dtype=np.int32)
            cv2.fillConvexPoly(frame, br_chip, chip_col)
            cv2.polylines(frame, [br_chip], True, crack_dark, 1, cv2.LINE_AA)

            # Cross shatter lines across center
            cross_shatter1 = [_rot(-w*0.45, 0.0), _rot(w*0.45, 0.0)]
            cross_shatter2 = [_rot(0.0, -h*0.45), _rot(0.0, h*0.45)]
            cv2.line(frame, cross_shatter1[0], cross_shatter1[1], crack_dark, 2, cv2.LINE_AA)
            cv2.line(frame, cross_shatter2[0], cross_shatter2[1], crack_dark, 2, cv2.LINE_AA)

            # Multiple missing chunks at critical health create a convincing
            # "about to break" silhouette while retaining the collision box.
            for lx, ly in [(-.38, .27), (.10, -.38), (.38, .12)]:
                px, py = _rot(lx*w, ly*h)
                rr = max(2, int(min(w, h) * .08))
                cv2.circle(frame, (px, py), rr + 1, crack_dark, -1, cv2.LINE_AA)
                cv2.circle(frame, (px - 1, py - 1), rr, chip_col, -1, cv2.LINE_AA)


    # ── Health bar ───────────────────────────────────────────────────────────

class Target(Block):
    """The circular creature you are trying to knock out. Artwork: TARGET_SPEC."""
    def __init__(self, x: int, y: int, radius: int = 20):
        # We store radius in width/height so AABB collision works reasonably well
        super().__init__(x, y, w=radius*2, h=radius*2, material=TARGET)
        
        # Override health for the target specifically
        from config import TARGET_HEALTH
        self.health = TARGET_HEALTH
        self.max_health = TARGET_HEALTH
        self.density = 1.2
        self.is_target = True
        self.radius = radius

    def draw(self, frame: np.ndarray, render_3d: bool = False):
        if not self.active:
            return

        cx, cy = int(self.cx), int(self.cy)
        h, w = frame.shape[:2]

        # 0. Ground Drop Shadow under target
        if LIGHTING_SETTINGS.get("SHADOWS_ENABLED", True):
            shadow_offset_y = int(FLOOR_Y - (self.cy + self.radius))
            if 0 <= shadow_offset_y < 160:
                base_opacity = LIGHTING_SETTINGS.get("SHADOW_OPACITY", 0.45)
                shadow_alpha = max(0.08, base_opacity * (1.0 - shadow_offset_y / 160.0))
                
                angle_rad = math.radians(LIGHTING_SETTINGS.get("LIGHT_ANGLE", 45.0))
                light_shift_x = int(math.cos(angle_rad) * LIGHTING_SETTINGS.get("SHADOW_OFFSET_X", 15.0) * (1.0 + shadow_offset_y / 120.0))
                
                scale_x = LIGHTING_SETTINGS.get("SHADOW_SCALE_X", 1.25)
                scale_y = LIGHTING_SETTINGS.get("SHADOW_SCALE_Y", 0.35)
                
                shadow_w = int(self.radius * scale_x * (1.2 - 0.3 * (shadow_offset_y / 160.0)))
                shadow_h = int(self.radius * scale_y * (1.0 - 0.4 * (shadow_offset_y / 160.0)))
                shadow_center = (cx + light_shift_x, int(FLOOR_Y - 4))
                shadow_col = LIGHTING_SETTINGS.get("SHADOW_COLOR", (10, 15, 20))
                
                # ROI-blended in the engine — see blit_ellipse_alpha.
                blit_ellipse_alpha(frame, shadow_center,
                                   (max(2, shadow_w), max(2, shadow_h)),
                                   shadow_col, shadow_alpha)

        # Body. The sphere-plus-hand-drawn-face this used to be is gone; the
        # creature is a spec in config.py that the engine rasterises, exactly
        # like the birds. A damaged target rocks slightly, which the blit gets
        # for free as a rotation.
        wobble = math.sin(self.health) * 8.0 if self.health < self.max_health else 0.0
        blit_sprite(frame, _target_sprite(), cx, cy,
                    size=int(self.radius * 2.3), angle=wobble)

        # Health bar — only shown when damaged
