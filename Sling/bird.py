"""
bird.py — Bird class (5 types, drawn with OpenCV primitives).

Improvements over original:
  - Per-type mass values
  - Flight trail (fading circles behind the bird)
  - Impact pop ring on collision
  - Speed lines during fast flight
  - Floor bounce + linger timer (bird rolls after landing)
  - Air drag during flight
"""

import cv2
import numpy as np
import math
import collections
import os
from physics import GRAVITY, FLOOR_Y, AIR_DRAG, BIRD_BOUNCE, BIRD_LINGER, magnitude, bird_hits_ground
from config import (
    TRAIL_LEN, SPEED_LINE_THRESHOLD, IMPACT_POP_DURATION, BIRD_IDLE_SPEED,
    RED, CHUCK, BOMB, BLUES, WHITE, BIRD_ORDER,
    BIRD_RADII as RADII, BIRD_MASSES as MASSES, BIRD_COLOURS as COLOURS,
    CHARACTER_3D_MATERIALS, LIGHTING_SETTINGS
)
from visual_ai import Renderer3D, Mesh3D, Transform3D, Camera3D, Vector2

# Trail length (number of past positions stored)
_TRAIL_LEN = TRAIL_LEN

# Shared 3D Camera & Renderer for Bird 3D Mesh rendering
_bird_cam3d = Camera3D(fov=60.0, position=(0.0, 0.0, 500.0))
_bird_renderer3d = Renderer3D(
    camera=_bird_cam3d,
    light_angle_deg=LIGHTING_SETTINGS.get("LIGHT_ANGLE", 45.0),
    ambient_intensity=LIGHTING_SETTINGS.get("AMBIENT_INTENSITY", 0.45),
    light_intensity=LIGHTING_SETTINGS.get("LIGHT_INTENSITY", 0.85),
)

# Cache 3D Mesh geometries per bird kind
_bird_3d_meshes = {
    RED: Mesh3D.create_sphere(radius=RADII[RED]),
    CHUCK: Mesh3D.create_pyramid(width=RADII[CHUCK]*1.8, height=RADII[CHUCK]*2.2),
    BOMB: Mesh3D.create_capsule(radius=RADII[BOMB]*0.9, height=RADII[BOMB]*0.6),
    BLUES: Mesh3D.create_sphere(radius=RADII[BLUES]),
    WHITE: Mesh3D.create_capsule(radius=RADII[WHITE]*0.8, height=RADII[WHITE]*0.8),
}

_bird_images_2d = {}
_bird_images_side_2d = {}


class Bird:
    def __init__(self, kind: str, x: float, y: float):
        self.kind    = kind
        self.x       = float(x)
        self.y       = float(y)
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.radius  = RADII[kind]
        self.mass    = MASSES[kind]
        self.active  = True
        self.launched = False
        self.on_slingshot = False

        # ── New state ─────────────────────────────────────────────────────
        self.trail = collections.deque(maxlen=_TRAIL_LEN) # O(1) past positions for trail effect
        self.impact_timer: int = 0                    # countdown for impact pop ring
        self.grounded: bool   = False                 # True after first floor/block impact
        self.linger_timer: int = 0                    # frames until deactivation after grounding
        self._hit_blocks: set  = set()                # ids of blocks already hit (multi-hit)
        self.rot_z: float = 0.0                       # 3D rotational angle

    # ── Physics ───────────────────────────────────────────────────────────────

    def update(self):
        if not self.launched or not self.active:
            return

        # Gravity
        self.vy += GRAVITY

        # Air drag (subtle deceleration)
        self.vx *= AIR_DRAG
        self.vy *= AIR_DRAG

        # Move
        self.x += self.vx
        self.y += self.vy

        # Roll rotation angle update
        self.rot_z += self.vx * 2.0

        # Store trail position
        self.trail.append((self.x, self.y))

        # Floor collision
        if bird_hits_ground(self):
            self.y = FLOOR_Y - self.radius
            self.vy = -self.vy * BIRD_BOUNCE
            self.vx *= 0.8
            if not self.grounded:
                self.grounded = True
                self.linger_timer = BIRD_LINGER

        # Linger countdown (after grounding, bird rolls then deactivates)
        if self.grounded:
            self.linger_timer -= 1
            speed = magnitude(self.vx, self.vy)
            if self.linger_timer <= 0 or speed < BIRD_IDLE_SPEED:
                self.active = False

        # Impact animation countdown
        if self.impact_timer > 0:
            self.impact_timer -= 1

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, scale: float = 1.0):
        cx, cy = int(self.x), int(self.y)
        r = int(self.radius * scale)
        col = COLOURS[self.kind]

        # 0. Ground Drop Shadow under bird (using global LIGHTING_SETTINGS)
        if LIGHTING_SETTINGS.get("SHADOWS_ENABLED", True):
            shadow_offset_y = int(FLOOR_Y - self.y)
            if 0 < shadow_offset_y < 160:
                base_opacity = LIGHTING_SETTINGS.get("SHADOW_OPACITY", 0.45)
                shadow_alpha = max(0.08, base_opacity * (1.0 - shadow_offset_y / 160.0))
                
                # Dynamic horizontal shadow displacement derived from LIGHT_ANGLE
                angle_rad = math.radians(LIGHTING_SETTINGS.get("LIGHT_ANGLE", 45.0))
                light_shift_x = int(math.cos(angle_rad) * LIGHTING_SETTINGS.get("SHADOW_OFFSET_X", 15.0) * (1.0 + shadow_offset_y / 120.0))
                
                scale_x = LIGHTING_SETTINGS.get("SHADOW_SCALE_X", 1.25)
                scale_y = LIGHTING_SETTINGS.get("SHADOW_SCALE_Y", 0.35)
                
                shadow_w = int(r * scale_x * (1.2 - 0.3 * (shadow_offset_y / 160.0)))
                shadow_h = int(r * scale_y * (1.0 - 0.4 * (shadow_offset_y / 160.0)))
                shadow_center = (cx + light_shift_x, int(FLOOR_Y - 4))
                shadow_col = LIGHTING_SETTINGS.get("SHADOW_COLOR", (10, 15, 20))
                
                # Draw translucent dark oval drop shadow onto frame
                shadow_overlay = frame.copy()
                cv2.ellipse(shadow_overlay, shadow_center, (max(2, shadow_w), max(2, shadow_h)), 0, 0, 360, shadow_col, -1, cv2.LINE_AA)
                cv2.addWeighted(shadow_overlay, shadow_alpha, frame, 1.0 - shadow_alpha, 0, frame)

        # 1. Trail (behind bird)
        if self.launched and len(self.trail) > 1:
            self._draw_trail(frame)

        # 3. Bird Base Body (2D Image or Fallback Silhouette)
        img = _bird_images_side_2d.get(self.kind) if self.on_slingshot else _bird_images_2d.get(self.kind)
        if img is None:
            img = _bird_images_2d.get(self.kind)
        if img is not None and img.shape[2] == 4:
            dim = int(r * 2.2)
            if dim > 0:
                resized = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
                center = (dim / 2.0, dim / 2.0)
                M = cv2.getRotationMatrix2D(center, -math.degrees(self.rot_z * 0.1), 1.0)
                rotated = cv2.warpAffine(resized, M, (dim, dim), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                
                y1, y2 = cy - dim // 2, cy - dim // 2 + dim
                x1, x2 = cx - dim // 2, cx - dim // 2 + dim
                
                y1_f, y2_f = max(0, y1), min(frame.shape[0], y2)
                x1_f, x2_f = max(0, x1), min(frame.shape[1], x2)
                
                if y1_f < y2_f and x1_f < x2_f:
                    y1_i, y2_i = y1_f - y1, y2_f - y1
                    x1_i, x2_i = x1_f - x1, x2_f - x1
                    
                    overlay_crop = rotated[y1_i:y2_i, x1_i:x2_i]
                    frame_crop = frame[y1_f:y2_f, x1_f:x2_f]
                    
                    alpha = overlay_crop[:, :, 3] / 255.0
                    alpha = np.expand_dims(alpha, axis=2)
                    
                    frame[y1_f:y2_f, x1_f:x2_f] = (alpha * overlay_crop[:, :, :3] + (1 - alpha) * frame_crop).astype(np.uint8)
        else:
            # Fallback drawing
            cv2.circle(frame, (cx, cy), r, col, -1)
            cv2.circle(frame, (cx, cy), r, (20, 20, 20), max(1, r // 12))
            if self.kind == RED:
                Bird._draw_red(frame, cx, cy, r, col)
            elif self.kind == CHUCK:
                Bird._draw_chuck(frame, cx, cy, r, col)
            elif self.kind == BOMB:
                Bird._draw_bomb(frame, cx, cy, r, col)
            elif self.kind == BLUES:
                Bird._draw_blues(frame, cx, cy, r, col)
            elif self.kind == WHITE:
                Bird._draw_white(frame, cx, cy, r, col)
            Bird._draw_finish(frame, cx, cy, r, col)

        # 2b. Emissive Spark for Bomb bird fuse
        if self.kind == BOMB and self.launched:
            fuse_x = cx + int(r * 0.1)
            fuse_y = cy - int(r * 0.85)
            cv2.circle(frame, (fuse_x, fuse_y), max(2, r // 5), (0, 220, 255), -1)
            cv2.circle(frame, (fuse_x, fuse_y), max(4, r // 3), (0, 120, 255), 1)

        # 3. Speed lines (during fast flight only)
        if self.launched and not self.grounded and scale >= 1.0:
            self._draw_speed_lines(frame, cx, cy, r)

        # 4. Impact pop ring
        if self.impact_timer > 0 and scale >= 1.0:
            self._draw_impact_pop(frame, cx, cy, r)

    def _draw_3d_body(self, frame: np.ndarray, cx: int, cy: int, scale: float):
        """Render subtle 3D character volume tilt using visual_ai camera/renderer."""
        h, w = frame.shape[:2]
        _bird_cam3d.screen_width = float(w)
        _bird_cam3d.screen_height = float(h)
        _bird_cam3d.focal_length = (float(w) / 2.0) / math.tan(math.radians(_bird_cam3d.fov / 2.0))
        _bird_cam3d.position[2] = _bird_cam3d.focal_length

        # Screen to camera translation
        rel_x = float(cx) - (w / 2.0)
        rel_y = (h / 2.0) - float(cy)

        # Determine dynamic 3D tilt based on flight angle
        rx = 0.0
        ry = 0.0
        rz = self.rot_z
        if self.launched and (abs(self.vx) > 0.1 or abs(self.vy) > 0.1):
            flight_angle = math.degrees(math.atan2(-self.vy, self.vx))
            ry = flight_angle * 0.4
            rx = 10.0

        mesh = _bird_3d_meshes.get(self.kind, _bird_3d_meshes[RED])
        mat = CHARACTER_3D_MATERIALS.get(self.kind, CHARACTER_3D_MATERIALS["red"])

        transform = Transform3D(x=rel_x, y=rel_y, z=0.0, rx=rx, ry=ry, rz=rz, sx=scale*0.95, sy=scale*0.95, sz=scale*0.95)
        _bird_renderer3d.render_mesh(frame, mesh, transform, material=mat)

    @staticmethod
    def _draw_finish(frame: np.ndarray, cx: int, cy: int, r: int, col):
        """Small illustration passes shared by all birds: wing, rim and shine."""
        if r < 5:
            return
        # A tucked wing adds depth without covering the face.
        wing_col = tuple(max(0, int(c * 0.68)) for c in col)
        wing = (cx - int(r * .47), cy + int(r * .18))
        cv2.ellipse(frame, wing, (max(3, int(r*.35)), max(3, int(r*.22))), 28, 10, 190, wing_col, -1, cv2.LINE_AA)
        cv2.ellipse(frame, wing, (max(3, int(r*.35)), max(3, int(r*.22))), 28, 10, 190, (20, 20, 20), 1, cv2.LINE_AA)
        # A restrained upper-left sheen makes the simple OpenCV artwork feel
        # round while preserving the crisp, game-like style.
        shine = tuple(min(255, int(c + (255-c)*.35)) for c in col)
        cv2.ellipse(frame, (cx - int(r*.22), cy - int(r*.35)),
                    (max(2, int(r*.25)), max(1, int(r*.10))), -28, 185, 340,
                    shine, max(1, r//12), cv2.LINE_AA)

    # ── New visual effects ────────────────────────────────────────────────────

    def _draw_trail(self, frame: np.ndarray):
        """Fading circles along past positions."""
        n = len(self.trail)
        col = COLOURS[self.kind]
        for i, (tx, ty) in enumerate(self.trail):
            alpha = (i + 1) / (n + 1)
            r = max(2, int(self.radius * 0.25 * alpha))
            faded = (
                max(0, min(255, int(col[0] * alpha * 0.5))),
                max(0, min(255, int(col[1] * alpha * 0.5))),
                max(0, min(255, int(col[2] * alpha * 0.5))),
            )
            cv2.circle(frame, (int(tx), int(ty)), r, faded, -1)

    def _draw_speed_lines(self, frame: np.ndarray, cx: int, cy: int, r: int):
        """Short lines behind the bird during fast flight."""
        speed = magnitude(self.vx, self.vy)
        if speed < SPEED_LINE_THRESHOLD:
            return
        vel_dir = Vector2(self.vx, self.vy).normalize()
        cos_a, sin_a = vel_dir.x, vel_dir.y
        for i in range(3):
            offset = r + 5 + i * 7
            lx = cx - int(cos_a * offset)
            ly = cy - int(sin_a * offset)
            length = 8 + i * 4
            ex = lx - int(cos_a * length)
            ey = ly - int(sin_a * length)
            bright = max(0, int(180 * (1.0 - i * 0.3)))
            cv2.line(frame, (lx, ly), (ex, ey), (bright, bright, bright), 1)

    def _draw_impact_pop(self, frame: np.ndarray, cx: int, cy: int, r: int):
        """Expanding ring that fades out over 15 frames."""
        progress = 1.0 - self.impact_timer / IMPACT_POP_DURATION
        pop_r = int(r + progress * 20)
        alpha_val = max(0, 1.0 - progress)
        col_base = COLOURS[self.kind]
        col = (
            min(255, int(col_base[0] * alpha_val) + 50),
            min(255, int(col_base[1] * alpha_val) + 50),
            min(255, int(col_base[2] * alpha_val) + 50),
        )
        thick = max(1, 3 - int(progress * 3))
        cv2.circle(frame, (cx, cy), pop_r, col, thick)

    # ── High Quality Species Character Artwork Helpers ────────────────────────

    @staticmethod
    def _draw_tail(f, pts_list, col=(20, 20, 20)):
        """Helper to draw tail feathers."""
        cv2.fillPoly(f, pts_list, col)

    @staticmethod
    def _draw_tuft(f, pts_list, col, line_col=None):
        """Helper to draw head tufts or crest feathers."""
        cv2.fillPoly(f, [pts_list], col)
        if line_col:
            cv2.polylines(f, [pts_list], True, line_col, 1)

    @staticmethod
    def _draw_belly(f, center, axes, angle, startAngle, endAngle, col):
        """Helper to draw belly patches."""
        cv2.ellipse(f, center, axes, angle, startAngle, endAngle, col, -1)

    @staticmethod
    def _draw_eyes(f, left_eye, right_eye, eye_r, pupil_r, bg_col=(255, 255, 255), pupil_col=(0, 0, 0), pupil_offset=(0, 0)):
        """Helper to draw character eyes with pupils and glints."""
        cv2.circle(f, left_eye, eye_r, bg_col, -1)
        cv2.circle(f, right_eye, eye_r, bg_col, -1)
        cv2.circle(f, left_eye, eye_r, (20, 20, 20), 1)
        cv2.circle(f, right_eye, eye_r, (20, 20, 20), 1)

        p_left = (left_eye[0] + pupil_offset[0], left_eye[1] + pupil_offset[1])
        p_right = (right_eye[0] - pupil_offset[0], right_eye[1] + pupil_offset[1])
        
        cv2.circle(f, p_left, pupil_r, pupil_col, -1)
        cv2.circle(f, p_right, pupil_r, pupil_col, -1)
        
        glint_r = max(1, pupil_r // 2)
        cv2.circle(f, (p_left[0] - 1, p_left[1] - 1), glint_r, (255, 255, 255), -1)
        cv2.circle(f, (p_right[0] - 1, p_right[1] - 1), glint_r, (255, 255, 255), -1)

    @staticmethod
    def _draw_beak(f, pts, fill_col, outline_col=None):
        """Helper to draw a beak."""
        cv2.fillPoly(f, [pts], fill_col)
        if outline_col:
            cv2.polylines(f, [pts], True, outline_col, 1)

    @staticmethod
    def _draw_eyebrows_poly(f, pts, col=(15, 15, 15)):
        """Helper to draw filled polygonal eyebrows."""
        cv2.fillPoly(f, [pts], col)

    @staticmethod
    def _draw_eyebrow_lines(f, pt1_left, pt2_left, pt1_right, pt2_right, col, thickness):
        """Helper to draw line-based eyebrows."""
        cv2.line(f, pt1_left, pt2_left, col, thickness)
        cv2.line(f, pt1_right, pt2_right, col, thickness)

    # ── High Quality Species Character Artwork ────────────────────────────────

    @staticmethod
    def _draw_red(f, cx, cy, r, col):
        # 1. Tail Feathers
        tail_pts = np.array([
            [[cx - r, cy - r//6], [cx - r - r//2, cy - r//3], [cx - r + 2, cy - r//8]],
            [[cx - r, cy], [cx - r - r//1.8, cy], [cx - r + 2, cy + r//10]],
            [[cx - r, cy + r//6], [cx - r - r//2, cy + r//3], [cx - r + 2, cy + r//8]]
        ], np.int32)
        Bird._draw_tail(f, tail_pts)

        # 2. Crest Head Feathers
        head_tuft = np.array([
            [cx - r//4, cy - r + 2], [cx - r//2, cy - r - r//2], [cx, cy - r + 4],
            [cx, cy - r + 2], [cx + r//6, cy - r - r//2.2], [cx + r//3, cy - r + 4]
        ], np.int32)
        Bird._draw_tuft(f, head_tuft, col, (20, 20, 100))

        # 4. Belly Patch
        Bird._draw_belly(f, (cx + r//6, cy + r//4), (int(r * 0.7), int(r * 0.55)), 15, 0, 180, (180, 220, 240))

        # 5. Expressive Angry Brows
        brow_pts = np.array([
            [cx - int(r * 0.55), cy - int(r * 0.35)],
            [cx + int(r * 0.55), cy - int(r * 0.35)],
            [cx + int(r * 0.05), cy - int(r * 0.12)],
            [cx - int(r * 0.05), cy - int(r * 0.12)]
        ], np.int32)
        Bird._draw_eyebrows_poly(f, brow_pts, (15, 15, 15))

        # 6. Eyes with Pupils & Glint
        Bird._draw_eyes(f, (cx - r // 4, cy - r // 8), (cx + r // 4, cy - r // 8), 
                        max(3, r // 4), max(1, r // 8), pupil_offset=(1, 0))

        # 7. Shaded Split Beak
        beak_upper = np.array([[cx - r//4, cy + r//10], [cx + r//4, cy + r//10], [cx, cy + int(r * 0.45)]], np.int32)
        beak_lower = np.array([[cx - r//5, cy + int(r * 0.25)], [cx + r//5, cy + int(r * 0.25)], [cx, cy + int(r * 0.55)]], np.int32)
        Bird._draw_beak(f, beak_upper, (0, 180, 255), (0, 100, 180))
        Bird._draw_beak(f, beak_lower, (0, 140, 220))
        cv2.line(f, (cx - r//4, cy + r//4), (cx + r//4, cy + r//4), (0, 100, 180), 1)

    @staticmethod
    def _draw_chuck(f, cx, cy, r, col):
        # 1. Spiky Black Tail Feathers
        tail_pts = np.array([
            [[cx - int(r*0.9), cy], [cx - int(r*1.6), cy - int(r*0.3)], [cx - int(r*0.8), cy + int(r*0.1)]],
            [[cx - int(r*0.9), cy + int(r*0.2)], [cx - int(r*1.7), cy + int(r*0.1)], [cx - int(r*0.8), cy + int(r*0.3)]]
        ], np.int32)
        Bird._draw_tail(f, tail_pts)

        # 2. Spiky Black Head Feathers
        head_tuft = np.array([
            [cx - r//6, cy - int(r * 0.9)],
            [cx - int(r * 0.7), cy - int(r * 1.7)],
            [cx + r//8, cy - int(r * 1.0)],
            [cx - int(r * 0.4), cy - int(r * 1.9)],
            [cx + r//3, cy - int(r * 0.8)]
        ], np.int32)
        Bird._draw_tuft(f, head_tuft, (20, 20, 20))

        # 4. Cream/White Belly Patch
        belly_pts = np.array([
            [cx - int(r * 0.4), cy + int(r * 0.3)],
            [cx + int(r * 1.1), cy + int(r * 0.95)],
            [cx - int(r * 0.95), cy + int(r * 0.95)]
        ], np.int32)
        Bird._draw_tuft(f, belly_pts, (200, 240, 255))

        # 5. Red/Dark Angry Eyebrows
        Bird._draw_eyebrow_lines(f, (cx - int(r * 0.5), cy - int(r * 0.3)), (cx - 2, cy - int(r * 0.48)), 
                                 (cx + int(r * 0.5), cy - int(r * 0.3)), (cx + 2, cy - int(r * 0.48)), 
                                 (20, 20, 140), 3)

        # 6. Eyes with Pupils & Glints
        Bird._draw_eyes(f, (cx - r // 4, cy - r // 8), (cx + r // 4, cy - r // 8), 
                        max(3, r // 4), max(1, r // 8))

        # 7. Long Sharp Orange Beak
        beak_pts = np.array([
            [cx - r//4, cy + r//8],
            [cx + r//4, cy + r//8],
            [cx + int(r * 0.1), cy + int(r * 0.65)]
        ], np.int32)
        Bird._draw_beak(f, beak_pts, (0, 170, 255), (0, 110, 190))

    @staticmethod
    def _draw_bomb(f, cx, cy, r, col):
        # 1. Tail Feathers
        tail_pts = np.array([
            [[cx - r, cy - r//5], [cx - r - r//2, cy - r//3], [cx - r + 2, cy]],
            [[cx - r, cy + r//5], [cx - r - r//2, cy + r//3], [cx - r + 2, cy]]
        ], np.int32)
        Bird._draw_tail(f, tail_pts)

        # 2. Fuse line on top of head
        fuse_start = (cx, cy - r + 2)
        fuse_end = (cx + r // 3, cy - r - r // 2)
        cv2.line(f, fuse_start, fuse_end, (80, 80, 80), 3)
        cv2.circle(f, fuse_end, max(2, r // 6), (40, 100, 160), -1)

        # 4. Cream/Grey Belly Patch
        Bird._draw_belly(f, (cx, cy + r//3), (int(r * 0.65), int(r * 0.45)), 0, 0, 180, (140, 150, 160))

        # 5. Signature Yellow Forehead Spot
        cv2.circle(f, (cx, cy - int(r * 0.5)), max(2, r // 6), (0, 220, 255), -1)

        # 6. Aggressive Fiery Red Eyebrows & Eye Glow
        Bird._draw_eyebrow_lines(f, (cx - int(r * 0.55), cy - int(r * 0.35)), (cx - 2, cy - int(r * 0.2)), 
                                 (cx + int(r * 0.55), cy - int(r * 0.35)), (cx + 2, cy - int(r * 0.2)), 
                                 (0, 60, 220), 3)

        # Red/Orange angry eyes
        Bird._draw_eyes(f, (cx - r // 4, cy - r // 8), (cx + r // 4, cy - r // 8), 
                        max(3, r // 4), max(1, r // 8), bg_col=(0, 0, 220))

        # 7. Stout Yellow Beak
        beak_pts = np.array([
            [cx - r//5, cy + r//8],
            [cx + r//5, cy + r//8],
            [cx, cy + int(r * 0.45)]
        ], np.int32)
        Bird._draw_beak(f, beak_pts, (0, 180, 255), (0, 100, 180))

    @staticmethod
    def _draw_blues(f, cx, cy, r, col):
        # 1. 2 Cute Head Tuft Feathers on top
        tuft = np.array([
            [cx - r//4, cy - r + 2], [cx - r//3, cy - r - r//3], [cx, cy - r + 2],
            [cx, cy - r + 2], [cx + r//6, cy - r - r//3], [cx + r//3, cy - r + 2]
        ], np.int32)
        Bird._draw_tuft(f, tuft, col)

        # 3. Red/Orange Ring Patches around Eyes
        cv2.ellipse(f, (cx - r//4, cy - r//8), (int(r * 0.35), int(r * 0.35)), 0, 0, 360, (50, 80, 200), -1)
        cv2.ellipse(f, (cx + r//4, cy - r//8), (int(r * 0.35), int(r * 0.35)), 0, 0, 360, (50, 80, 200), -1)

        # 4. Cute Big Eyes with Pupils & Glints
        Bird._draw_eyes(f, (cx - r // 4, cy - r // 8), (cx + r // 4, cy - r // 8), 
                        max(3, int(r * 0.28)), max(1, r // 7))

        # 5. Small Yellow Beak
        beak_pts = np.array([
            [cx - r//6, cy + r//8],
            [cx + r//6, cy + r//8],
            [cx, cy + int(r * 0.4)]
        ], np.int32)
        Bird._draw_beak(f, beak_pts, (0, 160, 220))

    @staticmethod
    def _draw_white(f, cx, cy, r, col):
        # 1. 3 Black Crest Head Feathers
        head_tuft = np.array([
            [cx - r//4, cy - int(r*1.1)], [cx - r//2, cy - int(r*1.6)], [cx, cy - int(r*1.1)],
            [cx, cy - int(r*1.1)], [cx, cy - int(r*1.7)], [cx + r//4, cy - int(r*1.1)],
            [cx + r//4, cy - int(r*1.1)], [cx + r//3, cy - int(r*1.5)], [cx + r//2, cy - int(r*1.0)]
        ], np.int32)
        Bird._draw_tuft(f, head_tuft, (20, 20, 20))

        # 2. 3 Black Tail Feathers
        tail_pts = np.array([
            [[cx - r, cy], [cx - int(r*1.5), cy - r//4], [cx - r + 2, cy + r//8]],
            [[cx - r, cy + r//5], [cx - int(r*1.4), cy + r//3], [cx - r + 2, cy + r//4]]
        ], np.int32)
        Bird._draw_tail(f, tail_pts)

        # 4. Soft Pink Cheeks
        cv2.circle(f, (cx - int(r * 0.55), cy + r//6), max(2, r // 4), (180, 180, 240), -1)
        cv2.circle(f, (cx + int(r * 0.55), cy + r//6), max(2, r // 4), (180, 180, 240), -1)

        # 5. Eyes with Pupils & Glints
        Bird._draw_eyes(f, (cx - r // 4, cy - r // 6), (cx + r // 4, cy - r // 6), 
                        max(2, r // 4), max(1, r // 8))

        # 6. Large Shaded Yellow Beak
        beak_pts = np.array([
            [cx - r//4, cy + r//10],
            [cx + r//4, cy + r//10],
            [cx, cy + int(r * 0.5)]
        ], np.int32)
        Bird._draw_beak(f, beak_pts, (0, 180, 255), (0, 120, 200))

# ── OpenCV / MediaPipe Optimization: Sprite Caching ──────────────────────────

# ── Pillow High-Quality Procedural Sprite Generator ──────────────────────────

def _render_supersampled_sprite(draw_fn, dim=200, scale=4):
    """
    Renders artwork onto a high-resolution 4x canvas using Pillow (PIL),
    then downsamples with LANCZOS for super-smooth vector-like anti-aliasing.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    canvas_dim = dim * scale
    img = Image.new("RGBA", (canvas_dim, canvas_dim), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Call the drawing callback with supersampled canvas dimensions
    draw_fn(draw, canvas_dim // 2, canvas_dim // 2, scale)
    
    # Downsample with high quality Lanczos filter
    img_resized = img.resize((dim, dim), Image.Resampling.LANCZOS)
    
    # Convert RGBA PIL image to BGR+Alpha OpenCV array
    arr = np.array(img_resized, dtype=np.uint8)
    bgra = np.zeros_like(arr)
    bgra[:, :, 0] = arr[:, :, 2]  # Red -> Blue
    bgra[:, :, 1] = arr[:, :, 1]  # Green -> Green
    bgra[:, :, 2] = arr[:, :, 0]  # Blue -> Red
    bgra[:, :, 3] = arr[:, :, 3]  # Alpha
    return bgra


def _generate_fallback_sprite(b_type):
    """
    Renders procedural bird artwork using Pillow (PIL) supersampling.
    Falls back to OpenCV dual-pass alpha if PIL is unavailable.
    """
    r_base = 80
    dim = int(r_base * 2.5)

    def draw_bird_pil(draw, cx, cy, scale):
        r = r_base * scale
        col_rgb = COLOURS[b_type][::-1] # BGR -> RGB
        
        # Shadow / Outer stroke
        draw.ellipse([cx - r - 4*scale, cy - r - 4*scale, cx + r + 4*scale, cy + r + 4*scale], fill=(20, 20, 20, 255))
        # Main Body
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*col_rgb, 255))
        
        # Eyes
        eye_r = r * 0.22
        pupil_r = eye_r * 0.45
        left_eye = (cx - r * 0.28, cy - r * 0.15)
        right_eye = (cx + r * 0.28, cy - r * 0.15)
        
        for eye_center in [left_eye, right_eye]:
            ex, ey = eye_center
            draw.ellipse([ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r], fill=(255, 255, 255, 255), outline=(20, 20, 20, 255), width=int(2*scale))
            draw.ellipse([ex - pupil_r, ey - pupil_r, ex + pupil_r, ey + pupil_r], fill=(20, 20, 20, 255))
            # Glint
            draw.ellipse([ex - pupil_r*0.4, ey - pupil_r*0.4, ex, ey], fill=(255, 255, 255, 255))
            
        # Eyebrows
        brow_w = max(2 * scale, 4)
        if b_type == RED:
            draw.polygon([(cx - r*0.6, cy - r*0.45), (cx + r*0.6, cy - r*0.45), (cx, cy - r*0.15)], fill=(120, 20, 20, 255))
        elif b_type == CHUCK:
            draw.line([(cx - r*0.5, cy - r*0.4), (cx - r*0.05, cy - r*0.25)], fill=(140, 20, 20, 255), width=int(brow_w*1.5))
            draw.line([(cx + r*0.05, cy - r*0.25), (cx + r*0.5, cy - r*0.4)], fill=(140, 20, 20, 255), width=int(brow_w*1.5))
        elif b_type == BOMB:
            draw.line([(cx - r*0.5, cy - r*0.35), (cx - r*0.05, cy - r*0.2)], fill=(220, 60, 0, 255), width=int(brow_w*1.8))
            draw.line([(cx + r*0.05, cy - r*0.2), (cx + r*0.5, cy - r*0.35)], fill=(220, 60, 0, 255), width=int(brow_w*1.8))
            
        # Beak (Yellow/Orange)
        beak_pts = [(cx - r*0.22, cy + r*0.05), (cx + r*0.22, cy + r*0.05), (cx, cy + r*0.45)]
        draw.polygon(beak_pts, fill=(255, 180, 0, 255), outline=(200, 120, 0, 255))

        # Top Feather / Crest Tuft
        tuft_pts = [(cx - r*0.15, cy - r*0.9), (cx, cy - r*1.3), (cx + r*0.15, cy - r*0.9)]
        draw.polygon(tuft_pts, fill=(*col_rgb, 255), outline=(20, 20, 20, 255))
        
        # Wing Sheen
        draw.arc([cx - r*0.7, cy - r*0.7, cx + r*0.7, cy + r*0.7], start=200, end=260, fill=(255, 255, 255, 140), width=int(4*scale))

    pil_sprite = _render_supersampled_sprite(draw_bird_pil, dim=dim)
    if pil_sprite is not None:
        return pil_sprite

    # Fallback to OpenCV primitive drawing if Pillow is missing
    r = r_base
    def _draw_to_bg(bg_color):
        img = np.full((dim, dim, 3), bg_color, dtype=np.uint8)
        col = COLOURS[b_type]
        cv2.circle(img, (cx, cy), r, col, -1)
        cv2.circle(img, (cx, cy), r, (20, 20, 20), max(1, r // 12))
        if b_type == RED: Bird._draw_red(img, cx, cy, r, col)
        elif b_type == CHUCK: Bird._draw_chuck(img, cx, cy, r, col)
        elif b_type == BOMB: Bird._draw_bomb(img, cx, cy, r, col)
        elif b_type == BLUES: Bird._draw_blues(img, cx, cy, r, col)
        elif b_type == WHITE: Bird._draw_white(img, cx, cy, r, col)
        Bird._draw_finish(img, cx, cy, r, col)
        return img.astype(np.float32)

    img_b = _draw_to_bg((0, 0, 0))
    img_w = _draw_to_bg((255, 255, 255))
    diff = img_w - img_b
    alpha = np.clip(255.0 - diff[:, :, 0], 0, 255)
    rgb = np.zeros_like(img_b)
    mask = alpha > 0
    for c in range(3):
        rgb[:, :, c][mask] = np.clip((img_b[:, :, c][mask] * 255.0) / alpha[mask], 0, 255)
    rgba = np.zeros((dim, dim, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb.astype(np.uint8)
    rgba[:, :, 3] = alpha.astype(np.uint8)
    return rgba


def _generate_fallback_sprite_side(b_type):
    """
    Renders detailed side-view bird artwork (slingshot aim) using Pillow supersampling.
    Features per-character body shapes, head crests, belly contours, expressive side eyebrows,
    eye glints, and two-piece shaded beaks pointing right.
    """
    r_base = 80
    dim = int(r_base * 2.6)

    def draw_bird_side_pil(draw, cx, cy, scale):
        r = r_base * scale
        col_rgb = COLOURS[b_type][::-1]  # BGR to RGB

        # 1. Tail Feathers (facing left - back of bird)
        if b_type == CHUCK:
            # Long spiky back feathers
            draw.polygon([(cx - r*0.7, cy - r*0.1), (cx - r*1.5, cy - r*0.45), (cx - r*0.6, cy + r*0.1)], fill=(20, 20, 20, 255))
            draw.polygon([(cx - r*0.7, cy + r*0.05), (cx - r*1.6, cy - r*0.15), (cx - r*0.6, cy + r*0.25)], fill=(20, 20, 20, 255))
            draw.polygon([(cx - r*0.65, cy + r*0.2), (cx - r*1.3, cy + r*0.4), (cx - r*0.5, cy + r*0.35)], fill=(20, 20, 20, 255))
        elif b_type == WHITE:
            # 3 plume feathers
            draw.polygon([(cx - r*0.7, cy - r*0.2), (cx - r*1.4, cy - r*0.5), (cx - r*0.6, cy - r*0.05)], fill=(20, 20, 20, 255))
            draw.polygon([(cx - r*0.75, cy), (cx - r*1.5, cy - r*0.1), (cx - r*0.65, cy + r*0.15)], fill=(20, 20, 20, 255))
            draw.polygon([(cx - r*0.7, cy + r*0.15), (cx - r*1.35, cy + r*0.3), (cx - r*0.6, cy + r*0.28)], fill=(20, 20, 20, 255))
        else:
            # Standard 2/3 tail feathers
            draw.polygon([(cx - r*0.8, cy - r*0.2), (cx - r*1.35, cy - r*0.35), (cx - r*0.7, cy + r*0.02)], fill=(20, 20, 20, 255))
            draw.polygon([(cx - r*0.8, cy + r*0.05), (cx - r*1.4, cy + r*0.22), (cx - r*0.7, cy + r*0.25)], fill=(20, 20, 20, 255))

        # 1b. Self Body Drop Shadow (Bottom-left ambient occlusion under body)
        draw.ellipse([cx - r*0.9, cy + r*0.2, cx + r*0.9, cy + r*1.05], fill=(15, 18, 25, 110))

        # 2. Main Body Silhouette
        stroke_w = int(3.5 * scale)
        if b_type == CHUCK:
            # Triangle silhouette leaning forward-right
            tri_pts = [(cx + r*0.8, cy + r*0.7), (cx - r*0.9, cy + r*0.75), (cx - r*0.2, cy - r*1.15)]
            draw.polygon(tri_pts, fill=(*col_rgb, 255), outline=(20, 20, 20, 255))
        elif b_type == WHITE:
            # Egg / Oval shape elongated vertically
            draw.ellipse([cx - r*0.85, cy - r*1.1, cx + r*0.85, cy + r*0.95], fill=(*col_rgb, 255), outline=(20, 20, 20, 255), width=stroke_w)
        else:
            # Round spherical body
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*col_rgb, 255), outline=(20, 20, 20, 255), width=stroke_w)

        # 2b. Ambient Lighting Arc (Top-Right key sun light highlight)
        draw.arc([cx - r*0.85, cy - r*0.85, cx + r*0.85, cy + r*0.85], start=200, end=290, fill=(255, 255, 255, 160), width=int(6*scale))

        # 3. Head Crest / Feathers / Fuse (top)
        if b_type == RED:
            # Dual curved head feathers backward tilt
            draw.polygon([(cx - r*0.2, cy - r*0.9), (cx - r*0.7, cy - r*1.35), (cx + r*0.05, cy - r*0.85)], fill=(*col_rgb, 255), outline=(20, 20, 20, 255))
            draw.polygon([(cx - r*0.35, cy - r*0.85), (cx - r*0.85, cy - r*1.15), (cx - r*0.1, cy - r*0.8)], fill=(*col_rgb, 255), outline=(20, 20, 20, 255))
        elif b_type == CHUCK:
            # Black spiky crown tuft
            draw.polygon([(cx - r*0.2, cy - r*1.0), (cx - r*0.5, cy - r*1.7), (cx + r*0.1, cy - r*0.8)], fill=(20, 20, 20, 255))
            draw.polygon([(cx - r*0.1, cy - r*0.9), (cx - r*0.2, cy - r*1.5), (cx + r*0.2, cy - r*0.7)], fill=(20, 20, 20, 255))
        elif b_type == BOMB:
            # Fuse & Spark Cap
            fuse_w = int(4 * scale)
            draw.line([(cx - r*0.05, cy - r*0.9), (cx - r*0.3, cy - r*1.45)], fill=(70, 70, 70, 255), width=fuse_w)
            draw.ellipse([cx - r*0.4, cy - r*1.6, cx - r*0.2, cy - r*1.35], fill=(255, 120, 0, 255))
            draw.ellipse([cx - r*0.35, cy - r*1.55, cx - r*0.25, cy - r*1.4], fill=(255, 230, 0, 255))
        elif b_type == BLUES:
            # 2 cute small head tufts
            draw.polygon([(cx - r*0.15, cy - r*0.9), (cx - r*0.4, cy - r*1.25), (cx + r*0.05, cy - r*0.85)], fill=(*col_rgb, 255), outline=(20, 20, 20, 255))
        elif b_type == WHITE:
            # 3 black head crest plumes
            draw.polygon([(cx - r*0.1, cy - r*1.0), (cx - r*0.3, cy - r*1.55), (cx + r*0.1, cy - r*0.95)], fill=(20, 20, 20, 255))

        # 4. Belly Patch (facing right side)
        if b_type == RED:
            # Cream/light brown underbelly chord
            draw.chord([cx - r*0.5, cy - r*0.4, cx + r*0.85, cy + r*0.92], start=280, end=100, fill=(235, 225, 200, 255))
        elif b_type == CHUCK:
            # White bottom belly sweep
            draw.polygon([(cx - r*0.7, cy + r*0.72), (cx + r*0.75, cy + r*0.68), (cx - r*0.1, cy + r*0.25)], fill=(240, 245, 255, 255))
        elif b_type == BOMB:
            # Grey underbelly arc
            draw.chord([cx - r*0.6, cy - r*0.2, cx + r*0.8, cy + r*0.9], start=280, end=100, fill=(140, 150, 160, 255))
        elif b_type == WHITE:
            # Soft pink cheeks patch
            draw.ellipse([cx + r*0.15, cy + r*0.05, cx + r*0.55, cy + r*0.45], fill=(255, 190, 200, 255))

        # 5. Bomb's Forehead Yellow Spot
        if b_type == BOMB:
            draw.ellipse([cx + r*0.1, cy - r*0.65, cx + r*0.45, cy - r*0.3], fill=(255, 215, 0, 255))

        # 6. Eye (Single side view eye facing right)
        eye_r = r * (0.24 if b_type in (RED, CHUCK, WHITE) else 0.22 if b_type == BLUES else 0.23)
        pupil_r = eye_r * 0.45
        ex, ey = (cx + r * 0.25, cy - r * 0.15)
        if b_type == CHUCK: ex, ey = (cx + r * 0.18, cy - r * 0.1)

        # Eye background / ring
        if b_type == BOMB:
            draw.ellipse([ex - eye_r - 2*scale, ey - eye_r - 2*scale, ex + eye_r + 2*scale, ey + eye_r + 2*scale], fill=(220, 40, 0, 255))
        elif b_type == BLUES:
            draw.ellipse([ex - eye_r - 3*scale, ey - eye_r - 3*scale, ex + eye_r + 3*scale, ey + eye_r + 3*scale], fill=(210, 70, 50, 255))

        draw.ellipse([ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r], fill=(255, 255, 255, 255), outline=(20, 20, 20, 255), width=int(2*scale))
        # Pupil slightly shifted forward (right) for directional gaze
        draw.ellipse([ex - pupil_r + r*0.06, ey - pupil_r, ex + pupil_r + r*0.06, ey + pupil_r], fill=(20, 20, 20, 255))
        # Catchlight Glint
        draw.ellipse([ex + r*0.02, ey - pupil_r*0.6, ex + r*0.08, ey], fill=(255, 255, 255, 255))

        # 7. Eyebrow (Angled aggressive side eyebrow)
        brow_col = (120, 20, 20, 255) if b_type in (RED, CHUCK) else (220, 50, 0, 255) if b_type == BOMB else (20, 20, 20, 255)
        if b_type == RED:
            draw.polygon([(cx - r*0.05, cy - r*0.42), (cx + r*0.6, cy - r*0.24), (cx + r*0.58, cy - r*0.12), (cx - r*0.05, cy - r*0.28)], fill=brow_col)
        elif b_type == CHUCK:
            draw.polygon([(cx - r*0.15, cy - r*0.38), (cx + r*0.55, cy - r*0.22), (cx + r*0.52, cy - r*0.12), (cx - r*0.15, cy - r*0.26)], fill=brow_col)
        elif b_type == BOMB:
            draw.polygon([(cx - r*0.05, cy - r*0.45), (cx + r*0.58, cy - r*0.28), (cx + r*0.55, cy - r*0.16), (cx - r*0.05, cy - r*0.3)], fill=brow_col)

        # 8. Side Beak (Two-piece shaded beak pointing right)
        if b_type == CHUCK:
            # Long conical yellow/orange beak
            upper_beak = [(cx + r*0.25, cy - r*0.12), (cx + r*1.35, cy + r*0.05), (cx + r*0.28, cy + r*0.14)]
            lower_beak = [(cx + r*0.28, cy + r*0.14), (cx + r*1.2, cy + r*0.18), (cx + r*0.3, cy + r*0.32)]
        elif b_type == WHITE:
            # Stout large yellow beak
            upper_beak = [(cx + r*0.3, cy - r*0.08), (cx + r*1.1, cy + r*0.1), (cx + r*0.32, cy + r*0.2)]
            lower_beak = [(cx + r*0.32, cy + r*0.2), (cx + r*0.95, cy + r*0.25), (cx + r*0.35, cy + r*0.38)]
        elif b_type == BLUES:
            # Small cute beak
            upper_beak = [(cx + r*0.3, cy - r*0.05), (cx + r*0.8, cy + r*0.08), (cx + r*0.32, cy + r*0.16)]
            lower_beak = [(cx + r*0.32, cy + r*0.16), (cx + r*0.7, cy + r*0.2), (cx + r*0.34, cy + r*0.28)]
        else:
            # Standard Red / Bomb beak
            upper_beak = [(cx + r*0.3, cy - r*0.08), (cx + r*1.15, cy + r*0.08), (cx + r*0.32, cy + r*0.18)]
            lower_beak = [(cx + r*0.32, cy + r*0.18), (cx + r*1.0, cy + r*0.22), (cx + r*0.35, cy + r*0.36)]

        # Upper Beak (brighter yellow with top sun highlight)
        draw.polygon(upper_beak, fill=(255, 195, 20, 255), outline=(190, 110, 0, 255))
        # Lower Beak (darker shaded yellow)
        draw.polygon(lower_beak, fill=(225, 130, 0, 255), outline=(170, 90, 0, 255))

        # 9. Rim Light highlight on Beak top
        draw.line([upper_beak[0], upper_beak[1]], fill=(255, 240, 180, 255), width=int(2*scale))

    pil_sprite = _render_supersampled_sprite(draw_bird_side_pil, dim=dim)
    if pil_sprite is not None:
        return pil_sprite

    # Fallback to OpenCV if PIL is not installed
    r = r_base
    def _draw_to_bg(bg_color):
        img = np.full((dim, dim, 3), bg_color, dtype=np.uint8)
        col = COLOURS[b_type]
        cv2.circle(img, (cx, cy), r, col, -1)
        cv2.circle(img, (cx, cy), r, (20, 20, 20), max(1, r // 12))
        Bird._draw_finish(img, cx, cy, r, col)
        return img.astype(np.float32)

    img_b = _draw_to_bg((0, 0, 0))
    img_w = _draw_to_bg((255, 255, 255))
    diff = img_w - img_b
    alpha = np.clip(255.0 - diff[:, :, 0], 0, 255)
    rgb = np.zeros_like(img_b)
    mask = alpha > 0
    for c in range(3):
        rgb[:, :, c][mask] = np.clip((img_b[:, :, c][mask] * 255.0) / alpha[mask], 0, 255)
    rgba = np.zeros((dim, dim, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb.astype(np.uint8)
    rgba[:, :, 3] = alpha.astype(np.uint8)
    return rgba

def _load_bird_images():
    if _bird_images_2d: return
    for b_type in [RED, CHUCK, BOMB, BLUES, WHITE]:
        path = os.path.join("assets", f"{b_type}.png")
        if os.path.exists(path):
            _bird_images_2d[b_type] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            # Cache the OpenCV primitives as a fast sprite!
            _bird_images_2d[b_type] = _generate_fallback_sprite(b_type)
            
        side_path = os.path.join("assets", f"{b_type}_side.png")
        if os.path.exists(side_path):
            _bird_images_side_2d[b_type] = cv2.imread(side_path, cv2.IMREAD_UNCHANGED)
        else:
            # Generate highly detailed procedural side view
            _bird_images_side_2d[b_type] = _generate_fallback_sprite_side(b_type)

_load_bird_images()
