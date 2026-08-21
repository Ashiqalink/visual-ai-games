"""
sculptor.py — a lump of geometry on a turntable that you shape with one hand.

What this is testing
--------------------
`render3d.py` is a pure-Python software rasteriser. `demo_3d` exercises it, and
Sling draws birds and blocks through it, but nothing has ever asked what it costs
under *continuous* input at a real resolution for a sustained stretch — which is
the only way the answer matters, because a gesture game that renders at 45 fps
for ten seconds and 22 fps thereafter is a broken game with a good benchmark.

So this has no failure state and no score. It cycles every mesh the renderer can
build, and reports per-mesh render cost with percentiles. `--soak` holds one mesh
for the whole run so thermal or allocation drift shows up as a rising p95.

Controls map hand to shape:
    move hand          orbit the turntable
    grip openness      scale
    ToF depth          extrude along Z
    SPACE / 1-8        next mesh

Scope: game side.

    python sculptor.py                              play it
    python sculptor.py --headless 600               cycle every mesh
    python sculptor.py --headless 900 --soak torus  one mesh, look for drift
"""

from __future__ import annotations

import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from visual_ai import Camera3D, Material, Mesh3D, Renderer3D, Transform3D

import labkit
from labkit import LabGame, draw_bar, draw_panel, draw_text

MESHES = ("cube", "pyramid", "sphere", "cylinder", "capsule", "torus",
          "prism", "icosahedron")

MESH_BUILDERS = {
    "cube":        lambda: Mesh3D.create_cube(size=90.0),
    "pyramid":     lambda: Mesh3D.create_pyramid(width=95.0, height=115.0),
    "sphere":      lambda: Mesh3D.create_sphere(radius=62.0, rings=14, sectors=22),
    "cylinder":    lambda: Mesh3D.create_cylinder(radius=52.0, height=115.0, segments=20),
    "capsule":     lambda: Mesh3D.create_capsule(radius=44.0, height=80.0, rings=10, sectors=18),
    "torus":       lambda: Mesh3D.create_torus(ring_radius=62.0, tube_radius=24.0,
                                               ring_segments=24, tube_segments=14),
    "prism":       lambda: Mesh3D.create_prism(width=84.0, height=84.0, depth=84.0),
    "icosahedron": lambda: Mesh3D.create_icosahedron(radius=66.0),
}

CYCLE_SECONDS = 2.5


def _tri_count(mesh) -> int:
    faces = getattr(mesh, "faces", None)
    if faces is None:
        return 0
    try:
        return int(len(faces))
    except TypeError:
        return int(np.asarray(faces).shape[0])


class Sculptor(LabGame):
    title = "Sculptor — 3D renderer soak"
    width, height = 1280, 800
    wants_max_hands = 1

    goal = ("Shape the solid on the turntable with one hand. There is no score "
            "and nothing to lose — it is here to hold the 3D renderer flat out.")
    controls = (
        ("Move your hand",
         "left/right spins the turntable, up/down tilts it"),
        ("Open and close your grip",
         "an open hand grows the model, a fist shrinks it"),
        ("Push toward the camera",
         "extrudes the model along its depth axis; pull back to flatten it"),
        ("Take your hand away",
         "the model keeps turning on its own"),
    )
    keys = (
        ("SPACE", "next mesh"),
        ("1 - 8", "pick a mesh directly"),
    )

    def add_args(self, parser):
        parser.add_argument("--soak", metavar="MESH",
                            help=f"hold one mesh for the whole run ({', '.join(MESHES)})")

    def configure(self):
        soak = getattr(self.args, "soak", None)
        if soak:
            if soak not in MESH_BUILDERS:
                raise SystemExit(f"unknown mesh {soak!r}; pick from {', '.join(MESHES)}")
            self.soak = soak

    def __init__(self):
        self.soak = None
        self.camera = Camera3D()
        self.renderer = Renderer3D(camera=self.camera, light_angle_deg=38.0)
        self.material = Material(base_color=(0.62, 0.42, 0.86), opacity=1.0)

        self.mesh_index = 0
        self.transform = Transform3D()
        self.render_ms: dict[str, list[float]] = {}
        self.tri_counts: dict[str, int] = {}
        self._set_mesh(MESHES[0])

        self.yaw = 0.0
        self.pitch = 0.0
        self.scale = 1.0
        self.extrude = 1.0
        self.time = 0.0
        self.cycle_timer = 0.0
        self.hand_visible = False

        # ── measurement ───────────────────────────────────────────────────────
        self.frames = 0

    def _set_mesh(self, name):
        self.mesh_name = name
        self.mesh = MESH_BUILDERS[name]()
        self.tri_counts[name] = _tri_count(self.mesh)

    def update(self, payload, dt):
        self.frames += 1
        self.time += dt
        if self.soak:
            if self.mesh_name != self.soak:
                self._set_mesh(self.soak)
        else:
            self.cycle_timer += dt
            if self.cycle_timer >= CYCLE_SECONDS:
                self.cycle_timer = 0.0
                self.mesh_index = (self.mesh_index + 1) % len(MESHES)
                self._set_mesh(MESHES[self.mesh_index])

        self.hand_visible = bool(payload.get("hand_visible"))
        if self.hand_visible:
            hx, hy = payload["index_pos"]
            # Hand position drives the turntable rather than setting it, so the
            # model keeps spinning smoothly instead of snapping to the hand.
            self.yaw = (hx / max(1, self.width) - 0.5) * 360.0
            self.pitch = (hy / max(1, self.height) - 0.5) * -120.0
            self.scale = 0.55 + 1.05 * float(payload.get("grip_openness", 0.5))
            depth = float(payload.get("tof_z_m", 0.45))
            # Nearer hand -> deeper extrusion.
            self.extrude = max(0.25, min(2.4, 1.0 + (0.45 - depth) * 4.0))
        else:
            self.yaw += dt * 40.0

    def render(self, canvas):
        canvas[:] = (20, 16, 26)

        mesh = self.mesh
        if abs(self.extrude - 1.0) > 0.01:
            mesh = mesh.apply_extrusion_depth(self.extrude)

        self.transform.position = (self.width / 2.0, self.height / 2.0, 0.0)
        self.transform.rotation = (self.pitch, self.yaw, math.sin(self.time) * 8.0)
        self.transform.scale = (self.scale, self.scale, self.scale)

        t0 = time.perf_counter()
        try:
            self.renderer.render_mesh(canvas, mesh, self.transform, self.material)
        except TypeError:
            # Older signature ordering; keep the soak running rather than dying
            # halfway through a measurement run.
            self.renderer.render_mesh(canvas, mesh, self.transform)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.render_ms.setdefault(self.mesh_name, []).append(elapsed)

        draw_panel(canvas, 12, 12, 340, 128)
        draw_text(canvas, self.mesh_name.upper(), 26, 46, 0.9, (200, 170, 255))
        draw_text(canvas, f"{self.tri_counts.get(self.mesh_name, 0)} faces", 26, 74,
                  0.55, (180, 190, 210), 1)
        samples = self.render_ms.get(self.mesh_name, [])
        if samples:
            recent = samples[-60:]
            draw_text(canvas, f"render {statistics.fmean(recent):5.2f} ms", 26, 100,
                      0.55, (150, 240, 190), 1)
        draw_text(canvas, f"scale {self.scale:4.2f}   extrude {self.extrude:4.2f}",
                  26, 126, 0.5, (200, 200, 215), 1)

        draw_bar(canvas, self.width - 60, 100, 26, self.height - 200,
                 0.0)   # placeholder rail keeps the layout stable
        draw_text(canvas, "grip = scale · depth = extrude", 26, self.height - 24,
                  0.55, (150, 160, 180), 1)

    def on_key(self, key):
        if key == ord(' '):
            self.mesh_index = (self.mesh_index + 1) % len(MESHES)
            self._set_mesh(MESHES[self.mesh_index])
        elif ord('1') <= key <= ord('8'):
            self.mesh_index = key - ord('1')
            self._set_mesh(MESHES[self.mesh_index])
        return True

    def metrics(self):
        out = {"frames": self.frames, "resolution": [self.width, self.height],
               "soak": self.soak, "per_mesh": {}}
        for name, samples in self.render_ms.items():
            if not samples:
                continue
            ordered = sorted(samples)
            entry = {
                "faces": self.tri_counts.get(name, 0),
                "samples": len(ordered),
                "mean_ms": round(statistics.fmean(ordered), 3),
                "p95_ms": round(ordered[int(len(ordered) * 0.95)], 3),
                "worst_ms": round(ordered[-1], 3),
                "fps_if_alone": round(1000.0 / statistics.fmean(ordered), 1),
            }
            # Drift: first quarter of the run against the last quarter. A
            # renderer that slows down over a session is a different problem
            # from one that is simply slow, and only this split tells them apart.
            if len(samples) >= 40:
                q = len(samples) // 4
                entry["drift_pct"] = round(
                    100.0 * (statistics.fmean(samples[-q:])
                             / statistics.fmean(samples[:q]) - 1.0), 1)
            out["per_mesh"][name] = entry
        return out


def _landmarks(t, i):
    """A hand that keeps orbiting, gripping and pushing so no frame is idle."""
    cx = 0.5 + 0.32 * math.sin(t * 0.9)
    cy = 0.5 + 0.20 * math.cos(t * 0.6)
    sign = "open_palm" if (int(t) % 2 == 0) else "fist"
    z = math.sin(t * 1.4) * 0.2
    return labkit.hand_landmarks(cx=cx, cy=cy, sign=sign, z=z)


Sculptor.headless_landmarks = staticmethod(_landmarks)


if __name__ == "__main__":
    raise SystemExit(labkit.run(Sculptor()))
