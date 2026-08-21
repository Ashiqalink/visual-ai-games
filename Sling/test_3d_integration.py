"""
test_3d_integration.py — Integration test to verify visual_ai 3D elements in Sling.
"""

import os
import sys

import numpy as np

# Ensure visual ai game engine is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'visual ai game engine'))

from visual_ai import Camera3D, Material, Mesh3D, Renderer3D, Transform3D


def test_3d_rendering():
    print("[TEST] Initializing 3D Camera and Renderer...")
    cam = Camera3D(fov=60.0, screen_width=800.0, screen_height=600.0, position=(0.0, 0.0, 500.0))
    renderer = Renderer3D(camera=cam)

    print("[TEST] Creating 3D Mesh Primitives (Sphere, Cube, Pyramid, Cylinder)...")
    sphere = Mesh3D.create_sphere(radius=35.0, rings=10, sectors=14)
    cube = Mesh3D.create_cube(size=50.0)
    pyramid = Mesh3D.create_pyramid(width=40.0, height=60.0)
    cylinder = Mesh3D.create_cylinder(radius=20.0, height=50.0, segments=12)

    canvas = np.zeros((600, 800, 3), dtype=np.uint8)

    # Render red bird 3D sphere
    mat_red = Material(base_color=(1.0, 0.2, 0.2, 1.0))
    t1 = Transform3D(x=-150.0, y=0.0, z=0.0, rx=15.0, ry=45.0, rz=0.0)
    renderer.render_mesh(canvas, sphere, t1, material=mat_red)

    # Render wood 3D block cube
    mat_wood = Material(base_color=(0.8, 0.5, 0.2, 1.0))
    t2 = Transform3D(x=-50.0, y=0.0, z=0.0, rx=25.0, ry=30.0, rz=0.0)
    renderer.render_mesh(canvas, cube, t2, material=mat_wood)

    # Render golden 3D trophy pyramid
    mat_gold = Material(base_color=(1.0, 0.84, 0.0, 1.0))
    t3 = Transform3D(x=50.0, y=0.0, z=0.0, rx=20.0, ry=60.0, rz=0.0)
    renderer.render_mesh(canvas, pyramid, t3, material=mat_gold)

    # Render ice 3D cylinder
    mat_ice = Material(base_color=(0.4, 0.8, 1.0, 0.7), opacity=0.7)
    t4 = Transform3D(x=150.0, y=0.0, z=0.0, rx=30.0, ry=15.0, rz=0.0)
    renderer.render_mesh(canvas, cylinder, t4, material=mat_ice)

    nonzero_count = np.count_nonzero(canvas)
    print(f"[TEST] Rendered frame with {nonzero_count} non-zero pixels.")
    assert nonzero_count > 1000, "Rendering output appears empty!"

def test_game_3d_integration():
    print("[TEST] Testing full Game scene rendering with 3D elements...")
    from game import Game

    game = Game(frame_w=1280, frame_h=720)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 1. Test SELECTION state with 3D Carousel Bird
    dummy_gesture = {"hand_visible": False, "index_pos": (0, 0)}
    game.update_game_state(dummy_gesture, key=255)
    game.draw(canvas)
    assert np.count_nonzero(canvas) > 5000, "Selection state rendering produced empty canvas!"
    print("[TEST] SELECTION state 3D Carousel rendered successfully.")

    # 2. Test WIN state with 3D Golden Trophy
    game.state = "WIN"
    canvas.fill(0)
    game.draw(canvas)
    assert np.count_nonzero(canvas) > 5000, "WIN state 3D Trophy rendering produced empty canvas!"
    print("[TEST] WIN state 3D Victory Trophy rendered successfully.")

    print("[SUCCESS] Game 3D integration tests passed completely.")
    return True

def test_block_3d_rendering():
    print("[TEST] Testing 3D Block rendering position accuracy...")
    from block import Block
    
    blk = Block(x=800, y=400, w=60, h=60, material="wood")
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    blk.draw(canvas, render_3d=True)

    # Check non-zero pixels around (830, 430)
    center_roi = canvas[400:460, 800:860]
    nonzero_center = np.count_nonzero(center_roi)
    print(f"[TEST] Block rendered at (800, 400) with {nonzero_center} non-zero pixels in target ROI.")
    assert nonzero_center > 100, f"Block pixels did not land in target ROI! Count: {nonzero_center}"
    print("[SUCCESS] 3D Block position accuracy verified.")
    return True

if __name__ == "__main__":
    test_3d_rendering()
    test_game_3d_integration()
    test_block_3d_rendering()

