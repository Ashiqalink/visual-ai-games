/* physics.js — Physics constants and Matter.js Engine Setup */

import Matter from 'matter-js';

export const GRAVITY = 1.0;        // Matter.js gravity scale
export const FLOOR_Y = 660;         // ground line (pixel row)
export const POWER_FACTOR = 0.025;  // pull-distance → launch force scale for matter.js 
export const MAX_PULL = 150;        // max slingshot pull radius (px)

export const PINCH_THRESHOLD = 30;       // px distance thumb↔index for pinch click
export const Z_CLICK_THRESHOLD_M = 0.025;  // ~1 inch in MediaPipe Z units (normalised ~= metres)
export const Z_CLICK_XY_MAX_PX = 30;     // max allowed X/Y drift during a Z-push to count as click

export const DAMAGE_FACTOR = 0.3;
export const BLOCK_HEALTH = 50;

export const BIRD_LINGER = 90;       // frames bird stays active after first ground hit
export const MIN_DAMAGE_VEL = 2.0;   // minimum impact speed to deal damage to blocks

/**
 * Setup Matter.js Engine
 */
export function setupPhysics() {
  const engine = Matter.Engine.create();

  // Configure gravity
  engine.world.gravity.y = GRAVITY;

  // Create a static ground body
  const ground = Matter.Bodies.rectangle(1280 / 2, FLOOR_Y + 50, 1280, 100, {
    isStatic: true,
    label: 'ground',
    friction: 0.8,
    restitution: 0.1
  });

  Matter.World.add(engine.world, ground);

  return engine;
}

/**
 * Distance between two points.
 */
export function distance(p1, p2) {
  return Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);
}
