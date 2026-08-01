/* game.js — Game State Machine, Levels configurations, and rules logic */

import Matter from 'matter-js';
import { SLING_X, SLING_Y, triggerSnapback, draw as drawSlingshot } from './slingshot.js';
import { Bird, BIRD_ORDER, RED, CHUCK, BOMB, BLUES, WHITE } from './bird.js';
import { Block } from './block.js';
import {
  setupPhysics,
  POWER_FACTOR,
  MAX_PULL,
  FLOOR_Y,
  distance
} from './physics.js';
import { drawGround, drawCarousel, drawTrajectory, drawHUD, drawDoneOverlay, drawScorePopups } from './ui.js';

export class Game {
  constructor(frameW = 1280, frameH = 720) {
    this.W = frameW;
    this.H = frameH;

    this.levelIdx = 0; // 0: Easy, 1: Medium, 2: Hard
    this.score = 0;
    this.popups = []; // Floating text popup elements [{ x, y, value, alpha }]
    this.blocksDestroyed = 0;

    // Aim smoothing settings
    this.smoothedIX = parseFloat(SLING_X);
    this.smoothedIY = parseFloat(SLING_Y);
    this.SMOOTH = 0.15; // EMA factor (lower = smoother)

    this.reset();
  }

  reset() {
    this.state = "SELECTION";
    this.birdQueue = [...BIRD_ORDER];
    this.selectedIdx = 0;
    this.currentBird = null;

    // Matter.js setup
    if (this.engine) {
      Matter.Engine.clear(this.engine);
      Matter.World.clear(this.engine.world);
    }
    this.engine = setupPhysics();
    this.world = this.engine.world;

    this.blocks = this.buildLevel(this.levelIdx);
    this.pullPos = [SLING_X, SLING_Y];
    this.lastClickMode = "Z-PUSH";
    this.zDeltaDisplay = 0.0;
    this.blocksDestroyed = 0;

    // Reset smoothing to slingshot coordinates
    this.smoothedIX = parseFloat(SLING_X);
    this.smoothedIY = parseFloat(SLING_Y);

    // Setup Collision Events
    Matter.Events.on(this.engine, 'collisionStart', (event) => {
      event.pairs.forEach((pair) => {
        const bodyA = pair.bodyA;
        const bodyB = pair.bodyB;

        // We can check if either is a bird hitting a block or block hitting a block
        const pA = bodyA.plugin;
        const pB = bodyB.plugin;

        if (!pA || !pB) return;

        // Calculate impact intensity
        // In Matter.js, we can estimate impact by relative velocity
        const relVel = {
          x: bodyA.velocity.x - bodyB.velocity.x,
          y: bodyA.velocity.y - bodyB.velocity.y
        };
        const impactSpeed = Math.sqrt(relVel.x * relVel.x + relVel.y * relVel.y);

        // Apply damage if speed is high enough
        if (impactSpeed > 2.0) {
          const dmg = impactSpeed * 3;
          if (pA.block) {
            const oldH = pA.block.health;
            pA.block.takeDamage(dmg);
            if (pA.block.health < oldH) {
              const points = Math.floor(oldH - Math.max(0, pA.block.health)) * 10;
              this.score += points;
              this.addPopup(bodyA.position.x, bodyA.position.y, points);
            }
          }
          if (pB.block) {
            const oldH = pB.block.health;
            pB.block.takeDamage(dmg);
            if (pB.block.health < oldH) {
              const points = Math.floor(oldH - Math.max(0, pB.block.health)) * 10;
              this.score += points;
              this.addPopup(bodyB.position.x, bodyB.position.y, points);
            }
          }
          if (pA.bird || pB.bird) {
            const bird = (pA.bird) ? pA.bird : pB.bird;
            bird.startImpactAnim();
          }
        }
      });
    });
  }

  buildLevel(levelIdx) {
    const blocks = [];
    const BW = 30; // standard block width
    const BH = 60; // standard block height
    const TH = 20; // thin plank height
    
    const X_OFF = -290;
    const Y_OFF = SLING_Y - FLOOR_Y; // shifts BASE_Y to SLING_Y
    const BASE_Y = FLOOR_Y + Y_OFF;

    this.platform = Matter.Bodies.rectangle(930 + X_OFF, BASE_Y + 10, 400, 20, {
      isStatic: true, label: 'ground', friction: 0.8, restitution: 0.1
    });
    Matter.World.add(this.world, this.platform);

    const addBlock = (gx, gy, w, h, mat) => {
        blocks.push(new Block(this.world, gx + X_OFF, gy + Y_OFF, w, h, mat));
    };

    if (levelIdx === 0) {
      for (const gx of [820, 900, 980, 1060]) {
        addBlock(gx, FLOOR_Y - BH, BW, BH, "wood");
      }
      for (const gx of [860, 980]) {
        addBlock(gx, FLOOR_Y - BH * 2, BW, BH, "wood");
      }
      addBlock(920, FLOOR_Y - BH * 3, BW, BH, "wood");
      addBlock(820, FLOOR_Y - BH - TH, 280, TH, "wood");
      addBlock(860, FLOOR_Y - BH * 2 - TH, 180, TH, "wood");

    } else if (levelIdx === 1) {
      addBlock(840, FLOOR_Y - BH, BW, BH, "wood");
      addBlock(840, FLOOR_Y - BH * 2, BW, BH, "wood");
      addBlock(1020, FLOOR_Y - BH, BW, BH, "ice");
      addBlock(1020, FLOOR_Y - BH * 2, BW, BH, "ice");
      addBlock(820, FLOOR_Y - BH * 2 - TH, 230, TH, "wood");
      addBlock(920, FLOOR_Y - BH * 2 - TH - BH, BW * 2, BH, "stone");

    } else if (levelIdx === 2) {
      for (const gx of [800, 900, 1000, 1100]) {
        addBlock(gx, FLOOR_Y - BH, BW, BH, "stone");
      }
      addBlock(780, FLOOR_Y - BH - TH, 370, TH, "stone");
      for (const gx of [840, 940, 1040]) {
        addBlock(gx, FLOOR_Y - BH - TH - BH, BW, BH, "wood");
      }
      addBlock(825, FLOOR_Y - BH * 2 - TH * 2, 260, TH, "wood");
      addBlock(860, FLOOR_Y - BH * 2 - TH * 2 - BH, BW, BH, "ice");
      addBlock(935, FLOOR_Y - BH * 2 - TH * 2 - BH, BW + 10, BH, "stone");
      addBlock(1000, FLOOR_Y - BH * 2 - TH * 2 - BH, BW, BH, "ice");
    }

    return blocks;
  }

  addPopup(x, y, val) {
    this.popups.push({ x: x, y: y, value: val, alpha: 1.0 });
  }

  update(gesture, key) {
    if (key === 'r' || key === 'R') { this.reset(); return; }
    if (key === '1') { this.levelIdx = 0; this.reset(); return; }
    if (key === '2') { this.levelIdx = 1; this.reset(); return; }
    if (key === '3') { this.levelIdx = 2; this.reset(); return; }

    if (this.state === "SELECTION") {
      this.updateSelection(gesture);
    } else if (this.state === "ARMED") {
      this.updateArmed(gesture);
    } else if (this.state === "FLIGHT") {
      this.updateFlight(gesture);
    }

    // Step Matter.js Engine
    Matter.Engine.update(this.engine, 1000 / 60);

    // Clean up destroyed blocks from our array
    for (let i = this.blocks.length - 1; i >= 0; i--) {
      if (!this.blocks[i].active) {
        this.blocksDestroyed++;
        this.blocks.splice(i, 1);
      }
    }

    // ── Update Floating Popups ───────────────────────────────────────────────
    for (let i = this.popups.length - 1; i >= 0; i--) {
      const p = this.popups[i];
      p.y -= 0.8;       // float upwards
      p.alpha -= 0.02;  // fade out
      if (p.alpha <= 0) {
        this.popups.splice(i, 1);
      }
    }
  }

  updateSelection(gesture) {
    if (!gesture.hand_visible) return;
    const ix = gesture.index_pos[0];
    const iy = gesture.index_pos[1];

    if (iy > 200) return;

    const total = this.birdQueue.length;
    if (total === 0) {
      this.state = "DONE";
      return;
    }

    const spacing = 120;
    const panelW = spacing * total + 80;
    const px = (this.W - panelW) / 2;

    const relativeX = ix - px;
    let idx = Math.floor((relativeX / panelW) * total);
    idx = Math.max(0, Math.min(total - 1, idx));
    this.selectedIdx = idx;

    if (gesture.click_just_fired) {
      this.lastClickMode = "Z-PUSH";
      this.loadBird();
    } else if (gesture.is_pinching) {
      this.lastClickMode = "PINCH";
      this.loadBird();
    }
  }

  loadBird() {
    const total = this.birdQueue.length;
    if (total > 0) {
      const kind = this.birdQueue[this.selectedIdx];
      this.currentBird = new Bird(this.world, kind, SLING_X, SLING_Y);
      this.smoothedIX = parseFloat(SLING_X);
      this.smoothedIY = parseFloat(SLING_Y);
      this.armedTimer = 60;
      this.state = "ARMED";
    }
  }

  updateArmed(gesture) {
    const margin = 15;
    let ix = this.smoothedIX;
    let iy = this.smoothedIY;

    if (this.armedTimer > 0) {
      this.armedTimer--;
    }

    if (gesture.hand_visible) {
      ix = gesture.index_pos[0];
      iy = gesture.index_pos[1];
    }

    const atEdge = (ix < margin || ix > this.W - margin || iy < margin || iy > this.H - margin);

    if (this.armedTimer <= 0 && (atEdge || (gesture.hand_visible && gesture.click_just_fired))) {
      this.fireBird();
      return;
    }

    if (!gesture.hand_visible) return;

    if (gesture.is_index_isolated) {
      const rawIX = gesture.index_pos[0];
      const rawIY = gesture.index_pos[1];
      const dDist = Math.sqrt((rawIX - this.smoothedIX) ** 2 + (rawIY - this.smoothedIY) ** 2);
      const scale = Math.max(0, Math.min(1, (dDist - 2) / 18));
      const alpha = 0.08 + scale * 0.14;
      this.smoothedIX = alpha * rawIX + (1 - alpha) * this.smoothedIX;
      this.smoothedIY = alpha * rawIY + (1 - alpha) * this.smoothedIY;
    }

    let dx = this.smoothedIX - SLING_X;
    let dy = this.smoothedIY - SLING_Y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;

    if (dist > MAX_PULL) {
      dx = dx / dist * MAX_PULL;
      dy = dy / dist * MAX_PULL;
    }

    const bx = SLING_X + dx;
    const by = SLING_Y + dy;

    // Position the bird manually while armed
    Matter.Body.setPosition(this.currentBird.body, { x: bx, y: by });
    Matter.Body.setVelocity(this.currentBird.body, { x: 0, y: 0 });
  }

  fireBird() {
    const bird = this.currentBird;
    const [fx, fy] = this.getLaunchVelocity();
    bird.launch(fx, fy);

    triggerSnapback(bird.body.position.x, bird.body.position.y);

    if (this.selectedIdx >= 0 && this.selectedIdx < this.birdQueue.length) {
      this.birdQueue.splice(this.selectedIdx, 1);
      this.selectedIdx = Math.max(0, this.selectedIdx - 1);
    }
    this.state = "FLIGHT";
  }

  updateFlight(gesture) {
    const bird = this.currentBird;
    if (!bird) {
      this.nextBird();
      return;
    }

    bird.update();

    const bx = bird.body.position.x;
    const by = bird.body.position.y;
    if (!bird.active || by > FLOOR_Y + 50 || bx > this.W + 100 || bx < -100) {
      this.nextBird();
    }
  }

  nextBird() {
    if (this.currentBird && this.currentBird.active) {
      Matter.World.remove(this.world, this.currentBird.body);
    }
    this.currentBird = null;
    if (this.birdQueue.length > 0) {
      this.state = "SELECTION";
    } else {
      this.state = "DONE";
    }
  }

  getLaunchVelocity() {
    const bird = this.currentBird;
    const dx = SLING_X - bird.body.position.x;
    const dy = SLING_Y - bird.body.position.y;
    // For Matter.js, forces should be scaled down significantly
    return [dx * POWER_FACTOR, dy * POWER_FACTOR];
  }

  // ── Render Calls ───────────────────────────────────────────────────────────

  draw(ctx) {
    // 1. Sky/Ground
    drawGround(ctx, FLOOR_Y);

    if (this.platform) {
      ctx.save();
      ctx.fillStyle = "rgb(101, 67, 33)"; // Dark wood/dirt color
      const pos = this.platform.position;
      ctx.fillRect(pos.x - 200, pos.y - 10, 400, 20);
      ctx.restore();
    }

    // 2. Active Level Blocks
    for (const b of this.blocks) {
      b.draw(ctx);
    }

    // 3. Game state overlays
    if (this.state === "SELECTION") {
      drawSlingshot(ctx, null, 0);
      drawCarousel(ctx, this.birdQueue, this.selectedIdx);
    } else if (this.state === "ARMED") {
      const bird = this.currentBird;
      const bx = bird.body.position.x;
      const by = bird.body.position.y;
      const pullD = distance([bx, by], [SLING_X, SLING_Y]);

      drawSlingshot(ctx, [bx, by], pullD);
      bird.draw(ctx);

      const [fx, fy] = this.getLaunchVelocity();
      // Adjust trajectory visualization for matter.js roughly (visual only)
      drawTrajectory(ctx, bx, by, fx * 20, fy * 20);

    } else if (this.state === "FLIGHT") {
      drawSlingshot(ctx, null, 0);
      if (this.currentBird && this.currentBird.active) {
        this.currentBird.draw(ctx);
      }
    } else if (this.state === "DONE") {
      drawSlingshot(ctx, null, 0);
      const unusedBirdBonus = this.birdQueue.length * 1000;
      drawDoneOverlay(ctx, this.score, unusedBirdBonus, this.blocksDestroyed, this.levelIdx);
    }

    // 4. Floating score texts
    drawScorePopups(ctx, this.popups);

    // 5. Draw overlaying HUD indicators
    drawHUD(
      ctx,
      this.state,
      this.birdQueue,
      this.lastClickMode,
      this.zDeltaDisplay,
      this.score,
      this.levelIdx
    );
  }
}
