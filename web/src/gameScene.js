import Phaser from 'phaser';
import { generateGameTextures } from './textureGenerator.js';
import { BIRD_ORDER, RADII, MASSES } from './bird.js';
import { MATERIALS } from './block.js';
import {
  drawCarousel,
  drawHUD,
  drawDoneOverlay,
  drawGround,
  drawScorePopups,
} from './ui.js';
import { draw as drawSlingshot, triggerSnapback } from './slingshot.js';

// ── Constants ────────────────────────────────────────────────────────────────
const SLING_X = 400;
const SLING_Y = 550;
const FLOOR_Y = 660;
const MAX_PULL = 150;
const POWER_FACTOR = 0.003;
const GRAVITY_Y = 1.5;

// ── Phaser Scene ─────────────────────────────────────────────────────────────
export class GameScene extends Phaser.Scene {
  constructor() {
    super({ key: 'GameScene' });
  }

  init() {
    this.tracker = this.game.registry.get('tracker');

    this.gameState = 'SELECTION';
    this.levelIdx = 0;
    this.score = 0;
    this.blocksDestroyed = 0;
    this.lastClickMode = 'PINCH';

    this.birdQueue = [...BIRD_ORDER];
    this.selectedIdx = 0;

    this.bird = null;
    this.blocks = [];
    this.wasPinching = false;
    this.resetTimer = null;
    this.popups = [];

    this.slingshotTexture = null;
    this.uiTexture = null;
  }

  preload() { }

  create() {
    generateGameTextures(this);

    // Physics
    this.matter.world.setBounds(0, 0, 1280, 720, 100, true, true, false, true);
    this.matter.world.setGravity(0, GRAVITY_Y);

    // Background
    const bgTex = this.textures.createCanvas('bgLayer', 1280, 720);
    drawGround(bgTex.context, FLOOR_Y);
    bgTex.refresh();
    this.add.image(640, 360, 'bgLayer').setDepth(-10);

    // Ground body
    this.matter.add.rectangle(640, 720, 1280, 120, { isStatic: true, label: 'ground' });

    // Slingshot overlay canvas (uses rich drawing from slingshot.js)
    this.slingshotTexture = this.textures.createCanvas('slingshotLayer', 1280, 720);
    this.add.image(640, 360, 'slingshotLayer').setDepth(2);

    // Phaser Graphics layers
    this.gameGraphics = this.add.graphics({ depth: 10 });
    this.pointerGraphics = this.add.graphics({ depth: 1000 });

    // UI canvas layer
    this.uiTexture = this.textures.createCanvas('uiLayer', 1280, 720);
    this.add.image(640, 360, 'uiLayer').setDepth(100);

    // Collision events
    this.matter.world.on('collisionstart', (event) => this._onCollision(event));

    // Keyboard
    this.input.keyboard.on('keydown', (e) => this._onKey(e.key));

    this.buildLevel(this.levelIdx);
  }

  // ── Keyboard ─────────────────────────────────────────────────────────────────

  _onKey(key) {
    if (key === 'r' || key === 'R') { this.resetLevel(); return; }
    if (key === '1') { this.levelIdx = 0; this.resetLevel(); return; }
    if (key === '2') { this.levelIdx = 1; this.resetLevel(); return; }
    if (key === '3') { this.levelIdx = 2; this.resetLevel(); return; }
  }

  // ── Level / Block Management ─────────────────────────────────────────────────

  buildLevel(levelIdx) {
    this.blocks.forEach(b => { if (b && b.gameObject) b.gameObject.destroy(); });
    this.blocks = [];

    const BW = 30, BH = 60, TH = 20;

    const addBlock = (x, y, w, h, material) => {
      const texKey = w > h ? `plank_${material}_horiz` : `block_${material}_vert`;
      const mat = MATERIALS[material];
      const cx = x + w / 2;
      const cy = y + h / 2;

      const go = this.matter.add.image(cx, cy, texKey, null, {
        restitution: 0.25,
        friction: 0.5,
        density: mat.density * 0.02,
        label: 'block',
      });
      go.setDepth(3);

      const entry = { gameObject: go, material, health: mat.health, maxHealth: mat.health, active: true };
      this.blocks.push(entry);
    };

    const X_OFF = -310;
    const Y_OFF = -110;
    const BASE_Y = FLOOR_Y + Y_OFF;

    // Platform
    if (this.platformGraphic) this.platformGraphic.destroy();
    this.platformGraphic = this.add.rectangle(950 + X_OFF, BASE_Y, 400, 20, 0x654321);
    this.platformGraphic.setDepth(2);
    if (this.platformBody) this.matter.world.remove(this.platformBody);
    this.platformBody = this.matter.add.rectangle(950 + X_OFF, BASE_Y, 400, 20, { isStatic: true, label: 'ground' });

    if (levelIdx === 0) {
      for (const gx of [820, 900, 980, 1060]) addBlock(gx + X_OFF, BASE_Y - 10 - BH, BW, BH, 'wood');
      for (const gx of [860, 980]) addBlock(gx + X_OFF, BASE_Y - 10 - BH * 2, BW, BH, 'wood');
      addBlock(920 + X_OFF, BASE_Y - 10 - BH * 3, BW, BH, 'wood');
      addBlock(820 + X_OFF, BASE_Y - 10 - BH - TH, 280, TH, 'wood');
      addBlock(860 + X_OFF, BASE_Y - 10 - BH * 2 - TH, 180, TH, 'wood');

    } else if (levelIdx === 1) {
      addBlock(840 + X_OFF, BASE_Y - 10 - BH, BW, BH, 'wood');
      addBlock(840 + X_OFF, BASE_Y - 10 - BH * 2, BW, BH, 'wood');
      addBlock(1020 + X_OFF, BASE_Y - 10 - BH, BW, BH, 'ice');
      addBlock(1020 + X_OFF, BASE_Y - 10 - BH * 2, BW, BH, 'ice');
      addBlock(820 + X_OFF, BASE_Y - 10 - BH * 2 - TH, 230, TH, 'wood');
      addBlock(920 + X_OFF, BASE_Y - 10 - BH * 2 - TH - BH, BW * 2, BH, 'stone');

    } else if (levelIdx === 2) {
      for (const gx of [800, 900, 1000, 1100]) addBlock(gx + X_OFF, BASE_Y - 10 - BH, BW, BH, 'stone');
      addBlock(780 + X_OFF, BASE_Y - 10 - BH - TH, 370, TH, 'stone');
      for (const gx of [840, 940, 1040]) addBlock(gx + X_OFF, BASE_Y - 10 - BH - TH - BH, BW, BH, 'wood');
      addBlock(825 + X_OFF, BASE_Y - 10 - BH * 2 - TH * 2, 260, TH, 'wood');
      addBlock(860 + X_OFF, BASE_Y - 10 - BH * 2 - TH * 2 - BH, BW, BH, 'ice');
      addBlock(935 + X_OFF, BASE_Y - 10 - BH * 2 - TH * 2 - BH, BW + 10, BH, 'stone');
      addBlock(1000 + X_OFF, BASE_Y - 10 - BH * 2 - TH * 2 - BH, BW, BH, 'ice');
    }
  }

  // ── Collision Events ──────────────────────────────────────────────────────────

  _onCollision(event) {
    for (const pair of event.pairs) {
      const { bodyA, bodyB } = pair;
      const isBird = (b) => b.label === 'bird';
      const isBlock = (b) => b.label === 'block';

      // Bird hits block
      if ((isBird(bodyA) && isBlock(bodyB)) || (isBird(bodyB) && isBlock(bodyA))) {
        const birdBody = isBird(bodyA) ? bodyA : bodyB;
        const blockBody = isBlock(bodyB) ? bodyB : bodyA;
        const vel = birdBody.velocity;
        const speed = Math.hypot(vel.x, vel.y);
        if (speed < 1) continue;

        const entry = this.blocks.find(b => b.active && b.gameObject && b.gameObject.body === blockBody);
        if (!entry) continue;

        const damage = speed * 3 * (1.0 / MATERIALS[entry.material].density);
        entry.health -= damage;
        const pts = Math.max(10, Math.floor(speed * 20));
        this.score += pts;
        this._addPopup(entry.gameObject.x, entry.gameObject.y - 20, pts);
        if (entry.health <= 0) this._destroyBlock(entry);
      }

      // Block hits block
      if (isBlock(bodyA) && isBlock(bodyB)) {
        const speed = Math.hypot(
          bodyA.velocity.x - bodyB.velocity.x,
          bodyA.velocity.y - bodyB.velocity.y
        );
        if (speed < 2) continue;
        for (const bb of [bodyA, bodyB]) {
          const entry = this.blocks.find(b => b.active && b.gameObject && b.gameObject.body === bb);
          if (!entry) continue;
          entry.health -= speed * 1.5 * (1.0 / MATERIALS[entry.material].density);
          if (entry.health <= 0) this._destroyBlock(entry);
        }
      }
    }
  }

  _destroyBlock(entry) {
    if (!entry.active) return;
    entry.active = false;
    this.score += 500;
    this.blocksDestroyed++;
    this._addPopup(entry.gameObject.x, entry.gameObject.y - 10, 500);
    entry.gameObject.destroy();
    entry.gameObject = null;

    const remaining = this.blocks.filter(b => b.active);
    if (remaining.length === 0) {
      this.gameState = 'DONE';
    }
  }

  // ── Bird Spawn ────────────────────────────────────────────────────────────────

  spawnBird(kind) {
    if (this.bird) { this.bird.destroy(); this.bird = null; }
    const r = RADII[kind];
    const m = MASSES[kind];
    this.bird = this.matter.add.image(SLING_X, SLING_Y, `bird_${kind}`, null, {
      shape: { type: 'circle', radius: r },
      restitution: 0.4,
      friction: 0.1,
      density: 0.04 * m,
      label: 'bird',
    });
    this.bird.setDepth(5);
    this.bird.setStatic(true);
  }

  // ── Popups ────────────────────────────────────────────────────────────────────

  _addPopup(x, y, value) {
    this.popups.push({ x, y, value, alpha: 1.0 });
  }

  _updatePopups() {
    for (let i = this.popups.length - 1; i >= 0; i--) {
      const p = this.popups[i];
      p.y -= 0.8;
      p.alpha -= 0.02;
      if (p.alpha <= 0) this.popups.splice(i, 1);
    }
  }

  // ── Main Update ───────────────────────────────────────────────────────────────

  update() {
    this.pointerGraphics.clear();
    this.gameGraphics.clear();

    if (!this.tracker) this.tracker = this.game.registry.get('tracker');

    let isPinching = false, isZClicking = false;
    let px = 0, py = 0, ix = 0, iy = 0;
    let zDebug = 0;

    if (this.tracker && this.tracker.gesture && this.tracker.gesture.hand_visible) {
      const g = this.tracker.gesture;
      isPinching = g.is_pinching;
      isZClicking = g.click_just_fired;
      zDebug = g.zDelta || 0;
      [ix, iy] = g.index_pos;
      [px, py] = g.pinch_pos;
      if (isZClicking) this.lastClickMode = 'Z-PUSH';
      if (isPinching) this.lastClickMode = 'PINCH';

      // Cursor ring
      this.pointerGraphics.lineStyle(2, 0x00ffc8, 1.0);
      this.pointerGraphics.strokeCircle(ix, iy, 12);
      this.pointerGraphics.fillStyle(0x00ffc8, 1.0);
      this.pointerGraphics.fillCircle(ix, iy, 4);

      if (isPinching) {
        this.pointerGraphics.fillStyle(0x00c8ff, 0.75);
        this.pointerGraphics.fillCircle(px, py, 10);
        this.pointerGraphics.lineStyle(1.5, 0x00c8ff, 1.0);
        this.pointerGraphics.strokeCircle(px, py, 14);
      }
    }

    // State machine
    if (this.gameState === 'SELECTION') this._updateSelection(isPinching, isZClicking, ix, iy);
    else if (this.gameState === 'ARMED') this._updateArmed(isPinching, px, py);
    else if (this.gameState === 'FLIGHT') this._updateFlight();

    this.wasPinching = isPinching;

    this._drawSlingshotLayer();
    this._updatePopups();
    this._drawUI(zDebug);
  }

  // ── State Handlers ────────────────────────────────────────────────────────────

  _updateSelection(isPinching, isZClicking, ix, iy) {
    const total = this.birdQueue.length;
    if (total === 0) { this.gameState = 'DONE'; return; }

    if (iy <= 200) {
      const spacing = 120;
      const panelW = spacing * total + 80;
      const panelX = 640 - panelW / 2;
      let idx = Math.floor(((ix - panelX) / panelW) * total);
      idx = Math.max(0, Math.min(total - 1, idx));
      this.selectedIdx = idx;
    }

    const nearSling = Math.abs(ix - SLING_X) < 150 && Math.abs(iy - SLING_Y) < 150;
    if (isZClicking || (!this.wasPinching && isPinching)) {
      if (iy <= 200 || nearSling) {
        this.gameState = 'ARMED';
        this.aimAnchor = { x: (ix || px || SLING_X), y: (iy || py || SLING_Y) };
        const kind = this.birdQueue[this.selectedIdx];
        this.spawnBird(kind);
        this.birdQueue.splice(this.selectedIdx, 1);
        if (this.selectedIdx >= this.birdQueue.length) this.selectedIdx = 0;
      }
    }
  }

  _updateArmed(isPinching, px, py) {
    if (!this.bird) return;

    const curX = isPinching ? px : (this.tracker && this.tracker.gesture ? this.tracker.gesture.index_pos[0] : 0);
    const curY = isPinching ? py : (this.tracker && this.tracker.gesture ? this.tracker.gesture.index_pos[1] : 0);

    if (isPinching || (this.aimAnchor && curX > 0)) {
      if (!this.aimAnchor) this.aimAnchor = { x: curX, y: curY };
      let relDx = (curX - this.aimAnchor.x) * 1.8;
      let relDy = (curY - this.aimAnchor.y) * 1.8;
      const d = Math.hypot(relDx, relDy);
      let dx = relDx, dy = relDy;
      if (d > MAX_PULL) { dx = (dx / d) * MAX_PULL; dy = (dy / d) * MAX_PULL; }
      this.bird.setPosition(SLING_X + dx, SLING_Y + dy);

    } else if (this.wasPinching && !isPinching) {
      // Fire!
      this.bird.setStatic(false);
      const pullDx = SLING_X - this.bird.x;
      const pullDy = SLING_Y - this.bird.y;
      this.bird.applyForce(new Phaser.Math.Vector2(pullDx * POWER_FACTOR, pullDy * POWER_FACTOR));
      triggerSnapback(this.bird.x, this.bird.y);
      this.gameState = 'FLIGHT';
      return;
    }

    // Trajectory preview dots
    if (this.bird) {
      const pullDx = SLING_X - this.bird.x;
      const pullDy = SLING_Y - this.bird.y;
      const birdMass = this.bird.body ? this.bird.body.mass : 1;
      const scale = 1 / (birdMass * 0.12);
      let tx = this.bird.x, ty = this.bird.y;
      let tvx = pullDx * POWER_FACTOR * scale;
      let tvy = pullDy * POWER_FACTOR * scale;
      const nDots = 26;
      for (let i = 0; i < nDots; i++) {
        tvy += GRAVITY_Y;
        tx += tvx;
        ty += tvy;
        const alpha = 1.0 - i / nDots;
        const rDot = Math.max(2, Math.floor(5 * alpha));
        this.gameGraphics.fillStyle(0xf0f0f5, alpha * 0.85);
        this.gameGraphics.fillCircle(tx, ty, rDot);
      }
    }
  }

  _updateFlight() {
    if (!this.bird) { this._triggerNextTurn(); return; }

    const { x: bx, y: by } = this.bird;
    if (bx > 1400 || bx < -100 || by > 820) { this._triggerNextTurn(); return; }

    const vel = this.bird.body ? this.bird.body.velocity : { x: 0, y: 0 };
    if (Math.abs(vel.x) < 0.15 && Math.abs(vel.y) < 0.15) {
      if (!this.resetTimer) {
        this.resetTimer = setTimeout(() => this._triggerNextTurn(), 1800);
      }
    } else if (this.resetTimer) {
      clearTimeout(this.resetTimer);
      this.resetTimer = null;
    }
  }

  // ── Slingshot Overlay ─────────────────────────────────────────────────────────

  _drawSlingshotLayer() {
    if (!this.slingshotTexture) return;
    const ctx = this.slingshotTexture.context;
    ctx.clearRect(0, 0, 1280, 720);

    let birdPos = null;
    let pullDist = 0;
    if (this.gameState === 'ARMED' && this.bird) {
      birdPos = [this.bird.x, this.bird.y];
      pullDist = Math.hypot(this.bird.x - SLING_X, this.bird.y - SLING_Y);
    }

    drawSlingshot(ctx, birdPos, pullDist);
    this.slingshotTexture.refresh();
  }

  // ── UI Canvas Pass ────────────────────────────────────────────────────────────

  _drawUI(zDebug) {
    if (!this.uiTexture) return;
    const ctx = this.uiTexture.context;
    ctx.clearRect(0, 0, 1280, 720);

    // Health bars for live blocks
    for (const entry of this.blocks) {
      if (!entry.active || !entry.gameObject) continue;
      const hpRatio = Math.max(0, entry.health / entry.maxHealth);
      if (hpRatio >= 1.0) continue;
      const go = entry.gameObject;
      const bw = go.displayWidth;
      const bh = 5;
      const bx = go.x - bw / 2;
      const by = go.y - go.displayHeight / 2 - 10;
      ctx.fillStyle = 'rgb(40,40,40)';
      ctx.fillRect(bx, by, bw, bh);
      const hpCol = `rgb(${Math.floor(200 * (1 - hpRatio))},${Math.floor(200 * hpRatio)},0)`;
      ctx.fillStyle = hpCol;
      ctx.fillRect(bx, by, bw * hpRatio, bh);
    }

    // Selection carousel
    if (this.gameState === 'SELECTION') {
      drawCarousel(ctx, this.birdQueue, this.selectedIdx);
    }

    // Done / level-complete overlay
    if (this.gameState === 'DONE') {
      const bonus = this.birdQueue.length * 1000;
      drawDoneOverlay(ctx, this.score, bonus, this.blocksDestroyed, this.levelIdx);
    }

    // Score popups
    drawScorePopups(ctx, this.popups);

    // HUD
    const g = this.lastGesture || {};
    drawHUD(
      ctx,
      this.gameState,
      this.birdQueue,
      this.lastClickMode,
      zDebug,
      this.score,
      this.levelIdx,
      g.tof_active || false,
      g.tof_z_m || 0.0,
      g.depth_source || "RGB MediaPipe Estimate"
    );

    this.uiTexture.refresh();
  }

  // ── Turn / Level Control ──────────────────────────────────────────────────────

  _triggerNextTurn() {
    clearTimeout(this.resetTimer);
    this.resetTimer = null;
    if (this.bird) { this.bird.destroy(); this.bird = null; }
    this.gameState = this.birdQueue.length > 0 ? 'SELECTION' : 'DONE';
  }

  resetLevel() {
    clearTimeout(this.resetTimer);
    this.resetTimer = null;
    this.gameState = 'SELECTION';
    this.birdQueue = [...BIRD_ORDER];
    this.selectedIdx = 0;
    this.score = 0;
    this.blocksDestroyed = 0;
    this.popups = [];
    this.wasPinching = false;
    if (this.bird) { this.bird.destroy(); this.bird = null; }
    this.buildLevel(this.levelIdx);
  }
}
