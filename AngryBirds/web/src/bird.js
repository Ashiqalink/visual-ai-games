/* bird.js — Bird class, properties, and canvas rendering */

import Matter from 'matter-js';
import { BIRD_LINGER } from './physics.js';

export const RED = "Red";
export const CHUCK = "Chuck";
export const BOMB = "Bomb";
export const BLUES = "Blues";
export const WHITE = "White";

export const BIRD_ORDER = [RED, CHUCK, BOMB, BLUES, WHITE];

export const RADII = {
  [RED]: 28,
  [CHUCK]: 26,
  [BOMB]: 30,
  [BLUES]: 22,
  [WHITE]: 26
};

export const MASSES = {
  [RED]: 1.0,
  [CHUCK]: 0.7,
  [BOMB]: 1.1,
  [BLUES]: 0.5,
  [WHITE]: 1.0
};

export const COLOURS = {
  [RED]: { r: 200, g: 50, b: 50, str: "rgb(200, 50, 50)" },
  [CHUCK]: { r: 255, g: 215, b: 0, str: "rgb(255, 215, 0)" },
  [BOMB]: { r: 40, g: 40, b: 40, str: "rgb(40, 40, 40)" },
  [BLUES]: { r: 50, g: 80, b: 220, str: "rgb(50, 80, 220)" },
  [WHITE]: { r: 240, g: 240, b: 240, str: "rgb(240, 240, 240)" }
};

const TRAIL_LEN = 14;

export class Bird {
  constructor(world, kind, x, y) {
    this.world = world;
    this.kind = kind;
    this.radius = RADII[kind];
    this.mass = MASSES[kind];
    this.active = true;
    this.launched = false;

    if (world) {
      // Matter.js Body
      this.body = Matter.Bodies.circle(parseFloat(x), parseFloat(y), this.radius, {
        density: this.mass * 0.001,
        friction: 0.3,
        restitution: 0.4,
        label: 'bird'
      });

      // Link class instance to body for collision callbacks
      this.body.plugin = { bird: this };

      // Initially static until launched
      Matter.Body.setStatic(this.body, true);
      Matter.World.add(world, this.body);
    } else {
      // UI / Mock mode
      this.body = {
        position: { x: parseFloat(x), y: parseFloat(y) },
        angle: 0
      };
    }

    // Visual / Animation states
    this.trail = []; // history of [x, y] coordinates
    this.impactTimer = 0;
    this.squashScale = 1.0;
  }

  launch(forceX, forceY) {
    this.launched = true;
    Matter.Body.setStatic(this.body, false);
    Matter.Body.applyForce(this.body, this.body.position, { x: forceX, y: forceY });
  }

  update() {
    if (!this.launched || !this.active) return;

    // Death / Impact squash animation
    if (this.impactTimer > 0) {
      this.impactTimer--;
      const t = this.impactTimer;
      if (t > 10) {
        this.squashScale = 0.6; // flat squash
      } else if (t > 5) {
        this.squashScale = 1.3; // expand pop
      } else {
        this.squashScale = Math.max(0.2, t / 5.0); // shrink away
      }
      if (this.impactTimer <= 0) {
        this.active = false;
        Matter.World.remove(this.world, this.body);
      }
      return;
    }

    // Save trail position
    this.trail.push([this.body.position.x, this.body.position.y]);
    if (this.trail.length > TRAIL_LEN) {
      this.trail.shift();
    }
  }

  startImpactAnim() {
    if (this.impactTimer <= 0) {
      this.impactTimer = 18;
    }
  }


  draw(ctx, scale = 1.0) {
    // Draw trail behind
    if (this.launched && this.trail.length > 1) {
      this.drawTrail(ctx);
    }

    const cx = this.body.position.x;
    const cy = this.body.position.y;
    const effectiveScale = scale * this.squashScale;
    const r = Math.max(1, Math.floor(this.radius * effectiveScale));
    const col = COLOURS[this.kind];

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(this.body.angle);

    if (this.kind === RED) {
      this.drawRed(ctx, r, col.str);
    } else if (this.kind === CHUCK) {
      this.drawChuck(ctx, r, col.str);
    } else if (this.kind === BOMB) {
      this.drawBomb(ctx, r, col.str);
    } else if (this.kind === BLUES) {
      this.drawBlues(ctx, r, col.str);
    } else if (this.kind === WHITE) {
      this.drawWhite(ctx, r, col.str);
    }

    ctx.restore();
  }

  drawTrail(ctx) {
    const col = COLOURS[this.kind];
    const n = this.trail.length;
    for (let i = 0; i < n; i++) {
      const [tx, ty] = this.trail[i];
      const alpha = (i + 1) / n;
      const rDot = Math.max(1, Math.floor(this.radius * 0.3 * alpha));
      ctx.beginPath();
      ctx.arc(tx, ty, rDot, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${col.r}, ${col.g}, ${col.b}, ${alpha * 0.5})`;
      ctx.fill();
    }
  }

  // ── Individual Render Methods (Relative to Local Origin [0, 0]) ─────────────────

  drawRed(ctx, r, col) {
    // 1. Tail Feathers
    ctx.fillStyle = "rgb(20, 20, 20)";
    ctx.beginPath();
    ctx.moveTo(-r, -r / 4);
    ctx.lineTo(-r - r / 2, -r / 3);
    ctx.lineTo(-r + 2, -r / 8);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(-r, 0);
    ctx.lineTo(-r - r / 1.8, 0);
    ctx.lineTo(-r + 2, r / 10);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(-r, r / 6);
    ctx.lineTo(-r - r / 2, r / 3);
    ctx.lineTo(-r + 2, r / 8);
    ctx.fill();

    // 2. Head Crest Feathers
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.moveTo(-r / 4, -r + 2);
    ctx.lineTo(-r / 2, -r - r / 2);
    ctx.lineTo(0, -r + 4);
    ctx.lineTo(r / 6, -r - r / 2.2);
    ctx.lineTo(r / 3, -r + 4);
    ctx.closePath();
    ctx.fill();

    // 3. Body Circle
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();

    // 4. Cream/Beige Belly Patch
    ctx.beginPath();
    ctx.ellipse(r / 6, r / 4, r * 0.7, r * 0.55, 15 * Math.PI / 180, 0, Math.PI);
    ctx.fillStyle = "rgb(240, 220, 180)";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgb(120, 20, 20)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // 5. Angry Brows (V shape)
    ctx.fillStyle = "rgb(15, 15, 15)";
    ctx.beginPath();
    ctx.moveTo(-r * 0.55, -r * 0.35);
    ctx.lineTo(r * 0.55, -r * 0.35);
    ctx.lineTo(r * 0.05, -r * 0.12);
    ctx.lineTo(-r * 0.05, -r * 0.12);
    ctx.closePath();
    ctx.fill();

    // 6. Eyes with Pupils & Glints
    const eyeR = Math.max(3, Math.floor(r / 4));
    const pupilR = Math.max(1, Math.floor(r / 8));

    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(-r / 4, -r / 8, eyeR, 0, Math.PI * 2);
    ctx.arc(r / 4, -r / 8, eyeR, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgb(20, 20, 20)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.arc(-r / 4 + 1, -r / 8, pupilR, 0, Math.PI * 2);
    ctx.arc(r / 4 - 1, -r / 8, pupilR, 0, Math.PI * 2);
    ctx.fill();

    // Glints
    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(-r / 4 - 1, -r / 8 - 1, Math.max(1, pupilR / 2), 0, Math.PI * 2);
    ctx.arc(r / 4 - 2, -r / 8 - 1, Math.max(1, pupilR / 2), 0, Math.PI * 2);
    ctx.fill();

    // 7. Shaded Split Beak
    ctx.beginPath();
    ctx.moveTo(-r / 4, r / 10);
    ctx.lineTo(r / 4, r / 10);
    ctx.lineTo(0, r * 0.45);
    ctx.closePath();
    ctx.fillStyle = "rgb(255, 180, 0)";
    ctx.fill();
    ctx.strokeStyle = "rgb(180, 100, 0)";
    ctx.stroke();
  }

  drawChuck(ctx, r, col) {
    // 1. Spiky Tail Feathers
    ctx.fillStyle = "rgb(20, 20, 20)";
    ctx.beginPath();
    ctx.moveTo(-r * 0.9, 0);
    ctx.lineTo(-r * 1.6, -r * 0.3);
    ctx.lineTo(-r * 0.8, r * 0.1);
    ctx.fill();

    // 2. Spiky Head Feathers
    ctx.beginPath();
    ctx.moveTo(-r / 6, -r * 0.9);
    ctx.lineTo(-r * 0.7, -r * 1.7);
    ctx.lineTo(r / 8, -r * 1.0);
    ctx.lineTo(-r * 0.4, -r * 1.9);
    ctx.lineTo(r / 3, -r * 0.8);
    ctx.closePath();
    ctx.fill();

    // 3. True Rounded Triangle Polygon Body
    ctx.beginPath();
    ctx.moveTo(0, -r * 1.3);
    ctx.lineTo(r * 1.25, r * 0.95);
    ctx.lineTo(-r * 1.15, r * 0.95);
    ctx.closePath();
    ctx.fillStyle = col;
    ctx.fill();

    // 4. Cream Belly Patch
    ctx.beginPath();
    ctx.moveTo(-r * 0.4, r * 0.3);
    ctx.lineTo(r * 1.1, r * 0.95);
    ctx.lineTo(-r * 0.95, r * 0.95);
    ctx.closePath();
    ctx.fillStyle = "rgb(255, 240, 200)";
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(0, -r * 1.3);
    ctx.lineTo(r * 1.25, r * 0.95);
    ctx.lineTo(-r * 1.15, r * 0.95);
    ctx.closePath();
    ctx.strokeStyle = "rgb(200, 160, 0)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // 5. Angry Brows
    ctx.strokeStyle = "rgb(140, 20, 20)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(-r * 0.5, -r * 0.3);
    ctx.lineTo(-2, -r * 0.48);
    ctx.moveTo(r * 0.5, -r * 0.3);
    ctx.lineTo(2, -r * 0.48);
    ctx.stroke();

    // 6. Eyes & Pupils
    const eyeR = Math.max(3, Math.floor(r / 4));
    const pupilR = Math.max(1, Math.floor(r / 8));

    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(-r / 4, -r / 8, eyeR, 0, Math.PI * 2);
    ctx.arc(r / 4, -r / 8, eyeR, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.arc(-r / 4, -r / 8, pupilR, 0, Math.PI * 2);
    ctx.arc(r / 4, -r / 8, pupilR, 0, Math.PI * 2);
    ctx.fill();

    // 7. Beak
    ctx.beginPath();
    ctx.moveTo(-r / 4, r / 8);
    ctx.lineTo(r / 4, r / 8);
    ctx.lineTo(r * 0.1, r * 0.65);
    ctx.closePath();
    ctx.fillStyle = "rgb(255, 170, 0)";
    ctx.fill();
  }

  drawBomb(ctx, r, col) {
    // 1. Tail Feathers
    ctx.fillStyle = "rgb(20, 20, 20)";
    ctx.beginPath();
    ctx.moveTo(-r, -r / 5);
    ctx.lineTo(-r - r / 2, -r / 3);
    ctx.lineTo(-r + 2, 0);
    ctx.fill();

    // 2. Fuse Line
    ctx.strokeStyle = "rgb(80, 80, 80)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, -r + 2);
    ctx.lineTo(r / 3, -r - r / 2);
    ctx.stroke();

    // Fuse spark
    ctx.beginPath();
    ctx.arc(r / 3, -r - r / 2, 4, 0, Math.PI * 2);
    ctx.fillStyle = "rgb(255, 220, 0)";
    ctx.fill();

    // 3. Body Circle
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();

    // 4. Belly Patch
    ctx.beginPath();
    ctx.ellipse(0, r / 3, r * 0.65, r * 0.45, 0, 0, Math.PI);
    ctx.fillStyle = "rgb(160, 150, 140)";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgb(15, 15, 15)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // 5. Forehead Yellow Spot
    ctx.beginPath();
    ctx.arc(0, -r * 0.5, Math.max(2, Math.floor(r / 6)), 0, Math.PI * 2);
    ctx.fillStyle = "rgb(255, 220, 0)";
    ctx.fill();

    // 6. Red Angry Eyes
    const eyeR = Math.max(3, Math.floor(r / 4));
    const pupilR = Math.max(1, Math.floor(r / 8));

    ctx.fillStyle = "rgb(220, 0, 0)";
    ctx.beginPath();
    ctx.arc(-r / 4, -r / 8, eyeR, 0, Math.PI * 2);
    ctx.arc(r / 4, -r / 8, eyeR, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.arc(-r / 4, -r / 8, pupilR, 0, Math.PI * 2);
    ctx.arc(r / 4, -r / 8, pupilR, 0, Math.PI * 2);
    ctx.fill();

    // 7. Stout Beak
    ctx.beginPath();
    ctx.moveTo(-r / 5, r / 8);
    ctx.lineTo(r / 5, r / 8);
    ctx.lineTo(0, r * 0.45);
    ctx.closePath();
    ctx.fillStyle = "rgb(255, 180, 0)";
    ctx.fill();
  }

  drawBlues(ctx, r, col) {
    // 1. Head Tuft
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.moveTo(-r / 4, -r + 2);
    ctx.lineTo(-r / 3, -r - r / 3);
    ctx.lineTo(0, -r + 2);
    ctx.lineTo(r / 6, -r - r / 3);
    ctx.lineTo(r / 3, -r + 2);
    ctx.closePath();
    ctx.fill();

    // 2. Body Circle
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();

    // 3. Red Patches around Eyes
    ctx.fillStyle = "rgb(200, 80, 50)";
    ctx.beginPath();
    ctx.arc(-r / 4, -r / 8, r * 0.35, 0, Math.PI * 2);
    ctx.arc(r / 4, -r / 8, r * 0.35, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgb(30, 50, 160)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // 4. Eyes & Pupils
    const eyeR = Math.max(3, Math.floor(r * 0.28));
    const pupilR = Math.max(1, Math.floor(r / 7));

    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(-r / 4, -r / 8, eyeR, 0, Math.PI * 2);
    ctx.arc(r / 4, -r / 8, eyeR, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.arc(-r / 4, -r / 8, pupilR, 0, Math.PI * 2);
    ctx.arc(r / 4, -r / 8, pupilR, 0, Math.PI * 2);
    ctx.fill();

    // 5. Beak
    ctx.beginPath();
    ctx.moveTo(-r / 6, r / 8);
    ctx.lineTo(r / 6, r / 8);
    ctx.lineTo(0, r * 0.4);
    ctx.closePath();
    ctx.fillStyle = "rgb(220, 160, 0)";
    ctx.fill();
  }

  drawWhite(ctx, r, col) {
    // 1. Crest Feathers
    ctx.fillStyle = "rgb(20, 20, 20)";
    ctx.beginPath();
    ctx.moveTo(-r / 4, -r * 1.1);
    ctx.lineTo(-r / 2, -r * 1.6);
    ctx.lineTo(0, -r * 1.1);
    ctx.lineTo(0, -r * 1.7);
    ctx.lineTo(r / 4, -r * 1.1);
    ctx.closePath();
    ctx.fill();

    // 2. Oval Body
    ctx.beginPath();
    ctx.ellipse(0, 0, r, Math.floor(r * 1.25), 0, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();

    // 3. Pink Cheeks
    ctx.fillStyle = "rgb(240, 180, 180)";
    ctx.beginPath();
    ctx.arc(-r * 0.55, r / 6, Math.max(2, Math.floor(r / 4)), 0, Math.PI * 2);
    ctx.arc(r * 0.55, r / 6, Math.max(2, Math.floor(r / 4)), 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.ellipse(0, 0, r, Math.floor(r * 1.25), 0, 0, Math.PI * 2);
    ctx.strokeStyle = "rgb(180, 180, 180)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // 4. Eyes & Pupils
    const eyeR = Math.max(2, Math.floor(r / 4));
    const pupilR = Math.max(1, Math.floor(r / 8));

    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(-r / 4, -r / 6, eyeR, 0, Math.PI * 2);
    ctx.arc(r / 4, -r / 6, eyeR, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.arc(-r / 4, -r / 6, pupilR, 0, Math.PI * 2);
    ctx.arc(r / 4, -r / 6, pupilR, 0, Math.PI * 2);
    ctx.fill();

    // 5. Yellow Beak
    ctx.beginPath();
    ctx.moveTo(-r / 4, r / 10);
    ctx.lineTo(r / 4, r / 10);
    ctx.lineTo(0, r * 0.5);
    ctx.closePath();
    ctx.fillStyle = "rgb(255, 180, 0)";
    ctx.fill();
  }
}
