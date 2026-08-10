/* block.js — Block class and material rendering */

import Matter from 'matter-js';
import { MIN_DAMAGE_VEL } from './physics.js';

export const MATERIALS = {
  wood: {
    health: 60,
    density: 0.001,
    fill: "rgb(172, 110, 60)",
    dark: "rgb(100, 60, 30)",
    grain: "rgb(195, 140, 90)"
  },
  stone: {
    health: 120,
    density: 0.003,
    fill: "rgb(128, 128, 128)",
    dark: "rgb(80, 80, 80)",
    grain: "rgb(160, 160, 160)"
  },
  ice: {
    health: 30,
    density: 0.0005,
    fill: "rgba(160, 220, 240, 0.6)",
    dark: "rgb(120, 180, 200)",
    grain: "rgba(255, 255, 255, 0.4)"
  }
};

export class Block {
  constructor(world, x, y, w = 60, h = 60, material = "wood") {
    this.w = parseFloat(w);
    this.h = parseFloat(h);
    this.material = material;

    const mat = MATERIALS[material];
    this.max_health = mat.health;
    this.health = mat.health;

    this.active = true;

    // Matter.js Body
    this.body = Matter.Bodies.rectangle(x + this.w / 2, y + this.h / 2, this.w, this.h, {
      density: mat.density,
      friction: 0.5,
      restitution: 0.2,
      label: 'block'
    });
    
    // Link class instance to body for collision callbacks
    this.body.plugin = { block: this };

    Matter.World.add(world, this.body);
    this.world = world;
  }

  takeDamage(amount) {
    this.health -= amount;
    if (this.health <= 0 && this.active) {
      this.active = false;
      Matter.World.remove(this.world, this.body);
    }
  }

  draw(ctx) {
    if (!this.active) return;

    const pos = this.body.position;
    const angle = this.body.angle;
    const w = this.w;
    const h = this.h;

    const mat = MATERIALS[this.material];
    const fill = mat.fill;
    const dark = mat.dark;
    const grain = mat.grain;

    const hpRatio = this.health / this.max_health;

    // Draw rotated block on canvas
    ctx.save();
    ctx.translate(pos.x, pos.y);
    ctx.rotate(angle);

    // Draw base fill
    ctx.fillStyle = fill;
    ctx.fillRect(-w / 2, -h / 2, w, h);

    // Draw material grain details
    ctx.strokeStyle = grain;
    ctx.lineWidth = 1;
    if (this.material === "wood") {
      for (const gy of [-h / 4, 0, h / 4]) {
        ctx.beginPath();
        ctx.moveTo(-w / 2, gy);
        ctx.lineTo(w / 2, gy);
        ctx.stroke();
      }
    } else if (this.material === "stone") {
      for (const gy of [-h / 3, h / 6]) {
        ctx.beginPath();
        ctx.moveTo(-w / 3, gy);
        ctx.lineTo(w / 3, gy);
        ctx.stroke();
      }
    } else if (this.material === "ice") {
      const shineY = -h / 4;
      ctx.beginPath();
      ctx.moveTo(-w / 3, shineY);
      ctx.lineTo(w / 4, shineY);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
      ctx.stroke();
    }

    // Border stroke
    ctx.strokeStyle = dark;
    ctx.lineWidth = 2;
    ctx.strokeRect(-w / 2, -h / 2, w, h);

    // Crack lines overlay
    if (hpRatio < 0.7) {
      this.drawCracks(ctx, w, h, hpRatio);
    }

    ctx.restore();

    // Health bar (drawn on top of the block, not rotated)
    if (hpRatio < 1.0) {
      const barW = w;
      const barH = 5;
      const bx = pos.x - w / 2;
      const by = pos.y - h / 2 - 10;

      ctx.fillStyle = "rgb(40, 40, 40)";
      ctx.fillRect(bx, by, barW, barH);

      const clamped = Math.max(0.0, hpRatio);
      const hpCol = `rgb(${Math.floor(200 * (1 - clamped))}, ${Math.floor(200 * clamped)}, 0)`;
      ctx.fillStyle = hpCol;
      ctx.fillRect(bx, by, barW * clamped, barH);
    }
  }

  drawCracks(ctx, w, h, hpRatio) {
    const crackCol = this.material === "ice" ? "rgba(100, 160, 180, 0.8)" : "rgb(30, 30, 30)";
    ctx.strokeStyle = crackCol;

    // Light cracks
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(-w * 0.3, -h * 0.2);
    ctx.lineTo(0, h * 0.1);
    ctx.lineTo(w * 0.2, -h * 0.1);
    ctx.stroke();

    // Heavy cracks at < 40% health
    if (hpRatio < 0.4) {
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(-w * 0.1, -h * 0.4);
      ctx.lineTo(w * 0.15, 0);
      ctx.lineTo(-w * 0.05, h * 0.3);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(w * 0.25, -h * 0.3);
      ctx.lineTo(w * 0.1, h * 0.2);
      ctx.stroke();
    }
  }
}
