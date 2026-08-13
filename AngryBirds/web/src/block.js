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
  },
  pig: {
    health: 40,
    density: 0.0012,
    fill: "rgb(40, 200, 80)",
    dark: "rgb(20, 120, 40)",
    grain: "rgb(60, 220, 100)"
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

    if (this.material === "pig") {
      const r = w / 2;
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = dark;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Snout
      ctx.beginPath();
      ctx.ellipse(0, r * 0.1, r * 0.45, r * 0.35, 0, 0, Math.PI * 2);
      ctx.fillStyle = "rgb(60, 220, 100)";
      ctx.fill();
      ctx.strokeStyle = dark;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Snout holes
      ctx.fillStyle = "rgb(10, 100, 30)";
      ctx.beginPath();
      ctx.arc(-r * 0.15, r * 0.1, Math.max(1.5, r * 0.08), 0, Math.PI * 2);
      ctx.arc(r * 0.15, r * 0.1, Math.max(1.5, r * 0.08), 0, Math.PI * 2);
      ctx.fill();

      // Eyes & pupils
      const eyeR = Math.max(2, r * 0.22);
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(-r * 0.35, -r * 0.25, eyeR, 0, Math.PI * 2);
      ctx.arc(r * 0.35, -r * 0.25, eyeR, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#000000";
      ctx.beginPath();
      ctx.arc(-r * 0.3, -r * 0.2, Math.max(1, eyeR * 0.4), 0, Math.PI * 2);
      ctx.arc(r * 0.4, -r * 0.2, Math.max(1, eyeR * 0.4), 0, Math.PI * 2);
      ctx.fill();
    } else {
      // Compute missing bite cutouts when damaged (HP < 80%)
      const halfW = w / 2;
      const halfH = h / 2;

      ctx.beginPath();
      if (hpRatio >= 0.75) {
        ctx.rect(-halfW, -halfH, w, h);
      } else {
        const depth = hpRatio >= 0.50 ? 0.22 : (hpRatio >= 0.25 ? 0.32 : 0.42);
        const nw = halfW * depth;
        const nh = halfH * depth;

        // Top-Left
        if (hpRatio < 0.25) {
          ctx.moveTo(-halfW + nw, -halfH);
          ctx.lineTo(-halfW, -halfH + nh);
        } else {
          ctx.moveTo(-halfW, -halfH);
        }

        // Top-Right bite (disappears first at HP < 75%)
        ctx.lineTo(halfW - nw, -halfH);
        ctx.lineTo(halfW - nw * 0.3, -halfH + nh * 0.7);
        ctx.lineTo(halfW, -halfH + nh);

        // Bottom-Right bite
        if (hpRatio < 0.50) {
          ctx.lineTo(halfW, halfH - nh);
          ctx.lineTo(halfW - nw, halfH);
        } else {
          ctx.lineTo(halfW, halfH);
        }

        // Bottom-Left
        if (hpRatio < 0.25) {
          ctx.lineTo(-halfW + nw, halfH);
          ctx.lineTo(-halfW, halfH - nh);
        } else {
          ctx.lineTo(-halfW, halfH);
        }
        ctx.closePath();
      }

      // Draw base fill
      ctx.fillStyle = fill;
      ctx.fill();

      // Draw material grain details
      ctx.strokeStyle = grain;
      ctx.lineWidth = 1;
      if (this.material === "wood") {
        for (const gy of [-h / 4, 0, h / 4]) {
          ctx.beginPath();
          ctx.moveTo(-w / 3, gy);
          ctx.lineTo(w / 3, gy);
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
        ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
        ctx.stroke();
      }

      // PBR Specular highlight edge (top-left bevel)
      if (this.material === "ice") {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(-w / 2, h / 2);
        ctx.lineTo(-w / 2, -h / 2);
        ctx.lineTo(w / 2, -h / 2);
        ctx.stroke();
      } else {
        ctx.strokeStyle = this.material === "wood" ? "rgba(255, 215, 160, 0.6)" : "rgba(230, 230, 240, 0.5)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(-w / 2 + 1, h / 2 - 1);
        ctx.lineTo(-w / 2 + 1, -h / 2 + 1);
        ctx.lineTo(w / 2 - 1, -h / 2 + 1);
        ctx.stroke();
      }

      // Border stroke
      ctx.strokeStyle = dark;
      ctx.lineWidth = 2;
      ctx.strokeRect(-w / 2, -h / 2, w, h);
    }

    // Crack lines overlay
    if (hpRatio < 0.75) {
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
    if (hpRatio >= 0.75) return;

    let darkCol, lightCol, chipCol;
    if (this.material === "ice") {
      darkCol = "rgba(40, 120, 160, 0.8)";
      lightCol = "rgba(230, 245, 255, 0.9)";
      chipCol = "rgba(110, 170, 190, 0.7)";
    } else if (this.material === "stone") {
      darkCol = "rgb(30, 30, 30)";
      lightCol = "rgb(150, 150, 150)";
      chipCol = "rgb(70, 70, 70)";
    } else {
      darkCol = "rgb(40, 20, 10)";
      lightCol = "rgb(190, 140, 90)";
      chipCol = "rgb(90, 50, 25)";
    }

    const drawPath = (path, width) => {
      if (path.length < 2) return;
      // Shadow pass
      ctx.strokeStyle = darkCol;
      ctx.lineWidth = width + 1;
      ctx.beginPath();
      ctx.moveTo(path[0][0], path[0][1]);
      for (let i = 1; i < path.length; i++) ctx.lineTo(path[i][0], path[i][1]);
      ctx.stroke();

      // Highlight pass
      ctx.strokeStyle = lightCol;
      ctx.lineWidth = Math.max(1, width - 1);
      ctx.beginPath();
      ctx.moveTo(path[0][0], path[0][1]);
      for (let i = 1; i < path.length; i++) ctx.lineTo(path[i][0], path[i][1]);
      ctx.stroke();
    };

    // ── Tier 1: Surface cracks (< 75% HP) ──
    if (this.material === "wood") {
      drawPath([[-w*0.35, -h*0.2], [-w*0.1, -h*0.18], [w*0.15, -h*0.22], [w*0.38, -h*0.2]], 1);
      drawPath([[-w*0.25, h*0.15], [0, h*0.18], [w*0.2, h*0.14]], 1);
    } else if (this.material === "stone") {
      drawPath([[-w*0.3, -h*0.35], [-w*0.15, -h*0.1], [w*0.05, h*0.1], [w*0.25, h*0.3]], 1.5);
      drawPath([[-w*0.1, -h*0.1], [w*0.15, -h*0.25]], 1);
    } else {
      drawPath([[0, 0], [-w*0.3, -h*0.3]], 1);
      drawPath([[0, 0], [w*0.35, -h*0.2]], 1);
      drawPath([[0, 0], [-w*0.2, h*0.35]], 1);
    }

    // ── Tier 2: Corner chip & deep fracture (< 60% HP) ──
    if (hpRatio < 0.6) {
      const cSize = Math.max(3, Math.min(w, h) * 0.22);
      ctx.fillStyle = chipCol;
      ctx.beginPath();
      ctx.moveTo(-w*0.5, -h*0.5);
      ctx.lineTo(-w*0.5 + cSize, -h*0.5);
      ctx.lineTo(-w*0.5, -h*0.5 + cSize * 0.8);
      ctx.closePath();
      ctx.fill();

      drawPath([[w*0.3, -h*0.4], [w*0.1, -h*0.05], [-w*0.15, h*0.2], [-w*0.35, h*0.38]], 2);
    }

    // ── Tier 3: Edge Notch & Shatter (< 30% HP) ──
    if (hpRatio < 0.3) {
      const brSize = Math.max(4, Math.min(w, h) * 0.28);
      ctx.fillStyle = chipCol;
      ctx.beginPath();
      ctx.moveTo(w*0.5, h*0.5);
      ctx.lineTo(w*0.5 - brSize, h*0.5);
      ctx.lineTo(w*0.5, h*0.5 - brSize * 0.85);
      ctx.closePath();
      ctx.fill();

      drawPath([[-w*0.45, 0], [w*0.45, 0]], 1.5);
      drawPath([[0, -h*0.45], [0, h*0.45]], 1.5);
    }
  }
}
