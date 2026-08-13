/* ui.js — Canvas HUD panels, carousel selection, score cards, and text helpers */

import { Bird } from './bird.js';
import { Z_CLICK_THRESHOLD_M } from './physics.js';

const HUD_TEXT = "#ffffff";
const LEVEL_NAMES = ["Easy", "Medium", "Hard"];
const STATE_HINTS = {
  "SELECTION": "Move hand horizontally to scroll  |  Pinch to select bird",
  "ARMED": "Aim index finger to pull slingshot  |  Release to launch",
  "FLIGHT": "Bird in flight... watching destruction!"
};

// Text drawing helper with a bold cartoon drop-shadow outline for maximum legibility.
export function drawText(ctx, txt, x, y, size = 18, col = HUD_TEXT, align = "left", bold = true) {
  ctx.save();
  ctx.font = `${bold ? '700' : '500'} ${size}px 'Fredoka', 'Outfit', sans-serif`;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";

  // Heavy cartoon shadow
  ctx.fillStyle = "#2d1906";
  ctx.fillText(txt, x + 2, y + 2);

  // Text fill
  ctx.fillStyle = col;
  ctx.fillText(txt, x, y);
  ctx.restore();
}

/** Draw sunny blue sky, rolling hills, and vibrant grass lawn.*/
export function drawGround(ctx, floorY = 660) {
  const w = 1280;
  const h = 720;

  // Sunny sky multi-stop gradient
  const skyGrad = ctx.createLinearGradient(0, 0, 0, floorY);
  skyGrad.addColorStop(0, "#4ab3ff");
  skyGrad.addColorStop(0.7, "#85d7ff");
  skyGrad.addColorStop(1, "#cbeeff");
  ctx.fillStyle = skyGrad;
  ctx.fillRect(0, 0, w, floorY);

  // Distant green rolling hill arches
  ctx.save();
  ctx.fillStyle = "#5cba47";
  ctx.beginPath();
  ctx.arc(300, floorY + 120, 420, Math.PI, 0);
  ctx.fill();

  ctx.fillStyle = "#4caf50";
  ctx.beginPath();
  ctx.arc(950, floorY + 150, 480, Math.PI, 0);
  ctx.fill();
  ctx.restore();

  // Grass soil floor
  ctx.fillStyle = "#5c3a21"; // Soil brown
  ctx.fillRect(0, floorY, w, h - floorY);

  // Grass blade surface strip
  ctx.fillStyle = "#388e3c"; // Deep grass green
  ctx.fillRect(0, floorY, w, 14);

  ctx.fillStyle = "#66bb6a"; // Top bright grass highlight
  ctx.fillRect(0, floorY, w, 4);
}

/** Render wooden bird selection carousel panel. */
export function drawCarousel(ctx, birdQueue, selectedIdx) {
  const cx = 640;
  const cy = 90;
  const spacing = 120;
  const total = birdQueue.length;

  if (total === 0) return;

  const panelW = spacing * total + 80;
  const panelH = 135;
  const px = cx - panelW / 2;

  // Wooden plank panel backing
  ctx.save();
  const woodGrad = ctx.createLinearGradient(px, cy - 60, px, cy + panelH - 60);
  woodGrad.addColorStop(0, "#c78d4e");
  woodGrad.addColorStop(1, "#8d5b4c");
  ctx.fillStyle = woodGrad;
  ctx.fillRect(px, cy - 60, panelW, panelH);
  ctx.strokeStyle = "#4d2d0c";
  ctx.lineWidth = 4;
  ctx.strokeRect(px, cy - 60, panelW, panelH);

  // Inner inset highlight
  ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
  ctx.lineWidth = 2;
  ctx.strokeRect(px + 3, cy - 57, panelW - 6, panelH - 6);
  ctx.restore();

  for (let i = 0; i < total; i++) {
    const kind = birdQueue[i];
    const bx = cx + (i - (total - 1) / 2) * spacing;
    const isSel = (i === selectedIdx);
    const scale = isSel ? 1.4 : 0.85;

    if (isSel) {
      // Golden selector halo
      ctx.save();
      ctx.beginPath();
      ctx.arc(bx, cy, 38, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 204, 0, 0.25)";
      ctx.fill();
      ctx.strokeStyle = "#ffcc00";
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.restore();
    }

    // Draw carousel bird
    const tempBird = new Bird(null, kind, bx, cy);
    tempBird.draw(ctx, scale);

    const labelCol = isSel ? "#ffcc00" : "#d1d5db";
    drawText(ctx, kind, bx, cy + 42, isSel ? 14 : 11, labelCol, "center");
  }

  drawText(ctx, "3-FINGER PINCH TO SELECT BIRD", cx, cy + 62, 12, "#fff", "center", true);
}

// Predict and draw flight trajectory curve dots.
export function drawTrajectory(ctx, startX, startY, vx, vy, gravity = 0.45, nDots = 28) {
  let x = startX;
  let y = startY;
  let cvx = vx;
  let cvy = vy;

  ctx.save();
  for (let i = 0; i < nDots; i++) {
    cvy += gravity;
    x += cvx;
    y += cvy;

    const alpha = 1.0 - i / nDots;
    const rDot = Math.max(3, Math.floor(6 * alpha));

    ctx.beginPath();
    ctx.arc(x, y, rDot, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.9})`;
    ctx.fill();
    ctx.strokeStyle = `rgba(61, 36, 18, ${alpha * 0.5})`;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  ctx.restore();
}

// Floating scores overlay updater.
export function drawScorePopups(ctx, popups) {
  for (const p of popups) {
    const alpha = p.alpha;
    const col = `rgba(255, 204, 0, ${alpha})`;
    const size = Math.floor((14 + 12 * alpha));
    drawText(ctx, `+${p.value}`, p.x, p.y, size, col, "center");
  }
}

/** Main HUD in-canvas fallback drawer (complements DOM overlay). */
export function drawHUD(ctx, state, birdsLeft, clickMode = "PINCH", zDebug = 0.0, score = 0, levelIdx = 0) {
  const h = 720;

  // Render Queue Drawer (Bottom Left)
  drawText(ctx, "BIRDS:", 24, h - 55, 12, "#fde68a");
  const qx = 45;
  const qy = h - 30;
  for (let i = 0; i < birdsLeft.length; i++) {
    const bkind = birdsLeft[i];
    const bx = qx + i * 40;
    const mini = new Bird(null, bkind, bx, qy);
    mini.draw(ctx, 0.55);
  }
}

// End level statistics popup overlay (when completed in canvas mode).
export function drawDoneOverlay(ctx, score = 0, bonus = 0, blocksDestroyed = 0, levelIdx = 0) {
  const w = 1280;
  const h = 720;

  ctx.save();
  ctx.fillStyle = "rgba(10, 14, 23, 0.78)";
  ctx.fillRect(0, 0, w, h);
  ctx.restore();

  const lvlName = LEVEL_NAMES[levelIdx] || "?";
  drawText(ctx, `LEVEL ${levelIdx + 1}: ${lvlName} CLEARED! 🎉`, w / 2, h / 2 - 80, 32, "#ffcc00", "center");
  drawText(ctx, `Blocks Shattered: ${blocksDestroyed}`, w / 2, h / 2 - 20, 18, "#ffffff", "center");
  drawText(ctx, `Base Score: ${score}`, w / 2, h / 2 + 15, 18, "#ffffff", "center");

  if (bonus > 0) {
    drawText(ctx, `Unused Bird Bonus: +${bonus}`, w / 2, h / 2 + 50, 18, "#4caf50", "center");
  }

  const total = score + bonus;
  drawText(ctx, `TOTAL SCORE: ${total}`, w / 2, h / 2 + 105, 30, "#00ffc8", "center");

  drawText(ctx, "Press  R  to Restart  |  1/2/3 to switch Level", w / 2, h / 2 + 160, 14, "#9ca3af", "center", false);
}
