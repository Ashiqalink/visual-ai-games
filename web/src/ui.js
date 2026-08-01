/* ui.js — Canvas HUD panels, carousel selection, score cards, and text helpers */

import { Bird } from './bird.js';

const HUD_TEXT = "rgb(240, 240, 240)";
const LEVEL_NAMES = ["Easy", "Medium", "Hard"];

// Text drawing helper with a clean dark drop-shadow for high legibility.
export function drawText(ctx, txt, x, y, size = 18, col = HUD_TEXT, align = "left", bold = true) {
  ctx.save();
  ctx.font = `${bold ? '600' : '400'} ${size}px 'Outfit', sans-serif`;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";

  // Shadow
  ctx.fillStyle = "rgba(10, 11, 15, 0.85)";
  ctx.fillText(txt, x + 1.5, y + 1.5);

  // Text fill
  ctx.fillStyle = col;
  ctx.fillText(txt, x, y);
  ctx.restore();
}

/** Draw background sky shade and base lawn.*/
export function drawGround(ctx, floorY = 660) {
  const w = 1280;
  const h = 720;

  // Sky tint
  ctx.fillStyle = "rgba(200, 160, 80, 0.12)";
  ctx.fillRect(0, 0, w, floorY);

  // Grass soil
  ctx.fillStyle = "rgb(40, 130, 60)";
  ctx.fillRect(0, floorY, w, h - floorY);

  // Surface grass blade indicator
  ctx.fillStyle = "rgb(30, 100, 45)";
  ctx.fillRect(0, floorY, w, 6);
}

/** Render selection carousel on top center. */
export function drawCarousel(ctx, birdQueue, selectedIdx) {
  const cx = 640;
  const cy = 90;
  const spacing = 120;
  const total = birdQueue.length;

  if (total === 0) return;

  const panelW = spacing * total + 80;
  const panelH = 140;
  const px = cx - panelW / 2;

  // Glassmorphic panel backing
  ctx.save();
  ctx.fillStyle = "rgba(15, 18, 30, 0.65)";
  ctx.fillRect(px, cy - 60, panelW, panelH);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(px, cy - 60, panelW, panelH);
  ctx.restore();

  for (let i = 0; i < total; i++) {
    const kind = birdQueue[i];
    const bx = cx + (i - (total - 1) / 2) * spacing;
    const isSel = (i === selectedIdx);
    const scale = isSel ? 1.5 : 0.9;

    if (isSel) {
      // Aim Selection Ring
      ctx.beginPath();
      ctx.arc(bx, cy, 38, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(0, 255, 200, 0.7)";
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(bx, cy, 32, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(0, 200, 255, 0.4)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Draw carousel item
    const tempBird = new Bird(null, kind, bx, cy);
    tempBird.draw(ctx, scale);

    const labelCol = isSel ? "#00ffc8" : "#9ca3af";
    drawText(ctx, kind, bx, cy + 42, isSel ? 14 : 11, labelCol, "center");
  }

  drawText(ctx, "PINCH or Z-PUSH on bird to select", cx, cy + 62, 13, "#b4dcff", "center", false);
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
    const rDot = Math.max(2, Math.floor(5 * alpha));

    ctx.beginPath();
    ctx.arc(x, y, rDot, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(240, 240, 245, ${alpha * 0.85})`;
    ctx.fill();
  }
  ctx.restore();
}

// Floating scores overlay updater.
export function drawScorePopups(ctx, popups) {
  for (const p of popups) {
    const alpha = p.alpha;
    const col = `rgba(0, 255, 200, ${alpha})`;
    const size = Math.floor((12 + 10 * alpha));
    drawText(ctx, `+${p.value}`, p.x, p.y, size, col, "center");
  }
}

/** Main HUD panel containing score labels, level numbers, modes, and ToF depth debugger. */
export function drawHUD(ctx, state, birdsLeft, clickMode = "PINCH", zDebug = 0.0, score = 0, levelIdx = 0, tofActive = false, tofZM = 0.0, depthSource = "RGB MediaPipe Estimate") {
  const w = 1280;
  const h = 720;

  // ── 1. State Label ───────────────────────────────────────────────────────
  const stateCols = {
    "SELECTION": "#00d2ff",
    "ARMED": "#00ffc8",
    "FLIGHT": "#ff3366"
  };
  const sCol = stateCols[state] || HUD_TEXT;
  drawText(ctx, `State: ${state}`, 20, 36, 17, sCol);

  // ── 2. Level and Score Indicators ─────────────────────────────────────────
  const lvlName = LEVEL_NAMES[levelIdx] || "?";
  drawText(ctx, `Level ${levelIdx + 1}: ${lvlName}`, w / 2, 28, 18, "#00d2ff", "center");
  drawText(ctx, `Score: ${score}`, w / 2, 54, 21, "#00ffc8", "center");

  // ── 3. Queue Drawer (Bottom Left) ─────────────────────────────────────────
  drawText(ctx, "Birds Remaining:", 20, h - 55, 14, "#9ca3af");
  const qx = 40;
  const qy = h - 30;
  for (let i = 0; i < birdsLeft.length; i++) {
    const bkind = birdsLeft[i];
    const bx = qx + i * 45;
    const mini = new Bird(null, bkind, bx, qy);
    mini.draw(ctx, 0.6);
  }

  // ── 4. Gesture mode & ToF Debugger (Top Right) ───────────────────────────
  const modeCol = clickMode === "Z-PUSH" ? "#00ffc8" : "#ffa800";
  drawText(ctx, `Gesture: ${clickMode}`, w - 220, 24, 13, modeCol);

  // ToF Sensor Badge
  const tofBadgeCol = tofActive ? "#00ffc8" : "#ffb000";
  const tofBadgeTxt = tofActive ? `ToF: ACTIVE (${tofZM.toFixed(2)}m)` : "ToF: INACTIVE (RGB Est)";
  drawText(ctx, tofBadgeTxt, w - 220, 42, 12, tofBadgeCol);

  // Z-Push slider debugger
  const barX = w - 220;
  const barY = 56;
  const barW = 180;
  const barH = 8;
  const Z_THRESHOLD = 0.025;
  const ratio = Math.min(1.0, Math.abs(zDebug) / Z_THRESHOLD);

  ctx.save();
  ctx.fillStyle = "rgba(255, 255, 255, 0.1)";
  ctx.fillRect(barX, barY, barW, barH);

  const barCol = ratio < 0.8 ? "rgb(0, 200, 100)" : "rgb(255, 80, 0)";
  ctx.fillStyle = barCol;
  ctx.fillRect(barX, barY, barW * ratio, barH);
  ctx.restore();

  drawText(ctx, `Z-push depth | ${depthSource}`, barX, barY + barH + 12, 10, "#9ca3af", "left", false);

  // ── 5. Hints ─────────────────────────────────────────────────────────────
  drawText(ctx, "1/2/3 = Level  |  R = Restart", w / 2, h - 20, 12, "#9ca3af", "center", false);

  const hints = {
    "SELECTION": "Move hand horizontally to scroll  |  Pinch/Z-Push to choose",
    "ARMED": "Aim index finger  |  Extend index only  |  Touch border to fire",
    "FLIGHT": "Bird in flight... waiting for impact"
  };
  const hint = hints[state] || "";
  drawText(ctx, hint, w - 20, h - 25, 13, "#d1d5db", "right", false);
}

// End level statistics popup dashboard overlay.
export function drawDoneOverlay(ctx, score = 0, bonus = 0, blocksDestroyed = 0, levelIdx = 0) {
  const w = 1280;
  const h = 720;

  // Dark overlay mask
  ctx.save();
  ctx.fillStyle = "rgba(10, 12, 22, 0.78)";
  ctx.fillRect(0, 0, w, h);
  ctx.restore();

  const lvlName = LEVEL_NAMES[levelIdx] || "?";
  drawText(ctx, `LEVEL ${levelIdx + 1}: ${lvlName} COMPLETE!`, w / 2, h / 2 - 80, 32, "#00d2ff", "center");

  drawText(ctx, `Blocks Shattered: ${blocksDestroyed}`, w / 2, h / 2 - 20, 18, "#d1d5db", "center");
  drawText(ctx, `Base Score: ${score}`, w / 2, h / 2 + 15, 18, "#d1d5db", "center");

  if (bonus > 0) {
    drawText(ctx, `Unused Bird Bonus: +${bonus}`, w / 2, h / 2 + 50, 18, "#00ffc8", "center");
  }

  const total = score + bonus;
  drawText(ctx, `TOTAL SCORE: ${total}`, w / 2, h / 2 + 105, 30, "#00ffc8", "center");

  drawText(ctx, "Press  R  to Restart  |  1/2/3 to switch Level", w / 2, h / 2 + 160, 14, "#9ca3af", "center", false);
}
