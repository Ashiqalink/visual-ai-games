/* slingshot.js — Slingshot structures and elastic rubber bands rendering */

export const SLING_X = 400;
export const SLING_Y = 550;

export const FORK_LEFT  = [SLING_X - 28, SLING_Y - 50];
export const FORK_RIGHT = [SLING_X + 28, SLING_Y - 50];
export const HANDLE_BOT = [SLING_X, SLING_Y + 100];

const WOOD_COL = "rgb(160, 100, 30)";
const WOOD_DARK = "rgb(100, 60, 15)";
const ELASTIC_COL_NEAR = { r: 255, g: 140, b: 0 }; // orange
const ELASTIC_COL_FAR = { r: 220, g: 40, b: 0 };   // red when stretched

const SNAPBACK_DURATION = 8;
let snapbackFrames = 0;
let snapbackOrigin = [SLING_X, SLING_Y];

export function triggerSnapback(birdX, birdY) {
  snapbackFrames = SNAPBACK_DURATION;
  snapbackOrigin = [Math.floor(birdX), Math.floor(birdY)];
}

function getCatenaryPoints(p1, p2, sag, nPts = 10) {
  const points = [];
  for (let i = 0; i <= nPts; i++) {
    const t = i / nPts;
    const x = p1[0] + (p2[0] - p1[0]) * t;
    let y = p1[1] + (p2[1] - p1[1]) * t;
    const sagAmount = sag * 4 * t * (1 - t);
    y += sagAmount;
    points.push([Math.floor(x), Math.floor(y)]);
  }
  return points;
}

function drawBand(ctx, pts, colorStr, thickness) {
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) {
    ctx.lineTo(pts[i][0], pts[i][1]);
  }
  ctx.strokeStyle = colorStr;
  ctx.lineWidth = thickness;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke();
}

export function draw(ctx, birdPos = null, pullDist = 0.0) {
  // Elastic color and thickness based on stretch
  const t = Math.min(1.0, pullDist / 150.0);
  const r = Math.floor(ELASTIC_COL_FAR.r + t * (ELASTIC_COL_NEAR.r - ELASTIC_COL_FAR.r));
  const g = Math.floor(ELASTIC_COL_FAR.g + t * (ELASTIC_COL_NEAR.g - ELASTIC_COL_FAR.g));
  const b = Math.floor(ELASTIC_COL_FAR.b + t * (ELASTIC_COL_NEAR.b - ELASTIC_COL_FAR.b));
  const eCol = `rgb(${r}, ${g}, ${b})`;
  const thickness = Math.floor(2 + t * 4); // 2px -> 6px stretched

  // Less sag when stretched
  const sag = Math.max(2, Math.floor(20 * (1 - t)));

  // ── Back Elastic Band (drawn behind the bird) ───────────────────────────
  if (birdPos !== null) {
    const bx = Math.floor(birdPos[0]);
    const by = Math.floor(birdPos[1]);
    const backPts = getCatenaryPoints(FORK_LEFT, [bx, by], sag);
    drawBand(ctx, backPts, eCol, thickness);
  } else if (snapbackFrames > 0) {
    // Snapback vibration
    snapbackFrames--;
    const progress = snapbackFrames / SNAPBACK_DURATION;
    const vibAmp = Math.floor(8 * progress * Math.sin(progress * Math.PI * 4));
    const midX = SLING_X;
    const midY = SLING_Y + vibAmp;

    const snapPtsL = getCatenaryPoints(FORK_LEFT, [midX, midY], 5);
    const snapPtsR = getCatenaryPoints(FORK_RIGHT, [midX, midY], 5);
    const snapCol = "rgb(200, 100, 0)";

    drawBand(ctx, snapPtsL, snapCol, 3);
    drawBand(ctx, snapPtsR, snapCol, 3);
  }

  // ── Slingshot Handle ────────────────────────────────────────────────────
  ctx.beginPath();
  ctx.moveTo(SLING_X, SLING_Y);
  ctx.lineTo(HANDLE_BOT[0], HANDLE_BOT[1]);
  ctx.strokeStyle = WOOD_COL;
  ctx.lineWidth = 14;
  ctx.lineCap = "round";
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(SLING_X, SLING_Y);
  ctx.lineTo(HANDLE_BOT[0], HANDLE_BOT[1]);
  ctx.strokeStyle = WOOD_DARK;
  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  ctx.stroke();

  // ── Slingshot Fork Arms ──────────────────────────────────────────────────
  // Left arm
  ctx.beginPath();
  ctx.moveTo(SLING_X, SLING_Y);
  ctx.lineTo(FORK_LEFT[0], FORK_LEFT[1]);
  ctx.strokeStyle = WOOD_COL;
  ctx.lineWidth = 12;
  ctx.lineCap = "round";
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(SLING_X, SLING_Y);
  ctx.lineTo(FORK_LEFT[0], FORK_LEFT[1]);
  ctx.strokeStyle = WOOD_DARK;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.stroke();

  // Right arm
  ctx.beginPath();
  ctx.moveTo(SLING_X, SLING_Y);
  ctx.lineTo(FORK_RIGHT[0], FORK_RIGHT[1]);
  ctx.strokeStyle = WOOD_COL;
  ctx.lineWidth = 12;
  ctx.lineCap = "round";
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(SLING_X, SLING_Y);
  ctx.lineTo(FORK_RIGHT[0], FORK_RIGHT[1]);
  ctx.strokeStyle = WOOD_DARK;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.stroke();

  // Fork tips circles
  ctx.beginPath();
  ctx.arc(FORK_LEFT[0], FORK_LEFT[1], 6, 0, Math.PI * 2);
  ctx.fillStyle = WOOD_COL;
  ctx.fill();

  ctx.beginPath();
  ctx.arc(FORK_RIGHT[0], FORK_RIGHT[1], 6, 0, Math.PI * 2);
  ctx.fillStyle = WOOD_COL;
  ctx.fill();

  // ── Front Elastic Band (drawn on top of the bird) ──────────────────────
  if (birdPos !== null) {
    const bx = Math.floor(birdPos[0]);
    const by = Math.floor(birdPos[1]);
    const frontPts = getCatenaryPoints(FORK_RIGHT, [bx, by], sag);
    drawBand(ctx, frontPts, eCol, thickness);
  }
}
