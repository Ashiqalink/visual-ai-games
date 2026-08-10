/* main.js — Web Entrypoint, camera logic, and double execution loops */

import { HandTracker } from './handTracker.js';
import { Game } from './game.js';

let lastKey = null;
let fps = 0.0;
let fpsSmooth = 60.0;
let prevTime = performance.now();
let highScore = 0;

// Listen to keyboard controls
window.addEventListener("keydown", (e) => {
  lastKey = e.key;
});

document.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById("gameCanvas");
  const ctx = canvas.getContext("2d");
  const video = document.getElementById("webcam");
  const startOverlay = document.getElementById("startOverlay");
  const startButton = document.getElementById("startButton");
  const loadingIndicator = document.getElementById("loadingIndicator");

  const scoreDisplay = document.getElementById("scoreDisplay");
  const highScoreDisplay = document.getElementById("highScoreDisplay");
  const fpsBadge = document.getElementById("fpsBadge");
  const gestureBadge = document.getElementById("gestureBadge");
  const hintDisplay = document.getElementById("hintDisplay");
  const restartBtn = document.getElementById("restartBtn");

  const victoryOverlay = document.getElementById("victoryOverlay");
  const victoryScore = document.getElementById("victoryScore");
  const retryVictoryBtn = document.getElementById("retryVictoryBtn");
  const nextLevelBtn = document.getElementById("nextLevelBtn");

  const lvlBtns = [
    document.getElementById("lvlBtn1"),
    document.getElementById("lvlBtn2"),
    document.getElementById("lvlBtn3")
  ];

  let tracker = null;
  let game = null;

  // Level selector click handlers
  lvlBtns.forEach((btn, idx) => {
    if (!btn) return;
    btn.addEventListener("click", () => {
      if (!game) return;
      game.levelIdx = idx;
      game.resetLevel();
      updateLevelButtons(idx);
    });
  });

  function updateLevelButtons(activeIdx) {
    lvlBtns.forEach((btn, idx) => {
      if (!btn) return;
      if (idx === activeIdx) {
        btn.classList.add("active-lvl");
      } else {
        btn.classList.remove("active-lvl");
      }
    });
  }

  if (restartBtn) {
    restartBtn.addEventListener("click", () => {
      if (game) game.resetLevel();
    });
  }

  if (retryVictoryBtn) {
    retryVictoryBtn.addEventListener("click", () => {
      if (game) game.resetLevel();
      if (victoryOverlay) victoryOverlay.classList.add("hidden");
    });
  }

  if (nextLevelBtn) {
    nextLevelBtn.addEventListener("click", () => {
      if (game) {
        game.levelIdx = (game.levelIdx + 1) % 3;
        game.resetLevel();
        updateLevelButtons(game.levelIdx);
      }
      if (victoryOverlay) victoryOverlay.classList.add("hidden");
    });
  }

  startButton.addEventListener("click", async () => {
    startButton.classList.add("hidden");
    loadingIndicator.classList.remove("hidden");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        },
        audio: false
      });
      video.srcObject = stream;

      tracker = new HandTracker(1280, 720);
      game = new Game(1280, 720);

      tracker.init(() => {
        startOverlay.classList.remove("active");

        const camera = new window.Camera(video, {
          onFrame: async () => {
            await tracker.process(video);
          },
          width: 1280,
          height: 720
        });
        camera.start();

        requestAnimationFrame(renderLoop);
      });

    } catch (err) {
      console.error("[ERROR] Failed to start webcam or initialize model:", err);
      alert("Could not access camera. Please allow webcam permissions and reload the page.");
      startButton.classList.remove("hidden");
      loadingIndicator.classList.add("hidden");
    }
  });

  function renderLoop() {
    // Calculate FPS
    const now = performance.now();
    const dt = (now - prevTime) / 1000;
    prevTime = now;
    if (dt > 0) {
      fps = 1.0 / dt;
    }
    fpsSmooth = 0.9 * fpsSmooth + 0.1 * fps;

    if (tracker.zHistory.length >= 2) {
      game.zDeltaDisplay = tracker.zHistory[0] - tracker.zHistory[tracker.zHistory.length - 1];
    } else {
      game.zDeltaDisplay = 0.0;
    }

    // Run game update step
    const currentKey = lastKey;
    lastKey = null;
    game.update(tracker.gesture, currentKey);

    // Update DOM telemetry
    if (scoreDisplay) scoreDisplay.textContent = game.score;
    if (game.score > highScore) {
      highScore = game.score;
      if (highScoreDisplay) highScoreDisplay.textContent = highScore;
    }
    if (fpsBadge) fpsBadge.textContent = `${Math.round(fpsSmooth)} FPS`;
    if (updateLevelButtons) updateLevelButtons(game.levelIdx);

    // Update gesture badge
    if (gestureBadge && tracker.gesture) {
      if (!tracker.gesture.hand_visible) {
        gestureBadge.textContent = "NO HAND";
        gestureBadge.className = "gesture-badge status-ready";
        gestureBadge.style.backgroundColor = "#7f1d1d";
        gestureBadge.style.color = "#fca5a5";
      } else if (tracker.gesture.is_pinching) {
        gestureBadge.textContent = "PINCH LOCK 🤏";
        gestureBadge.className = "gesture-badge status-ready";
        gestureBadge.style.backgroundColor = "#854d0e";
        gestureBadge.style.color = "#fef08a";
      } else {
        gestureBadge.textContent = "AIMING 👆";
        gestureBadge.className = "gesture-badge status-ready";
        gestureBadge.style.backgroundColor = "#166534";
        gestureBadge.style.color = "#86efac";
      }
    }

    // Hint text update
    if (hintDisplay && game.state) {
      const hints = {
        "SELECTION": "Move hand horizontally to scroll birds | Pinch to select!",
        "ARMED": "Aim index finger to pull slingshot | Release to launch!",
        "FLIGHT": "Bird in flight... watching destruction!"
      };
      hintDisplay.textContent = hints[game.state] || "Pinch or move index finger to play!";
    }

    // Check Victory condition: pigs eliminated or all blocks destroyed
    const pigsRemaining = game.blocks.filter(b => b.active && b.material === "pig").length;
    const activeBlocks = game.blocks.filter(b => b.active).length;
    const isVictorious = (game.blocks.length > 0 && pigsRemaining === 0) || activeBlocks === 0;

    if ((game.state === "DONE" || pigsRemaining === 0) && isVictorious) {
      if (victoryOverlay && victoryOverlay.classList.contains("hidden")) {
        victoryOverlay.classList.remove("hidden");
        if (victoryScore) victoryScore.textContent = game.score;
      }
    }

    // Render Canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    game.draw(ctx);

    // Pointer cursor
    if (tracker.gesture.hand_visible) {
      const [ix, iy] = tracker.gesture.index_pos;
      const [px, py] = tracker.gesture.pinch_pos;

      ctx.save();
      ctx.beginPath();
      ctx.arc(ix, iy, 14, 0, Math.PI * 2);
      ctx.strokeStyle = "#ffcc00";
      ctx.lineWidth = 3;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(ix, iy, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#ffcc00";
      ctx.fill();

      if (tracker.gesture.is_pinching) {
        ctx.beginPath();
        ctx.arc(px, py, 12, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255, 59, 48, 0.75)";
        ctx.fill();
      }
      ctx.restore();
    }

    requestAnimationFrame(renderLoop);
  }
});

