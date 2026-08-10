/* main.js — Web Entrypoint, camera logic, and double execution loops */

import { HandTracker } from './handTracker.js';
import { Game } from './game.js';

let lastKey = null;
let fps = 0.0;
let fpsSmooth = 60.0;
let prevTime = performance.now();

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

  let tracker = null;
  let game = null;

  startButton.addEventListener("click", async () => {
    // Hide startup button, show loading spinner
    startButton.classList.add("hidden");
    loadingIndicator.classList.remove("hidden");

    try {
      // 1. Request Webcam Permission
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        },
        audio: false
      });
      video.srcObject = stream;

      // 2. Initialize modules
      tracker = new HandTracker(1280, 720);
      game = new Game(1280, 720);

      tracker.init(() => {
        // MediaPipe initialized callback: Start tracking and rendering
        startOverlay.classList.remove("active");

        // Set up MediaPipe camera tracking loop
        const camera = new window.Camera(video, {
          onFrame: async () => {
            await tracker.process(video);
          },
          width: 1280,
          height: 720
        });
        camera.start();

        // Start drawing canvas frames
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
    // ── 1. Calculate FPS ─────────────────────────────────────────────────────
    const now = performance.now();
    const dt = (now - prevTime) / 1000;
    prevTime = now;
    if (dt > 0) {
      fps = 1.0 / dt;
    }
    fpsSmooth = 0.9 * fpsSmooth + 0.1 * fps; // EMA smoothing

    // ── 2. Pass Z-delta to HUD display ──────────────────────────────────────
    if (tracker.zHistory.length >= 2) {
      game.zDeltaDisplay = tracker.zHistory[0] - tracker.zHistory[tracker.zHistory.length - 1];
    } else {
      game.zDeltaDisplay = 0.0;
    }

    // ── 3. Run game updating rules ──────────────────────────────────────────
    const currentKey = lastKey;
    lastKey = null; // consume key
    game.update(tracker.gesture, currentKey);

    // ── 4. Render Graphics onto Canvas ────────────────────────────────────────
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    game.draw(ctx);

    // ── 5. Overlay index pointer cursor rings ──────────────────────────────
    if (tracker.gesture.hand_visible) {
      const [ix, iy] = tracker.gesture.index_pos;
      const [px, py] = tracker.gesture.pinch_pos;

      // Outer cursor pointer ring
      ctx.save();
      ctx.beginPath();
      ctx.arc(ix, iy, 12, 0, Math.PI * 2);
      ctx.strokeStyle = "rgb(0, 255, 200)";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Inner solid point
      ctx.beginPath();
      ctx.arc(ix, iy, 4, 0, Math.PI * 2);
      ctx.fillStyle = "rgb(0, 255, 200)";
      ctx.fill();

      // Pinch indicator
      if (tracker.gesture.is_pinching) {
        ctx.beginPath();
        ctx.arc(px, py, 10, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0, 200, 255, 0.75)";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(px, py, 14, 0, Math.PI * 2);
        ctx.strokeStyle = "rgb(0, 200, 255)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Click event flash ring
      if (tracker.gesture.click_just_fired) {
        ctx.beginPath();
        ctx.arc(ix, iy, 28, 0, Math.PI * 2);
        ctx.strokeStyle = "rgb(255, 80, 0)";
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.font = "bold 16px 'Outfit', sans-serif";
        ctx.fillStyle = "rgb(255, 80, 0)";
        ctx.textAlign = "left";
        ctx.fillText("Z-CLICK!", ix + 20, iy - 20);
      }
      ctx.restore();
    }

    // ── 6. Display Framerate FPS text ─────────────────────────────────────────
    ctx.save();
    ctx.font = "600 13px 'Outfit', sans-serif";
    const fpsText = `FPS: ${Math.round(fpsSmooth)}`;
    
    // shadow
    ctx.fillStyle = "rgba(10, 11, 15, 0.85)";
    ctx.fillText(fpsText, 22, canvas.height - 8);

    // text
    ctx.fillStyle = "rgb(0, 255, 100)";
    ctx.fillText(fpsText, 20, canvas.height - 10);
    ctx.restore();

    // Call next frame loop
    requestAnimationFrame(renderLoop);
  }
});
