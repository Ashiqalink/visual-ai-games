# Workspace Rules for Angry Birds OpenCV & Visual AI

## 🎯 Scope Disambiguation Rule

Whenever asking questions, asking for clarification, proposing architectural changes, or suggesting code edits, **ALWAYS explicitly label whether the change belongs to:**

1. ⚙️ **Game Engine SDK (`visual_ai`)** — `d:\visual ai game engine`
   *(Camera feed capture, MediaPipe tracking pipeline, EMA smoothing, frame queues, C++ pybind11 physics core)*

2. 🎮 **The Game (`angry-birds-opencv`)** — `d:\angry birds using opencv`
   *(Game state machine, bird/block entities, slingshot rendering, level layouts, HUD UI, web frontend)*
   *(Design philosophy: This game is supposed to be a laid back and enjoy game, not a fast paced game.)*


---

## ⚡ Token Efficiency Rules

- Use `replace_file_content` with tight `StartLine`/`EndLine` for minor/CSS edits.
- Use `view_file` with explicit line ranges when reading UI/CSS files.
- Use `grep_search` to target specific selectors or variables directly.
