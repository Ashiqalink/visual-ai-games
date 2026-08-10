# Angry Birds OpenCV — Agent Skill & Token Efficiency Guidelines

Refer to [.agents/skills/angry-birds-opencv/SKILL.md](file:///d:/angry%20birds%20using%20opencv/.agents/skills/angry-birds-opencv/SKILL.md) for full skill directives.

---

## ⚡ Token-Efficient Editing Strategy (CSS & Minor Tweaks)

When performing CSS styling, minor UI layout tweaks, or localized parameter adjustments in this project:

1. **Targeted Reading (`view_file` line ranges):**
   - ALWAYS specify `StartLine` and `EndLine` when reading CSS files (`web/src/style.css`), UI JS files (`web/src/ui.js`), or Python UI modules (`ui.py`).
   - Do NOT load entire files into memory when inspecting or modifying specific selectors/functions.

2. **Surgical Edits (`replace_file_content`):**
   - NEVER use `write_to_file` to overwrite full files for minor CSS or code edits.
   - Use `replace_file_content` with precise `StartLine`/`EndLine` targeting only the lines being modified.

3. **Fast Selector Lookup (`grep_search`):**
   - Use `grep_search` to locate exact CSS class names (e.g. `.camera-box`, `#canvas-container`) or variable definitions instead of reading large sections of code.

4. **Concise Chat Responses:**
   - Keep responses focused on line changes and avoid dumping full file outputs in conversational replies.

---

## 🎯 Target Scope Disambiguation Rule

Whenever asking questions, proposing designs, or suggesting code changes, **ALWAYS explicitly state which repository / scope the change applies to:**

- ⚙️ **Game Engine SDK (`visual_ai`)**: Camera pipeline, MediaPipe hand tracking, EMA smoothing, threaded frame queue, pybind11 C++ physics core (`d:\visual ai game engine`).
- 🎮 **The Game (`angry-birds-opencv`)**: Game loop state machine, level layouts, bird/block entities, slingshot rendering, HUD UI (`d:\angry birds using opencv`).

