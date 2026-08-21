"""
instructions.py — the how-to-play card every game in this repo opens with.

Why this exists
---------------
These are gesture games. A keyboard game can be poked at until it gives its
controls up, but a webcam game gives no hint that the hand should be a fist
rather than a point, or that depth matters and position does not — a player who
does not already know simply waves at a window where nothing happens. So every
game opens on a card that says what it is for and what to do with your hands,
and can bring it back with `H` once play has started.

Deliberately standalone: this imports cv2 and numpy and nothing else from this
repo. `labkit` renders its card through here, and so do Sling, flappy and
punchy — which predate labkit and are no part of the lab harness. One card, no
coupling between the two halves of the repo.

Scope: game side, and purely cosmetic. Nothing here touches the payload
schema, the pipeline, or the engine SDK.
"""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Set once so the card looks the same in every game rather than inheriting each
#: game's own HUD palette — it is a system screen, not part of the game's art.
TITLE_COLOR = (110, 215, 255)
GOAL_COLOR = (228, 232, 240)
HEAD_COLOR = (120, 235, 190)
LABEL_COLOR = (255, 200, 130)
BODY_COLOR = (200, 206, 218)
FOOT_COLOR = (150, 160, 178)

DEFAULT_FOOTER = "press any key to start   ·   H reopens this card"


def _text(img, text, x, y, size=0.5, color=BODY_COLOR, thickness=1):
    """Text with a dark outline, so the card stays legible over a bright game."""
    cv2.putText(img, text, (int(x), int(y)), FONT, size, (0, 0, 0),
                thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (int(x), int(y)), FONT, size, color, thickness,
                cv2.LINE_AA)


def _plate(img, x, y, w, h, alpha=0.88, color=(12, 11, 18)):
    """Translucent backing panel, clipped to the canvas."""
    x, y = max(0, int(x)), max(0, int(y))
    region = img[y:y + int(h), x:x + int(w)]
    if region.size == 0:
        return
    fill = np.full(region.shape, color, dtype=np.uint8)
    img[y:y + int(h), x:x + int(w)] = cv2.addWeighted(
        region, 1.0 - alpha, fill, alpha, 0)


def wrap(text: str, limit: int) -> list[str]:
    """
    Greedy word wrap by character count.

    Hershey glyphs are close enough to fixed width that counting characters
    tracks the rendered width well, and it avoids a `getTextSize` call per word
    on a screen that is redrawn every frame while it is up.
    """
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > limit and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_card(canvas, title, goal, controls=(), keys=(), footer=DEFAULT_FOOTER):
    """
    Draw the how-to-play card over `canvas`, in place.

    `controls` and `keys` are (left, right) pairs: the left column is the input,
    the right is what it does. The card is drawn *over* the running game rather
    than instead of it, so the player reads the rules against the real screen
    and dismissing them does not move anything.
    """
    height, width = canvas.shape[:2]
    pad, label_x = 26, 250
    panel_w = min(width - 60, 820)
    goal_limit = max(30, (panel_w - pad * 2) // 11)
    text_limit = max(24, (panel_w - label_x - pad * 2) // 9)

    rows: list[tuple[str, str, str]] = []
    for line in wrap(goal, goal_limit):
        rows.append(("goal", line, ""))
    if controls:
        rows.append(("gap", "", ""))
        rows.append(("head", "YOUR HANDS", ""))
        for left, right in controls:
            for i, line in enumerate(wrap(right, text_limit)):
                rows.append(("row", left if i == 0 else "", line))
    if keys:
        rows.append(("gap", "", ""))
        rows.append(("head", "KEYS", ""))
        for left, right in keys:
            for i, line in enumerate(wrap(right, text_limit)):
                rows.append(("row", left if i == 0 else "", line))
    if footer:
        rows.append(("gap", "", ""))
        rows.append(("foot", footer, ""))

    # Tighten the leading rather than let a long card run off a short window —
    # 800x600 is a real resolution here (flappy, punchy) and a clipped card is
    # worse than a cramped one.
    line_h = 26
    while line_h > 17 and 84 + len(rows) * line_h + pad > height - 16:
        line_h -= 1

    panel_h = 84 + len(rows) * line_h + pad
    x = (width - panel_w) // 2
    y = max(8, (height - panel_h) // 2)

    _plate(canvas, x, y, panel_w, panel_h)
    cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), (80, 120, 170), 1)

    _text(canvas, title, x + pad, y + 46, 0.82, TITLE_COLOR, 2)
    cv2.line(canvas, (x + pad, y + 62), (x + panel_w - pad, y + 62),
             (55, 75, 105), 1)

    ty = y + 92
    for kind, left, right in rows:
        if kind == "goal":
            _text(canvas, left, x + pad, ty, 0.6, GOAL_COLOR, 1)
        elif kind == "head":
            _text(canvas, left, x + pad, ty, 0.5, HEAD_COLOR, 1)
        elif kind == "row":
            if left:
                cv2.circle(canvas, (x + pad + 4, ty - 5), 3, TITLE_COLOR, -1,
                           cv2.LINE_AA)
                _text(canvas, left, x + pad + 18, ty, 0.5, LABEL_COLOR, 1)
            _text(canvas, right, x + pad + label_x, ty, 0.5, BODY_COLOR, 1)
        elif kind == "foot":
            _text(canvas, left, x + pad, ty, 0.45, FOOT_COLOR, 1)
        ty += line_h


def draw_hint(canvas, text="H  how to play", color=(130, 140, 160)):
    """The one-line reminder left on screen once the card is dismissed."""
    _text(canvas, text, 20, canvas.shape[0] - 16, 0.45, color, 1)
