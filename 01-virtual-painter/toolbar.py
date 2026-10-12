# ─────────────────────────────────────────────
#  toolbar.py  —  Semi-transparent top-bar UI
# ─────────────────────────────────────────────

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from config import (
    PALETTE, DEFAULT_COLOR,
    TOOLBAR_HEIGHT, SWATCH_SIZE, SWATCH_MARGIN,
    BTN_WIDTH, BTN_MARGIN,
)


@dataclass
class HitResult:
    kind: str                            # 'color' | 'clear' | 'eraser'
    value: Optional[Tuple[int, int, int]] = None  # BGR color if kind=='color'


class Toolbar:
    """
    Renders the semi-transparent toolbar and handles hit-testing.
    """

    def __init__(self, frame_width: int):
        self._w = frame_width
        self.active_color: Tuple[int, int, int] = DEFAULT_COLOR
        self.eraser_mode: bool = False

        # Pre-compute swatch rects (x, y, x2, y2)
        self._swatches = []
        x = SWATCH_MARGIN
        y = (TOOLBAR_HEIGHT - SWATCH_SIZE) // 2
        for name, bgr in PALETTE:
            self._swatches.append({
                "name": name,
                "bgr":  bgr,
                "rect": (x, y, x + SWATCH_SIZE, y + SWATCH_SIZE),
            })
            x += SWATCH_SIZE + SWATCH_MARGIN

        # Action buttons
        btn_y  = (TOOLBAR_HEIGHT - SWATCH_SIZE) // 2
        btn_x1 = self._w - 2 * (BTN_WIDTH + BTN_MARGIN)
        self._btn_clear  = (btn_x1, btn_y, btn_x1 + BTN_WIDTH, btn_y + SWATCH_SIZE)
        btn_x2 = btn_x1 + BTN_WIDTH + BTN_MARGIN
        self._btn_eraser = (btn_x2, btn_y, btn_x2 + BTN_WIDTH, btn_y + SWATCH_SIZE)

    # ── hit-testing ───────────────────────────────────────────────────────

    def hit_test(self, x: int, y: int) -> Optional[HitResult]:
        """
        Returns HitResult if (x, y) falls on an interactive element,
        None otherwise. Only valid for y < TOOLBAR_HEIGHT.
        """
        if y > TOOLBAR_HEIGHT:
            return None

        for swatch in self._swatches:
            x1, y1, x2, y2 = swatch["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return HitResult(kind="color", value=swatch["bgr"])

        x1, y1, x2, y2 = self._btn_clear
        if x1 <= x <= x2 and y1 <= y <= y2:
            return HitResult(kind="clear")

        x1, y1, x2, y2 = self._btn_eraser
        if x1 <= x <= x2 and y1 <= y <= y2:
            return HitResult(kind="eraser")

        return None

    def point_in_toolbar(self, pt: Tuple[int, int]) -> bool:
        return pt is not None and pt[1] < TOOLBAR_HEIGHT

    # ── rendering ─────────────────────────────────────────────────────────

    def render(self, frame: np.ndarray):
        """Overlay the toolbar onto frame in-place."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self._w, TOOLBAR_HEIGHT), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Color swatches
        for swatch in self._swatches:
            x1, y1, x2, y2 = swatch["rect"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), swatch["bgr"], -1, cv2.LINE_AA)
            # Active border
            if swatch["bgr"] == self.active_color and not self.eraser_mode:
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2, cv2.LINE_AA)

        # "Clear all" button
        x1, y1, x2, y2 = self._btn_clear
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 180), -1, cv2.LINE_AA)
        cv2.putText(frame, "Clear all", (x1 + 8, y1 + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # "Eraser" button
        x1, y1, x2, y2 = self._btn_eraser
        btn_col = (60, 140, 60) if self.eraser_mode else (70, 70, 70)
        cv2.rectangle(frame, (x1, y1), (x2, y2), btn_col, -1, cv2.LINE_AA)
        label = "Eraser: ON " if self.eraser_mode else "Eraser mode"
        cv2.putText(frame, label, (x1 + 6, y1 + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)