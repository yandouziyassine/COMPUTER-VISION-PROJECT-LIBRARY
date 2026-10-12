# ─────────────────────────────────────────────
#  canvas.py  —  NumPy BGRA drawing overlay
# ─────────────────────────────────────────────

from typing import Optional, Tuple
import numpy as np
import cv2

from config import STROKE_THICKNESS, ERASER_RADIUS


class Canvas:
    """
    Transparent BGRA layer that sits above the camera feed.
    All drawing operations mutate this layer; composite() blends it onto a frame.
    """

    def __init__(self, width: int, height: int):
        self._w = width
        self._h = height
        self._layer = np.zeros((height, width, 4), dtype=np.uint8)  # BGRA, all transparent

    # ── drawing ───────────────────────────────────────────────────────────

    def draw_stroke(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color_bgr: Tuple[int, int, int],
        thickness: int = STROKE_THICKNESS,
    ):
        """Draw an anti-aliased line segment from p1 to p2."""
        if p1 is None or p2 is None:
            return
        bgra = (*color_bgr, 255)
        cv2.line(self._layer, p1, p2, bgra, thickness, cv2.LINE_AA)

    def erase(self, center: Tuple[int, int], radius: int = ERASER_RADIUS):
        """Erase a circular area by setting alpha to 0."""
        if center is None:
            return
        mask = np.zeros((self._h, self._w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        self._layer[:, :, 3][mask == 255] = 0

    def erase_brush(self, center: Tuple[int, int], radius: int = ERASER_RADIUS):
        """Alias for erase — used by the toolbar eraser-mode button."""
        self.erase(center, radius)

    def clear(self):
        """Wipe the entire canvas."""
        self._layer[:] = 0

    # ── compositing ───────────────────────────────────────────────────────

    def composite(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Alpha-blend the canvas onto frame_bgr (in-place) and return it.
        """
        alpha = self._layer[:, :, 3:4].astype(np.float32) / 255.0
        canvas_bgr = self._layer[:, :, :3].astype(np.float32)
        frame_f    = frame_bgr.astype(np.float32)
        blended    = frame_f * (1 - alpha) + canvas_bgr * alpha
        frame_bgr[:] = blended.astype(np.uint8)
        return frame_bgr

    # ── properties ────────────────────────────────────────────────────────

    @property
    def size(self) -> Tuple[int, int]:
        return self._w, self._h