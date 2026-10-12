# ─────────────────────────────────────────────
#  gesture_detector.py  —  Gesture → Mode state
# ─────────────────────────────────────────────

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from config import DEBOUNCE_FRAMES


class Mode(Enum):
    IDLE  = auto()   # fist or no hand
    DRAW  = auto()   # 1 finger
    ERASE = auto()   # 2 fingers (not on toolbar)


@dataclass
class GestureState:
    mode: Mode
    position: Optional[Tuple[int, int]]   # primary action point (index tip or 2-finger midpoint)
    index_tip: Optional[Tuple[int, int]]  # raw index fingertip
    midpoint: Optional[Tuple[int, int]]   # midpoint between index & middle tips


class GestureDetector:
    """
    Maps finger count + landmark positions → GestureState.
    Applies N-frame debounce before committing a mode change.
    """

    def __init__(self):
        self._current_mode  = Mode.IDLE
        self._pending_mode  = Mode.IDLE
        self._pending_count = 0

    # ── public ────────────────────────────────────────────────────────────

    def update(self, landmarks, finger_count: int) -> GestureState:
        """
        Call once per frame with the latest landmarks and finger count.
        Returns the current (debounced) GestureState.
        """
        candidate = self._fingers_to_mode(finger_count)

        if candidate == self._pending_mode:
            self._pending_count += 1
        else:
            self._pending_mode  = candidate
            self._pending_count = 1

        if self._pending_count >= DEBOUNCE_FRAMES:
            self._current_mode = self._pending_mode

        # Build position data
        index_tip = landmarks[8]  if landmarks else None
        mid_tip   = landmarks[12] if landmarks else None

        midpoint = None
        if index_tip and mid_tip:
            midpoint = (
                (index_tip[0] + mid_tip[0]) // 2,
                (index_tip[1] + mid_tip[1]) // 2,
            )

        if self._current_mode == Mode.DRAW:
            position = index_tip
        elif self._current_mode == Mode.ERASE:
            position = midpoint
        else:
            position = index_tip  # still useful for toolbar hit-test in IDLE

        return GestureState(
            mode=self._current_mode,
            position=position,
            index_tip=index_tip,
            midpoint=midpoint,
        )

    def reset(self):
        self._current_mode  = Mode.IDLE
        self._pending_mode  = Mode.IDLE
        self._pending_count = 0

    # ── private ───────────────────────────────────────────────────────────

    @staticmethod
    def _fingers_to_mode(count: int) -> Mode:
        if count == 1:
            return Mode.DRAW
        if count == 2:
            return Mode.ERASE
        return Mode.IDLE