from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class CanvasManager:
    """Administra el lienzo y las operaciones de dibujo."""

    def __init__(self, width: int, height: int, bg_color=(0, 0, 0)) -> None:
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if bg_color != (0, 0, 0):
            self.canvas[:] = bg_color
        self.prev_point: Optional[Tuple[int, int]] = None
        self._history: deque = deque()
        self._max_history = 25

    def set_history_limit(self, max_steps: int) -> None:
        self._max_history = max(1, int(max_steps))

    def _push_history(self) -> None:
        self._history.append(self.canvas.copy())
        if len(self._history) > self._max_history:
            self._history.popleft()

    def reset_prev(self) -> None:
        self.prev_point = None

    def clear(self) -> None:
        self._push_history()
        self.canvas[:] = self.bg_color
        self.prev_point = None

    def undo(self) -> bool:
        if not self._history:
            return False
        self.canvas = self._history.pop()
        self.prev_point = None
        return True

    def update(self, point: Tuple[int, int], draw_enabled: bool, color: Tuple[int, int, int], thickness: int) -> None:
        if not draw_enabled:
            self.prev_point = point
            return

        if self.prev_point is None:
            self._push_history()
            self.prev_point = point
            return

        cv2.line(self.canvas, self.prev_point, point, color, thickness, cv2.LINE_AA)
        self.prev_point = point

    def draw_polyline(self, points, color: Tuple[int, int, int], thickness: int, closed: bool = False) -> None:
        if len(points) < 2:
            return
        self._push_history()
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(self.canvas, [pts], closed, color, thickness, cv2.LINE_AA)
        self.prev_point = None

    def blend_on_frame(self, frame, alpha=0.9):
        bg = np.array(self.bg_color, dtype=np.uint8)
        mask2d = np.any(self.canvas != bg, axis=2)
        if not mask2d.any():
            return frame.copy()
        out = frame.copy()
        if alpha >= 1.0:
            out[mask2d] = self.canvas[mask2d]
        else:
            blended = cv2.addWeighted(frame, 1.0 - alpha, self.canvas, alpha, 0)
            out[mask2d] = blended[mask2d]
        return out

    def save(self, output_dir: str) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_name = f"airdraw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = output_path / file_name
        cv2.imwrite(str(full_path), self.canvas)
        return full_path

    def save_transparent(self, output_dir: str) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        bg = np.array(self.bg_color, dtype=np.uint8)
        mask = np.any(self.canvas != bg, axis=2).astype(np.uint8) * 255
        rgba = np.dstack((self.canvas, mask))

        file_name = f"airdraw_transparent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = output_path / file_name
        cv2.imwrite(str(full_path), rgba)
        return full_path
