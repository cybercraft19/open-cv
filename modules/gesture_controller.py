from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.hand_tracker import HandState


@dataclass
class GestureResult:
    mode: str
    drawing_enabled: bool


class GestureController:
    """Traduce estado de dedos a un modo de interaccion."""

    def __init__(self, pinch_threshold_px: float) -> None:
        self.pinch_threshold_px = pinch_threshold_px

    def resolve(self, hand_state: Optional[HandState]) -> GestureResult:
        if hand_state is None:
            return GestureResult(mode="SIN MANO", drawing_enabled=False)

        f = hand_state.finger_up
        up_count = sum(1 for v in f.values() if v)

        if hand_state.pinch_distance < self.pinch_threshold_px:
            return GestureResult(mode="BORRADOR", drawing_enabled=True)

        if up_count == 5:
            return GestureResult(mode="PAUSA", drawing_enabled=False)

        if f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
            return GestureResult(mode="MOVER", drawing_enabled=False)

        if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
            return GestureResult(mode="DIBUJAR", drawing_enabled=True)

        return GestureResult(mode="LISTO", drawing_enabled=False)
