from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

TOOLBAR_HEIGHT = 110
COLORS_START_X = 238
COLOR_BOX_W = 52
COLOR_BOX_H = 52
COLOR_GAP = 14
COLOR_Y = 26

ACTIONS_START_X = 620
ACTION_W = 78
ACTION_H = 52
ACTION_GAP = 8
ACTION_Y = 26
HIT_MARGIN = 8

ACTION_KEYS = ["THICK-", "THICK+", "UNDO", "CLEAR", "SAVE", "SHAPE", "VIEW3D"]
ACTION_LABELS = {
    "THICK-": "-",
    "THICK+": "+",
    "UNDO": "UNDO",
    "CLEAR": "CLR",
    "SAVE": "SAVE",
    "SHAPE": "",  # label set dynamically from shape_mode
    "VIEW3D": "3D",
}

_toolbar_bg_cache: Optional[np.ndarray] = None
_toolbar_bg_width: int = 0


def _get_toolbar_bg(w: int) -> np.ndarray:
    global _toolbar_bg_cache, _toolbar_bg_width
    if _toolbar_bg_cache is None or _toolbar_bg_width != w:
        bg = np.zeros((TOOLBAR_HEIGHT, w, 3), dtype=np.uint8)
        for y in range(TOOLBAR_HEIGHT):
            t = y / max(1, TOOLBAR_HEIGHT - 1)
            bg[y] = (int(15 + 16 * t), int(10 + 8 * t), int(28 + 30 * t))
        _toolbar_bg_cache = bg
        _toolbar_bg_width = w
    return _toolbar_bg_cache


def _draw_neon_glow_rect(frame, x1: int, y1: int, x2: int, y2: int, glow_color: Tuple[int, int, int]) -> None:
    for i in (8, 5, 2):
        cv2.rectangle(frame, (x1 - i, y1 - i), (x2 + i, y2 + i), glow_color, 1)


def draw_toolbar(
    frame,
    colors: List[Tuple[int, int, int]],
    active_color_idx: int,
    hover_color_idx: int,
    hover_action: str,
    dwell_progress: float,
    thickness: int,
    mode: str,
    last_message: str,
    fps: float,
    hand_visible: bool,
    show_help: bool,
    shape_mode: str = "FREE",
) -> None:
    h, w, _ = frame.shape

    frame[:TOOLBAR_HEIGHT] = _get_toolbar_bg(w)
    cv2.rectangle(frame, (0, TOOLBAR_HEIGHT), (w, TOOLBAR_HEIGHT + 1), (82, 82, 82), -1)
    cv2.putText(frame, "dibujos insanos", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 230, 120), 2)
    cv2.putText(frame, "ola", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 205, 240), 1)

    for i, color in enumerate(colors):
        x1 = COLORS_START_X + i * (COLOR_BOX_W + COLOR_GAP)
        y1 = COLOR_Y
        x2 = x1 + COLOR_BOX_W
        y2 = y1 + COLOR_BOX_H
        border = (255, 255, 255) if i == active_color_idx else (120, 120, 120)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 3 if i == active_color_idx else 2)
        if i == active_color_idx:
            _draw_neon_glow_rect(frame, x1, y1, x2, y2, (255, 255, 180))
        if i == hover_color_idx:
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (120, 220, 255), 2)

    for i, key in enumerate(ACTION_KEYS):
        x1 = ACTIONS_START_X + i * (ACTION_W + ACTION_GAP)
        y1 = ACTION_Y
        x2 = x1 + ACTION_W
        y2 = y1 + ACTION_H

        fill = (58, 58, 58) if key == hover_action else (42, 42, 42)
        border = (120, 220, 255) if key == hover_action else (130, 130, 130)
        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2)
        label = shape_mode if key == "SHAPE" else ACTION_LABELS[key]
        label_x = x1 + 8 if len(label) > 2 else x1 + 34
        cv2.putText(frame, label, (label_x, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (230, 230, 230), 2)

    if hover_color_idx != -1 or hover_action:
        px = int(20 + 220 * max(0.0, min(1.0, dwell_progress)))
        cv2.rectangle(frame, (20, 84), (242, 100), (35, 30, 50), -1)
        cv2.rectangle(frame, (20, 84), (242, 100), (90, 100, 130), 1)
        cv2.rectangle(frame, (20, 84), (px, 100), (255, 220, 80), -1)
        cv2.putText(frame, "Seleccion", (252, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 220, 235), 1)

    cv2.putText(frame, f"Grosor: {thickness}", (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (220, 220, 220), 1)
    cv2.putText(frame, f"Modo: {mode}", (185, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (120, 255, 220), 1)

    hand_chip = "HAND OK" if hand_visible else "NO HAND"
    hand_color = (120, 255, 200) if hand_visible else (130, 130, 130)
    cv2.rectangle(frame, (358, 90), (470, 110), (30, 30, 45), -1)
    cv2.rectangle(frame, (358, 90), (470, 110), hand_color, 1)
    cv2.putText(frame, hand_chip, (366, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

    fps_text = f"FPS {fps:.1f}"
    cv2.rectangle(frame, (480, 90), (565, 110), (30, 30, 45), -1)
    cv2.rectangle(frame, (480, 90), (565, 110), (200, 200, 120), 1)
    cv2.putText(frame, fps_text, (488, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 180), 1)

    hint = "Teclas: 1-5 color | F libre | M figuras | V 3D | T alpha | Z undo | H ayuda | Q"
    cv2.putText(frame, hint, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    if last_message:
        cv2.putText(frame, last_message, (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 220, 100), 2)

    if show_help:
        overlay = frame.copy()
        x1, y1, x2, y2 = 20, 150, 540, 360
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 16, 30), -1)
        cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 200, 255), 2)

        lines = [
            "Atajos Pro",
            "H: mostrar/ocultar ayuda",
            "F: modo libre inmediato",
            "M: cambia FREE/LINE/RECT/CIRCLE",
            "V: abre visor 3D (con agarre por mano)",
            "T: exporta PNG transparente",
            "Z: deshacer ultimo trazo",
            "Pinza o permanencia para click en la barra",
        ]
        yy = y1 + 30
        for i, text in enumerate(lines):
            size = 0.66 if i == 0 else 0.56
            color = (255, 230, 130) if i == 0 else (220, 220, 235)
            cv2.putText(frame, text, (x1 + 16, yy), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)
            yy += 28


def color_index_from_toolbar(point: Tuple[int, int], colors_count: int) -> int:
    x, y = point
    if y < COLOR_Y - HIT_MARGIN or y > COLOR_Y + COLOR_BOX_H + HIT_MARGIN:
        return -1

    for i in range(colors_count):
        x1 = COLORS_START_X + i * (COLOR_BOX_W + COLOR_GAP)
        x2 = x1 + COLOR_BOX_W
        if x1 - HIT_MARGIN <= x <= x2 + HIT_MARGIN:
            return i
    return -1


def toolbar_action_from_point(point: Tuple[int, int]) -> str:
    x, y = point
    if y < ACTION_Y - HIT_MARGIN or y > ACTION_Y + ACTION_H + HIT_MARGIN:
        return ""

    for i, action in enumerate(ACTION_KEYS):
        x1 = ACTIONS_START_X + i * (ACTION_W + ACTION_GAP)
        x2 = x1 + ACTION_W
        if x1 - HIT_MARGIN <= x <= x2 + HIT_MARGIN:
            return action
    return ""


def point_in_toolbar(point: Tuple[int, int]) -> bool:
    _, y = point
    return 0 <= y <= TOOLBAR_HEIGHT
