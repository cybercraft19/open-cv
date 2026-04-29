"""Configuracion central del proyecto AirDraw."""

from __future__ import annotations

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FLIP = True

DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.4
MAX_NUM_HANDS = 1
DETECT_WIDTH = 640          # MediaPipe runs on this width; landmarks re-mapped to full res

CANVAS_BG_COLOR = (0, 0, 0)
DEFAULT_COLOR = (255, 0, 0)
DEFAULT_THICKNESS = 6
ERASER_THICKNESS = 28
MAX_UNDO_STEPS = 25

SMOOTHING_FACTOR = 0.35
SMOOTHING_MIN = 0.20
SMOOTHING_MAX = 0.72
PINCH_THRESHOLD_PX = 45
UI_PINCH_THRESHOLD_PX = 72
UI_DWELL_FRAMES = 16
UI_ACTION_COOLDOWN_FRAMES = 12

OUTPUT_DIR = "outputs"
WINDOW_NAME = "ventana pro"

# BGR (OpenCV)
COLOR_PALETTE = [
    (255, 255, 255),
    (57, 255, 20),
    (0, 0, 255),
    (0, 170, 255),
    (255, 0, 0),
]

COLOR_NAMES = ["blanco nazi", "verdolaga nacional", "azul", "naranja", "rojo pasion"]
