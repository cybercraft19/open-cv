from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp


@dataclass
class HandState:
    landmarks_px: Dict[int, Tuple[int, int]]
    index_tip: Tuple[int, int]
    finger_up: Dict[str, bool]
    pinch_distance: float


class HandTracker:
    """Gestiona MediaPipe Hands y expone estado util para interaccion."""

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )

    def _ensure_model(self) -> Path:
        model_path = Path("assets") / "models" / "hand_landmarker.task"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists():
            return model_path

        # Descarga una vez el modelo para uso local.
        try:
            urlretrieve(self.MODEL_URL, model_path)
        except Exception as exc:
            raise RuntimeError(
                "No se pudo descargar el modelo de mano de MediaPipe. "
                "Verifica conexion a internet y reintenta."
            ) from exc
        return model_path

    def __init__(
        self,
        detection_conf: float,
        tracking_conf: float,
        max_hands: int = 1,
        detect_width: int = 640,
    ) -> None:
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        model_path = self._ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._start_ms = int(time.perf_counter() * 1000)
        self._detect_width = detect_width

    def _normalized_to_px(self, normalized_points, frame_shape) -> Dict[int, Tuple[int, int]]:
        h, w, _ = frame_shape
        pts: Dict[int, Tuple[int, int]] = {}
        for idx, lm in enumerate(normalized_points):
            pts[idx] = (int(lm.x * w), int(lm.y * h))
        return pts

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _finger_states(self, pts: Dict[int, Tuple[int, int]]) -> Dict[str, bool]:
        # En coordenadas de imagen, menor y significa mas arriba.
        finger_up = {
            "thumb": pts[4][0] > pts[3][0],
            "index": pts[8][1] < pts[6][1],
            "middle": pts[12][1] < pts[10][1],
            "ring": pts[16][1] < pts[14][1],
            "pinky": pts[20][1] < pts[18][1],
        }
        return finger_up

    def process(self, frame) -> Tuple[Optional[HandState], any]:
        orig_h, orig_w = frame.shape[:2]
        if orig_w > self._detect_width:
            det_h = int(orig_h * self._detect_width / orig_w)
            detect_frame = cv2.resize(
                frame, (self._detect_width, det_h), interpolation=cv2.INTER_NEAREST
            )
        else:
            detect_frame = frame

        rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.perf_counter() * 1000) - self._start_ms
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        if not result.hand_landmarks:
            return None, result

        hand_landmarks = result.hand_landmarks[0]
        # Always map to original frame dimensions — landmarks are normalized [0,1]
        pts = self._normalized_to_px(hand_landmarks, frame.shape)
        finger_up = self._finger_states(pts)
        pinch_distance = self._distance(pts[4], pts[8])

        state = HandState(
            landmarks_px=pts,
            index_tip=pts[8],
            finger_up=finger_up,
            pinch_distance=pinch_distance,
        )
        return state, result

    def draw_landmarks(self, frame, result) -> None:
        if not result.hand_landmarks:
            return
        h, w, _ = frame.shape
        for hand_landmarks in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for a, b in self.HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (120, 190, 255), 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, 3, (255, 255, 255), -1)

    def close(self) -> None:
        self.landmarker.close()
