from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from modules.camera import CameraManager
from modules.hand_tracker import HandTracker

Stroke = dict


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ rx @ ry


def _project_points(points_3d: np.ndarray, frame_w: int, frame_h: int, focal: float = 520.0) -> np.ndarray:
    z = points_3d[:, 2:3]
    z = np.clip(z, 20.0, None)
    xy = points_3d[:, :2] * (focal / z)
    xy[:, 0] += frame_w / 2.0
    xy[:, 1] = frame_h / 2.0 - xy[:, 1]
    return xy.astype(np.int32)


def _build_3d_strokes(strokes: Sequence[Stroke], width: int, height: int) -> List[Tuple[np.ndarray, Tuple[int, int, int], int]]:
    out: List[Tuple[np.ndarray, Tuple[int, int, int], int]] = []
    if not strokes:
        return out

    for si, stroke in enumerate(strokes):
        pts_2d = stroke.get("points", [])
        if len(pts_2d) < 2:
            continue

        color = tuple(stroke.get("color", (255, 255, 255)))
        thickness = int(stroke.get("thickness", 2))

        n = len(pts_2d)
        z_base = -120.0 + si * 8.0
        pts_3d = []
        for i, (x, y) in enumerate(pts_2d):
            depth_wave = (i / max(1, n - 1)) * 40.0
            px = float(x - width / 2)
            py = float(height / 2 - y)
            pz = z_base + depth_wave
            pts_3d.append((px, py, pz))

        out.append((np.array(pts_3d, dtype=np.float32), color, thickness))
    return out


def run_3d_viewer(strokes: Sequence[Stroke], width: int, height: int) -> None:
    """Renderiza los trazos en un visor 3D simple con controles de transformacion."""
    data = _build_3d_strokes(strokes, width, height)

    win_name = "AirDraw 3D Viewer"
    frame_w, frame_h = 1280, 720

    yaw = 0.0
    pitch = 0.0
    roll = 0.0
    tx, ty, tz = 0.0, 0.0, 480.0

    camera = CameraManager(0, 640, 360)
    hand_tracker = None
    hand_enabled = camera.is_opened()
    if hand_enabled:
        hand_tracker = HandTracker(0.6, 0.6, 1)

    grab_active = False
    prev_tip = None
    pinch_threshold = 74.0
    _rot_angles: tuple = (None, None, None)
    _rot_mat: np.ndarray = _rotation_matrix(yaw, pitch, roll)

    while True:
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        cv2.putText(frame, "AirDraw 3D", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 230, 120), 2)
        if hand_enabled:
            tip_text = "Pinza para agarrar y mover | 2 dedos para mover XY | 1 dedo para rotar"
        else:
            tip_text = "Control mano no disponible, usando teclado"
        cv2.putText(frame, tip_text, (24, frame_h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(
            frame,
            "W/S pitch  A/D yaw  Q/E roll  IJKL mover  U/O zoom  R reset  X salir",
            (24, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
        )

        if (yaw, pitch, roll) != _rot_angles:
            _rot_mat = _rotation_matrix(yaw, pitch, roll)
            _rot_angles = (yaw, pitch, roll)
        rot = _rot_mat

        for pts_3d, color, thickness in data:
            transformed = (pts_3d @ rot.T) + np.array([tx, ty, tz], dtype=np.float32)
            pts_2d = _project_points(transformed, frame_w, frame_h)

            for i in range(1, len(pts_2d)):
                cv2.line(frame, tuple(pts_2d[i - 1]), tuple(pts_2d[i]), color, max(1, thickness), cv2.LINE_AA)

        if hand_enabled and hand_tracker is not None:
            ok, cam_frame = camera.read()
            if ok:
                cam_frame = cv2.flip(cam_frame, 1)
                hand_state, mp_result = hand_tracker.process(cam_frame)
                hand_tracker.draw_landmarks(cam_frame, mp_result)

                if hand_state is not None:
                    tip = hand_state.index_tip
                    pinch = hand_state.pinch_distance < pinch_threshold
                    mode_xy = hand_state.finger_up.get("middle", False)

                    if pinch and prev_tip is not None:
                        dx = tip[0] - prev_tip[0]
                        dy = tip[1] - prev_tip[1]

                        if mode_xy:
                            tx += dx * 1.5
                            ty -= dy * 1.5
                        else:
                            yaw += dx * 0.006
                            pitch += dy * 0.006

                        grab_active = True
                    else:
                        grab_active = False

                    prev_tip = tip
                else:
                    prev_tip = None
                    grab_active = False

                small = cv2.resize(cam_frame, (320, 180))
                frame[16:196, frame_w - 336:frame_w - 16] = small
                state_text = "AGARRANDO" if grab_active else "LISTO"
                cv2.putText(frame, f"Mano: {state_text}", (frame_w - 332, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 220), 2)
            else:
                hand_enabled = False

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(16) & 0xFF

        if key == ord("x"):
            break
        if key == ord("a"):
            yaw -= 0.06
        elif key == ord("d"):
            yaw += 0.06
        elif key == ord("w"):
            pitch -= 0.06
        elif key == ord("s"):
            pitch += 0.06
        elif key == ord("q"):
            roll -= 0.06
        elif key == ord("e"):
            roll += 0.06
        elif key == ord("j"):
            tx -= 12.0
        elif key == ord("l"):
            tx += 12.0
        elif key == ord("i"):
            ty += 12.0
        elif key == ord("k"):
            ty -= 12.0
        elif key == ord("u"):
            tz -= 14.0
        elif key == ord("o"):
            tz += 14.0
        elif key == ord("r"):
            yaw = pitch = roll = 0.0
            tx, ty, tz = 0.0, 0.0, 480.0

    if hand_tracker is not None:
        hand_tracker.close()
    camera.release()
    cv2.destroyWindow(win_name)
