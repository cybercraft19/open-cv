from __future__ import annotations

import math
import time

import cv2
import numpy as np

import config
from modules.camera import CameraManager
from modules.canvas_manager import CanvasManager
from modules.gesture_controller import GestureController
from modules.hand_tracker import HandTracker
from modules.ui import color_index_from_toolbar, draw_toolbar, point_in_toolbar, toolbar_action_from_point
from modules.viewer3d import run_3d_viewer


SHAPE_MODES = ["FREE", "LINE", "RECT", "CIRCLE"]


def _shape_polyline(points, mode: str):
    if not points:
        return []

    if mode == "LINE":
        return [points[0], points[-1]]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    if mode == "RECT":
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]

    if mode == "CIRCLE":
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rx = max(2, (x2 - x1) // 2)
        ry = max(2, (y2 - y1) // 2)
        r = int((rx + ry) / 2)
        ring = []
        for i in range(48):
            t = (i / 48.0) * 2.0 * 3.141592653589793
            ring.append((int(cx + r * math.cos(t)), int(cy + r * math.sin(t))))
        ring.append(ring[0])
        return ring

    return list(points)


def smooth_point(prev, curr, factor: float):
    if prev is None:
        return curr
    x = int(prev[0] + factor * (curr[0] - prev[0]))
    y = int(prev[1] + factor * (curr[1] - prev[1]))
    return x, y


def adaptive_smooth_point(prev, curr, base_factor: float):
    if prev is None:
        return curr
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    dist = (dx * dx + dy * dy) ** 0.5

    # Mas movimiento => menos suavizado para no generar retraso.
    dyn = max(config.SMOOTHING_MIN, min(config.SMOOTHING_MAX, base_factor + 0.30 - dist / 120.0))
    x = int(prev[0] + dyn * dx)
    y = int(prev[1] + dyn * dy)
    return x, y


def handle_keyboard(key: int, color_idx: int, thickness: int, canvas: CanvasManager, message: str):
    if key == ord("q"):
        return False, color_idx, thickness, ""
    if key == ord("c"):
        canvas.clear()
        return True, color_idx, thickness, "Lienzo limpiado"
    if key == ord("s"):
        path = canvas.save(config.OUTPUT_DIR)
        return True, color_idx, thickness, f"Guardado: {path.name}"
    if key == ord("t"):
        path = canvas.save_transparent(config.OUTPUT_DIR)
        return True, color_idx, thickness, f"PNG alpha: {path.name}"
    if key == ord("z"):
        ok = canvas.undo()
        return True, color_idx, thickness, "Undo aplicado" if ok else "Nada para deshacer"
    if key in (ord("+"), ord("=")):
        return True, color_idx, min(24, thickness + 1), f"Grosor: {min(24, thickness + 1)}"
    if key == ord("-"):
        return True, color_idx, max(1, thickness - 1), f"Grosor: {max(1, thickness - 1)}"
    if ord("1") <= key <= ord("5"):
        idx = key - ord("1")
        if idx < len(config.COLOR_PALETTE):
            return True, idx, thickness, f"Color: {config.COLOR_NAMES[idx]}"
    return True, color_idx, thickness, message


def main() -> None:
    camera = CameraManager(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    if not camera.is_opened():
        raise RuntimeError("No se pudo abrir la webcam. Verifica permisos y dispositivo.")

    tracker = HandTracker(
        detection_conf=config.DETECTION_CONFIDENCE,
        tracking_conf=config.TRACKING_CONFIDENCE,
        max_hands=config.MAX_NUM_HANDS,
        detect_width=config.DETECT_WIDTH,
    )
    gestures = GestureController(config.PINCH_THRESHOLD_PX)
    canvas = CanvasManager(config.FRAME_WIDTH, config.FRAME_HEIGHT, config.CANVAS_BG_COLOR)
    canvas.set_history_limit(config.MAX_UNDO_STEPS)

    color_idx = 0
    thickness = config.DEFAULT_THICKNESS
    smooth_tip = None
    last_message = "Modo figura: FREE"
    action_cooldown = 0
    hover_color_idx = -1
    hover_action = ""
    dwell_target = ""
    dwell_frames = 0
    shape_mode_idx = 0
    shape_mode = SHAPE_MODES[shape_mode_idx]
    current_stroke = []
    current_stroke_color = config.COLOR_PALETTE[color_idx]
    current_stroke_thickness = thickness
    strokes_2d = []
    show_help = False
    fps = 0.0
    last_tick = time.perf_counter()

    def reopen_tracking() -> None:
        nonlocal camera, tracker
        camera = CameraManager(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        if not camera.is_opened():
            raise RuntimeError("No se pudo reabrir la webcam despues del visor 3D.")
        tracker = HandTracker(
            detection_conf=config.DETECTION_CONFIDENCE,
            tracking_conf=config.TRACKING_CONFIDENCE,
            max_hands=config.MAX_NUM_HANDS,
            detect_width=config.DETECT_WIDTH,
        )

    def open_viewer_3d() -> None:
        nonlocal last_message, smooth_tip
        tracker.close()
        camera.release()
        cv2.destroyWindow(config.WINDOW_NAME)
        try:
            run_3d_viewer(strokes_2d, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        finally:
            reopen_tracking()
            smooth_tip = None
            last_message = "Visor 3D cerrado"

    def finalize_stroke() -> None:
        nonlocal current_stroke, current_stroke_color, current_stroke_thickness, strokes_2d
        if len(current_stroke) < 2:
            current_stroke = []
            return

        stored_points = list(current_stroke)
        if shape_mode != "FREE":
            # No raw stroke was drawn on canvas (effective_draw was False), draw clean shape now.
            figure_points = _shape_polyline(current_stroke, shape_mode)
            if len(figure_points) >= 2:
                canvas.draw_polyline(
                    figure_points,
                    color=current_stroke_color,
                    thickness=current_stroke_thickness,
                    closed=(shape_mode in ("RECT", "CIRCLE")),
                )
                stored_points = figure_points

        strokes_2d.append(
            {
                "points": stored_points,
                "color": tuple(int(c) for c in current_stroke_color),
                "thickness": int(current_stroke_thickness),
            }
        )
        current_stroke = []

    try:
        while True:
            now = time.perf_counter()
            dt = max(1e-6, now - last_tick)
            instant_fps = 1.0 / dt
            fps = instant_fps if fps == 0.0 else (fps * 0.9 + instant_fps * 0.1)
            last_tick = now

            ok, frame = camera.read()
            if not ok:
                break

            if config.FRAME_FLIP:
                frame = cv2.flip(frame, 1)

            hand_state, mp_result = tracker.process(frame)
            tracker.draw_landmarks(frame, mp_result)

            mode_result = gestures.resolve(hand_state)
            draw_enabled = mode_result.drawing_enabled
            mode_label = mode_result.mode

            if hand_state is not None:
                smooth_tip = adaptive_smooth_point(smooth_tip, hand_state.index_tip, config.SMOOTHING_FACTOR)
                hover_color_idx = color_index_from_toolbar(smooth_tip, len(config.COLOR_PALETTE))
                hover_action = toolbar_action_from_point(smooth_tip)
                in_toolbar = point_in_toolbar(smooth_tip)
                current_target = ""
                if hover_color_idx != -1:
                    current_target = f"C{hover_color_idx}"
                elif hover_action:
                    current_target = f"A{hover_action}"

                if action_cooldown > 0:
                    action_cooldown -= 1

                if in_toolbar:
                    draw_enabled = False
                    mode_label = "UI"
                    if current_target and current_target == dwell_target:
                        dwell_frames += 1
                    else:
                        dwell_target = current_target
                        dwell_frames = 1 if current_target else 0

                    pinch_click = hand_state.pinch_distance < config.UI_PINCH_THRESHOLD_PX
                    dwell_click = dwell_frames >= config.UI_DWELL_FRAMES

                    if (pinch_click or dwell_click) and action_cooldown == 0:
                        if hover_color_idx != -1:
                            color_idx = hover_color_idx
                            last_message = f"Color: {config.COLOR_NAMES[color_idx]}"
                            action_cooldown = config.UI_ACTION_COOLDOWN_FRAMES
                        elif hover_action == "THICK-":
                            thickness = max(1, thickness - 1)
                            last_message = f"Grosor: {thickness}"
                            action_cooldown = config.UI_ACTION_COOLDOWN_FRAMES
                        elif hover_action == "THICK+":
                            thickness = min(24, thickness + 1)
                            last_message = f"Grosor: {thickness}"
                            action_cooldown = config.UI_ACTION_COOLDOWN_FRAMES
                        elif hover_action == "CLEAR":
                            canvas.clear()
                            strokes_2d = []
                            last_message = "Lienzo limpiado"
                            action_cooldown = config.UI_ACTION_COOLDOWN_FRAMES + 2
                        elif hover_action == "UNDO":
                            ok = canvas.undo()
                            if ok and strokes_2d:
                                strokes_2d.pop()
                            last_message = "Undo aplicado" if ok else "Nada para deshacer"
                            action_cooldown = config.UI_ACTION_COOLDOWN_FRAMES
                        elif hover_action == "SAVE":
                            path = canvas.save(config.OUTPUT_DIR)
                            last_message = f"Guardado: {path.name}"
                            action_cooldown = config.UI_ACTION_COOLDOWN_FRAMES + 2
                        elif hover_action == "SHAPE":
                            shape_mode_idx = (shape_mode_idx + 1) % len(SHAPE_MODES)
                            shape_mode = SHAPE_MODES[shape_mode_idx]
                            last_message = f"Modo figura: {shape_mode}"
                            action_cooldown = config.UI_ACTION_COOLDOWN_FRAMES
                        elif hover_action == "VIEW3D":
                            open_viewer_3d()
                            action_cooldown = config.UI_ACTION_COOLDOWN_FRAMES + 2

                        dwell_frames = 0
                else:
                    dwell_target = ""
                    dwell_frames = 0

                draw_color = config.COLOR_PALETTE[color_idx]
                active_thickness = thickness
                if mode_result.mode == "BORRADOR":
                    draw_color = config.CANVAS_BG_COLOR
                    active_thickness = config.ERASER_THICKNESS

                # For non-FREE shape modes, suppress raw stroke; only eraser draws directly.
                effective_draw = draw_enabled and (
                    shape_mode == "FREE" or mode_result.mode == "BORRADOR"
                )
                canvas.update(
                    point=smooth_tip,
                    draw_enabled=effective_draw,
                    color=draw_color,
                    thickness=active_thickness,
                )

                if draw_enabled and mode_result.mode != "BORRADOR":
                    if not current_stroke:
                        current_stroke_color = draw_color
                        current_stroke_thickness = active_thickness
                    current_stroke.append(smooth_tip)
                elif current_stroke:
                    finalize_stroke()

                cv2.circle(frame, smooth_tip, 12, (255, 255, 255), 2)
                cv2.circle(frame, smooth_tip, 4, (120, 220, 255), -1)
                cv2.circle(frame, smooth_tip, max(4, active_thickness // 2), draw_color, 1)
            else:
                if current_stroke:
                    finalize_stroke()
                canvas.reset_prev()
                smooth_tip = None
                action_cooldown = 0
                hover_color_idx = -1
                hover_action = ""
                dwell_target = ""
                dwell_frames = 0

            merged = canvas.blend_on_frame(frame)

            # Live preview for non-FREE shape modes
            if current_stroke and shape_mode != "FREE" and len(current_stroke) >= 2:
                preview_pts = _shape_polyline(current_stroke, shape_mode)
                if len(preview_pts) >= 2:
                    pts_arr = np.array(preview_pts, dtype=np.int32)
                    closed = shape_mode in ("RECT", "CIRCLE")
                    cv2.polylines(
                        merged, [pts_arr], closed,
                        config.COLOR_PALETTE[color_idx], thickness, cv2.LINE_AA,
                    )

            draw_toolbar(
                merged,
                colors=config.COLOR_PALETTE,
                active_color_idx=color_idx,
                hover_color_idx=hover_color_idx,
                hover_action=hover_action,
                dwell_progress=(dwell_frames / max(1, config.UI_DWELL_FRAMES)),
                thickness=thickness,
                mode=f"{mode_label} | {shape_mode}",
                last_message=last_message,
                fps=fps,
                hand_visible=(hand_state is not None),
                show_help=show_help,
                shape_mode=shape_mode,
            )

            cv2.imshow(config.WINDOW_NAME, merged)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("m"):
                shape_mode_idx = (shape_mode_idx + 1) % len(SHAPE_MODES)
                shape_mode = SHAPE_MODES[shape_mode_idx]
                last_message = f"Modo figura: {shape_mode}"
            elif key == ord("f"):
                shape_mode_idx = 0
                shape_mode = SHAPE_MODES[shape_mode_idx]
                last_message = "Modo figura: FREE"
            elif key == ord("v"):
                open_viewer_3d()
            elif key == ord("c"):
                strokes_2d = []
            elif key == ord("z") and strokes_2d:
                strokes_2d.pop()
            elif key == ord("h"):
                show_help = not show_help
                last_message = "Ayuda visible" if show_help else "Ayuda oculta"

            running, color_idx, thickness, last_message = handle_keyboard(
                key,
                color_idx,
                thickness,
                canvas,
                last_message,
            )
            if not running:
                break

    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
