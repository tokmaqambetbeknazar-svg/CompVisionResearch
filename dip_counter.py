import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(r"C:\CVModels\pose_landmarker_lite.task")

if not MODEL_PATH.is_file():
    MODEL_PATH = PROJECT_DIR / "pose_landmarker_lite.task"

LOG_PATH = PROJECT_DIR / "workout_log.csv"

CAMERA_INDEX = 0
VISIBILITY_THRESHOLD = 0.5
TOP_ANGLE = 155
BOTTOM_ANGLE = 105
READY_HOLD_FRAMES = 15
TRACKING_LOSS_SECONDS = 0.75
SHOULDER_ELBOW_LEVEL_TOLERANCE = 0.45


def is_visible(point):
    return (point.visibility or 0) >= VISIBILITY_THRESHOLD


def to_pixel(point, width, height):
    return int(point.x * width), int(point.y * height)


def calculate_angle(point_a, point_b, point_c):
    """Вычисляет угол в точке B между направлениями к A и C."""
    vector_a = (point_a[0] - point_b[0], point_a[1] - point_b[1])
    vector_c = (point_c[0] - point_b[0], point_c[1] - point_b[1])

    length_a = math.hypot(*vector_a)
    length_c = math.hypot(*vector_c)

    if length_a == 0 or length_c == 0:
        return None

    dot_product = vector_a[0] * vector_c[0] + vector_a[1] * vector_c[1]
    cosine = dot_product / (length_a * length_c)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def calculate_torso_lean(shoulder, hip):
    """Приблизительно оценивает наклон корпуса от вертикали на изображении."""
    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]

    if math.hypot(dx, dy) == 0:
        return None

    return abs(math.degrees(math.atan2(dx, dy)))


def draw_rounded_rectangle(image, top_left, bottom_right, color, radius):
    x1, y1 = top_left
    x2, y2 = bottom_right

    if x2 <= x1 or y2 <= y1:
        return

    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    if radius == 0:
        cv2.rectangle(image, top_left, bottom_right, color, -1)
        return

    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    for center in (
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ):
        cv2.circle(image, center, radius, color, -1)


def draw_panel(frame, repetitions, phase, torso_feedback, progress):
    height, width, _ = frame.shape
    left, top = 12, 12
    panel_width = min(490, width - 24)
    panel_height = min(148, height - 24)

    if panel_width < 220 or panel_height < 110:
        return

    right = left + panel_width
    bottom = top + panel_height

    overlay = frame.copy()
    draw_rounded_rectangle(
        overlay,
        (left, top),
        (right, bottom),
        (165, 165, 155),
        22,
    )
    draw_rounded_rectangle(
        overlay,
        (left + 4, top + 4),
        (right - 4, bottom - 4),
        (27, 28, 33),
        18,
    )
    cv2.addWeighted(overlay, 0.94, frame, 0.06, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (245, 245, 245)
    gray = (195, 197, 204)
    teal = (0, 205, 145)
    text_x = left + 18

    cv2.putText(
        frame, "REPS:", (text_x, top + 37),
        font, 0.64, white, 2, cv2.LINE_AA,
    )
    label_width = cv2.getTextSize("REPS:", font, 0.64, 2)[0][0]
    cv2.putText(
        frame, str(repetitions), (text_x + label_width + 8, top + 37),
        font, 0.68, teal, 2, cv2.LINE_AA,
    )

    bar_x = right - 46
    bar_top = top + 43
    bar_bottom = bottom - 24
    bar_width = 15

    cv2.line(
        frame,
        (text_x, top + 48),
        (bar_x - 12, top + 48),
        (75, 76, 82),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"PHASE: {phase}", (text_x, top + 70),
        font, 0.40, white, 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"BODY: {torso_feedback}", (text_x, top + 101),
        font, 0.37, gray, 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "FORM: Keep shoulders down; move with control",
        (text_x, top + 130),
        font, 0.34, gray, 1, cv2.LINE_AA,
    )

    cv2.putText(
        frame, "UP", (bar_x - 2, top + 36),
        font, 0.31, white, 1, cv2.LINE_AA,
    )
    draw_rounded_rectangle(
        frame,
        (bar_x, bar_top),
        (bar_x + bar_width, bar_bottom),
        (75, 77, 84),
        7,
    )

    if progress is not None and progress > 0:
        fill_bottom = int(bar_top + progress * (bar_bottom - bar_top))
        if fill_bottom > bar_top + 6:
            draw_rounded_rectangle(
                frame,
                (bar_x + 2, bar_top + 2),
                (bar_x + bar_width - 2, fill_bottom),
                teal,
                5,
            )

    cv2.putText(
        frame, "DOWN", (bar_x - 8, bottom - 7),
        font, 0.31, white, 1, cv2.LINE_AA,
    )


if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        "Не найдена модель pose_landmarker_lite.task. "
        "Положи её в C:\\CVModels или рядом с dip_counter.py."
    )

print(f"Модель: {MODEL_PATH}")

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
)

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError(
        f"Не удалось открыть камеру {CAMERA_INDEX}. "
        "Проверь Camo Studio и номер камеры."
    )

repetitions = 0
is_armed = False
start_pose_frames = 0
arm_is_at_top = False
bottom_reached = False
movement_progress = None
tracking_lost_since = None

try:
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        print(
            "Встань на стационарные брусья и удерживай верхний упор "
            "с выпрямленными руками. Нажми Q, чтобы закончить."
        )

        while True:
            success, frame = camera.read()

            if not success:
                raise RuntimeError("Не удалось получить изображение с камеры.")

            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = time.monotonic_ns() // 1_000_000
            result = landmarker.detect_for_video(image, timestamp_ms)

            arm_angle = None
            shoulder_elbow_level = None
            torso_lean = None
            phase = "Step into camera view"
            torso_feedback = "Show shoulder and hip"

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # Выбираем лучше видимую руку.
                sides = [
                    (11, 13, 15, 23),  # левая сторона
                    (12, 14, 16, 24),  # правая сторона
                ]

                best_side = None
                best_visibility = -1.0

                for shoulder_i, elbow_i, wrist_i, hip_i in sides:
                    arm_points = (
                        landmarks[shoulder_i],
                        landmarks[elbow_i],
                        landmarks[wrist_i],
                    )
                    visibility = min(
                        (point.visibility or 0)
                        for point in arm_points
                    )

                    if visibility > best_visibility:
                        best_visibility = visibility
                        best_side = (
                            shoulder_i,
                            elbow_i,
                            wrist_i,
                            hip_i,
                        )

                if (
                    best_side is not None
                    and best_visibility >= VISIBILITY_THRESHOLD
                ):
                    shoulder_i, elbow_i, wrist_i, hip_i = best_side
                    shoulder_lm = landmarks[shoulder_i]
                    elbow_lm = landmarks[elbow_i]
                    wrist_lm = landmarks[wrist_i]
                    hip_lm = landmarks[hip_i]

                    shoulder = to_pixel(shoulder_lm, width, height)
                    elbow = to_pixel(elbow_lm, width, height)
                    wrist = to_pixel(wrist_lm, width, height)

                    arm_angle = calculate_angle(shoulder, elbow, wrist)

                    cv2.line(frame, shoulder, elbow, (0, 180, 0), 3)
                    cv2.line(frame, elbow, wrist, (0, 180, 0), 3)
                    for point in (shoulder, elbow, wrist):
                        cv2.circle(frame, point, 6, (0, 0, 255), -1)

                    upper_arm_length = math.hypot(
                        shoulder[0] - elbow[0],
                        shoulder[1] - elbow[1],
                    )
                    if upper_arm_length > 0:
                        shoulder_elbow_level = (
                            abs(shoulder[1] - elbow[1]) / upper_arm_length
                        )

                    if is_visible(hip_lm):
                        hip = to_pixel(hip_lm, width, height)
                        torso_lean = calculate_torso_lean(shoulder, hip)

                        cv2.line(frame, shoulder, hip, (255, 0, 0), 3)

                        if torso_lean is None:
                            torso_feedback = "Adjust camera to side view"
                        elif torso_lean < 15:
                            torso_feedback = "Lean slightly forward"
                        elif torso_lean <= 35:
                            torso_feedback = "Forward lean looks controlled"
                        else:
                            torso_feedback = "Reduce forward lean"
                    else:
                        torso_feedback = "Show shoulder and hip"

                    if arm_angle is None:
                        phase = "Keep arm visible"
                    else:
                        movement_progress = max(
                            0.0,
                            min(
                                1.0,
                                (TOP_ANGLE - arm_angle)
                                / (TOP_ANGLE - BOTTOM_ANGLE),
                            ),
                        )

                        # Для старта проверяем устойчиво видимые выпрямленные руки.
                        # Положение запястья относительно плеча не используется.
                        start_pose_detected = arm_angle >= TOP_ANGLE
                        now = time.monotonic()

                        if not is_armed:
                            if start_pose_detected:
                                start_pose_frames += 1
                                phase = (
                                    f"Hold top support "
                                    f"{start_pose_frames}/{READY_HOLD_FRAMES}"
                                )

                                if start_pose_frames >= READY_HOLD_FRAMES:
                                    is_armed = True
                                    arm_is_at_top = True
                                    bottom_reached = False
                                    phase = "Ready - lower slowly"
                            else:
                                start_pose_frames = 0
                                phase = "Straighten arms and hold"
                        else:
                            tracking_lost_since = None

                            at_bottom = (
                                arm_angle <= BOTTOM_ANGLE
                                and shoulder_elbow_level is not None
                                and shoulder_elbow_level
                                <= SHOULDER_ELBOW_LEVEL_TOLERANCE
                            )

                            if at_bottom and arm_is_at_top:
                                bottom_reached = True
                                arm_is_at_top = False
                                phase = "Push up smoothly"

                            elif (
                                arm_angle <= BOTTOM_ANGLE
                                and not at_bottom
                            ):
                                phase = "Shoulders near elbow level"

                            elif arm_angle >= TOP_ANGLE:
                                if bottom_reached:
                                    repetitions += 1
                                    bottom_reached = False

                                arm_is_at_top = True
                                phase = "Lower slowly"

                            elif bottom_reached:
                                phase = "Push up smoothly"

                            elif arm_is_at_top:
                                phase = "Lower slowly"

                            else:
                                phase = "Move under control"
                else:
                    phase = "Show shoulder, elbow and wrist"

            now = time.monotonic()

            if arm_angle is None:
                movement_progress = None
                start_pose_frames = 0

                if tracking_lost_since is None:
                    tracking_lost_since = now

                if (
                    is_armed
                    and now - tracking_lost_since >= TRACKING_LOSS_SECONDS
                ):
                    is_armed = False
                    arm_is_at_top = False
                    bottom_reached = False
                    phase = "Tracking lost - reset at top"

            draw_panel(
                frame,
                repetitions,
                phase,
                torso_feedback,
                movement_progress,
            )

            cv2.imshow("Dip counter", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    camera.release()
    cv2.destroyAllWindows()


file_is_new = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0

with LOG_PATH.open("a", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)

    if file_is_new:
        writer.writerow(["Дата и время", "Упражнение", "Повторы"])

    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Дипы на брусьях",
        repetitions,
    ])

print(f"Тренировка сохранена: {repetitions} повторов.")