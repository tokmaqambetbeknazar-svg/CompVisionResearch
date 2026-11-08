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

VISIBILITY_THRESHOLD = 0.5
TOP_ANGLE = 160
BOTTOM_ANGLE = 90
READY_HOLD_FRAMES = 15
TRACKING_LOSS_SECONDS = 0.75


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


def average_visible_points(landmarks, indices, width, height):
    points = [
        to_pixel(landmarks[index], width, height)
        for index in indices
        if is_visible(landmarks[index])
    ]

    if not points:
        return None

    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def calculate_signed_body_angle(shoulder, hip, ankle):
    """Приблизительно различает таз выше и ниже линии тела."""
    shoulder_to_hip = (
        shoulder[0] - hip[0],
        shoulder[1] - hip[1],
    )
    ankle_to_hip = (
        ankle[0] - hip[0],
        ankle[1] - hip[1],
    )

    length_a = math.hypot(*shoulder_to_hip)
    length_b = math.hypot(*ankle_to_hip)
    body_length = math.hypot(
        ankle[0] - shoulder[0],
        ankle[1] - shoulder[1],
    )

    if length_a == 0 or length_b == 0 or body_length == 0:
        return None

    if abs(ankle[0] - shoulder[0]) < 0.15 * body_length:
        return None

    dot_product = (
        shoulder_to_hip[0] * ankle_to_hip[0]
        + shoulder_to_hip[1] * ankle_to_hip[1]
    )
    cosine = dot_product / (length_a * length_b)
    cosine = max(-1.0, min(1.0, cosine))
    interior_angle = math.degrees(math.acos(cosine))

    cross_product = (
        shoulder_to_hip[0] * ankle_to_hip[1]
        - shoulder_to_hip[1] * ankle_to_hip[0]
    )
    facing_direction = ankle[0] - shoulder[0]
    pelvis_is_high = cross_product * facing_direction < 0

    if pelvis_is_high:
        return interior_angle

    return 360.0 - interior_angle


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


def draw_panel(frame, repetitions, phase, body_feedback, elbow_feedback, progress):
    height, width, _ = frame.shape
    left, top = 12, 12
    panel_width = min(420, width - 24)
    panel_height = min(148, height - 24)

    if panel_width < 200 or panel_height < 100:
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
        frame,
        "REPS:",
        (text_x, top + 37),
        font,
        0.64,
        white,
        2,
        cv2.LINE_AA,
    )
    label_width = cv2.getTextSize("REPS:", font, 0.64, 2)[0][0]
    cv2.putText(
        frame,
        str(repetitions),
        (text_x + label_width + 8, top + 37),
        font,
        0.68,
        teal,
        2,
        cv2.LINE_AA,
    )

    bar_x = right - 46
    bar_top = top + 43
    bar_bottom = bottom - 25
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
        frame,
        f"PHASE: {phase}",
        (text_x, top + 70),
        font,
        0.42,
        white,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"BODY: {body_feedback}",
        (text_x, top + 94),
        font,
        0.38,
        gray,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"ELBOWS: {elbow_feedback}",
        (text_x, top + 118),
        font,
        0.38,
        gray,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "UP",
        (bar_x - 2, top + 36),
        font,
        0.31,
        white,
        1,
        cv2.LINE_AA,
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
        frame,
        "DOWN",
        (bar_x - 8, bottom - 8),
        font,
        0.31,
        white,
        1,
        cv2.LINE_AA,
    )


if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        "Не найдена модель pose_landmarker_lite.task. "
        "Положи её в C:\\CVModels или рядом с pushup_counter.py."
    )

print(f"Модель: {MODEL_PATH}")

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Не удалось открыть камеру. Проверь подключение и разрешения Windows."
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
            "Поставь камеру примерно под углом 45 градусов спереди. "
            "Нажми Q в окне видео, чтобы закончить."
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

            elbow_angles = []
            elbow_flare_angles = []
            body_angle = None
            phase = "Step into camera view"
            body_feedback = "Show full body"
            elbow_feedback = "Show arms and shoulders"

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                sides = [
                    (11, 13, 15, 23),
                    (12, 14, 16, 24),
                ]

                for shoulder_i, elbow_i, wrist_i, hip_i in sides:
                    shoulder_lm = landmarks[shoulder_i]
                    elbow_lm = landmarks[elbow_i]
                    wrist_lm = landmarks[wrist_i]
                    hip_lm = landmarks[hip_i]

                    arm_visible = all(
                        is_visible(point)
                        for point in (shoulder_lm, elbow_lm, wrist_lm)
                    )

                    if arm_visible:
                        shoulder = to_pixel(shoulder_lm, width, height)
                        elbow = to_pixel(elbow_lm, width, height)
                        wrist = to_pixel(wrist_lm, width, height)

                        angle = calculate_angle(shoulder, elbow, wrist)
                        if angle is not None:
                            elbow_angles.append(angle)

                        cv2.line(frame, shoulder, elbow, (0, 180, 0), 3)
                        cv2.line(frame, elbow, wrist, (0, 180, 0), 3)

                        for point in (shoulder, elbow, wrist):
                            cv2.circle(frame, point, 6, (0, 0, 255), -1)

                    flare_visible = all(
                        is_visible(point)
                        for point in (shoulder_lm, elbow_lm, hip_lm)
                    )

                    if flare_visible:
                        shoulder = to_pixel(shoulder_lm, width, height)
                        elbow = to_pixel(elbow_lm, width, height)
                        hip = to_pixel(hip_lm, width, height)

                        flare_angle = calculate_angle(elbow, shoulder, hip)
                        if flare_angle is not None:
                            elbow_flare_angles.append(flare_angle)

                shoulder_center = average_visible_points(
                    landmarks, [11, 12], width, height
                )
                hip_center = average_visible_points(
                    landmarks, [23, 24], width, height
                )
                ankle_center = average_visible_points(
                    landmarks, [27, 28], width, height
                )

                if (
                    shoulder_center is not None
                    and hip_center is not None
                    and ankle_center is not None
                ):
                    body_angle = calculate_signed_body_angle(
                        shoulder_center,
                        hip_center,
                        ankle_center,
                    )

                    if body_angle is not None:
                        shoulder_point = tuple(map(int, shoulder_center))
                        hip_point = tuple(map(int, hip_center))
                        ankle_point = tuple(map(int, ankle_center))

                        cv2.line(
                            frame,
                            shoulder_point,
                            hip_point,
                            (255, 0, 0),
                            3,
                        )
                        cv2.line(
                            frame,
                            hip_point,
                            ankle_point,
                            (255, 0, 0),
                            3,
                        )

                        if body_angle < 165:
                            body_feedback = "Hips too high"
                        elif body_angle > 190:
                            body_feedback = "Hips too low"
                        elif 170 <= body_angle <= 185:
                            body_feedback = "Body line steady"
                        else:
                            body_feedback = "Check body alignment"
                    else:
                        body_feedback = "Turn slightly side-on"
                else:
                    body_feedback = "Show shoulders, hips, ankles"

            average_elbow_angle = (
                sum(elbow_angles) / len(elbow_angles)
                if elbow_angles
                else None
            )
            average_flare_angle = (
                sum(elbow_flare_angles) / len(elbow_flare_angles)
                if elbow_flare_angles
                else None
            )

            now = time.monotonic()

            if average_elbow_angle is None:
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

                phase = "Show arms and body" if not is_armed else "Tracking"

            else:
                tracking_lost_since = None
                movement_progress = max(
                    0.0,
                    min(
                        1.0,
                        (TOP_ANGLE - average_elbow_angle)
                        / (TOP_ANGLE - BOTTOM_ANGLE),
                    ),
                )

                body_is_straight = (
                    body_angle is not None
                    and 170 <= body_angle <= 190
                )
                start_pose_detected = (
                    average_elbow_angle >= TOP_ANGLE
                    and body_is_straight
                )

                if not is_armed:
                    if start_pose_detected:
                        start_pose_frames += 1
                        phase = (
                            f"Hold start "
                            f"{start_pose_frames}/{READY_HOLD_FRAMES}"
                        )

                        if start_pose_frames >= READY_HOLD_FRAMES:
                            is_armed = True
                            arm_is_at_top = True
                            bottom_reached = False
                            phase = "Ready - lower"
                    else:
                        start_pose_frames = 0
                        phase = "Get into plank position"

                elif average_elbow_angle <= BOTTOM_ANGLE and arm_is_at_top:
                    bottom_reached = True
                    arm_is_at_top = False
                    phase = "Push up"

                elif average_elbow_angle >= TOP_ANGLE:
                    if bottom_reached:
                        repetitions += 1
                        bottom_reached = False

                    arm_is_at_top = True
                    phase = "Lower"

                elif bottom_reached:
                    phase = "Push up"
                elif arm_is_at_top:
                    phase = "Lower"
                else:
                    phase = "Move between top and bottom"

            if average_flare_angle is not None:
                if 30 <= average_flare_angle <= 45:
                    elbow_feedback = "Elbows near target"
                elif average_flare_angle < 30:
                    elbow_feedback = "Elbows close to body"
                elif average_flare_angle <= 60:
                    elbow_feedback = "Elbows somewhat wide"
                else:
                    elbow_feedback = "Elbows wide; adjust comfortably"

            draw_panel(
                frame,
                repetitions,
                phase,
                body_feedback,
                elbow_feedback,
                movement_progress,
            )

            cv2.imshow("Push-up form check", frame)

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
        "Отжимания",
        repetitions,
    ])

print(f"Тренировка сохранена: {repetitions} повторов.")