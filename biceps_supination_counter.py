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
ARM_EXTENDED_ANGLE = 150
ARM_CURL_ANGLE = 65
READY_HOLD_FRAMES = 15
TOP_PAUSE_SECONDS = 1.0
TRACKING_LOSS_SECONDS = 0.75
MAX_ELBOW_DRIFT = 0.16
DRIFT_FRAMES_TO_REJECT = 3


def is_visible(point):
    return (point.visibility or 0) >= VISIBILITY_THRESHOLD


def to_pixel(point, width, height):
    return int(point.x * width), int(point.y * height)


def calculate_angle(point_a, point_b, point_c):
    """Вычисляет угол в локте по точкам плеча, локтя и запястья."""
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


def draw_panel(frame, repetitions, phase, elbow_feedback):
    height, width, _ = frame.shape
    left, top = 12, 12
    panel_width = min(535, width - 24)
    panel_height = min(124, height - 24)

    if panel_width < 230 or panel_height < 90:
        return

    right = left + panel_width
    bottom = top + panel_height

    overlay = frame.copy()
    draw_rounded_rectangle(
        overlay,
        (left, top),
        (right, bottom),
        (165, 165, 155),
        20,
    )
    draw_rounded_rectangle(
        overlay,
        (left + 4, top + 4),
        (right - 4, bottom - 4),
        (27, 28, 33),
        16,
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
        (text_x, top + 34),
        font,
        0.62,
        white,
        2,
        cv2.LINE_AA,
    )
    label_width = cv2.getTextSize("REPS:", font, 0.62, 2)[0][0]
    cv2.putText(
        frame,
        str(repetitions),
        (text_x + label_width + 8, top + 34),
        font,
        0.66,
        teal,
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (text_x, top + 46),
        (right - 18, top + 46),
        (75, 76, 82),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"PHASE: {phase}",
        (text_x, top + 73),
        font,
        0.42,
        white,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"ELBOW: {elbow_feedback}",
        (text_x, top + 103),
        font,
        0.38,
        gray,
        1,
        cv2.LINE_AA,
    )


if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        "Не найдена модель pose_landmarker_lite.task. "
        "Положи её в C:\\CVModels или рядом с biceps_supination_counter.py."
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
rep_in_progress = False
rep_form_valid = True
reached_top = False
top_hold_started = None
drift_frames = 0
tracking_lost_since = None
baseline_elbow_offset = None

try:
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        print(
            "Поставь камеру сбоку, чтобы были видны плечо, локоть, запястье "
            "и бедро. Нажми Q, чтобы закончить."
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
            elbow_offset = None
            phase = "Show one arm from the side"
            elbow_feedback = "Keep elbow beside your body"
            selected_points = None

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # Из двух рук выбираем ту, которую камера видит лучше.
                sides = [
                    (11, 13, 15, 23),  # левая сторона
                    (12, 14, 16, 24),  # правая сторона
                ]

                best_side = None
                best_visibility = -1.0

                for shoulder_i, elbow_i, wrist_i, hip_i in sides:
                    points = (
                        landmarks[shoulder_i],
                        landmarks[elbow_i],
                        landmarks[wrist_i],
                        landmarks[hip_i],
                    )
                    visibility = min(
                        (point.visibility or 0)
                        for point in points
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
                    hip = to_pixel(hip_lm, width, height)

                    arm_angle = calculate_angle(shoulder, elbow, wrist)

                    torso_length = math.hypot(
                        hip[0] - shoulder[0],
                        hip[1] - shoulder[1],
                    )

                    if torso_length > 0:
                        # Положение локтя относительно плеча,
                        # нормированное по длине корпуса.
                        elbow_offset = (
                            (elbow[0] - shoulder[0]) / torso_length,
                            (elbow[1] - shoulder[1]) / torso_length,
                        )

                    selected_points = (shoulder, elbow, wrist, hip)

                    cv2.line(frame, shoulder, elbow, (0, 180, 0), 3)
                    cv2.line(frame, elbow, wrist, (0, 180, 0), 3)
                    cv2.line(frame, shoulder, hip, (255, 0, 0), 2)

                    for point in selected_points:
                        cv2.circle(frame, point, 6, (0, 0, 255), -1)

            now = time.monotonic()

            if arm_angle is None or elbow_offset is None:
                start_pose_frames = 0
                top_hold_started = None

                if tracking_lost_since is None:
                    tracking_lost_since = now

                if (
                    is_armed
                    and now - tracking_lost_since >= TRACKING_LOSS_SECONDS
                ):
                    is_armed = False
                    rep_in_progress = False
                    rep_form_valid = True
                    reached_top = False
                    baseline_elbow_offset = None
                    drift_frames = 0

                phase = "Keep shoulder, elbow, wrist and hip visible"

            else:
                tracking_lost_since = None

                if not is_armed:
                    if arm_angle >= ARM_EXTENDED_ANGLE:
                        start_pose_frames += 1
                        phase = (
                            f"Hold start position "
                            f"{start_pose_frames}/{READY_HOLD_FRAMES}"
                        )

                        if start_pose_frames >= READY_HOLD_FRAMES:
                            is_armed = True
                            baseline_elbow_offset = elbow_offset
                            phase = "Ready - curl slowly"
                    else:
                        start_pose_frames = 0
                        phase = "Start with arm extended"

                else:
                    if (
                        not rep_in_progress
                        and arm_angle < ARM_EXTENDED_ANGLE
                    ):
                        rep_in_progress = True
                        rep_form_valid = True
                        reached_top = False
                        top_hold_started = None
                        drift_frames = 0

                    if rep_in_progress:
                        elbow_drift = math.hypot(
                            elbow_offset[0] - baseline_elbow_offset[0],
                            elbow_offset[1] - baseline_elbow_offset[1],
                        )

                        if elbow_drift > MAX_ELBOW_DRIFT:
                            drift_frames += 1
                        else:
                            drift_frames = 0

                        if drift_frames >= DRIFT_FRAMES_TO_REJECT:
                            rep_form_valid = False

                        if not rep_form_valid:
                            elbow_feedback = "Elbow moved - rep will not count"
                        elif elbow_drift > MAX_ELBOW_DRIFT:
                            elbow_feedback = "Keep elbow still"
                        else:
                            elbow_feedback = "Elbow steady"

                        if arm_angle <= ARM_CURL_ANGLE:
                            if top_hold_started is None:
                                top_hold_started = now

                            hold_duration = now - top_hold_started

                            if hold_duration >= TOP_PAUSE_SECONDS:
                                reached_top = True
                                phase = "Lower slowly"
                            else:
                                phase = f"Pause at top {hold_duration:.1f}/1.0 sec"

                        elif arm_angle < ARM_EXTENDED_ANGLE:
                            top_hold_started = None
                            phase = "Curl and turn palms up"

                        else:
                            if reached_top and rep_form_valid:
                                repetitions += 1
                                phase = "Rep counted - curl again"
                            elif not rep_form_valid:
                                phase = "Rep not counted - keep elbow still"
                            else:
                                phase = "Reach the top and pause"

                            rep_in_progress = False
                            rep_form_valid = True
                            reached_top = False
                            top_hold_started = None
                            drift_frames = 0

                            # Обновляем исходную точку на верхнем положении.
                            baseline_elbow_offset = elbow_offset
                            elbow_feedback = "Elbow steady"

                    else:
                        phase = "Curl and turn palms up"
                        elbow_feedback = "Keep elbow beside your body"

            draw_panel(
                frame,
                repetitions,
                phase,
                elbow_feedback,
            )

            cv2.imshow("Biceps curl with supination", frame)

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
        "Подъем гантелей на бицепс с супинацией",
        repetitions,
    ])

print(f"Тренировка сохранена: {repetitions} повторов.")