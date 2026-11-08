import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "pose_landmarker_lite.task"
VISIBILITY_THRESHOLD = 0.5
BENT_ARM_ANGLE = 110
EXTENDED_ARM_ANGLE = 155


def calculate_angle(shoulder, elbow, wrist, frame_width, frame_height):
    """Вычисляет угол локтя с учётом размеров изображения."""
    upper_arm = (
        (shoulder.x - elbow.x) * frame_width,
        (shoulder.y - elbow.y) * frame_height,
    )
    forearm = (
        (wrist.x - elbow.x) * frame_width,
        (wrist.y - elbow.y) * frame_height,
    )

    upper_length = math.hypot(*upper_arm)
    forearm_length = math.hypot(*forearm)

    if upper_length == 0 or forearm_length == 0:
        return None

    dot_product = (
        upper_arm[0] * forearm[0]
        + upper_arm[1] * forearm[1]
    )
    cosine = dot_product / (upper_length * forearm_length)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Не удалось открыть камеру. Проверь подключение и разрешения Windows."
    )

repetitions = 0
arm_was_bent = False

try:
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        print("Встань боком к камере. Нажми Q в окне видео, чтобы выйти.")

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

            angle = None
            phase = "Show your right arm"

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # Правая рука: плечо 12, локоть 14, запястье 16.
                shoulder = landmarks[12]
                elbow = landmarks[14]
                wrist = landmarks[16]

                visible = all(
                    (point.visibility or 0) > VISIBILITY_THRESHOLD
                    for point in (shoulder, elbow, wrist)
                )

                if visible:
                    angle = calculate_angle(
                        shoulder,
                        elbow,
                        wrist,
                        width,
                        height,
                    )

                    if angle is not None:
                        if angle < BENT_ARM_ANGLE:
                            arm_was_bent = True
                            phase = "Press overhead"

                        elif angle > EXTENDED_ARM_ANGLE:
                            if arm_was_bent:
                                repetitions += 1
                                arm_was_bent = False

                            phase = "Lower to shoulder level"

                        elif arm_was_bent:
                            phase = "Continue pressing"

                        else:
                            phase = "Lower to shoulder level"

                        points = []

                        for point in (shoulder, elbow, wrist):
                            x = int(point.x * width)
                            y = int(point.y * height)
                            points.append((x, y))
                            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

                        cv2.line(
                            frame,
                            points[0],
                            points[1],
                            (0, 255, 0),
                            3,
                        )
                        cv2.line(
                            frame,
                            points[1],
                            points[2],
                            (0, 255, 0),
                            3,
                        )
                else:
                    phase = "Show shoulder, elbow and wrist"
            else:
                phase = "Step into camera view"

            if angle is not None:
                cv2.putText(
                    frame,
                    f"Elbow angle: {int(angle)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.putText(
                frame,
                f"Reps: {repetitions}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                phase,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Shoulder press counter", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    camera.release()
    cv2.destroyAllWindows()


log_path = Path("workout_log.csv")
file_is_new = not log_path.exists() or log_path.stat().st_size == 0

with log_path.open("a", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)

    if file_is_new:
        writer.writerow(["Дата и время", "Упражнение", "Повторы"])

    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Жим гантелей вверх",
        repetitions,
    ])

print(f"Тренировка сохранена: {repetitions} повторов.")