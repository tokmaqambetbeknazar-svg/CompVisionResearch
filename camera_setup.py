import math
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "pose_landmarker_lite.task"
VISIBILITY_THRESHOLD = 0.5
TARGET_MIN_DEGREES = 30
TARGET_MAX_DEGREES = 60

# Точки для проверки кадрирования: плечи, локти, бёдра и лодыжки.
REQUIRED_POINTS = [11, 12, 13, 14, 23, 24, 27, 28]


def is_visible(point):
    return (point.visibility or 0) >= VISIBILITY_THRESHOLD


options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Не удалось открыть камеру. Проверь подключение и разрешения Windows.")

try:
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        print(
            "Встань в полный рост перед камерой и повернись к ней примерно на 45 градусов. "
            "Нажми Q в окне видео, чтобы выйти."
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

            messages = ["Step into camera view"]
            estimated_angle = None

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                for index, point in enumerate(landmarks):
                    if is_visible(point):
                        x = int(point.x * width)
                        y = int(point.y * height)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                visible_points = [
                    landmarks[index]
                    for index in REQUIRED_POINTS
                    if is_visible(landmarks[index])
                ]

                if len(visible_points) == len(REQUIRED_POINTS):
                    messages = ["Key body points detected"]

                    min_x = min(point.x for point in visible_points)
                    max_x = max(point.x for point in visible_points)
                    min_y = min(point.y for point in visible_points)
                    max_y = max(point.y for point in visible_points)

                    if min_x < 0.04 or max_x > 0.96 or min_y < 0.04 or max_y > 0.96:
                        messages.append("Move farther back to fit in the frame")
                    else:
                        messages.append("Full body fits in the frame")

                    if result.pose_world_landmarks:
                        world = result.pose_world_landmarks[0]
                        left_shoulder = world[11]
                        right_shoulder = world[12]

                        shoulder_dx = abs(left_shoulder.x - right_shoulder.x)
                        shoulder_dz = abs(left_shoulder.z - right_shoulder.z)

                        if shoulder_dx > 0.001:
                            estimated_angle = math.degrees(
                                math.atan2(shoulder_dz, shoulder_dx)
                            )

                            if estimated_angle < TARGET_MIN_DEGREES:
                                messages.append("Turn your body slightly sideways")
                            elif estimated_angle > TARGET_MAX_DEGREES:
                                messages.append("Turn your body slightly toward the camera")
                            else:
                                messages.append("Approximate angle is near 45 degrees")
                else:
                    messages = ["Show shoulders, elbows, hips and ankles"]
            else:
                messages = ["Step into camera view"]

            if estimated_angle is not None:
                messages.insert(0, f"Estimated body angle: {int(estimated_angle)} deg")

            for line_number, message in enumerate(messages):
                cv2.putText(
                    frame,
                    message,
                    (15, 30 + line_number * 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Camera setup guide", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    camera.release()
    cv2.destroyAllWindows()