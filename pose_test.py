import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


model_path = "pose_landmarker_lite.task"

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
)

# Пары точек, которые соединяем линиями, чтобы показать скелет.
connections = [
    (11, 12),  # плечи
    (11, 13), (13, 15),  # левая рука
    (12, 14), (14, 16),  # правая рука
    (11, 23), (12, 24), (23, 24),  # корпус
    (23, 25), (25, 27),  # левая нога
    (24, 26), (26, 28),  # правая нога
]

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Не удалось открыть камеру. Проверь подключение и разрешения Windows.")

try:
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        print("Найди своё тело в кадре. Нажми Q в окне видео, чтобы выйти.")

        while True:
            success, frame = camera.read()

            if not success:
                raise RuntimeError("Не удалось получить изображение с камеры.")

            # MediaPipe принимает цветное изображение RGB, а камера даёт BGR.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(image, timestamp_ms)

            if result.pose_landmarks:
                height, width, _ = frame.shape
                landmarks = result.pose_landmarks[0]

                for start_index, end_index in connections:
                    start = landmarks[start_index]
                    end = landmarks[end_index]

                    if (start.visibility or 0) > 0.5 and (end.visibility or 0) > 0.5:
                        start_point = (int(start.x * width), int(start.y * height))
                        end_point = (int(end.x * width), int(end.y * height))
                        cv2.line(frame, start_point, end_point, (0, 255, 0), 3)

                for point in landmarks:
                    if (point.visibility or 0) > 0.5:
                        center = (int(point.x * width), int(point.y * height))
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)

            cv2.imshow("Проверка позы", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    camera.release()
    cv2.destroyAllWindows()