"""Счётчик приседаний: стоя -> присед -> снова стоя."""

import csv
import math
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
TOP_ANGLE = 160
BOTTOM_ANGLE = 95
HOLD_SECONDS = 0.15
VISIBILITY_THRESHOLD = 0.6


def calculate_angle(a, b, c):
    """Угол в колене B между бедром A и лодыжкой C."""
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    length = math.hypot(*ab) * math.hypot(*cb)
    if length == 0:
        return None
    cosine = (ab[0] * cb[0] + ab[1] * cb[1]) / length
    return math.degrees(math.acos(max(-1, min(1, cosine))))


class SquatCounter:
    def __init__(self):
        self.repetitions = 0
        self.stage = None
        self.candidate = None
        self.candidate_since = None

    def update(self, angle, now):
        # При потере ноги незавершённый повтор сбрасываем, счёт сохраняем.
        if angle is None:
            self.stage = None
            self.candidate = None
            self.candidate_since = None
            return "Show full body side-on"

        position = "up" if angle >= TOP_ANGLE else "down" if angle <= BOTTOM_ANGLE else None
        if position != self.candidate:
            self.candidate = position
            self.candidate_since = now

        # Небольшая выдержка защищает от одиночных скачков распознавания.
        if position is not None and now - self.candidate_since >= HOLD_SECONDS:
            if position == "up":
                if self.stage == "down":
                    self.repetitions += 1
                self.stage = "up"
            elif self.stage == "up":
                self.stage = "down"

        if self.stage is None:
            return "Stand straight to start"
        return "Stand up" if self.stage == "down" else "Squat down"


def save_workout(repetitions):
    log_path = PROJECT_DIR / "workout_log.csv"
    file_is_new = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        if file_is_new:
            writer.writerow(["Дата и время", "Упражнение", "Повторы"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Приседания",
            repetitions,
        ])


def main():
    # Импорты здесь позволяют проверять логику без камеры и тяжёлых библиотек.
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = PROJECT_DIR / "pose_landmarker_lite.task"
    if not model_path.is_file():
        model_path = Path(r"C:\CVModels\pose_landmarker_lite.task")
    if not model_path.is_file():
        raise FileNotFoundError("Положи pose_landmarker_lite.task рядом с squat_counter.py.")

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Не удалось открыть камеру. Проверь, не занята ли она.")

    counter = SquatCounter()
    selected_side = None
    last_timestamp = -1
    print("Встань боком к камере: плечи, таз, колени и стопы должны быть в кадре.")
    print("Начни стоя. Присядь и выпрямись. Q или Esc — закончить и сохранить.")

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Не удалось получить изображение с камеры.")
                height, width = frame.shape[:2]
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                timestamp = max(last_timestamp + 1, time.monotonic_ns() // 1_000_000)
                last_timestamp = timestamp
                result = landmarker.detect_for_video(image, timestamp)
                angle = None

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    # Плечо, таз, колено, лодыжка: левая либо правая сторона.
                    sides = [(11, 23, 25, 27), (12, 24, 26, 28)]
                    visible_sides = [
                        side for side in sides
                        if all(
                            (landmarks[i].visibility or 0) >= VISIBILITY_THRESHOLD
                            and 0.02 <= landmarks[i].x <= 0.98
                            and 0.02 <= landmarks[i].y <= 0.98
                            for i in side
                        )
                    ]
                    if selected_side not in visible_sides:
                        # Не соединяем половинки движения с разных ног.
                        counter.update(None, time.monotonic())
                        selected_side = max(
                            visible_sides,
                            key=lambda side: sum(landmarks[i].visibility or 0 for i in side),
                            default=None,
                        )
                    if selected_side is not None:
                        points = [
                            (int(landmarks[i].x * width), int(landmarks[i].y * height))
                            for i in selected_side
                        ]
                        angle = calculate_angle(*points[1:])
                        for a, b in zip(points, points[1:]):
                            cv2.line(frame, a, b, (0, 200, 0), 3)
                        for point in points:
                            cv2.circle(frame, point, 6, (0, 200, 255), -1)
                else:
                    selected_side = None

                message = counter.update(angle, time.monotonic())
                cv2.rectangle(frame, (8, 8), (min(width - 8, 520), 142), (30, 30, 30), -1)
                lines = [
                    f"SQUATS: {counter.repetitions}",
                    message,
                    f"Knee angle: {angle:.0f}" if angle is not None else "Knee angle: --",
                    "Side view | Q / Esc: finish",
                ]
                for row, text in enumerate(lines):
                    cv2.putText(frame, text, (18, 36 + row * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
                cv2.imshow("Squat counter", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty("Squat counter", cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        save_workout(counter.repetitions)
        print(f"Сохранено приседаний: {counter.repetitions}")


if __name__ == "__main__":
    main()
