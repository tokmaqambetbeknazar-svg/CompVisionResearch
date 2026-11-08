"""Общая камера и простой цикл повторов для подтягиваний и скручиваний."""

import csv
import math
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
HOLD_SECONDS = 0.15
VISIBILITY = 0.6
EXERCISES = {
    "pullup": {
        "name": "Подтягивания", "title": "Pull-ups",
        "start": "Hang with straight arms", "move": "Pull up", "return": "Lower to straight arms",
        "instructions": "Поставь камеру сбоку или немного спереди. В кадре должны быть лицо, руки и таз. Начни с виса на прямых руках.",
    },
    "crunch": {
        "name": "Скручивания", "title": "Crunches",
        "start": "Lie down, bend knees", "move": "Lift shoulders", "return": "Lower shoulders",
        "instructions": "Поставь камеру сбоку примерно на уровне пола. Ляг на спину, согни колени. В кадре нужны плечи, таз, колени и стопы.",
    },
}


def angle(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    length = math.hypot(*ab) * math.hypot(*cb)
    if length == 0:
        return None
    cosine = (ab[0] * cb[0] + ab[1] * cb[1]) / length
    return math.degrees(math.acos(max(-1, min(1, cosine))))


class RepetitionCounter:
    """start -> target -> start: только полный цикл добавляет повтор."""
    def __init__(self):
        self.repetitions = 0
        self.stage = None
        self.candidate = None
        self.since = None

    def update(self, position, now):
        # lost — потеря точек; middle — видимая промежуточная поза.
        if position == "lost":
            self.stage = self.candidate = self.since = None
            return
        if position != self.candidate:
            self.candidate = position
            self.since = now
        if position == "middle" or now - self.since < HOLD_SECONDS:
            return
        if position == "start":
            if self.stage == "target":
                self.repetitions += 1
            self.stage = "start"
        elif position == "target" and self.stage == "start":
            self.stage = "target"


def measure(landmarks, side, width, height, exercise):
    """Возвращает положение, измеренный угол и точки для рисунка."""
    if not all(
        (landmarks[i].visibility or 0) >= VISIBILITY
        and 0.02 <= landmarks[i].x <= 0.98
        and 0.02 <= landmarks[i].y <= 0.98
        for i in side
    ):
        return "lost", None, []
    points = [(landmarks[i].x * width, landmarks[i].y * height) for i in side]
    if exercise == "pullup":
        shoulder, elbow, wrist, hip, mouth = points
        value = angle(shoulder, elbow, wrist)
        # Руки должны быть над плечами: обычное сгибание рук не подходит.
        if value is None or wrist[1] >= shoulder[1] or hip[1] <= shoulder[1]:
            return "lost", value, points[:3]
        if value >= 155:
            position = "start"
        elif value <= 80 and mouth[1] < wrist[1]:
            position = "target"
        else:
            position = "middle"
        return position, value, points[:3]

    shoulder, hip, knee, ankle = points
    knee_angle = angle(hip, knee, ankle)
    thigh_length = math.dist(hip, knee)
    torso_length = math.dist(shoulder, hip)
    if thigh_length == 0 or torso_length == 0 or knee_angle is None:
        return "lost", None, points
    # Ожидаем согнутые колени и таз примерно на высоте лодыжки.
    if knee_angle > 140 or abs(hip[1] - ankle[1]) > 0.5 * thigh_length:
        return "lost", None, points
    value = math.degrees(math.atan2(hip[1] - shoulder[1], abs(shoulder[0] - hip[0])))
    if value < -15 or value > 65:
        return "lost", value, points
    position = "start" if value <= 15 else "target" if value >= 30 else "middle"
    return position, value, points


def save_workout(name, repetitions):
    path = PROJECT_DIR / "workout_log.csv"
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(["Дата и время", "Упражнение", "Повторы"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, repetitions])


def run_exercise(exercise):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    config = EXERCISES[exercise]
    sides = [(11, 13, 15, 23, 9), (12, 14, 16, 24, 10)] if exercise == "pullup" else [
        (11, 23, 25, 27), (12, 24, 26, 28)
    ]
    model = PROJECT_DIR / "pose_landmarker_lite.task"
    if not model.is_file():
        model = Path(r"C:\CVModels\pose_landmarker_lite.task")
    if not model.is_file():
        raise FileNotFoundError("Положи pose_landmarker_lite.task в папку проекта.")
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
    )
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Не удалось открыть камеру. Проверь, не занята ли она.")

    counter = RepetitionCounter()
    selected_side = None
    timestamp = -1
    print(config["instructions"])
    print("Q или Esc в окне видео — закончить и сохранить результат.")
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Не удалось получить изображение с камеры.")
                height, width = frame.shape[:2]
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamp = max(timestamp + 1, time.monotonic_ns() // 1_000_000)
                result = landmarker.detect_for_video(image, timestamp)
                position, value, points = "lost", None, []
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    measurements = {side: measure(landmarks, side, width, height, exercise) for side in sides}
                    if selected_side is None or measurements[selected_side][0] == "lost":
                        counter.update("lost", time.monotonic())
                        selected_side = max(
                            (side for side in sides if measurements[side][0] != "lost"),
                            key=lambda side: sum(landmarks[i].visibility or 0 for i in side),
                            default=None,
                        )
                    if selected_side is not None:
                        position, value, points = measurements[selected_side]
                else:
                    selected_side = None
                counter.update(position, time.monotonic())
                if position == "lost":
                    message = "Show body / check camera view"
                elif counter.stage is None:
                    message = config["start"]
                elif counter.stage == "target":
                    message = config["return"]
                else:
                    message = config["move"]
                pixels = [tuple(map(int, p)) for p in points]
                for a, b in zip(pixels, pixels[1:]):
                    cv2.line(frame, a, b, (0, 200, 0), 3)
                for point in pixels:
                    cv2.circle(frame, point, 6, (0, 200, 255), -1)
                cv2.rectangle(frame, (8, 8), (min(width - 8, 560), 142), (30, 30, 30), -1)
                lines = [f"{config['title']}: {counter.repetitions}", message,
                         f"Angle: {value:.0f}" if value is not None else "Angle: --",
                         "Q / Esc: finish and save"]
                for row, text in enumerate(lines):
                    cv2.putText(frame, text, (18, 36 + row * 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (240, 240, 240), 2)
                cv2.imshow(config["title"], frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(config["title"], cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        save_workout(config["name"], counter.repetitions)
        print(f"{config['name']}: сохранено повторов — {counter.repetitions}")
