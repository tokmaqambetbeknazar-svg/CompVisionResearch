try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    if exc.name != "ultralytics":
        raise
    raise SystemExit(
        "Ultralytics is not installed. Run: python -m pip install ultralytics"
    ) from exc
import cv2
import numpy as np

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

# точки позы: плечо, локоть, запястье
LEFT = (5, 7, 9)
RIGHT = (6, 8, 10)

DOWN_ANGLE = 150   # рука выпрямлена
UP_ANGLE = 50      # рука согнута
DRIFT_LIMIT = 0.4  # насколько локоть может уйти вперёд

stage = "down"
good_reps = 0
bad_reps = 0
rep_has_error = False
smooth_angle = None


def angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = model(frame, verbose=False)[0]

    warnings = []
    ang = None

    if result.keypoints is not None and len(result.boxes) > 0:
        xy = result.keypoints.xy[0].cpu().numpy()
        conf = result.keypoints.conf[0].cpu().numpy()

        # выбираем руку, которую камера видит лучше
        side = LEFT if conf[list(LEFT)].sum() > conf[list(RIGHT)].sum() else RIGHT
        s, e, w = side

        if min(conf[s], conf[e], conf[w]) > 0.5:
            sh, el, wr = xy[s], xy[e], xy[w]

            raw = angle(sh, el, wr)
            smooth_angle = raw if smooth_angle is None else 0.6 * smooth_angle + 0.4 * raw
            ang = smooth_angle

            # проверка 1: локоть не должен "гулять" вперёд
            upper_arm = np.linalg.norm(sh - el) + 1e-6
            drift = abs(el[0] - sh[0]) / upper_arm
            if drift > DRIFT_LIMIT:
                warnings.append("Keep elbow close to body")

            # счётчик повторов
            if ang > DOWN_ANGLE:
                if stage == "up":
                    if rep_has_error:
                        bad_reps += 1
                    else:
                        good_reps += 1
                stage = "down"
                rep_has_error = False
            elif ang < UP_ANGLE and stage == "down":
                stage = "up"

            if warnings and stage != "down":
                rep_has_error = True
            elif warnings:
                rep_has_error = True

            # рисуем руку
            pts = [tuple(map(int, p)) for p in (sh, el, wr)]
            color = (0, 0, 255) if warnings else (0, 255, 0)
            cv2.line(frame, pts[0], pts[1], color, 4)
            cv2.line(frame, pts[1], pts[2], color, 4)
            for p in pts:
                cv2.circle(frame, p, 8, (255, 255, 255), -1)
            cv2.putText(frame, f"{int(ang)}", (pts[1][0] + 10, pts[1][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            warnings.append("Show your arm from the side")
    else:
        warnings.append("No person found")

    # панель статистики
    cv2.rectangle(frame, (0, 0), (360, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"Good reps: {good_reps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Bad reps: {bad_reps}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Stage: {stage}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for i, msg in enumerate(warnings):
        cv2.putText(frame, msg, (10, 150 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("AI Trainer", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()