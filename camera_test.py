import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Не удалось открыть камеру. Проверь подключение и разрешения Windows.")

print("Камера работает. Нажми Q в окне видео, чтобы выйти.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Не удалось получить изображение с камеры.")
            break

        cv2.imshow("Проверка камеры", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()