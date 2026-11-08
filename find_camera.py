import cv2

stop_search = False

for camera_index in range(10):
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        camera.release()
        continue

    print(f"Проверяю камеру {camera_index}. Нажми Q, чтобы проверить следующую.")

    while True:
        success, frame = camera.read()

        if not success:
            print(f"Камера {camera_index} не передала изображение.")
            break

        cv2.imshow(f"Камера {camera_index}", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 27:  # Esc — завершить поиск
            stop_search = True
            break

    camera.release()
    cv2.destroyAllWindows()

    if stop_search:
        break

print("Поиск завершён.")