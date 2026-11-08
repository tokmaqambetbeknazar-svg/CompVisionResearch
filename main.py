import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

MUSCLE_GROUPS = {
    "1": {
        "name": "Мышцы груди (грудные)",
        "exercises": [
    ("Отжимания", "pushup_counter.py"),
    ("Отжимания с ногами на возвышении", "decline_pushup_counter.py"),
    ("Дипы на брусьях", "dip_counter.py"),
],
    },
    "2": {
        "name": "Мышцы спины",
        "exercises": [],
    },
    "3": {
        "name": "Мышцы ног и ягодиц",
        "exercises": [
            ("Приседания", "squat_counter.py"),
        ],
    },
    "4": {
        "name": "Мышцы плеч (дельтовидные мышцы)",
        "exercises": [
            ("Жим гантелей вверх", "shoulder_press_counter.py"),
        ],
    },
    "5": {
        "name": "Мышцы рук",
        "exercises": [
            ("Подъем гантелей с супинацией", "biceps_supination_counter.py"),
        ],
    },
    "6": {
        "name": "Мышцы живота и кора (пресс)",
        "exercises": [],
    },
}


def run_script(script_name):
    script_path = PROJECT_DIR / script_name

    if not script_path.exists():
        print(f"Файл упражнения не найден: {script_name}")
        return

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIR,
        check=False,
    )

    if result.returncode != 0:
        print(f"Программа завершилась с ошибкой. Код: {result.returncode}")


def choose_muscle_group(group):
    exercises = group["exercises"]

    if not exercises:
        print(f"\nДля раздела «{group['name']}» счётчиков пока нет.")
        return

    while True:
        print(f"\n=== {group['name']} ===")

        for number, (exercise_name, _) in enumerate(exercises, start=1):
            print(f"{number}. {exercise_name}")

        print("0. Назад")
        choice = input("Выбери упражнение: ").strip()

        if choice == "0":
            return

        if not choice.isdigit() or not 1 <= int(choice) <= len(exercises):
            print("Нет такого пункта меню.")
            continue

        exercise_name, script_name = exercises[int(choice) - 1]
        print(f"\nЗапускаю: {exercise_name}")
        run_script(script_name)


def main():
    while True:
        print("\n=== ИИ-фитнес-тренер ===")

        for number, group in MUSCLE_GROUPS.items():
            print(f"{number}. {group['name']}")

        print("7. История тренировок")
        print("0. Выход")

        choice = input("Выбери группу мышц: ").strip()

        if choice == "0":
            print("До встречи!")
            return

        if choice == "7":
            run_script("workout_history.py")
        elif choice in MUSCLE_GROUPS:
            choose_muscle_group(MUSCLE_GROUPS[choice])
        else:
            print("Нет такого пункта меню.")


if __name__ == "__main__":
    main()
