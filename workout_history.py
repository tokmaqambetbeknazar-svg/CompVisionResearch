import csv
from collections import defaultdict
from pathlib import Path


log_path = Path("workout_log.csv")

if not log_path.exists():
    print("Журнал пока пуст. Сначала заверши тренировку.")
else:
    with log_path.open("r", newline="", encoding="utf-8-sig") as file:
        sessions = list(csv.DictReader(file))

    if not sessions:
        print("В журнале пока нет тренировок.")
    else:
        repetitions_by_exercise = defaultdict(int)

        print("История тренировок:")
        for session in sessions:
            exercise = session["Упражнение"]
            repetitions = int(session["Повторы"])
            repetitions_by_exercise[exercise] += repetitions

            print(
                f'{session["Дата и время"]} — '
                f'{exercise}: {repetitions} повторов'
            )

        print("\nИтого по упражнениям:")
        for exercise, repetitions in repetitions_by_exercise.items():
            print(f"{exercise}: {repetitions} повторов")

        total_repetitions = sum(repetitions_by_exercise.values())
        print(f"\nТренировок: {len(sessions)}")
        print(f"Всего повторов: {total_repetitions}")