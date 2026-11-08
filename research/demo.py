"""Demonstrate the saved model on existing held-out examples; no camera."""
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
TRANSLATIONS = {
    "pushup_correct": "отжимание — правильное по разметке",
    "pushup_incorrect": "отжимание — неправильное по разметке",
    "squat_correct": "приседание — правильное по разметке",
    "squat_incorrect": "приседание — неправильное по разметке",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Index of a held-out repetition (0..61)")
    args = parser.parse_args()
    # This file is our own saved model, not an arbitrary downloaded pickle.
    bundle = joblib.load(ROOT / "results" / "model.joblib")
    with (ROOT / "data" / "features.csv").open(encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if int(row["person"]) in (8, 9)]
    if not 0 <= args.index < len(rows):
        parser.error(f"index must be between 0 and {len(rows) - 1}")
    row = rows[args.index]
    vector = np.array([[float(row[name]) for name in bundle["features"]]])
    predicted = bundle["model"].predict(vector)[0]
    print("Пример из готового датасета. Камера не используется.")
    print(f"Участник: {row['person']}; запись: {row['video']}; повтор: {row['repetition']}")
    print("Метка авторов:", TRANSLATIONS[row["label"]])
    print("Ответ модели:", TRANSLATIONS[predicted])
    print("Совпадение:", "ДА" if predicted == row["label"] else "НЕТ")
    print("Это учебное предсказание метки, не медицинская оценка техники.")


if __name__ == "__main__":
    main()
