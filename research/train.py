"""Train, tune only on development people, then evaluate on held-out people."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid

from features import CLASSES, build_table, read_repetitions

ROOT = Path(__file__).resolve().parent
SEED = 42
# Fixed before any model was trained. Never changed based on test results.
TEST_PEOPLE = [8, 9]
BASELINE = {"n_estimators": 50, "max_depth": 3, "min_samples_leaf": 3}
GRID = {"n_estimators": [50, 150], "max_depth": [3, 6, None], "min_samples_leaf": [1, 3]}


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_features():
    from download_data import download
    folder = download()
    records, audit = read_repetitions(folder)
    values, names = build_table(records, "geometry")
    with (folder / "features.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["person", "video", "repetition", "label", *names])
        for record, vector in zip(records, values):
            writer.writerow([record["person"], record["video"], record["repetition"], record["label"], *vector])
    write_json(folder / "audit.json", audit)


def read_features():
    path = ROOT / "data" / "features.csv"
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        names = reader.fieldnames[4:]
        rows = list(reader)
    keys = [(row["video"], row["repetition"]) for row in rows]
    assert len(set(keys)) == len(keys), "Duplicate repetitions"
    assert not set(names) & {"person", "video", "repetition", "label", "is_correct", "movement"}
    x = np.array([[float(row[name]) for name in names] for row in rows])
    y = np.array([row["label"] for row in rows])
    people = np.array([int(row["person"]) for row in rows])
    assert np.isfinite(x).all() and set(y) == set(CLASSES)
    return x, y, people, names, rows


def make_model(params):
    return RandomForestClassifier(**params, random_state=SEED, class_weight="balanced", n_jobs=1)


def score(y_true, predicted):
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=CLASSES).tolist(),
        "classification_report": classification_report(y_true, predicted, labels=CLASSES, output_dict=True, zero_division=0),
    }


def evaluate_cv(params, x, y, groups):
    folds = []
    for train, validation in LeaveOneGroupOut().split(x, y, groups):
        model = make_model(params)
        model.fit(x[train], y[train])
        predicted = model.predict(x[validation])
        metrics = score(y[validation], predicted)
        folds.append({"person": int(groups[validation][0]),
                      "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
    return {
        "params": params,
        "mean_macro_f1": float(np.mean([fold["macro_f1"] for fold in folds])),
        "mean_accuracy": float(np.mean([fold["accuracy"] for fold in folds])),
        "folds": folds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-features", action="store_true", help="Download raw tables and recompute features")
    args = parser.parse_args()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    if args.rebuild_features or not (ROOT / "data" / "features.csv").exists():
        prepare_features()
    x, y, people, names, rows = read_features()
    test = np.isin(people, TEST_PEOPLE)
    development = ~test
    assert not set(people[test]) & set(people[development])
    assert set(y[test]) == set(CLASSES) and set(y[development]) == set(CLASSES)
    x_train, y_train, groups = x[development], y[development], people[development]
    write_json(output / "experiment_plan.json", {
        "seed": SEED, "test_people": TEST_PEOPLE, "baseline": BASELINE, "grid": GRID,
        "selection_metric": "mean macro-F1 over leave-one-development-person-out folds; per-fold F1 uses labels present in truth or predictions",
        "notes": "Test people are never used for parameter selection. Same features and split for both models.",
    })
    comparisons = []
    for params in ParameterGrid(GRID):
        result = evaluate_cv(params, x_train, y_train, groups)
        comparisons.append(result)
        print("CV:", params, "macro-F1:", round(result["mean_macro_f1"], 4), flush=True)
    # Stable first maximum; no choice based on held-out scores.
    best_cv = max(comparisons, key=lambda result: result["mean_macro_f1"])
    baseline_cv = next(result for result in comparisons if result["params"] == BASELINE)
    write_json(output / "validation_search.json", comparisons)
    result = {
        "classes": CLASSES, "seed": SEED, "feature_count": len(names),
        "train_people": sorted(set(groups.tolist())), "test_people": TEST_PEOPLE,
        "train_repetitions": int(development.sum()), "test_repetitions": int(test.sum()),
        "feature_file_sha256": hashlib.sha256((ROOT / "data" / "features.csv").read_bytes()).hexdigest(),
        "class_counts": {
            split: {label: int(np.sum(y[mask] == label)) for label in CLASSES}
            for split, mask in (("development", development), ("test", test))
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "models": {},
    }
    all_predictions = {}
    for name, validation in (("baseline", baseline_cv), ("tuned", best_cv)):
        model = make_model(validation["params"])
        started = time.perf_counter()
        model.fit(x_train, y_train)
        elapsed = time.perf_counter() - started
        prediction = model.predict(x[test])
        all_predictions[name] = prediction
        metrics = score(y[test], prediction)
        # Exercise-only accuracy is separate from the harder four-class task.
        metrics["exercise_accuracy"] = float(accuracy_score(
            [label.split("_")[0] for label in y[test]],
            [label.split("_")[0] for label in prediction],
        ))
        metrics["correctness_accuracy"] = float(accuracy_score(
            [label.split("_", 1)[1] for label in y[test]],
            [label.split("_", 1)[1] for label in prediction],
        ))
        metrics["per_person"] = {
            str(person): score(y[test][people[test] == person], prediction[people[test] == person])
            for person in TEST_PEOPLE
        }
        result["models"][name] = {
            "params": validation["params"], "cv_macro_f1": validation["mean_macro_f1"],
            "train_accuracy": float(accuracy_score(y_train, model.predict(x_train))),
            "fit_seconds": elapsed, **metrics,
        }
        if name == "tuned":
            joblib.dump({"model": model, "features": names, "classes": CLASSES}, output / "model.joblib")
            importances = sorted(zip(names, model.feature_importances_), key=lambda item: -item[1])
            with (output / "feature_importance.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["feature", "importance"])
                writer.writerows(importances)
    baseline, tuned = result["models"]["baseline"], result["models"]["tuned"]
    result["delta_accuracy_percentage_points"] = 100 * (tuned["accuracy"] - baseline["accuracy"])
    result["delta_macro_f1"] = tuned["macro_f1"] - baseline["macro_f1"]
    write_json(output / "metrics.json", result)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["person", "video", "repetition", "true_label", "baseline", "tuned"])
        for row, first, second in zip(np.array(rows)[test], all_predictions["baseline"], all_predictions["tuned"]):
            writer.writerow([row["person"], row["video"], row["repetition"], row["label"], first, second])
    with (output / "split_manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["person", "video", "repetition", "label", "split"])
        for row, is_test in zip(rows, test):
            writer.writerow([row["person"], row["video"], row["repetition"], row["label"], "test" if is_test else "development"])
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "validation_macro_f1", "test_accuracy", "test_macro_f1", "exercise_accuracy", "correctness_accuracy"])
        for name, values in result["models"].items():
            writer.writerow([name, values["cv_macro_f1"], values["accuracy"], values["macro_f1"], values["exercise_accuracy"], values["correctness_accuracy"]])
    print(json.dumps({"models": result["models"], "delta_accuracy_percentage_points": result["delta_accuracy_percentage_points"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
