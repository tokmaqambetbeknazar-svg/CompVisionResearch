"""Check provenance, split isolation, saved predictions and feature invariants."""
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from features import build_table, feature_vector, read_repetitions
from train import BASELINE, CLASSES, ROOT, TEST_PEOPLE, make_model, read_features


def main():
    x, y, people, names, rows = read_features()
    test = np.isin(people, TEST_PEOPLE)
    development = ~test
    assert not set(people[test]) & set(people[development])
    assert not set(np.array([row['video'] for row in rows])[test]) & set(np.array([row['video'] for row in rows])[development])
    assert len(x) == 287 and len(names) == 48
    result = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((ROOT / "data" / "features.csv").read_bytes()).hexdigest() == result["feature_file_sha256"]
    assert result["train_people"] == sorted(set(people[development].tolist()))
    assert result["train_repetitions"] == int(development.sum())
    assert result["test_repetitions"] == int(test.sum())
    search = json.loads((ROOT / "results" / "validation_search.json").read_text(encoding="utf-8"))
    for candidate in search:
        assert {fold["person"] for fold in candidate["folds"]} == set(people[development])
        assert not {fold["person"] for fold in candidate["folds"]} & set(TEST_PEOPLE)
    assert result["models"]["tuned"]["params"] == max(search, key=lambda row: row["mean_macro_f1"])["params"]
    bundle = joblib.load(ROOT / "results" / "model.joblib")
    assert bundle["features"] == names
    predicted = bundle["model"].predict(x[test])
    with (ROOT / "results" / "predictions.csv").open(encoding="utf-8", newline="") as file:
        saved = list(csv.DictReader(file))
    assert [row["true_label"] for row in saved] == y[test].tolist()
    assert [row["tuned"] for row in saved] == predicted.tolist()
    assert np.isclose(accuracy_score(y[test], predicted), result["models"]["tuned"]["accuracy"])
    assert np.isclose(f1_score(y[test], predicted, average="macro"), result["models"]["tuned"]["macro_f1"])
    baseline = make_model(BASELINE).fit(x[development], y[development])
    assert baseline.predict(x[test]).tolist() == [row["baseline"] for row in saved]
    # Optional full trace back to coordinate tables (not needed for offline use).
    if (ROOT / "data" / "original_segmentation.csv").exists():
        records, audit = read_repetitions(ROOT / "data")
        recomputed, feature_names = build_table(records, "geometry")
        assert names == feature_names and np.allclose(x, recomputed)
        pose = records[0]["positions"]
        original = feature_vector(pose, "geometry")[0]
        transformed = feature_vector(pose * 2.3 + np.array([2., 3., -1.]), "geometry")[0]
        assert np.allclose(original, transformed, atol=1e-5)
        mirrored = pose.copy()
        mirrored[:, :, 0] *= -1
        assert np.allclose(original, feature_vector(mirrored, "geometry")[0], atol=1e-5)
        assert audit == json.loads((ROOT / "data" / "audit.json").read_text(encoding="utf-8"))
    print("PASS: original data, feature invariants, no subject leakage, validation-only selection, model reload, predictions and metrics")


if __name__ == "__main__":
    main()
