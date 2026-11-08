"""One repetition becomes one feature row; no IDs or labels enter features."""
import csv
from collections import defaultdict

import numpy as np

JOINTS = [
    "LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow",
    "LeftWrist", "RightWrist", "LeftKnee", "RightKnee",
    "LeftAnkle", "RightAnkle", "LeftFootIndex", "RightFootIndex",
    "LeftHip", "RightHip",
]
CLASSES = ["pushup_correct", "pushup_incorrect", "squat_correct", "squat_incorrect"]


def read_repetitions(folder):
    with (folder / "original_segmentation.csv").open(encoding="utf-8-sig", newline="") as file:
        metadata = {
            (row["video_id"], int(row["repetition_number"])): row
            for row in csv.DictReader(file, delimiter=";")
            if int(row["exercise_id"]) in (3, 6)
        }
    records = []
    audit = {"raw_frames": 0, "raw_repetitions": 0, "excluded_mocap_repetitions": 0}
    for exercise, exercise_id in (("pushup", 3), ("squat", 6)):
        path = folder / f"rehab246_{exercise}_positions_standardized.csv"
        groups = defaultdict(list)
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                key = (row["subject_id"], int(row["repetition"]))
                groups[key].append(row)
                audit["raw_frames"] += 1
        for (video, repetition), rows in sorted(groups.items()):
            audit["raw_repetitions"] += 1
            meta = metadata[(video, repetition)]
            assert int(meta["exercise_id"]) == exercise_id
            rows.sort(key=lambda row: int(row["frame"]))
            expected_frames = int(meta["last_frame"]) - int(meta["first_frame"]) + 1
            assert len(rows) == expected_frames, (video, repetition, "frame count")
            assert [int(row["frame"]) for row in rows] == list(range(expected_frames))
            correctness = int(meta["correctness"])
            assert all(row["is_correct"].lower() == str(bool(correctness)).lower() for row in rows)
            if int(meta["mocap_erroneous"]):
                audit["excluded_mocap_repetitions"] += 1
                continue
            positions = np.array([
                [[float(row[f"{joint}_{axis}"]) for axis in "XYZ"] for joint in JOINTS]
                for row in rows
            ])
            assert np.isfinite(positions).all()
            # The derivative uses one central Hips joint for both hips.
            assert np.allclose(positions[:, 12], positions[:, 13])
            label = f"{exercise}_{'correct' if correctness else 'incorrect'}"
            records.append({
                "video": video, "repetition": repetition, "person": int(meta["person_id"]),
                "label": label, "positions": positions,
            })
    assert audit["raw_repetitions"] == len(metadata), "Missing or extra repetitions"
    audit["usable_repetitions"] = len(records)
    audit["usable_frames"] = sum(len(record["positions"]) for record in records)
    return records, audit


def joint_angle(a, b, c):
    ab, cb = a - b, c - b
    denominator = np.linalg.norm(ab, axis=1) * np.linalg.norm(cb, axis=1)
    if np.any(denominator < 1e-8):
        raise ValueError("Degenerate joint angle")
    cosine = np.sum(ab * cb, axis=1) / denominator
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def feature_vector(positions, representation):
    # Inputs already hip-centred; normalize by torso length for body scale.
    shoulder_center = (positions[:, 0] + positions[:, 1]) / 2
    hip_center = (positions[:, 12] + positions[:, 13]) / 2
    scale = np.linalg.norm(shoulder_center - hip_center, axis=1)
    if np.any(scale < 1e-8):
        raise ValueError("Zero torso length")
    normalized = (positions - hip_center[:, None, :]) / scale[:, None, None]
    if representation == "coordinates":
        # Initial model: mean and variation of the 12 nonzero 3D joints.
        values = normalized[:, :12, :].reshape(len(positions), -1)
        names = [f"{joint}_{axis}" for joint in JOINTS[:12] for axis in "XYZ"]
        stats = [("mean", np.mean), ("std", np.std)]
    elif representation == "geometry":
        columns, names = [], []
        for side in ("Left", "Right"):
            index = {joint: JOINTS.index(side + joint) for joint in
                     ("Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle")}
            for name, (a, b, c) in {
                "elbow": ("Shoulder", "Elbow", "Wrist"),
                "shoulder": ("Elbow", "Shoulder", "Hip"),
                "hip": ("Shoulder", "Hip", "Knee"),
                "knee": ("Hip", "Knee", "Ankle"),
            }.items():
                columns.append(joint_angle(*(positions[:, index[joint]] for joint in (a, b, c))))
                names.append(side + "_" + name + "_angle")
        # Y is vertical in the source MoCap skeleton.
        columns.append((shoulder_center[:, 1] - hip_center[:, 1]) / scale)
        names.append("torso_vertical_ratio")
        for name, a, b in [("wrists_distance", 4, 5), ("knees_distance", 6, 7),
                           ("ankles_distance", 8, 9)]:
            columns.append(np.linalg.norm(normalized[:, a] - normalized[:, b], axis=1))
            names.append(name)
        values = np.column_stack(columns)
        stats = [("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]
    else:
        raise ValueError(representation)
    vector = np.concatenate([function(values, axis=0) for _, function in stats])
    feature_names = [f"{name}_{statistic}" for statistic, _ in stats for name in names]
    assert np.isfinite(vector).all()
    return vector, feature_names


def build_table(records, representation):
    vectors = [feature_vector(record["positions"], representation)[0] for record in records]
    names = feature_vector(records[0]["positions"], representation)[1]
    return np.stack(vectors), names
