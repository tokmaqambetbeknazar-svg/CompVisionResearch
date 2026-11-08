"""Create presentation figures from saved measurements, never invented values."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "results"
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    names = ["Baseline", "Tuned"]
    colors = ["#74839b", "#008974"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy (%)", "Macro-F1 (%)")):
        values = [100 * metrics["models"][name][metric] for name in ("baseline", "tuned")]
        ax.bar(names, values, color=colors, width=.55)
        ax.set_ylim(0, 110)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
        for x, value in enumerate(values):
            ax.text(x, value + 2, f"{value:.1f}", ha="center", weight="bold")
    fig.suptitle(f"Held-out people: 8 and 9 | {metrics['test_repetitions']} repetitions | 4 classes", fontsize=12)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)

    labels = ["Push-up\ncorrect", "Push-up\nincorrect", "Squat\ncorrect", "Squat\nincorrect"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained")
    maximum = max(max(row) for model in metrics["models"].values() for row in model["confusion_matrix"])
    for ax, key, title in zip(axes, ("baseline", "tuned"), names):
        matrix = np.array(metrics["models"][key]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        ax.set_xticks(range(4), labels, fontsize=9)
        ax.set_yticks(range(4), labels, fontsize=9)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Dataset label")
        ax.set_title(title)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > maximum / 2 else "#1a2533")
    fig.suptitle("Where the models make mistakes (number of repetitions)")
    fig.savefig(output / "confusion_matrices.png", dpi=180)
    fig.savefig(output / "confusion_matrices.svg")
    plt.close(fig)
    # Matplotlib writes trailing spaces in SVG paths; keep Git diffs clean.
    for path in output.glob("*.svg"):
        text = path.read_text(encoding="utf-8")
        path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8", newline="\n")
    print("Saved comparison and confusion-matrix figures")


if __name__ == "__main__":
    main()
