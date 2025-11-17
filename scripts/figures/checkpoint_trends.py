from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

from scripts.figures.utils import (
    MODEL_COLORS,
    FIGSIZE,
    format_model_name,
    sort_by_model_preference,
)


def eval_checkpoints_by_wa(metrics_root, output_dir):
    metrics_root = Path(metrics_root)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    fig.suptitle("Checkpoint Evaluation", fontweight="bold")

    exclude_models = {"crnn-base-3m"}
    model_dirs = [
        d for d in metrics_root.iterdir() if d.is_dir() and d.name not in exclude_models
    ]

    # Sort models by preferred order for legend consistency
    sorted_model_dirs = sort_by_model_preference(
        model_dirs, name_extractor=lambda p: p.name
    )

    # Axis labels and formatting
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Word Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.8)
    ax.grid(False, axis="x")

    # Format x-axis ticks with thousand separators
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    for model_idx, model_dir in enumerate(sorted_model_dirs):
        model_name = format_model_name(model_dir.name)
        color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]

        steps, accuracies, cers, f1s = [], [], [], []

        for metrics_file in model_dir.glob("*_test.json"):
            name = metrics_file.stem  # checkpoint-45000_test or epoch-17_test
            if "checkpoint-" in name:
                step_str = name.split("checkpoint-")[1].split("_", 1)[0]
            elif "epoch-" in name:
                step_str = name.split("epoch-")[1].split("_", 1)[0]
            else:
                continue
            try:
                step = int(step_str)
            except ValueError:
                continue

            with open(metrics_file, "r") as f:
                data = json.load(f)

            steps.append(step)
            accuracies.append(data.get("manchu_word_accuracy", 0))
            cers.append(data.get("manchu_cer", 0))
            f1s.append(data.get("manchu_f1_score", 0))

        if not steps:
            continue

        sorted_idx = np.argsort(steps)
        steps = np.array(steps)[sorted_idx]
        accuracies = np.array(accuracies)[sorted_idx]
        cers = np.array(cers)[sorted_idx]
        f1s = np.array(f1s)[sorted_idx]

        # Plot accuracy curve
        ax.plot(steps, accuracies, label=model_name, color=color, linewidth=2)

        # Mark the best checkpoint with a star
        best_idx = int(np.argmax(accuracies))
        ax.plot(
            steps[best_idx],
            accuracies[best_idx],
            marker="*",
            color=color,
            markersize=16,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )

    ax.legend(loc="best")

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12)

    plt.savefig(output_dir / "checkpoint_trends.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "checkpoint_trends.pdf", bbox_inches="tight")
    plt.close()


def main():
    output_dir = Path("results/paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_checkpoints_by_wa("results/metrics", output_dir)
    print("Saved checkpoint trends figure to", output_dir)


if __name__ == "__main__":
    main()
