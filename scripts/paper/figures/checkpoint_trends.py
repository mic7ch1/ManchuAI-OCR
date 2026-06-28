from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

from scripts.paper.utils import (
    MODEL_COLORS,
    FIGSIZE,
    FIGURES_OUTPUT_DIR,
    format_model_name,
    sort_by_model_preference,
    load_json,
    save_figure,
)


def eval_checkpoints_by_wa(metrics_root, output_dir):
    metrics_root = Path(metrics_root)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    fig.suptitle("Checkpoint Evaluation", fontweight="bold")

    exclude_models = {"crnn-base-3m"}
    model_dirs = [
        d for d in metrics_root.iterdir() if d.is_dir() and d.name not in exclude_models
    ]

    sorted_model_dirs = sort_by_model_preference(
        model_dirs, name_extractor=lambda p: p.name
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Word Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.8)
    ax.grid(False, axis="x")

    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    for model_idx, model_dir in enumerate(sorted_model_dirs):
        model_name = format_model_name(model_dir.name)
        color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]

        steps, accuracies, cers, f1s = [], [], [], []

        for metrics_file in model_dir.glob("*_validation.json"):
            name = metrics_file.stem  # checkpoint-45000_validation or epoch-17_validation
            if name.startswith("best_checkpoint"):
                continue
            if "checkpoint-" in name:
                step_str = name.split("checkpoint-")[1].split("_validation", 1)[0]
            elif "epoch-" in name:
                step_str = name.split("epoch-")[1].split("_validation", 1)[0]
            else:
                continue
            try:
                step = int(step_str)
            except ValueError:
                continue

            data = load_json(metrics_file, {})

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

        ax.plot(steps, accuracies, label=model_name, color=color, linewidth=2)

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
        ax.annotate(
            f"{int(steps[best_idx]):,}",
            xy=(steps[best_idx], accuracies[best_idx]),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color=color,
        )

    ax.legend(loc="best")

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12)

    save_figure(fig, output_dir, "05a_checkpoint_trends")


def main():
    FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_checkpoints_by_wa("results/metrics", FIGURES_OUTPUT_DIR)
    print("Saved checkpoint trends figure to", FIGURES_OUTPUT_DIR)


if __name__ == "__main__":
    main()
