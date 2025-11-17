from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures.utils import load_metrics_data, COLORS, FIGSIZE


def create_f1_bar_plot(metrics_data, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    models = list(metrics_data.keys())

    manchu_test = []
    manchu_val = []
    model_names = []

    for model in models:
        test_f1 = (
            metrics_data[model].get("test", {}).get("manchu_f1_score", 0)
            if "test" in metrics_data[model]
            else 0
        )
        val_f1 = (
            metrics_data[model].get("validation", {}).get("manchu_f1_score", 0)
            if "validation" in metrics_data[model]
            else 0
        )

        manchu_test.append(test_f1)
        manchu_val.append(val_f1)
        model_names.append(model)

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        manchu_test,
        width,
        label="Test",
        alpha=0.8,
        color=COLORS["accent3"],
    )
    bars2 = ax.bar(
        x + width / 2,
        manchu_val,
        width,
        label="Validation",
        alpha=0.8,
        color=COLORS["accent4"],
    )

    ax.set_xlabel("Models")
    ax.set_ylabel("Manchu F1 Score")
    ax.set_title("Manchu F1 Score Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_dir / "f1_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "f1_comparison.pdf", bbox_inches="tight")
    plt.close()


def main():
    output_dir = Path("results/paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_data = load_metrics_data()
    create_f1_bar_plot(metrics_data, output_dir)
    print("Saved F1 comparison to", output_dir)


if __name__ == "__main__":
    main()
