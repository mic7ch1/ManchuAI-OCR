from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from scripts.paper.utils import (
    load_metrics_data,
    FIGSIZE,
    FIGURES_OUTPUT_DIR,
    annotate_bar_values,
    save_figure,
)


def create_performance_comparison_chart(metrics_data, output_dir):
    models = list(metrics_data.keys())

    metrics_to_plot = [
        "manchu_cer",
        "manchu_word_accuracy",
        "manchu_f1_score",
        "inference_time",
    ]

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle("Model Performance Comparison (VLM Models)", fontweight="bold")

    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        test_values = []
        val_values = []
        model_names = []

        vlm_models = [
            m for m in models if any(k in m.lower() for k in ("qwen", "llama"))
        ]

        for model in vlm_models:
            test_val = (
                metrics_data[model].get("test", {}).get(metric, 0)
                if "test" in metrics_data[model]
                else 0
            )
            val_val = (
                metrics_data[model].get("validation", {}).get(metric, 0)
                if "validation" in metrics_data[model]
                else 0
            )

            test_values.append(test_val)
            val_values.append(val_val)
            model_names.append(model)

        x = np.arange(len(model_names)) if model_names else np.arange(len(vlm_models))
        width = 0.35

        bars1 = ax.bar(x - width / 2, test_values, width, label="Test", alpha=0.8)
        bars2 = ax.bar(x + width / 2, val_values, width, label="Validation", alpha=0.8)

        ax.set_xlabel("Models")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")

        annotate_bar_values(ax, bars1)
        annotate_bar_values(ax, bars2)

    test_patch = Patch(facecolor="C0", alpha=0.8, label="Test")
    val_patch = Patch(facecolor="C1", alpha=0.8, label="Validation")

    fig.legend(
        handles=[test_patch, val_patch],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure(fig, output_dir, "performance_comparison")


def main():
    FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_data = load_metrics_data()
    create_performance_comparison_chart(metrics_data, FIGURES_OUTPUT_DIR)
    print("Saved performance comparison chart to", FIGURES_OUTPUT_DIR)


if __name__ == "__main__":
    main()
