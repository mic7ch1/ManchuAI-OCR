from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import json
import sys

sys.path.append("src")
from evaluation.metrics import (
    calculate_cer,
    calculate_f1_score,
)
from matplotlib.patches import Patch

from scripts.figures.utils import FIGSIZE


def create_vlm_vs_crnn_chart(metrics_data, output_dir):
    # Original model identifiers used to fetch metrics
    models = ["llama-32-11b", "crnn-base-3m"]

    # Categories on the x-axis (data splits)
    split_labels = ["Validation", "Test"]

    # Colour-blind friendly palette
    model_info = {
        "llama-32-11b": {"label": "LLaMA-3.2-11B", "color": "#4E79A7"},  # blue
        "crnn-base-3m": {"label": "CRNN-3M", "color": "#E15759"},  # red
    }

    metrics_to_plot = [
        "manchu_cer",
        "manchu_word_accuracy",
        "manchu_f1_score",
        "inference_time",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7, 4.33))
    fig.suptitle("LLaMA-3.2-11B vs CRNN-3M", fontweight="bold")

    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        def transform_metric(m_key, val, err):
            """Return transformed metric value and error based on key."""
            if "cer" in m_key:
                # 1-CER, error is unchanged since std(1-X) = std(X)
                return (1 - val if val is not None else 0, err)
            if m_key == "inference_time":
                # ms/img -> img/s
                if val > 0:
                    t_val = 1000 / val
                    # Propagate error for Y=1000/X -> std(Y) ~ |(dY/dX)|std(X)
                    t_err = (1000 / (val**2)) * err
                    return (t_val, t_err)
                return (0, 0)
            return (val, err)

        split_to_values = {split: [] for split in ["validation", "test"]}
        split_to_errors = {split: [] for split in ["validation", "test"]}
        for split in ["validation", "test"]:
            for model in models:
                raw_val = metrics_data.get(model, {}).get(split, {}).get(metric, 0)
                raw_err = (
                    metrics_data.get(model, {}).get(split, {}).get(f"{metric}_std", 0)
                )
                t_val, t_err = transform_metric(metric, raw_val, raw_err)
                split_to_values[split].append(t_val)
                split_to_errors[split].append(t_err)

        x = np.arange(len(split_labels))
        width = 0.35

        # Bars: one per model for each x category (split)
        bars_by_model = {}
        for idx_model, model in enumerate(models):
            # Extract values in order Validation, Test
            y_vals = [
                split_to_values["validation"][idx_model],
                split_to_values["test"][idx_model],
            ]
            # Offset: first model left, second model right
            offset = (-0.5 + idx_model) * width

            err_vals = [
                split_to_errors["validation"][idx_model],
                split_to_errors["test"][idx_model],
            ]

            # Only draw error bars for 1-CER, F1 Score and Inference Speed
            draw_error_metrics = {"manchu_cer", "manchu_f1_score", "inference_time"}
            draw_error = metric in draw_error_metrics and any(err_vals)

            bars = ax.bar(
                x + offset,
                y_vals,
                width,
                alpha=0.9,
                label=model_info[model]["label"],
                color=model_info[model]["color"],
                yerr=err_vals if draw_error else None,
                capsize=4 if draw_error else 0,
            )

            # Annotate each bar right after creation to align with its error value
            for idx_bar, bar in enumerate(bars):
                height = bar.get_height()
                err = err_vals[idx_bar] if draw_error else 0
                bar.set_edgecolor("black")
                bar.set_linewidth(0.5)
                if height > 0:
                    ax.annotate(
                        f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height + err),
                        xytext=(0, 0),  # small offset above bar/errorbar
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            bars_by_model[model] = bars

        # Remove x-axis label ("Models") per request
        # Format y-axis label without the "Manchu" prefix
        metric_key = metric.replace("manchu_", "")

        if metric_key == "cer":
            y_label = "1 - CER"
        elif metric_key == "inference_time":
            y_label = "Inference Speed (img/s)"
        else:
            y_label = metric_key.replace("_", " ").title()

        # Show metric name as subplot title and remove y-axis label
        ax.set_title(y_label, fontsize=12)
        ax.set_ylabel("")

        ax.set_xticks(x)
        ax.set_xticklabels(split_labels)

        # Use log scale for transformed CER (1-CER) and inference speed
        if metric_key in ["inference_time"]:
            ax.set_yscale("log")

    # Legend with consistent colors
    fig.legend(
        handles=[
            Patch(facecolor="#4E79A7", alpha=0.9, label="LLaMA-3.2-11B"),
            Patch(facecolor="#E15759", alpha=0.9, label="CRNN-3M"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.975, 1.0),
        borderaxespad=0.2,
    )

    # fig.subplots_adjust(hspace=0.6, top=0.8)
    plt.tight_layout(pad=1.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "comparison_vlm_vs_crnn.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "comparison_vlm_vs_crnn.pdf", bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Helper: compute mean and standard deviation metrics from detailed result files
# -----------------------------------------------------------------------------


def compute_metrics_data(models, splits):
    """Return nested dict of metrics and std using results/{split}/{model}.json files."""

    metrics_data = {}

    for model in models:
        metrics_data[model] = {}
        for split in splits:
            file_path = Path("results") / split / f"{model}.json"

            if not file_path.exists():
                print("Warning: missing", file_path)
                continue

            with open(file_path, "r") as f:
                results = json.load(f)

            # Per-sample metric lists (except word accuracy which is loaded separately)
            cer_vals = []
            f1_vals = []
            inf_time_vals = []

            for r in results:
                manchu_gt = r["manchu_gt"]
                manchu_pred = r["manchu_pred"]

                cer = calculate_cer(manchu_gt, manchu_pred)
                cer_vals.append(cer)

                f1_vals.append(calculate_f1_score(manchu_gt, manchu_pred))

                inf_time_vals.append(r.get("inference_time", 0))

            # Compute means and stds (population std)
            def _mean_std(arr):
                arr_np = np.array(arr, dtype=float)
                return float(arr_np.mean()), float(arr_np.std(ddof=0))

            cer_mean, cer_std = _mean_std(cer_vals)
            f1_mean, f1_std = _mean_std(f1_vals)
            inf_mean, inf_std = _mean_std(inf_time_vals)

            # Load pre-computed word accuracy metrics
            metrics_file_path = Path("results") / "metrics" / f"{model}_{split}.json"

            if metrics_file_path.exists():
                with open(metrics_file_path, "r") as mf:
                    metrics_json = json.load(mf)

                # Try common key variations for robustness
                word_acc_mean = float(
                    metrics_json.get(
                        "manchu_word_accuracy",
                        metrics_json.get("word_accuracy", 0.0),
                    )
                )
                word_acc_std = float(
                    metrics_json.get(
                        "manchu_word_accuracy_std",
                        metrics_json.get("word_accuracy_std", 0.0),
                    )
                )
            else:
                print("Warning: missing", metrics_file_path)
                word_acc_mean, word_acc_std = 0.0, 0.0

            metrics_data[model][split] = {
                "manchu_cer": cer_mean,
                "manchu_cer_std": cer_std,
                "manchu_word_accuracy": word_acc_mean,
                "manchu_word_accuracy_std": word_acc_std,
                "manchu_f1_score": f1_mean,
                "manchu_f1_score_std": f1_std,
                "inference_time": inf_mean,
                "inference_time_std": inf_std,
            }

    return metrics_data


def main():
    output_dir = Path("results/paper/figures")
    models = ["llama-32-11b", "crnn-base-3m"]
    splits = ["validation", "test"]

    metrics_data = compute_metrics_data(models, splits)
    create_vlm_vs_crnn_chart(metrics_data, output_dir)
    print("Saved VLM vs CRNN comparison chart to", output_dir)


if __name__ == "__main__":
    main()
