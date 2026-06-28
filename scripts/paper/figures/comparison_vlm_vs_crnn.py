from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.evaluation.metrics import (
    calculate_cer,
    calculate_f1_score,
    calculate_word_accuracy,
)
from scripts.paper.utils import (
    FIGSIZE,
    FIGURES_OUTPUT_DIR,
    MODEL_NAME_MAP,
    load_json,
    save_figure,
    transform_metric_for_display,
)


def create_vlm_vs_crnn_chart(metrics_data, output_dir):
    models = ["llama-32-11b", "crnn-base-3m"]

    split_labels = ["Validation", "Test"]

    model_colors = {
        "llama-32-11b": "#4E79A7",  # blue
        "crnn-base-3m": "#E15759",  # red
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

        split_to_values = {split: [] for split in ["validation", "test"]}
        split_to_errors = {split: [] for split in ["validation", "test"]}
        for split in ["validation", "test"]:
            for model in models:
                raw_val = metrics_data.get(model, {}).get(split, {}).get(metric, 0)
                raw_err = (
                    metrics_data.get(model, {}).get(split, {}).get(f"{metric}_std", 0)
                )
                t_val, t_err = transform_metric_for_display(metric, raw_val, raw_err)
                split_to_values[split].append(t_val)
                split_to_errors[split].append(t_err)

        x = np.arange(len(split_labels))
        width = 0.35

        bars_by_model = {}
        for idx_model, model in enumerate(models):
            y_vals = [
                split_to_values["validation"][idx_model],
                split_to_values["test"][idx_model],
            ]
            offset = (-0.5 + idx_model) * width

            err_vals = [
                split_to_errors["validation"][idx_model],
                split_to_errors["test"][idx_model],
            ]

            draw_error_metrics = {"manchu_cer", "manchu_word_accuracy", "manchu_f1_score", "inference_time"}
            draw_error = metric in draw_error_metrics and any(err_vals)

            bars = ax.bar(
                x + offset,
                y_vals,
                width,
                alpha=0.9,
                label=MODEL_NAME_MAP.get(model, model),
                color=model_colors[model],
                yerr=err_vals if draw_error else None,
                capsize=4 if draw_error else 0,
            )

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

        metric_key = metric.replace("manchu_", "")

        if metric_key == "cer":
            y_label = "1 - CER"
        elif metric_key == "inference_time":
            y_label = "Inference Speed (img/s)"
        else:
            y_label = metric_key.replace("_", " ").title()

        ax.set_title(y_label, fontsize=12)
        ax.set_ylabel("")

        ax.set_xticks(x)
        ax.set_xticklabels(split_labels)

        if metric_key in ["inference_time"]:
            ax.set_yscale("log")

    fig.legend(
        handles=[
            Patch(facecolor="#4E79A7", alpha=0.9, label="LLaMA-3.2-11B"),
            Patch(facecolor="#E15759", alpha=0.9, label="CRNN-3M"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.975, 1.0),
        borderaxespad=0.2,
    )

    plt.tight_layout(pad=1.5)

    save_figure(fig, output_dir, "07_comparison_vlm_vs_crnn")




def compute_metrics_data(models, splits):
    """Return nested dict of metrics and std using results/predictions/{model}/best_checkpoint/{split}.json files."""

    metrics_data = {}

    file_name_map = {
        "validation": "validation.json",
        "test": "test.json",
    }

    for model in models:
        metrics_data[model] = {}
        for split in splits:
            file_name = file_name_map.get(split, f"{split}.json")
            file_path = Path("results/predictions") / model / "best_checkpoint" / file_name

            if not file_path.exists():
                print("Warning: missing", file_path)
                continue

            results = load_json(file_path, [])

            cer_vals = []
            f1_vals = []
            word_acc_vals = []
            inf_time_vals = []

            for r in results:
                manchu_gt = r["manchu_gt"]
                manchu_pred = r["manchu_pred"]

                cer = calculate_cer(manchu_gt, manchu_pred)
                cer_vals.append(cer)

                f1_vals.append(calculate_f1_score(manchu_gt, manchu_pred))

                word_acc_vals.append(1.0 if calculate_word_accuracy(manchu_gt, manchu_pred) else 0.0)

                inf_time_vals.append(r.get("inference_time", 0))

            def _mean_std(arr):
                arr_np = np.array(arr, dtype=float)
                return float(arr_np.mean()), float(arr_np.std(ddof=0))

            def _bootstrap_ci(arr, n_bootstrap=1000, ci=0.95):
                arr_np = np.array(arr, dtype=float)
                n = len(arr_np)
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    sample = np.random.choice(arr_np, size=n, replace=True)
                    bootstrap_means.append(sample.mean())
                bootstrap_means = np.array(bootstrap_means)
                mean = arr_np.mean()
                lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
                upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
                half_width = (upper - lower) / 2
                return float(mean), float(half_width)

            cer_mean, cer_std = _bootstrap_ci(cer_vals)
            f1_mean, f1_std = _bootstrap_ci(f1_vals)
            word_acc_mean, word_acc_std = _bootstrap_ci(word_acc_vals)
            inf_mean, inf_std = _bootstrap_ci(inf_time_vals)

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
    FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models = ["llama-32-11b", "crnn-base-3m"]
    splits = ["validation", "test"]

    metrics_data = compute_metrics_data(models, splits)
    create_vlm_vs_crnn_chart(metrics_data, FIGURES_OUTPUT_DIR)
    print("Saved VLM vs CRNN comparison chart to", FIGURES_OUTPUT_DIR)


if __name__ == "__main__":
    main()
