from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import sys

sys.path.append("src")

plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent1": "#F18F01",
    "accent2": "#C73E1D",
    "accent3": "#1B998B",
    "accent4": "#7209B7",
    "neutral": "#495057",
    "light": "#F8F9FA",
}


MODEL_COLORS = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["accent1"],
    COLORS["accent2"],
    COLORS["accent3"],
    COLORS["accent4"],
]


sns.set_palette(MODEL_COLORS)


plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
    }
)


def load_metrics_data():
    metrics_dir = Path("results/metrics")
    metrics_data = {}

    for metrics_file in metrics_dir.glob("*.json"):
        with open(metrics_file, "r") as f:
            data = json.load(f)

        filename = metrics_file.stem
        if "_" in filename:
            model, dataset = filename.rsplit("_", 1)
            if model not in metrics_data:
                metrics_data[model] = {}
            metrics_data[model][dataset] = data

    return metrics_data


def load_detailed_results(results_dir):
    results_path = Path(results_dir)
    detailed_results = {}

    for result_file in results_path.glob("*.json"):
        model_name = result_file.stem
        with open(result_file, "r") as f:
            detailed_results[model_name] = json.load(f)

    return detailed_results


def create_performance_comparison_chart(metrics_data, output_dir):
    models = list(metrics_data.keys())
    datasets = ["test", "validation"]

    metrics_to_plot = [
        "manchu_word_accuracy",
        "roman_word_accuracy",
        "manchu_cer",
        "roman_cer",
        "manchu_f1_score",
        "roman_f1_score",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        test_values = []
        val_values = []
        model_names = []

        for model in models:
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

        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax.bar(x - width / 2, test_values, width, label="Test", alpha=0.8)
        bars2 = ax.bar(x + width / 2, val_values, width, label="Validation", alpha=0.8)

        ax.set_xlabel("Models")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend()

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
    plt.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "performance_comparison.pdf", bbox_inches="tight")
    plt.close()


def create_accuracy_bar_plot(metrics_data, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    models = list(metrics_data.keys())

    manchu_test = []
    manchu_val = []
    model_names = []

    for model in models:
        test_acc = (
            metrics_data[model].get("test", {}).get("manchu_word_accuracy", 0)
            if "test" in metrics_data[model]
            else 0
        )
        val_acc = (
            metrics_data[model].get("validation", {}).get("manchu_word_accuracy", 0)
            if "validation" in metrics_data[model]
            else 0
        )

        manchu_test.append(test_acc)
        manchu_val.append(val_acc)
        model_names.append(model)

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        manchu_test,
        width,
        label="Test",
        alpha=0.8,
        color=COLORS["primary"],
    )
    bars2 = ax.bar(
        x + width / 2,
        manchu_val,
        width,
        label="Validation",
        alpha=0.8,
        color=COLORS["secondary"],
    )

    ax.set_xlabel("Models")
    ax.set_ylabel("Manchu Word Accuracy")
    ax.set_title("Manchu Word Accuracy Comparison")
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
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "accuracy_comparison.pdf", bbox_inches="tight")
    plt.close()


def create_cer_bar_plot(metrics_data, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    models = list(metrics_data.keys())

    manchu_test = []
    manchu_val = []
    model_names = []

    for model in models:
        test_cer = (
            metrics_data[model].get("test", {}).get("manchu_cer", 0)
            if "test" in metrics_data[model]
            else 0
        )
        val_cer = (
            metrics_data[model].get("validation", {}).get("manchu_cer", 0)
            if "validation" in metrics_data[model]
            else 0
        )

        manchu_test.append(test_cer)
        manchu_val.append(val_cer)
        model_names.append(model)

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        manchu_test,
        width,
        label="Test",
        alpha=0.8,
        color=COLORS["accent1"],
    )
    bars2 = ax.bar(
        x + width / 2,
        manchu_val,
        width,
        label="Validation",
        alpha=0.8,
        color=COLORS["accent2"],
    )

    ax.set_xlabel("Models")
    ax.set_ylabel("Manchu Character Error Rate")
    ax.set_title("Manchu Character Error Rate Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(manchu_test), max(manchu_val)) * 1.1)

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
    plt.savefig(output_dir / "cer_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "cer_comparison.pdf", bbox_inches="tight")
    plt.close()


def create_f1_bar_plot(metrics_data, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

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


def create_inference_time_comparison(metrics_data, output_dir):
    models = []
    test_times = []

    for model in metrics_data:
        if (
            "test" in metrics_data[model]
            and "inference_time" in metrics_data[model]["test"]
        ):
            models.append(model)
            test_times.append(metrics_data[model]["test"]["inference_time"])

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(models, test_times, alpha=0.8, color=COLORS["primary"])

    ax.set_xlabel("Models")
    ax.set_ylabel("Inference Time (microseconds)")
    ax.set_title("Inference Time Comparison (Test Set)")
    ax.tick_params(axis="x", rotation=45)
    ax.set_yscale("log")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}μs",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "inference_time_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "inference_time_comparison.pdf", bbox_inches="tight")
    plt.close()


def analyze_word_length_performance_by_model(detailed_results, output_dir):

    model_word_length_stats = {}
    overall_word_length_counts = defaultdict(int)

    for model, results in detailed_results.items():
        model_word_length_stats[model] = defaultdict(lambda: {"correct": 0, "total": 0})

        for result in results:
            manchu_gt = result.get("manchu_gt", "")
            manchu_pred = result.get("manchu_pred", "")
            word_length = len(manchu_gt)

            model_word_length_stats[model][word_length]["total"] += 1
            overall_word_length_counts[word_length] += 1

            if manchu_gt == manchu_pred:
                model_word_length_stats[model][word_length]["correct"] += 1

    all_lengths = sorted(set(overall_word_length_counts.keys()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for model in model_word_length_stats:
        accuracies = []
        lengths_for_model = []

        for length in all_lengths:
            stats = model_word_length_stats[model][length]
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                accuracies.append(accuracy)
                lengths_for_model.append(length)

        if accuracies:
            ax1.plot(
                lengths_for_model,
                accuracies,
                marker="o",
                label=model,
                linewidth=2,
                markersize=6,
            )

    ax1.set_xlabel("Word Length (characters)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy by Word Length (by Model)")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    counts = [overall_word_length_counts[length] for length in all_lengths]
    ax2.bar(all_lengths, counts, alpha=0.7, color=COLORS["accent1"])
    ax2.set_xlabel("Word Length (characters)")
    ax2.set_ylabel("Number of Samples")
    ax2.set_title("Sample Distribution by Word Length")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "word_length_analysis_by_model.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "word_length_analysis_by_model.pdf", bbox_inches="tight")
    plt.close()


def main():

    output_dir = Path("results/paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metrics data...")
    metrics_data = load_metrics_data()

    print("Loading detailed test results...")
    test_results = load_detailed_results("results/test")

    print("Generating performance comparison chart...")
    create_performance_comparison_chart(metrics_data, output_dir)

    print("Generating accuracy bar plots...")
    create_accuracy_bar_plot(metrics_data, output_dir)

    print("Generating CER comparison...")
    create_cer_bar_plot(metrics_data, output_dir)

    print("Generating F1 score comparison...")
    create_f1_bar_plot(metrics_data, output_dir)

    print("Generating inference time comparison...")
    create_inference_time_comparison(metrics_data, output_dir)

    print("Analyzing word length performance by model...")
    analyze_word_length_performance_by_model(test_results, output_dir)

    print(f"All figures saved to {output_dir}")
    print("Generated files:")
    for file in sorted(output_dir.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
