from pathlib import Path

import matplotlib.pyplot as plt

from scripts.figures.utils import load_metrics_data, COLORS, FIGSIZE


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

    if not models:
        print("No inference time data found in metrics; skipping figure.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

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


def main():
    output_dir = Path("results/paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_data = load_metrics_data()
    create_inference_time_comparison(metrics_data, output_dir)
    print("Saved inference time comparison to", output_dir)


if __name__ == "__main__":
    main()
