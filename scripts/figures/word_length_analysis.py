from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

from scripts.figures.utils import load_detailed_results, COLORS, FIGSIZE


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
    if not all_lengths:
        print("No word length data found; skipping figure.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE)

    # Accuracy curves per model
    for model in model_word_length_stats:
        accuracies = []
        lengths_for_model = []

        for length in all_lengths:
            stats = model_word_length_stats[model][length]
            if stats["total"] > 0:
                accuracies.append(stats["correct"] / stats["total"])
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

    # Sample distribution
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

    detailed_results = load_detailed_results("results/test")
    analyze_word_length_performance_by_model(detailed_results, output_dir)
    print("Saved word length analysis figure to", output_dir)


if __name__ == "__main__":
    main()
