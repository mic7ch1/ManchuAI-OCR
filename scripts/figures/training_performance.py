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


def plot_training_performance(metrics_root, output_dir):
    metrics_root = Path(metrics_root)

    # Single line subplot for gradient norm
    fig, ax = plt.subplots(figsize=FIGSIZE)

    fig.suptitle(
        "Training Stability",
        fontweight="bold",
    )

    ax.set_ylabel("Gradient Norm")
    ax.set_xlabel("Training Step")

    # Log scale for y-axis to reduce overlap
    ax.set_yscale("log")

    # Format x-axis ticks with thousand separators
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    # Sort models by preferred order for legend consistency
    all_files = list(metrics_root.glob("trainer_state_*.json"))
    trainer_state_files = sort_by_model_preference(
        all_files, name_extractor=lambda p: p.stem.replace("trainer_state_", "")
    )

    # Cycle through colours & basic line styles for differentiation
    line_styles = ["-", "-", "-", "-"]

    bin_size = 1000  # aggregate window

    for idx, state_file in enumerate(trainer_state_files):
        model_name = state_file.stem.replace("trainer_state_", "")
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ls = line_styles[idx % len(line_styles)]

        try:
            with open(state_file, "r") as f:
                state_data = json.load(f)
        except Exception:
            continue

        steps, grad_norms = [], []

        for rec in state_data.get("log_history", []):
            # Record step as the x-axis
            step = rec.get("step")
            if step is None:
                continue

            grad_norm = rec.get("grad_norm")

            if grad_norm is None:
                continue

            steps.append(step)
            grad_norms.append(grad_norm)

        if not steps:
            continue

        steps = np.array(steps)
        grad_norms = np.array(grad_norms)

        # Aggregate into bins
        bin_ids = steps // bin_size
        unique_bins = np.unique(bin_ids)

        bin_centers = []
        bin_means = []
        bin_lowers = []
        bin_uppers = []

        for b in unique_bins:
            mask = bin_ids == b
            if not np.any(mask):
                continue
            bin_steps = steps[mask]
            bin_norms = grad_norms[mask]

            center = bin_steps.mean()
            bin_centers.append(center)
            bin_means.append(bin_norms.mean())
            bin_lowers.append(bin_norms.min())
            bin_uppers.append(bin_norms.max())

        bin_centers = np.array(bin_centers)
        sort_idx = np.argsort(bin_centers)
        bin_centers = bin_centers[sort_idx]
        bin_means = np.array(bin_means)[sort_idx]
        bin_lowers = np.array(bin_lowers)[sort_idx]
        bin_uppers = np.array(bin_uppers)[sort_idx]

        # Shaded variability band
        ax.fill_between(
            bin_centers,
            bin_lowers,
            bin_uppers,
            color=color,
            alpha=0.15,
            linewidth=0,
        )

        # Estimated line (mean)
        ax.plot(
            bin_centers,
            bin_means,
            label=format_model_name(model_name),
            color=color,
            linestyle=ls,
            linewidth=1.2,
        )

    ax.legend(loc="best")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.12)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "training_performance.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "training_performance.pdf", bbox_inches="tight")
    plt.close()


# Helper CLI wrapper


def main():
    output_dir = Path("results/paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_training_performance("results/metrics", output_dir)
    print("Saved training performance figure to", output_dir)


if __name__ == "__main__":
    main()
