from pathlib import Path

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


def plot_training_performance(models_root, output_dir):
    """Plot training stability (gradient norms) from trainer_state.json files.

    Args:
        models_root: Root directory containing model folders (e.g., models/VLM/)
        output_dir: Output directory for figures
    """
    models_root = Path(models_root)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    fig.suptitle(
        "Training Stability",
        fontweight="bold",
    )

    ax.set_ylabel("Gradient Norm")
    ax.set_xlabel("Training Step")

    ax.set_yscale("log")

    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    all_files = []
    for model_dir in models_root.iterdir():
        if not model_dir.is_dir():
            continue
        checkpoints_dir = model_dir / "checkpoints"
        if not checkpoints_dir.exists():
            continue
        checkpoints = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if not checkpoints:
            continue
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
        trainer_state = latest_checkpoint / "trainer_state.json"
        if trainer_state.exists():
            all_files.append(trainer_state)

    trainer_state_files = sort_by_model_preference(
        all_files, name_extractor=lambda p: p.parent.parent.parent.name
    )

    line_styles = ["-", "-", "-", "-"]

    bin_size = 1000  # aggregate window

    for idx, state_file in enumerate(trainer_state_files):
        model_name = state_file.parent.parent.parent.name
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ls = line_styles[idx % len(line_styles)]

        state_data = load_json(state_file)
        if state_data is None:
            continue

        steps, grad_norms = [], []

        for rec in state_data.get("log_history", []):
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

        ax.fill_between(
            bin_centers,
            bin_lowers,
            bin_uppers,
            color=color,
            alpha=0.15,
            linewidth=0,
        )

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

    save_figure(fig, output_dir, "05b_training_performance")




def main():
    FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_performance("models/VLM", FIGURES_OUTPUT_DIR)
    print("Saved training performance figure to", FIGURES_OUTPUT_DIR)


if __name__ == "__main__":
    main()
