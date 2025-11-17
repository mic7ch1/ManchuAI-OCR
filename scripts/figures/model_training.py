from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

from scripts.figures.utils import MODEL_COLORS, FIGSIZE, format_model_name


# --------------------------------------------------
# Data loading helpers
# --------------------------------------------------


def _load_checkpoint_metrics(metrics_root):
    metrics_root = Path(metrics_root)
    exclude_models = {"crnn-base-3m"}
    model_dirs = [
        d for d in metrics_root.iterdir() if d.is_dir() and d.name not in exclude_models
    ]

    data = {}

    for model_dir in model_dirs:
        model_name = model_dir.name
        steps, accs, cers, f1s = [], [], [], []

        for metrics_file in model_dir.glob("*_test.json"):
            name = metrics_file.stem
            if "checkpoint-" in name:
                step_str = name.split("checkpoint-")[1].split("_", 1)[0]
            elif "epoch-" in name:
                step_str = name.split("epoch-")[1].split("_", 1)[0]
            else:
                continue
            try:
                step = int(step_str)
            except ValueError:
                continue

            with open(metrics_file, "r") as f:
                record = json.load(f)

            steps.append(step)
            accs.append(record.get("manchu_word_accuracy", 0))
            cers.append(record.get("manchu_cer", 0))
            f1s.append(record.get("manchu_f1_score", 0))

        if not steps:
            continue

        idx = np.argsort(steps)
        steps = np.array(steps)[idx]
        accs = np.array(accs)[idx]
        cers = np.array(cers)[idx]
        f1s = np.array(f1s)[idx]

        data[model_name] = {
            "steps": steps,
            "accuracy": accs,
            "cer": cers,
            "f1": f1s,
        }

    return data


def _load_training_stability(metrics_root, bin_size=1000):
    metrics_root = Path(metrics_root)
    state_files = sorted(metrics_root.glob("trainer_state_*.json"))

    data = {}

    for state_file in state_files:
        model_name = state_file.stem.replace("trainer_state_", "")
        try:
            with open(state_file, "r") as f:
                state_data = json.load(f)
        except Exception:
            continue

        steps, norms = [], []
        for rec in state_data.get("log_history", []):
            step = rec.get("step")
            g = rec.get("grad_norm")
            if step is None or g is None:
                continue
            steps.append(step)
            norms.append(g)

        if not steps:
            continue

        steps = np.array(steps)
        norms = np.array(norms)

        bin_ids = steps // bin_size
        uniq_bins = np.unique(bin_ids)

        centers, means, lowers, uppers = [], [], [], []
        for b in uniq_bins:
            m = bin_ids == b
            if not np.any(m):
                continue
            bs = steps[m]
            bn = norms[m]
            centers.append(bs.mean())
            means.append(bn.mean())
            lowers.append(bn.min())
            uppers.append(bn.max())

        order = np.argsort(centers)
        centers = np.array(centers)[order]
        means = np.array(means)[order]
        lowers = np.array(lowers)[order]
        uppers = np.array(uppers)[order]

        data[model_name] = {
            "centers": centers,
            "mean": means,
            "lower": lowers,
            "upper": uppers,
        }

    return data


# --------------------------------------------------
# Plotting
# --------------------------------------------------


def plot_model_training(metrics_root, output_dir):
    ckpt_data = _load_checkpoint_metrics(metrics_root)
    grad_data = _load_training_stability(metrics_root)

    if not ckpt_data:
        print("No checkpoint data found.")
        return

    n_models = len(ckpt_data)
    # Preferred ordering
    pref_list = ["qwen3b", "qwen7b", "llama11b"]

    def _canonical(n):
        return n.lower().replace("-", "").replace("_", "").replace(" ", "")

    def _pref_index(n):
        canon = _canonical(n)
        try:
            return pref_list.index(canon)
        except ValueError:
            return len(pref_list)

    ordered_models = sorted(ckpt_data.keys(), key=_pref_index)

    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig_w = FIGSIZE[0] * n_cols
    fig_h = FIGSIZE[1] * n_rows
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        sharex=False,
    )

    axes = np.array(axes).reshape(-1)

    fig.suptitle("Training Stability & Checkpoint Evaluation", fontweight="bold")

    last_idx = len(ordered_models) - 1

    for idx, model_name in enumerate(ordered_models):
        mdata = ckpt_data[model_name]
        if idx >= len(axes):
            break
        ax = axes[idx]
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]

        # Accuracy curve
        steps = mdata["steps"]
        accs = mdata["accuracy"]
        ax.plot(steps, accs, color=color, linewidth=2)

        # Best checkpoint marker
        best_i = int(np.argmax(accs))
        ax.plot(
            steps[best_i],
            accs[best_i],
            marker="*",
            color=color,
            markersize=16,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )

        # Format x-axis ticks with thousand separators
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

        ax.set_ylabel("Word Accuracy")
        ax.set_title(format_model_name(model_name))
        ax.grid(True, axis="y", alpha=0.3)

        # Gradient norm overlay
        if model_name in grad_data:
            g = grad_data[model_name]
            twin = ax.twinx()
            twin.set_yscale("log")
            twin.fill_between(
                g["centers"],
                g["lower"],
                g["upper"],
                color=color,
                alpha=0.1,
                linewidth=0,
            )
            twin.plot(
                g["centers"],
                g["mean"],
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )
            if idx == last_idx:
                twin.set_ylabel("Gradient Norm (log)")

    for ax in axes[len(ckpt_data) :]:
        ax.axis("off")

    for ax in axes[: len(ckpt_data)]:
        ax.set_xlabel("Training Step")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(top=0.92, wspace=0.3, hspace=0.4)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "model_training.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "model_training.pdf", bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# CLI helper
# --------------------------------------------------


def main():
    output_dir = Path("results/paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_model_training("results/metrics", output_dir)
    print("Saved model training overview figure to", output_dir)


if __name__ == "__main__":
    main()
