"""
Generate LaTeX table for performance comparison across validation and test datasets.
"""

from pathlib import Path
import sys

import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.paper.utils import (
    TABLES_OUTPUT_DIR,
    PROJECT_ROOT,
    MODEL_NAME_MAP,
    load_json,
    write_latex_table,
    format_value_with_bold,
    format_with_ci,
)
from src.evaluation.metrics import (
    calculate_cer,
    calculate_f1_score,
    calculate_word_accuracy,
)


VLM_MODELS = ["llama-32-11b", "qwen-25-7b", "qwen-25-3b"]


def compute_bootstrap_ci(results, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap CI for test results from per-sample metrics."""
    cer_vals = []
    f1_vals = []
    word_acc_vals = []
    inf_time_vals = []

    for r in results:
        manchu_gt = r["manchu_gt"]
        manchu_pred = r["manchu_pred"]

        cer_vals.append(calculate_cer(manchu_gt, manchu_pred))
        f1_vals.append(calculate_f1_score(manchu_gt, manchu_pred))
        word_acc_vals.append(1.0 if calculate_word_accuracy(manchu_gt, manchu_pred) else 0.0)
        inf_time_vals.append(r.get("inference_time", 0))

    def _bootstrap(arr):
        arr_np = np.array(arr, dtype=float)
        n = len(arr_np)
        np.random.seed(42)  # For reproducibility
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(arr_np, size=n, replace=True)
            bootstrap_means.append(sample.mean())
        bootstrap_means = np.array(bootstrap_means)
        mean = arr_np.mean()
        lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
        return float(mean), float(lower), float(upper)

    return {
        "cer": _bootstrap(cer_vals),
        "f1": _bootstrap(f1_vals),
        "word_acc": _bootstrap(word_acc_vals),
        "inf_time": _bootstrap(inf_time_vals),
    }


def load_metrics():
    """Load metrics for VLM models only."""
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    predictions_dir = PROJECT_ROOT / "results" / "predictions"
    metrics = {}

    for model_name in VLM_MODELS:
        model_metrics_dir = metrics_dir / model_name
        model_pred_dir = predictions_dir / model_name

        if not model_metrics_dir.exists():
            continue

        metrics[model_name] = {}

        val_file = model_pred_dir / "best_checkpoint" / "validation.json"
        if val_file.exists():
            results = load_json(val_file, [])
            metrics[model_name]["val_bootstrap"] = compute_bootstrap_ci(results)

        test_file = model_pred_dir / "best_checkpoint" / "test.json"
        if test_file.exists():
            results = load_json(test_file, [])
            metrics[model_name]["test_bootstrap"] = compute_bootstrap_ci(results)

    return metrics


def generate_performance_table(metrics):
    """Generate LaTeX table for performance comparison."""

    display_order = VLM_MODELS

    def extract_bootstrap_metrics(bootstrap):
        """Extract metrics from bootstrap results."""
        wa_mean, wa_lower, wa_upper = bootstrap["word_acc"]
        cer_mean, cer_lower, cer_upper = bootstrap["cer"]
        f1_mean, f1_lower, f1_upper = bootstrap["f1"]
        time_mean, time_lower, time_upper = bootstrap["inf_time"]

        return {
            "wa_mean": wa_mean * 100,
            "wa_ci_lower": wa_lower * 100,
            "wa_ci_upper": wa_upper * 100,
            "cer_mean": cer_mean,
            "cer_ci_lower": cer_lower,
            "cer_ci_upper": cer_upper,
            "f1_mean": f1_mean,
            "f1_ci_lower": f1_lower,
            "f1_ci_upper": f1_upper,
            "time_mean": time_mean / 1000,  # Convert ms to s
            "time_ci_lower": time_lower / 1000,
            "time_ci_upper": time_upper / 1000,
        }

    val_metrics = []
    for model_name in display_order:
        if model_name not in metrics or "val_bootstrap" not in metrics[model_name]:
            continue
        bootstrap = metrics[model_name]["val_bootstrap"]
        m = extract_bootstrap_metrics(bootstrap)
        m["model"] = model_name
        val_metrics.append(m)

    test_metrics = []
    for model_name in display_order:
        if model_name not in metrics or "test_bootstrap" not in metrics[model_name]:
            continue
        bootstrap = metrics[model_name]["test_bootstrap"]
        m = extract_bootstrap_metrics(bootstrap)
        m["model"] = model_name
        test_metrics.append(m)

    if val_metrics:
        best_val_wa = max(m["wa_mean"] for m in val_metrics)
        best_val_cer = min(m["cer_mean"] for m in val_metrics)
        best_val_f1 = max(m["f1_mean"] for m in val_metrics)
        best_val_time = min(m["time_mean"] for m in val_metrics)

    if test_metrics:
        best_test_wa = max(m["wa_mean"] for m in test_metrics)
        best_test_cer = min(m["cer_mean"] for m in test_metrics)
        best_test_f1 = max(m["f1_mean"] for m in test_metrics)
        best_test_time = min(m["time_mean"] for m in test_metrics)

    def format_with_pm(mean, ci_lower, ci_upper, is_best, fmt):
        """Format value as mean ± half-width."""
        half_width = (ci_upper - ci_lower) / 2
        if fmt == ".1f":
            val_str = f"{mean:.1f}$\\pm${half_width:.1f}"
        elif fmt == ".2f":
            val_str = f"{mean:.2f}$\\pm${half_width:.2f}"
        elif fmt == ".3f":
            val_str = f"{mean:.3f}$\\pm${half_width:.3f}"
        else:
            val_str = f"{mean:{fmt}}$\\pm${half_width:{fmt}}"
        if is_best:
            return f"\\textbf{{{val_str}}}"
        return val_str

    lines = [
        r"\begin{table}[!htb]",
        r"\TBL{\caption{Manchu performance comparison across validation and test datasets.}\label{tab:performance_comparison_manchu}}{\centering",
        r"\begin{tabular}{lcccc}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{WA (\%)}  & \textbf{CER} & \textbf{F1} & \textbf{Time (s)} \\",
        r"    \midrule",
        r"    \multicolumn{5}{c}{\textit{Validation Dataset (1,000 synthetic samples)}} \\",
        r"    \midrule",
    ]

    for m in val_metrics:
        model_display = MODEL_NAME_MAP.get(m["model"], m["model"])
        wa = format_with_pm(m["wa_mean"], m["wa_ci_lower"], m["wa_ci_upper"],
                          m["wa_mean"] == best_val_wa, ".1f")
        cer = format_with_pm(m["cer_mean"], m["cer_ci_lower"], m["cer_ci_upper"],
                           m["cer_mean"] == best_val_cer, ".2f")
        f1 = format_with_pm(m["f1_mean"], m["f1_ci_lower"], m["f1_ci_upper"],
                          m["f1_mean"] == best_val_f1, ".3f")
        time = format_with_pm(m["time_mean"], m["time_ci_lower"], m["time_ci_upper"],
                            m["time_mean"] == best_val_time, ".1f")
        lines.append(f"    {model_display} & {wa} & {cer} & {f1} & {time} \\\\")

    lines.extend([
        r"    \midrule",
        r"    \multicolumn{5}{c}{\textit{Test Dataset (753 real-world samples)}} \\",
        r"    \midrule",
    ])

    for m in test_metrics:
        model_display = MODEL_NAME_MAP.get(m["model"], m["model"])
        wa = format_with_pm(m["wa_mean"], m["wa_ci_lower"], m["wa_ci_upper"],
                          m["wa_mean"] == best_test_wa, ".1f")
        cer = format_with_pm(m["cer_mean"], m["cer_ci_lower"], m["cer_ci_upper"],
                           m["cer_mean"] == best_test_cer, ".2f")
        f1 = format_with_pm(m["f1_mean"], m["f1_ci_lower"], m["f1_ci_upper"],
                          m["f1_mean"] == best_test_f1, ".3f")
        time = format_with_pm(m["time_mean"], m["time_ci_lower"], m["time_ci_upper"],
                            m["time_mean"] == best_test_time, ".1f")
        lines.append(f"    {model_display} & {wa} & {cer} & {f1} & {time} \\\\")

    lines.extend([
        r"    \bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    metrics = load_metrics()
    table_latex = generate_performance_table(metrics)

    write_latex_table(
        TABLES_OUTPUT_DIR,
        "02_performance_comparison_manchu.tex",
        table_latex,
        "performance comparison table",
    )


if __name__ == "__main__":
    main()
