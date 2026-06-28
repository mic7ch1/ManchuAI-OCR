"""Shared utilities for paper figure and table generation."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.files import load_json


FIGURES_OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"
TABLES_OUTPUT_DIR = PROJECT_ROOT / "paper" / "tables"


MODEL_NAME_MAP = {
    "llama-32-11b": "LLaMA-3.2-VL-11B",
    "qwen-25-3b": "Qwen-2.5-VL-3B",
    "qwen-25-7b": "Qwen-2.5-VL-7B",
    "crnn-base-3m": "CRNN-3M",
}

PREFERRED_MODEL_ORDER = ["qwen-25-3b", "qwen-25-7b", "llama-32-11b", "crnn-base-3m"]


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


plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette(MODEL_COLORS)

FIGSIZE = (7, 4.33)
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12

plt.rcParams.update(
    {
        "figure.figsize": FIGSIZE,
        "figure.titlesize": TITLE_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "font.family": "Times New Roman",
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.8,
    }
)

if not any(f.name == "Times New Roman" for f in _fm.fontManager.ttflist):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]




def save_figure(fig, output_dir, filename_base, dpi=300):
    """Save matplotlib figure in PNG and PDF formats.

    Args:
        fig: Matplotlib figure object
        output_dir: Output directory path
        filename_base: Base filename without extension
        dpi: DPI for PNG output (default 300)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / f"{filename_base}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{filename_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def annotate_bar_values(ax, bars, format_str="{:.3f}", fontsize=8, offset=3):
    """Add value labels above bar chart bars.

    Args:
        ax: Matplotlib axes object
        bars: Bar container from ax.bar()
        format_str: Format string for values
        fontsize: Font size for annotations
        offset: Vertical offset in points
    """
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                format_str.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )


def transform_metric_for_display(metric_key, value, error=0):
    """Transform metric values for better visualization.

    Applies transformations:
    - CER -> 1-CER (for accuracy interpretation)
    - Inference time (ms/img) -> (img/s)

    Args:
        metric_key: Metric name (e.g., 'manchu_cer', 'inference_time')
        value: Raw metric value
        error: Standard deviation/error value

    Returns:
        Tuple of (transformed_value, transformed_error)
    """
    if "cer" in metric_key.lower():
        return (1 - value if value is not None else 0, error)

    if metric_key == "inference_time" and value > 0:
        t_val = 1000 / value
        t_err = (1000 / (value**2)) * error
        return (t_val, t_err)

    return (value, error)




def write_latex_table(output_dir, filename, latex_content, description=""):
    """Write LaTeX table to file with consistent logging.

    Args:
        output_dir: Output directory path
        filename: Output filename (should end with .tex)
        latex_content: LaTeX table content string
        description: Optional description for logging
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    with open(output_path, "w") as f:
        f.write(latex_content)

    desc = description or filename
    print(f"Saved {desc} to {output_path}")


def format_value_with_bold(value, is_best, fmt=".1f"):
    """Format a numeric value, making it bold if it's the best.

    Args:
        value: Numeric value to format
        is_best: Whether this is the best value (will be bolded)
        fmt: Format specifier string

    Returns:
        Formatted string with LaTeX bold if is_best
    """
    formatted = f"{value:{fmt}}"
    if is_best:
        return f"\\textbf{{{formatted}}}"
    return formatted


def format_with_ci(mean, ci_lower, ci_upper, is_best, fmt=".1f"):
    """Format a value with confidence interval.

    Args:
        mean: Mean value
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        is_best: Whether this is the best value
        fmt: Format specifier string

    Returns:
        Formatted string like "value [lower--upper]" with bold if is_best
    """
    mean_str = f"{mean:{fmt}}"
    ci_str = f"[{ci_lower:{fmt}}--{ci_upper:{fmt}}]"
    if is_best:
        return f"\\textbf{{{mean_str}}} {ci_str}"
    return f"{mean_str} {ci_str}"


def get_display_order(reverse=True):
    """Get model display order for tables/figures.

    Args:
        reverse: If True, reverse the order (LLaMA first)

    Returns:
        List of model names in display order
    """
    if reverse:
        return list(reversed(PREFERRED_MODEL_ORDER))
    return list(PREFERRED_MODEL_ORDER)




def sort_by_model_preference(paths, name_extractor):
    """Sort a list of file/dir paths based on PREFERRED_MODEL_ORDER.

    Args:
        paths: A list of pathlib.Path objects to sort
        name_extractor: Function that takes a Path and returns the model name

    Returns:
        Sorted list of paths
    """

    def sort_key(path):
        name = name_extractor(path)
        try:
            return PREFERRED_MODEL_ORDER.index(name)
        except ValueError:
            return len(PREFERRED_MODEL_ORDER)

    return sorted(paths, key=sort_key)


def load_metrics_data(metrics_dir="results/metrics"):
    """Load metrics data from model subdirectories.

    Structure: metrics_dir/{model}/best_checkpoint/test.json, validation.json

    Args:
        metrics_dir: Path to metrics directory

    Returns:
        Dict mapping model_name -> {split_name -> metrics_dict}
    """
    metrics_path = Path(metrics_dir)
    metrics_data = {}

    for model_dir in metrics_path.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        metrics_data[model_name] = {}

        for split_name, file_name in [
            ("test", "test.json"),
            ("validation", "validation.json"),
        ]:
            split_file = model_dir / "best_checkpoint" / file_name
            if split_file.exists():
                metrics_data[model_name][split_name] = load_json(split_file, {})

    return metrics_data


def load_model_metrics(metrics_dir=None, splits=None):
    """Load metrics for all models with consistent structure.

    Args:
        metrics_dir: Path to metrics directory (default: PROJECT_ROOT/results/metrics)
        splits: List of splits to load (default: ["validation", "test"])

    Returns:
        Dict mapping model_name -> {split_name -> metrics_dict, "replications" -> ...}
    """
    if metrics_dir is None:
        metrics_dir = PROJECT_ROOT / "results" / "metrics"
    else:
        metrics_dir = Path(metrics_dir)

    if splits is None:
        splits = ["validation", "test"]

    metrics = {}

    for model_name in PREFERRED_MODEL_ORDER:
        model_dir = metrics_dir / model_name
        if not model_dir.exists():
            continue

        metrics[model_name] = {}

        file_map = {
            "validation": "validation.json",
            "test": "test.json",
        }

        for split in splits:
            file_name = file_map.get(split, f"{split}.json")
            split_file = model_dir / "best_checkpoint" / file_name
            if split_file.exists():
                metrics[model_name][split] = load_json(split_file, {})

        replication_file = model_dir / "best_checkpoint" / "validation_replications.json"
        if replication_file.exists():
            metrics[model_name]["replications"] = load_json(replication_file, {})

    return metrics


def load_detailed_results(results_dir, split="test"):
    """Load detailed prediction results from model subdirectories.

    Structure: results_dir/{model}/best_checkpoint/{split}.json

    Args:
        results_dir: Path to results directory
        split: Split name ("test" or "validation")

    Returns:
        Dict mapping model_name -> results_list
    """
    results_path = Path(results_dir)
    detailed_results = {}

    file_name_map = {
        "test": "test.json",
        "validation": "validation.json",
    }
    file_name = file_name_map.get(split, f"{split}.json")

    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue

        split_file = model_dir / "best_checkpoint" / file_name
        if split_file.exists():
            detailed_results[model_dir.name] = load_json(split_file, [])

    return detailed_results


def format_model_name(dir_name):
    """Convert directory name to display name.

    Args:
        dir_name: Model directory name (e.g., 'llama-32-11b')

    Returns:
        Display name (e.g., 'LLaMA-3.2-11B')
    """
    key = dir_name.lower().replace("_", "-")
    return MODEL_NAME_MAP.get(key, dir_name)
