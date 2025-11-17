from pathlib import Path
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as _fm

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

MODEL_NAME_MAP = {
    "llama-32-11b": "LLaMA-3.2-11B",
    "qwen-25-3b": "Qwen-2.5-3B",
    "qwen-25-7b": "Qwen-2.5-7B",
}

# Preferred model order for consistent legends across all plots
PREFERRED_MODEL_ORDER = ["qwen-25-3b", "qwen-25-7b", "llama-32-11b"]


def sort_by_model_preference(paths, name_extractor):
    """Sort a list of file/dir paths based on PREFERRED_MODEL_ORDER.

    Args:
        paths (list[Path]): A list of pathlib.Path objects to sort.
        name_extractor (callable): A function that takes a Path object and returns
            the model name string.

    Returns:
        list[Path]: The sorted list of paths.
    """

    def sort_key(path):
        name = name_extractor(path)
        try:
            return PREFERRED_MODEL_ORDER.index(name)
        except ValueError:
            return len(PREFERRED_MODEL_ORDER)

    return sorted(paths, key=sort_key)


def load_metrics_data(metrics_dir="results/metrics"):
    metrics_path = Path(metrics_dir)
    metrics_data = {}

    for metrics_file in metrics_path.glob("*.json"):
        with open(metrics_file, "r") as f:
            data = json.load(f)

        filename = metrics_file.stem  # e.g. crnn-large_test
        if "_" not in filename:
            continue

        model, dataset = filename.rsplit("_", 1)
        metrics_data.setdefault(model, {})[dataset] = data

    return metrics_data


def load_detailed_results(results_dir):
    results_path = Path(results_dir)
    detailed_results = {}

    for result_file in results_path.glob("*.json"):
        model_name = result_file.stem
        with open(result_file, "r") as f:
            detailed_results[model_name] = json.load(f)

    return detailed_results


def format_model_name(dir_name):

    key = dir_name.lower().replace("_", "-")
    return MODEL_NAME_MAP.get(key, dir_name)
