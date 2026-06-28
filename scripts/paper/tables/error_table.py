"""
Generate LaTeX table for character-level error analysis on test and validation data.
"""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.paper.utils import (
    TABLES_OUTPUT_DIR,
    PROJECT_ROOT,
    MODEL_NAME_MAP,
    load_json,
    write_latex_table,
)

VLM_MODELS = ["llama-32-11b", "qwen-25-7b", "qwen-25-3b"]

CHAR_TO_ROMAN = {
    "ᠠ": "A",
    "ᡝ": "E",
    "ᡳ": "I",
    "ᠣ": "O",
    "ᡠ": "U",
    "ᡡ": "Ū",
    "ᠨ": "N",
    "ᠩ": "NG",
    "ᡴ": "K",
    "ᡤ": "G",
    "ᡥ": "H",
    "ᠪ": "B",
    "ᡦ": "P",
    "ᠰ": "S",
    "ᡧ": "Š",
    "ᡨ": "T",
    "ᡩ": "D",
    "ᠯ": "L",
    "ᠮ": "M",
    "ᠴ": "C",
    "ᠵ": "J",
    "ᠶ": "Y",
    "ᡵ": "R",
    "ᡶ": "F",
    "ᠸ": "W",
    "ᡯ": "Z",
}


def load_error_data():
    """Load error frequency data for VLM models only."""
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    error_data = {}

    for model_name in VLM_MODELS:
        model_dir = metrics_dir / model_name
        if not model_dir.exists():
            continue

        error_data[model_name] = {}

        val_file = model_dir / "best_checkpoint" / "validation.json"
        data = load_json(val_file)
        if data:
            error_data[model_name]["validation"] = {
                "errors": data.get("frequent_error", {}).get("manchu", {}),
                "total": data.get("total_predictions", 1),
            }

        test_file = model_dir / "best_checkpoint" / "test.json"
        data = load_json(test_file)
        if data:
            error_data[model_name]["test"] = {
                "errors": data.get("frequent_error", {}).get("manchu", {}),
                "total": data.get("total_predictions", 1),
            }

    return error_data


def get_top_errors(errors_dict, top_n=3):
    """Get top N error characters with their counts and percentages."""
    if not errors_dict:
        return []

    total_errors = sum(errors_dict.values())
    if total_errors == 0:
        return []

    sorted_errors = sorted(errors_dict.items(), key=lambda x: x[1], reverse=True)

    top_errors = []
    for char, count in sorted_errors[:top_n]:
        percentage = (count / total_errors) * 100
        roman = CHAR_TO_ROMAN.get(char, "?")
        top_errors.append({
            "char": char,
            "roman": roman,
            "count": count,
            "percentage": percentage,
        })

    return top_errors


def format_top_errors(top_errors):
    """Format top errors as LaTeX string."""
    parts = []
    for err in top_errors:
        parts.append(f"\\manchu{{{err['char']}}} ({err['roman']}) ({err['percentage']:.1f}\\%)")
    return ", ".join(parts)


def calculate_share(top_errors):
    """Calculate total share of top errors."""
    return sum(err["percentage"] for err in top_errors)


def generate_error_table(error_data):
    """Generate LaTeX table for character error analysis."""

    display_order = VLM_MODELS

    lines = [
        r"\begin{table}[t]",
        r"\TBL{\caption{Character-level error analysis on test and validation data}\label{tab:top_errors}}{\centering",
        r"\begin{tabular}{lcc}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Top 3 Characters} & \textbf{Share} \\",
        r"    \midrule",
        r"    \multicolumn{3}{c}{\textit{Validation Dataset (1,000 synthetic samples)}} \\",
        r"    \midrule",
    ]

    for model_name in display_order:
        if model_name not in error_data or "validation" not in error_data[model_name]:
            continue

        errors = error_data[model_name]["validation"]["errors"]
        top_errors = get_top_errors(errors, top_n=3)

        if top_errors:
            model_display = MODEL_NAME_MAP.get(model_name, model_name)
            top_str = format_top_errors(top_errors)
            share = calculate_share(top_errors)
            lines.append(f"    {model_display} & {top_str} & {share:.1f}\\% \\\\")

    lines.extend([
        r"    \midrule",
        r"    \multicolumn{3}{c}{\textit{Test Dataset (753 real-world samples)}} \\",
        r"    \midrule",
    ])

    for model_name in display_order:
        if model_name not in error_data or "test" not in error_data[model_name]:
            continue

        errors = error_data[model_name]["test"]["errors"]
        top_errors = get_top_errors(errors, top_n=3)

        if top_errors:
            model_display = MODEL_NAME_MAP.get(model_name, model_name)
            top_str = format_top_errors(top_errors)
            share = calculate_share(top_errors)
            lines.append(f"    {model_display} & {top_str} & {share:.1f}\\% \\\\")

    lines.extend([
        r"    \bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    error_data = load_error_data()
    table_latex = generate_error_table(error_data)

    write_latex_table(
        TABLES_OUTPUT_DIR,
        "04_error_analysis.tex",
        table_latex,
        "error analysis table",
    )


if __name__ == "__main__":
    main()
