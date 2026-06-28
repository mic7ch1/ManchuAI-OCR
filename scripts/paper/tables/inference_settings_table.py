"""
Generate LaTeX table for VLM inference settings.
"""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.paper.utils import (
    TABLES_OUTPUT_DIR,
    write_latex_table,
)


def generate_inference_settings_table():
    """Generate LaTeX table for VLM inference settings."""

    lines = [
        r"\begin{table}[H]",
        r"\TBL{\caption{VLM inference settings.}\label{tab:inference_settings}}{\centering",
        r"\begin{tabular}{ll}",
        r"    \toprule",
        r"    \textbf{Parameter} & \textbf{Value} \\",
        r"    \midrule",
        r"    max\_new\_tokens & 1536 \\",
        r"    temperature & 1.0 (default) \\",
        r"    top\_p & 1.0 (default) \\",
        r"    do\_sample & False (greedy) \\",
        r"    use\_cache & True \\",
        r"    pad\_token\_id & eos\_token\_id \\",
        r"    \bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def main():
    table_latex = generate_inference_settings_table()

    write_latex_table(
        TABLES_OUTPUT_DIR,
        "07_inference_settings.tex",
        table_latex,
        "inference settings table",
    )


if __name__ == "__main__":
    main()
