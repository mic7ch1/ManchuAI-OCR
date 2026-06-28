"""
Generate LaTeX table for CRNN model architecture overview.
"""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.paper.utils import TABLES_OUTPUT_DIR, write_latex_table
from src.utils.config import ConfigLoader


def create_crnn_architecture_table(output_dir):
    """Generate LaTeX table for CRNN model architecture."""
    config_loader = ConfigLoader()
    raw_config = config_loader.training_config

    crnn_config = raw_config.get("crnn-base-3m", {}).get("training", {})

    input_height = crnn_config.get("input_height", 64)
    input_width = crnn_config.get("input_width", 480)
    hidden_size = crnn_config.get("hidden_size", 256)
    dropout = crnn_config.get("dropout", 0.3)

    dropout_pct = int(dropout * 100)

    latex_content = rf"""\begin{{table}}[hbt!]
\TBL{{\caption{{CRNN model architecture overview.}}\label{{tab:crnn_arch}}}}{{\centering
\begin{{tabular}}{{l@{{\hspace{{1em}}}}l}}
\toprule
\textbf{{Component}} & \textbf{{Specification}} \\
\midrule
\multicolumn{{2}}{{l}}{{\textit{{CNN Feature Extractor}}}} \\
Layers & 9 convolutional layers \\
Channel progression & 3 \textrightarrow{{}} 64 \textrightarrow{{}} 128 \textrightarrow{{}} 256 \textrightarrow{{}} 512 \\
Input size & {input_height} $\times$ {input_width} pixels \\
Output features & 512-dimensional per position \\
\midrule
\multicolumn{{2}}{{l}}{{\textit{{LSTM Sequence Modeler}}}} \\
Architecture & 4-layer bidirectional LSTM \\
Hidden units & {hidden_size} per direction ({hidden_size * 2} total) \\
Dropout rate & {dropout_pct}\% between layers \\
\midrule
\multicolumn{{2}}{{l}}{{\textit{{Output Layer}}}} \\
Type & Linear + CTC decoding \\
Function & Character class probabilities \\
\botrule%
\end{{tabular}}
}}
\end{{table}}
"""

    write_latex_table(
        output_dir,
        "08_crnn_architecture.tex",
        latex_content,
        "CRNN architecture table",
    )


def main():
    TABLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_crnn_architecture_table(TABLES_OUTPUT_DIR)


if __name__ == "__main__":
    main()
