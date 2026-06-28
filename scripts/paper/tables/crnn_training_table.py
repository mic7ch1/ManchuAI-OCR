"""
Generate LaTeX table for CRNN training configuration.
"""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.paper.utils import TABLES_OUTPUT_DIR, write_latex_table
from src.utils.config import ConfigLoader


def create_crnn_training_table(output_dir):
    """Generate LaTeX table for CRNN training configuration."""
    config_loader = ConfigLoader()
    raw_config = config_loader.training_config

    crnn_config = raw_config.get("crnn-base-3m", {}).get("training", {})

    num_epochs = crnn_config.get("num_train_epochs", 100)
    input_height = crnn_config.get("input_height", 64)
    input_width = crnn_config.get("input_width", 480)
    batch_size = crnn_config.get("batch_size", 16)
    learning_rate = crnn_config.get("learning_rate", 1e-3)
    hidden_size = crnn_config.get("hidden_size", 256)
    dropout = crnn_config.get("dropout", 0.3)
    warmup_epochs = crnn_config.get("warmup_epochs", 5)
    mixed_precision = crnn_config.get("mixed_precision", True)

    optimizer_config = crnn_config.get("optimizer", {})
    optimizer_type = optimizer_config.get("type", "AdamW")
    weight_decay = optimizer_config.get("weight_decay", 0.05)

    scheduler_config = crnn_config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "CosineAnnealingWarmRestarts")

    gradient_config = crnn_config.get("gradient_clipping", {})
    max_norm = gradient_config.get("max_norm", 1.0)

    mixed_precision_str = "True" if mixed_precision else "False"

    latex_content = rf"""\begin{{table}}[t!]
\TBL{{\caption{{CRNN training configuration.}}\label{{tab:crnn_training}}}}{{\centering
\begin{{tabular}}{{l@{{\hspace{{1em}}}}l}}
\toprule
\textbf{{Parameter}} & \textbf{{Value}} \\
\midrule
Dataset & 60,000 validation samples \\
Optimizer & {optimizer_type} \\
Initial learning rate & {learning_rate} \\
Weight decay & {weight_decay} \\
LR scheduler & {scheduler_type} \\
Input image size & {input_height} $\times$ {input_width} pixels \\
Hidden size & {hidden_size} \\
Dropout & {dropout} \\
Normalization & [0, 1] \\
Training epochs & {num_epochs} \\
Warmup epochs & {warmup_epochs} \\
Mixed precision & {mixed_precision_str} \\
Gradient clipping & Max norm {max_norm} \\
Batch size & {batch_size} \\
\botrule%
\end{{tabular}}
}}
\end{{table}}
"""

    write_latex_table(
        output_dir,
        "09_crnn_training.tex",
        latex_content,
        "CRNN training table",
    )


def main():
    TABLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_crnn_training_table(TABLES_OUTPUT_DIR)


if __name__ == "__main__":
    main()
