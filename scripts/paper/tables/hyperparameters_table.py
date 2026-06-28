from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.paper.utils import TABLES_OUTPUT_DIR, write_latex_table
from src.utils.config import ConfigLoader


def create_hyperparameters_table(output_dir):
    """Generate LaTeX table for VLM training hyperparameters from training.yaml."""
    config_loader = ConfigLoader()

    raw_config = config_loader.training_config

    default_training = raw_config["default"]["training"]
    default_peft = raw_config["default"]["peft"]

    qwen_3b = raw_config.get("qwen-25-3b", {}).get("training", {})
    qwen_7b = raw_config.get("qwen-25-7b", {}).get("training", {})
    llama_11b = raw_config.get("llama-32-11b", {}).get("training", {})

    epochs_3b = qwen_3b.get("num_train_epochs", default_training["num_train_epochs"])
    epochs_7b = qwen_7b.get("num_train_epochs", default_training["num_train_epochs"])
    epochs_11b = llama_11b.get("num_train_epochs", default_training["num_train_epochs"])

    lr_3b = qwen_3b.get("learning_rate", default_training["learning_rate"])
    lr_7b = qwen_7b.get("learning_rate", default_training["learning_rate"])
    lr_11b = llama_11b.get("learning_rate", default_training["learning_rate"])

    def format_lr(lr):
        if lr >= 1e-3:
            return f"{lr:.0e}".replace("e-0", r" \times 10^{-") + "}"
        exp = int(f"{lr:.0e}".split("e")[1])
        coef = lr / (10 ** exp)
        return f"{coef:.0f}" + r" \times 10^{" + str(exp) + "}"

    if lr_3b == lr_7b == lr_11b:
        lr_value = f"${format_lr(lr_3b)}$"
    else:
        lr_value = f"${format_lr(lr_3b)}$ (3B), ${format_lr(lr_7b)}$ (7B), ${format_lr(lr_11b)}$ (11B)"

    optim = default_training["optim"]
    if "adamw" in optim.lower():
        if "8bit" in optim:
            optim_display = "AdamW (8-bit)"
        else:
            optim_display = "AdamW"
    else:
        optim_display = optim

    scheduler = default_training["lr_scheduler_type"]
    if scheduler == "cosine_with_restarts":
        scheduler_display = "Cosine with restarts"
    elif scheduler == "cosine":
        scheduler_display = "Cosine"
    else:
        scheduler_display = scheduler.replace("_", " ").title()

    warmup_3b = qwen_3b.get("warmup_steps", default_training["warmup_steps"])
    warmup_7b = qwen_7b.get("warmup_steps", default_training["warmup_steps"])
    warmup_11b = llama_11b.get("warmup_steps", default_training["warmup_steps"])

    if warmup_3b == warmup_7b == warmup_11b:
        warmup_value = str(warmup_3b)
    else:
        warmup_value = f"{warmup_3b} (3B), {warmup_7b} (7B), {warmup_11b} (11B)"

    batch_size = default_training["per_device_train_batch_size"]
    grad_accum = default_training["gradient_accumulation_steps"]
    effective_batch = batch_size * grad_accum

    if default_training.get("bf16"):
        precision = "BFloat16 (bf16)"
    elif default_training.get("fp16"):
        precision = "Float16 (fp16)"
    else:
        precision = "Float32"

    weight_decay = default_training["weight_decay"]
    seed = default_training["seed"]

    lora_r = default_peft["r"]
    lora_alpha = default_peft["lora_alpha"]
    lora_dropout = default_peft["lora_dropout"]

    latex_content = rf"""\begin{{table}}[hbt!]
\TBL{{\caption{{Training hyperparameters for VLM fine-tuning}}\label{{tab:hyperparams}}}}{{\centering
\begin{{tabular}}{{l@{{\hspace{{1em}}}}l}}
\toprule
\textbf{{Hyperparameter}} & \textbf{{Value}} \\
\midrule
Number of epochs & {epochs_3b} (3B), {epochs_7b} (7B), {epochs_11b} (11B) \\
Optimizer & {optim_display} \\
Learning rate & {lr_value} \\
LR scheduler & {scheduler_display} \\
Warmup steps & {warmup_value} \\
Per-device batch size & {batch_size} \\
Gradient accumulation steps & {grad_accum} \\
Effective batch size & {effective_batch} \\
Mixed precision & {precision} \\
Weight decay & {weight_decay} \\
Random seed & {seed} \\
LoRA rank & {lora_r} \\
LoRA alpha & {lora_alpha} \\
LoRA dropout & {lora_dropout} \\
\botrule%
\end{{tabular}}}}
\end{{table}}
"""

    write_latex_table(
        output_dir,
        "01_hyperparameters_table.tex",
        latex_content,
        "hyperparameters table",
    )


def main():
    TABLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_hyperparameters_table(TABLES_OUTPUT_DIR)


if __name__ == "__main__":
    main()
