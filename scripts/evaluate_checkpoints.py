import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.dataset import prepare_evaluation_datasets
from src.utils.config import ConfigLoader
from src.evaluation.vlm_evaluator import evaluate_vlm
from src.evaluation.crnn_evaluator import evaluate_crnn
from src.evaluation.utils import print_header, create_best_model


def main(target_models=None):
    config_loader = ConfigLoader()
    models_config_list = config_loader.get_config("models")
    dataset_config = config_loader.get_config("data")
    default_config = config_loader.get_config("evaluation")
    checkpoints_config = config_loader.evaluation_config["checkpoints"]

    num_samples = checkpoints_config["num_samples"]
    models_to_evaluate = checkpoints_config["models"]
    max_new_tokens = default_config.get("max_new_tokens", 1536)

    if target_models:
        models_to_evaluate = target_models

    val_dataset, _ = prepare_evaluation_datasets(
        dataset_config["val_split"], dataset_config["test_split"]
    )

    print(f"Loaded validation dataset with {len(val_dataset)} samples")
    print(f"Evaluating on {num_samples} samples per checkpoint")

    evaluation_output = project_root / "results"
    evaluation_output.mkdir(parents=True, exist_ok=True)

    model_config_map = {m["name"]: m for m in models_config_list}
    models_to_process = [
        model_config_map[name]
        for name in models_to_evaluate
        if name in model_config_map
    ]

    print(f"\nModels to evaluate: {[m['name'] for m in models_to_process]}")

    for model_config in models_to_process:
        model_name = model_config["name"]
        model_class = model_config.get("model_class")

        if model_class == "VLM":
            result = evaluate_vlm(
                model_config,
                dataset_config,
                val_dataset,
                None,
                evaluation_output,
                num_samples,
                mode="checkpoints",
                max_new_tokens=max_new_tokens,
            )
        elif model_class == "CRNN":
            result = evaluate_crnn(
                model_config,
                dataset_config,
                val_dataset,
                None,
                evaluation_output,
                num_samples,
                mode="checkpoints",
            )
        else:
            print(f"Unknown model class {model_class}, skipping...")
            continue

        if result and result.get("best_step"):
            create_best_model(
                model_class,
                model_name,
                result["best_step"],
                result.get("best_accuracy"),
            )

    print_header("Checkpoint evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints for Manchu OCR models")
    parser.add_argument(
        "--target-model",
        type=str,
        nargs="*",
        default=None,
        help="Model(s) to evaluate (space-separated). Available models: qwen-25-3b, qwen-25-7b, llama-32-11b, crnn-base-3m",
    )

    args = parser.parse_args()
    target_models = args.target_model if args.target_model else None

    main(target_models=target_models)
