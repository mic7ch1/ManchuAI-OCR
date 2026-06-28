import sys
import argparse
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.dataset import prepare_evaluation_datasets
from src.utils.config import ConfigLoader
from src.evaluation.vlm_evaluator import evaluate_vlm
from src.evaluation.crnn_evaluator import evaluate_crnn


def main(target_models=None):
    config_loader = ConfigLoader()
    models_config_list = config_loader.get_config("models")
    dataset_config = config_loader.get_config("data")
    evaluation_config = config_loader.get_config("evaluation")

    if target_models:
        config_map = {m["name"]: m for m in models_config_list}
        ordered_models_list = []
        for model_name in target_models:
            if model_name in config_map:
                ordered_models_list.append(config_map[model_name])
            else:
                print(
                    f"Warning: Model '{model_name}' not found in configuration. Skipping."
                )
        models_config_list = ordered_models_list

    val_dataset, test_dataset = prepare_evaluation_datasets(
        dataset_config["val_split"], dataset_config["test_split"]
    )

    evaluation_output = project_root / "results"
    evaluation_output.mkdir(parents=True, exist_ok=True)

    num_samples = evaluation_config["validation"]["num_samples"]
    max_new_tokens = evaluation_config.get("max_new_tokens", 1536)

    for model_config in models_config_list:
        model_name = model_config["name"]

        if not model_config:
            print(f"Warning: Model {model_name} not found in models.yaml. Skipping.")
            continue

        model_class = model_config.get("model_class")

        if model_class == "VLM":
            evaluate_vlm(
                model_config,
                dataset_config,
                val_dataset,
                test_dataset,
                evaluation_output,
                num_samples,
                mode="best",
                max_new_tokens=max_new_tokens,
            )
        elif model_class == "CRNN":
            evaluate_crnn(
                model_config,
                dataset_config,
                val_dataset,
                test_dataset,
                evaluation_output,
                num_samples,
                mode="best",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Manchu OCR models")
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
