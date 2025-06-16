import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.dataset import prepare_evaluation_datasets
from src.utils.config import ConfigLoader
from src.evaluation.vlm_evaluator import evaluate_vlm_model
from src.evaluation.crnn_evaluator import evaluate_crnn_model
from src.evaluation.openai_evaluator import evaluate_openai_model


def main(target_models=None):
    config_loader = ConfigLoader()
    models_config_list = config_loader.get_config("models")
    dataset_config = config_loader.get_config("data")
    if target_models:
        models_config_list = [
            m for m in models_config_list if m["name"] in target_models
        ]
    val_dataset, test_dataset = prepare_evaluation_datasets(
        dataset_config["val_split"], dataset_config["test_split"]
    )
    evaluation_output = project_root / "results"
    evaluation_output.mkdir(parents=True, exist_ok=True)
    for model_config in models_config_list:
        model_name = model_config["name"]

        if not model_config:
            print(f"Warning: Model {model_name} not found in models.yaml. Skipping.")
            continue

        model_class = model_config.get("model_class")

        evaluation_config = config_loader.get_config("evaluation", model_name)

        if model_class == "VLM":
            evaluate_vlm_model(
                model_config,
                evaluation_config,
                dataset_config,
                val_dataset,
                test_dataset,
                evaluation_output,
            )
        elif model_class == "CRNN":
            evaluate_crnn_model(
                model_config,
                evaluation_config,
                dataset_config,
                val_dataset,
                test_dataset,
                evaluation_output,
            )
        elif model_class == "OPENAI":
            evaluate_openai_model(
                model_config,
                evaluation_config,
                dataset_config,
                val_dataset,
                test_dataset,
                evaluation_output,
            )


if __name__ == "__main__":
    main(target_models=["qwen-25-3b", "qwen-25-7b", "llama-32-11b", "crnn-base-3m"])
