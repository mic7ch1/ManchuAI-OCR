import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.config import ConfigLoader
from src.utils.dataset import prepare_training_datasets, convert_to_conversation
from src.training.vlm_trainer import train_vlm_model
from src.training.crnn_trainer import train_crnn_model
from src.training.openai_trainer import train_openai_model


def main(target_models=None):
    config_loader = ConfigLoader()
    models_config_list = config_loader.get_config("models")
    dataset_config = config_loader.get_config("data")
    if target_models:
        models_config_list = [
            m for m in models_config_list if m["name"] in target_models
        ]
    print(f"target_models: {target_models}")
    train_dataset, val_dataset = prepare_training_datasets(
        dataset_config.get("train_split"), dataset_config.get("val_split")
    )
    for model_config in models_config_list:
        model_name = model_config["name"]

        if not model_config:
            print(f"Warning: Model {model_name} not found in models.yaml. Skipping.")
            continue

        model_class = model_config.get("model_class")
        training_config = config_loader.get_config("training", model_config["name"])
        if not training_config:
            training_config = training_config.get("default")

        training_output = project_root / "models" / model_class / model_config["name"]
        training_output.mkdir(parents=True, exist_ok=True)

        if model_class == "VLM":
            converted_train_data_vlm = [
                convert_to_conversation(s, dataset_config) for s in train_dataset
            ]
            converted_val_data_vlm = [
                convert_to_conversation(s, dataset_config) for s in val_dataset
            ]

            train_vlm_model(
                model_config,
                training_config,
                training_output,
                converted_train_data_vlm,
                converted_val_data_vlm,
            )

        elif model_class == "CRNN":
            train_crnn_model(
                model_config,
                training_config,
                dataset_config,
                train_dataset,
                val_dataset,
                training_output,
            )
            print(f"CRNN training pipeline finished for {model_config['name']}.")

        elif model_class == "OPENAI":
            train_openai_model(
                model_config,
                training_config,
                dataset_config,
                train_dataset,
                val_dataset,
                training_output,
            )
            print(f"OpenAI training pipeline finished for {model_config['name']}.")


if __name__ == "__main__":
    # Possible models:
    # qwen-25-3b, qwen-25-7b, llama-32-11b, crnn-base-3m
    main(target_models=["qwen-25-3b", "qwen-25-7b", "llama-32-11b", "crnn-base-3m"])
