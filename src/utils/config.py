import yaml
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


class ConfigLoader:
    def __init__(self, config_dir=Path("configs")):
        self.project_root = project_root
        self.config_dir = self.project_root / config_dir

        self.models_config = self.load_yaml_config("models.yaml")
        self.training_config = self.load_yaml_config("training.yaml")
        self.evaluation_config = self.load_yaml_config("evaluation.yaml")
        self.data_config = self.load_yaml_config("data.yaml")
        self.inference_config = self.load_yaml_config("inference.yaml")

    def load_yaml_config(self, file_name):
        file_path = self.config_dir / file_name
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            return config_data
        except Exception as e:
            raise Exception(
                f"An unexpected error occurred while loading {file_path}: {e}"
            )

    def deep_merge_dicts(self, base_dict, update_dict):
        merged = base_dict.copy()
        for key, value in update_dict.items():
            if (
                isinstance(value, dict)
                and key in merged
                and isinstance(merged[key], dict)
            ):
                merged[key] = self.deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def get_config(self, config_type, section_key=None):
        config_map = {
            "models": self.models_config,
            "training": self.training_config,
            "evaluation": self.evaluation_config,
            "data": self.data_config,
            "inference": self.inference_config,
        }

        config_content = config_map.get(config_type)

        if config_type == "models":
            return config_content.get("models")

        if section_key is None:
            section_key = "default"

        if section_key != "default":
            section_specific_data = config_content.get(section_key)
            if section_specific_data is not None:
                default_data = config_content.get("default", {})
                config_to_return = self.deep_merge_dicts(
                    default_data, section_specific_data
                )
                return config_to_return

        default_data = config_content.get("default")
        if default_data is None:
            raise KeyError(
                f"Mandatory 'default' section not found or is empty in '{config_type}.yaml'."
            )

        return default_data.copy()


if __name__ == "__main__":
    # Example usage
    config_loader = ConfigLoader()

    print("\n--- Models Config ---")
    models_data = config_loader.get_config("models")

    print(models_data)
