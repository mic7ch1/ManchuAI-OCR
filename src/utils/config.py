import yaml
from pathlib import Path
import sys
import shutil
import os

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.dataset import download_and_prepare_data


class ConfigLoader:
    def __init__(self, config_dir=Path("configs")):
        self.project_root = project_root
        self.config_dir = self.project_root / config_dir

        self.base_config = self.load_yaml_config("base.yaml")
        self.training_config = self.load_yaml_config("training.yaml")
        self.evaluation_config = self.load_yaml_config("evaluation.yaml")

        self._handle_data_cache()

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

    def _handle_data_cache(self):
        """Check cache setting and remove data directory if cache is disabled."""
        if os.environ.get("MANCHU_OCR_DOWNLOADING") == "1":
            return

        data_config = self.base_config.get("data", {})
        use_cache = data_config.get("cache", True)

        if not use_cache:
            data_path = self.project_root / "data"
            if data_path.exists():
                print(f"Cache disabled. Removing existing data directory: {data_path}")
                shutil.rmtree(data_path)
                print("Existing data removed successfully.")

            print("Downloading fresh data...")
            os.environ["MANCHU_OCR_DOWNLOADING"] = "1"  # Prevent infinite recursion
            try:
                download_and_prepare_data(data_config)
                print("Data download completed successfully.")
            except Exception as e:
                print(f"Warning: Data download failed with error: {e}")
            finally:
                os.environ.pop("MANCHU_OCR_DOWNLOADING", None)

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
            "models": self.base_config,
            "training": self.training_config,
            "evaluation": self.evaluation_config,
            "data": self.base_config,
        }

        config_content = config_map.get(config_type)

        if config_type == "models":
            return config_content.get("models")

        if config_type == "data":
            return config_content.get("data")

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
    config_loader = ConfigLoader()

    print("\n--- Models Config ---")
    models_data = config_loader.get_config("models")

    print(models_data)
