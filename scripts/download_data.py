import sys
from pathlib import Path
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import ConfigLoader


def download_and_prepare_data():
    config_loader = ConfigLoader()
    data_config = config_loader.get_config("data")

    dataset_name = data_config["dataset_name"]
    image_key = data_config["image_key"]
    text_keys = data_config["text_key"]

    splits = {
        "train": data_config.get("train_split", "train"),
        "validation": data_config.get("val_split", "validation"),
        "test": data_config.get("test_split", "test"),
    }

    base_data_path = project_root / "data"

    if not base_data_path.exists():
        base_data_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting data download from {dataset_name}...")

    for split_name, hf_split_name in splits.items():
        print(f"Processing {split_name} split...")
        output_dir = base_data_path / split_name
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_list = []

        dataset_split = load_dataset(
            dataset_name,
            name="default",
            split=hf_split_name,
            trust_remote_code=True,
        )

        for i, example in tqdm(enumerate(dataset_split)):
            image_data = example[image_key]
            image_filename = f"{split_name}_{i:05d}.png"
            image_path = images_dir / image_filename
            image_data.save(image_path)

            label_entry = {"image_filename": image_filename}
            for tk in text_keys:
                label_entry[tk] = example[tk]

            labels_list.append(label_entry)

        labels_file_path = output_dir / "labels.json"
        with open(labels_file_path, "w", encoding="utf-8") as f:
            json.dump(labels_list, f, ensure_ascii=False, indent=4)

        print(
            f"Finished processing {split_name} split. {len(labels_list)} images saved."
        )

    print("Data download and preparation complete.")


if __name__ == "__main__":
    download_and_prepare_data()
