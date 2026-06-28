import base64
import io
import random
from datasets import Dataset, Features, Value, Image as HFImageFeatures, load_dataset
from PIL import Image as PILImage
from pathlib import Path
from tqdm import tqdm

from src.utils.files import load_json, save_json

project_root = Path(__file__).resolve().parent.parent.parent


def download_and_prepare_data(data_config):
    """Download and prepare dataset from HuggingFace.

    Args:
        data_config: Dictionary containing dataset configuration with keys:
            - dataset_name: HuggingFace dataset name
            - image_key: Key for image data in dataset
            - text_key: List of text keys (e.g., ['roman', 'manchu'])
            - train_split, val_split, test_split: Split names
    """
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
        save_json(labels_file_path, labels_list)

        print(
            f"Finished processing {split_name} split. {len(labels_list)} images saved."
        )

    print("Data download and preparation complete.")


def load_split(data_root, split_name):
    features = Features(
        {
            "im": HFImageFeatures(),
            "manchu": Value("string"),
            "roman": Value("string"),
            "image_path": Value("string"),
        }
    )
    label_file = data_root / split_name / "labels.json"

    if not label_file.exists():
        print(f"Error: Labels file not found for {split_name} at {label_file}")
        return Dataset.from_dict({}, features=features)

    label_data = load_json(label_file, [])

    def generator():
        for item in label_data:
            try:
                yield {
                    "im": PILImage.open(
                        data_root / split_name / "images" / item["image_filename"]
                    ).convert("RGB"),
                    "manchu": item["manchu"] or "",
                    "roman": item["roman"] or "",
                    "image_path": str(
                        data_root / split_name / "images" / item["image_filename"]
                    ),
                }
            except Exception as e:
                print(
                    f"Warning: Failed to load image {item.get('image_filename', 'unknown')}: {e}"
                )
                continue

    return Dataset.from_generator(generator, features=features)


def prepare_training_datasets(train_key, val_key):
    data_root = project_root / "data"
    train_dataset = load_split(data_root, train_key)
    val_dataset = load_split(data_root, val_key)
    return train_dataset, val_dataset


def prepare_evaluation_datasets(val_key, test_key):
    data_root = project_root / "data"
    val_dataset = load_split(data_root, val_key)
    test_dataset = load_split(data_root, test_key)
    return val_dataset, test_dataset


def convert_to_conversation(sample, dataset_config):

    query = f'Manchu: {sample[dataset_config["text_key"][0]]}\nRoman: {sample[dataset_config["text_key"][1]]}'
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample[dataset_config["image_key"]]},
                {"type": "text", "text": dataset_config["instruction"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": query,
                }
            ],
        },
    ]
    return {"messages": conversation}


def image_to_base64_data_url(image, format="JPEG"):
    if image.mode == "RGBA" or (
        image.mode == "P" and "A" in image.info.get("transparency", b"")
    ):
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_byte = buffered.getvalue()
    base64_encoded_data = base64.b64encode(img_byte)
    base64_message = base64_encoded_data.decode("utf-8")
    return f"data:image/{format.lower()};base64,{base64_message}"


if __name__ == "__main__":
    train_dataset, val_dataset = prepare_training_datasets("train", "validation")
    print(train_dataset[0])
    print(val_dataset[0])
