import json
import base64
import io
import random
from datasets import Dataset, Features, Value, Image as HFImageFeatures, load_dataset
from PIL import Image as PILImage
from pathlib import Path
from tqdm import tqdm
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


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

    with open(label_file, "r") as f:
        label_data = json.load(f)

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
    print("prepare_evaluation_datasets", val_dataset, test_dataset, test_key, val_key)
    return val_dataset, test_dataset


def convert_to_conversation(sample, dataset_config):

    query = f'Manchu: {sample[dataset_config["text_key"][0]]}\nRoman: {sample[dataset_config["text_key"][1]]}'
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": dataset_config["instruction"]},
                {"type": "image", "image": sample[dataset_config["image_key"]]},
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


def create_openai_jsonl_file(
    dataset_split, num_samples, output_filename, instruction=None
):
    if num_samples > len(dataset_split):
        num_samples = len(dataset_split)

    if num_samples == 0:
        return None

    if num_samples < len(dataset_split):
        indices = random.sample(range(len(dataset_split)), num_samples)
        sampled_data = dataset_split.select(indices)
    else:
        sampled_data = dataset_split.select(range(num_samples))

    lines_written = 0
    with open(output_filename, "w", encoding="utf-8") as f:
        for item in tqdm(sampled_data, desc=f"Processing samples"):
            try:
                pil_image = item["im"]
                manchu_text = item["manchu"]
                roman_text = item["roman"]

                if not isinstance(pil_image, PILImage.Image):
                    continue
                if not manchu_text or not roman_text:
                    continue

                base64_url = image_to_base64_data_url(pil_image)

                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert OCR system for Manchu script.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract the text from the provided image with perfect accuracy. Format your answer exactly as follows: first line with 'Manchu:' followed by the Manchu script, then a new line with 'Roman:' followed by the romanized transliteration.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_url, "detail": "auto"},
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": f"Manchu:{manchu_text}\nRoman:{roman_text}",
                    },
                ]

                json_line_data = {"messages": messages}
                f.write(json.dumps(json_line_data) + "\n")
                lines_written += 1
            except Exception:
                continue

    return output_filename if lines_written > 0 else None


if __name__ == "__main__":
    train_dataset, val_dataset = prepare_training_datasets("train", "validation")
    print(train_dataset[0])
    print(val_dataset[0])
