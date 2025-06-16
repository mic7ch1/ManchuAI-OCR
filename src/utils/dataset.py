import json
import base64
import io
import random
from datasets import Dataset, Features, Value, Image as HFImageFeatures
from PIL import Image as PILImage
from pathlib import Path
from tqdm import tqdm
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


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
        return Dataset.from_generator(
            lambda: (
                {
                    "im": PILImage.open(
                        data_root / split_name / "images" / item["image_filename"]
                    ).convert("RGB"),
                    "manchu": item["manchu"] or "",
                    "roman": item["roman"] or "",
                    "image_path": str(
                        data_root / split_name / "images" / item["image_filename"]
                    ),
                }
                for item in json.load(f)
            ),
            features=features,
        )


def prepare_training_datasets(train_key, val_key):
    data_root = project_root / "data_padded"
    train_dataset = load_split(data_root, train_key)
    val_dataset = load_split(data_root, val_key)
    return train_dataset, val_dataset


def prepare_evaluation_datasets(val_key, test_key):
    data_root = project_root / "data_padded"
    val_dataset = load_split(data_root, val_key)
    test_dataset = load_split(data_root, test_key)
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
