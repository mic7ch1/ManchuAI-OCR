import json
import sys
import time
import base64
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from PIL import Image
import io

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.evaluation.metrics import calculate_metrics
from src.utils.files import save_json


def evaluate_openai_model(
    model_config,
    evaluation_config,
    dataset_config,
    val_dataset,
    test_dataset,
    evaluation_output,
):
    model_name = model_config["name"]
    api_key = model_config.get("api_key")
    model_id = model_config.get("model_id")

    if not api_key:
        raise ValueError("OpenAI API key must be provided in model_config")

    validation_config = evaluation_config["validation"]
    test_config = evaluation_config["test"]

    num_samples = validation_config["num_samples"]
    metrics_dir = evaluation_output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Start evaluating {model_name} using OpenAI API...")
    print(f"Model ID: {model_id}")

    validation_dir = evaluation_output / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    test_dir = evaluation_output / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key)

    print("Running validation...")
    validation_results = inference_openai_model(
        client,
        model_id,
        val_dataset,
        dataset_config,
        num_samples,
    )

    save_json(str(validation_dir / model_name) + ".json", validation_results)

    validation_metrics = calculate_metrics(validation_results)
    save_json(str(metrics_dir / model_name) + "_validation.json", validation_metrics)

    print("Running test...")
    test_num_samples = test_config["num_samples"]

    test_results = inference_openai_model(
        client,
        model_id,
        test_dataset,
        dataset_config,
        test_num_samples,
    )
    save_json(str(test_dir / model_name) + ".json", test_results)
    test_metrics = calculate_metrics(test_results)
    save_json(str(metrics_dir / model_name) + "_test.json", test_metrics)

    print(f"Evaluation completed for {model_name}")


def encode_image_to_base64(image):
    if isinstance(image, str):
        with open(image, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    elif hasattr(image, "save"):
        buffer = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError("Image must be a file path or PIL Image")


def inference_openai_model(client, model_id, dataset, dataset_config, num_samples):
    sampled_dataset = dataset.shuffle().select(range(min(num_samples, len(dataset))))

    results = []
    instruction = dataset_config["instruction"]

    for sample in tqdm(sampled_dataset, desc="Processing samples"):
        image = sample[dataset_config["image_key"]]
        image_path = sample["image_path"]
        manchu_ground_truth = sample[dataset_config["text_key"][0]]
        roman_ground_truth = sample[dataset_config["text_key"][1]]

        try:
            base64_image = encode_image_to_base64(image)
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]

        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=128,
                temperature=0.0,
            )
            predicted_text = response.choices[0].message.content
        except Exception as e:
            print(f"Error with OpenAI API for image {image_path}: {e}")
            predicted_text = ""

        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

        manchu_pred, roman_pred = "", ""
        for line in predicted_text.split("\n"):
            line_stripped = line.strip()
            if line_stripped.lower().startswith("manchu:"):
                manchu_pred = line_stripped.split(":", 1)[-1].strip()
            elif line_stripped.lower().startswith("roman:"):
                roman_pred = line_stripped.split(":", 1)[-1].strip()

        results.append(
            {
                "manchu_gt": manchu_ground_truth,
                "roman_gt": roman_ground_truth,
                "manchu_pred": manchu_pred,
                "roman_pred": roman_pred,
                "image_path": image_path,
                "inference_time": inference_time,
            }
        )

    return results
