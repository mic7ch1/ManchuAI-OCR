import sys
import time
from pathlib import Path
from unsloth import FastVisionModel
from tqdm import tqdm
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.model import get_model_path
from src.evaluation.metrics import calculate_metrics
from src.utils.files import save_json


def evaluate_vlm_model(
    model_config,
    evaluation_config,
    dataset_config,
    val_dataset,
    test_dataset,
    evaluation_output,
):

    model_name = model_config["name"]
    model_class = model_config["model_class"]

    validation_config = evaluation_config["validation"]
    test_config = evaluation_config["test"]

    val_step_num = validation_config["step_num"]
    num_samples = validation_config["num_samples"]
    metrics_dir = evaluation_output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if val_step_num == "best":
        model_path = get_model_path(model_class, model_name, val_step_num)

    elif val_step_num == "latest":
        model_path = get_model_path(model_class, model_name, val_step_num)
    else:
        model_path = get_model_path(model_class, model_name, val_step_num)

    print(f"Start validating {model_name}...")
    print(f"Model path: {model_path}")

    validation_dir = evaluation_output / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    test_dir = evaluation_output / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    model = model.to("cuda")
    FastVisionModel.for_inference(model)

    validation_results = inference_model(
        model,
        tokenizer,
        val_dataset,
        dataset_config,
        num_samples,
    )

    save_json(str(validation_dir / model_name) + ".json", validation_results)

    validation_metrics = calculate_metrics(validation_results)
    save_json(str(metrics_dir / model_name) + "_validation.json", validation_metrics)

    # Test
    test_num_samples = test_config["num_samples"]
    test_step_num = test_config["step_num"]
    if test_step_num == "best":
        model_path = get_model_path(model_class, model_name, test_step_num)
    elif test_step_num == "latest":
        model_path = get_model_path(model_class, model_name, test_step_num)
    else:
        model_path = get_model_path(model_class, model_name, test_step_num)

    test_results = inference_model(
        model,
        tokenizer,
        test_dataset,
        dataset_config,
        test_num_samples,
    )
    save_json(str(test_dir / model_name) + ".json", test_results)
    test_metrics = calculate_metrics(test_results)
    save_json(str(metrics_dir / model_name) + "_test.json", test_metrics)


def inference_model(model, tokenizer, dataset, dataset_config, num_samples):
    sampled_dataset = dataset.shuffle().select(range(min(num_samples, len(dataset))))

    results = []
    instruction = dataset_config["instruction"]

    for sample in tqdm(sampled_dataset, desc="Processing validation samples"):
        image = sample[dataset_config["image_key"]]
        image_path = sample["image_path"]
        manchu_ground_truth = sample[dataset_config["text_key"][0]]
        roman_ground_truth = sample[dataset_config["text_key"][1]]

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": instruction}],
            }
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1536,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000

        if hasattr(output, "sequences"):
            generated_tokens = output.sequences[0]
        else:
            generated_tokens = output[0] if output.dim() > 1 else output

        predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
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
