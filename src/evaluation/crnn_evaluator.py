import sys
import time
from pathlib import Path
from tqdm import tqdm


project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.model import get_model_path
from src.evaluation.metrics import calculate_metrics
from src.utils.files import save_json
from src.CRNN.inference import CRNNInference


def evaluate_crnn_model(
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

    print("Loading CRNN model...")
    crnn_model = CRNNInference.from_pretrained(model_path, device="cuda")

    validation_results = inference_crnn_model(
        crnn_model,
        val_dataset,
        dataset_config,
        num_samples,
    )

    save_json(str(validation_dir / model_name) + ".json", validation_results)

    validation_metrics = calculate_metrics(validation_results)
    save_json(str(metrics_dir / model_name) + "_validation.json", validation_metrics)

    test_num_samples = test_config["num_samples"]
    test_step_num = test_config["step_num"]
    if test_step_num == "best":
        model_path = get_model_path(model_class, model_name, test_step_num)
    elif test_step_num == "latest":
        model_path = get_model_path(model_class, model_name, test_step_num)
    else:
        model_path = get_model_path(model_class, model_name, test_step_num)

    if test_step_num != val_step_num:
        print(f"Loading test model from {model_path}...")
        crnn_model = CRNNInference.from_pretrained(model_path, device="cuda")

    test_results = inference_crnn_model(
        crnn_model,
        test_dataset,
        dataset_config,
        test_num_samples,
    )

    save_json(str(test_dir / model_name) + ".json", test_results)
    test_metrics = calculate_metrics(test_results)
    save_json(str(metrics_dir / model_name) + "_test.json", test_metrics)


def inference_crnn_model(crnn_model, dataset, dataset_config, num_samples):
    sampled_dataset = dataset.shuffle().select(range(min(num_samples, len(dataset))))

    results = []

    for sample in tqdm(sampled_dataset, desc="Processing samples"):
        image = sample[dataset_config["image_key"]]
        image_path = sample["image_path"]
        manchu_ground_truth = sample[dataset_config["text_key"][0]]
        roman_ground_truth = sample[dataset_config["text_key"][1]]

        start_time = time.time()
        predicted_text = crnn_model.inference(image)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000

        manchu_pred = predicted_text
        roman_pred = ""

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
