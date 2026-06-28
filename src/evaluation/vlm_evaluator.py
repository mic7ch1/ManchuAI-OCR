import sys
import time
from pathlib import Path
from tqdm import tqdm
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.evaluation.metrics import (
    calculate_metrics,
    update_best_by_metric,
    get_completed_iterations,
    load_iteration_metrics,
    compute_replication_summary,
)
from src.evaluation.utils import get_model_path, load_vlm_model, cleanup_gpu, get_checkpoint_path, run_repeated_inference, print_header
from src.utils.files import load_json, save_json


def evaluate_vlm(
    model_config,
    dataset_config,
    val_dataset,
    test_dataset,
    evaluation_output,
    num_samples,
    mode="best",
    checkpoint_override=None,
    max_new_tokens=1536,
):
    """
    Evaluate VLM model.

    Args:
        mode: "best", "checkpoints", or "replications"
        checkpoint_override: Specific checkpoint step (for replications mode)
        max_new_tokens: Maximum tokens to generate (default: 1536)
    """
    model_name = model_config["name"]
    model_class = model_config["model_class"]

    metrics_dir = evaluation_output / "metrics" / model_name
    predictions_dir = evaluation_output / "predictions" / model_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if mode == "checkpoints":
        return _evaluate_all_checkpoints(
            model_name, model_class, dataset_config, val_dataset,
            metrics_dir, predictions_dir, num_samples, max_new_tokens
        )
    elif mode == "replications":
        return _evaluate_replications(
            model_name, model_class, dataset_config, test_dataset,
            metrics_dir, predictions_dir, num_samples, checkpoint_override, max_new_tokens
        )
    else:
        _evaluate_best_checkpoint(
            model_name, model_class, dataset_config, val_dataset, test_dataset,
            metrics_dir, predictions_dir, num_samples, max_new_tokens
        )


def _evaluate_best_checkpoint(
    model_name, model_class, dataset_config, val_dataset, test_dataset,
    metrics_dir, predictions_dir, num_samples, max_new_tokens
):
    """Evaluate best checkpoint on validation and test sets."""
    try:
        model_path = get_model_path(model_class, model_name, "best")
    except FileNotFoundError:
        print(f"\n[ERROR] Best model not found for {model_name}")
        print(f"\n  Please run checkpoint evaluation first to determine the best checkpoint:")
        print(f"    python scripts/evaluate_checkpoints.py --target-model {model_name}")
        return

    print(f"Start validating {model_name}...")
    print(f"Model path: {model_path}")

    best_model_dir = project_root / "models" / model_class / model_name / "best_model"
    trainer_state_file = best_model_dir / "trainer_state.json"
    best_step = None
    if trainer_state_file.exists():
        trainer_state = load_json(trainer_state_file, {})
        best_step = trainer_state.get("best_step")

    model, tokenizer = load_vlm_model(model_path)

    best_pred_dir = predictions_dir / "best_checkpoint"
    best_metrics_dir = metrics_dir / "best_checkpoint"
    best_pred_dir.mkdir(parents=True, exist_ok=True)
    best_metrics_dir.mkdir(parents=True, exist_ok=True)

    results = inference_vlm(model, tokenizer, val_dataset, dataset_config, num_samples, max_new_tokens)
    save_json(best_pred_dir / "validation.json", results)
    val_metrics = calculate_metrics(results, model_name, best_step)
    val_metrics["best_step"] = best_step
    save_json(best_metrics_dir / "validation.json", val_metrics)

    results = inference_vlm(model, tokenizer, test_dataset, dataset_config, num_samples, max_new_tokens)
    save_json(best_pred_dir / "test.json", results)
    test_metrics = calculate_metrics(results, model_name, best_step)
    test_metrics["best_step"] = best_step
    save_json(best_metrics_dir / "test.json", test_metrics)


def _evaluate_all_checkpoints(
    model_name, model_class, dataset_config, val_dataset,
    metrics_dir, predictions_dir, num_samples, max_new_tokens
):
    """Evaluate all checkpoints for a VLM model."""
    checkpoints_dir = project_root / "models" / model_class / model_name / "checkpoints"

    if not checkpoints_dir.exists():
        print(f"\n[ERROR] No checkpoints directory found for {model_name}")
        print(f"  Expected path: {checkpoints_dir}")
        print(f"\n  Please train the model first:")
        print(f"    python scripts/train.py --target-model {model_name}")
        return

    checkpoint_paths = sorted(
        checkpoints_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )

    if not checkpoint_paths:
        print(f"\n[ERROR] No checkpoints found for {model_name}")
        print(f"  Looked in: {checkpoints_dir}")
        print(f"\n  Please train the model first:")
        print(f"    python scripts/train.py --target-model {model_name}")
        return

    print_header(f"Evaluating {model_name} ({len(checkpoint_paths)} checkpoints)")

    best_primary_value = -1.0
    best_metrics = None
    best_step = None
    best_by_metric = {}

    for ckpt_path in checkpoint_paths:
        step_num = int(ckpt_path.name.split("-")[-1])
        metrics_file = metrics_dir / f"checkpoint-{step_num}_validation.json"
        predictions_file = predictions_dir / f"checkpoint-{step_num}_validation.json"

        if metrics_file.exists() and predictions_file.exists():
            print(f"Loading cached results for checkpoint-{step_num}")
            metrics = load_json(metrics_file, {})
        else:
            print(f"Evaluating {model_name} checkpoint-{step_num} ...")
            model, tokenizer = load_vlm_model(str(ckpt_path))
            results = inference_vlm(model, tokenizer, val_dataset, dataset_config, num_samples, max_new_tokens)
            metrics = calculate_metrics(results, model_name, step_num)

            save_json(metrics_file, metrics)
            save_json(predictions_file, results)

            del model, tokenizer
            cleanup_gpu()

        if metrics:
            primary_value = metrics.get("manchu_word_accuracy", 0)
            print(f"  checkpoint-{step_num}: manchu_word_accuracy = {primary_value:.4f}")

            if primary_value > best_primary_value:
                best_primary_value = primary_value
                best_metrics = metrics.copy()
                best_step = step_num

            best_by_metric = update_best_by_metric(best_by_metric, metrics, step_num)

    if best_metrics is not None:
        best_metrics["model_name"] = model_name
        best_metrics["checkpoint"] = best_step
        best_metrics["best_step"] = best_step
        best_metrics["best_by_metric"] = best_by_metric
        best_metrics_dir = metrics_dir / "best_checkpoint"
        best_metrics_dir.mkdir(parents=True, exist_ok=True)
        save_json(best_metrics_dir / "validation.json", best_metrics)
        print(f"\n*** Best: step {best_step}, manchu_word_accuracy {best_primary_value:.4f} ***")

    cleanup_gpu()

    return {"best_step": best_step, "best_accuracy": best_primary_value} if best_step else None


def _evaluate_replications(
    model_name, model_class, dataset_config, test_dataset,
    metrics_dir, predictions_dir, num_iterations, checkpoint_override, max_new_tokens
):
    """Run replication evaluation with repeated full inference."""
    if checkpoint_override is not None:
        result_name = f"{model_name}/checkpoint-{checkpoint_override}"
    else:
        result_name = f"{model_name}/best_checkpoint"

    print_header(f"Replication evaluation: {result_name}")

    checkpoint_path = get_checkpoint_path(model_name, model_class, checkpoint_override)
    if checkpoint_path is None:
        print(f"\n[ERROR] Checkpoint not found for {model_name}")
        if checkpoint_override is None:
            print(f"\n  Please run checkpoint evaluation first to determine the best checkpoint:")
            print(f"    python scripts/evaluate_checkpoints.py --target-model {model_name}")
        else:
            print(f"\n  Checkpoint {checkpoint_override} does not exist.")
            print(f"  Please train the model first:")
            print(f"    python scripts/train.py --target-model {model_name}")
        return None

    print(f"Using checkpoint: {checkpoint_path}")

    if checkpoint_override is not None:
        repl_predictions_dir = predictions_dir / f"checkpoint-{checkpoint_override}" / "validation_replications"
        repl_metrics_dir = metrics_dir / f"checkpoint-{checkpoint_override}" / "validation_replications"
    else:
        repl_predictions_dir = predictions_dir / "best_checkpoint" / "validation_replications"
        repl_metrics_dir = metrics_dir / "best_checkpoint" / "validation_replications"
    repl_predictions_dir.mkdir(parents=True, exist_ok=True)
    repl_metrics_dir.mkdir(parents=True, exist_ok=True)

    completed = get_completed_iterations(repl_predictions_dir, repl_metrics_dir)

    checkpoint_step = int(Path(checkpoint_path).name.split("-")[-1])

    if len(completed) >= num_iterations:
        print(f"Already completed. Loading results...")
        all_metrics = load_iteration_metrics(repl_metrics_dir)
    else:
        model, tokenizer = load_vlm_model(str(checkpoint_path))
        inference_fn = lambda ds, dc, n: inference_vlm(model, tokenizer, ds, dc, n, max_new_tokens)
        all_metrics = run_repeated_inference(
            inference_fn, test_dataset, dataset_config,
            repl_predictions_dir, repl_metrics_dir, num_iterations, len(completed),
            model_name, checkpoint_step
        )
        del model, tokenizer
        cleanup_gpu()

    summary = compute_replication_summary(
        model_name, model_class, checkpoint_path, checkpoint_override,
        num_iterations, len(test_dataset), all_metrics
    )
    summary_dir = repl_metrics_dir.parent
    save_json(summary_dir / "validation_replications.json", summary)
    return summary


def inference_vlm(model, tokenizer, dataset, dataset_config, num_samples, max_new_tokens=1536):
    """Run VLM inference on dataset samples."""
    sampled = dataset.shuffle().select(range(min(num_samples, len(dataset))))
    instruction = dataset_config["instruction"]
    results = []

    for sample in tqdm(sampled, desc="Processing samples"):
        image = sample[dataset_config["image_key"]]
        manchu_gt = sample[dataset_config["text_key"][0]]
        roman_gt = sample[dataset_config["text_key"][1]]

        messages = [{"role": "user", "content": [{"type": "text", "text": instruction}, {"type": "image"}]}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, use_cache=True, pad_token_id=tokenizer.eos_token_id
            )
        inference_time = (time.time() - start_time) * 1000

        tokens = output.sequences[0] if hasattr(output, "sequences") else output[0]
        predicted_text = tokenizer.decode(tokens, skip_special_tokens=True)

        manchu_pred, roman_pred = "", ""
        for line in predicted_text.split("\n"):
            line = line.strip()
            if line.lower().startswith("manchu:"):
                manchu_pred = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("roman:"):
                roman_pred = line.split(":", 1)[-1].strip()

        results.append({
            "manchu_gt": manchu_gt, "roman_gt": roman_gt,
            "manchu_pred": manchu_pred, "roman_pred": roman_pred,
            "image_path": sample["image_path"], "inference_time": inference_time,
        })

    return results
