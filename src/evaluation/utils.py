import gc
import time
from pathlib import Path
from tqdm import tqdm
import torch

from unsloth import FastVisionModel
from src.CRNN.inference import CRNNInference
from src.evaluation.metrics import calculate_metrics, load_iteration_metrics
from src.utils.files import load_json, save_json

project_root = Path(__file__).resolve().parent.parent.parent




def get_model_path(model_class, model_name, step_num=None):
    """Get model path for a given checkpoint step.

    Args:
        model_class: "VLM" or "CRNN"
        model_name: Name of the model (e.g., "llama-32-11b")
        step_num: "best", "latest", or an integer step number

    Returns:
        Path to the model checkpoint
    """
    model_path = project_root / "models" / model_class / model_name
    if step_num == "best":
        best_model_dir = model_path / "best_model"
        best_model_pth = best_model_dir / "best_model.pth"

        if best_model_pth.exists():
            model_path = best_model_pth
        elif best_model_dir.exists():
            model_path = best_model_dir
        else:
            raise FileNotFoundError(
                f"Best model not found for {model_name}. "
                f"Please run: python scripts/evaluate_checkpoints.py --target-model {model_name}"
            )

    elif step_num == "latest":
        checkpoints_dir = model_path / "checkpoints"

        vlm_checkpoints = list(checkpoints_dir.glob("checkpoint-*"))
        vlm_checkpoints = [p for p in vlm_checkpoints if p.is_dir()]

        crnn_checkpoints = list(checkpoints_dir.glob("checkpoint-*.pth"))

        if vlm_checkpoints:
            model_path = max(vlm_checkpoints, key=lambda p: int(p.name.split("-")[-1]))
        elif crnn_checkpoints:
            latest_checkpoint = max(
                crnn_checkpoints, key=lambda p: int(p.stem.split("-")[-1])
            )
            model_path = latest_checkpoint
        else:
            model_path = None

    elif isinstance(step_num, int) and step_num > 0:
        checkpoints_dir = model_path / "checkpoints"
        vlm_checkpoint = checkpoints_dir / f"checkpoint-{step_num}"
        crnn_checkpoint = checkpoints_dir / f"checkpoint-{step_num}.pth"

        if vlm_checkpoint.exists():
            model_path = vlm_checkpoint
        elif crnn_checkpoint.exists():
            model_path = crnn_checkpoint
        else:
            model_path = vlm_checkpoint  # Will raise FileNotFoundError below
    else:
        raise ValueError(
            f"Invalid step number: {step_num}. step_num should be 'best', 'latest', or an integer."
        )

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return str(model_path)


def create_best_model(model_class, model_name, best_step, best_accuracy=None, metric="manchu_word_accuracy"):
    """Create best_model directory with checkpoint and trainer_state.json.

    Creates best_model/ directory containing:
    - best_model.pth: Copy of the best checkpoint
    - trainer_state.json: Contains best_step and best_accuracy info

    Args:
        model_class: "VLM" or "CRNN"
        model_name: Name of the model
        best_step: The step number of the best checkpoint
        best_accuracy: The accuracy value of the best checkpoint
        metric: Metric used for selection (saved in trainer_state.json)

    Returns:
        Path to the best_model directory
    """
    import shutil

    model_path = project_root / "models" / model_class / model_name
    checkpoints_dir = model_path / "checkpoints"
    best_model_dir = model_path / "best_model"

    if model_class == "CRNN":
        best_checkpoint_path = checkpoints_dir / f"checkpoint-{best_step}.pth"
    else:
        best_checkpoint_path = checkpoints_dir / f"checkpoint-{best_step}"

    if not best_checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint_path}")

    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    if model_class == "CRNN":
        shutil.copy2(best_checkpoint_path, best_model_dir / "best_model.pth")
    else:
        for item in best_checkpoint_path.iterdir():
            if item.is_file():
                shutil.copy2(item, best_model_dir / item.name)
            else:
                shutil.copytree(item, best_model_dir / item.name)

    trainer_state = {
        "best_step": best_step,
        "best_accuracy": best_accuracy,
        "metric": metric,
    }
    save_json(best_model_dir / "trainer_state.json", trainer_state)

    print(f"Created best_model directory: {best_model_dir}")
    print(f"  - best_step: {best_step}")
    print(f"  - best_accuracy: {best_accuracy}")

    return best_model_dir




def load_vlm_model(model_path):
    """Load VLM model."""
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path, load_in_4bit=False, load_in_8bit=False
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    FastVisionModel.for_inference(model)
    return model, tokenizer


def load_crnn_model(model_path):
    """Load CRNN model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CRNNInference.from_pretrained(model_path, device=device)


def cleanup_gpu():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    time.sleep(2)


def get_checkpoint_path(model_name, model_class, checkpoint_override=None, extension=""):
    """Get checkpoint path - specific step or best.

    Args:
        extension: File extension for checkpoint (e.g., ".pth" for CRNN, "" for VLM directories)
    """
    checkpoints_dir = project_root / "models" / model_class / model_name / "checkpoints"

    if checkpoint_override is not None:
        path = checkpoints_dir / f"checkpoint-{checkpoint_override}{extension}"
        return path if path.exists() else None

    best_file = project_root / "results" / "metrics" / model_name / "best_checkpoint" / "validation.json"
    if not best_file.exists():
        return None

    best_step = load_json(best_file, {}).get("best_step")

    if best_step is None:
        return None

    path = checkpoints_dir / f"checkpoint-{best_step}{extension}"
    return path if path.exists() else None


def run_repeated_inference(
    inference_fn, dataset, dataset_config,
    predictions_dir, metrics_dir, num_iterations, start_iteration,
    model_name=None, checkpoint=None
):
    """Run inference num_iterations times on full dataset.

    Args:
        inference_fn: Callable that takes (dataset, dataset_config, num_samples) and returns results
        model_name: Name of the model for metadata
        checkpoint: Checkpoint identifier for metadata ("best" or step number)
    """
    remaining = num_iterations - start_iteration
    print(f"Running {remaining} inference passes on {len(dataset)} samples...")

    for iteration in tqdm(range(start_iteration, num_iterations), desc="Iterations"):
        results = inference_fn(dataset, dataset_config, len(dataset))
        metrics = calculate_metrics(results, model_name, checkpoint)

        iter_num = f"{iteration + 1:03d}"
        save_json(predictions_dir / f"checkpoint-{checkpoint}-{iter_num}_validation.json", results)
        save_json(metrics_dir / f"checkpoint-{checkpoint}-{iter_num}_validation.json", metrics)

    return load_iteration_metrics(metrics_dir)


def print_header(message, width=60):
    """Print a formatted header with separators."""
    print(f"\n{'=' * width}")
    print(message)
    print('=' * width)
