import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


def get_model_path(model_class, model_name, step_num=None):
    model_path = project_root / "models" / model_class / model_name
    if step_num == "best":
        best_model_path = model_path / "best_model"
        if best_model_path.exists():
            model_path = best_model_path
            print(model_path)
        else:
            # Check for CRNN-style best_model.pth file
            crnn_best_model = model_path / "checkpoints" / "best_model.pth"
            if crnn_best_model.exists():
                model_path = crnn_best_model
            else:
                # If best_model directory doesn't exist, find the best checkpoint
                model_path = find_best_model(model_class, model_name)

    elif step_num == "latest":
        checkpoints_dir = model_path / "checkpoints"

        # Check for VLM-style checkpoints (checkpoint-* directories)
        checkpoints = list(checkpoints_dir.glob("checkpoint-*"))
        if checkpoints:
            model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        else:
            # Check for CRNN-style checkpoints (ckpt_epoch_*.pth files)
            crnn_checkpoints = list(checkpoints_dir.glob("ckpt_epoch_*.pth"))
            if crnn_checkpoints:
                # Extract epoch numbers and find the latest one
                latest_checkpoint = max(
                    crnn_checkpoints, key=lambda p: int(p.stem.split("_")[-1])
                )
                model_path = latest_checkpoint
            else:
                model_path = None

    elif isinstance(step_num, int) and step_num > 0:
        model_path = model_path / "checkpoints" / f"checkpoint-{step_num}"
    else:
        raise ValueError(
            f"Invalid step number: {step_num}. step_num should be 'best', 'latest', or an integer."
        )

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return str(model_path)


def find_best_model(model_class, model_name):
    model_path = project_root / "models" / model_class / model_name
    checkpoints_dir = model_path / "checkpoints"
    checkpoints = list(checkpoints_dir.glob("checkpoint-*"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    latest_model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    trainer_state_file = latest_model_path / "trainer_state.json"

    with open(trainer_state_file, "r") as f:
        trainer_state = json.load(f)
        log_history = trainer_state["log_history"]
        log_history = [l for l in log_history if l.get("eval_loss") is not None]

        if not log_history:
            raise ValueError("No evaluation logs found with eval_loss")

        log_history = sorted(log_history, key=lambda x: x["eval_loss"])

        # Try to find the best available checkpoint
        best_checkpoint_path = None
        for entry in log_history:
            step = entry["step"]
            checkpoint_path = model_path / "checkpoints" / f"checkpoint-{step}"
            if checkpoint_path.exists():
                best_checkpoint_path = checkpoint_path
                break

        if best_checkpoint_path is None:
            raise FileNotFoundError(
                f"No valid checkpoint found for any of the best models in {model_path / 'checkpoints'}"
            )

    # Create best_model directory as a symbolic link to the best checkpoint
    best_model_path = model_path / "best_model"

    # Remove existing best_model if it exists
    if best_model_path.exists() or best_model_path.is_symlink():
        best_model_path.unlink()

    # Create symbolic link to the best checkpoint
    best_model_path.symlink_to(best_checkpoint_path)

    return best_model_path
