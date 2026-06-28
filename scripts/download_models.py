"""
Download pre-trained Manchu OCR models from Hugging Face.
Models are saved to models/{model_class}/{model_name}/best_model/
"""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "llama-32-11b": {
        "repo_id": "mic7ch/manchu-ocr-llama-32-11b",
        "model_class": "VLM",
    },
    "qwen-25-3b": {
        "repo_id": "mic7ch/manchu-ocr-qwen-25-3b",
        "model_class": "VLM",
    },
    "qwen-25-7b": {
        "repo_id": "mic7ch/manchu-ocr-qwen-25-7b",
        "model_class": "VLM",
    },
    "crnn-base-3m": {
        "repo_id": "mic7ch/manchu-ocr-crnn-base-3m",
        "model_class": "CRNN",
    },
}


def download_model(model_name: str, force: bool = False):
    """Download a single model from Hugging Face."""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False

    model_info = MODELS[model_name]
    target_dir = PROJECT_ROOT / "models" / model_info["model_class"] / model_name / "best_model"

    if target_dir.exists() and not force:
        print(f"Model {model_name} already exists at {target_dir}")
        print("Use --force to re-download")
        return True

    print(f"Downloading {model_name} from {model_info['repo_id']}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_info["repo_id"],
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded {model_name} to {target_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained Manchu OCR models")
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        default=None,
        help=f"Model(s) to download. Available: {', '.join(MODELS.keys())}. Downloads all if not specified.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists",
    )

    args = parser.parse_args()

    models_to_download = args.model if args.model else list(MODELS.keys())

    print(f"Downloading {len(models_to_download)} model(s)...\n")

    for model_name in models_to_download:
        download_model(model_name, force=args.force)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
