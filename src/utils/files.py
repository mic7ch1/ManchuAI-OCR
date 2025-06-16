import json
from pathlib import Path
import os


def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def create_dir(file_path):
    if isinstance(file_path, Path):
        file_path.mkdir(parents=True, exist_ok=True)
    elif isinstance(file_path, str):
        os.makedirs(file_path, exist_ok=True)
    else:
        raise ValueError(f"Invalid file path: {file_path}")
