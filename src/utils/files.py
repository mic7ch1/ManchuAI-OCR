import json
from pathlib import Path
import os


def load_json(file_path, default=None):
    """Safely load JSON file with error handling.

    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist or is invalid

    Returns:
        Loaded JSON data or default value
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {file_path}")
        return default
    except Exception as e:
        print(f"Warning: Error loading {file_path}: {e}")
        return default


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
