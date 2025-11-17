import json
from pathlib import Path
import sys

# Project root is two levels above this script
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.evaluation.metrics import calculate_frequent_error


def update_metrics_with_error_frequency():
    results_dir = project_root / "results"

    for split in ["test", "validation"]:
        split_dir = results_dir / split
        if not split_dir.exists():
            continue

        for result_file in split_dir.glob("*.json"):
            # Load evaluation results (list of dicts)
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except Exception as e:
                print(f"Skipping {result_file}: {e}")
                continue

            # Compute character error frequency
            error_freq = calculate_frequent_error(results)

            # Determine corresponding metrics file path
            model_name = result_file.stem  # filename without .json
            metrics_filename = f"{model_name}_{split}.json"
            metrics_path = results_dir / "metrics" / metrics_filename

            if not metrics_path.exists():
                print(
                    f"Metrics file not found for {model_name} ({split}), creating new one."
                )
                metrics_data = {}
            else:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)

            metrics_data["character_error_frequency"] = error_freq

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, ensure_ascii=False, indent=4)

            total_chars = (
                sum(len(v) for v in error_freq.values())
                if isinstance(error_freq, dict)
                else len(error_freq)
            )
            print(
                f"Updated {metrics_path} with character_error_frequency (total {total_chars} distinct chars – Manchu: {len(error_freq.get('manchu', {}))}, Roman: {len(error_freq.get('roman', {}))})."
            )


if __name__ == "__main__":
    update_metrics_with_error_frequency()
