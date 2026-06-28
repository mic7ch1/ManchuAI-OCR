import difflib
import numpy as np
from Levenshtein import distance

from src.utils.files import load_json

METRICS_TO_TRACK = [
    "manchu_cer",
    "roman_cer",
    "manchu_word_accuracy",
    "roman_word_accuracy",
    "manchu_f1_score",
    "roman_f1_score",
    "avg_inference_time",
]


def calculate_word_accuracy(gt, pred):
    gt = gt.strip()
    pred = pred.strip()
    return gt == pred


def calculate_cer(gt, pred):

    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    edit_distance = distance(gt, pred)
    cer = edit_distance / len(gt)

    return cer


def calculate_f1_score(gt, pred):

    if len(gt) == 0 and len(pred) == 0:
        return 1.0
    if len(gt) == 0 or len(pred) == 0:
        return 0.0

    gt_chars = {}
    pred_chars = {}

    for i, char in enumerate(gt):
        gt_chars[i] = char

    for i, char in enumerate(pred):
        pred_chars[i] = char

    matcher = difflib.SequenceMatcher(None, gt, pred)
    matches = matcher.get_matching_blocks()

    true_positives = sum(match.size for match in matches[:-1])

    precision = true_positives / len(pred) if len(pred) > 0 else 0.0
    recall = true_positives / len(gt) if len(gt) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def calculate_metrics(results, model_name=None, checkpoint=None):
    """Calculate evaluation metrics from results.

    Args:
        results: List of prediction results
        model_name: Name of the model (e.g., "llama-32-11b")
        checkpoint: Checkpoint identifier ("best" or step number like 1000)

    Returns:
        Dictionary with metrics including model_name and checkpoint if provided
    """
    if not results:
        return {}

    total_predictions = len(results)
    manchu_word_correct = 0
    roman_word_correct = 0
    total_manchu_cer = 0
    total_roman_cer = 0
    total_manchu_f1 = 0
    total_roman_f1 = 0
    total_inference_time = 0

    error_counter = calculate_frequent_error(results)

    for result in results:
        manchu_gt = result["manchu_gt"]
        manchu_pred = result["manchu_pred"]
        roman_gt = result["roman_gt"]
        roman_pred = result["roman_pred"]
        inference_time = result.get("inference_time", 0)

        if calculate_word_accuracy(manchu_gt, manchu_pred):
            manchu_word_correct += 1
        if calculate_word_accuracy(roman_gt, roman_pred):
            roman_word_correct += 1

        total_manchu_cer += calculate_cer(manchu_gt, manchu_pred)
        total_roman_cer += calculate_cer(roman_gt, roman_pred)
        total_manchu_f1 += calculate_f1_score(manchu_gt, manchu_pred)
        total_roman_f1 += calculate_f1_score(roman_gt, roman_pred)
        total_inference_time += inference_time

    metrics = {
        "model_name": model_name,
        "checkpoint": checkpoint,
        "total_predictions": total_predictions,
        "manchu_word_accuracy": manchu_word_correct / total_predictions,
        "roman_word_accuracy": roman_word_correct / total_predictions,
        "manchu_cer": total_manchu_cer / total_predictions,
        "roman_cer": total_roman_cer / total_predictions,
        "manchu_f1_score": total_manchu_f1 / total_predictions,
        "roman_f1_score": total_roman_f1 / total_predictions,
        "inference_time": total_inference_time / total_predictions,
        "frequent_error": error_counter,
    }

    if model_name is None:
        del metrics["model_name"]
    if checkpoint is None:
        del metrics["checkpoint"]

    return metrics


def calculate_frequent_error(results):
    errors_manchu = {}
    errors_roman = {}

    def _accumulate(errors_dict, gt, pred):
        max_len = max(len(gt), len(pred))
        for i in range(max_len):
            gt_ch = gt[i] if i < len(gt) else ""
            pred_ch = pred[i] if i < len(pred) else ""
            if gt_ch != pred_ch and gt_ch:
                errors_dict[gt_ch] = errors_dict.get(gt_ch, 0) + 1

    for r in results:
        _accumulate(errors_manchu, r["manchu_gt"], r["manchu_pred"])
        _accumulate(errors_roman, r["roman_gt"], r["roman_pred"])

    return {"manchu": errors_manchu, "roman": errors_roman}


def update_best_by_metric(best_by_metric, metrics, step_num):
    """Update best checkpoint tracking for each metric."""
    metric_preferences = {
        "manchu_cer": "min",
        "roman_cer": "min",
        "manchu_word_accuracy": "max",
        "roman_word_accuracy": "max",
        "manchu_f1_score": "max",
        "roman_f1_score": "max",
        "avg_inference_time": "min",
    }
    for metric, preference in metric_preferences.items():
        value = metrics.get(metric)
        if value is None:
            continue

        current_best = best_by_metric.get(metric)
        if current_best is None:
            best_by_metric[metric] = {"best_step": step_num, "value": value}
        else:
            is_better = (preference == "max" and value > current_best["value"]) or (
                preference == "min" and value < current_best["value"]
            )
            if is_better:
                best_by_metric[metric] = {"best_step": step_num, "value": value}

    return best_by_metric


def compute_statistics(metric_values):
    """Compute mean, std, and 95% confidence interval."""
    arr = np.array(metric_values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    ci_lower = np.percentile(arr, 2.5)
    ci_upper = np.percentile(arr, 97.5)
    return {
        "mean": float(mean),
        "std": float(std),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def get_completed_iterations(predictions_dir, metrics_dir):
    """Get list of completed iteration numbers.

    Handles both old (iteration_NNN.json) and new ({checkpoint}-NNN_validation.json) naming.
    """
    if not predictions_dir.exists() or not metrics_dir.exists():
        return []

    def extract_iteration_num(filename):
        """Extract iteration number from filename."""
        stem = filename.stem
        if "_validation" in stem:
            parts = stem.replace("_validation", "").split("-")
            if len(parts) >= 2:
                try:
                    return int(parts[-1])
                except ValueError:
                    pass
        if stem.startswith("iteration_"):
            try:
                return int(stem.split("_")[1])
            except (ValueError, IndexError):
                pass
        return None

    pred_completed = set()
    for f in predictions_dir.glob("*_validation.json"):
        num = extract_iteration_num(f)
        if num is not None:
            pred_completed.add(num)
    for f in predictions_dir.glob("iteration_*.json"):
        num = extract_iteration_num(f)
        if num is not None:
            pred_completed.add(num)

    metrics_completed = set()
    for f in metrics_dir.glob("*_validation.json"):
        num = extract_iteration_num(f)
        if num is not None:
            metrics_completed.add(num)
    for f in metrics_dir.glob("iteration_*.json"):
        num = extract_iteration_num(f)
        if num is not None:
            metrics_completed.add(num)

    return sorted(pred_completed & metrics_completed)


def load_iteration_metrics(metrics_dir):
    """Load all completed iteration metrics.

    Handles both old (iteration_NNN.json) and new ({checkpoint}-NNN_validation.json) naming.
    """
    all_metrics = {metric: [] for metric in METRICS_TO_TRACK}

    iteration_files = list(metrics_dir.glob("*_validation.json"))
    iteration_files.extend(metrics_dir.glob("iteration_*.json"))

    def get_sort_key(f):
        stem = f.stem
        if "_validation" in stem:
            parts = stem.replace("_validation", "").split("-")
            if len(parts) >= 2:
                try:
                    return int(parts[-1])
                except ValueError:
                    pass
        if stem.startswith("iteration_"):
            try:
                return int(stem.split("_")[1])
            except (ValueError, IndexError):
                pass
        return 0

    iteration_files = sorted(set(iteration_files), key=get_sort_key)

    for iter_file in iteration_files:
        metrics = load_json(iter_file, {})

        for metric in METRICS_TO_TRACK:
            if metric in metrics:
                all_metrics[metric].append(metrics[metric])

    return all_metrics


def compute_replication_summary(
    model_name, model_class, checkpoint_path, checkpoint_override,
    num_iterations, dataset_size, all_metrics
):
    """Compute and display replication summary."""
    result_name = (
        f"{model_name}_{checkpoint_override}"
        if checkpoint_override is not None
        else model_name
    )

    summary = {
        "model_name": model_name,
        "model_class": model_class,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_override": checkpoint_override,
        "num_iterations": num_iterations,
        "samples_per_iteration": dataset_size,
        "total_inferences": num_iterations * dataset_size,
        "test_dataset_size": dataset_size,
        "replication_metrics": {},
        "replication_raw_values": {},
    }

    print(f"\nRepeated inference results for {result_name}:")
    print(
        f"({num_iterations} iterations × {dataset_size} samples = "
        f"{num_iterations * dataset_size:,} total inferences)"
    )

    for metric in METRICS_TO_TRACK:
        if all_metrics[metric]:
            stats = compute_statistics(all_metrics[metric])
            summary["replication_metrics"][metric] = stats
            summary["replication_raw_values"][metric] = all_metrics[metric]
            print(
                f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"(95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])"
            )

    return summary
