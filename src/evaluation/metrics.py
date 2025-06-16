import difflib
from Levenshtein import distance


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


def calculate_metrics(results):

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

    return {
        "total_predictions": total_predictions,
        "manchu_word_accuracy": manchu_word_correct / total_predictions,
        "roman_word_accuracy": roman_word_correct / total_predictions,
        "manchu_cer": total_manchu_cer / total_predictions,
        "roman_cer": total_roman_cer / total_predictions,
        "manchu_f1_score": total_manchu_f1 / total_predictions,
        "roman_f1_score": total_roman_f1 / total_predictions,
        "inference_time": total_inference_time / total_predictions,
    }
