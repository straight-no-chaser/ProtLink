import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score


def select_threshold(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5

    f1_values = 2 * precision[:-1] * recall[:-1]
    f1_values = f1_values / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_idx = int(np.nanargmax(f1_values))
    return float(thresholds[best_idx])


def compute_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)

    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = float("nan")

    try:
        ap = float(average_precision_score(y_true, scores))
    except ValueError:
        ap = float("nan")

    return {
        "roc_auc": roc_auc,
        "ap": ap,
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }


def summarize_metric_dicts(metric_dicts):
    if not metric_dicts:
        return {}

    summary = {}
    for key in metric_dicts[0]:
        values = np.asarray([metrics[key] for metrics in metric_dicts], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return summary
