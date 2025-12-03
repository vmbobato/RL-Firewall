def compute_confusion_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }
