import numpy as np
from sklearn.metrics import average_precision_score


def multilabel_metrics(targets, probabilities, threshold=0.5):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= threshold
    ap = []
    for c in range(targets.shape[1]):
        ap.append(average_precision_score(targets[:, c], probabilities[:, c]))
    tp = (predictions & (targets == 1)).sum(0)
    fp = (predictions & (targets == 0)).sum(0)
    fn = ((~predictions) & (targets == 1)).sum(0)
    per_class_precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    per_class_recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    per_class_f1 = np.divide(
        2 * per_class_precision * per_class_recall,
        per_class_precision + per_class_recall,
        out=np.zeros_like(per_class_precision),
        where=(per_class_precision + per_class_recall) > 0,
    )
    class_precision = per_class_precision.mean()
    class_recall = per_class_recall.mean()
    cf1 = per_class_f1.mean()
    tp_o, fp_o, fn_o = tp.sum(), fp.sum(), fn.sum()
    overall_precision = tp_o / max(tp_o + fp_o, 1)
    overall_recall = tp_o / max(tp_o + fn_o, 1)
    of1 = 2 * overall_precision * overall_recall / max(overall_precision + overall_recall, 1e-12)
    fpr = fp_o / max((targets == 0).sum(), 1)
    return {"mAP": 100 * float(np.mean(ap)), "AP": [100 * float(x) for x in ap], "CP": 100 * float(class_precision), "CR": 100 * float(class_recall), "CF1": 100 * float(cf1), "OP": 100 * float(overall_precision), "OR": 100 * float(overall_recall), "OF1": 100 * float(of1), "FPR": 100 * float(fpr), "threshold": float(threshold)}

