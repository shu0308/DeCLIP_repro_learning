"""Conservative old-class pseudo-labels for an incremental task.

Selection uses only predictions from the model *before* the task starts.  The
ground-truth old labels, when available in a benchmark, are diagnostic only.
"""

import torch
from torch.utils.data import DataLoader, Subset


def select_old_labels(probabilities, positive_threshold=0.98,
                      negative_threshold=0.02, positive_fraction=0.10,
                      negative_fraction=0.30, positive_weight=0.20,
                      negative_weight=0.20):
    """Return binary pseudo-targets and weights from two deterministic views.

    probabilities has shape [examples, old classes, 2]. Both views must pass
    the relevant threshold. Each class also has a coverage cap. A class with
    no selected positives or negatives supplies no supervision at this stage.
    """
    if probabilities.ndim != 3 or probabilities.shape[-1] != 2:
        raise ValueError("probabilities must have shape [examples, classes, 2]")
    if not 0 <= negative_threshold < positive_threshold <= 1:
        raise ValueError("require 0 <= negative_threshold < positive_threshold <= 1")
    if not 0 <= positive_fraction <= 1 or not 0 <= negative_fraction <= 1:
        raise ValueError("selection fractions must be in [0, 1]")
    if positive_weight < 0 or negative_weight < 0:
        raise ValueError("loss weights must be non-negative")
    n, classes, _ = probabilities.shape
    means = probabilities.mean(-1)
    targets = torch.zeros((n, classes), dtype=torch.long)
    weights = torch.zeros((n, classes), dtype=torch.float32)
    positive_counts, negative_counts = [], []
    for class_idx in range(classes):
        positive = torch.nonzero(
            (probabilities[:, class_idx] >= positive_threshold).all(-1)
        ).flatten()
        negative = torch.nonzero(
            (probabilities[:, class_idx] <= negative_threshold).all(-1)
        ).flatten()
        positive_budget = int(n * positive_fraction)
        negative_budget = int(n * negative_fraction)
        if positive.numel() > positive_budget:
            order = means[positive, class_idx].argsort(descending=True)
            positive = positive[order[:positive_budget]]
        if negative.numel() > negative_budget:
            order = means[negative, class_idx].argsort()
            negative = negative[order[:negative_budget]]
        # One-sided supervision caused the all-positive failure in the prior run.
        if positive.numel() == 0 or negative.numel() == 0:
            positive_counts.append(0)
            negative_counts.append(0)
            continue
        weights[positive, class_idx] = positive_weight
        # Bound the total negative influence by the positive influence.
        balanced_negative_weight = min(
            negative_weight, positive_weight * positive.numel() / negative.numel()
        )
        weights[negative, class_idx] = balanced_negative_weight
        targets[negative, class_idx] = 1  # CE sign 1 means absent.
        positive_counts.append(positive.numel())
        negative_counts.append(negative.numel())
    summary = {"positive_count": positive_counts,
               "negative_count": negative_counts,
               "positive_coverage": sum(positive_counts) / max(n * classes, 1),
               "negative_coverage": sum(negative_counts) / max(n * classes, 1)}
    return targets, weights, summary


@torch.no_grad()
def build_old_label_cache(model, dataset, old_class_indices, device, batch_size,
                          workers=0, amp=True, **selection_kwargs):
    """Score the current task's images once with the frozen pre-task model.

    The dataset must use a deterministic evaluation transform. A horizontal
    flip supplies a second view; no hidden old labels enter the selection.
    """
    old_class_indices = list(old_class_indices)
    if not old_class_indices:
        return {}, {"positive_count": [], "negative_count": [],
                    "positive_coverage": 0.0, "negative_coverage": 0.0}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    was_training = model.training
    model.eval()
    ids, scores = [], []
    try:
        for images, _, image_ids in loader:
            images = images.to(device, non_blocking=True)
            views = []
            for view in (images, images.flip(-1)):
                with torch.autocast(device_type=device.type,
                                    enabled=amp and device.type == "cuda"):
                    logits = model(view, old_class_indices, ast_temperature=1.0)
                    views.append(logits.softmax(-1)[..., 0].float().cpu())
            ids.extend(image_ids)
            scores.append(torch.stack(views, dim=-1))
    finally:
        model.train(was_training)
    probabilities = torch.cat(scores) if scores else torch.empty(0, len(old_class_indices), 2)
    targets, weights, summary = select_old_labels(probabilities, **selection_kwargs)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate image IDs in old-label cache")
    cache = {image_id: (targets[row], weights[row]) for row, image_id in enumerate(ids)}
    return cache, summary


def diagnose_old_label_cache(cache, dataset, old_class_indices):
    """Offline benchmark diagnostic; never used to create pseudo-labels."""
    old_class_indices = list(old_class_indices)
    indices = list(range(len(dataset)))
    while isinstance(dataset, Subset):
        indices = [dataset.indices[index] for index in indices]
        dataset = dataset.dataset
    positive_correct = positive_total = negative_correct = negative_total = 0
    for row in indices:
        true_labels, image_id = dataset.labels[row], dataset.ids[row]
        targets, weights = cache[image_id]
        actual = true_labels[old_class_indices]
        positive = (weights > 0) & (targets == 0)
        negative = (weights > 0) & (targets == 1)
        positive_total += int(positive.sum())
        negative_total += int(negative.sum())
        positive_correct += int(((actual == 1) & positive).sum())
        negative_correct += int(((actual == 0) & negative).sum())
    return {"positive_precision": (positive_correct / positive_total
                                   if positive_total else None),
            "negative_precision": (negative_correct / negative_total
                                   if negative_total else None),
            "positive_selected": positive_total,
            "negative_selected": negative_total}
