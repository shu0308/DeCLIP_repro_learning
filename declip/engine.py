import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from .metrics import multilabel_metrics


def _class_chunks(class_indices, class_chunk_size):
    class_indices = list(class_indices)
    size = class_chunk_size or len(class_indices)
    if size <= 0:
        raise ValueError("class_chunk_size must be non-negative")
    return [(start, min(start + size, len(class_indices)),
             class_indices[start:start + size])
            for start in range(0, len(class_indices), size)]


def train_task(model, loader, class_indices, optimizer, device, epochs=5, amp=True,
               negative_weight=1.0, class_chunk_size=0, old_class_indices=(),
               reliable_old_labels=False, old_confidence_threshold=0.80,
               old_positive_loss_weight=0.20, old_logit_margin=0.0,
               old_label_cache=None):
    class_indices = list(class_indices)
    negative_weight = torch.as_tensor(negative_weight, dtype=torch.float32)
    if negative_weight.numel() not in (1, len(class_indices)) or (negative_weight <= 0).any():
        raise ValueError("negative_weight must contain one or one-per-class positive value")
    chunks = _class_chunks(class_indices, class_chunk_size)
    old_class_indices = list(old_class_indices)
    old_positions = [class_indices.index(index) for index in old_class_indices]
    if reliable_old_labels and old_label_cache is not None:
        raise ValueError("choose either the legacy oracle filter or pseudo-label cache")
    if old_label_cache is not None and not old_class_indices:
        raise ValueError("old_class_indices are required for an old-label cache")
    if not 0.0 <= old_confidence_threshold <= 1.0:
        raise ValueError("old_confidence_threshold must be in [0, 1]")
    if old_positive_loss_weight < 0:
        raise ValueError("old_positive_loss_weight must be non-negative")
    if old_logit_margin < 0:
        raise ValueError("old_logit_margin must be non-negative")
    model.freeze_except(class_indices)
    model.train()
    scaler = GradScaler(enabled=amp and device.type == "cuda")
    for epoch in range(epochs):
        total_loss = 0.0
        reliable_old_count = 0
        old_positive_count = 0
        for images, labels, image_ids in loader:
            images = images.to(device, non_blocking=True)
            targets = (1 - labels[:, class_indices]).to(device, non_blocking=True).long()
            negative_weight_device = negative_weight.to(device)
            if negative_weight_device.numel() > 1:
                negative_weight_device = negative_weight_device.view(1, -1)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                shared = model._shared_visual_prefix(images)
                if reliable_old_labels and old_class_indices:
                    old_logits = model.forward_from_shared(shared, old_class_indices,
                                                           ast_temperature=1.0)
                    old_probabilities = old_logits.softmax(-1)[..., 0]
                    # A high positive probability alone can be overconfident.
                    # Require its positive prompt score to also beat the paired
                    # negative prompt score by a meaningful margin.
                    old_margins = old_logits[..., 0] - old_logits[..., 1]
                    old_targets = labels[:, old_class_indices].to(device, non_blocking=True)
                    reliable_old = ((old_targets > 0.5) &
                                    (old_probabilities >= old_confidence_threshold) &
                                    (old_margins >= old_logit_margin))
                    reliable_old_count += reliable_old.sum().item()
                    old_positive_count += (old_targets > 0.5).sum().item()
            full_weights = torch.where(targets == 1, negative_weight_device, 1.0)
            if reliable_old_labels and old_class_indices:
                # Do not make partial old negatives into hard supervision.
                # Only confidence-screened, real old positives get a weak loss.
                full_weights[:, old_positions] = (
                    reliable_old.to(full_weights.dtype) * old_positive_loss_weight)
            if old_label_cache is not None:
                cached = [old_label_cache[image_id] for image_id in image_ids]
                old_targets = torch.stack([row[0] for row in cached]).to(device)
                old_weights = torch.stack([row[1] for row in cached]).to(device)
                targets[:, old_positions] = old_targets
                full_weights[:, old_positions] = old_weights
            denominator = full_weights.sum().clamp_min(1.0)
            batch_loss = 0.0
            for start, end, chunk in chunks:
                chunk_targets = targets[:, start:end]
                with autocast(enabled=scaler.is_enabled()):
                    logits = model.forward_from_shared(shared, chunk, ast_temperature=1.0)
                    per_label = F.cross_entropy(
                        logits.reshape(-1, 2), chunk_targets.reshape(-1),
                        reduction="none").reshape_as(chunk_targets)
                    loss = (per_label * full_weights[:, start:end]).sum() / denominator
                scaler.scale(loss).backward()
                batch_loss += loss.detach().item()
                del logits, per_label, loss
            del shared
            scaler.step(optimizer)
            scaler.update()
            total_loss += batch_loss * images.shape[0]
        log = {"epoch": epoch + 1, "loss": total_loss / len(loader.dataset)}
        if reliable_old_labels and old_class_indices:
            log["reliable_old_ratio"] = reliable_old_count / max(old_positive_count, 1)
        yield log


@torch.no_grad()
def evaluate(model, loader, seen_indices, task_number, device, ast_strength=16.0,
             amp=True, decision_threshold=0.80, class_chunk_size=0):
    model.eval()
    temperature = model.ast_temperature(task_number, len(seen_indices), ast_strength)
    chunks = _class_chunks(seen_indices, class_chunk_size)
    predictions, targets = [], []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            shared = model._shared_visual_prefix(images)
        batch_probabilities = []
        for _, _, chunk in chunks:
            with autocast(enabled=amp and device.type == "cuda"):
                logits = model.forward_from_shared(shared, chunk, ast_temperature=temperature)
                batch_probabilities.append(logits.softmax(-1)[..., 0])
        predictions.append(torch.cat(batch_probabilities, dim=1).cpu())
        del shared, batch_probabilities
        targets.append(labels[:, seen_indices])
    result = multilabel_metrics(torch.cat(targets).numpy(),
                                torch.cat(predictions).numpy(), decision_threshold)
    result["AST_tau"] = temperature
    return result


@torch.no_grad()
def calibrate_threshold(model, loader, class_indices, task_number, seen_class_count, device,
                        ast_strength=16.0, amp=True, objective="OF1", class_chunk_size=0):
    model.eval()
    temperature = model.ast_temperature(task_number, seen_class_count, ast_strength)
    chunks = _class_chunks(class_indices, class_chunk_size)
    probabilities, targets = [], []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            shared = model._shared_visual_prefix(images)
        batch_probabilities = []
        for _, _, chunk in chunks:
            with autocast(enabled=amp and device.type == "cuda"):
                logits = model.forward_from_shared(shared, chunk, ast_temperature=temperature)
                batch_probabilities.append(logits.softmax(-1)[..., 0])
        probabilities.append(torch.cat(batch_probabilities, dim=1).cpu())
        del shared, batch_probabilities
        targets.append(labels[:, class_indices])
    probabilities = torch.cat(probabilities).numpy()
    targets = torch.cat(targets).numpy()
    candidates = np.arange(0.50, 0.951, 0.01)
    scored = [(float(multilabel_metrics(targets, probabilities, t)[objective]), float(t))
              for t in candidates]
    score, threshold = max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))
    return threshold, score


def save_checkpoint(path, model, protocol, task_id, history, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if k.startswith("prompts.") or k == "prompt_tokens"
    }
    torch.save({"prompts": prompt_state, "protocol": protocol.name,
                "task_id": task_id, "history": history, "args": vars(args)}, path)
    path.with_suffix(".json").write_text(json.dumps(history, indent=2))
