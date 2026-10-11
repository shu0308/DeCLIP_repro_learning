import argparse
import random
from pathlib import Path

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from declip.data import VOCMultiLabel
from declip.engine import calibrate_threshold, evaluate, save_checkpoint, train_task
from declip.model import DeCLIP
from declip.old_labels import build_old_label_cache, diagnose_old_label_cache
from declip.protocols import voc_protocol


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce DeCLIP on PASCAL VOC 2007")
    parser.add_argument("--data-root", default="/root/autodl-tmp/VOCdevkittest/VOC2007")
    parser.add_argument("--protocol", choices=["b0-c4", "b4-c2"], default="b4-c2")
    parser.add_argument("--joint-upper-bound", action="store_true",
                        help="Train all classes jointly with full labels as a non-incremental upper bound")
    parser.add_argument("--all-images-per-task", action="store_true",
                        help="Use all training images at every stage while supervising only current classes")
    parser.add_argument("--oracle-old-labels", action="store_true",
                        help="Oracle ablation: supervise and update old prompts with ground-truth old labels")
    parser.add_argument("--reliable-old-labels", action="store_true",
                        help="Use only confidence-screened real old positives, with weak loss weight")
    parser.add_argument("--pseudo-old-labels", action="store_true",
                        help="Train with two-sided old-class pseudo-labels from the pre-task model")
    parser.add_argument("--pseudo-positive-threshold", type=float, default=0.98)
    parser.add_argument("--pseudo-negative-threshold", type=float, default=0.02)
    parser.add_argument("--pseudo-positive-fraction", type=float, default=0.10,
                        help="Maximum selected positive fraction per old class")
    parser.add_argument("--pseudo-negative-fraction", type=float, default=0.30,
                        help="Maximum selected negative fraction per old class")
    parser.add_argument("--pseudo-positive-weight", type=float, default=0.20)
    parser.add_argument("--pseudo-negative-weight", type=float, default=0.20)
    parser.add_argument("--diagnose-pseudo-labels", action="store_true",
                        help="Offline VOC-only diagnostic using hidden old labels; never used for selection")
    parser.add_argument("--old-confidence-threshold", type=float, default=0.80,
                        help="Minimum pre-update old-class probability for old-label supervision")
    parser.add_argument("--old-positive-loss-weight", type=float, default=0.20,
                        help="Weak loss weight for each retained old positive label")
    parser.add_argument("--old-logit-margin", type=float, default=0.0,
                        help="Minimum positive-minus-negative prompt logit margin for a retained old label")
    parser.add_argument("--pretrained", default="openai", help="OpenCLIP pretrained tag or local checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--prompt-length", type=int, default=4)
    parser.add_argument("--prompted-blocks", type=int, default=5)
    parser.add_argument("--pooling", choices=["attention", "mean", "max", "logsumexp"], default="attention")
    parser.add_argument("--pool-temperature", type=float, default=1.0,
                        help="Spatial attention temperature. 1.0 matches the unscaled softmax aggregation in DualCoOp")
    parser.add_argument("--exclude-cls", action="store_true")
    parser.add_argument("--semantic-init", action="store_true")
    parser.add_argument("--balanced-sampling", action="store_true",
                        help="Balance current-class positive exposure with a weighted sampler")
    parser.add_argument("--balance-min-ratio", type=float, default=0.0,
                        help="Apply balanced sampling only when max/min positive count reaches this ratio")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Adam weight decay for prompt regularization")
    parser.add_argument("--negative-loss-weight", type=float, default=1.0,
                        help="Loss multiplier for current-class negative labels")
    parser.add_argument("--adaptive-negative-weight", action="store_true",
                        help="Set each negative weight to 1 + current-class positive prevalence")
    parser.add_argument("--ast-strength", type=float, default=16.0)
    parser.add_argument("--decision-threshold", type=float, default=0.80,
                        help="Fixed positive-confidence threshold used by the referenced MLCIL protocol")
    parser.add_argument("--calibration-ratio", type=float, default=0.0,
                        help="Fraction of each stage trainval set held out for threshold calibration")
    parser.add_argument("--calibration-objective", choices=["CF1", "OF1"], default="OF1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="outputs/voc_b4_c2_seed1")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Debug-only subset")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if sum((args.joint_upper_bound, args.oracle_old_labels,
            args.reliable_old_labels, args.pseudo_old_labels)) > 1:
        raise ValueError("joint, oracle, legacy reliable, and pseudo-label modes are exclusive")
    if args.diagnose_pseudo_labels and not args.pseudo_old_labels:
        raise ValueError("--diagnose-pseudo-labels requires --pseudo-old-labels")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_checkpoint = Path(args.pretrained).is_file()
    model_clip, train_tf, val_tf = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=args.pretrained, device=device,
        force_quick_gelu=local_checkpoint, load_weights_only=True,
    )
    protocol = voc_protocol(args.protocol)
    model = DeCLIP(model_clip, open_clip.get_tokenizer("ViT-B-16"), protocol.classes,
                   args.prompt_length, args.prompted_blocks, pooling=args.pooling,
                   pool_temperature=args.pool_temperature, exclude_cls=args.exclude_cls,
                   semantic_init=args.semantic_init).to(device)
    history = []
    tasks = [list(range(len(protocol.classes)))] if args.joint_upper_bound else protocol.tasks
    for task_id, current in enumerate(tasks):
        # The standard MLCIL setting updates only prompts for newly introduced
        # labels.  The oracle switch keeps the same current-task image stream,
        # but exposes all already-seen ground-truth labels and updates their
        # prompts as an upper-bound measurement for reliable old-label signals.
        train_indices = (list(range(len(protocol.classes))) if args.joint_upper_bound
                         else (protocol.seen(task_id)
                               if (args.oracle_old_labels or args.reliable_old_labels
                                   or args.pseudo_old_labels) else current))
        train_set = VOCMultiLabel(args.data_root, "trainval", train_tf, current,
                                  require_current_positive=not (args.all_images_per_task or args.joint_upper_bound))
        pseudo_set = (VOCMultiLabel(args.data_root, "trainval", val_tf, current,
                                   require_current_positive=not args.all_images_per_task)
                      if args.pseudo_old_labels and task_id > 0 else None)
        calibration_set = None
        if args.calibration_ratio:
            if not 0.0 < args.calibration_ratio < 0.5:
                raise ValueError("--calibration-ratio must be in (0, 0.5)")
            calibration_base = VOCMultiLabel(
                args.data_root, "trainval", val_tf, current,
                require_current_positive=not (args.all_images_per_task or args.joint_upper_bound))
            split_generator = torch.Generator().manual_seed(args.seed + 1000 + task_id)
            order = torch.randperm(len(train_set), generator=split_generator).tolist()
            calibration_size = max(1, round(len(order) * args.calibration_ratio))
            calibration_indices, training_indices = order[:calibration_size], order[calibration_size:]
            train_set = torch.utils.data.Subset(train_set, training_indices)
            if pseudo_set is not None:
                pseudo_set = torch.utils.data.Subset(pseudo_set, training_indices)
            calibration_set = torch.utils.data.Subset(calibration_base, calibration_indices)
        if args.max_train_samples:
            chosen = range(min(args.max_train_samples, len(train_set)))
            train_set = torch.utils.data.Subset(train_set, chosen)
            if pseudo_set is not None:
                pseudo_set = torch.utils.data.Subset(pseudo_set, chosen)
        generator = torch.Generator().manual_seed(args.seed + task_id)
        sampler = None
        if args.balanced_sampling:
            labels = train_set.dataset.labels[train_set.indices][:, current] if isinstance(
                train_set, torch.utils.data.Subset) else train_set.labels[:, current]
            positive_counts = labels.sum(0).clamp_min(1)
            imbalance_ratio = (positive_counts.max() / positive_counts.min()).item()
            if imbalance_ratio >= args.balance_min_ratio:
                # Each class contributes equal total sampling mass. Averaging over
                # labels prevents multi-positive images from being over-counted.
                sample_weights = (labels / positive_counts).sum(1) / labels.sum(1).clamp_min(1)
                sampler = WeightedRandomSampler(sample_weights.double(), len(train_set),
                                                replacement=True, generator=generator)
            print(f"task={task_id + 1} positive_counts={positive_counts.tolist()} "
                  f"imbalance={imbalance_ratio:.2f} balanced={sampler is not None}")
        train_loader = DataLoader(train_set, args.batch_size, shuffle=sampler is None, sampler=sampler,
                                  generator=generator if sampler is None else None,
                                  num_workers=args.workers, pin_memory=True, drop_last=False)
        old_indices = [] if task_id == 0 or args.joint_upper_bound else protocol.seen(task_id - 1)
        old_label_cache, pseudo_summary = None, None
        if pseudo_set is not None:
            old_label_cache, pseudo_summary = build_old_label_cache(
                model, pseudo_set, old_indices, device, args.batch_size,
                workers=args.workers, amp=not args.no_amp,
                positive_threshold=args.pseudo_positive_threshold,
                negative_threshold=args.pseudo_negative_threshold,
                positive_fraction=args.pseudo_positive_fraction,
                negative_fraction=args.pseudo_negative_fraction,
                positive_weight=args.pseudo_positive_weight,
                negative_weight=args.pseudo_negative_weight)
            if args.diagnose_pseudo_labels:
                pseudo_summary.update(diagnose_old_label_cache(
                    old_label_cache, pseudo_set, old_indices))
            active_positions = [i for i, count in enumerate(pseudo_summary["positive_count"])
                                if count > 0 and pseudo_summary["negative_count"][i] > 0]
            pseudo_summary["active_old_classes"] = [protocol.classes[old_indices[i]]
                                                    for i in active_positions]
            if active_positions:
                old_label_cache = {image_id: (targets[active_positions], weights[active_positions])
                                   for image_id, (targets, weights) in old_label_cache.items()}
                old_indices = [old_indices[i] for i in active_positions]
                train_indices = old_indices + current
            else:
                old_label_cache = None
                old_indices = []
                train_indices = current
            print(f"task={task_id + 1} pseudo_old={pseudo_summary}")
        optimizer = torch.optim.Adam(model.prompt_parameters(train_indices), lr=args.lr,
                                     betas=(0.9, 0.999), weight_decay=args.weight_decay)
        negative_weight = args.negative_loss_weight
        if args.adaptive_negative_weight:
            labels = train_set.dataset.labels[train_set.indices][:, current] if isinstance(
                train_set, torch.utils.data.Subset) else train_set.labels[:, current]
            current_weights = (1.0 + labels.sum(0) / len(labels)).tolist()
            negative_weight = ([1.0] * len(old_indices) + current_weights
                               if len(train_indices) != len(current) else current_weights)
            print(f"task={task_id + 1} adaptive_negative_weights={negative_weight}")
        for train_log in train_task(
                model, train_loader, train_indices, optimizer, device, args.epochs,
                not args.no_amp, negative_weight, old_class_indices=old_indices,
                reliable_old_labels=args.reliable_old_labels,
                old_confidence_threshold=args.old_confidence_threshold,
                old_positive_loss_weight=args.old_positive_loss_weight,
                old_logit_margin=args.old_logit_margin,
                old_label_cache=old_label_cache):
            suffix = (f" reliable_old_ratio={train_log['reliable_old_ratio']:.3f}"
                      if 'reliable_old_ratio' in train_log else '')
            print(f"task={task_id + 1}/{len(tasks)} epoch={train_log['epoch']} "
                  f"loss={train_log['loss']:.5f}{suffix}")
        seen = current if args.joint_upper_bound else protocol.seen(task_id)
        evaluation_task = 1 if args.joint_upper_bound else task_id + 1
        test_set = VOCMultiLabel(args.data_root, "test", val_tf, seen,
                                 require_current_positive=True)
        test_loader = DataLoader(test_set, args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
        decision_threshold = args.decision_threshold
        calibration_score = None
        if calibration_set is not None:
            calibration_loader = DataLoader(calibration_set, args.batch_size, shuffle=False,
                                            num_workers=args.workers, pin_memory=True)
            decision_threshold, calibration_score = calibrate_threshold(
                model, calibration_loader, current, evaluation_task, len(seen), device,
                args.ast_strength, not args.no_amp, args.calibration_objective)
            print(f"task={task_id + 1} calibrated_threshold={decision_threshold:.2f} "
                  f"calibration_{args.calibration_objective}={calibration_score:.4f}")
        metrics = evaluate(model, test_loader, seen, evaluation_task, device,
                           args.ast_strength, not args.no_amp, decision_threshold)
        if calibration_score is not None:
            metrics["calibration_objective"] = args.calibration_objective
            metrics["calibration_score"] = calibration_score
        metrics["oracle_old_labels"] = args.oracle_old_labels
        metrics["reliable_old_labels"] = args.reliable_old_labels
        metrics["pseudo_old_labels"] = args.pseudo_old_labels
        if pseudo_summary is not None:
            metrics["pseudo_summary"] = pseudo_summary
        metrics["class_AP"] = {protocol.classes[i]: ap for i, ap in zip(seen, metrics.pop("AP"))}
        history.append({"task": task_id + 1, "classes": [protocol.classes[i] for i in current], **metrics})
        print(history[-1])
        save_checkpoint(Path(args.output) / f"task_{task_id + 1}.pt", model, protocol, task_id, history, args)
    averages = {key: float(np.mean([row[key] for row in history])) for key in ("mAP", "CF1", "OF1")}
    print({"Average": averages, "Last": history[-1]})


if __name__ == "__main__":
    main()

