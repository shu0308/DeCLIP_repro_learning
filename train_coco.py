"""Train DeCLIP on an extracted COCO 2017 directory."""
import argparse
import random
from pathlib import Path
import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader
from declip.data import COCOMultiLabel
from declip.engine import evaluate, save_checkpoint, train_task
from declip.model import DeCLIP
from declip.protocols import coco_protocol


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce DeCLIP on MS-COCO 2017")
    parser.add_argument("--data-root", required=True,
                        help="Extracted COCO root with trainYEAR/, valYEAR/, annotations/")
    parser.add_argument("--year", choices=["2014", "2017"], default="2014")
    parser.add_argument("--protocol", choices=["b0-c10", "b40-c10"], default="b40-c10")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--class-chunk-size", type=int, default=10,
                        help="Classes per forward pass; zero disables chunking")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--prompt-length", type=int, default=4)
    parser.add_argument("--prompted-blocks", type=int, default=5)
    parser.add_argument("--ast-strength", type=float, default=16.0)
    parser.add_argument("--decision-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="outputs/coco_b40_c10_seed1")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_checkpoint = Path(args.pretrained).is_file()
    model_clip, train_tf, val_tf = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=args.pretrained, device=device,
        force_quick_gelu=local_checkpoint, load_weights_only=True)
    protocol = coco_protocol(args.protocol)
    model = DeCLIP(model_clip, open_clip.get_tokenizer("ViT-B-16"), protocol.classes,
                   args.prompt_length, args.prompted_blocks).to(device)
    history = []
    for task_id, current in enumerate(protocol.tasks):
        train_set = COCOMultiLabel(args.data_root, f"train{args.year}", train_tf, current,
                                   require_current_positive=True)
        if args.max_train_samples:
            train_set = torch.utils.data.Subset(train_set, range(min(args.max_train_samples, len(train_set))))
        generator = torch.Generator().manual_seed(args.seed + task_id)
        train_loader = DataLoader(train_set, args.batch_size, shuffle=True, generator=generator,
                                  num_workers=args.workers, pin_memory=True, drop_last=False)
        optimizer = torch.optim.Adam(model.prompt_parameters(current), lr=args.lr,
                                     betas=(0.9, 0.999), weight_decay=0.0)
        for train_log in train_task(model, train_loader, current, optimizer, device,
                                    args.epochs, not args.no_amp, 1.0, args.class_chunk_size):
            print(f"task={task_id + 1}/{len(protocol.tasks)} epoch={train_log['epoch']} loss={train_log['loss']:.5f}")
        seen = protocol.seen(task_id)
        test_set = COCOMultiLabel(args.data_root, f"val{args.year}", val_tf, seen,
                                  require_current_positive=True)
        test_loader = DataLoader(test_set, args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
        metrics = evaluate(model, test_loader, seen, task_id + 1, device,
                           args.ast_strength, not args.no_amp, args.decision_threshold,
                           args.class_chunk_size)
        metrics["class_AP"] = {protocol.classes[index]: ap
                               for index, ap in zip(seen, metrics.pop("AP"))}
        history.append({"task": task_id + 1,
                        "classes": [protocol.classes[index] for index in current], **metrics})
        print(history[-1])
        save_checkpoint(Path(args.output) / f"task_{task_id + 1}.pt",
                        model, protocol, task_id, history, args)
    averages = {key: float(np.mean([row[key] for row in history]))
                for key in ("mAP", "CF1", "OF1")}
    print({"Average": averages, "Last": history[-1]})


if __name__ == "__main__":
    main()
