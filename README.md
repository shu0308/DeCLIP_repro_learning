# DeCLIP reproduction and old-label experiments

`train_voc.py` reproduces the VOC incremental protocol. The new
`--pseudo-old-labels` experiment generates old-class supervision from the
model's predictions **before each incremental task**. It does not use hidden
old-class ground truth for training.

For each current-task image, the pre-task model scores the original and a
horizontally flipped deterministic view. An old class is selected as positive
only if both views exceed `--pseudo-positive-threshold`; it is selected as
negative only if both fall below `--pseudo-negative-threshold`. Per-class
coverage caps limit the number selected. If either side has no selected
examples for a class, that class provides no pseudo-label loss for the task.
The total negative weight per class cannot exceed its positive weight.

On the AutoDL instance, run a small diagnostic first:

```bash
cd /root/autodl-tmp/DeCLIP_repro_github
.venv/bin/python -u train_voc.py --protocol b0-c4 --pseudo-old-labels \
  --diagnose-pseudo-labels --epochs 1 --batch-size 4 --workers 0 \
  --max-train-samples 16 --pretrained checkpoints/ViT-B-16.pt \
  --seed 0 --output outputs/voc_b0c4_pseudo_smoke
```

For a full comparison, run the original and pseudo-label modes with exactly
the same seed, batch size, epochs, training images, and decision threshold.
The `--diagnose-pseudo-labels` switch reports selection precision against VOC's
hidden old labels for research analysis only; it does not affect selection or
training. The `pseudo_summary` in each task result records positive and
negative selection counts and coverage. If coverage or diagnostic precision is
poor, inspect those numbers before running a full five-task experiment.

This is an exploratory two-sided selection method, not a calibrated
probability model. Its fixed default thresholds and coverage caps are starting
settings rather than validated optimal values.
