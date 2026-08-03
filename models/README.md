# Final model weights

The two deliverable detectors. They are trained fully independently and are
fused only at the output level (see `context.md`).

| File | Val mAP50 | Val mAP50-95 | imgsz | Source run |
|---|---|---|---|---|
| `child_best.pt` | 0.9554 | 0.8353 | **416** | `runs/detect/final_child` (Phase 4a) |
| `hazard_best.pt` | 0.5657 | 0.4072 | **640** | `runs/detect/hazard_baseline` (Phase 1) |

The two models use **different input resolutions**. This is deliberate and
legitimate — they are separate models — and follows from the Phase 3a
ablation (§5 of `results_and_findings.md`). Any inference code must resize
per model, not once for both.

## Why the hazard model comes from Phase 1, not Phase 4a

Phase 4a trained the hazard detector at the configuration selected by
Phases 2–3 and it scored **0.5256 mAP50 — 0.040 below the untuned
baseline**, losing on 10 of 12 classes (§7.1). The 20-epoch tuning proxy did
not transfer to 100-epoch training. The baseline configuration is therefore
the better model and is shipped as the deliverable; the tuned weights are
kept at `runs/detect/final_hazard` for the record but are not the product.

## Reproducibility caveat on `hazard_best.pt`

Two things about this run are worth stating plainly in the write-up:

1. **It was trained with `optimizer=auto`.** Its `args.yaml` records
   `lr0: 0.01`, but that value was **discarded at runtime**: ultralytics'
   `auto` mode hard-codes AdamW with `lr = 0.002 x 5/(4 + nc)`, i.e.
   **6.25e-4** for these 12 classes. Anyone reproducing this run from
   `args.yaml` alone would use the wrong learning rate.
2. **Its ultralytics version is unrecorded.** The Phase 1 baselines predate
   the decision to pin `ultralytics==8.4.106`, so the exact version is
   unknown. Everything from Phase 2 onward is pinned.

To remove both caveats, retrain this configuration explicitly — AdamW,
`lr0=6.25e-4`, default loss weights (7.5 / 0.5 / 1.5), `imgsz=640`, 100
epochs — under the pinned version (~2.4 h on a T4). That would make the
headline model exactly reproducible. Until then, quote the numbers with the
caveat rather than presenting the run as reproducible.

## Regenerating

```bash
# child (reproducible as-is)
python scripts/train_final.py --model child --data data/child/data.yaml --epochs 100

# hazard: the shipped weights came from Phase 1; the explicit equivalent is
python scripts/train.py --model hazard --data data/hazard/data.yaml --epochs 100
```
