#!/usr/bin/env python3
"""
ablation_imgsz.py — Phase 3a: input resolution ablation for one detector.

Trains the same model at several `imgsz` values with EVERYTHING else held
fixed, so the only variable is input resolution (context.md: one ablation,
one variable).

    python scripts/ablation_imgsz.py --model hazard --data data/hazard/data.yaml
    python scripts/ablation_imgsz.py --model child  --data data/child/data.yaml

HYPERPARAMETERS: for hazard, the Phase 2 winner is loaded from
results/metrics/tuning_hazard_best.yaml and held fixed across all
resolutions. The child detector was never tuned, so it runs ultralytics
defaults — an asymmetry to state in the write-up rather than hide. Either
way, within a single model every run uses identical hyperparameters, so the
resolution comparison itself stays clean.

OPTIMIZER: explicit AdamW, never 'auto' — auto ignores lr0 (see tune.py),
which would silently discard the tuned learning rate.

EPOCHS: 30 by default (~2h per model on a T4). This is a comparison budget,
not convergence — the baseline plateaued near epoch 50. Note in the write-up
that higher resolutions can need more epochs to pay off, so a short budget
may understate 832.

RESUMABLE: each finished resolution is appended to
results/metrics/ablation_imgsz_<model>.csv, and sizes already present are
skipped. If Colab disconnects, re-run the same command to finish the rest.
Pass --project to write runs to mounted Drive so they survive a disconnect.
"""

import argparse
import csv
import os
import time

RESULTS_DIR = os.path.join("results", "metrics")
TUNED_YAML = os.path.join(RESULTS_DIR, "tuning_{model}_best.yaml")
FIELDS = ["model", "imgsz", "epochs", "optimizer", "lr0", "box", "cls", "dfl",
          "precision", "recall", "mAP50", "mAP50_95", "train_min", "infer_ms"]


def load_tuned(model):
    """Phase 2 winner for this model, or {} if it was never tuned."""
    path = TUNED_YAML.format(model=model)
    if not os.path.isfile(path):
        return {}
    import yaml
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return {k: float(cfg[k]) for k in ("lr0", "box", "cls", "dfl") if k in cfg}


def done_sizes(path):
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {int(r["imgsz"]) for r in csv.DictReader(fh) if r.get("imgsz")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["child", "hazard"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[416, 640, 832])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--project", default=None,
                    help="output root; use mounted Drive on Colab")
    args = ap.parse_args()

    if args.optimizer == "auto":
        raise SystemExit("optimizer='auto' ignores lr0 — pass AdamW or SGD.")
    if not os.path.isfile(args.data):
        raise SystemExit(f"data.yaml not found: {args.data}")

    from ultralytics import YOLO

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, f"ablation_imgsz_{args.model}.csv")
    already = done_sizes(out_csv)
    tuned = load_tuned(args.model)
    print(f"[{args.model}] hyperparameters held fixed: "
          f"{tuned if tuned else 'ultralytics defaults (never tuned)'}")
    if already:
        print(f"[{args.model}] skipping already-done sizes: {sorted(already)}")

    for size in args.sizes:
        if size in already:
            continue
        print(f"\n[{args.model}] === imgsz={size} ===")
        model = YOLO(args.weights)
        t0 = time.time()
        model.train(data=args.data, imgsz=size, epochs=args.epochs,
                    batch=args.batch, optimizer=args.optimizer,
                    name=f"ablation_imgsz_{args.model}_{size}",
                    project=args.project, plots=False, exist_ok=True, **tuned)
        train_min = (time.time() - t0) / 60
        m = model.val(data=args.data, imgsz=size)
        # inference ms/image — resolution's real cost, needed for the
        # computational-cost comparison the proposal promises
        infer_ms = float(m.speed.get("inference", 0)) if hasattr(m, "speed") else 0

        row = {
            "model": args.model, "imgsz": size, "epochs": args.epochs,
            "optimizer": args.optimizer,
            "lr0": tuned.get("lr0", "default"), "box": tuned.get("box", "default"),
            "cls": tuned.get("cls", "default"), "dfl": tuned.get("dfl", "default"),
            "precision": round(float(m.box.mp), 5),
            "recall": round(float(m.box.mr), 5),
            "mAP50": round(float(m.box.map50), 5),
            "mAP50_95": round(float(m.box.map), 5),
            "train_min": round(train_min, 2),
            "infer_ms": round(infer_ms, 3),
        }
        write_header = not os.path.isfile(out_csv)
        with open(out_csv, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"[{args.model}] imgsz={size}: mAP50={row['mAP50']} "
              f"mAP50-95={row['mAP50_95']} -> {out_csv}")

    print(f"\n[{args.model}] done. Results: {out_csv}")


if __name__ == "__main__":
    main()
