#!/usr/bin/env python3
"""
train.py — train one YOLOv8n detector (child OR hazard) on its own dataset.

The two detectors are identical except for their data.yaml, so one
parametrized script covers both (kept simpler than duplicate
train_child.py / train_hazard.py; the model name only tags the output).

    # Phase 0 smoke test — 1 epoch on a tiny subset, just proves the
    # pipeline runs end-to-end locally before spending Colab GPU time:
    python scripts/train.py --model child  --data data/child/data.yaml  --smoke
    python scripts/train.py --model hazard --data data/hazard/data.yaml --smoke

    # Phase 1 baseline — default hyperparameters, full budget, on Colab T4:
    python scripts/train.py --model child  --data data/child/data.yaml  --epochs 100
    python scripts/train.py --model hazard --data data/hazard/data.yaml --epochs 100

Ultralytics writes full logs/weights to runs/detect/<name>/ (gitignored).
This script additionally appends the run's config + final mAP to
results/metrics/baseline_<model>.csv so the numbers survive the Colab
session (context.md: log configs + metrics, not just stdout).
"""

import argparse
import csv
import os
from datetime import datetime

RESULTS_DIR = os.path.join("results", "metrics")


def log_metrics(model_name, args, metrics):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Smoke runs are throwaway (1 epoch on a data fraction) — keep them out of
    # the baseline file so it only ever holds real, reportable runs.
    prefix = "smoke" if args.smoke else "baseline"
    path = os.path.join(RESULTS_DIR, f"{prefix}_{model_name}.csv")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": args.optimizer,
        "smoke": args.smoke,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
    }
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[{model_name}] logged -> {path}: mAP50={row['mAP50']} "
          f"mAP50-95={row['mAP50_95']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["child", "hazard"],
                    help="which detector (tags the output run + metrics file)")
    ap.add_argument("--data", required=True, help="path to that dataset's data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--optimizer", default="auto",
                    help="auto|SGD|Adam|AdamW (ablated in Phase 3)")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="pretrained weights to fine-tune from")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on a small data fraction — pipeline sanity check")
    args = ap.parse_args()

    if not os.path.isfile(args.data):
        raise SystemExit(f"data.yaml not found: {args.data} "
                         f"(run scripts/download_data.py first)")

    from ultralytics import YOLO  # imported here so --help works without the dep

    epochs = 1 if args.smoke else args.epochs
    fraction = 0.02 if args.smoke else 1.0  # ~tiny subset for the smoke test
    run_name = f"{args.model}_{'smoke' if args.smoke else 'baseline'}"

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        fraction=fraction,
        name=run_name,
        exist_ok=True,
    )
    metrics = model.val(data=args.data)  # eval on the dataset's val split
    log_metrics(args.model, args, metrics)


if __name__ == "__main__":
    main()
