#!/usr/bin/env python3
"""
ablation_optimizer.py — Phase 3b: optimizer ablation for one detector.

Compares SGD / Adam / AdamW at imgsz=640 (the Phase 3a winner), with the
loss weights (box/cls/dfl) held fixed.

    python scripts/ablation_optimizer.py --model hazard --data data/hazard/data.yaml
    python scripts/ablation_optimizer.py --model child  --data data/child/data.yaml

WHY EACH OPTIMIZER GETS ITS OWN lr0 (this is deliberate, not a confound):
    Holding a single lr0 across optimizers is NOT the fair comparison it
    looks like — it just measures which optimizer happens to suit that one
    learning rate. Our own Phase 2 sweep shows the effect is enormous:
    AdamW at lr0=0.01 reaches only ~0.24 mAP50, while the same optimizer at
    lr0=8.8e-4 reaches 0.50. Ultralytics agrees — its `optimizer='auto'`
    picks (SGD, 0.01) or (AdamW, 0.002*5/(4+nc)) depending on the optimizer.
    So the unit of comparison here is "optimizer + the learning rate
    appropriate to it", which must be stated in the write-up.

    Defaults used (override with --lr-sgd / --lr-adam):
      SGD          -> 0.01      (ultralytics' standard SGD rate)
      Adam / AdamW -> hazard: 8.8e-4, the Phase 2 tuned value
                      child : 0.002, ultralytics' auto-derived rate for nc=1
                      (Adam and AdamW share the same learning-rate scale;
                       they differ in how weight decay is applied)

RESUMABLE: each finished optimizer is appended to
results/metrics/ablation_optimizer_<model>.csv and skipped on re-run.
Pass --project to write runs to mounted Drive so they survive a disconnect.
"""

import argparse
import csv
import os
import time

RESULTS_DIR = os.path.join("results", "metrics")
TUNED_YAML = os.path.join(RESULTS_DIR, "tuning_{model}_best.yaml")
FIELDS = ["model", "optimizer", "lr0", "imgsz", "epochs", "box", "cls", "dfl",
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


def adaptive_lr(model, tuned):
    """Learning rate for Adam/AdamW: the tuned value if we have one,
    else ultralytics' own auto formula for this dataset's class count."""
    if "lr0" in tuned:
        return tuned["lr0"]
    nc = 1 if model == "child" else 12
    return round(0.002 * 5 / (4 + nc), 6)


def done_optimizers(path):
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {r["optimizer"] for r in csv.DictReader(fh) if r.get("optimizer")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["child", "hazard"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--optimizers", nargs="+", default=["SGD", "Adam", "AdamW"])
    ap.add_argument("--imgsz", type=int, default=640, help="Phase 3a winner")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-sgd", type=float, default=0.01)
    ap.add_argument("--lr-adam", type=float, default=None,
                    help="Adam/AdamW rate; defaults to the tuned value, "
                         "or ultralytics' auto rate if the model was never tuned")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--project", default=None,
                    help="output root; use mounted Drive on Colab")
    args = ap.parse_args()

    if "auto" in args.optimizers:
        raise SystemExit("'auto' is not a comparable optimizer — it overrides "
                         "lr0 and picks the optimizer itself. Name them explicitly.")
    if not os.path.isfile(args.data):
        raise SystemExit(f"data.yaml not found: {args.data}")

    from ultralytics import YOLO

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, f"ablation_optimizer_{args.model}.csv")
    already = done_optimizers(out_csv)
    tuned = load_tuned(args.model)
    # loss weights held fixed across all optimizers; only optimizer+lr0 vary
    weights = {k: tuned[k] for k in ("box", "cls", "dfl") if k in tuned}
    lr_adam = args.lr_adam if args.lr_adam is not None else adaptive_lr(args.model, tuned)

    print(f"[{args.model}] imgsz={args.imgsz} epochs={args.epochs}")
    print(f"[{args.model}] loss weights held fixed: "
          f"{weights if weights else 'ultralytics defaults (never tuned)'}")
    print(f"[{args.model}] lr0 per optimizer: SGD={args.lr_sgd}, "
          f"Adam/AdamW={lr_adam}")
    if already:
        print(f"[{args.model}] skipping already-done: {sorted(already)}")

    for opt in args.optimizers:
        if opt in already:
            continue
        lr0 = args.lr_sgd if opt == "SGD" else lr_adam
        print(f"\n[{args.model}] === {opt} (lr0={lr0}) ===")
        model = YOLO(args.weights)
        t0 = time.time()
        model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs,
                    batch=args.batch, optimizer=opt, lr0=lr0,
                    name=f"ablation_opt_{args.model}_{opt}",
                    project=args.project, plots=False, exist_ok=True, **weights)
        train_min = (time.time() - t0) / 60
        m = model.val(data=args.data, imgsz=args.imgsz)
        infer_ms = float(m.speed.get("inference", 0)) if hasattr(m, "speed") else 0

        row = {
            "model": args.model, "optimizer": opt, "lr0": lr0,
            "imgsz": args.imgsz, "epochs": args.epochs,
            "box": weights.get("box", "default"),
            "cls": weights.get("cls", "default"),
            "dfl": weights.get("dfl", "default"),
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
        print(f"[{args.model}] {opt}: mAP50={row['mAP50']} "
              f"mAP50-95={row['mAP50_95']} -> {out_csv}")

    print(f"\n[{args.model}] done. Results: {out_csv}")


if __name__ == "__main__":
    main()
