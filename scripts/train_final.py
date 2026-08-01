#!/usr/bin/env python3
"""
train_final.py - Phase 4a: train the final detectors at the locked config.

    python scripts/train_final.py --model hazard --data data/hazard/data.yaml --project "$DRIVE"
    python scripts/train_final.py --model child  --data data/child/data.yaml  --project "$DRIVE"

The configuration is BAKED IN rather than passed on the command line, so the
final run cannot silently disagree with what Phases 2-3 decided. Each value
records the phase that chose it:

    child   imgsz=416  lr0=0.002   default loss weights
            imgsz  <- Phase 3a re-run (416 beat 640 once lr0 was corrected)
            lr0    <- ultralytics' auto rate for nc=1; child was never tuned
    hazard  imgsz=640  lr0=8.8e-4  box/cls/dfl from the Phase 2 sweep
            imgsz  <- Phase 3a
            rest   <- Phase 2 winner (results/metrics/tuning_hazard_best.yaml)

    both    optimizer=AdamW  <- Phase 3b
            epochs=100       <- matches the Phase 1 baselines

WHY 100 EPOCHS MATTERS HERE: this is the first FAIR comparison against the
baselines. The Phase 2 sweep ran 20-epoch schedules, and ultralytics anneals
the learning rate across the *scheduled* epoch count, so a 20-epoch run is
fully annealed at epoch 20 while a 100-epoch run is only a fifth of the way
through its decay. Comparing them at epoch 20 favours the short run on
schedule alone. Training the tuned config for the same 100 epochs is the
only way to tell whether the tuning actually helped.

Outputs, per model:
    results/metrics/final_<model>.csv            overall metrics + timings
    results/metrics/per_class_<model>_final.csv  per-class (the 12 hazards)
    <project>/final_<model>/weights/best.pt      the deliverable weights

RESUMABLE: pass --project pointing at mounted Drive. A run that already
reached the full epoch count is re-validated in minutes instead of retrained;
an interrupted one continues from its checkpoint.
"""

import argparse
import csv
import os
import time

RESULTS_DIR = os.path.join("results", "metrics")
TUNED_YAML = os.path.join(RESULTS_DIR, "tuning_{model}_best.yaml")

# Phase 1 baselines, for the fair-comparison print at the end.
BASELINE = {"child": (0.9469, 0.8271), "hazard": (0.5657, 0.4072)}

FINAL = {
    "child":  {"imgsz": 416, "lr0": 0.002},   # 3a re-run; auto rate for nc=1
    "hazard": {"imgsz": 640},                 # 3a; lr0 + weights from Phase 2
}
FIELDS = ["model", "imgsz", "epochs", "optimizer", "lr0", "box", "cls", "dfl",
          "precision", "recall", "mAP50", "mAP50_95", "train_min", "infer_ms",
          "baseline_mAP50", "baseline_mAP50_95", "delta_mAP50", "delta_mAP50_95"]


def load_tuned(model):
    path = TUNED_YAML.format(model=model)
    if not os.path.isfile(path):
        return {}
    import yaml
    cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
    return {k: float(cfg[k]) for k in ("lr0", "box", "cls", "dfl") if k in cfg}


def run_dir(project, model):
    return os.path.join(project or os.path.join("runs", "detect"), f"final_{model}")


def epochs_completed(project, model):
    """Epochs a previous run actually finished. Read from results.csv (one row
    per epoch) - the run directory and best.pt both exist mid-run, so neither
    proves completion."""
    p = os.path.join(run_dir(project, model), "results.csv")
    if not os.path.isfile(p):
        return 0
    with open(p, encoding="utf-8") as fh:
        return sum(1 for r in csv.DictReader(fh) if any(v.strip() for v in r.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["child", "hazard"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--project", default=None,
                    help="output root; use mounted Drive on Colab so a "
                         "disconnect does not cost the whole run")
    args = ap.parse_args()

    if not os.path.isfile(args.data):
        raise SystemExit(f"data.yaml not found: {args.data}")

    from ultralytics import YOLO

    cfg = dict(FINAL[args.model])
    cfg.update(load_tuned(args.model))        # Phase 2 winner, hazard only
    imgsz = cfg.pop("imgsz")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"[{args.model}] FINAL CONFIG  imgsz={imgsz} optimizer=AdamW "
          f"epochs={args.epochs}")
    print(f"[{args.model}]   {cfg if cfg else 'ultralytics defaults'}")
    print(f"[{args.model}] baseline to beat: mAP50={BASELINE[args.model][0]} "
          f"mAP50-95={BASELINE[args.model][1]} (same 100 epochs)")

    rd = run_dir(args.project, args.model)
    done = epochs_completed(args.project, args.model)
    best = os.path.join(rd, "weights", "best.pt")
    last = os.path.join(rd, "weights", "last.pt")
    print(f"[{args.model}] run dir: {rd}")

    train_min = ""
    if done >= args.epochs and os.path.isfile(best):
        print(f"[{args.model}] PLAN: complete run found ({done}/{args.epochs}) "
              f"- validating only, not retraining")
        model = YOLO(best)
    elif done and os.path.isfile(last):
        print(f"[{args.model}] PLAN: resuming from epoch {done}/{args.epochs}")
        model = YOLO(last)
        t0 = time.time()
        # Pass the checkpoint PATH, not resume=True: resume=True falls back to
        # get_latest_run(), which picks the newest run anywhere on disk and
        # could continue the wrong one.
        model.train(resume=last, data=args.data, imgsz=imgsz)
        train_min = round((time.time() - t0) / 60, 2)   # resumed portion only
    else:
        if done:
            print(f"[{args.model}] PLAN: partial run ({done}/{args.epochs}) but "
                  f"no last.pt - retraining from scratch")
        else:
            print(f"[{args.model}] PLAN: training from scratch ({args.epochs} epochs)")
        model = YOLO(args.weights)
        t0 = time.time()
        model.train(data=args.data, imgsz=imgsz, epochs=args.epochs,
                    batch=args.batch, optimizer="AdamW",
                    name=f"final_{args.model}", project=args.project,
                    plots=True, exist_ok=True, **cfg)
        train_min = round((time.time() - t0) / 60, 2)

    m = model.val(data=args.data, imgsz=imgsz)
    infer_ms = float(m.speed.get("inference", 0)) if hasattr(m, "speed") else 0
    b50, b95 = BASELINE[args.model]

    row = {
        "model": args.model, "imgsz": imgsz, "epochs": args.epochs,
        "optimizer": "AdamW",
        "lr0": cfg.get("lr0", "default"), "box": cfg.get("box", "default"),
        "cls": cfg.get("cls", "default"), "dfl": cfg.get("dfl", "default"),
        "precision": round(float(m.box.mp), 5),
        "recall": round(float(m.box.mr), 5),
        "mAP50": round(float(m.box.map50), 5),
        "mAP50_95": round(float(m.box.map), 5),
        "train_min": train_min, "infer_ms": round(infer_ms, 3),
        "baseline_mAP50": b50, "baseline_mAP50_95": b95,
        "delta_mAP50": round(float(m.box.map50) - b50, 5),
        "delta_mAP50_95": round(float(m.box.map) - b95, 5),
    }
    out = os.path.join(RESULTS_DIR, f"final_{args.model}.csv")
    hdr = not os.path.isfile(out)
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if hdr:
            w.writeheader()
        w.writerow(row)

    # Per-class table - a stated deliverable for the 12 hazard classes, and
    # regenerated here under the pinned ultralytics so it matches these weights.
    names = getattr(m, "names", None) or {}
    if names:
        pc = os.path.join(RESULTS_DIR, f"per_class_{args.model}_final.csv")
        with open(pc, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["class", "precision", "recall", "mAP50", "mAP50_95"])
            w.writerow(["all", row["precision"], row["recall"],
                        row["mAP50"], row["mAP50_95"]])
            for i, c in enumerate(getattr(m.box, "ap_class_index", [])):
                p_, r_, ap50_, ap_ = m.box.class_result(i)
                w.writerow([names.get(int(c), int(c)), round(float(p_), 5),
                            round(float(r_), 5), round(float(ap50_), 5),
                            round(float(ap_), 5)])
        print(f"[{args.model}] per-class -> {pc}")

    print(f"\n[{args.model}] FINAL   mAP50={row['mAP50']}  mAP50-95={row['mAP50_95']}")
    print(f"[{args.model}] BASELINE mAP50={b50}  mAP50-95={b95}")
    print(f"[{args.model}] DELTA   {row['delta_mAP50']:+.5f} / "
          f"{row['delta_mAP50_95']:+.5f}   <- did the tuning help?")
    print(f"[{args.model}] weights: {best}")
    print(f"[{args.model}] metrics -> {out}")


if __name__ == "__main__":
    main()
