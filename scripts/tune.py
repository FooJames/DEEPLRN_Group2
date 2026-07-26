#!/usr/bin/env python3
"""
tune.py — Phase 2 hyperparameter tuning for one detector.

Tunes ONLY the four parameters the project committed to (context.md):
lr0, box, cls, dfl. Ultralytics' default search space also evolves ~16
augmentation/optimizer params; that is deliberately overridden here so the
sweep answers the question we actually asked.

    python scripts/tune.py --model hazard --data data/hazard/data.yaml \
        --iterations 6 --epochs 20

WHY AN EXPLICIT OPTIMIZER (do not pass --optimizer auto):
    With optimizer='auto', ultralytics logs "ignoring 'lr0=...'" and hard-codes
    AdamW with lr = 0.002*5/(4+nc). Any lr0 the tuner proposes is silently
    discarded, so a quarter of the search would be wasted and the reported
    "best lr0" would be meaningless. The baseline ran auto (=AdamW, lr=6.25e-4
    for nc=12), so AdamW is the default here to stay comparable.

RESUMABILITY (two things are required, both easy to get wrong):
  1. `resume=True` — the Tuner does `exist_ok = resume`, so passing exist_ok
     alone is silently overwritten and each run would start a FRESH
     tune_<model>2/3/... dir instead of continuing. This script always passes
     resume=True, so the dir is stable and the tuner seeds new mutations from
     whatever tune_results.csv already exists.
  2. `--project` on persistent storage — on Colab, runs/ lives in the
     ephemeral session and is wiped on disconnect, which would delete the
     very CSV that makes resume work. Point --project at a mounted Drive
     path so the sweep survives across sessions.
With both, re-running the same command continues the sweep and only the
in-flight iteration is lost.

Every iteration's hyperparameters + fitness are copied to
results/metrics/tuning_<model>.csv (the whole sweep, not just the winner).
"""

import argparse
import os
import shutil

RESULTS_DIR = os.path.join("results", "metrics")

# Narrower than the ultralytics default lr0 range (1e-5, 1e-1): 1e-1 diverges
# under AdamW and would burn iterations. This brackets the auto-chosen 6.25e-4.
SPACE = {
    "lr0": (1e-5, 1e-2),
    "box": (1.0, 20.0),
    "cls": (0.2, 4.0),
    "dfl": (0.4, 6.0),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["child", "hazard"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--iterations", type=int, default=6,
                    help="tuning iterations (default 6; ~2.9h for hazard at 20 epochs on a T4)")
    ap.add_argument("--epochs", type=int, default=20,
                    help="epochs per iteration — a ranking proxy, not convergence "
                         "(hazard hits ~0.46 mAP50 by ep20 vs 0.53 at ep100)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--optimizer", default="AdamW",
                    help="explicit optimizer; 'auto' is rejected (see module docstring)")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--project", default=None,
                    help="output root; point at mounted Drive on Colab so the "
                         "sweep survives a disconnect (default: runs/detect)")
    args = ap.parse_args()

    if args.optimizer == "auto":
        raise SystemExit(
            "optimizer='auto' ignores lr0 — the sweep would be meaningless. "
            "Use --optimizer AdamW (baseline-comparable) or SGD.")
    if not os.path.isfile(args.data):
        raise SystemExit(f"data.yaml not found: {args.data}")

    from ultralytics import YOLO

    tune_name = f"tune_{args.model}"
    tune_dir = os.path.join(args.project or os.path.join("runs", "detect"), tune_name)
    prior = os.path.join(tune_dir, "tune_results.csv")
    if os.path.isfile(prior):
        with open(prior, encoding="utf-8") as fh:
            done = max(sum(1 for _ in fh) - 1, 0)  # minus header
        print(f"[{args.model}] resuming: {done} iteration(s) already in {prior}")

    print(f"[{args.model}] tuning {list(SPACE)} | {args.iterations} iterations "
          f"x {args.epochs} epochs | optimizer={args.optimizer}")

    model = YOLO(args.weights)
    model.tune(
        data=args.data,
        space=SPACE,
        iterations=args.iterations,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        name=tune_name,
        project=args.project,
        plots=False,      # tuning only needs the fitness number
        save=False,       # per-iteration weights are throwaway; Phase 4 retrains
        resume=True,      # keeps a stable tune dir; exist_ok alone is ignored
    )

    # Keep the WHOLE sweep as a deliverable, not just the winner.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for src, dst in (
        ("tune_results.csv", f"tuning_{args.model}.csv"),
        ("best_hyperparameters.yaml", f"tuning_{args.model}_best.yaml"),
    ):
        s = os.path.join(tune_dir, src)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(RESULTS_DIR, dst))
            print(f"[{args.model}] saved -> {os.path.join(RESULTS_DIR, dst)}")
        else:
            print(f"[{args.model}] WARNING: {s} not found")


if __name__ == "__main__":
    main()
