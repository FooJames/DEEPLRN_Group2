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
import csv
import glob
import json
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


def _flatten(obj, prefix=""):
    """Flatten nested dicts into single-level {a.b: value} — the ndjson schema
    nests hyperparameters/metrics, and column names shouldn't depend on it."""
    out = {}
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _clean_col(name):
    """Readable column names: the ndjson nests everything under
    hyperparameters./datasets.data.metrics/ etc, which makes an unreadable
    header for a deliverable table."""
    for prefix in ("hyperparameters.", "datasets.data.metrics/",
                   "datasets.data.val/", "datasets.data."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("(B)", "").replace("mAP50-95", "mAP50_95").replace("/", "_")


def ndjson_to_rows(path):
    """One JSON object per line -> list of flat dicts sharing a column set."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            flat = _flatten(json.loads(line))
            row = {}
            for k, v in flat.items():
                if k.startswith("save_dirs"):
                    continue          # session-specific paths, not results
                c = _clean_col(k)
                if c not in row:      # keeps top-level fitness over the nested dup
                    row[c] = v
            rows.append(row)
    cols = list(dict.fromkeys(k for r in rows for k in r))  # union, stable order
    return [{c: r.get(c, "") for c in cols} for r in rows]


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

    # Keep the WHOLE sweep as a deliverable, not just the winner. The results
    # file format changed across ultralytics versions: 8.3.x wrote
    # tune_results.csv, 8.4.x writes tune_results.ndjson. Handle both.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dst = os.path.join(RESULTS_DIR, f"tuning_{args.model}.csv")
    found = sorted(glob.glob(os.path.join(tune_dir, "tune_results.*")))
    if not found:
        print(f"[{args.model}] WARNING: no tune_results.* in {tune_dir} — "
              f"the sweep table is a required deliverable; copy it manually.")
    elif found[0].endswith(".csv"):
        shutil.copy2(found[0], dst)
        print(f"[{args.model}] saved -> {dst}")
    else:
        rows = ndjson_to_rows(found[0])
        if rows:
            cols = list(rows[0])
            with open(dst, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
            print(f"[{args.model}] saved -> {dst} "
                  f"({len(rows)} iterations, converted from "
                  f"{os.path.basename(found[0])})")

    best = os.path.join(tune_dir, "best_hyperparameters.yaml")
    if os.path.isfile(best):
        dst = os.path.join(RESULTS_DIR, f"tuning_{args.model}_best.yaml")
        shutil.copy2(best, dst)
        print(f"[{args.model}] saved -> {dst}")
    else:
        print(f"[{args.model}] WARNING: {best} not found")


if __name__ == "__main__":
    main()
