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

RESUMABLE — always pass --project pointing at mounted Drive. Resume keys off
the trained weights in <project>/ablation_imgsz_<model>_<size>/weights/,
because on Colab the repo (and so results/metrics/) is wiped on disconnect
while Drive is not. A size that already has weights is re-validated in
minutes rather than retrained for hours. The results table is also mirrored
to <project>/ and restored from there. Re-run the identical command to
continue; at most the in-flight resolution is lost.
"""

import argparse
import csv
import os
import shutil
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
    """Sizes already written to the results CSV."""
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {int(r["imgsz"]) for r in csv.DictReader(fh) if r.get("imgsz")}


def run_dir(project, model, size):
    root = project or os.path.join("runs", "detect")
    return os.path.join(root, f"ablation_imgsz_{model}_{size}")


def epochs_completed(project, model, size):
    """How many epochs a previous run actually finished (0 if none).

    Read from the run's results.csv — one row per completed epoch. Neither the
    run directory nor weights/best.pt proves completion: the directory is
    created when training STARTS, and best.pt is rewritten every time the
    model improves, so both exist mid-run.
    """
    p = os.path.join(run_dir(project, model, size), "results.csv")
    if not os.path.isfile(p):
        return 0
    with open(p, encoding="utf-8") as fh:
        return sum(1 for r in csv.DictReader(fh) if any(v.strip() for v in r.values()))


def trained_weights(project, model, size, want_epochs):
    """best.pt for this size, but ONLY if the run reached want_epochs.

    Resume must key off persistent storage, not the results CSV: on Colab the
    CSV lives in the cloned repo (wiped on disconnect) while --project points
    at Drive. A COMPLETE run is re-validated in minutes instead of retrained
    for hours; an INCOMPLETE one must be retrained, not silently recorded as
    if it had finished.
    """
    done = epochs_completed(project, model, size)
    w = os.path.join(run_dir(project, model, size), "weights", "best.pt")
    if not os.path.isfile(w):
        return None, done
    return (w if done >= want_epochs else None), done


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
    if args.project:  # recover the table from Drive if the repo copy was wiped
        mirror = os.path.join(args.project, os.path.basename(out_csv))
        if not already and os.path.isfile(mirror):
            shutil.copy2(mirror, out_csv)
            already = done_sizes(out_csv)
            print(f"[{args.model}] restored results table from {mirror}")
    tuned = load_tuned(args.model)
    print(f"[{args.model}] hyperparameters held fixed: "
          f"{tuned if tuned else 'ultralytics defaults (never tuned)'}")
    if already:
        print(f"[{args.model}] skipping already-done sizes: {sorted(already)}")

    # Show the plan for EVERY size before doing any work. Retraining from
    # scratch when you expected a resume costs hours, so it must be visible
    # in the first second, not discovered later.
    print(f"\n[{args.model}] project dir: "
          f"{args.project or os.path.join('runs', 'detect')}")
    print(f"[{args.model}] plan:")
    plan_retrain = 0
    for size in args.sizes:
        if size in already:
            what = "skip (already in results table)"
        else:
            w, done = trained_weights(args.project, args.model, size, args.epochs)
            last = os.path.isfile(os.path.join(
                run_dir(args.project, args.model, size), "weights", "last.pt"))
            if w:
                what = f"validate only ({done}/{args.epochs} epochs done)"
            elif done and last:
                what = f"resume from epoch {done} ({args.epochs - done} left)"
            else:
                what = (f"TRAIN FROM SCRATCH ({args.epochs} epochs)"
                        + (f" — partial {done} but no last.pt" if done else ""))
                plan_retrain += 1
        print(f"    imgsz={size:<4} -> {what}")
    if plan_retrain:
        print(f"[{args.model}] NOTE: {plan_retrain} size(s) will train from "
              f"scratch. If you expected a resume, stop now and check "
              f"--project points at the dir holding "
              f"ablation_imgsz_{args.model}_<size>/weights/.")
    print()

    for size in args.sizes:
        if size in already:
            continue
        print(f"\n[{args.model}] === imgsz={size} ===")

        recovered, done_ep = trained_weights(args.project, args.model, size,
                                             args.epochs)
        if recovered:
            # Complete run from a previous session: don't burn hours redoing it
            # just because the results CSV was on ephemeral storage.
            print(f"[{args.model}] complete run found ({done_ep}/{args.epochs} "
                  f"epochs) — validating instead of retraining")
            model = YOLO(recovered)
            train_min = ""          # not recoverable after the fact
        else:
            last = os.path.join(run_dir(args.project, args.model, size),
                                "weights", "last.pt")
            if done_ep and os.path.isfile(last):
                # Continue the interrupted run instead of paying for the epochs
                # it already did. Pass the checkpoint PATH, not resume=True:
                # resume=True falls back to get_latest_run(), which picks the
                # newest run anywhere and could resume the wrong one. The
                # checkpoint carries its own args (lr0/box/cls/dfl/epochs), so
                # the continuation matches the original configuration.
                print(f"[{args.model}] resuming PARTIAL run from epoch "
                      f"{done_ep}/{args.epochs}")
                model = YOLO(last)
                model.train(resume=last, data=args.data, imgsz=size)
                train_min = ""      # only the resumed portion; not a full-run time
            else:
                if done_ep:
                    print(f"[{args.model}] partial run ({done_ep}/{args.epochs}) "
                          f"but no last.pt — retraining from scratch")
                model = YOLO(args.weights)
                t0 = time.time()
                model.train(data=args.data, imgsz=size, epochs=args.epochs,
                            batch=args.batch, optimizer=args.optimizer,
                            name=f"ablation_imgsz_{args.model}_{size}",
                            project=args.project, plots=False,
                            exist_ok=True, **tuned)
                train_min = round((time.time() - t0) / 60, 2)
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
            "train_min": train_min,
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

    # Mirror the table to persistent storage as well. The run dirs are the
    # source of truth for resume, but this avoids re-validating every size
    # after each reconnect.
    if args.project and os.path.isfile(out_csv):
        mirror = os.path.join(args.project, os.path.basename(out_csv))
        shutil.copy2(out_csv, mirror)
        print(f"[{args.model}] mirrored -> {mirror}")

    print(f"\n[{args.model}] done. Results: {out_csv}")


if __name__ == "__main__":
    main()
