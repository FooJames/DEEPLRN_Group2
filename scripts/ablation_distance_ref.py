#!/usr/bin/env python3
"""
ablation_distance_ref.py - Phase 3c: centroid vs nearest-edge distance.

Runs BOTH final detectors over the co-occurrence set, applies the risk-fusion
rule under each candidate distance metric, and calibrates a threshold for
each. This ablates the FUSION LAYER, not the detectors: the same predictions
feed both metrics, so the only thing that varies is how child-hazard
proximity is measured.

    python scripts/ablation_distance_ref.py            # val split
    python scripts/ablation_distance_ref.py --split test --confirm-test

CALIBRATE ON VAL, TOUCH TEST ONCE. The threshold is swept on the validation
split only; the test split is refused unless --confirm-test is passed, so
Phase 5 keeps a genuinely held-out number.

THE DETECTORS USE DIFFERENT INPUT RESOLUTIONS (child 416, hazard 640; see
models/README.md). Each is run at its own imgsz - resizing the frame once
for both would silently degrade one of them.

WHY THIS RUNS ON PREDICTIONS, NOT GROUND-TRUTH BOXES: on exact geometry both
metrics separate the classes perfectly, because the reachability label and
the diagonal-normalised predictors differ only by a child-size factor
(see results_and_findings.md 9.1). Real detections carry box-extent and
localisation error, which the two metrics absorb differently - edge distance
depends on box borders, centroid distance on box centres - so predictions are
where a difference can actually show up.

Reported per metric: accuracy, precision/recall for the unsafe class, and the
calibrated threshold. Detection failures are reported separately so fusion
error is not confused with detector error.
"""

import argparse
import csv
import math
import os

RESULTS_DIR = os.path.join("results", "metrics")


def rect_gap(a, b):
    """Nearest-edge distance between two boxes (0 if they overlap)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def centroid_gap(a, b):
    return math.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2,
                      (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)


def sweep(rows, key):
    """Best single threshold on `key`: predict unsafe when value <= t.

    Images where a detector found nothing are Safe by the fusion rule
    regardless of threshold, so they are scored but never contribute a
    candidate threshold.
    """
    vals = sorted({r[key] for r in rows if r[key] is not None})
    best = (-1.0, None)
    for t in vals:
        m = score(rows, key, t)
        if m["balanced"] > best[0]:      # calibrate on BALANCED accuracy
            best = (m["balanced"], t)
    return best


def score(rows, key, t):
    tp = fp = fn = tn = 0
    for r in rows:
        pred_unsafe = r[key] is not None and r[key] <= t
        if r["truth"] == "unsafe":
            tp += pred_unsafe
            fn += not pred_unsafe
        else:
            fp += pred_unsafe
            tn += not pred_unsafe
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0          # recall on unsafe (TPR)
    spec = tn / (tn + fp) if tn + fp else 0.0         # recall on safe (TNR)
    # The set is ~74% unsafe, so plain accuracy rewards a constant "unsafe"
    # predictor (0.74). Balanced accuracy is the honest headline: a constant
    # predictor scores 0.5 on it regardless of class balance.
    return dict(accuracy=(tp + tn) / len(rows), balanced=(rec + spec) / 2,
                precision=prec, recall=rec, specificity=spec,
                tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join("data", "cooccurrence"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--confirm-test", action="store_true",
                    help="required to touch the test split (Phase 5 only)")
    ap.add_argument("--child-weights", default=os.path.join("models", "child_best.pt"))
    ap.add_argument("--hazard-weights", default=os.path.join("models", "hazard_best.pt"))
    ap.add_argument("--child-imgsz", type=int, default=416)
    ap.add_argument("--hazard-imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=None,
                    help="use this threshold instead of calibrating (Phase 5)")
    ap.add_argument("--metric", default=None, choices=["centroid", "edge"],
                    help="with --threshold, which metric it belongs to")
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_test:
        raise SystemExit(
            "Refusing to touch the test split. It is held out for Phase 5; "
            "calibrate on val. Pass --confirm-test when you genuinely mean it.")

    labels = os.path.join(args.root, "cooccurrence_labels.csv")
    if not os.path.isfile(labels):
        raise SystemExit(f"missing {labels} - run make_cooccurrence_eval.py first")
    truth = {r["filename"]: r for r in csv.DictReader(open(labels, encoding="utf-8"))
             if r["split"] == args.split}
    if not truth:
        raise SystemExit(f"no rows for split={args.split}")

    from ultralytics import YOLO
    from PIL import Image
    child = YOLO(args.child_weights)
    hazard = YOLO(args.hazard_weights)
    img_dir = os.path.join(args.root, args.split, "images")
    print(f"[3c] {len(truth)} images, split={args.split}, conf={args.conf}")
    print(f"[3c] child {args.child_weights} @ {args.child_imgsz} | "
          f"hazard {args.hazard_weights} @ {args.hazard_imgsz}")

    rows, no_child, no_hazard = [], 0, 0
    for fn, t in sorted(truth.items()):
        p = os.path.join(img_dir, fn)
        W, H = Image.open(p).size
        diag = math.hypot(W, H)
        cb = child.predict(p, imgsz=args.child_imgsz, conf=args.conf,
                           verbose=False)[0].boxes.xyxy.tolist()
        hb = hazard.predict(p, imgsz=args.hazard_imgsz, conf=args.conf,
                            verbose=False)[0].boxes.xyxy.tolist()
        no_child += not cb
        no_hazard += not hb
        if cb and hb:
            cen = min(centroid_gap(a, b) for a in cb for b in hb) / diag
            edg = min(rect_gap(a, b) for a in cb for b in hb) / diag
        else:
            cen = edg = None            # fusion rule: nothing detected -> Safe
        rows.append({"filename": fn, "truth": t["label"], "centroid": cen,
                     "edge": edg, "n_child": len(cb), "n_hazard": len(hb)})

    n_un = sum(r["truth"] == "unsafe" for r in rows)
    print(f"\n[3c] detection: {no_child} images with NO child, "
          f"{no_hazard} with NO hazard "
          f"({100*sum(1 for r in rows if r['centroid'] is None)/len(rows):.0f}% "
          f"forced Safe by the fusion rule)")
    print(f"[3c] truth: {n_un} unsafe / {len(rows)-n_un} safe | constant-'unsafe' "
          f"predictor scores plain {n_un/len(rows):.3f} but balanced 0.500")

    out, results = [], {}
    for key in ("centroid", "edge"):
        if args.threshold is not None and args.metric == key:
            t, acc = args.threshold, None
        elif args.threshold is not None:
            continue
        else:
            acc, t = sweep(rows, key)
        m = score(rows, key, t)
        results[key] = (t, m)
        print(f"\n[3c] {key.upper()} distance   threshold={t:.4f}")
        print(f"       BALANCED acc {m['balanced']:.4f}  (plain {m['accuracy']:.4f})")
        print(f"       unsafe: precision {m['precision']:.4f} recall {m['recall']:.4f}"
              f" | safe recall {m['specificity']:.4f}")
        print(f"       tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")
        out.append({"split": args.split, "metric": key, "threshold": round(t, 5),
                    "n_images": len(rows), "n_unsafe": n_un,
                    "balanced_accuracy": round(m["balanced"], 5),
                    "accuracy": round(m["accuracy"], 5),
                    "specificity": round(m["specificity"], 5),
                    "precision": round(m["precision"], 5),
                    "recall": round(m["recall"], 5),
                    "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                    "no_child_imgs": no_child, "no_hazard_imgs": no_hazard,
                    "conf": args.conf})

    if len(results) == 2:
        d = results["edge"][1]["balanced"] - results["centroid"][1]["balanced"]
        print(f"\n[3c] EDGE - CENTROID = {d:+.4f}  -> "
              f"{'edge' if d > 0 else 'centroid' if d < 0 else 'tie'} wins on {args.split}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summ = os.path.join(RESULTS_DIR, f"ablation_distance_ref_{args.split}.csv")
    with open(summ, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    per = os.path.join(RESULTS_DIR, f"distance_ref_predictions_{args.split}.csv")
    with open(per, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n[3c] -> {summ}\n[3c] -> {per}")


if __name__ == "__main__":
    main()
