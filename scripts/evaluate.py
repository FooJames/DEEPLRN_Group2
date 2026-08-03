#!/usr/bin/env python3
"""
evaluate.py - Phase 5: final evaluation on the held-out test splits.

Produces every test-set number in one pass:
  1. child detector  mAP on data/child   test split
  2. hazard detector mAP on data/hazard  test split, overall + per class
  3. risk classification on the co-occurrence test split, using the
     threshold CALIBRATED ON VALIDATION in Phase 3c/4b
  4. inference cost for the two-detector pipeline
  5. a comparison table against the published baselines

    python scripts/evaluate.py --dry-run     # rehearse on val, test untouched
    python scripts/evaluate.py --confirm-test

THIS IS A ONE-SHOT EVALUATION. The test splits have been untouched for the
whole project. Nothing here calibrates, tunes or selects anything: the
threshold, the confidence and the metric are all fixed inputs, taken from
the validation-calibrated Phase 3c result. Re-running this after changing
anything in response to its output would turn the test set into a
validation set and invalidate every number in it. Use --dry-run to rehearse
against validation as often as you like.

Defaults come from Phase 3c (results_and_findings.md 8.3):
    metric = centroid, threshold = 0.3625, detector confidence = 0.05
"""

import argparse
import csv
import math
import os

RESULTS_DIR = os.path.join("results", "metrics")

# Published numbers, for the comparison table. These are NOT directly
# comparable to ours - different datasets, classes, model sizes and in some
# cases a different task - so the table records what each one measured.
LITERATURE = [
    ("Ahmad et al. (2025)", "YOLOv8l", "own merged child+hazard set",
     "~0.90 mAP", "different dataset and classes; larger model (l vs n)"),
    ("AlMhdawi et al. (2026)", "YOLOv8 dual", "fire/engineering sites",
     "n/a", "different domain; one detector is COCO-pretrained"),
    ("Ramadan et al. (2025)", "YOLOv11n + pose", "hand-to-mouth",
     "0.92 / 0.74 acc", "different task: hand-to-mouth, not child-hazard distance"),
    ("Khan & Dey (2024)", "classifier", "ChildSUn",
     "n/a", "classification only; no spatial reasoning"),
]


def rect_gap(a, b):
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def centroid_gap(a, b):
    return math.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2,
                      (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)


def detector_eval(weights, data, imgsz, split):
    from ultralytics import YOLO
    m = YOLO(weights).val(data=data, imgsz=imgsz, split=split,
                          plots=False, verbose=False)
    per = []
    names = getattr(m, "names", {}) or {}
    for i, c in enumerate(getattr(m.box, "ap_class_index", [])):
        p_, r_, a50, a = m.box.class_result(i)
        per.append({"class": names.get(int(c), int(c)),
                    "precision": round(float(p_), 5), "recall": round(float(r_), 5),
                    "mAP50": round(float(a50), 5), "mAP50_95": round(float(a), 5)})
    return {"precision": round(float(m.box.mp), 5), "recall": round(float(m.box.mr), 5),
            "mAP50": round(float(m.box.map50), 5), "mAP50_95": round(float(m.box.map), 5),
            "infer_ms": round(float(m.speed.get("inference", 0)), 3)}, per


def fusion_eval(root, split, cw, hw, cz, hz, conf, metric, thr):
    from ultralytics import YOLO
    from PIL import Image
    child, hazard = YOLO(cw), YOLO(hw)
    truth = {r["filename"]: r for r in
             csv.DictReader(open(os.path.join(root, "cooccurrence_labels.csv"),
                                 encoding="utf-8")) if r["split"] == split}
    gap = centroid_gap if metric == "centroid" else rect_gap
    tp = fp = fn = tn = 0
    missed = 0
    rows = []
    for fn_, t in sorted(truth.items()):
        p = os.path.join(root, split, "images", fn_)
        W, H = Image.open(p).size
        cb = child.predict(p, imgsz=cz, conf=conf, verbose=False)[0].boxes.xyxy.tolist()
        hb = hazard.predict(p, imgsz=hz, conf=conf, verbose=False)[0].boxes.xyxy.tolist()
        if cb and hb:
            d = min(gap(a, b) for a in cb for b in hb) / math.hypot(W, H)
            pred = "unsafe" if d <= thr else "safe"
        else:
            d, pred, missed = None, "safe", missed + 1   # fusion rule
        if t["label"] == "unsafe":
            tp += pred == "unsafe"; fn += pred == "safe"
        else:
            fp += pred == "unsafe"; tn += pred == "safe"
        rows.append({"filename": fn_, "truth": t["label"], "pred": pred,
                     "distance": None if d is None else round(d, 5),
                     "n_child": len(cb), "n_hazard": len(hb)})
    n = len(rows)
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"n_images": n, "n_unsafe": tp + fn, "metric": metric, "threshold": thr,
            "conf": conf, "accuracy": round((tp + tn) / n, 5),
            "balanced_accuracy": round((rec + spec) / 2, 5),
            "precision": round(tp / (tp + fp), 5) if tp + fp else 0.0,
            "recall": round(rec, 5), "specificity": round(spec, 5),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "no_detection_imgs": missed}, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm-test", action="store_true",
                    help="required: evaluates on the held-out test splits")
    ap.add_argument("--dry-run", action="store_true",
                    help="rehearse on the validation splits instead")
    ap.add_argument("--child-weights", default=os.path.join("models", "child_best.pt"))
    ap.add_argument("--hazard-weights", default=os.path.join("models", "hazard_best.pt"))
    ap.add_argument("--child-imgsz", type=int, default=416)
    ap.add_argument("--hazard-imgsz", type=int, default=640)
    ap.add_argument("--cooc-root", default=os.path.join("data", "cooccurrence"))
    ap.add_argument("--metric", default="centroid", choices=["centroid", "edge"])
    ap.add_argument("--threshold", type=float, default=0.3625)
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()

    if not args.dry_run and not args.confirm_test:
        raise SystemExit(
            "Refusing to touch the test splits without --confirm-test.\n"
            "This is a ONE-SHOT evaluation: acting on its output and re-running "
            "would turn the test set into a validation set.\n"
            "Rehearse with --dry-run (uses validation) first.")
    # ultralytics keys its data dict by "val"/"test" (the yaml says
    # `val: valid/images`), so the split NAME is "val", not "valid".
    split = "val" if args.dry_run else "test"
    cooc_split = "val" if args.dry_run else "test"
    tag = "dryrun" if args.dry_run else "test"

    print("=" * 70)
    print(f"PHASE 5 {'DRY RUN (validation)' if args.dry_run else 'FINAL EVALUATION (TEST)'}")
    print(f"  fusion: {args.metric} distance, threshold {args.threshold}, "
          f"detector conf {args.conf}   [calibrated on validation, fixed here]")
    print("=" * 70)

    det, per_class = {}, {}
    for name, w, data, imgsz in (
            ("child", args.child_weights, os.path.join("data", "child", "data.yaml"),
             args.child_imgsz),
            ("hazard", args.hazard_weights, os.path.join("data", "hazard", "data.yaml"),
             args.hazard_imgsz)):
        d, pc = detector_eval(w, data, imgsz, split)
        det[name], per_class[name] = d, pc
        print(f"\n[{name}] {split} split @ imgsz {imgsz}")
        print(f"  mAP50 {d['mAP50']:.4f}  mAP50-95 {d['mAP50_95']:.4f}  "
              f"P {d['precision']:.4f}  R {d['recall']:.4f}  {d['infer_ms']:.2f} ms/img")

    fus, fus_rows = fusion_eval(args.cooc_root, cooc_split, args.child_weights,
                                args.hazard_weights, args.child_imgsz,
                                args.hazard_imgsz, args.conf, args.metric,
                                args.threshold)
    print(f"\n[risk fusion] {cooc_split} split, {fus['n_images']} images "
          f"({fus['n_unsafe']} unsafe)")
    print(f"  BALANCED accuracy {fus['balanced_accuracy']:.4f}   "
          f"(plain {fus['accuracy']:.4f}; a constant predictor scores 0.500 balanced)")
    print(f"  unsafe precision {fus['precision']:.4f} recall {fus['recall']:.4f} | "
          f"safe recall {fus['specificity']:.4f}")
    print(f"  tp={fus['tp']} fp={fus['fp']} fn={fus['fn']} tn={fus['tn']}  "
          f"| {fus['no_detection_imgs']} images had no detection (forced Safe)")

    total_ms = det["child"]["infer_ms"] + det["hazard"]["infer_ms"]
    print(f"\n[cost] {det['child']['infer_ms']:.2f} + {det['hazard']['infer_ms']:.2f}"
          f" = {total_ms:.2f} ms/frame for both detectors "
          f"(~{1000/total_ms:.0f} fps)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{tag}_detectors.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "split", "imgsz"] + list(det["child"]))
        w.writeheader()
        for k, v in det.items():
            w.writerow({"model": k, "split": split,
                        "imgsz": args.child_imgsz if k == "child" else args.hazard_imgsz,
                        **v})
    for k, pc in per_class.items():
        if pc:
            with open(os.path.join(RESULTS_DIR, f"{tag}_per_class_{k}.csv"), "w",
                      newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(pc[0])); w.writeheader(); w.writerows(pc)
    with open(os.path.join(RESULTS_DIR, f"{tag}_risk_fusion.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fus)); w.writeheader(); w.writerow(fus)
    with open(os.path.join(RESULTS_DIR, f"{tag}_risk_predictions.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fus_rows[0])); w.writeheader(); w.writerows(fus_rows)

    comp = os.path.join(RESULTS_DIR, f"{tag}_literature_comparison.csv")
    with open(comp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "model", "evaluated on", "reported", "why not directly comparable"])
        w.writerow(["This work (child)", "YOLOv8n",
                    f"child-detection-piuns {split}", f"{det['child']['mAP50']:.4f} mAP50", "-"])
        w.writerow(["This work (hazard)", "YOLOv8n",
                    f"harmful-objects {split}", f"{det['hazard']['mAP50']:.4f} mAP50", "-"])
        w.writerow(["This work (risk fusion)", "dual YOLOv8n + distance rule",
                    f"composited co-occurrence {cooc_split}",
                    f"{fus['balanced_accuracy']:.4f} balanced acc", "-"])
        for row in LITERATURE:
            w.writerow(row)
    print(f"\n[comparison] -> {comp}")
    print("  NOTE: literature numbers are on different datasets, class sets and")
    print("  model sizes. The claim supported is that our specialists are")
    print("  COMPETITIVE with published figures, not that they beat them.")
    print(f"\nwrote {tag}_detectors.csv, {tag}_per_class_*.csv, "
          f"{tag}_risk_fusion.csv, {tag}_risk_predictions.csv")


if __name__ == "__main__":
    main()
