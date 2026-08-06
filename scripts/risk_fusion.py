#!/usr/bin/env python3
"""
risk_fusion.py - the fusion layer, standalone and runnable on any image.

Runs BOTH detectors on a frame, applies the distance rule from context.md,
and writes an annotated copy: every child box, every hazard box with its
class name, a line joining the closest child-hazard pair, and a Safe/Unsafe
banner carrying the measured distance.

    python scripts/risk_fusion.py path/to/image.jpg
    python scripts/risk_fusion.py path/to/folder/ --out runs/risk_fusion

THE DEFAULTS ARE THE VALIDATION-CALIBRATED ONES (Phase 3c, results_and_
findings.md 8.3): centroid distance, threshold 0.3625, detector confidence
0.05. Overriding them changes what this demo draws; it does not change any
number in the paper, which comes from results/metrics/.

THE DETECTORS USE DIFFERENT INPUT RESOLUTIONS (child 416, hazard 640; see
models/README.md). Each is run at its own imgsz - resizing the frame once
for both would silently degrade one of them.

This is an inference entry point only. It scores nothing and reads no
ground truth: risk accuracy lives in evaluate.py, the centroid-vs-edge
comparison in ablation_distance_ref.py. Pointing it at test images is
therefore harmless - it cannot leak a test label into a decision.
"""

import argparse
import math
import os

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

CHILD_RGB = (60, 200, 255)
HAZARD_RGB = (255, 160, 40)
UNSAFE_RGB = (220, 50, 50)
SAFE_RGB = (60, 180, 90)


def rect_gap(a, b):
    """Nearest-edge distance between two boxes (0 if they overlap)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def centroid_gap(a, b):
    return math.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2,
                      (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)


def classify(child_boxes, hazard_boxes, W, H, metric="centroid", threshold=0.3625):
    """The fusion rule. Returns (label, normalised distance, closest pair).

    Either detector returning nothing means Safe with distance None - that is
    rule 1 in context.md, and it is also the pipeline's main failure mode
    (57% of hazards are missed at default confidence, see 8.2), so callers
    should report it rather than let it pass as a confident Safe.
    """
    if not child_boxes or not hazard_boxes:
        return "safe", None, None
    gap = centroid_gap if metric == "centroid" else rect_gap
    diag = math.hypot(W, H)
    d, pair = min(((gap(a, b) / diag, (a, b))
                   for a in child_boxes for b in hazard_boxes),
                  key=lambda x: x[0])
    return ("unsafe" if d <= threshold else "safe"), d, pair


def link_points(a, b, metric):
    """The two points whose separation is what the metric actually measured."""
    if metric == "centroid":
        return ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), \
               ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    pts = []
    for lo_a, hi_a, lo_b, hi_b in ((a[0], a[2], b[0], b[2]),
                                   (a[1], a[3], b[1], b[3])):
        if hi_a < lo_b:
            pts.append((hi_a, lo_b))
        elif hi_b < lo_a:
            pts.append((lo_a, hi_b))
        else:
            mid = (max(lo_a, lo_b) + min(hi_a, hi_b)) / 2
            pts.append((mid, mid))
    return (pts[0][0], pts[1][0]), (pts[0][1], pts[1][1])


def annotate(img, child_boxes, hazard_boxes, hazard_labels, label, d, pair, metric):
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    size = max(14, img.width // 45)
    font = ImageFont.load_default(size=size)
    w = max(2, img.width // 400)

    for box in child_boxes:
        draw.rectangle(box, outline=CHILD_RGB, width=w)
        draw.text((box[0], max(0, box[1] - size - 2)), "child",
                  fill=CHILD_RGB, font=font)
    for box, name in zip(hazard_boxes, hazard_labels):
        draw.rectangle(box, outline=HAZARD_RGB, width=w)
        draw.text((box[0], max(0, box[1] - size - 2)), name,
                  fill=HAZARD_RGB, font=font)

    banner = UNSAFE_RGB if label == "unsafe" else SAFE_RGB
    if pair is not None:
        p, q = link_points(pair[0], pair[1], metric)
        draw.line([p, q], fill=banner, width=w)
    if d is None:
        text = f"{label.upper()}  (no child-hazard pair detected)"
    else:
        text = f"{label.upper()}  {metric} d={d:.4f}"
    draw.rectangle([0, 0, img.width, size + 8], fill=banner)
    draw.text((6, 4), text, fill=(255, 255, 255), font=font)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="image file or a folder of images")
    ap.add_argument("--out", default=os.path.join("runs", "risk_fusion"))
    ap.add_argument("--child-weights", default=os.path.join("models", "child_best.pt"))
    ap.add_argument("--hazard-weights", default=os.path.join("models", "hazard_best.pt"))
    ap.add_argument("--child-imgsz", type=int, default=416)
    ap.add_argument("--hazard-imgsz", type=int, default=640)
    ap.add_argument("--metric", default="centroid", choices=["centroid", "edge"])
    ap.add_argument("--threshold", type=float, default=0.3625)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--no-save", action="store_true",
                    help="print the verdict only, write no annotated image")
    args = ap.parse_args()

    if os.path.isdir(args.source):
        paths = [os.path.join(args.source, f) for f in sorted(os.listdir(args.source))
                 if f.lower().endswith(EXTS)]
    else:
        paths = [args.source]
    if not paths:
        raise SystemExit(f"no images found in {args.source}")

    from ultralytics import YOLO
    from PIL import Image
    child = YOLO(args.child_weights)
    hazard = YOLO(args.hazard_weights)
    names = hazard.model.names

    # Only the shipped defaults carry the calibration; say so honestly when
    # they have been overridden, so a demo run is never mistaken for one.
    calibrated = (args.metric == "centroid" and args.threshold == 0.3625
                  and args.conf == 0.05)
    print(f"[fusion] {args.metric} distance, threshold {args.threshold}, "
          f"conf {args.conf}   "
          f"[{'calibrated on the co-occurrence val split' if calibrated else 'OVERRIDDEN - not the calibrated setting'}]")
    print(f"[fusion] child {args.child_weights} @ {args.child_imgsz} | "
          f"hazard {args.hazard_weights} @ {args.hazard_imgsz}")
    if not args.no_save:
        os.makedirs(args.out, exist_ok=True)

    n_unsafe = n_nodetect = 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        cr = child.predict(p, imgsz=args.child_imgsz, conf=args.conf, verbose=False)[0]
        hr = hazard.predict(p, imgsz=args.hazard_imgsz, conf=args.conf, verbose=False)[0]
        cb = cr.boxes.xyxy.tolist()
        hb = hr.boxes.xyxy.tolist()
        hl = [names[int(c)] for c in hr.boxes.cls.tolist()]

        label, d, pair = classify(cb, hb, img.width, img.height,
                                  args.metric, args.threshold)
        n_unsafe += label == "unsafe"
        n_nodetect += d is None

        dtxt = "   n/a" if d is None else f"{d:.4f}"
        note = "  <- no pair detected, Safe by rule" if d is None else ""
        print(f"  {os.path.basename(p):<28} {label.upper():<6} d={dtxt}  "
              f"child={len(cb)} hazard={len(hb)}{note}")

        if not args.no_save:
            stem = os.path.splitext(os.path.basename(p))[0]
            dest = os.path.join(args.out, f"{stem}_risk.jpg")
            annotate(img, cb, hb, hl, label, d, pair, args.metric).save(dest)

    if len(paths) > 1:
        print(f"\n[fusion] {len(paths)} images: {n_unsafe} unsafe, "
              f"{len(paths) - n_unsafe} safe "
              f"({n_nodetect} of them forced Safe by missing detections)")
    if not args.no_save:
        print(f"[fusion] annotated images -> {args.out}")


if __name__ == "__main__":
    main()
