#!/usr/bin/env python3
"""
label_safe_unsafe.py

A labeling *helper* for the DEEPLRN Group 2 child-safety risk detector.

This script does NOT decide final Safe/Unsafe labels. Its only job is to
make MANUAL labeling faster by pre-sorting images and reporting the
child<->hazard proximity so a human can eyeball and confirm:

  - reads YOLOv8-format annotations for a "co-occurrence" dataset
    (child + hazard objects in the same images),
  - for each image, computes the minimum normalized centroid distance
    between any child box and any hazard box (the same geometry the
    fusion layer / risk_fusion.py will use -- see context.md),
  - suggests "safe"/"unsafe" against a PLACEHOLDER threshold,
  - writes labels.csv with a blank `final_label` column for you to fill
    in by hand.

Optionally also ingests a second "child-only" dataset (no hazard objects
at all) and auto-tags every image "safe (trivial)" to pad out easy cases.

Output is CSV only -- the dataset's own annotations are never modified,
and no images are written.

Only third-party dependency: Pillow, used solely to read each image's
pixel dimensions so the distance can be normalized by the true image
diagonal (aspect ratio matters when images aren't square).

Usage:
    python label_safe_unsafe.py /path/to/cooccurrence_dataset/ [/path/to/child_only_dataset/]

--------------------------------------------------------------------------
LABEL FORMAT NOTE (read this):
The task assumed each label line is `class_id xc yc w h` (5 fields,
detection/bbox format). The datasets currently in this repo are actually
exported as POLYGON / segmentation labels: `class_id x1 y1 x2 y2 ...`.
This script handles BOTH: for a polygon it derives the axis-aligned
bounding box and uses that box's center as the centroid -- the same
centroid a detection box would have, so the fusion distance is consistent
either way. When a polygon label is seen, the script prints a warning so
this is never silent.
--------------------------------------------------------------------------
"""

import argparse
import csv
import math
import os
import re

from PIL import Image


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

# Name of the "child" class WITHIN EACH dataset's own data.yaml. These are
# two separate constants ON PURPOSE: the co-occurrence set and the
# child-only set were exported independently and may name the child class
# differently (e.g. "toddler" vs "child"). Do NOT assume they match.
CHILD_CLASS_NAME_COOCCURRENCE = "child"
CHILD_CLASS_NAME_CHILDONLY = "child"

# PLACEHOLDER sorting threshold on the normalized centroid distance
# (min over all child-hazard pairs, range ~0..1.4). This exists ONLY to
# pre-sort images for manual review -- it is NOT the real calibrated risk
# threshold. The real threshold gets derived on the validation set later,
# once we actually have final hand labels (context.md, Phase 4). Anything
# closer than this is *suggested* "unsafe", anything farther "safe".
# Changing it changes only the suggestion, never a final label.
#
# NOTE on this dataset: because the child box tends to fill much of the
# frame, centroid distances here cluster high (~0.2-0.48, median ~0.29)
# even when the child is right next to the hazard, so most images will be
# *suggested* "safe" under centroid geometry. That is the known centroid-
# vs-edge effect (context.md ablation #4), not a safe scene -- rely on the
# min_centroid_distance_norm column and your own eyes, not the suggestion.
SUGGESTED_THRESHOLD = 0.15

# Splits to look for inside each dataset folder (standard Roboflow export).
# Missing splits are skipped silently -- e.g. an export with only train/.
SPLITS = ("train", "valid", "test")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTPUT_CSV = "labels.csv"


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------

def _fs(path):
    """Return a filesystem path safe to open on Windows past the 260-char
    MAX_PATH limit, via the \\?\ extended-length prefix. Roboflow filenames
    are long enough that a dataset on the Desktop can blow past MAX_PATH.
    No-op on non-Windows."""
    if os.name != "nt":
        return path
    abspath = os.path.abspath(path)
    if abspath.startswith("\\\\?\\"):
        return abspath
    if abspath.startswith("\\\\"):  # UNC share
        return "\\\\?\\UNC\\" + abspath[2:]
    return "\\\\?\\" + abspath


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def parse_class_names(data_yaml_path):
    """Pull the `names:` list out of a Roboflow data.yaml without a YAML dep.

    Handles inline flow style:   names: ['a', 'b']
    and block style:
        names:
          - a
          - b
    """
    with open(_fs(data_yaml_path), "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue
        after = stripped[len("names:"):].strip()
        if after.startswith("["):
            # inline flow list (possibly spanning lines): join until ']'
            buf = after
            j = i
            while "]" not in buf and j + 1 < len(lines):
                j += 1
                buf += " " + lines[j].strip()
            inner = buf[buf.index("[") + 1: buf.index("]")]
            return [n.strip().strip("'\"") for n in inner.split(",") if n.strip()]
        # block style: collect the following "- item" lines
        names = []
        for block_line in lines[i + 1:]:
            if block_line.strip() == "":
                continue
            m = re.match(r"\s*-\s*(.+?)\s*$", block_line)
            if not m:
                break
            names.append(m.group(1).strip().strip("'\""))
        return names
    raise ValueError(f"No `names:` key found in {data_yaml_path}")


def parse_boxes(label_path):
    """Return (boxes, saw_polygon) for one YOLO label file.

    Each box is a dict with normalized centroid + bounding box:
        {class_id, cx, cy, xmin, ymin, xmax, ymax}   (all 0-1)

    Handles BOTH label formats we might get from a Roboflow YOLOv8 export:
      * detection:     `class_id xc yc w h`               (4 coords)
      * segmentation:  `class_id x1 y1 x2 y2 ... xn yn`   (even #coords, >=6)
    For a polygon we derive the axis-aligned bounding box and use its
    center as the centroid. A missing label file -> no boxes (not an error).
    """
    boxes = []
    saw_polygon = False
    if not os.path.exists(_fs(label_path)):
        return boxes, saw_polygon

    with open(_fs(label_path), "r", encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                continue

            if len(coords) == 4:
                xc, yc, w, h = coords
                xmin, ymin = xc - w / 2, yc - h / 2
                xmax, ymax = xc + w / 2, yc + h / 2
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                saw_polygon = True
                xs, ys = coords[0::2], coords[1::2]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
            else:
                # unexpected arity -- skip rather than guess
                continue

            boxes.append({
                "class_id": class_id,
                "cx": xc, "cy": yc,
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            })
    return boxes, saw_polygon


def idx_name(class_names, class_id):
    """Class name for an id, or '' if the id is out of range."""
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return ""


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def min_normalized_centroid_distance(child_boxes, hazard_boxes, img_w, img_h):
    """Minimum over all child-hazard pairs of:
        euclidean(child_centroid, hazard_centroid) in pixels / image diagonal

    Centroids arrive normalized (0-1); convert to pixels first so the image
    aspect ratio is respected, then divide by the pixel diagonal. This is
    the d_norm formula the fusion layer uses (context.md)."""
    diag = math.hypot(img_w, img_h)
    best = None
    for c in child_boxes:
        cx, cy = c["cx"] * img_w, c["cy"] * img_h
        for h in hazard_boxes:
            hx, hy = h["cx"] * img_w, h["cy"] * img_h
            d = math.hypot(cx - hx, cy - hy) / diag
            if best is None or d < best:
                best = d
    return best


# --------------------------------------------------------------------------
# Dataset iteration + per-dataset processing
# --------------------------------------------------------------------------

def iter_split_images(dataset_dir):
    """Yield (split, filename, image_path, label_path) for every image in
    whichever of train/valid/test splits actually exist."""
    for split in SPLITS:
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
                continue
            img_path = os.path.join(img_dir, fn)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
            yield split, fn, img_path, lbl_path


def process_cooccurrence(dataset_dir):
    """Process the child+hazard co-occurrence dataset. Returns
    (rows, n_images, saw_polygon)."""
    class_names = parse_class_names(os.path.join(dataset_dir, "data.yaml"))
    child_name = CHILD_CLASS_NAME_COOCCURRENCE
    if child_name not in class_names:
        print(f"[WARN] co-occurrence child class '{child_name}' is not in "
              f"data.yaml names {class_names}. No boxes will be treated as "
              f"child -- fix CHILD_CLASS_NAME_COOCCURRENCE.")

    rows = []
    n = 0
    saw_polygon = False
    for _split, fn, img_path, lbl_path in iter_split_images(dataset_dir):
        boxes, saw_poly = parse_boxes(lbl_path)
        saw_polygon = saw_polygon or saw_poly

        # "hazard" = every non-child class in this dataset.
        child_boxes = [b for b in boxes
                       if idx_name(class_names, b["class_id"]) == child_name]
        hazard_boxes = [b for b in boxes
                        if idx_name(class_names, b["class_id"]) != child_name]

        if child_boxes and hazard_boxes:
            with Image.open(_fs(img_path)) as im:
                W, H = im.size
            d = min_normalized_centroid_distance(child_boxes, hazard_boxes, W, H)
            suggested = "unsafe" if d < SUGGESTED_THRESHOLD else "safe"
            dist_str = f"{d:.4f}"
        else:
            # Missing a child OR a hazard box -> can't compute proximity.
            # Mark it rather than guessing a label.
            suggested = "N/A (missing child or hazard box)"
            dist_str = ""

        rows.append([fn, "cooccurrence", suggested, dist_str, ""])
        n += 1

    return rows, n, saw_polygon


def process_child_only(dataset_dir):
    """Process the optional child-only dataset (no hazards at all). Returns
    (rows, n_images, saw_polygon)."""
    child_name = CHILD_CLASS_NAME_CHILDONLY  # kept separate from co-occ on purpose

    rows = []
    n = 0
    saw_polygon = False
    for _split, fn, _img_path, lbl_path in iter_split_images(dataset_dir):
        _boxes, saw_poly = parse_boxes(lbl_path)
        saw_polygon = saw_polygon or saw_poly

        # A child-only image is a TRIVIAL safe case by our own fusion rule:
        # "if either detector finds nothing, label Safe" -- here the hazard
        # detector has nothing to detect, so there is no distance to compute.
        # These are tagged source="child_only" (NOT "cooccurrence") on
        # purpose, so the easy trivial-safe cases stay SEPARABLE from the
        # hard "child + hazard but far apart" cases when risk-classification
        # accuracy is reported later. They must not be blended into one
        # accuracy number.
        rows.append([fn, "child_only", "safe (trivial)", "", ""])
        n += 1

    return rows, n, saw_polygon


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Suggest Safe/Unsafe labels for manual review "
                    "(pre-sorting helper, not a final labeler).")
    ap.add_argument("cooccurrence_dir",
                    help="co-occurrence dataset folder (child + hazard)")
    ap.add_argument("child_only_dir", nargs="?", default=None,
                    help="optional child-only dataset folder")
    args = ap.parse_args()

    all_rows = []
    co_rows, co_n, co_poly = process_cooccurrence(args.cooccurrence_dir)
    all_rows.extend(co_rows)
    if co_n == 0:
        print(f"[WARN] no images found under {args.cooccurrence_dir} "
              f"(expected train/valid/test with images/ + labels/).")

    child_n = 0
    child_poly = False
    if args.child_only_dir:
        c_rows, child_n, child_poly = process_child_only(args.child_only_dir)
        all_rows.extend(c_rows)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "source", "suggested_label",
                    "min_centroid_distance_norm", "final_label"])
        w.writerows(all_rows)

    print("\n==== summary ====")
    print(f"co-occurrence images processed: {co_n}")
    if args.child_only_dir:
        print(f"child-only images processed:    {child_n}")
    print(f"total images processed:         {co_n + child_n}")
    print(f"labels csv -> {os.path.abspath(OUTPUT_CSV)}")

    if co_poly or child_poly:
        print("\n[FORMAT NOTE] Some/all label files are POLYGON (segmentation) "
              "format, not `class_id xc yc w h` bbox format. Centroids were "
              "derived from each polygon's bounding box (mathematically the "
              "same centroid). If you expected bbox labels, re-export the "
              "dataset as YOLOv8 'Object Detection', not 'Instance Segmentation'.")


if __name__ == "__main__":
    main()
