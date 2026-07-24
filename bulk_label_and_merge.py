#!/usr/bin/env python3
"""
bulk_label_and_merge.py

Second-stage helper on top of label_safe_unsafe.py + labels.csv. It:

  1. Bulk-assigns `final_label`:
       - source == "cooccurrence"  -> "unsafe"
       - source == "child_only"    -> "safe"
     Rows still marked "N/A (missing child or hazard box)" are NOT
     bulk-labeled -- left blank for a human, since they may be hazard-only
     or child-only images that landed in the co-occurrence export.

     WHY bulk-unsafe for the co-occurrence set: it is staged "dangerous
     situation" stock photography. Prior analysis (see PROGRESS.md) found
     35% of its images have overlapping child/hazard boxes and a median
     child<->hazard gap of ~2.5% of the frame -- i.e. uniformly close by
     design, not a natural near/far mix. So "unsafe" is a justified
     default here, not a shortcut.

  2. Flags likely exceptions for a human spot-check via a new
     `flagged_for_review` column (true/false). A co-occurrence row is
     flagged when its min_centroid_distance_norm sits above the 90th
     percentile of that dataset's distances -- the images most likely to
     be a genuine "actually far apart" exception to the bulk-unsafe rule.

  3. Merges into datasets/labeled/{safe,unsafe}/ by COPYING each image
     (originals untouched). N/A / blank rows are skipped. Filenames are
     preserved; on a cross-source collision the copy is prefixed with the
     source value instead of overwriting.

  4. Prints a full report: per-folder counts, the flag list + cutoff
     rationale, and an explicit accounting of every row not copied.

Usage:
    python bulk_label_and_merge.py /path/to/cooccurrence_dataset/ [/path/to/child_only_dataset/] [--csv labels.csv]
"""

import argparse
import csv
import os
import shutil

# Reuse the long-path helper + constants from the first script so behaviour
# (MAX_PATH handling, which splits/extensions count) stays identical.
from label_safe_unsafe import _fs, IMAGE_EXTS, SPLITS

LABELED_ROOT = os.path.join("datasets", "labeled")
FLAG_PERCENTILE = 90  # co-occurrence rows above this distance percentile get flagged
NA_PREFIX = "N/A"     # how process_cooccurrence marks missing child/hazard rows


def build_image_index(dataset_dir):
    """filename -> full path, across whichever train/valid/test splits exist."""
    index = {}
    if not dataset_dir:
        return index
    for split in SPLITS:
        img_dir = os.path.join(dataset_dir, split, "images")
        if not os.path.isdir(img_dir):
            continue
        for fn in os.listdir(img_dir):
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                index.setdefault(fn, os.path.join(img_dir, fn))
    return index


def percentile(sorted_values, p):
    """Linear-interpolation percentile of an already-sorted list."""
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser(
        description="Bulk-label labels.csv and merge images into "
                    "datasets/labeled/{safe,unsafe}/.")
    ap.add_argument("cooccurrence_dir",
                    help="co-occurrence dataset folder (source of unsafe images)")
    ap.add_argument("child_only_dir", nargs="?", default=None,
                    help="optional child-only dataset folder (source of safe images)")
    ap.add_argument("--csv", default="labels.csv",
                    help="labels.csv to read and rewrite (default: labels.csv)")
    args = ap.parse_args()

    # ---- read ----
    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # image lookup per source
    source_index = {
        "cooccurrence": build_image_index(args.cooccurrence_dir),
        "child_only": build_image_index(args.child_only_dir),
    }

    # ---- Step 1: bulk final_label ----
    for r in rows:
        is_na = r["suggested_label"].startswith(NA_PREFIX)
        if is_na:
            r["final_label"] = ""            # leave N/A for a human
        elif r["source"] == "cooccurrence":
            r["final_label"] = "unsafe"
        elif r["source"] == "child_only":
            r["final_label"] = "safe"
        else:
            r["final_label"] = ""            # unknown source -> don't guess

    # ---- Step 2: flag high-distance co-occurrence exceptions ----
    cooc_dists = sorted(
        float(r["min_centroid_distance_norm"])
        for r in rows
        if r["source"] == "cooccurrence" and r["min_centroid_distance_norm"]
    )
    cutoff = percentile(cooc_dists, FLAG_PERCENTILE)

    flagged = []
    for r in rows:
        r["flagged_for_review"] = "false"
        if (r["source"] == "cooccurrence"
                and r["min_centroid_distance_norm"]
                and cutoff is not None
                and float(r["min_centroid_distance_norm"]) > cutoff):
            r["flagged_for_review"] = "true"
            flagged.append((r["filename"], r["min_centroid_distance_norm"]))

    # ---- rewrite CSV with the new column ----
    fieldnames = ["filename", "source", "suggested_label",
                  "min_centroid_distance_norm", "final_label",
                  "flagged_for_review"]
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ---- Step 3: merge (copy) into labeled folders ----
    safe_dir = os.path.join(LABELED_ROOT, "safe")
    unsafe_dir = os.path.join(LABELED_ROOT, "unsafe")
    os.makedirs(safe_dir, exist_ok=True)
    os.makedirs(unsafe_dir, exist_ok=True)

    counts = {"safe": 0, "unsafe": 0}
    skipped_na = []       # blank final_label (N/A or unknown source)
    skipped_missing = []  # final_label set but source image not found
    collisions = []       # copied under a source-prefixed name

    for r in rows:
        label = r["final_label"]
        if label not in ("safe", "unsafe"):
            skipped_na.append(r)
            continue

        src_path = source_index.get(r["source"], {}).get(r["filename"])
        if not src_path:
            skipped_missing.append(r)
            continue

        dest_dir = safe_dir if label == "safe" else unsafe_dir
        dest_name = r["filename"]
        dest_path = os.path.join(dest_dir, dest_name)
        # collision guard: never silently overwrite a different source's file
        if os.path.exists(_fs(dest_path)):
            dest_name = f"{r['source']}_{r['filename']}"
            dest_path = os.path.join(dest_dir, dest_name)
            collisions.append(dest_name)

        shutil.copy2(_fs(src_path), _fs(dest_path))
        counts[label] += 1

    # ---- Step 4: report ----
    total = len(rows)
    print("==== bulk label + merge report ====")
    print(f"labels.csv rows: {total}")
    print(f"copied -> {unsafe_dir}: {counts['unsafe']}")
    print(f"copied -> {safe_dir}:   {counts['safe']}")
    print(f"total copied: {counts['safe'] + counts['unsafe']}")

    print(f"\nflagged_for_review = true: {len(flagged)} rows")
    print(f"  cutoff: co-occurrence min_centroid_distance_norm > "
          f"{cutoff:.4f} (the {FLAG_PERCENTILE}th percentile of the "
          f"{len(cooc_dists)} co-occurrence distances).")
    print(f"  rationale: this set is close-proximity by design, so the "
          f"most-distant ~{100 - FLAG_PERCENTILE}% are the likeliest genuine "
          f"'actually far apart' exceptions to the bulk-unsafe default -- "
          f"worth a human glance before trusting the label.")
    if flagged:
        print("  flagged filenames (distance):")
        for fn, dist in sorted(flagged, key=lambda x: -float(x[1])):
            print(f"    {dist}  {fn}")

    # full accounting: every row is copied or explicitly skipped
    print(f"\nnot copied (accounted for): {len(skipped_na) + len(skipped_missing)}")
    if skipped_na:
        na_true = [r for r in skipped_na
                   if r["suggested_label"].startswith(NA_PREFIX)]
        other = [r for r in skipped_na
                 if not r["suggested_label"].startswith(NA_PREFIX)]
        print(f"  {len(na_true)} left BLANK for human decision "
              f"(marked '{NA_PREFIX} (missing child or hazard box)'):")
        for r in na_true:
            print(f"    {r['source']}  {r['filename']}")
        if other:
            print(f"  {len(other)} blank for another reason "
                  f"(unexpected source value):")
            for r in other:
                print(f"    source={r['source']!r}  {r['filename']}")
    if skipped_missing:
        print(f"  {len(skipped_missing)} had a final_label but the source "
              f"image was NOT FOUND under the given dataset dir(s):")
        for r in skipped_missing:
            print(f"    {r['source']}  {r['filename']}")

    if collisions:
        print(f"\ncross-source filename collisions handled (prefixed): "
              f"{len(collisions)}")
        for name in collisions:
            print(f"    {name}")

    accounted = counts['safe'] + counts['unsafe'] + len(skipped_na) + len(skipped_missing)
    print(f"\nreconciliation: {counts['safe'] + counts['unsafe']} copied + "
          f"{len(skipped_na) + len(skipped_missing)} skipped = {accounted} "
          f"of {total} rows {'(OK)' if accounted == total else '(MISMATCH!)'}")


if __name__ == "__main__":
    main()
