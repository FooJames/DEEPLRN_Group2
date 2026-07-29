#!/usr/bin/env python3
"""
make_cooccurrence_eval.py — build the child+hazard evaluation set (§8.1).

The existing labelled set cannot test the fusion layer: every "safe" image
has NO hazard, so the label tracks hazard *presence*, not proximity, and a
distance threshold calibrated on it degenerates to "hazard detected →
unsafe". What is missing is the case where a child and a hazard are BOTH
present but far apart, i.e. safe.

This composites that case: a hazard crop is pasted onto a real child image
at a controlled separation, so the geometry is exact by construction.

    python scripts/make_cooccurrence_eval.py --n 200

GROUND TRUTH IS *NOT* THE METRIC UNDER TEST — this matters.
    Phase 3c compares two candidate predictors: centroid-to-centroid
    distance and nearest-edge distance, both normalised by the image
    diagonal. If the label were derived from either one, that predictor
    would win by construction and the ablation would be a tautology.

    So the label uses an independent criterion: REACHABILITY, measured in
    units of the child's own body size —

        reach_ratio = (edge gap between boxes) / (child box height)
        unsafe  <=>  reach_ratio <= --reach (default 0.5, ~arm's length)

    That is scale-relative (normalised by the child), while both predictors
    are image-diagonal-normalised. Neither recovers it automatically: a
    small child far away and a large child close by can share a centroid
    distance yet differ in reachability. The ablation is therefore a real
    test of which predictor better tracks physical risk.

SAMPLING: reach_ratio is drawn uniformly over --span (default 0.0-1.5) so
the set brackets the decision boundary and includes genuinely ambiguous
cases. A set where every safe image sits far beyond every unsafe one would
be trivially separable and would prove nothing.

SPLITS: grouped by source child image, so no child appears in both val and
test. Calibrate on val; touch test once, at the end.

LIMITATION TO STATE IN THE WRITE-UP: composited images test the pipeline's
mechanics (detection + geometric fusion) under known geometry. They do not
establish that proximity predicts real-world danger — that would need real
images with independent human judgement.
"""

import argparse
import csv
import math
import os
import random

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_boxes(label_path):
    """YOLO label file -> [(cls, xc, yc, w, h)] in normalised units."""
    out = []
    if not os.path.isfile(label_path):
        return out
    with open(label_path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) == 5:
                out.append((int(p[0]), *(float(v) for v in p[1:])))
    return out


def index_dataset(root):
    """[(image_path, label_path)] across whichever splits exist."""
    pairs = []
    for sp in SPLITS:
        img_dir = os.path.join(root, sp, "images")
        lbl_dir = os.path.join(root, sp, "labels")
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                lbl = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
                if os.path.isfile(lbl):
                    pairs.append((os.path.join(img_dir, fn), lbl))
    return pairs


def rect_gap(a, b):
    """Nearest-edge distance between two pixel boxes (0 if they overlap).
    Boxes are (x1, y1, x2, y2)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


def to_pixels(box, W, H):
    _, xc, yc, w, h = box
    return ((xc - w / 2) * W, (yc - h / 2) * H,
            (xc + w / 2) * W, (yc + h / 2) * H)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child-root", default=os.path.join("data", "child"))
    ap.add_argument("--hazard-root", default=os.path.join("data", "hazard"))
    ap.add_argument("--out", default=os.path.join("data", "cooccurrence"))
    ap.add_argument("--n", type=int, default=200, help="images to generate")
    ap.add_argument("--reach", type=float, default=0.5,
                    help="unsafe if gap <= reach * child height (arm's length)")
    ap.add_argument("--span", type=float, nargs=2, default=[0.0, 1.5],
                    help="range of reach_ratio to sample across")
    ap.add_argument("--haz-scale", type=float, nargs=2, default=[0.12, 0.35],
                    help="hazard height as a fraction of child box height")
    ap.add_argument("--val-frac", type=float, default=0.6,
                    help="fraction of child images used for the val split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image

    rng = random.Random(args.seed)
    children = index_dataset(args.child_root)
    hazards = index_dataset(args.hazard_root)
    if not children or not hazards:
        raise SystemExit(f"need both datasets: {args.child_root}, {args.hazard_root}")
    print(f"source pools: {len(children)} child images, {len(hazards)} hazard images")

    haz_names = []
    yml = os.path.join(args.hazard_root, "data.yaml")
    if os.path.isfile(yml):
        import yaml
        haz_names = (yaml.safe_load(open(yml, encoding="utf-8")) or {}).get("names", [])

    # split by SOURCE CHILD IMAGE so no child leaks across val/test
    order = list(range(len(children)))
    rng.shuffle(order)
    cut = int(len(order) * args.val_frac)
    split_of = {i: ("val" if k < cut else "test") for k, i in enumerate(order)}

    for sp in ("val", "test"):
        os.makedirs(os.path.join(args.out, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, sp, "labels"), exist_ok=True)

    rows, made, attempts = [], 0, 0
    while made < args.n and attempts < args.n * 40:
        attempts += 1
        ci = rng.randrange(len(children))
        cimg_p, clbl_p = children[ci]
        cboxes = [b for b in load_boxes(clbl_p)]
        if not cboxes:
            continue
        try:
            cim = Image.open(cimg_p).convert("RGB")
        except Exception:
            continue
        W, H = cim.size
        # largest child box — the most reliable subject in the frame
        cb = max(cboxes, key=lambda b: b[3] * b[4])
        cx1, cy1, cx2, cy2 = to_pixels(cb, W, H)
        ch = cy2 - cy1
        if ch < 40:                      # too small to place around sensibly
            continue

        hi = rng.randrange(len(hazards))
        himg_p, hlbl_p = hazards[hi]
        hboxes = load_boxes(hlbl_p)
        if not hboxes:
            continue
        hb = max(hboxes, key=lambda b: b[3] * b[4])
        try:
            him = Image.open(himg_p).convert("RGB")
        except Exception:
            continue
        HW, HH = him.size
        hx1, hy1, hx2, hy2 = to_pixels(hb, HW, HH)
        if hx2 - hx1 < 8 or hy2 - hy1 < 8:
            continue
        crop = him.crop((int(hx1), int(hy1), int(hx2), int(hy2)))

        # scale the hazard to a plausible size relative to the child
        target_h = ch * rng.uniform(*args.haz_scale)
        scale = target_h / crop.height
        nw, nh = max(int(crop.width * scale), 6), max(int(crop.height * scale), 6)
        crop = crop.resize((nw, nh))

        # place it at a sampled edge-gap, in units of child height
        want_ratio = rng.uniform(*args.span)
        gap = want_ratio * ch
        ccx, ccy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
        placed = None
        for _ in range(24):
            th = rng.uniform(0, 2 * math.pi)
            # approximate radial offset; the true gap is measured after placing
            r = math.hypot((cx2 - cx1) / 2, ch / 2) + gap + math.hypot(nw, nh) / 2
            px, py = ccx + r * math.cos(th) - nw / 2, ccy + r * math.sin(th) - nh / 2
            if 0 <= px <= W - nw and 0 <= py <= H - nh:
                placed = (px, py)
                break
        if placed is None:
            continue
        px, py = placed
        out_im = cim.copy()
        out_im.paste(crop, (int(px), int(py)))

        hbox_px = (px, py, px + nw, py + nh)
        diag = math.hypot(W, H)
        hcx, hcy = px + nw / 2, py + nh / 2

        # Ground truth must match what the fusion layer does: it takes the
        # MINIMUM over all child-hazard pairs. Scoring against only the box we
        # placed around would disagree with the rule whenever a different
        # child in the frame is nearer, for reasons unrelated to the distance
        # metric under test.
        cand = []
        for b in cboxes:
            bx = to_pixels(b, W, H)
            cand.append((rect_gap(bx, hbox_px), bx))
        edge, cbox_px = min(cand, key=lambda t: t[0])
        ref_h = cbox_px[3] - cbox_px[1]          # height of the CLOSEST child
        centroid = min(math.hypot((bx[0] + bx[2]) / 2 - hcx,
                                  (bx[1] + bx[3]) / 2 - hcy) for _, bx in cand)
        reach_ratio = edge / ref_h
        label = "unsafe" if reach_ratio <= args.reach else "safe"

        sp = split_of[ci]
        stem = f"cooc_{made:05d}"
        out_im.save(os.path.join(args.out, sp, "images", stem + ".jpg"), quality=92)
        # reference boxes (0=child, 1=hazard) for diagnosing detection vs fusion
        with open(os.path.join(args.out, sp, "labels", stem + ".txt"),
                  "w", encoding="utf-8") as fh:
            for cls, (x1, y1, x2, y2) in ((0, cbox_px), (1, hbox_px)):
                fh.write(f"{cls} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                         f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")

        rows.append({
            "filename": stem + ".jpg", "split": sp, "label": label,
            "reach_ratio": round(reach_ratio, 4),
            "centroid_dist_norm": round(centroid / diag, 4),
            "edge_dist_norm": round(edge / diag, 4),
            "hazard_class": (haz_names[hb[0]] if hb[0] < len(haz_names) else hb[0]),
            "n_children": len(cboxes),
            "child_src": os.path.basename(cimg_p),
            "hazard_src": os.path.basename(himg_p),
        })
        made += 1

    out_csv = os.path.join(args.out, "cooccurrence_labels.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_un = sum(r["label"] == "unsafe" for r in rows)
    print(f"\ngenerated {made} images ({attempts} attempts) -> {args.out}")
    print(f"  unsafe {n_un}  |  safe {made - n_un}")
    for sp in ("val", "test"):
        s = [r for r in rows if r["split"] == sp]
        u = sum(r["label"] == "unsafe" for r in s)
        print(f"  {sp:4}: {len(s):4d}  (unsafe {u}, safe {len(s)-u})")
    print(f"  labels -> {out_csv}")
    print(f"\nground truth = reachability (gap <= {args.reach} x child height); "
          f"predictors under test are centroid_dist_norm and edge_dist_norm.")


if __name__ == "__main__":
    main()
