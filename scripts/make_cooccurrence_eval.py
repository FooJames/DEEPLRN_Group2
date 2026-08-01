#!/usr/bin/env python3
"""
make_cooccurrence_eval.py -- build the child+hazard evaluation set (§8.1).

The existing labelled set cannot test the fusion layer: every "safe" image
has NO hazard, so the label tracks hazard *presence*, not proximity, and a
distance threshold calibrated on it degenerates to "hazard detected ->
unsafe". What is missing is the case where a child and a hazard are BOTH
present but far apart, i.e. safe.

This composites that case: a hazard crop is pasted onto a real child image
at a controlled separation, so the geometry is exact by construction.

    python scripts/make_cooccurrence_eval.py --n 200

GROUND TRUTH IS *NOT* THE METRIC UNDER TEST -- this matters.
    Phase 3c compares two candidate predictors: centroid-to-centroid
    distance and nearest-edge distance, both normalised by the image
    diagonal. If the label were derived from either one, that predictor
    would win by construction and the ablation would be a tautology.

    So the label uses an independent criterion: REACHABILITY, measured in
    units of the child's own body size --

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

BACKGROUND SELECTION: child backgrounds are drawn only from the child
dataset's HELD-OUT splits (so composites do not reuse images the child
detector trained on) and are screened for salt-and-pepper noise, since the
child export carries 5% noise while hazard crops carry none -- pasting a
clean crop onto a speckled background would be visually inconsistent and
would make the hazard easier to detect than it should be.

LIMITATIONS TO STATE IN THE WRITE-UP:
  - Composited images test the pipeline's mechanics (detection + geometric
    fusion) under known geometry. They do not establish that proximity
    predicts real-world danger -- that needs real images with independent
    human judgement.
  - The child dataset annotates a head/face in roughly 40% of images and a
    full body in 57%. Because the reachability label is relative to box
    height, "arm's length" is not a consistent physical distance across the
    set. --max-aspect 0.7 restricts to body-like boxes if that matters.
"""

import argparse
import csv
import math
import os
import random
import re

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


def index_dataset(root, splits=SPLITS):
    """[(image_path, label_path)] across the requested splits."""
    pairs = []
    for sp in splits:
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


def sp_noise(path):
    """Salt-and-pepper level: fraction of isolated extreme pixels.

    The child dataset ships with noise applied to 5% of pixels while hazard
    crops are clean, so pasting a clean crop onto a speckled background makes
    the composite visually inconsistent and the hazard easier to spot than it
    should be. Screening backgrounds by this keeps the two consistent.
    """
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"), dtype=np.int16)
    if min(a.shape) < 20:
        return 1.0
    st = np.stack([np.roll(np.roll(a, dy, 0), dx, 1)
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    dev = np.abs(a - np.median(st, axis=0))
    return float((((a < 15) | (a > 240)) & (dev > 60)).mean())


def whitespace(path):
    """Fraction of near-white pixels. The child dataset contains collages and
    infographics whose panels are separated by white gutters; a hazard pasted
    into a different panel from the child is at a physically meaningless
    "distance", so such backgrounds must be rejected."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L").resize((128, 128)))
    return float((a > 235).mean())


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
    ap.add_argument("--max-noise", type=float, default=0.002,
                    help="reject child backgrounds whose salt-and-pepper level "
                         "exceeds this (0 disables). The child dataset was "
                         "exported with 5%% noise; hazard crops have none.")
    ap.add_argument("--child-splits", nargs="+", default=["valid", "test"],
                    choices=["train", "valid", "test"],
                    help="which child splits to draw backgrounds from. Defaults "
                         "to the held-out splits so composites do not reuse "
                         "images the child detector was trained on.")
    ap.add_argument("--max-aspect", type=float, default=0.7,
                    help="reject child boxes wider than this (w/h). The child "
                         "dataset annotates a head/face in ~40%% of images and a "
                         "full body in ~57%%, and the reachability label is "
                         "relative to box height, so mixing them makes "
                         "\"arm's length\" mean different physical distances. "
                         "Set 0.7 to keep body-like boxes only.")
    ap.add_argument("--min-box-area", type=float, default=0.03,
                    help="reject child boxes smaller than this fraction of the "
                         "frame; tiny boxes are usually thumbnails inside a "
                         "collage rather than the subject of the scene")
    ap.add_argument("--max-whitespace", type=float, default=0.15,
                    help="reject backgrounds with more near-white area than "
                         "this (collage/infographic detector)")
    ap.add_argument("--safe-margin", type=float, default=None,
                    help="require reach_ratio >= this to label SAFE, skipping "
                         "the band between --reach and this value. Without a "
                         "margin, an image just past the boundary is labelled "
                         "safe while the hazard is still nearly within reach. "
                         "Defaults to 2x --reach.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.safe_margin is None:
        args.safe_margin = args.reach * 2
    if args.safe_margin < args.reach:
        raise SystemExit("--safe-margin must be >= --reach")

    from PIL import Image

    rng = random.Random(args.seed)
    children = index_dataset(args.child_root, args.child_splits)
    hazards = index_dataset(args.hazard_root)
    if not children or not hazards:
        raise SystemExit(f"need both datasets: {args.child_root}, {args.hazard_root}")
    print(f"source pools: {len(children)} child images "
          f"(splits={args.child_splits}), {len(hazards)} hazard images")
    if args.max_noise:
        print(f"screening child backgrounds at noise <= {args.max_noise}")

    haz_names = []
    yml = os.path.join(args.hazard_root, "data.yaml")
    if os.path.isfile(yml):
        import yaml
        haz_names = (yaml.safe_load(open(yml, encoding="utf-8")) or {}).get("names", [])

    # Split by SOURCE PHOTOGRAPH, not by file. The child export contains
    # several near-identical copies of each source (Roboflow names them
    # <source>_jpg.rf.<hash>.jpg), so keying on the file would scatter copies
    # of one child across val and test -- reproducing exactly the train/val
    # leakage found in the child dataset itself (§2.2).
    def src_of(path):
        return re.split(r"\.rf\.", os.path.basename(path))[0]

    sources = sorted({src_of(f) for f, _ in children})
    rng.shuffle(sources)
    cut = int(len(sources) * args.val_frac)
    split_of_src = {s: ("val" if k < cut else "test") for k, s in enumerate(sources)}
    print(f"  {len(sources)} unique source photographs -> "
          f"{cut} val / {len(sources) - cut} test")

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
        if args.max_noise and sp_noise(cimg_p) > args.max_noise:
            continue                     # speckled background; would not match the crop
        try:
            cim = Image.open(cimg_p).convert("RGB")
        except Exception:
            continue
        W, H = cim.size
        # largest child box -- the most reliable subject in the frame
        cb = max(cboxes, key=lambda b: b[3] * b[4])
        if cb[4] > 0 and cb[3] / cb[4] > args.max_aspect:
            continue                     # head/face box: "0.5x height" would be ~10cm
        if cb[3] * cb[4] < args.min_box_area:
            continue                     # thumbnail inside a collage, not the subject
        # A hazard can only be placed --safe-margin child-heights away if the
        # child is small enough for that gap to fit in the frame. Requiring it
        # for EVERY background keeps the same children eligible for both
        # labels; otherwise "safe" images would systematically contain smaller
        # children than "unsafe" ones, and that size difference - not the
        # distance - could drive the result.
        if cb[4] > 1.0 / (1.0 + args.safe_margin):
            continue
        if args.max_whitespace and whitespace(cimg_p) > args.max_whitespace:
            continue                     # collage/infographic; cross-panel gap is meaningless
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
        # Draw from the UNSAFE band or the SAFE band, never the ambiguous
        # middle: a hazard just past the reach threshold is still practically
        # within reach, so labelling it "safe" would be wrong.
        if rng.random() < 0.5:
            want_ratio = rng.uniform(0.0, args.reach)
        else:
            want_ratio = rng.uniform(args.safe_margin, max(args.span[1],
                                                           args.safe_margin * 1.6))
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
        if reach_ratio <= args.reach:
            label = "unsafe"
        elif reach_ratio >= args.safe_margin:
            label = "safe"
        else:
            continue            # ambiguous band - not confidently labellable

        sp = split_of_src[src_of(cimg_p)]
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
            "child_split": os.path.basename(
                os.path.dirname(os.path.dirname(cimg_p))),
            "child_src": src_of(cimg_p),
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
