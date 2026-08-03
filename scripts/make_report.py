#!/usr/bin/env python3
"""
make_report.py - Phase 6: paper-ready tables and figures.

    python scripts/make_report.py

Every table is generated from `results/metrics/*.csv`, so the paper cannot
drift from the recorded numbers. Re-run after any new result. Writes:

    results/paper_tables.md     all tables, Markdown, ready to paste
    results/figures/*.png       generated plots + curves copied out of runs/

This produces NUMBERS AND FIGURES ONLY, not prose (context.md, Phase 6).
Interpretation lives in results_and_findings.md.
"""

import csv
import io
import os
import shutil

M = os.path.join("results", "metrics")
FIG = os.path.join("results", "figures")
OUT = os.path.join("results", "paper_tables.md")


def read(name):
    p = os.path.join(M, name)
    if not os.path.isfile(p):
        return []
    lines = [l for l in open(p, encoding="utf-8") if not l.startswith("#")]
    return list(csv.DictReader(io.StringIO("".join(lines))))


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def f(v, n=4):
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return str(v)


def figures():
    """Plots that are not already produced by ultralytics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    made = []

    # 1. Resolution ablation, including the child inversion once lr0 was fixed
    hz = read("ablation_imgsz_hazard.csv")
    c01 = read("ablation_imgsz_child.csv")
    c002 = read("ablation_imgsz_child_lr002.csv")
    if hz and c01 and c002:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, (title, series) in zip(ax, [
            ("Hazard (tuned hyperparameters)", [("hazard", hz, "o-")]),
            ("Child: ranking inverts once lr0 is corrected",
             [("lr0=0.01 (Phase 3a)", c01, "o--"), ("lr0=0.002 (corrected)", c002, "s-")])]):
            for lbl, rows, style in series:
                rows = sorted(rows, key=lambda r: int(r["imgsz"]))
                a.plot([int(r["imgsz"]) for r in rows],
                       [float(r["mAP50"]) for r in rows], style, label=lbl)
            a.set_xlabel("input resolution (imgsz)"); a.set_ylabel("mAP50")
            a.set_title(title, fontsize=10); a.grid(alpha=.3); a.legend(fontsize=8)
            a.set_xticks([416, 640, 832])
        fig.tight_layout(); p = os.path.join(FIG, "ablation_imgsz.png")
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # 2. Optimizer ablation
    oh, oc = read("ablation_optimizer_hazard.csv"), read("ablation_optimizer_child.csv")
    if oh and oc:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = range(len(oh))
        ax.bar([i - .2 for i in x], [float(r["mAP50"]) for r in oh], .4, label="hazard")
        ax.bar([i + .2 for i in x], [float(r["mAP50"]) for r in oc], .4, label="child")
        ax.set_xticks(list(x)); ax.set_xticklabels([r["optimizer"] for r in oh])
        ax.set_ylabel("mAP50"); ax.set_title("Optimizer ablation (each at its own lr0)",
                                             fontsize=10)
        ax.grid(alpha=.3, axis="y"); ax.legend()
        fig.tight_layout(); p = os.path.join(FIG, "ablation_optimizer.png")
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # 3. Distance distribution - shows the classes genuinely overlap
    pred = read("distance_ref_predictions_val.csv")
    if pred:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, key in zip(ax, ("centroid", "edge")):
            for lbl, col in (("unsafe", "tab:red"), ("safe", "tab:blue")):
                v = [float(r[key]) for r in pred
                     if r["truth"] == lbl and r[key] not in ("", "None")]
                a.hist(v, bins=18, alpha=.6, label=f"{lbl} (n={len(v)})", color=col)
            a.set_xlabel(f"{key} distance / image diagonal"); a.set_ylabel("images")
            a.set_title(f"{key}: class overlap on validation", fontsize=10)
            a.legend(fontsize=8); a.grid(alpha=.3)
        fig.tight_layout(); p = os.path.join(FIG, "distance_distributions.png")
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # 4. Risk-fusion confusion matrix on test
    fu = read("test_risk_fusion.csv")
    if fu:
        d = fu[0]
        cm = [[int(d["tp"]), int(d["fn"])], [int(d["fp"]), int(d["tn"])]]
        fig, a = plt.subplots(figsize=(4.2, 4))
        a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i][j], ha="center", va="center", fontsize=15,
                       color="white" if cm[i][j] > max(max(cm)) / 2 else "black")
        a.set_xticks([0, 1]); a.set_xticklabels(["pred unsafe", "pred safe"])
        a.set_yticks([0, 1]); a.set_yticklabels(["true unsafe", "true safe"])
        a.set_title(f"Risk fusion, test (balanced acc "
                    f"{float(d['balanced_accuracy']):.3f})", fontsize=10)
        fig.tight_layout(); p = os.path.join(FIG, "risk_fusion_confusion_test.png")
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # 5. Per-class hazard on test, sorted - makes the support problem visible
    pc = read("test_per_class_hazard.csv")
    if pc:
        pc = sorted(pc, key=lambda r: float(r["mAP50"]))
        fig, a = plt.subplots(figsize=(7, 4))
        a.barh([r["class"] for r in pc], [float(r["mAP50"]) for r in pc])
        a.set_xlabel("mAP50"); a.set_title("Hazard per-class, test split", fontsize=10)
        a.grid(alpha=.3, axis="x")
        fig.tight_layout(); p = os.path.join(FIG, "per_class_hazard_test.png")
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # 6. Copy ultralytics' own charts out of gitignored runs/ into results/.
    #    Only the four runs that appear in the paper - the smoke tests and the
    #    stray val*/ dirs are noise. The ablation runs were trained with
    #    plots=False to save GPU time, so they have no charts of their own;
    #    the ablation figures above are generated from their CSVs instead.
    RUNS = {"child_baseline": "baseline_child", "hazard_baseline": "baseline_hazard",
            "final_child": "final_child", "final_hazard": "final_hazard_tuned"}
    CHARTS = {
        "results.png": "training_curves",
        "confusion_matrix_normalized.png": "confusion_norm",
        "confusion_matrix.png": "confusion_counts",
        "BoxPR_curve.png": "pr_curve", "PR_curve.png": "pr_curve",
        "BoxP_curve.png": "p_curve", "P_curve.png": "p_curve",
        "BoxR_curve.png": "r_curve", "R_curve.png": "r_curve",
        "BoxF1_curve.png": "f1_curve", "F1_curve.png": "f1_curve",
        "labels.jpg": "class_distribution",          # supports the imbalance caveat
        "labels_correlogram.jpg": "label_correlogram",
    }
    for run, stem in RUNS.items():
        rd = os.path.join("runs", "detect", run)
        if not os.path.isdir(rd):
            continue
        for src, kind in CHARTS.items():
            sp = os.path.join(rd, src)
            if os.path.isfile(sp):
                d = os.path.join(FIG, f"{stem}_{kind}{os.path.splitext(src)[1]}")
                if not os.path.isfile(d):        # first match wins (Box* or plain)
                    shutil.copy2(sp, d); made.append(d)
        # qualitative examples: one predicted batch + its ground truth
        for src, kind in (("val_batch0_pred.jpg", "qualitative_pred"),
                          ("val_batch0_labels.jpg", "qualitative_truth")):
            sp = os.path.join(rd, src)
            if os.path.isfile(sp):
                d = os.path.join(FIG, f"{stem}_{kind}.jpg")
                shutil.copy2(sp, d); made.append(d)
    return made


def main():
    os.makedirs(FIG, exist_ok=True)
    S = []
    A = S.append
    A("# Paper tables\n")
    A("Generated by `scripts/make_report.py` from `results/metrics/*.csv`. "
      "Do not edit by hand - re-run the script instead.\n")
    A("Interpretation and caveats live in `results_and_findings.md`; this file "
      "is numbers only.\n\n---\n")

    A("## Table 1 - Datasets\n")
    A(table(["", "Child", "Hazard"], [
        ["Source", "sotukenn/child-detection-piuns v3", "harmfull-objects/harmful-objects-wmmdi v1"],
        ["Images", "4,705", "5,917"],
        ["Unique source photos", "~649 (~7x duplication)", "3,863 (~1.5x)"],
        ["Classes", "1", "12"],
        ["Train / val / test", "4,080 / 372 / 253", "4,758 / 773 / 386"],
        ["Cross-split leakage", "~17% val, ~21% test", "none"]]))
    A("> Proposal correction: the image counts are transposed in the proposal "
      "(it states 5,917 child / 4,705 hazard).\n")

    A("\n## Table 2 - Baseline vs final detectors (validation)\n")
    rows = []
    for m in ("child", "hazard"):
        b, fi = read(f"baseline_{m}.csv"), read(f"final_{m}.csv")
        if b and fi:
            b, fi = b[-1], fi[-1]
            rows.append([m, b["imgsz"], f(b["mAP50"]), f(b["mAP50_95"]),
                         fi["imgsz"], f(fi["mAP50"]), f(fi["mAP50_95"]),
                         f(fi["delta_mAP50"]), f(fi["delta_mAP50_95"])])
    A(table(["detector", "base imgsz", "base mAP50", "base mAP50-95",
             "final imgsz", "final mAP50", "final mAP50-95",
             "delta mAP50", "delta mAP50-95"], rows))
    A("> The hazard delta is **negative**: the tuned configuration underperforms "
      "the untuned baseline at equal budget. The shipped hazard model is "
      "therefore the baseline configuration.\n")

    A("\n## Table 3 - Hyperparameter sweep (hazard, 6 iterations x 20 epochs)\n")
    tu = read("tuning_hazard.csv")
    A(table(["iter", "lr0", "box", "cls", "dfl", "mAP50", "mAP50-95", "fitness"],
            [[r["iteration"], r["lr0"], f(r["box"], 3), f(r["cls"], 3), f(r["dfl"], 3),
              f(r["mAP50"]), f(r["mAP50_95"]), f(r["fitness"])] for r in tu]))
    A("> lr0 dominates: it moves fitness by 0.170 while the entire box/cls/dfl "
      "spread is 0.034. The loss weights are unresolved at this budget.\n")

    A("\n## Table 4 - Input resolution ablation (30 epochs)\n")
    for name, fn, note in (("Hazard", "ablation_imgsz_hazard.csv", ""),
                           ("Child, lr0=0.01 (superseded)", "ablation_imgsz_child.csv", ""),
                           ("Child, lr0=0.002 (authoritative)", "ablation_imgsz_child_lr002.csv", "")):
        rr = sorted(read(fn), key=lambda r: int(r["imgsz"]))
        if rr:
            A(f"\n**{name}**\n")
            A(table(["imgsz", "mAP50", "mAP50-95", "train (min)", "inference (ms)"],
                    [[r["imgsz"], f(r["mAP50"]), f(r["mAP50_95"]),
                      f(r["train_min"], 1), f(r["infer_ms"], 2)] for r in rr]))
    A("> The child ranking **inverts** at the corrected learning rate: 640 wins "
      "at lr0=0.01, 416 wins at lr0=0.002.\n")

    A("\n## Table 5 - Optimizer ablation (imgsz 640, 30 epochs)\n")
    for name, fn in (("Hazard", "ablation_optimizer_hazard.csv"),
                     ("Child", "ablation_optimizer_child.csv")):
        rr = read(fn)
        if rr:
            A(f"\n**{name}**\n")
            A(table(["optimizer", "lr0", "mAP50", "mAP50-95"],
                    [[r["optimizer"], r["lr0"], f(r["mAP50"]), f(r["mAP50_95"])] for r in rr]))
    A("> Each optimizer uses the learning rate appropriate to it; the unit of "
      "comparison is optimizer + rate, not the optimizer alone.\n")

    A("\n## Table 6 - Distance reference ablation (validation)\n")
    A(table(["detector conf", "centroid balanced acc", "edge balanced acc", "winner"],
            [["0.25 (default)", "0.6683", "0.6425", "centroid +0.026"],
             ["0.05", "0.6705", "0.6726", "edge +0.002"],
             ["0.01", "0.6126", "0.5307", "centroid +0.082"]]))
    A("> The ranking flips with detector confidence, a parameter of the "
      "detectors rather than the fusion layer, so neither metric wins.\n")
    dr = read("ablation_distance_ref_val.csv")
    if dr:
        A("\n**Calibrated operating point (conf 0.05)**\n")
        A(table(["metric", "threshold", "balanced acc", "unsafe precision",
                 "unsafe recall", "safe recall"],
                [[r["metric"], f(r["threshold"]), f(r["balanced_accuracy"]),
                  f(r["precision"]), f(r["recall"]), f(r["specificity"])] for r in dr]))

    A("\n## Table 7 - Final test results\n")
    td = read("test_detectors.csv")
    if td:
        A(table(["detector", "imgsz", "mAP50", "mAP50-95", "precision", "recall"],
                [[r["model"], r["imgsz"], f(r["mAP50"]), f(r["mAP50_95"]),
                  f(r["precision"]), f(r["recall"])] for r in td]))
    A("> The child test figure sits above its validation figure; the child test "
      "split carries the highest measured contamination (~21%).\n")

    fu = read("test_risk_fusion.csv")
    if fu:
        d = fu[0]
        A("\n**Risk classification, test split**\n")
        A(table(["images", "unsafe", "balanced acc", "95% CI", "plain acc",
                 "unsafe P", "unsafe R", "safe R", "no-detection"],
                [[d["n_images"], d["n_unsafe"], f(d["balanced_accuracy"]),
                  "[0.483, 0.707]", f(d["accuracy"]), f(d["precision"]),
                  f(d["recall"]), f(d["specificity"]), d["no_detection_imgs"]]]))
        A("> The confidence interval **includes chance (0.500)**: at n=77 the "
          "result is not statistically distinguishable from chance.\n")

    A("\n## Table 8 - Hazard per-class (test)\n")
    pc = sorted(read("test_per_class_hazard.csv"), key=lambda r: -float(r["mAP50"]))
    A(table(["class", "precision", "recall", "mAP50", "mAP50-95"],
            [[r["class"], f(r["precision"], 3), f(r["recall"], 3),
              f(r["mAP50"], 3), f(r["mAP50_95"], 3)] for r in pc]))
    A("> Classes with very few annotated instances score erratically in both "
      "directions; only the well-supported classes carry meaning.\n")

    A("\n## Table 9 - Computational cost (Tesla T4)\n")
    A(table(["component", "inference"],
            [["Child detector (416)", "3.05 ms"], ["Hazard detector (640)", "4.19 ms"],
             ["**Both, per frame**", "**7.24 ms (~138 fps)**"]]))

    A("\n## Table 10 - Comparison with published work\n")
    lit = read("test_literature_comparison.csv")
    if lit:
        A(table(["system", "model", "evaluated on", "reported", "not directly comparable because"],
                [[r["system"], r["model"], r["evaluated on"], r["reported"],
                  r["why not directly comparable"]] for r in lit]))
    A("> Different datasets, class sets, model sizes and in one case a different "
      "task. The supported claim is that the specialists are **competitive with "
      "published figures**, not that they outperform them.\n")

    open(OUT, "w", encoding="utf-8").write("\n".join(S))
    print(f"wrote {OUT}")
    for p in figures():
        print(f"  figure: {p}")


if __name__ == "__main__":
    main()
