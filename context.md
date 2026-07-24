# Context: Contextual Child Safety Risk Detector

## What this project is

A dual-model computer vision pipeline that detects (1) children and (2)
household hazards in a single image frame using two **independently
trained** YOLOv8n models, then fuses their outputs with a
distance-based rule to classify the scene as **Safe** or **Unsafe**.

Course: Deep Learning (DEEPLRN), Group 2 S08, De La Salle University.
Team: Stephen Co, James Foo, Simon Luis Ambata, Nathaniel Tolentino.

## Research question

Does a fully independent, dual-specialized detector design (one model
per task) match or exceed the performance of:
- a single unified multi-class model (Ahmad et al., 2025 — child +
  hazards trained jointly, ~90% mAP), or
- a hybrid specialist-plus-general design (AlMhdawi et al., 2026 —
  one purpose-trained detector + one COCO-pretrained detector)?

The hypothesis, grounded in Weighted Boxes Fusion (Solovyev et al.,
2021), is that fusing at the **output level** (post-hoc, via distance)
rather than the **training level** (joint multi-class training) avoids
class-imbalance and shared-capacity issues, and is sufficient without
requiring joint training.

## Architecture (see proposal Figure 1)

```
Input Frame
    |
    +-----------------+------------------+
    |                                    |
Child Detector                    Hazard Detector
YOLOv8n, fine-tuned                YOLOv8n, fine-tuned
single class: "child"              12 classes (collapsed to
                                    one "hazard" label at fusion,
                                    per-class name kept for display)
    |                                    |
    +-----------------+------------------+
                       |
                Risk-Fusion Layer
   Normalized Euclidean distance between every detected
   child-box centroid and every detected hazard-box centroid,
   normalized by image diagonal. Take the minimum over all
   pairs. Compare against a calibrated threshold.
                       |
                    Output
        Safe / Unsafe label + both detectors' boxes overlaid
```

### Hazard classes (12)
Axe, Chainsaw, Chisel, Coin, Drink, Dumbbell, Fork, Screwdriver,
Stapler, Knife, Hammer, Scissors.

### Risk-fusion logic
1. If either detector returns zero detections (no child OR no hazard)
   → label **Safe**.
2. If both return ≥1 detection → compute, for every child-hazard pair:

   `d_norm = euclidean_distance(child_centroid, hazard_centroid) / image_diagonal`

3. Take `min(d_norm)` across all pairs in the frame.
4. If `min(d_norm)` is below the calibrated threshold → **Unsafe**,
   else **Safe**.
5. Reference threshold from Ahmad et al. (2025): **700 pixels**
   (un-normalized) — use as a starting point, then calibrate on our
   own validation set once images/resolutions differ.

There is no learned model in the fusion layer — it's a threshold rule
on geometry, by design (see Solovyev et al. justification above).

## Datasets

Both from Roboflow Universe. Download via the Roboflow API/CLI (need
a Roboflow API key) or manual export in YOLOv8 (`.txt` label per
image, `data.yaml`) format.

1. **Child detection dataset**
   - URL: https://universe.roboflow.com/sotukenn/child-detection-piuns
   - 4,705 images, single class (`child`) — Roboflow v3
   - Pre-defined split: 87% train / 8% val / 5% test — **keep the
     creator's split**, do not reshuffle.

2. **Harmful objects dataset**
   - URL: https://universe.roboflow.com/harmfull-objects/harmful-objects-wmmdi
   - 5,917 images, 12 hazard classes (listed above) — Roboflow v1
   - Pre-defined split: 80% train / 13% val / 7% test — **keep the
     creator's split**.

Each model is trained, validated, and tested **entirely on its own
dataset** — no shared backbone, no joint training, no mixing of the
two label sets.

## Training setup

- Framework: `ultralytics` (YOLOv8), built on PyTorch.
- Model: YOLOv8n (nano) for both detectors.
- Compute: Google Colab, T4 GPU.
- Default epochs: 100 (adjust based on convergence / time budget).
- Hyperparameters to tune (grid or Bayesian sweep, per model):
  - `lr0` (initial learning rate)
  - `box` (box loss weight)
  - `cls` (classification loss weight)
  - `dfl` (distribution focal loss weight)

## Ablation studies

Run these on **both** detectors unless noted:

1. **Input resolution (`imgsz`)** — compare a small set of resolutions
   (e.g. 416 / 640 / 832) to see if higher resolution meaningfully
   improves precision, since this is a fixed pipeline decision, not a
   trained parameter.
2. **Optimizer** — compare SGD vs. Adam (and AdamW if time allows) to
   pick the optimizer used for the final runs.
3. **Object detection capability** — compare each specialist's
   detection performance (mAP) against baselines (Ahmad et al.,
   AlMhdawi et al.) to confirm the specialists are competitive before
   trusting the fusion layer's output.
4. **Distance reference point: centroid vs. bounding-box edge** —
   compare using box centroids (current design) vs. nearest-point/edge
   distance between boxes for risk classification. Hypothesis: edge
   distance may better reflect true proximity risk than centroid
   distance in some configurations (e.g. large boxes that are close at
   the edges but far at centroids).

## Evaluation metrics

- **mAP (mean Average Precision)** — primary metric for both
  detectors individually, especially the 12-class hazard detector.
- **Risk classification accuracy** — Safe/Unsafe label accuracy against
  a manually labeled test set (child+hazard co-occurring frames).
- **Euclidean distance** formula used in the fusion layer:
  `d = sqrt((x2-x1)^2 + (y2-y1)^2)` on centroid coordinates.

## Baselines to compare against

- Ahmad et al. (2025) — single YOLOv8 multi-class model (child +
  hazards jointly), reported **~90% mAP**. Primary baseline since it
  uses the same detect-then-distance framework.
- AlMhdawi et al. (2026) — dual YOLOv8 (one specialist + one
  COCO-pretrained general model), fire domain, exponential-decay risk
  tiers (Low–Critical). Secondary reference for the dual-model idea.
- Ramadan et al. (2025) — YOLOv11n + pose estimation, hand-to-mouth
  distance; reported 92% (no objects) vs. 74% (with objects) accuracy.
  Cited as evidence that combining tasks in one detector can hurt
  accuracy vs. separating them.
- Khan & Dey (2024) — ChildSUn dataset, classification-only (no
  spatial/distance reasoning). Not a direct pipeline baseline, but
  relevant for hazard object visual variety.

## Deliverables / outputs expected

- Two trained YOLOv8n weight files (`child_best.pt`, `hazard_best.pt`)
- Risk-fusion script that takes both models' outputs + an image and
  returns an annotated frame with bounding boxes + Safe/Unsafe label
- mAP tables per detector (overall + per-class for hazards)
- Risk classification accuracy on a held-out manually labeled test set
- Ablation results for: resolution, optimizer, centroid-vs-edge distance
- Comparison table against the four related-work baselines above

## Repo structure

```
data/
  child/          # downloaded child-detection dataset (gitignored)
  hazard/         # downloaded harmful-objects dataset (gitignored)
scripts/
  download_data.py
  train_child.py
  train_hazard.py
  ablation_imgsz.py
  ablation_optimizer.py
  ablation_distance_ref.py
  risk_fusion.py       # the actual fusion layer, standalone + testable
  evaluate.py           # mAP + risk-accuracy computation
runs/             # ultralytics output dirs (gitignored, large)
results/
  metrics/        # csv/json result tables per experiment
  figures/         # any plots
notebooks/
  colab_train.ipynb    # thin notebook that just calls scripts/*
context.md
CLAUDE.md
PROGRESS.md
README.md
docs/
  proposal.pdf    # original submitted course proposal
```

Keep `data/` and `runs/` out of git (`.gitignore`). Keep `results/`
in git — those are the actual deliverable artifacts (numbers, plots).

## Execution plan — run phases in this order

### Phase 0 — Setup
- Scaffold the repo structure above.
- `scripts/download_data.py`: pulls both Roboflow datasets via API,
  verifies the pre-existing train/val/test split is preserved (don't
  reshuffle), writes/validates each `data.yaml`.
- Smoke test: train each model for 1 epoch on a 20-image subset
  locally to confirm the pipeline runs end-to-end before touching
  Colab.

### Phase 1 — Baseline training (no tuning yet)
- Train child detector: YOLOv8n, default ultralytics hyperparameters,
  100 epochs, on Colab T4.
- Train hazard detector: same, on its own dataset.
- Log everything ultralytics gives you for free (`runs/detect/*`) —
  don't discard it.
- Record baseline mAP for both models in `results/metrics/`.

### Phase 2 — Hyperparameter tuning
- Tune `lr0`, `box`, `cls`, `dfl` per model (grid search or
  `ultralytics`'s built-in `model.tune()` if time allows — check
  current ultralytics docs before assuming API surface, since this
  library changes fast).
- One sweep per model. Save every run's config + resulting mAP to
  `results/metrics/tuning_<model>.csv` — don't just keep the winner,
  keep the whole sweep so the paper can show the search.

### Phase 3 — Ablations (fixed setup, one variable at a time)
Each ablation is its own script and its own results file. Do not
conflate them.
- `ablation_imgsz.py`: 416 / 640 / 832, best optimizer/hyperparams
  from Phase 2 held fixed.
- `ablation_optimizer.py`: SGD vs Adam (vs AdamW), imgsz held fixed
  at whatever Phase 3a picked.
- `ablation_distance_ref.py`: centroid-to-centroid vs. nearest-edge
  distance, evaluated on risk classification accuracy, not mAP — this
  ablation is about the fusion layer, not the detectors.

### Phase 4 — Final models + risk fusion
- Retrain both detectors with the winning hyperparams/imgsz/optimizer
  from Phases 2–3, full epoch budget.
- Implement `risk_fusion.py`: takes both models' predictions on a
  frame, computes normalized min-distance across all child-hazard
  pairs, applies the calibrated threshold, returns Safe/Unsafe +
  annotated frame.
- Calibrate the threshold on the validation set (start from Ahmad et
  al.'s 700px reference, but re-derive since resolutions differ) —
  don't just hardcode 700px without checking it makes sense for our
  own image sizes.

### Phase 5 — Evaluation + baseline comparison
- `evaluate.py`: computes final mAP per detector (overall + per-class
  for the 12 hazard classes) and risk classification accuracy on the
  manually labeled test set.
- Build the comparison table against Ahmad et al. (~90% mAP),
  AlMhdawi et al., Ramadan et al., Khan & Dey — see baselines section
  above for what each one actually measured, don't compare apples to
  oranges (e.g. Ramadan's number is hand-to-mouth accuracy, not
  child-vs-hazard mAP).

### Phase 6 — Write-up support
- Generate the plots/tables the paper needs from `results/`.
- Don't write prose for the paper unless asked — focus on producing
  correct, labeled, reproducible numbers and figures.

## Applying the CLAUDE.md principles to this pipeline

- **Look at the data before touching the model.** Before training,
  open a handful of images from each dataset and eyeball the labels.
  The hazard set has 12 easily-confusable small objects (screwdriver
  vs. chisel, fork vs. knife) — label noise there will silently cap
  mAP, and that's worth surfacing before blaming the model.
- **Get a trivial end-to-end baseline before tuning anything** — one
  working forward pass on both models plus the fusion layer, on a
  handful of images, before the full 100-epoch Colab run.
- **Overfit a tiny subset (~20-50 images) first.** If a model can't
  drive loss near zero on a tiny subset, that's a labels/loss/data-
  loading bug, not something more GPU hours will fix.
- **One ablation, one variable** — enforced by the phase structure
  above; each ablation script's diff from baseline should be exactly
  the one variable it's testing.
- **Log every run's config + metrics** to `results/metrics/` — not
  just stdout that dies with the Colab session.
- **Calibrate the risk threshold on validation only; touch test once,
  at the end.** If risk accuracy ends up worse than the joint-model
  baseline, that's a real, reportable answer to the research question
  — don't quietly retune the threshold against the test set to make
  the number look better.

## When Claude Code is unsure

- If the `ultralytics` API for something (e.g. built-in tuning,
  export formats) is unclear or might have changed, check the current
  docs rather than assuming from training data — this library moves
  fast.
- If a dataset's actual class distribution or image count doesn't
  match what's in this file after downloading, flag the mismatch
  rather than silently proceeding.
- If GPU/time budget looks tight for the full ablation matrix, say so
  explicitly and propose which ablations to cut, rather than quietly
  shrinking epoch counts everywhere.

## Key references (for citation, not to be re-derived from memory)

1. Ahmad, M. H. et al. (2025). *A Computer Vision Based Child Safety
   Solution Using YOLOv8 Architecture.* IJIST, 7(7), 297-306.
2. AlMhdawi, A. K. et al. (2026). *Intelligent Spatial Estimation for
   Fire Hazards in Engineering Sites.* arXiv:2603.09069.
3. Solovyev, R. et al. (2021). *Weighted boxes fusion: Ensembling
   boxes from different object detection models.* Image and Vision
   Computing, 107, 104117.
4. Ramadan, N. et al. (2025). *Risk Detection System for Children
   Putting Objects into Mouth Based on Computer Vision using YOLOv11n.*
   IJIS, 26(4).
5. Khan, F. A. & Dey, A. (2024). *Towards enhancing child safety: A
   deep learning approach to detect child safe and unsafe objects.*
   IEEE WIECON-ECE 2024, pp. 123–128.
