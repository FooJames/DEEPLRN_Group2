# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project Context — Child Safety Risk Detector

## What this is
Deep learning course project (DEEPLRN Group 2 S08, DLSU). Two
independently trained YOLOv8n models (child detector, 12-class hazard
detector) fused post-hoc via normalized Euclidean centroid distance
into a Safe/Unsafe risk label. Course project, not production software
— optimize for a working, honest, reproducible experiment over polish.

## Read before starting any work
- context.md — full spec: architecture, datasets, phased execution
  plan, ablations, evaluation metrics, baseline comparisons
- PROGRESS.md — what's been done and decided so far. READ THIS FIRST
  every session. A teammate using a different tool may have worked on
  this since your last session.
- docs/proposal.pdf — the original submitted course proposal.
  context.md is the distilled/working version of this; if the two
  ever disagree, flag it rather than silently trusting one over the
  other.

## Non-negotiable architecture rule
Child detector and hazard detector are trained fully independently —
no shared backbone, no joint training, no merging the two datasets.
Fusion happens only at the output level (distance between predicted
boxes). This is the actual research question the project is testing;
don't "simplify" it into one merged model.

## Hard constraints
- Keep each dataset's pre-existing train/val/test split. Don't reshuffle.
- Training happens on Google Colab (T4 GPU). Local machine has no
  reliable GPU — smoke-test on a tiny subset locally, train for real
  on Colab.
- One variable per ablation. Never fold two changed variables into
  one run and call it an ablation.
- Calibrate the risk-distance threshold on the validation set only.
  Touch the test set once, at the end.
- Never commit a Roboflow API key — read from env var or gitignored `.env`.

## Target environment
`ultralytics` (PyTorch-based), Google Colab T4 GPU. See context.md for
exact package/version notes as they're pinned.

## After finishing any work session
Update PROGRESS.md with: what was completed, what was decided, and
what the next session needs to know. Keep it to 1-3 lines.
