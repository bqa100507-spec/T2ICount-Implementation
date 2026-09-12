# Experiment Ledger

This is the append-only research history. Preserve negative results and obsolete
interpretations. Do not delete or rewrite an old entry to hide a failed branch;
append a superseding entry or note and mark the old conclusion `OBSOLETE`.

Artifact paths record where evidence was reported or observed. An external
Drive path is not a claim that the artifact still exists unless it was checked
at the time stated. `MSE` in legacy T2ICount logs denotes RMSE.

## Summary

| ID | Experiment | Status | Branch decision |
| --- | --- | --- | --- |
| BASE-001 | Official T2ICount checkpoint reproduction | CLOSED | Valid reproduction |
| BASE-002 | Controlled FSC-147 1000x10 baseline | CONTROL BASELINE | Reuse for limited-compute comparisons |
| DUMLO-000 | Early DUMLO 500x10 pilot | CLOSED / HISTORICAL | Preserve provenance; configuration incomplete |
| DUMLO-001 | Direct DUMLO integration | CLOSED-NEGATIVE | Worse than control |
| DUMLO-002 | Reduced count-loss weight | CLOSED-NEGATIVE | Tuning did not rescue branch |
| DUMLO-003 | Reduced OT weight | CLOSED-NEGATIVE | Substantially worse |
| DUMLO-004 | DUMLO + SSIM | CLOSED-NEGATIVE | Did not recover control |
| DUMLO-005 | DUMLO + full original regression loss | CLOSED-NEGATIVE | Did not recover control |
| DUMLO-006 | Zero-point / synthetic-negative incompatibility | PARKED | Reopen only with a new sample-type hypothesis |
| DUMLO-007 | IDCIA 500x10 diagnostic | CLOSED / DIAGNOSTIC | Lower DUMLO MAE is collapse, not transfer |
| RP-001 | RichPrompt prompt-bank construction | CLOSED | Data preparation complete |
| RP-002 | Direct rich prompts on official checkpoint | CLOSED-NEGATIVE | Do not rerun unchanged |
| RP-003 | Multi-prompt supervision, `lambda_cons=0` | CLOSED | Informative but rich branches remain poor |
| RP-004 | Multi-prompt consistency, `lambda_cons=0.1` | CLOSED-NEGATIVE | Complete rerun was not useful |
| RP-005 | Multi-prompt consistency, `lambda_cons=1` | CLOSED / INFORMATIVE | Agreement improved; class accuracy degraded |
| RP-006 | Multi-prompt consistency, `lambda_cons=100` | CLOSED-NEGATIVE | Degenerate agreement without correctness |
| RP-007 | Consistency-loss synthesis | CLOSED | Stop raw weight search |
| ALIGN-000 | Initial raw FFN + Adapter smoke | OBSOLETE | Debug history, not scientific evidence |
| ALIGN-001 | Embedding normalization / metric diagnostic | CLOSED / INFORMATIVE | Frozen CLIP has strong global alignment |
| ALIGN-002 | Random-FFN initialization / margin diagnostic | CLOSED / INFORMATIVE | Initialization destroys CLIP geometry; old conclusion obsolete |
| ALIGN-003 | Stage-A minimal FFN, 5 epochs | CLOSED / INFORMATIVE | Valid learned candidate, but compressed geometry |
| ALIGN-004 | Stage-A Figure3+BN FFN, 5 epochs | ACTIVE-CANDIDATE | Current best Stage-A FFN; counting untested |

## BASE-001 — Official T2ICount checkpoint reproduction

### Question

Does the original implementation/environment/inference path reproduce the
official T2ICount result before any research modification?

### Hypothesis

The official checkpoint should reproduce the paper's FSC-147 performance within
normal numerical tolerance.

### Change

No research variable changed; this was a reproduction/evaluation control.

### Controls

The official checkpoint and original T2ICount inference behavior were retained.
The dataset/evaluator varied only for the separately reported FSC-147,
FSC-147-S, CARPK, and IDCIA evaluations.

### Result

- FSC-147 test: MAE 11.763, RMSE 97.949.
- Paper reference (approximately): MAE 11.76, RMSE 97.86.
- FSC-147-S: MAE 5.989, RMSE 10.551.
- CARPK: MAE ≈8.615, RMSE ≈13.478.
- IDCIA: historical official-checkpoint MAE 90.26. Later prompt/evaluator
  variants exist, so this is preserved as the BASE-001 historical result rather
  than a single definitive value across variants.

### Interpretation

The official checkpoint reproduction is valid. FSC-147-S and CARPK are
approximately consistent with reported T2ICount behavior; IDCIA is a strong
domain shift.

### What this proves

The base inference pipeline and official checkpoint can reproduce the expected
FSC-147 test result closely enough to support controlled follow-up work.

### What this does NOT prove

It does not validate later retraining, DUMLO, RichPrompt training, or alignment
modules. It does not establish one definitive IDCIA score across prompt/evaluator
variants.

### Decision

Close reproduction and use it as the full-checkpoint reference.

### Artifacts

- `results/rich_prompt_phase1b_official/` contains the raw FSC-147-S prompt
  evaluation, manifest, and official-checkpoint fingerprint.
- External checkpoint recorded as
  `D:\T2ICount-assets\checkpoints\official\best_model_paper.pth`.
- Raw FSC-147 test, CARPK, and IDCIA-official artifact locations are not
  recorded here.

### Status

CLOSED

## BASE-002 — Controlled FSC-147 1000x10 retraining baseline

### Question

What performance is achieved by a reproducible limited-compute T2ICount run for
use as the control in later experiments?

### Hypothesis

A deterministic 1,000-sample run would underperform the full official model but
provide a fair controlled reference for similarly budgeted ablations.

### Change

Train on a deterministic 1,000-of-3,659 FSC-147 subset for 10 epochs.

### Controls

Subset seed 3407; training seed 3407; batch size 1; learning rate `5e-5`;
weight decay `5e-4`; original T2ICount architecture, regression objective,
optimizer setup, dataset transforms, and inference path.

### Result

- Validation: MAE 23.16, RMSE 84.09.
- Test: MAE 22.82, RMSE 120.25.
- FSC-147-S class prompt: MAE 33.58, RMSE 52.74.

### Interpretation

The model is worse than the full official checkpoint but is the accepted
limited-compute control.

### What this proves

It establishes the performance of this specific 1000x10 protocol.

### What this does NOT prove

It does not estimate full-data retraining performance and must not be compared
as though it had the official checkpoint's compute/data budget.

### Decision

Retain as the control baseline for limited-compute DUMLO and RichPrompt runs.

### Artifacts

- Historical manifest: `DUMLO:experiments/baseline_1000x10/manifest.yaml`.
- Reported Drive directory:
  `/content/drive/MyDrive/T2ICount-assets/checkpoints/baseline_retrain/pilot_1000x10/baseline_1000x10/`.
- `notebooks/dumlo_gradient_diagnostic.ipynb` recorded
  `best_model_7.pth` as existing when that notebook was executed; current Drive
  existence was not rechecked during this reconstruction.

### Status

CONTROL BASELINE

## DUMLO-000 — Early DUMLO 500x10 pilot

### Question

Not explicitly recorded.

### Hypothesis

Not explicitly recorded.

### Change

An early DUMLO pilot was labeled `dumlo_500x10`. The surviving artifact
designation supports recording a 500-sample/10-epoch pilot and no more specific
training configuration.

### Controls

The surviving artifacts do not provide the command, subset identity or seed,
training seed, batch size, optimizer settings, DUMLO coefficients, epsilon,
Trihorn iterations, augmented-point count, radius factor, commit SHA, checkpoint
path, or checkpoint hash. Those values must not be inherited from later 1000x10
runs.

### Result

No FSC-147 training, validation, or test metric for this pilot survives in the
current checkout. A later 53-image IDCIA prediction artifact attributed to the
`dumlo_500x10` model is analyzed separately as DUMLO-007.

### Interpretation

This is preserved as early execution history, not as a reproducible controlled
comparison or evidence that DUMLO improved counting.

### What this proves

The surviving run designation and IDCIA prediction artifact preserve evidence
that an early model labeled `dumlo_500x10` was evaluated.

### What this does NOT prove

It does not recover the full training configuration, establish a fair baseline,
verify FSC-147 performance, or justify importing settings from DUMLO-001 and
later experiments.

### Decision

Retain the pilot as historical provenance only. Do not use it as a control or
rerun it without a new, explicit manifest and hypothesis.

### Artifacts

- `results/idcia_dumlo_500x10.csv`: 53 prediction rows; SHA-256
  `5a8aee70e4d40a27cf8381cd06bef3f16ed1ffd91d41e7e8de592543f239346d`.
- Historical audit: `DUMLO:docs/REPO_GUIDE_VI.md`; it explicitly notes that the
  500x10 filename does not preserve command, commit SHA, or checkpoint path.

### Status

CLOSED / HISTORICAL

## DUMLO-001 — Direct DUMLO integration

### Question

Does directly replacing the controlled baseline's regression objective with the
operational DUMLO-inspired point-distribution loss improve counting?

### Hypothesis

Not explicitly recorded.

### Change

Use the DUMLO path with `lambda_count=1.0`, `lambda_OT=0.1`, and
`lambda_TV=0.01`.

### Controls

The 1,000-sample subset, seeds, 10-epoch budget, batch size, T2ICount
architecture, RRC path, optimizer setup, dataset transforms, and inference path
were held consistent with BASE-002 where recorded. Operational settings were
epsilon 10, 100 OT iterations, 10 augmented points, radius factor 0.5, and
sampling seed 3407.

### Result

- Validation: MAE 36.96, RMSE 116.37.
- Test: MAE 32.81, RMSE 139.77.

### Interpretation

Direct integration was worse than BASE-002.

### What this proves

This operational DUMLO integration and configuration did not improve the
controlled T2ICount run.

### What this does NOT prove

It does not prove that DUMLO's paper method is ineffective, nor that every
sample-aware integration must fail.

### Decision

Test whether objective weighting caused the degradation.

### Artifacts

- Historical manifest: `DUMLO:experiments/dumlo_1000x10/manifest.yaml`.
- Reported Drive directory:
  `/content/drive/MyDrive/T2ICount-assets/checkpoints/dumlo_retrain/pilot_1000x10/dumlo_1000x10/`.
- `notebooks/dumlo_gradient_diagnostic.ipynb` recorded
  `best_model_7.pth` as existing when executed; current Drive existence was not
  rechecked.

### Status

CLOSED-NEGATIVE

## DUMLO-002 — Reduced count-loss weight

### Question

Would reducing the count component fix the poor direct-DUMLO result?

### Hypothesis

Reducing `lambda_count` might allow the point-distribution terms to contribute
more usefully.

### Change

Reduce `lambda_count` from 1.0 to 0.1 while retaining `lambda_OT=0.1` and the
other recorded DUMLO settings.

### Controls

The controlled 1,000-sample subset, seeds, budget, architecture, RRC, optimizer,
transforms, inference path, and remaining DUMLO configuration matched
DUMLO-001.

### Result

- Validation: MAE 38.16, RMSE 116.36.
- Test: MAE 34.13, RMSE 139.89.

### Interpretation

Reducing the count component did not fix the problem and was slightly worse in
MAE.

### What this proves

`lambda_count=0.1` is not a rescue under this controlled configuration.

### What this does NOT prove

It does not identify the structural failure or justify arbitrary further
`lambda_count` search.

### Decision

Close this tuning direction and test a reduced OT weight separately.

### Artifacts

- Historical manifest: `DUMLO:experiments/dumlo_lc01_1000x10/manifest.yaml`.
- Reported Drive directory:
  `/content/drive/MyDrive/T2ICount-assets/checkpoints/dumlo_retrain/lambda_count_01_1000x10/dumlo_lc01_1000x10/`.
- Continued Drive existence was not verified during this reconstruction.

### Status

CLOSED-NEGATIVE

## DUMLO-003 — Reduced OT weight

### Question

Would reducing the OT contribution recover controlled-baseline performance?

### Hypothesis

The original OT weight might be too strong for T2ICount's training behavior.

### Change

Reduce `lambda_OT` from 0.1 to 0.01, with `lambda_count=1.0` and the other
recorded DUMLO settings unchanged.

### Controls

The controlled subset, seeds, budget, architecture, RRC, optimizer, transforms,
inference path, and non-OT DUMLO settings matched DUMLO-001.

### Result

- Validation: MAE 51.33, RMSE 130.60.
- Test: MAE 50.19, RMSE 153.73.

### Interpretation

This was substantially worse. Simple OT-weight reduction is not a valid rescue.

### What this proves

`lambda_OT=0.01` degraded this operational integration under the fixed control.

### What this does NOT prove

It does not show that OT itself is generally harmful or that sample-type-aware
use of OT would fail.

### Decision

Stop direct DUMLO weight tuning.

### Artifacts

- Historical manifest: `DUMLO:experiments/dumlo_lot001_1000x10/manifest.yaml`.
- Reported Drive directory:
  `/content/drive/MyDrive/T2ICount-assets/checkpoints/dumlo_retrain/lambda_ot_001_1000x10/dumlo_lot001_1000x10/`.
- That manifest records branch `DUMLO`, commit
  `8706cf712e79c43a338cbd37003a959f95bd631f`, clean state, and a
  `console.log`; date remains unknown.

### Status

CLOSED-NEGATIVE

## DUMLO-004 — DUMLO + SSIM

### Question

Can restoring SSIM supervision recover the controlled baseline while retaining
DUMLO?

### Hypothesis

SSIM might restore spatial density-map supervision missing from the direct
DUMLO objective.

### Change

Add the SSIM component to the DUMLO objective.

### Controls

The controlled 1,000-sample/10-epoch framework and operational DUMLO integration
were retained. A complete independently verified run configuration is not
recorded in the current checkout.

### Result

- Validation: MAE 39.68, RMSE 120.51.
- Test: MAE 37.94, RMSE 147.08.

### Interpretation

Adding SSIM did not recover BASE-002.

### What this proves

This tested DUMLO + SSIM run remained materially worse than the control.

### What this does NOT prove

It does not establish that SSIM is ineffective in all objectives or budgets.

### Decision

Do not repeat the same SSIM addition; test the complete original regression
objective once.

### Artifacts

Artifact location not recorded here.

### Status

CLOSED-NEGATIVE

## DUMLO-005 — DUMLO + full original L_reg

### Question

Can restoring the complete original T2ICount regression objective alongside
DUMLO recover BASE-002?

### Hypothesis

The full original density objective might restore supervision not supplied by
the operational DUMLO terms alone.

### Change

Use DUMLO with `lambda_count=1.0`, `lambda_OT=0.1`, `lambda_TV=0.01`, plus
SSIM coefficient 1.0 and normalized-L1 coefficient 0.1.

### Controls

The executed notebook records the same 1,000-sample subset, seed 3407,
10 epochs, batch size 1, learning rate `5e-5`, weight decay `5e-4`, crop 384,
downsample ratio 8, stride 384, and the existing model/optimizer/inference path.

### Result

- Validation: MAE 43.57, RMSE 124.18 (best epoch 7).
- Test: MAE 42.44, RMSE 149.78.

### Interpretation

Restoring the full original regression objective together with DUMLO did not
recover the controlled baseline.

### What this proves

Adding the recorded SSIM and normalized-L1 terms does not rescue this DUMLO
integration.

### What this does NOT prove

It does not prove that the original regression objective is harmful; BASE-002
uses it successfully without DUMLO.

### Decision

Stop additive loss reconstruction as a rescue strategy and investigate the
sample semantics.

### Artifacts

- `notebooks/train_colab.ipynb` contains the executed command and full log.
- Reported Drive directory:
  `/content/drive/MyDrive/T2ICount-assets/checkpoints/dumlo_retrain/full_lreg_1000x10/dumlo_full_lreg_1000x10/`.
- Current Drive existence was not rechecked.

### Status

CLOSED-NEGATIVE

## DUMLO-006 — Zero-point / synthetic-negative incompatibility analysis

### Question

Why do naive DUMLO replacements conflict with T2ICount's text-conditioned
training samples?

### Hypothesis

Zero-point samples may combine semantically different cases that require
different objectives.

### Change

Diagnostic analysis only; no new training variable was introduced.

### Controls

The analysis examined the existing T2ICount synthetic-negative behavior and
current DUMLO zero-point path. Model architecture, training math, and outputs
were not changed by the diagnostic.

### Result

T2ICount can retain objects in an image while deliberately replacing the class
prompt and assigning a zero density/count target. In the current DUMLO path,
zero points make `L_OT=0` and `L_TV=0`, leaving only the remaining count-related
term. This is indistinguishable inside that path from a transformed image with
genuinely no valid target points.

### Interpretation

Naive DUMLO replacement is structurally mismatched with T2ICount's
text-conditioned synthetic negatives.

### What this proves

The current integration fails to distinguish two semantically different
zero-point cases and disables its main point-distribution terms for both.

### What this does NOT prove

It does not prove that the DUMLO paper is ineffective, that positive-only DUMLO
would fail, or that a dedicated synthetic-negative objective cannot work.

### Decision

Park DUMLO. Reopen only with a new hypothesis: apply DUMLO only to positive
samples with valid points while retaining original T2ICount loss for synthetic
text-negatives, or define a separate negative objective.

### Artifacts

- `losses/dumlo.py`, `utils/regression_trainer.py`, and DUMLO history describe
  the operational integration.
- `notebooks/dumlo_gradient_diagnostic.ipynb` contains later diagnostic evidence,
  but it is not itself a new training experiment.

### Status

PARKED

## DUMLO-007 — IDCIA 500x10 diagnostic

### Question

Do the surviving IDCIA CSVs show successful cross-domain transfer by the early
DUMLO 500x10 pilot relative to the limited-compute baseline?

### Hypothesis

A lower aggregate MAE could reflect improved transfer, or it could be produced
by a degenerate prediction regime; the per-image predictions must distinguish
these cases.

### Change

Diagnostic only. Recompute metrics from
`results/idcia_baseline_500x10.csv` and
`results/idcia_dumlo_500x10.csv` for the generic `cell` and
staining-specific prompt columns. No model, training objective, dataset, or
inference code was changed.

### Controls

Both CSVs contain the same ordered 53 IDCIA images and identical ground-truth
counts. Four zero-GT images are excluded from mean and median APE, matching the
IDCIA evaluator convention. The CSVs do not encode preprocessing mode,
checkpoint path/hash, training command, or commit SHA, so none is inferred.

### Result

Mean ground truth is 91.04 counts.

| Artifact / prompt | MAE | RMSE | WAPE | Mean APE | Median APE | Mean prediction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `idcia_baseline_500x10.csv`, generic `cell` | 325.17 | 378.72 | 357.18% | 6699.32% | 332.77% | 416.21 |
| `idcia_baseline_500x10.csv`, staining-specific | 285.83 | 338.36 | 313.96% | 5947.19% | 316.10% | 376.86 |
| `idcia_dumlo_500x10.csv`, generic `cell` | 90.91 | 134.15 | 99.86% | 98.82% | 99.80% | 0.137 |
| `idcia_dumlo_500x10.csv`, staining-specific | 90.82 | 134.09 | 99.76% | 97.50% | 99.74% | 0.248 |

All 53 DUMLO generic predictions are below 1 count (range 0.032–0.274),
and all 53 staining-specific predictions are below 1 count (range 0.067–0.779).

### Interpretation

DUMLO's lower MAE is caused by near-zero prediction collapse against a baseline
that overcounts severely. It is not successful IDCIA transfer. The negligible
generic/specific difference also provides no useful prompt-sensitivity evidence
for this checkpoint.

### What this proves

For these exact 53-image CSV artifacts, the DUMLO checkpoint's apparently lower
MAE is a collapse diagnostic, not a model-quality win.

### What this does NOT prove

It does not prove DUMLO generally fails, recover the 500x10 training
configuration, establish a fair controlled comparison, or supersede BASE-001's
historical official-checkpoint IDCIA MAE 90.26. Later IDCIA prompt/evaluator
variants remain separate records.

### Decision

Close as diagnostic evidence. Do not rank `dumlo_500x10` as successful transfer
or use its lower MAE to reopen the parked DUMLO branch without a new hypothesis.

### Artifacts

- `results/idcia_baseline_500x10.csv`: 53 rows; SHA-256
  `d0b3f4ae73c6d3fb3f78032a05aa1716064be1ac8c1a6e01195de731789a9e84`.
- `results/idcia_dumlo_500x10.csv`: 53 rows; SHA-256
  `5a8aee70e4d40a27cf8381cd06bef3f16ed1ffd91d41e7e8de592543f239346d`.
- Both artifacts are ignored by `/results/` in `.gitignore`; the ledger records
  their provenance without adding the CSVs to Git.

### Status

CLOSED / DIAGNOSTIC

## RP-001 — RichPrompt prompt-bank construction

### Question

Can reproducible class/detailed/generalized prompts be prepared without count
leakage for the fixed datasets/subsets?

### Hypothesis

Offline prompt banks can hold image-aware descriptions while excluding counts
and preserving exact provenance.

### Change

Generate detailed descriptions using `gemma-4-26b-a4b-it`; derive generalized
descriptions by class-name-to-object replacement; save all prompts offline.

### Controls

Training selection used the exact deterministic FSC-147 1,000-sample subset and
seed 3407. FSC-147-S used all 230 images. Prompts were constrained not to count,
state quantities, or leak ground-truth counts.

### Result

- `fsc147_train1000_seed3407_gemma4_v3.json`: 1,000 selected samples,
  fingerprint
  `sha256:dd96b36bf15013e194b1a8ece06452a19822aae028fc00c0de019cbb7a311f24`.
- `fsc147s_gemma4_26b_v3.json`: all 230 FSC-147-S samples.
- Protocol `rich-prompt-phase1-v3`; generator `gemma-4-26b-a4b-it`.

### Interpretation

The prompt inputs needed by later RichPrompt work are fixed and reproducible.

### What this proves

The banks exist with recorded selection/generator provenance and can be reused
as controls.

### What this does NOT prove

It does not prove prompt correctness, alignment quality, or counting improvement.

### Decision

Close data preparation and require later experiments to use the same bank unless
prompt bank is the explicit changed variable.

### Artifacts

- `prompts/fsc147_train1000_seed3407_gemma4_v3.json`
- `prompts/fsc147s_gemma4_26b_v3.json`
- `docs/RICH_PROMPT_PHASE1.md`
- `docs/RICH_PROMPT_PHASE2A.md`

### Status

CLOSED

## RP-002 — Direct rich prompts on official T2ICount checkpoint

### Question

Can detailed or generalized prompts be used directly by the official checkpoint
without retraining?

### Hypothesis

Richer descriptions might improve target identification while holding the image
and counting model fixed.

### Change

Change only the text input among class, exact detailed, and exact generalized
prompts on FSC-147-S.

### Controls

All 230 images, official checkpoint, image patches, inference path, evaluator,
prompt bank, and density scale were held fixed.

### Result

- Class: MAE 5.9889466, RMSE 10.5507222.
- Detailed: MAE 35.9713112, RMSE 109.8975685; 55 images improved and 175
  worsened versus class.
- Generalized: MAE 37.8577242, RMSE 114.1148350; 54 improved and 176 worsened.

### Interpretation

Direct rich-prompt substitution causes severe degradation.

### What this proves

The official checkpoint is not zero-shot compatible with these stored rich
prompts under the fixed evaluator.

### What this does NOT prove

It does not prove that the prompts are semantically wrong, that CLIP lacks
global understanding, or that training/alignment cannot make them useful.

### Decision

Close negative. Do not rerun unless the representation/model changes.

### Artifacts

- `results/rich_prompt_phase1b_official/rich_prompt_eval_summary.json`
- `results/rich_prompt_phase1b_official/rich_prompt_eval_predictions.csv`
- `results/rich_prompt_phase1b_official/rich_prompt_eval_manifest.json`
- `docs/RICH_PROMPT_PHASE1B.md`

### Status

CLOSED-NEGATIVE

## RP-003 — Multi-prompt supervised training, lambda_cons = 0

### Question

Does supervising class, detailed, and generalized density branches improve rich
prompt behavior without an explicit consistency penalty?

### Hypothesis

Not explicitly recorded.

### Change

Train all three prompt branches with equal supervised T2ICount density loss;
set `lambda_cons=0`.

### Controls

The controlled 1,000-sample protocol and prompt bank were retained. Rich
supervision applied only to semantically compatible positive samples;
synthetic text-negatives and mosaic-style augmentations kept original behavior.

### Result

- FSC-147-S class: MAE 24.11, RMSE 40.26.
- Detailed: MAE 44.76, RMSE 77.04.
- Generalized: MAE 46.46, RMSE 80.47.
- Standard validation: MAE 23.06, RMSE 86.91.
- Standard test: MAE 19.98, RMSE 107.13.

### Interpretation

Multi-prompt supervision changed behavior and improved the standard test/class
side relative to BASE-002, but detailed/generalized FSC-147-S behavior remained
poor and did not demonstrate useful rich semantics.

### What this proves

Training on all three prompts has a measurable effect even with no consistency
penalty.

### What this does NOT prove

It does not prove semantic transfer to detailed/generalized prompts or isolate
which prompt branch caused the standard-test change.

### Decision

Test explicit consistency pressure while preserving the same representation.

### Artifacts

Artifact location not recorded here.

### Status

CLOSED

## RP-004 — Multi-prompt consistency, lambda_cons = 0.1

### Question

Does weak consistency pressure provide a useful compromise between prompt
agreement and counting accuracy?

### Hypothesis

A small consistency term might transfer useful behavior without overwhelming
density supervision.

### Change

Set `lambda_cons=0.1` in otherwise multi-prompt supervised training.

### Controls

The prompt modes, supervised objective, compatible-positive routing, controlled
subset, seeds, and evaluation protocol were retained. An earlier
interrupted/disconnected run is distinct from the complete rerun reported here.

### Result

- FSC-147-S class: MAE 47.573644, RMSE 79.345336.
- Detailed: MAE 47.606861, RMSE 78.697640.
- Generalized: MAE 47.840366, RMSE 79.447222.
- Standard validation: MAE 26.77, RMSE 86.55.
- Standard test: MAE 21.98, RMSE 115.42.

### Interpretation

Weak consistency did not produce a useful intermediate solution.

### What this proves

The complete `lambda_cons=0.1` rerun did not solve rich-prompt counting under
this protocol.

### What this does NOT prove

It does not merge or validate the earlier interrupted run and does not prove all
forms of consistency regularization are ineffective.

### Decision

Close negative and compare stronger consistency settings already in the branch.

### Artifacts

- `notebooks/rich_prompt_phase2b_diagnostic.ipynb` contains the complete-rerun
  checkpoint reference and executed FSC-147-S evaluation.
- Reported checkpoint:
  `/content/drive/MyDrive/T2ICount-assets/checkpoints/rich_prompt/pilot_1000x10_lambda0p1_rerun/rich_prompt_1000x10_lambda0p1_rerun/best_model_9.pth`.
- Reported evaluation output:
  `/content/drive/MyDrive/T2ICount-assets/outputs/rich_prompt_phase2c_lambda0p1/`.

### Status

CLOSED-NEGATIVE

## RP-005 — Multi-prompt consistency, lambda_cons = 1

### Question

Does stronger consistency align prompt outputs while retaining acceptable class
counting?

### Hypothesis

A weight of 1 might balance density supervision and output agreement.

### Change

Set `lambda_cons=1`.

### Controls

The multi-prompt supervised protocol and evaluation were retained. The run had
disconnect/resume behavior and is not treated as an uninterrupted gold-standard
run without further artifact proof.

### Result

- FSC-147-S class: MAE 36.39, RMSE 52.89.
- Detailed: MAE 36.48, RMSE 52.99.
- Generalized: MAE 36.49, RMSE 53.00.
- Standard validation/test metrics are not recorded here.

### Interpretation

The three outputs became very similar. Detailed/generalized improved relative to
RP-003, but the class branch degraded.

### What this proves

Output consistency can transfer behavior among prompt branches.

### What this does NOT prove

Agreement does not prove correct semantics or accurate counting, and the
disconnect/resume provenance does not support calling the run uninterrupted.

### Decision

Keep as informative evidence, not as a winning model.

### Artifacts

Artifact location not recorded here.

### Status

CLOSED / INFORMATIVE

## RP-006 — Multi-prompt consistency, lambda_cons = 100

### Question

Does forcing near-equality among prompt density outputs produce accurate
counting?

### Hypothesis

Very strong consistency should make the three outputs nearly identical; whether
that agreement is correct was the key test.

### Change

Set `lambda_cons=100`.

### Controls

The multi-prompt supervised protocol, prompt bank, compatible-positive routing,
controlled data, and evaluator were retained.

### Result

- FSC-147-S class: MAE 49.99, RMSE 77.47.
- Detailed: MAE 49.98, RMSE 77.45.
- Generalized: MAE 49.98, RMSE 77.45.
- Standard validation: MAE 25.73, RMSE 85.35.
- Standard test: MAE 25.32, RMSE 113.34.

### Interpretation

The outputs became almost identical while all remained wrong: output
consistency is not correctness.

### What this proves

Large consistency weight can produce a degenerate agreement solution.

### What this does NOT prove

It does not show that all moderate consistency formulations are useless after a
representation change.

### Decision

Close negative and stop using large weights merely to force equality.

### Artifacts

Artifact location not recorded here.

### Status

CLOSED-NEGATIVE

## RP-007 — Consistency-loss synthesis

### Question

What conclusion is justified by RP-003 through RP-006, and should the raw
`lambda_cons` search continue?

### Hypothesis

Not explicitly recorded.

### Change

No new training change; this is a cross-experiment synthesis.

### Controls

Only evidence from the recorded multi-prompt experiments was considered.

### Result

There was no monotonic relation between `lambda_cons` and counting quality.
Weight 1 balanced outputs at the class branch's expense; weight 100 forced
near-identical but inaccurate outputs. Detailed/generalized counting remained
unsolved.

### Interpretation

Output agreement alone cannot repair a potentially unsuitable representation.

### What this proves

More blind weight tuning on the same representation is not scientifically
justified by the existing sweep.

### What this does NOT prove

It does not show that consistency can never help after a meaningful
representation or architecture change.

### Decision

Stop the raw `lambda_cons` grid search and investigate image-text representation
alignment.

### Artifacts

See RP-003 through RP-006. No separate artifact was produced for the synthesis.

### Status

CLOSED

## ALIGN-000 — Initial raw FFN + Adapter smoke

### Question

Can the initial shallow FFN and Adapter execute end to end under the first
raw-Euclidean alignment prototype?

### Hypothesis

Not explicitly recorded.

### Change

Train an approximately 1.18M-parameter shallow visual FFN for one epoch in raw
Euclidean space, then train an approximately 1.18M-parameter Adapter after that
FFN. Frozen CLIP had approximately 427.6M parameters.

### Controls

Frozen CLIP, the deterministic alignment data, and class/detailed/generalized
prompt modes were retained. This was only a smoke run.

### Result

The one-epoch FFN output looked collapsed/almost random. The Adapter was then
trained against that already-poor visual space. Exact metrics and artifact
location are not recorded here.

### Interpretation

The run identified a debugging problem but cannot diagnose the fundamental
learnability of either module.

### What this proves

The initial raw prototype executed and produced a bad one-epoch FFN state.

### What this does NOT prove

It does not prove that the FFN cannot learn, that the Adapter fails, or that
RichCount-style alignment is ineffective.

### Decision

Supersede its scientific interpretation with normalization, initialization, and
longer Stage-A diagnostics.

### Artifacts

Artifact location not recorded here.

### Status

OBSOLETE

## ALIGN-001 — Embedding normalization / metric diagnostic

### Question

What does global CLIP alignment look like when final embeddings and the
Euclidean metric use a meaningful normalized scale?

### Hypothesis

L2-normalizing only the final projected embeddings would put distances in
`[0,2]`, make margins around 1.0–1.3 interpretable, and reveal whether frozen
CLIP already separates matched from mismatched pairs.

### Change

Enable L2 normalization only on final projected image/text embeddings before
Euclidean loss, metrics, and retrieval.

### Controls

Frozen local CLIP ViT-L/14; full stored images; 1,000-source subset; 900/100
split; seed 3407; same prompt bank and prompt modes. CLIP hidden states, FFN
inputs, and Adapter token outputs were not normalized.

### Result

Representative frozen-CLIP overall metrics on 100 validation images:

- contrastive loss 0.7595;
- pairwise alignment 0.9900;
- retrieval R@1 0.7100;
- class-aware R@1 0.8833;
- sample-aware R@1 0.5200;
- separation gap 0.1034;
- mean positive distance 1.2320; mean negative distance 1.3354.

Per-prompt R@1: class 0.87, detailed 0.70, generalized 0.56.

### Interpretation

Frozen CLIP already matches detailed/generalized descriptions reasonably well
at a global level. The earlier claim that CLIP simply does not understand rich
text is too strong.

### What this proves

Global frozen-CLIP geometry contains strong matching signal on this held-out
alignment split.

### What this does NOT prove

It does not prove local/spatial grounding, T2ICount conditioning quality, or
counting improvement.

### Decision

Mark the old global-understanding hypothesis obsolete and diagnose FFN
initialization/training separately.

### Artifacts

- `notebooks/rich_prompt_alignment_colab_v4.ipynb` contains the executed
  normalized smoke summary.
- Reported Drive directory:
  `/content/drive/MyDrive/T2ICount-assets/checkpoints/rich_alignment/smoke_norm_m12/`.

### Status

CLOSED / INFORMATIVE

## ALIGN-002 — Random-FFN initialization / margin diagnostic

### Question

Does a random FFN preserve frozen CLIP geometry before training, and how does
normalized distance interact with the margin?

### Hypothesis

A random trainable projection may destroy pretrained geometry before learning;
normalized random distances should cluster near `sqrt(2)`.

### Change

Evaluate the exact randomly initialized FFN instance that will train, before
optimizer steps, under normalized Euclidean diagnostics. Examine margin
activation around 1.2/1.3.

### Controls

The frozen CLIP, prompt bank, source subset, split, prompt modes, and exact FFN
module identity were held fixed. The diagnostic did not replace the module or
modify its parameters.

### Result

- Pairwise accuracy approximately 0.46–0.48.
- Retrieval R@1 approximately 0.02–0.04.
- Separation gap around zero or slightly negative.
- Mean distances near `sqrt(2) ≈ 1.414`.
- In the executed smoke summary, initial-FFN overall pairwise accuracy was 0.46,
  R@1 0.0367, gap -0.00145, `d_pos` 1.4341, and `d_neg` 1.4326.

### Interpretation

Random initialization destroys useful CLIP neighborhoods. Margin choice changes
whether the negative hinge activates. The old conclusion "FFN cannot learn" is
obsolete; it described initialization/one-epoch behavior.

### What this proves

The untrained FFN is near random on these metrics and is an essential baseline
for interpreting later recovery.

### What this does NOT prove

It does not establish the trained FFN's ceiling or Adapter quality.

### Decision

Run a longer controlled Stage-A FFN before making a learnability conclusion.

### Artifacts

- `notebooks/rich_prompt_alignment_colab_v4.ipynb`
- `tools/train_rich_alignment.py` initialization-diagnostic outputs include
  `stage0_ffn_init_metrics.json` when persistence is enabled.

### Status

CLOSED / INFORMATIVE

## ALIGN-003 — Stage-A minimal FFN, 5 epochs

### Question

Can the minimal FFN recover from random initialization and learn alignment over
five epochs?

### Hypothesis

Longer controlled training should determine whether the one-epoch collapse was
an optimization-duration artifact.

### Change

Train the minimal FFN
`Linear(768,768) -> ReLU -> Dropout -> Linear(768,768)` for five epochs.

### Controls

Frozen local CLIP ViT-L/14; feature dimension 768; deterministic 1,000-source
subset and 900/100 split; all seeds 3407; batch size 32; learning rate `1e-4`;
dropout 0.1; normalized final embeddings; normalized Euclidean distance; margin
1.3; all three prompt modes; Stage A only; validation-loss checkpoint selection.

### Result

Approximately 1,181,184 trainable parameters. Final reported overall metrics:

- contrastive loss 0.389;
- pairwise accuracy 0.883;
- R@1 0.307; class-aware R@1 0.483; sample R@1 0.150;
- gap 0.168; `d_pos` 0.801; `d_neg` 0.969;
- negative fraction below margin approximately 1.0;
- per-prompt R@1: class 0.63, detailed 0.18, generalized 0.11.

### Interpretation

The FFN clearly learned, superseding the old "cannot recover" conclusion. Its
low scalar loss partly reflects global distance compression, with nearly every
negative still inside the margin.

### What this proves

The minimal FFN can recover meaningful paired separation over five epochs.

### What this does NOT prove

It does not prove healthier semantic neighborhoods than frozen CLIP, useful
spatial alignment, or improved counting. Its lower scalar loss does not by
itself make it the better FFN.

### Decision

Treat as a valid learned control and compare a paper-figure-inspired architecture
with healthier scale behavior.

### Artifacts

- Reported Drive directory:
  `/content/drive/MyDrive/T2ICount-assets/checkpoints/rich_alignment/stageA_minimal_5e_m13/`.
- The raw directory was not available locally during this reconstruction.

### Status

CLOSED / INFORMATIVE

## ALIGN-004 — Stage-A Figure3+BN FFN, 5 epochs

### Question

Does the Figure3+BN FFN produce healthier rich-prompt alignment geometry than
the minimal FFN under the same five-epoch controls?

### Hypothesis

The deeper nonlinearity and terminal batch normalization may preserve distance
scale/separation and improve detailed/generalized retrieval.

### Change

Replace only the minimal FFN with
`Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> BatchNorm1d`, all at
dimension 768.

### Controls

The ALIGN-003 CLIP, subset, 900/100 split, seeds, batch size, five epochs,
learning rate, dropout, final normalization, normalized Euclidean metric, margin
1.3, prompt modes, Stage-A-only boundary, and checkpoint selection were retained.
The Adapter was not trained.

### Result

The notebook records 1,182,720 trainable FFN parameters and best epoch 5:

- contrastive loss 0.7658586;
- pairwise accuracy 0.9500000;
- R@1 0.3900; class-aware R@1 0.5667; sample R@1 0.2633;
- separation gap 0.16856;
- `d_pos` 1.23504; `d_neg` 1.40360;
- negative fraction below margin 0.0900.

Per-prompt R@1 / class-aware R@1:

- class: 0.58 / 0.58;
- detailed: 0.37 / 0.66;
- generalized: 0.22 / 0.46.

### Interpretation

ALIGN-004 is currently more promising than ALIGN-003 for rich-prompt alignment:
it has better retrieval/pairwise behavior, much stronger detailed/generalized
retrieval, and healthier distance scale. Its higher scalar loss is not evidence
that it is worse because ALIGN-003 compressed both positive and negative
distances.

### What this proves

Under the fixed Stage-A validation protocol, Figure3+BN produces better recorded
retrieval and distance geometry than the minimal FFN.

### What this does NOT prove

It does not beat frozen CLIP on generic/global retrieval, improve T2ICount
counting, establish local/spatial grounding, justify direct insertion into the
diffusion visual path, or validate a Stage-B Adapter.

### Decision

Keep as the current best Stage-A candidate and stop here pending explicit
scientific approval of a next experiment.

### Artifacts

- `notebooks/rich_prompt_alignment_colab_v4.ipynb` contains the executed command
  and five epoch records.
- Reported Drive directory:
  `/content/drive/MyDrive/T2ICount-assets/checkpoints/rich_alignment/stageA_figure3bn_5e_m13/`.

### Status

ACTIVE-CANDIDATE
