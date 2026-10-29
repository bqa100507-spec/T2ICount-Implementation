# RP-010 — controlled supervision: audit and execution contract

Preparation, 2026-10-29, `rich-prompt-alignment`, audit source `e8a31e1` plus
the local RP-010 patch. No full training or scientific conclusion is recorded.
Historical records and outputs are unchanged. Read this before launching either arm.

## Research definition (before implementation)

- **Question:** Does image-specific rich auxiliary supervision improve short
  class-name counting relative to generic class templates under paired controls?
- **Hypothesis:** RP-010B may improve standard FSC147 class-name test MAE and
  RMSE relative to RP-010A. This is a hypothesis, not an interpretation of RP-003.
- **Changed variable:** Only the two auxiliary strings and their corresponding
  validated CLIP masks: A uses `a photo of {class}` / `the {class} in the image`;
  B uses the saved detailed / generalized descriptions. The class slot is identical.
- **Controls:** Same shared initial weights, trainable parameters, 1,000 anchor
  subset, full-train augmentation donor pool, per-epoch sample/augmentation
  schedule, seed, three compatible-positive forwards, class-only incompatible
  forwards, density/RRC losses, zero consistency coefficient, optimizer, budget,
  validation, checkpoint rule, and production inference.
- **Metrics:** Primary standard FSC147 test class MAE/RMSE, per-image absolute
  error and signed bias. Secondary FSC147-S class metrics, GT-count strata and
  over/under counts. Report auxiliary token lengths and truncation. No localization
  accuracy is inferred, and no evaluation prompts construct training supervision.
- **Decision rule:** Review paired B-minus-A MAE/RMSE and bias. Both lower primary
  errors in the first pair support replication, not stable superiority. Mixed
  metrics are inconclusive. Replicate with additional independent training seeds
  before a stable claim; image-bootstrap uncertainty is conditional on a trained
  pair and does not measure training-seed variability. No automatic conclusions.

## Phase A — actual implementation audit

Inspected `train.py`, `utils/regression_trainer.py`, `datasets/dataset.py`,
`utils/rich_prompt_training.py`, `utils/train_subset.py`, `models/build.py`,
`models/reg_model.py`, `utils/checkpoints.py`, `utils/inference.py`, `test.py`,
RP-009 preparation/smoke utilities, PROJECT_STATE, ledger RP-003/RP-008/RP-009,
and DIAG-001/DIAG-002 source/artifacts/documentation. Historical filenames are
not evidence of identical initial weights or runtimes.

| Item | Verified current RP-003/RP-009 shared implementation | Difference / provenance limit |
|---|---|---|
| Initialization | Fresh SD v1.5 U-Net/VAE, local CLIP, seeded counting decoder; no trained counting checkpoint in `Reg_Trainer.setup` | Historical RP-003 initial tensor hash, raw command and runtime are unavailable; cannot certify historical matching. RP-009 preparation records fresh initialization, not a measured initial hash. |
| Parameters | VAE and CLIP frozen; U-Net and decoder trainable | No ALIGN-004 FFN or architecture modification. |
| Subset/order | Torch-generator seed-3407 randperm selects 1,000 primary anchors from 3,659 train images | DataLoader shuffle uses global RNG. Subset does **not** restrict negative-class/mosaic donors: all 3,659 remain available. Historical exact order/augmentations unrecorded. |
| Bank | `fsc147_train1000_seed3407_gemma4_v3.json`, protocol `rich-prompt-phase1-v3`, Gemma-4-26B-A4B-IT; validated selected IDs/classes/status | RP-009 uses this same bank as provenance, then constructs fixed wrappers. It has no separate generated template bank. |
| Routing | Three forwards on compatible unchanged positives, same transformed RGB/GT; negative mismatches/mosaics use class only | RP-003 auxiliary strings from bank; RP-009 exact generic wrappers. Same-class negative sampling remains compatible. |
| Loss | Each forward: original reg + .01 RRC1 + .01 RRC2; compatible losses averaged /3; sample-fraction replacement; lambda_cons=0 | Rich mode also computes a zero-weight consistency diagnostic; generic mode skips it. No gradient accumulation, AMP or gradient clipping. |
| Budget | Proposed/published pilot 1,000 anchors x 10 epochs, batch 1 = 10,000 optimizer updates | Three-forward frequency depends on compatibility; not 30,000 optimizer steps. Historical RP-003 full invocation not retained. |
| Optimizer | AdamW; decoder LR5e-5/WD5e-4; U-Net LR5e-6/WD5e-5; default betas(.9,.999), eps1e-8 | No scheduler in actual code. |
| Seeds | Requested seed/subset seed3407; Python/NumPy/Torch/CUDA seeded | `train.py` enables cuDNN benchmark while setup enables deterministic; historical CUDA/library nondeterminism cannot be excluded. |
| Validation | Standard FSC147 class inference; zero-based epochs7,8,9; strict improvement in MAE+RMSE | Parent `val_epoch` calls test **at every improvement**. RP-010 must suppress this and evaluate selected best once after completion. |
| Targets/preprocess | Original random positive/negative/mosaic, resize1..2, crop384, concat224, flip; sum8x8 density then x60 in loss | Keep full train pool explicit; do not claim 1,000 unique total exposed images. |
| Checkpoints/resume | Weights-only best `.pth`; `.tar` stores optimizer, next epoch, RNG, rich config | Historical resume guards omit full optimizer/data/source identity and step counter; epoch0,5 saving misses final9. RP-010 needs isolated stronger guards and every-epoch latest state. |
| Evaluation | Production patch384/stride384, RGB[-1,1], x8 density interpolation with /64 reassembly, /60 count scaling | Standard test targets are original FSC147; FSC147-S uses different target/counts. No matching positive dots for S. |

Bank locally verified: exact 1,000 keys; byte SHA256
`c3ec587b37fd9f351cdedbce888b26e0ba7a020de956c4ef18202a1268e7e32f`;
ordered selection SHA256
`dd96b36bf15013e194b1a8ece06452a19822aae028fc00c0de019cbb7a311f24`.
These identify the existing bank, not a new prompt generation run. Semantic
truthfulness of every generated sentence is not certified by schema validation.
Descriptions refer to the source image, not necessarily every augmented crop.
Compatibility follows the original routing rule; it does not certify that all
described attributes remain visible after cropping. No recaptioning is introduced.

Historical standard-test MAE/RMSE: BASE-002 22.82/120.25,
RP-003 19.98/107.13, RP-009 23.45/115.18. DIAG-002 all-230 S class MAEs
33.5840/24.1215/14.2940. These are different protocols, not a controlled treatment
effect. DIAG-002 reports reduced overcounting and some density redistribution;
HSCM associations vary across pairs; detailed inference generally degrades counts.
That motivates training controls, not selecting a feature-transfer mechanism.

The already retained DIAG-001 manifest records 4,263,115,917-byte weights-only
artifacts (no optimizer/RNG/initialization metadata). Recorded SHA256s:
BASE-002 `e97aa037939a8df779162166a5131b8245e7b1ac122f7c5165472d9b18cc9a6e`,
RP-003 `bf1b372429132e070c337cc02daeb331a61a4e2666f7557f4e8bc4727f9ec461`,
RP-009 `006f6968cdb18cbb2cc73679e28df48de5cb97351a5ca64f22af057547c14f72`.
These are previously recorded artifact identities, not new downloads or tensor
loads in this audit. No historical full training-state checkpoint/log has been
independently loaded here to recover RP-003's missing provenance.

## Minimum corrections for a new fair pair

These corrections apply to **both new arms only**; original training/evaluation
files remain unchanged.

1. Materialize one newly seeded full Count initialization per training seed;
   both arms load the same SHA256-verified weights. Never initialize from a
   trained BASE/RP checkpoint. Keep an initialization manifest with SD/CLIP asset
   identity, source/runtime, parameter names/shapes/dtypes and frozen flags.
2. Use an independent deterministic epoch permutation and per-anchor augmentation
   RNG context, restoring surrounding RNG afterward. Use workers=0, no AMP, no
   accumulation and cuDNN benchmark=False. Record per-epoch input/target/routing
   digests so a pair can be checked rather than assumed. This is a new paired
   random schedule; it does not reproduce historical trajectories.
3. Preserve the full official train donor pool identically. Validate original
   train/val/test disjointness; no val/test supervision is used. Do not tune on S.
4. Override only RP-010 validation/checkpoint bookkeeping: strict val MAE+RMSE,
   epochs7/8/9, no automatic test call, latest atomic state every completed epoch,
   guarded same-arm/seed/config/data/init/source/runtime resume. Interrupted epochs
   replay from the last completed epoch. Do not claim mid-epoch continuation.
5. During the shared production train loop, both arms execute its `rich` loss
   branch, including the **zero-weight** consistency diagnostic. A's dataset
   still emits the exact generic templates. This removes the old difference
   between computing consistency times zero and skipping it in template mode;
   the mathematical supervised loss and three-forward routing remain unchanged.

Locally checked split sizes 3659/1286/1190; no train intersection with standard
val/test. S shares **124** images with full train and **34** with the 1,000
anchors; those34 have different original/S target classes. S is **not an
image-held-out benchmark**. Its metadata/bank must never supply training text or
selection; overlap is recorded as a secondary diagnostic limitation, not repaired
by silently changing splits. No unresolved fairness issue requires changing the
architecture, loss, original augmentation probabilities or primary held-out split.

## Remaining limitations / review gate

The strategies differ in length, lexical content and semantic specificity; this
pair does not isolate those factors. CLIP content truncation at75 tokens is
reported, not length matched. A single pair has training-seed uncertainty. Exact
historical RP-003 provenance remains unknown. GPU arithmetic can differ across
hardware/packages: resume and pair guards require matching runtime signatures.
Full training requires researcher review and successful **both-arm real GPU smoke**.
Local GPU is RTX4060 Laptop8GiB (~6.9GiB free at audit). Full-precision training
needs weights+gradients+Adam states plus three retained forward graphs; a runtime
parameter-byte conservative budget check must precede smoke. Do not reduce crop, freeze
parameters, introduce AMP, or change losses to squeeze this experiment onto it.

## First paired run

Use seed3407, subset seed3407, 10 epochs, batch1, workers0, no scheduler;
10,000 updates/arm. Run preparation once, review token/data manifests, then run
A and B smoke separately. Use an L4 (24GiB) candidate runtime; this is a capacity
target, not a verified peak-memory estimate. Full training remains disabled by
default. Run A and B separately using the same prepared init and source/runtime.
Validate saved step counts, schedules and completion records before evaluating
each selected best on standard test and S once. Preserve both per-image tables;
report paired deltas and conditional image-bootstrap intervals, then consider
independent seeds3408/3409 with newly shared initializations but the same subset.
See `RP010_RUNBOOK.md` for commands and exact prerequisites.
