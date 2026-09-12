# T2ICount Research Project State

This document is the current, reviewable scientific state of the repository. It
may be rewritten when reviewed evidence changes the accepted direction. The
append-only history is in `EXPERIMENT_LEDGER.md`.

State reconstructed: 2026-09-12. Current repository branch at reconstruction:
`rich-prompt-alignment` (`e4ec6ba`). Metrics below are accepted research records;
where a raw artifact was available in this checkout, it was preferred over a
narrative report. Approximate values remain marked with `≈`.

## Goal and active direction

The project studies zero-shot object counting, based primarily on T2ICount. The
long-term goal is to improve T2ICount's text sensitivity and rich-prompt
capability through controlled experiments.

The active direction is RichPrompt / RichCount-inspired visual-text alignment.
DUMLO is parked, not deleted. This project is not currently a faithful
reproduction of RichCount; it adapts selected ideas and components.

## Architectural boundary

T2ICount does not use the CLIP image encoder as its counting visual path. Its
counting path is approximately:

```text
image
  -> Stable-Diffusion VAE latent
  -> one-step diffusion U-Net
  -> multi-scale diffusion features / cross-attention
  -> HSCM
  -> counter
  -> density map
```

CLIP mainly provides text conditioning. Consequently, a RichCount visual FFN
trained on CLIP image features cannot be assumed to plug directly into T2ICount
inference. Its downstream role remains unresolved.

## Chronological research history

### 1. Official checkpoint reproduction — BASE-001

The original implementation, environment, and inference pipeline were checked
before research changes.

| Evaluation | MAE | RMSE |
| --- | ---: | ---: |
| FSC-147 test, reproduced official checkpoint | 11.763 | 97.949 |
| FSC-147 test, paper (approximately) | 11.76 | 97.86 |
| FSC-147-S, official checkpoint | 5.989 | 10.551 |
| CARPK, official checkpoint | ≈8.615 | ≈13.478 |

The historical official-checkpoint IDCIA evaluation recorded MAE 90.26. Later
IDCIA prompt/evaluator variants exist, so 90.26 is preserved as the BASE-001
historical result rather than treated as one definitive value across variants.

Accepted state: the official checkpoint reproduction is valid. FSC-147-S and
CARPK are approximately consistent with the paper. IDCIA is a strong domain
shift and performs poorly. This experiment is closed.

### 2. Controlled limited-compute baseline — BASE-002

A deterministic FSC-147 baseline was trained on 1,000 of 3,659 training
samples, using subset seed 3407, training seed 3407, 10 epochs, batch size 1,
learning rate `5e-5`, and weight decay `5e-4`.

| Evaluation | MAE | RMSE |
| --- | ---: | ---: |
| Validation | 23.16 | 84.09 |
| Test | 22.82 | 120.25 |
| FSC-147-S, class prompt | 33.58 | 52.74 |

Accepted state: this model is worse than the full official checkpoint, but it
is the control baseline for limited-compute DUMLO and RichPrompt experiments.

### 3. DUMLO loss branch — DUMLO-000 through DUMLO-007

This branch kept the T2ICount architecture and inference path unchanged while
replacing or augmenting density regression with an operational DUMLO-inspired
objective:

```text
L_DUMLO = lambda_count * L_count
        + lambda_OT * L_OT
        + lambda_TV * N * L_TV
```

DUMLO uses optimal transport / Trihorn-related matching between predicted
density distributions and point annotations. The run sequence and results were:

| ID | Main intervention | Validation MAE / RMSE | Test MAE / RMSE | Outcome |
| --- | --- | ---: | ---: | --- |
| DUMLO-000 | Early pilot labeled `dumlo_500x10`; only the 500-sample/10-epoch designation survives | Not recorded | Not recorded | Historical pilot; full configuration and FSC metrics are unrecoverable |
| DUMLO-001 | Direct integration; `lambda_count=1`, `lambda_OT=0.1` | 36.96 / 116.37 | 32.81 / 139.77 | Worse than BASE-002 |
| DUMLO-002 | Reduce `lambda_count` to `0.1` | 38.16 / 116.36 | 34.13 / 139.89 | Did not fix the problem |
| DUMLO-003 | Reduce `lambda_OT` to `0.01` | 51.33 / 130.60 | 50.19 / 153.73 | Substantially worse |
| DUMLO-004 | Add SSIM | 39.68 / 120.51 | 37.94 / 147.08 | Did not recover baseline |
| DUMLO-005 | Restore full original `L_reg` with DUMLO | 43.57 / 124.18 | 42.44 / 149.78 | Did not recover baseline |
| DUMLO-006 | Diagnose zero-point / synthetic-negative incompatibility | Not applicable | Not applicable | Structural mismatch identified; branch parked |
| DUMLO-007 | IDCIA diagnostic of baseline/DUMLO 500x10 CSVs | Not applicable | Not applicable | Lower DUMLO MAE is near-zero prediction collapse, not successful transfer |

DUMLO-006 established the main compatibility problem. T2ICount creates
synthetic text-negative samples in which the image can still contain objects,
the class prompt is intentionally wrong, and the target density/count is zero.
In the current DUMLO integration these samples also have zero point annotations,
which effectively turns off `L_OT` and `L_TV` and leaves only the remaining
count-related contribution.

"Zero points" therefore conflates two cases:

1. a transformed image genuinely has no valid annotated target points; and
2. objects remain in the image, but a deliberately mismatched text prompt makes
   the correct text-conditioned target count zero.

DUMLO-007 preserved the later IDCIA diagnostic from the two surviving 500x10
prediction CSVs. Both files contain the same ordered 53 images and ground
truths; four zero-GT images are excluded from percentage-error metrics. Metrics
recomputed directly from the CSVs are:

| Artifact / prompt | MAE | RMSE | WAPE | Mean APE | Median APE | Mean prediction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `idcia_baseline_500x10.csv`, generic `cell` | 325.17 | 378.72 | 357.18% | 6699.32% | 332.77% | 416.21 |
| `idcia_baseline_500x10.csv`, staining-specific | 285.83 | 338.36 | 313.96% | 5947.19% | 316.10% | 376.86 |
| `idcia_dumlo_500x10.csv`, generic `cell` | 90.91 | 134.15 | 99.86% | 98.82% | 99.80% | 0.137 |
| `idcia_dumlo_500x10.csv`, staining-specific | 90.82 | 134.09 | 99.76% | 97.50% | 99.74% | 0.248 |

Mean ground truth is 91.04. Every DUMLO generic and staining-specific prediction
is below 1 count. Therefore DUMLO's lower IDCIA MAE is caused by near-zero
prediction collapse against a strongly overcounting limited-compute baseline,
not successful cross-domain transfer or useful prompt sensitivity.

Accepted state: the experiments do not prove that the DUMLO paper is
ineffective. They show that naive DUMLO replacement is structurally mismatched
with T2ICount's synthetic text-negative behavior. If reopened, a legitimate new
hypothesis would apply DUMLO only to positive samples with valid points while
retaining the original loss for synthetic text-negatives, or define a separate
negative-sample objective. The DUMLO branch is parked.

### 4. RichPrompt prompt banks — RP-001

RichPrompt preserves three text forms for the same target:

- `T_c`: class name;
- `T_d`: detailed description;
- `T_g`: generalized description.

Detailed prompts were generated offline with `gemma-4-26b-a4b-it`. They were
constrained not to reveal counts or use quantity cues. Generalized descriptions
were derived by replacing/removing explicit class identity with a generic
object-style reference.

The checked-in training bank covers the deterministic FSC-147 1,000-sample
subset selected with seed 3407. The FSC-147-S bank covers all 230 images. These
saved banks make later comparisons use the same prompt inputs. Data preparation
is closed.

### 5. Direct rich-prompt inference — RP-002

The unchanged official T2ICount checkpoint was evaluated on all 230 FSC-147-S
images.

| Prompt | MAE | RMSE | Better than class | Worse than class |
| --- | ---: | ---: | ---: | ---: |
| Class | 5.989 | 10.551 | — | — |
| Detailed | 35.971 | 109.898 | 55 / 230 | 175 / 230 |
| Generalized | 37.858 | 114.115 | 54 / 230 | 176 / 230 |

Accepted state: directly replacing the short class prompt with a rich
description severely degrades counting. The exact experiment is closed-negative
and should not be repeated unless the representation or model changes.

### 6. Multi-prompt supervision and consistency — RP-003 through RP-007

For semantically compatible positive samples, class, detailed, and generalized
prompts produced density maps `D_c`, `D_d`, and `D_g`. Each branch received the
same T2ICount density supervision:

```text
L_sup = (L_T2I(D_c, D_GT) + L_T2I(D_d, D_GT) + L_T2I(D_g, D_GT)) / 3

L_cons = (MSE(D_c, D_d) + MSE(D_c, D_g) + MSE(D_d, D_g)) / 3

L = L_sup + lambda_cons * L_cons
```

Synthetic text-negative and mosaic-style augmentations retained original
T2ICount behavior rather than blindly receiving detailed/generalized prompts.

| ID | `lambda_cons` | FSC-147-S class MAE / RMSE | detailed MAE / RMSE | generalized MAE / RMSE | Val MAE / RMSE | Test MAE / RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RP-003 | 0 | 24.11 / 40.26 | 44.76 / 77.04 | 46.46 / 80.47 | 23.06 / 86.91 | 19.98 / 107.13 |
| RP-004 | 0.1 | 47.57 / 79.35 | 47.61 / 78.70 | 47.84 / 79.45 | 26.77 / 86.55 | 21.98 / 115.42 |
| RP-005 | 1 | 36.39 / 52.89 | 36.48 / 52.99 | 36.49 / 53.00 | Not recorded | Not recorded |
| RP-006 | 100 | 49.99 / 77.47 | 49.98 / 77.45 | 49.98 / 77.45 | 25.73 / 85.35 | 25.32 / 113.34 |

RP-004 had an earlier interrupted/disconnected run; the table represents the
complete rerun. RP-005 involved disconnect/resume behavior and is not accepted
as an uninterrupted gold-standard run without stronger artifact evidence.

Accepted synthesis (RP-007):

- multi-prompt training can change and sometimes improve class/test behavior,
  while detailed/generalized counting remains poor;
- `lambda_cons` has no simple monotonic relationship with counting quality;
- `lambda_cons=1` makes prompt outputs similar but degrades the class branch;
- `lambda_cons=100` shows a degenerate agreement solution: output consistency
  is not correctness;
- further blind `lambda_cons` search on the same raw representation is not
  justified.

This branch decision motivated investigating representation alignment before
any further consistency tuning.

### 7. RichCount-inspired Stage-1 investigation

The earlier working hypothesis was that detailed/generalized prompts were not
represented or aligned well enough, so density-output consistency merely
propagated bad representations. Later frozen-CLIP evidence made the stronger
claim "CLIP does not understand detailed/generalized text" obsolete.

The repository's alignment prototype is an adaptation, not an exact RichCount
reproduction. It uses:

- frozen local CLIP ViT-L/14 with a 768-dimensional joint space;
- each full stored FSC image rather than RichCount-style visual prompts/crops;
- the deterministic 1,000-image source subset and saved prompt bank;
- a deterministic 900/100 train/validation split with seed 3407;
- class, detailed, and generalized positive prompt modes;
- trainable visual FFN and text Adapter modules.

RichCount differs in material ways, including CLIP ViT-B/16 / different
dimensionality, visual prompts/cropped regions, and deeper reported FFN/Adapter
variants.

### Repository alignment-stage naming

The repository further splits the RichCount-inspired Stage-1 alignment
procedure into two internal sub-stages:

- Stage-A: train the visual FFN while CLIP is frozen.
- Stage-B: freeze the selected visual FFN and train the text Adapter
  against that visual representation.

These names are repository-specific sub-stages.

They must NOT be confused with the RichCount paper's major stages:
- RichCount Stage 1: Visual-Text Alignment
- RichCount Stage 2: Text-Based Counting

### 8. Alignment smoke and diagnostics — ALIGN-000 through ALIGN-002

ALIGN-000 was a shallow one-epoch raw-Euclidean FFN + Adapter smoke. FFN and
Adapter each had approximately 1.18M trainable parameters; frozen CLIP had
approximately 427.6M parameters. The FFN appeared collapsed/almost random, and
the Adapter was then trained on that already-damaged visual space. This is
useful debugging history but obsolete as scientific evidence against FFNs,
Adapters, or RichCount alignment.

ALIGN-001 introduced L2 normalization only for final projected image/text
embeddings and used normalized Euclidean diagnostics. Representative frozen
CLIP validation metrics were:

| Loss | Pairwise | R@1 | Class-aware R@1 | Sample-aware R@1 | Gap | `d_pos` | `d_neg` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ≈0.760 | ≈0.990 | ≈0.710 | ≈0.883 | ≈0.520 | ≈0.103 | ≈1.232 | ≈1.335 |

Per-prompt R@1 was approximately 0.87 class, 0.70 detailed, and 0.56
generalized. Frozen CLIP therefore already has strong global rich-text matching.

ALIGN-002 measured the exact randomly initialized FFN used for training. It
destroyed the useful CLIP geometry: pairwise accuracy was approximately
0.46–0.48, R@1 approximately 0.02–0.04, and separation gap around zero or
slightly negative. Normalized random-pair distances were near `sqrt(2) ≈ 1.414`;
margin choices around 1.2/1.3 changed negative-hinge activation. The old
conclusion "FFN cannot learn" is obsolete because later five-epoch runs learned.

### 9. Controlled Stage-A FFNs — ALIGN-003 and ALIGN-004

Both final Stage-A runs held approximately the following controls fixed:

- frozen local CLIP ViT-L/14; joint dimension 768;
- 1,000-source-sample subset, 900/100 split, all seeds 3407;
- batch size 32, 5 epochs, FFN learning rate `1e-4`, dropout 0.1;
- final-embedding L2 normalization, normalized Euclidean distance, margin 1.3;
- class/detailed/generalized modes;
- Stage A only; no Adapter training;
- best checkpoint selected by validation overall contrastive loss.

ALIGN-003 used the minimal FFN:

```text
Linear(768,768) -> ReLU -> Dropout -> Linear(768,768)
```

It had approximately 1,181,184 trainable parameters and clearly learned over
five epochs. Final reported overall metrics were:

| Loss | Pairwise | R@1 | Class-aware R@1 | Sample R@1 | Gap | `d_pos` | `d_neg` | Negatives below margin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ≈0.389 | ≈0.883 | ≈0.307 | ≈0.483 | ≈0.150 | ≈0.168 | ≈0.801 | ≈0.969 | ≈1.00 |

Per-prompt R@1 was approximately 0.63 class, 0.18 detailed, and 0.11
generalized. The low loss partly came from compressing both positive and
negative distances; nearly all negatives remained inside the margin. It is a
valid candidate, but low scalar loss alone is not evidence of healthier
geometry.

ALIGN-004 used the Figure3+BN FFN:

```text
Linear(768,768) -> ReLU -> Dropout
-> Linear(768,768) -> ReLU -> Dropout -> BatchNorm1d(768)
```

It had 1,182,720 trainable parameters. The executed notebook records the final
epoch/best-checkpoint validation metrics:

| Loss | Pairwise | R@1 | Class-aware R@1 | Sample R@1 | Gap | `d_pos` | `d_neg` | Negatives below margin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7659 | 0.9500 | 0.3900 | 0.5667 | 0.2633 | 0.1686 | 1.2350 | 1.4036 | 0.0900 |

Per-prompt metrics were approximately:

| Prompt | R@1 | Class-aware R@1 |
| --- | ---: | ---: |
| Class | 0.58 | 0.58 |
| Detailed | 0.37 | 0.66 |
| Generalized | 0.22 | 0.46 |

Accepted state: ALIGN-004 is more promising than ALIGN-003 for rich-prompt
alignment: it has higher retrieval and pairwise alignment, much stronger
detailed/generalized retrieval, and healthier distance scale/separation. Its
higher scalar loss does not make it worse; ALIGN-003 lowered loss partly through
global distance compression.

Neither trained FFN "beats CLIP" on generic/global retrieval. Frozen CLIP has
approximately 0.99 pairwise accuracy, 0.71 R@1, and 0.883 class-aware R@1,
versus approximately 0.95, 0.39, and 0.567 for ALIGN-004. Both trained FFNs
increase positive-vs-negative separation gap but degrade generic retrieval
neighborhoods. Their relevance must ultimately be judged by downstream
counting/spatial evidence, not retrieval alone.

## Current scientific interpretation

The hypothesis that rich prompts fail simply because CLIP cannot understand
detailed/generalized language is obsolete. Frozen CLIP already shows strong
global alignment. The current, unproven hypothesis is that the bottleneck lies
between global semantic understanding and the local/spatial representations
needed by T2ICount's density-prediction pipeline.

Possible failure locations include U-Net cross-attention/spatial grounding,
HSCM semantic correction, the counter/density output, or the mismatch between
CLIP-based Stage-1 alignment and T2ICount's diffusion-based visual path. No one
location has been proven.

## Exact current stopping point

```text
direct rich prompts -> failed
multi-prompt supervision + consistency
  -> outputs can agree without becoming accurate
  -> stop raw lambda_cons tuning
RichCount-inspired Stage-1 investigation
  -> raw smoke/debug
  -> normalization, margin, and initialization diagnostics
  -> frozen CLIP has strong global rich-text alignment
  -> ALIGN-003 minimal FFN, 5 epochs
  -> ALIGN-004 Figure3+BN FFN, 5 epochs
CURRENT STOP: ALIGN-004 finished
```

Not yet done:

- no paper-like Stage-B Adapter has been trained from ALIGN-004;
- no downstream T2ICount counting experiment has used ALIGN-004;
- no decisive diagnostic has localized failure to U-Net attention, HSCM, or
  density output;
- DUMLO remains parked.

No proposed future experiment is a completed result.

## Do not repeat without a new hypothesis

1. Official-checkpoint direct detailed/generalized inference.
2. Raw multi-prompt `lambda_cons=0` training.
3. The existing `lambda_cons` values 0.1, 1, and 100.
4. Large consistency weights merely to force output equality.
5. Blind `lambda_cons` grid search on the same representation.
6. The old one-epoch FFN as evidence that an FFN cannot learn.
7. The old Adapter smoke as evidence that an Adapter fails.
8. The minimal five-epoch FFN merely to prove it learns.
9. The tested DUMLO `lambda_count` tuning.
10. The tested DUMLO `lambda_OT` tuning.
11. DUMLO + SSIM.
12. DUMLO + full original regression loss.
13. Naive DUMLO replacement for every training sample.
14. Combining DUMLO and RichPrompt at the current stopping point.
15. Claiming ALIGN-004 improves counting before downstream validation.

## Open questions

1. Why does T2ICount count poorly with rich prompts when frozen CLIP already
   shows good global image-text alignment?
2. Does the failure occur at spatial grounding / U-Net attention, HSCM, or
   density prediction?
3. What is the correct role of RichCount's visual FFN in a T2ICount system whose
   visual pathway is diffusion-based rather than CLIP-based?
4. Should a paper-like text Adapter be trained after ALIGN-004?
5. If an aligned representation is integrated into counting, does
   `lambda_cons` become useful again?
6. If DUMLO is reopened, should it be restricted to positive samples with valid
   points while synthetic text-negatives retain the original T2ICount loss?

No next experiment has been scientifically approved yet.
