# RP-010 workflow and verification

Start with [the audit](RP010_CONTROLLED_TRAINING_AUDIT.md). Entry points:
`notebooks/RP010_controlled_supervision.ipynb`, `tools/train_rp010.py`,
`configs/rp010a.json`, `configs/rp010b.json`. No historical trainer/model/loss/
inference file is modified. No full training, commit, push, or conclusion update
was performed for preparation.

## What is paired

Both arms: fresh common full-model initialization per seed; SD/CLIP pretrained
assets; exact 1,000 original FSC147 train anchors (subset seed3407); identical
full-train donor pool; deterministic per-epoch ordering and per-anchor
augmentation; batch1/workers0; all three compatible-positive supervised forwards
averaged /3; unchanged class-only negatives/mosaics; unchanged density/RRC losses;
lambda_cons0; identical zero-weight consistency graph; AdamW groups; ten epochs;
10,000 optimizer updates; validation epochs7/8/9; strict val MAE+RMSE selection;
production class-name patch inference.

Only A's auxiliary strings/masks (fixed RP-009 wrappers) versus B's saved
detailed/generalized strings/masks differ. All alternatives are used on every
compatible positive; there is no random prompt-selection schedule. Class
supervision is the same, including the compatible-class /3 weighting.

Remaining confounders: length, lexical composition and semantic specificity vary
together; historical RP-003 initial weights/runtime remain uncertified; S has
image overlap with training; one seed cannot establish stable superiority.
Cross-machine numerical equality is not claimed. GPU peak memory is unverified.

## Required external assets

Under `/content/drive/MyDrive/T2ICount-assets/`:

- `pretrained/sd-v1-5/v1-5-pruned-emaonly.ckpt`
- `pretrained/clip-vit-large-patch14/` including weights and tokenizer files
- `datasets/FSC147/FSC_147/{Train_Test_Val_FSC_147.json,annotation_FSC147_384.json,ImageClasses_FSC147.txt}`
- `datasets/FSC147/images_384_VarV2/` (all train/val/test images)
- `datasets/FSC147/gt_density_map_adaptive_384_VarV2/` (full train donor densities)
- `prompts/fsc147_train1000_seed3407_gemma4_v3.json`
- For final secondary evaluation only: `prompts/fsc147s_gemma4_26b_v3.json`.

The repository supplies `FSC-147-S.json`. Existing trained BASE/RP checkpoints are
**not** RP-010 initialization assets. No prompt generation/API is used.
The notebook can extract only the dataset from the existing asset ZIP into local
scratch, with stable archive timestamps for safe repeat runs. It does not need a
source ZIP or duplicate pretrained weights. Keep external data immutable. Dataset
provenance includes metadata byte hashes and path/size/mtime fingerprints of
RGB/density payloads; it does not claim every payload byte was hashed. SD/CLIP
files are hashed once during shared preparation, then their stat identities are
checked; the common initialization is hashed on each new command for integrity.

## Notebook sequence (VS Code with Colab runtime)

1. Review these eight new source files. They must be committed and pushed by the
   researcher to `rich-prompt-alignment` before GitHub setup can find them:
   `docs/RP010_CONTROLLED_TRAINING_AUDIT.md`, `docs/RP010_RUNBOOK.md`,
   `configs/rp010a.json`, `configs/rp010b.json`, `utils/rp010.py`,
   `tools/train_rp010.py`, `tests/test_rp010.py`,
   `notebooks/RP010_controlled_supervision.ipynb`. Keep earlier unrelated notebook
   edits/DIAG-002 work separate. Never stage runtime artifacts or prompt banks.
2. Open the **new** notebook from that pushed revision. Prefer a Colab L4 runtime,
   with sufficient host RAM and Drive capacity. L4 is a candidate, not a measured
   memory guarantee. Set EXPECTED_COMMIT to the reviewed full SHA. Pin the same
   source, package versions, GPU type and data paths for both arms and resumes.
3. Set `RUN_STAGE=True`, `STAGE='prepare'`, seed3407. Keep
   `ENABLE_FULL_TRAINING=False`. Run all cells. Preparation validates exact bank
   coverage/joins, creates one shared seeded CPU initialization, and writes its
   manifest and per-prompt token report. Full model initialization requires at
   least8GiB free host RAM as a safety headroom policy; this is not a measured peak.
   Model-derived weights+gradients+Adam byte accounting is recorded afterward.
   The conservative budget assumes gradients/two Adam moments for every trainable
   parameter (some unused SD parameters may allocate neither); it is not a strict
   lower bound. Three-forward activations, buffers and optimizer temporaries
   require extra memory.
4. Run `STAGE='smoke'`, first ARM A then ARM B, separately. Both must report
   **passed** with matching inputs, routing and budgets. These are real optimizer
   updates to throwaway models, never resumable full-run checkpoints. The smoke
   validates one val image, saves/reloads full state and does a second update.
   Its own large temporary state is deleted on success; report/log remain on Drive.
5. Review the audit, prompt-token reports and both smoke reports. Only then set
   `STAGE='train'`, ARM A, and explicitly `ENABLE_FULL_TRAINING=True`. Repeat
   separately for B, same seed/PAIR_ID/runtime/source/assets. The CLI also requires
   `--allow-full-training` and validates both smoke reports before any full run.
6. To resume, keep that arm's settings and set `RESUME=True`. `latest.tar` is
   replaced atomically after each complete epoch, including validation. An
   interrupted epoch is replayed; there is no mid-epoch claim. A cross-arm, seed,
   init, bank/data/config/source/runtime mismatch fails before state loading.
   If no epoch completed, preserve the old folder and use a new PAIR_ID, rerunning
   the two smokes. If initial preparation was interrupted, preserve its partial
   directory outside the configured init path before retrying; do not overwrite it.
7. On completion, verify ten epoch records, 10,000 updates, selected best `.pth`
   and `latest.tar`. Then set `STAGE='evaluate'` for each arm. It uses only the
   saved validation selection, class inference, and separate standard-test/S
   tables. An existing evaluation directory is preserved, including partial runs;
   inspect failures before choosing a new reviewed output namespace. Never choose
   another checkpoint or tune settings from test/S results.
8. Run `STAGE='compare'` once. It checks both complete epoch schedules, input/
   target/routing digests, initialization/source/runtime identities and exact
   evaluation joins. Paired B-minus-A error/bias/RMSE deltas and a seed3407,
   2,000-resample paired image-bootstrap MAE interval are descriptive results for
   researcher review. Repeat seeds3408/3409 with separately shared initializations
   before a stable training-strategy claim. Keep the subset seed3407 fixed.

The safe notebook default executes no external setup or job (`RUN_STAGE=False`),
and full training has a separate False flag. Progress is streamed from unbuffered
Python subprocesses. Nothing disconnects the runtime automatically.

## CLI examples (Colab terminal, from repository root)

The notebook assembles these commands; manual cell assembly is unnecessary.

```bash
python -u tools/train_rp010.py prepare --asset-root /content/drive/MyDrive/T2ICount-assets
python -u tools/train_rp010.py smoke --config configs/rp010a.json --asset-root /content/drive/MyDrive/T2ICount-assets
python -u tools/train_rp010.py smoke --config configs/rp010b.json --asset-root /content/drive/MyDrive/T2ICount-assets
```

Use `--data-dir` / `--prompt-bank` for explicit existing local asset locations.
Both overrides are applied identically to the paired control configuration.
`--seed 3408` selects a separate initial checkpoint and output tree.
`--pair-id pair_v2` creates a new paired output namespace; it does not overwrite
an existing run or change the common seed-specific initialization.
The full command additionally requires `train --allow-full-training`; run it
only after the above review/smoke gate. Evaluation is an independent `evaluate`
mode; comparison uses `compare --comparison-output <new-directory>`.

## Outputs

`checkpoints/rp010/init_seed3407/`: `initial.pth`, `initial_manifest.json`,
`prompt_tokens.csv` (exact texts, raw/retained CLIP lengths, truncation).

`outputs/rp010/seed3407/pair_v1/RP-010{A,B}/`:

- `manifest.json`: arm/shared settings, source-file hashes/commit, runtime,
  subset, bank/data identities, initialization hash.
- `train.log`: live training and validation progress/losses.
- `latest.tar`: model/optimizer/RNG/next epoch, step counts, control history,
  best selection, full resume guard. Keep enough Drive space for an atomic
  checkpoint replacement (old + temporary state) and selected weights.
- `best_epoch*.pth`: new validation improvements only; no automatic test call.
- `epoch_records.json`, `selection.json`, `completed.json`.
- `evaluation_class/{manifest.json,fsc147_test_class.csv,fsc147s_class.csv,summary.json}`.

Sibling `RP-010A_smoke/`, `RP-010B_smoke/` contain smoke reports/logs;
`paired_analysis/` contains paired tables and summary. New output folders are
created exclusively; matching completed-epoch resume is the only training reuse.
FSC147-S GT groups are1,2–5,6–20,>20 (unexpected counts use an explicit other
group); signed error=prediction−GT. Dot/localization metrics remain unavailable.

## Verification performed locally (2026-10-29)

- 125 CPU tests passed:23 RP-010 tests plus102 shared trainer/checkpoint/subset/
  inference/evaluator regression tests. Tested real production loss/three-forward
  routing/backward/AdamW **on a tiny CPU mock model**, not the SD model.
- Safe-default notebook cells compiled and executed top-to-bottom without setup,
  network, Drive, training or GPU model initialization. Live Colab execution and
  full dependency installation remain unverified.
- Actual local bank revalidated:1,000 exact records, subset seed3407, existing bank
  and ordered subset SHA256s match the audit. All train/donor RGB/density files
  and standard val/test images exist; official splits are disjoint.
- Real ObjectCount data paths, both arms: first32 sample transforms at epochs0
  and9 match exactly, including density/class masks and compatibility. Compatible
  counts were10/32 and18/32. Full1,000 epoch permutations are fingerprinted;
  this prefix check is not a claim to have trained/evaluated1,000 model samples.
- Actual CLIP raw-token mean/max: class1.261/3, generic1 4.261/6,
  generic2 5.261/7, detailed15.528/33, generalized15.267/33. No truncation in this
  bank. Exact per-image reports remain ignored local runtime artifacts under
  `outputs/rp010_cpu_validation_20261029_v3/`.
- Local hardware: RTX4060 Laptop8GiB (~6.9GiB GPU free). Real preparation stopped
  at the host-RAM preflight (**1.35GiB available**, below8GiB safety headroom).
  No real Count initialization/forward/backward/update/validation/resume ran.
  Both arm smoke preflights recorded **pending** in the separate external
  `D:/T2ICount-assets/outputs/rp010/seed3407/local_preflight_20261029/` tree.
  No initial-weight hash, real GPU smoke pass, peak memory or training-time
  estimate is claimed. Run both smokes on the intended runtime before training.

CPU command used:

```bash
python -m pytest tests/test_rp010.py tests/test_checkpoint_resume.py tests/test_rich_prompt_phase2b.py tests/test_train_samples.py tests/test_smoke_train_samples.py tests/test_t2icount_checkpoint_compatibility.py tests/test_infrastructure.py tests/test_progress_reporting.py tests/test_rich_prompt_evaluator.py -q
```

The existing local torchvision image-extension warning is unchanged; PIL image
loading/tests passed. It is not evidence of a Colab Torch installation failure.
