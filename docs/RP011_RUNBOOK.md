# RP-011: two-epoch warm-start HNW pilot

Preparation only. Full pilot training has not been launched. Preflight must pass
on the intended GPU runtime before either arm can start.

## Research specification

- **Question:** Does HNW improve short-class counting after two epochs of
  additional RP-010B rich-prompt supervision from the same selected model?
- **Hypothesis:** Giving retained high-similarity negatives larger relative
  gradients reduces false activation and overcounting. This is unproven.
- **Changed variable:** `rrcl_hnw=False` (control) versus `True` (HNW).
- **Controls:** Both arms load exactly RP-010B `best_epoch9.pth`, weight-only.
  Preserve its rich prompt bank, 1,000 anchors, subset/training seeds 3407,
  full 3,659-image training donor pool, three-forward rich supervision,
  consistency weight 0, density/RRCL masks, coefficients, transforms, AdamW
  parameter groups, crop384, concat224, stride384, batch1, workers0, and FP32.
  Each arm starts a fresh optimizer. RP-010B had no scheduler, so neither arm
  has a scheduler or scheduler state. No checkpoint resume is used.
- **Budget/schedule:** Two new local epochs (0 and 1), 1,000 optimizer updates
  per epoch, 2,000 per arm, 4,000 total. Reuse RP-010's deterministic epoch and
  per-anchor augmentation RNG at these local epochs in both arms; do not resume
  the RP-010 epoch/RNG/optimizer state. Validation follows each local epoch.
- **Metrics:** Full standard FSC147 validation (1,286 images), short class-name
  inference, MAE, RMSE, and mean signed bias = prediction minus ground truth.
  Report HNW minus control differences for each epoch on the same images.
- **Decision rule:** Advance only if validation MAE improves without worse
  RMSE; bias is diagnostic. Researcher review is required for scientific
  interpretation. Keep the original best-checkpoint rule, minimum MAE+RMSE.
  No standard-test or FSC147-S evaluation is included in this pilot.

The two effective configurations differ only in HNW. Arm names and output/log
paths are bookkeeping. Relative to RP-010B, only the user-requested warm start,
fresh optimizer, two-epoch budget, validation schedule, and new output location
are changed. `configs/rp010b.json` itself is never modified.

## Source checkpoint

The actual [RP-010B selection record](https://drive.google.com/file/d/1VB6kR33Eo_bPMGV4i7FWoHvGkAXCfg2x/view)
selects [best_epoch9.pth](https://drive.google.com/file/d/1JtdAfBwhmpsNoXxZw_mYa6UDYq18vDP9/view):

```text
SHA256 a85c92790e29fe6c1a2c5f0232e7ccacaf205d4cc0e49eaaa6fc7400d6026513
size   4,263,114,751 bytes
epoch  9
rule   val_mae_plus_rmse
```

`--rp010-run` must contain `manifest.json`, `selection.json`, and that exact
checkpoint. Source files are read-only. The runner checks the recorded RP-010B
configuration, selected epoch, checkpoint SHA256, and portable data identities.
Model loading uses the production strict key/shape loader, including its narrow
legacy nonpersistent-buffer reconciliation. Optimizer/epoch/RNG state is never
loaded from RP-010. Output paths overlapping the source run are rejected.

## Commands on the intended GPU runtime

Run from the repository root containing the completed HNW patch and RP-011
files. Use the same package versions and GPU for both arms. The actual RP-010B
manifest records Linux, Python3.13.16, Torch2.11.0+cu130,
torchvision0.26.0+cu130, NumPy2.1.3, transformers4.57.6,
OmegaConf2.4.0, Pillow11.3.0, OpenCV4.14.0.94, NVIDIA L4.
No dependency installation or GPU job is triggered merely by importing the runner.

The original run used local extracted dataset/pretrained assets and a Drive
prompt bank. Preserve those existing assets; these flags only locate them.

```bash
ASSETS=/content/T2ICount-assets
DRIVE=/content/drive/MyDrive/T2ICount-assets
COMMON=(
  --asset-root "$ASSETS"
  --rp010-run "$DRIVE/outputs/rp010/seed3407/pair_v1/RP-010B"
  --prompt-bank "$DRIVE/prompts/fsc147_train1000_seed3407_gemma4_v3.json"
  --output-dir "$DRIVE/outputs/rp011/seed3407/pair_v1"
)

python -u tools/train_rp011.py preflight "${COMMON[@]}"
```

Preflight runs arms sequentially. Each strictly loads the same model, records a
full starting-state fingerprint, uses the same first compatible sample in the
fixed first32 anchors, and executes the existing three supervised forwards and
one backward pass. It checks finite final loss and gradients, nonzero gradient
norm, unchanged model/optimizer state (zero optimizer steps), and GPU peak
allocated/reserved memory. Starting weights, sample, data trace and prompts must
match. The actual pilot creates fresh models/optimizers again.

Training remains gated until `preflight/report.json` has `status=passed` for the
current checkpoint/data/config/source/runtime. Review that report first. The
following commands are supplied for later execution; they have not been run:

```bash
python -u tools/train_rp011.py train "${COMMON[@]}" --arm control --allow-training
python -u tools/train_rp011.py train "${COMMON[@]}" --arm hnw --allow-training
python -u tools/train_rp011.py compare "${COMMON[@]}"
```

Existing arm directories are preserved; no resume option is provided. An
interrupted arm is rerun from the same source weights in a new RP-011 output
directory after a new paired preflight. No RP-010 artifact is overwritten.

## Outputs and memory

- `preflight/report.json`: readiness, both smoke measurements, starting-state
  fingerprints, source/data/runtime guard, effective configs and step budget.
- Each arm: `manifest.json`, `train.log`, `val_epoch0.csv`, `val_epoch1.csv`,
  `epoch_records.json`, selected `best.pth`, and `completed.json`.
- `paired_validation.csv/json`: epoch-specific MAE/RMSE/bias and HNW-minus-control
  deltas. Comparison checks training data traces and validation image/class/GT
  joins, coverage and metrics. It records raw observations, not conclusions.

The real RP-010B L4 smoke report recorded 22,715,654,144 peak allocated bytes
(21.16GiB), with two throwaway optimizer updates. This is historical capacity
evidence, not an RP-011 measurement. A 24GiB L4 is the existing candidate;
actual RP-011 peaks remain pending. Do not change precision, crop, freezing or
losses to fit an 8GiB GPU.

RP-011's zero-step smoke does not allocate Adam moments. Its training estimate
adds two moment tensors for every trainable parameter to the smoke reserved
peak, and also reports 20% suggested headroom. This is conservative and is not
a measured full-training peak. At least8GiB free host RAM is required before
the existing full-model/checkpoint initialization path is attempted.

## Local preparation evidence (2026-11-08)

- Downloaded checkpoint moved from `D:/Downloads/best_epoch9.pth` into the
  ignored `outputs/rp011_source/RP-010B/` input copy. Its byte hash matches the
  real selection record. Drive originals and historical RP-010 files were not
  changed. The earlier350-byte ZIP contained only `selection.json` and was
  extracted separately.
- The actual checkpoint deserialized successfully with CPU memory mapping
  (Torch2.13.0+cpu):1,160 state entries,4,262,594,212 tensor bytes. This is not a
  strict Count model load or GPU smoke. Report:
  `outputs/rp011_source/checkpoint_cpu_load.json`.
- Its unet/decoder state occupies3,633,697,556 bytes. Counting one gradient and
  two Adam moments gives a conservative state-based estimate of15,163,686,880
  bytes (14.12GiB) before activations/workspace; state buffers are not separated
  from parameters in this estimate. It exceeds the local GPU's total8GiB.
- Actual local data validation matched the recorded RP-010B config, prompt-bank
  SHA, ordered1,000-image subset, official split/class/annotation hashes, and
  full3,659-image donor pool. Payload bytes are assumed immutable; the historical
  manifest only supplies path/size/mtime fingerprints for RGB/density files.
  Report: `outputs/rp011_source/data_validation.json`.
- 161 CPU tests passed; one pre-existing RP-010 notebook test was deselected
  because `notebooks/RP010_controlled_supervision.ipynb` was already deleted in
  the working tree. The existing torchvision image-extension warning remains.
  Tiny-model CPU smoke fixtures do not certify real-model GPU readiness.
- Local preflight report: `outputs/rp011_local_preflight/preflight/report.json`.
  Status is **pending**:1,153,904,640 bytes (~1.07GiB) free host RAM, below the
  existing8GiB initialization threshold. The local GPU has8,585,216,000 total
  bytes (~8GiB), with7,451,181,056 free (~6.94GiB) at this check. GPU smokes,
  strict full-model loads and equal starting full-model fingerprints remain
  pending on a suitable runtime. No pilot optimizer updates or counting
  evaluation ran.

```bash
python -m pytest tests/test_rp011.py tests/test_rrcl_hnw.py tests/test_rp010.py tests/test_checkpoint_resume.py tests/test_rich_prompt_phase2b.py tests/test_train_samples.py tests/test_smoke_train_samples.py tests/test_t2icount_checkpoint_compatibility.py tests/test_infrastructure.py tests/test_progress_reporting.py tests/test_rich_prompt_evaluator.py -q -k 'not test_notebook_executes_all_cells_in_safe_default_order'
```

## Colab notebook and upload snapshot

Open `notebooks/RP011_Colab.ipynb` in Colab. Upload the generated
`outputs/rp011_colab_upload/RP011_source.zip` into
`MyDrive/RP011_upload/`. A copy of the notebook is in that same local upload
directory. The ZIP contains the current source, including the uncommitted HNW
and RP-011 preparation, a per-file hash manifest, the Colab requirements and
protocol files. It contains no dataset, prompt bank, model checkpoint or Git
history. No push is required to use this snapshot.

Default Drive layout:

```text
MyDrive/
  RP011_upload/
    RP011_Colab.ipynb
    RP011_source.zip
  T2ICount-assets.zip
  T2ICount-assets/
    prompts/fsc147_train1000_seed3407_gemma4_v3.json
    outputs/rp010/seed3407/pair_v1/RP-010B/
      best_epoch9.pth
      manifest.json
      selection.json
    outputs/rp011/seed3407/pair_v1/    # new outputs, created by notebook
```

Reuse existing Drive inputs rather than uploading duplicate RP-010 artifacts.
The original asset ZIP must supply `pretrained/sd-v1-5/v1-5-pruned-emaonly.ckpt`,
`pretrained/clip-vit-large-patch14/` (weights, config and tokenizer), and
`datasets/FSC147/{FSC_147,images_384_VarV2,gt_density_map_adaptive_384_VarV2}`.
These may be at ZIP root or below `T2ICount-assets/`. If the archive is absent,
the notebook can copy those existing expanded subtrees from the Drive asset
directory. All model/data extraction goes to `/content/T2ICount-assets`; the
source RP-010B model, manifest, selection and bank are only read from Drive.

Cells mount Drive, verify/extract the source snapshot, install the existing
Colab dependency bridge with constraints protecting the installed core stack,
check GPU/host capacity, prepare local input assets, run the existing input
validators, and execute the existing paired preflight. The notebook creates a
local-only Git snapshot under `/content/T2ICount-RP011` because the runner's
unchanged provenance code requires Git; the original repository is untouched.

Choose A10040GB for headroom or L424GB. The notebook rejects GPUs with less
than22GiB total or21GiB free VRAM and keeps the existing8GiB free-host-RAM
requirement. The runner then performs its own conservative memory checks and
actual paired forward/backward tests. These initial capacity thresholds do
not constitute an RP-011 GPU smoke pass.

Any failed/pending preflight raises and leaves `PREFLIGHT_PASSED=False`. The
separate pilot cell defaults to `RUN_PILOT=False`. When explicitly enabled
after review, it invokes only the supported `train --arm control`,
`train --arm hnw` (both with `--allow-training`), and `compare` commands with
the same common paths. There is no new runner flag or training-logic change.
Logs, package snapshot, input validation and all runner outputs are saved to
the new RP-011 Drive directory. Use a fresh `PAIR_ID` for an interrupted run;
no automatic resume or overwrite is provided.

Local verification is limited to notebook syntax, nested subprocess syntax,
source-ZIP hashes, supported runner arguments and disabled/failed-preflight
gates using mocks. Live Colab installation and real GPU execution remain
unverified; no local training or GPU job is launched by notebook preparation.
