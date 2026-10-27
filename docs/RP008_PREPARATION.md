# RP-008: Matched Class-Only Control — preparation only

Prepared 2026-10-27 on `rich-prompt-alignment`. No full training, evaluation, or
scientific conclusion is authorized by this document. Existing experiment
records remain unchanged.

## Experiment definition

- **Question:** Does RP-003's class-name counting gain depend on rich semantic
  content rather than the three-forward procedure?
- **Hypothesis:** With identical class text/masks/GT, deterministic and
  state-preserving forwards should give the baseline objective and gradients
  within numerical precision. Stochastic or stateful forwards may differ.
- **Changed variable:** Relative to RP-003, replace only detailed/generalized
  auxiliary captions and masks with the original class caption and mask.
  Retain three forwards and `lambda_cons=0`.
- **Controls:** The same validated bank and compatibility routing; FSC147
  1000-sample subset, subset/training seeds 3407; 10 epochs, batch size 1;
  original AdamW parameter groups, learning rates and weight decay;
  augmentation, density/RRC losses, validation/evaluation, and best-checkpoint
  selection. Negatives and mosaics remain class-only.
- **Metrics:** Preflight batch losses, gradient-vector difference L2 and
  relative L2 across every trainable parameter, per-parameter differences,
  unused-gradient mismatches, RNG progression, and model-state changes.
  Later counting comparisons, if reviewed, use class-name MAE/RMSE on the fixed
  protocols; rich-prompt/alignment behavior remains diagnostic.
- **Decision rule:** Preflight compares baseline, a baseline replay, and
  repeat-class from identical parameters, buffers, inputs, and RNG. Loss and
  global/per-parameter gradient L2 differences must each satisfy
  `difference <= 1e-6 + 1e-5 * reference_magnitude`. Gradients must be finite,
  nonzero overall, and have the same unused-parameter pattern. Extra RNG
  progression or registered-state changes prevent a skip recommendation.
  If all checks pass for both requested batches, recommend **skipping full
  10-epoch RP-008** and propose a template-diversity control for review. Otherwise
  report the effect for the researcher to decide; do not launch training.

The finite-batch check is not proof of equality over an entire training
trajectory and cannot establish that rich semantics caused RP-003's gain.

## Colab preparation and provenance

Use the existing asset/bootstrap environment with the prepared source files
available under `/content/T2ICount-Implementation` on `rich-prompt-alignment`.
Fetch/pull the pushed preparation commit on this branch before invoking either
command. Verify that `train.py --help` lists `--rich-supervision-mode` and that
`tools/preflight_repeat_class.py` exists in the Colab checkout.
Do not execute the DUMLO checkout/training/disconnect cells in
`notebooks/train_colab.ipynb` for this experiment.

The local bank validator passed against `D:/T2ICount-assets/datasets/FSC147`:

- selected count: 1000;
- ordered subset fingerprint:
  `sha256:dd96b36bf15013e194b1a8ece06452a19822aae028fc00c0de019cbb7a311f24`;
- bank fingerprint:
  `sha256:c3ec587b37fd9f351cdedbce888b26e0ba7a020de956c4ef18202a1268e7e32f`;
- protocol: `rich-prompt-phase1-v3`.

Preflight revalidates the bank/subset before constructing the model. Compare
its reported fingerprints with these values; do not substitute another bank.
The bank is required even though auxiliary forwards do not use its rich text.

`BASE-002`'s historical manifest records `start_val=7`, `val_epoch=1`.
The commands below use that schedule and the current original defaults
`concat_size=224`, checkpoint interval 5. The original RP-003 command,
initial-weight hash, full runtime, and artifact location are not preserved in
its ledger entry. Consequently exact historical initialization/runtime/schedule
matching cannot be certified from this checkout. Review surviving RP-003 logs
before deciding on full training. Do not start from a trained BASE-002,
RP-003, or official counting checkpoint: training builds from the SD/CLIP assets
and the seeded initial counting decoder, matching the current training path.

## Exact preflight command (Colab Bash cell)

Run after mounting Drive, extracting the existing assets, and installing the
normal project dependencies. This performs only bounded forward/backward
diagnostics; there are **zero optimizer steps** and no validation/test calls.
Use a new report/run name when rerunning; an existing JSON is rejected.

```bash
%%bash
set -euo pipefail
cd /content/T2ICount-Implementation
test "$(git branch --show-current)" = rich-prompt-alignment
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset SSLKEYLOGFILE
python -u tools/preflight_repeat_class.py \
  --output /content/drive/MyDrive/T2ICount-assets/outputs/rp008_preflight.json \
  --probe-batches 2 --max-scan-batches 32 --scratch-dir /content \
  --asset-root /content/T2ICount-assets \
  --config configs/v1-inference.yaml \
  --save-dir /content/drive/MyDrive/T2ICount-assets/checkpoints/rich_prompt/preflight \
  --content rp008_preflight \
  --rich-prompt-bank /content/drive/MyDrive/T2ICount-assets/prompts/fsc147_train1000_seed3407_gemma4_v3.json \
  --rich-supervision-mode repeat-class --rich-consistency-weight 0 \
  --seed 3407 --train-samples 1000 --train-subset-seed 3407 \
  --batch-size 1 --epochs 10 --start-epoch 0 --start-val 7 --val-epoch 1 \
  --num-workers 0 --crop-size 384 --concat-size 224 --downsample-ratio 8 \
  --stride 384 --lr 5e-5 --weight-decay 5e-4 --checkpoint-interval 5
```

The diagnostic scans up to 32 normally augmented shuffled training batches to
find two compatible positives; it never forces the augmentation branch. Each
selected batch is reused unchanged in all three passes. The same model instance
is restored from a streamed tensor snapshot before each pass. Sequential extra
forwards retain normal stochastic behavior; RNG is not reset between them.
The baseline replay checks repeatability under the runtime's existing cuDNN
settings. This is a bounded probe, not a training epoch over the dataset.

Local scratch holds a full model tensor snapshot plus one reference-gradient
set (several GB). Files are streamed per tensor to limit RAM usage and removed
on normal exit. GPU memory for three forward graphs is still required, as in
RP-003. Do not reduce precision, disable dropout, cache features, or change
batch size to make the experiment fit without a reviewed protocol change.

## Expected preflight outputs and interpretation

- `outputs/rp008_preflight.json` on Drive: both losses and their signed
  difference; baseline-replay and repeat-class gradient norms, absolute/relative
  vector differences, per-parameter details; RNG and per-forward buffer/parameter
  mutation checks; recommendation, args, bank/subset provenance, source hashes,
  Git state, and Torch/CUDA/cuDNN details.
- `checkpoints/rich_prompt/preflight/rp008_preflight/train.log`: setup and probe
  logs only. No model checkpoint is saved by preflight.
- The registered-state checks include nonpersistent buffers. The U-Net's
  transient attention store is reset at each forward and before each probe;
  this cache is not learned state. Arbitrary Python attributes are not
  exhaustively compared. No claim is made about all possible hidden state.
- `SKIP` means the measured compatible batches passed the stated numerical,
  RNG, and state checks. Proceed to designing a template-diversity control for
  researcher review, not to automatically implementing or training it.
- `REVIEW` means loss/gradient differences, state/RNG effects, nonfinite values,
  or insufficient compatible batches need interpretation. A changed RNG endpoint
  can affect subsequent training even when this batch's gradients agree.
  Two compatible batches not found yields exit code 2 and an inconclusive JSON.
- No actual T2ICount GPU preflight result is included in this preparation.
  CPU fixture tests validate the diagnostic's deterministic/stochastic/stateful
  detection paths; they cannot establish real-model equivalence.

## Exact full training command — prepared only, DO NOT RUN yet

```bash
%%bash
set -euo pipefail
cd /content/T2ICount-Implementation
test "$(git branch --show-current)" = rich-prompt-alignment
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset SSLKEYLOGFILE
python -u train.py \
  --asset-root /content/T2ICount-assets \
  --config configs/v1-inference.yaml \
  --save-dir /content/drive/MyDrive/T2ICount-assets/checkpoints/rich_prompt/rp008 \
  --content repeat_class_1000x10 \
  --rich-prompt-bank /content/drive/MyDrive/T2ICount-assets/prompts/fsc147_train1000_seed3407_gemma4_v3.json \
  --rich-supervision-mode repeat-class --rich-consistency-weight 0 \
  --seed 3407 --train-samples 1000 --train-subset-seed 3407 \
  --batch-size 1 --epochs 10 --start-epoch 0 --start-val 7 --val-epoch 1 \
  --num-workers 0 --crop-size 384 --concat-size 224 --downsample-ratio 8 \
  --stride 384 --lr 5e-5 --weight-decay 5e-4 --checkpoint-interval 5
```

If subsequently reviewed and run, the normal run directory contains `train.log`,
full `.tar` checkpoints on the original epoch cadence, and best-model `.pth`
weights selected by the unchanged **validation MAE + RMSE** rule. Full resumable
checkpoints include `rich_prompt_config.rich_supervision_mode`; missing mode in
historical rich checkpoints means `rich`. Cross-mode resume is rejected before
model/optimizer loading. Best `.pth` files remain weights-only evaluation
artifacts and cannot resume training. No evaluator or checkpoint-selection code
was changed.

## Local implementation verification

The following PowerShell command passed **137 tests** with exit code 0 under
`t2itest` (PyTorch 1.11.0, Torchvision 0.12.0). This is local CPU fixture/static
evidence, not Colab or real-model gradient evidence. Torchvision emitted an
existing optional image-extension warning. An initial checkpoint round-trip
test hit Windows access denial on an atomic rename in the sandbox temp folder;
it passed using the workspace-local temp directory below.

```powershell
$env:SSLKEYLOGFILE = ''
$taskTemp = Join-Path (Get-Location) '.pytest_cache\rp008_tmp'
[System.IO.Directory]::CreateDirectory($taskTemp) | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
& C:\Users\MSl\miniconda3\envs\t2itest\python.exe -m pytest tests/test_rich_prompt_phase2b.py tests/test_checkpoint_resume.py tests/test_train_samples.py tests/test_smoke_train_samples.py tests/test_progress_reporting.py tests/test_rich_prompt_evaluator.py tests/test_rich_prompt_generator.py tests/test_rich_prompt_phase2a.py tests/test_infrastructure.py -q
```

The full repository suite, alignment experiments, and notebooks were not run.
No training/inference architecture, dataset, bank, augmentation, evaluator, or
historical experiment record was edited. The current preparation changes only
the opt-in routing, resumable-mode metadata, diagnostic, tests, and this guide.
