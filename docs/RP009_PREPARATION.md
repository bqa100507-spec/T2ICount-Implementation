# RP-009 — Template Diversity Control (preparation only)

Prepared 2026-10-27 on `rich-prompt-alignment`, starting from commit
`b8aa367ba36f24dc33a7fc22079c045bf596214a`. No training, Gemma/API prompt generation,
or counting evaluation was launched. The researcher approved the core
implementation; use the exact commit reported after the preparation push, not
the starting commit above. Full training remains a separate research decision.
Historical research records are unchanged.

## Researcher-reviewed context

The researcher reports that the completed RP-008 preflight's official
recommendation was **REVIEW**, because of small gradient differences. The
researcher decided to forgo the full 10-epoch RP-008 run and prioritize RP-009.
This is a researcher's prioritization decision, not a finding of numerical
equivalence. The raw official diagnostic JSON was not independently inspected
for this preparation; no gradient values or artifact path are invented here.
The original RP-008 preparation and diagnostic remain unchanged.

RP-003 (`lambda_cons=0`) recorded standard class-name test MAE/RMSE
19.98 / 107.13 versus BASE-002's 22.82 / 120.25. This does not establish that rich
semantics caused the gain. RP-009 tests generic prompt diversity while retaining
the three-forward procedure, compatible-positive routing, and loss weighting.

## Experiment definition

- **Question:** Can RP-003's improvement in class-name counting be explained by
  generic prompt diversity rather than image-specific rich semantic information?
- **Hypothesis:** If generic text diversity is sufficient to account for the
  gain, deterministic class-based templates may improve counting over BASE-002
  and approach RP-003 without image-specific attributes.
- **Changed variable:** Relative to RP-003, replace only the detailed/generalized
  auxiliary texts and their masks with these exact two templates, generated
  from the original class caption: `a photo of {class_name}` and
  `the {class_name} in the image`. The first class caption remains unchanged.
- **Controls:** FSC147 1000-sample train subset; subset/training seeds 3407;
  10 epochs; batch size 1; learning rate `5e-5`, weight decay `5e-4`; original
  AdamW groups (U-Net LR/weight decay each scaled by 0.1); original augmentation,
  image preprocessing, density/RRC losses, U-Net, decoder, and evaluation.
  Validation starts at zero-based epoch 7, every epoch. Best checkpoint is
  still selected by validation **MAE + RMSE**. Use the existing validated bank,
  same compatible-positive routing, and `lambda_cons=0`.
- **Metrics:** Primary: class-name counting MAE/RMSE on the appropriate fixed
  standard FSC147 validation/test protocol. Keep any FSC147-S comparison
  separate and use its existing fixed protocol. Template/rich inference,
  alignment, and feature-map behavior are secondary diagnostics.
- **Decision rule for review:** After a separately approved run, compare RP-009
  with BASE-002 and RP-003 using their recorded MAE/RMSE pairs. Improving both
  BASE-002 test metrics would support an effect of this generic-template
  procedure in that run. Matching or improving both RP-003 test metrics would
  support generic diversity as a candidate explanation of its recorded gain.
  Mixed metrics or a shortfall remain inconclusive about the value of rich
  semantics. Do not declare statistical equivalence, causal attribution, or a
  final scientific conclusion from a single run with unverified historical
  provenance. The researcher must review all interpretation.

## Implementation

`--rich-supervision-mode template-diversity` is opt-in; `rich` remains the
default and `repeat-class` remains available.

For each compatible positive, the dataset constructs the templates from the
exact current/original class caption (no stripping, case changes, singularizing,
LLM calls, counts, attributes, or image-dependent choices). Each auxiliary mask
uses `build_rich_prompt_attention_mask` and the same dataset CLIP tokenizer:
77 positions, BOS/EOS excluded, at most 75 raw prompt tokens. Longer templates
receive their own masks; the class mask is unchanged.

The existing metadata keys `detailed_prompt` and `generalized_prompt` carry
template 1 and template 2 in this mode. These legacy slot names also occur in
loss logs; they do not mean bank descriptions were used. The validated bank
is still loaded and checked in full for metadata, subset, classes, status, and
text validity. Its descriptions are not read by the template dataset path,
tokenized into auxiliary masks, or passed to the counting model. The bank is
never modified or regenerated. Incompatible samples carry empty auxiliary text
and zero auxiliary masks, and receive only the original class/current-prompt
forward. Same-class unchanged positives retain existing compatibility.

The shared trainer still performs three independent forwards for compatible
samples, using the same transformed images, GT density, and image attention
masks. Each supervised loss is unchanged:

```text
L_branch = L_reg + 0.01 * L_RRC_stage1 + 0.01 * L_RRC_stage2
L_sup = (L_class + L_template1 + L_template2) / 3
L_batch = L_class_all + (n_compatible / N) * (L_sup - L_class_compatible)
```

Nonzero consistency weight is rejected; template mode does not compute the
consistency loss. Optimizer updates, inference, and checkpoint selection are
unchanged. Resumable `.tar` metadata records:

```text
rich_supervision_mode = template-diversity
template_scheme_version = class-name-templates-v1
```

The scheme identifier fixes the exact wrappers, order, and use of the original
caption. Any future change requires a new scheme version and review. Cross-mode
resumes and missing/changed template schemes are rejected before model or
optimizer loading. Historical checkpoints without a mode remain `rich`;
historical rich/repeat-class checkpoints do not need a template scheme.
Best-model `.pth` files remain weights-only evaluation artifacts.

## Provenance and limitations

The existing bank was revalidated locally without a GPU model:

- bank: `fsc147_train1000_seed3407_gemma4_v3.json`;
- selected count: 1000; protocol: `rich-prompt-phase1-v3`;
- ordered subset fingerprint:
  `sha256:dd96b36bf15013e194b1a8ece06452a19822aae028fc00c0de019cbb7a311f24`;
- bank file fingerprint (unchanged):
  `sha256:c3ec587b37fd9f351cdedbce888b26e0ba7a020de956c4ef18202a1268e7e32f`.

RP-003's ledger does not preserve its raw training command, initial-weight
fingerprint, full runtime/package versions, or raw artifact location. The
requested validation schedule agrees with BASE-002's historical manifest and
the RP-008 preparation, but exact historical RP-003 runtime/initialization
matching cannot be certified. Current training freshly initializes from the
existing SD/CLIP assets and a seeded counting decoder; it does not load a
trained BASE-002, RP-003, RP-008, or official counting checkpoint.

`initial_model_weight_fingerprint: null`: no real model was initialized in this
preparation. Do not substitute the Git SHA or an asset filename for this value.
For a future approved run, retain the exact source revision/patch, runtime
versions, args, asset/bank fingerprints, logs, and initial weight hash if
collected before the first optimizer step. Colab runtime behavior is untested
by this local CPU verification.

## Colab preparation: pinned imports and exact pushed revision

After mounting Drive, extracting the existing assets, and installing the normal
`requirements-colab.txt`, run this **Python notebook cell**. Replace the one
placeholder with the full 40-character SHA from the push report. Do not replace
it with a floating branch name or the old RP-008 SHA. The cell refuses tracked
local changes rather than discarding them, fast-forwards only the research
branch, and verifies both local HEAD and the remote branch against that SHA.

The RP-008 notebook required an import workaround even after the editable
`taming-transformers` installation. Preserve the pinned upstream commit
`3ba01b241669f5ade541ce990f7650a3b8f65318`, already specified in
`requirements-colab.txt`; do not install an unpinned replacement. `sys.path`
alone affects only the notebook process. Setting `os.environ['PYTHONPATH']`
also supplies the path to later Bash/Python subprocesses. The smoke and full
training Bash cells additionally export it explicitly.

```python
import os
import re
import subprocess
import sys
from pathlib import Path

EXPECTED_COMMIT = "REPLACE_WITH_FULL_RP009_SHA_FROM_PUSH_REPORT"
assert re.fullmatch(r"[0-9a-f]{40}", EXPECTED_COMMIT), "Use the reported full SHA"
REPO = Path("/content/T2ICount-Implementation")
def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()

assert not git("status", "--porcelain", "--untracked-files=no"), "Inspect/preserve local edits first"
subprocess.run(["git", "-C", str(REPO), "fetch", "origin", "rich-prompt-alignment"], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "rich-prompt-alignment"], check=True)
subprocess.run(["git", "-C", str(REPO), "merge", "--ff-only", "origin/rich-prompt-alignment"], check=True)
assert git("branch", "--show-current") == "rich-prompt-alignment"
assert git("rev-parse", "HEAD") == EXPECTED_COMMIT, "Local commit mismatch"
assert git("ls-remote", "origin", "refs/heads/rich-prompt-alignment").split()[0] == EXPECTED_COMMIT, "Remote commit mismatch"
subprocess.run(["git", "-C", str(REPO), "diff", "--exit-code", "HEAD", "--", "."], check=True)
os.chdir(REPO)
os.environ["RP009_EXPECTED_COMMIT"] = EXPECTED_COMMIT

TAMING_DIR = Path("/content/taming-transformers")
TAMING_COMMIT = "3ba01b241669f5ade541ce990f7650a3b8f65318"
if not TAMING_DIR.exists():
    subprocess.run(["git", "clone", "https://github.com/CompVis/taming-transformers.git", str(TAMING_DIR)], check=True)
subprocess.run(["git", "-C", str(TAMING_DIR), "diff", "--exit-code", "HEAD", "--", "."], check=True)
subprocess.run(["git", "-C", str(TAMING_DIR), "checkout", "--detach", TAMING_COMMIT], check=True)
assert subprocess.check_output(["git", "-C", str(TAMING_DIR), "rev-parse", "HEAD"], text=True).strip() == TAMING_COMMIT
sys.path.insert(0, str(TAMING_DIR))
os.environ["PYTHONPATH"] = str(TAMING_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")
from taming.modules.vqvae.quantize import VectorQuantizer2
subprocess.run(["bash", "-c", "python -c \"from taming.modules.vqvae.quantize import VectorQuantizer2; print('Bash subprocess import: OK')\""], check=True)
print("Verified RP-009 commit:", EXPECTED_COMMIT)
print("Verified taming commit:", TAMING_COMMIT)
```

If the remote has since advanced or the checkout has edits, stop and inspect;
do not silently run another revision or reset/delete work. This cell does not
modify `main`, checkpoints, datasets, banks, or historical artifacts.

## One-compatible-batch real-model GPU smoke procedure

**Question:** Does this exact checkout execute the intended RP-009 batch path?
**Hypothesis:** The original class and two templates have the expected masks,
three forwards, and finite supervised loss/backward gradients.
**Changed variable:** None in the scientific path; a diagnostic-only trainer
subclass omits optimizer updates. **Controls:** The documented bank, subset,
seeds, model/assets, augmentation, full-precision loss, and three-forward path.
**Metrics/decision:** PASS requires exact captions/masks/images/GT, three
forwards, four real criterion calls (class all/subset + auxiliaries), finite
loss and nonzero finite gradients, zero optimizer calls, and an identical
before/after parameter fingerprint. Any failure blocks running the experiment
pending inspection; PASS does not approve training or establish counting gains.

`tools/smoke_template_diversity.py` uses the production `train_epoch()` and
criterion. Observers only validate inputs and record real loss components; they
do not replace loss math. It scans at most 32 normally augmented batches until
one is compatible, without forcing augmentation branches, then runs just that
one batch. It never calls `trainer.train()`, validation, test, checkpoint save,
or `optimizer.step()`; a guard raises if an optimizer step is attempted.
Training-mode buffers/attention caches can change normally; the fingerprint
covers parameters (including frozen ones), not buffers. Hashing streams one
parameter at a time through CPU and retains no full model copy. GPU memory for
the normal three-forward graph is still required. No precision, dropout,
architecture, or activation-checkpointing workaround is introduced.

After the preparation cell, run this **Bash notebook cell**. It rechecks the
commit and pinned dependency and uses new smoke-only report/log paths. Existing
report or run-directory names are rejected; rename only the new smoke run for
a rerun. The `--epochs 10` argument records the unchanged protocol; it does not
start a ten-epoch loop in this diagnostic.

```bash
%%bash
set -euo pipefail
cd /content/T2ICount-Implementation
: "${RP009_EXPECTED_COMMIT:?Run the preparation cell with the reported SHA first}"
test "$(git branch --show-current)" = rich-prompt-alignment
test "$(git rev-parse HEAD)" = "$RP009_EXPECTED_COMMIT"
test "$(git -C /content/taming-transformers rev-parse HEAD)" = 3ba01b241669f5ade541ce990f7650a3b8f65318
export PYTHONPATH="/content/taming-transformers:${PYTHONPATH:-}"
python -c "from taming.modules.vqvae.quantize import VectorQuantizer2; print('taming import: OK')"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset SSLKEYLOGFILE
python -u tools/smoke_template_diversity.py \
  --expected-commit "$RP009_EXPECTED_COMMIT" \
  --output /content/drive/MyDrive/T2ICount-assets/outputs/rp009/smoke_template_diversity_v1.json \
  --max-scan-batches 32 \
  --asset-root /content/T2ICount-assets --config configs/v1-inference.yaml \
  --save-dir /content/drive/MyDrive/T2ICount-assets/checkpoints/rich_prompt/rp009/smoke \
  --content template_diversity_one_batch_v1 \
  --rich-prompt-bank /content/drive/MyDrive/T2ICount-assets/prompts/fsc147_train1000_seed3407_gemma4_v3.json \
  --rich-supervision-mode template-diversity --rich-consistency-weight 0 \
  --seed 3407 --train-samples 1000 --train-subset-seed 3407 \
  --batch-size 1 --epochs 10 --start-epoch 0 --start-val 7 --val-epoch 1 \
  --num-workers 0 --crop-size 384 --concat-size 224 --downsample-ratio 8 \
  --stride 384 --lr 5e-5 --weight-decay 5e-4 --checkpoint-interval 5
```

Expected: `Smoke PASS; zero optimizer updates`, a new JSON with the three
captions/full masks, density/RRC components, total loss, gradient norm, Git SHA,
bank/subset/scheme metadata, runtime, and identical initial/final parameter
hashes (`named-parameters-v1`). The new smoke run directory contains setup/
one-batch logs only, no `.pth`/`.tar` checkpoints. The initial parameter hash is
available for that smoke initialization only; a future training run must record
its own initialization if a corresponding fingerprint is required. This real
GPU procedure is prepared for Colab and has not been executed locally.

## Validation results

- **152 tests passed**, exit code 0: all existing/new Phase 2B, RP-008 preflight,
  checkpoint/resume, subset, progress, rich evaluator/generator, Phase 2A, and
  infrastructure tests in the command below. This includes 10 new RP-009 tests
  covering templates/masks, independence from stored descriptions, routing,
  three forwards, GT and gradient weighting, zero consistency, CLI checks, and
  mode/scheme compatibility including a full `.tar` round trip.
  Five additional smoke tests cover captions/masks and backward, zero updates,
  optimizer-step blocking, nonfinite-loss rejection, and parameter hashing.
- Real local CLIP tokenizer, offline, no model initialization: **2000/2000**
  auxiliary masks for the validated 1000-sample subset matched masks built
  independently from CLIP sequences including BOS/EOS. Prompt-token lengths
  ranged from **4 to 7**. Bank/subset fingerprints match those above.
- Python compilation passed for the six modified/new Python files. Scoped
  `git diff --check` passed. The full repository suite and notebooks were not
  executed; no real-model training or counting evaluation was run.
- Local environment: `C:/Users/MSl/miniconda3/envs/t2itest/python.exe`, PyTorch
  1.11.0, Torchvision 0.12.0. The existing optional Torchvision image-extension
  warning was emitted. Workspace-local temp avoids the previously observed
  Windows sandbox rename restriction. These results do not validate Colab's
  different runtime or establish any counting gain.

Exact test command, from the repository root in **PowerShell**:

```powershell
$env:SSLKEYLOGFILE = ''
$taskTemp = Join-Path (Get-Location) '.pytest_cache\rp009_tmp'
[System.IO.Directory]::CreateDirectory($taskTemp) | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
& C:\Users\MSl\miniconda3\envs\t2itest\python.exe -m pytest tests/test_rich_prompt_phase2b.py tests/test_checkpoint_resume.py tests/test_train_samples.py tests/test_smoke_train_samples.py tests/test_progress_reporting.py tests/test_rich_prompt_evaluator.py tests/test_rich_prompt_generator.py tests/test_rich_prompt_phase2a.py tests/test_infrastructure.py -q
& C:\Users\MSl\miniconda3\envs\t2itest\python.exe -m py_compile train.py datasets/dataset.py utils/regression_trainer.py utils/rich_prompt_training.py tools/smoke_template_diversity.py tests/test_rich_prompt_phase2b.py
```

## Proposed Colab command — DO NOT RUN before researcher review

Use the exact pushed preparation commit and pinned import workaround verified
by the preparation cell above, with the unchanged assets/environment. Do not
run the old DUMLO training/checkout cells from `train_colab.ipynb`.

This command is proposed only. Both checkpoint and log/output destinations are
new RP-009 paths. It rejects existing run directories; choose a new run name
for a reviewed rerun instead of overwriting anything.

```bash
%%bash
set -euo pipefail
cd /content/T2ICount-Implementation
test "$(git branch --show-current)" = rich-prompt-alignment
: "${RP009_EXPECTED_COMMIT:?Run the preparation cell with the reported SHA first}"
test "$(git rev-parse HEAD)" = "$RP009_EXPECTED_COMMIT"
git diff --exit-code HEAD -- .
test "$(git -C /content/taming-transformers rev-parse HEAD)" = 3ba01b241669f5ade541ce990f7650a3b8f65318
export PYTHONPATH="/content/taming-transformers:${PYTHONPATH:-}"
python -c "from taming.modules.vqvae.quantize import VectorQuantizer2; print('taming import: OK')"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset SSLKEYLOGFILE
RUN_NAME=template_diversity_1000x10_v1
SAVE_ROOT=/content/drive/MyDrive/T2ICount-assets/checkpoints/rich_prompt/rp009
OUTPUT_DIR=/content/drive/MyDrive/T2ICount-assets/outputs/rp009/$RUN_NAME
test ! -e "$SAVE_ROOT/$RUN_NAME"
test ! -e "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
python -u train.py \
  --asset-root /content/T2ICount-assets \
  --config configs/v1-inference.yaml \
  --save-dir "$SAVE_ROOT" --content "$RUN_NAME" \
  --rich-prompt-bank /content/drive/MyDrive/T2ICount-assets/prompts/fsc147_train1000_seed3407_gemma4_v3.json \
  --rich-supervision-mode template-diversity --rich-consistency-weight 0 \
  --seed 3407 --train-samples 1000 --train-subset-seed 3407 \
  --batch-size 1 --epochs 10 --start-epoch 0 --start-val 7 --val-epoch 1 \
  --num-workers 0 --crop-size 384 --concat-size 224 --downsample-ratio 8 \
  --stride 384 --lr 5e-5 --weight-decay 5e-4 --checkpoint-interval 5 \
  2>&1 | tee "$OUTPUT_DIR/console.log"
```

If later approved and executed, checkpoints and `train.log` will appear under
`checkpoints/rich_prompt/rp009/template_diversity_1000x10_v1/`, with the console
log in `outputs/rp009/template_diversity_1000x10_v1/`. Full checkpoints follow
the existing epoch cadence (0, 5); best weights can be selected during the
unchanged validation schedule at epochs 7–9. No artifact in BASE-002, RP-003,
RP-008, or other historical directories is used as an output target.
