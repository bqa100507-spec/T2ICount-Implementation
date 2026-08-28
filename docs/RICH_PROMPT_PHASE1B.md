# Rich-prompt research: Phase 1B

## Purpose and interpretation

Phase 1B is an inference-only FSC-147-S diagnostic. It compares the unchanged
official T2ICount checkpoint under three text inputs while holding image and
inference processing fixed:

1. `class`: the authoritative class name from `FSC-147-S.json`.
2. `detailed`: the exact detailed description stored in the Phase 1A bank.
3. `generalized`: the exact generalized description stored in the Phase 1A
   bank.

The official checkpoint was not trained with these rich prompts. Consequently,
this experiment measures zero-shot prompt compatibility or sensitivity; it does
not establish the benefit of rich-prompt training. The checkpoint, architecture,
VAE, CLIP, U-Net, density scaling, and count calculation are unchanged. No
training or fine-tuning API is used.

The evaluator requires protocol `rich-prompt-phase1-v3`. It records the bank's
generator provenance without requiring one particular generator. Class names
and ground-truth counts always come from `FSC-147-S.json`; any class mismatch
between that metadata and the prompt bank stops evaluation. Prompt-bank counts,
if unexpectedly present, are never read.

## Controlled inference

For each image, `tools/evaluate_rich_prompts.py` applies the same RGB tensor
normalization as normal FSC147 evaluation, moves that tensor to the requested
device, and calls `prepare_image_patches` exactly once. The resulting patch
tuple and the same input tensor are reused for all three calls to
`predict_count`. Only the exact prompt string and its normal T2ICount prompt
attention mask change.

The model is built with `build_t2icount(..., checkpoint_path=<explicit path>,
mode="eval")`, followed by `model.eval()`. Predictions run under
`torch.no_grad()`. Patch size and stride remain 384, density is divided by 60 in
the existing inference helper, and predictions are not rounded before metrics.
There is no checkpoint fallback.

## Metrics and paired diagnostics

For predictions `p_i` and ground truth `y_i`, each prompt mode reports:

```text
MAE  = mean(abs(p_i - y_i))
RMSE = sqrt(mean((p_i - y_i)^2))
```

Mean signed count error is also reported separately. It is not labeled as MAE
or RMSE.

For detailed versus class and generalized versus class, the evaluator computes
per-image `rich_abs_error - class_abs_error`, its mean and median, and the number
of images improved, worsened, or tied. Negative delta means the rich prompt had
lower absolute error.

Primary `all_samples` metrics retain every FSC-147-S entry, including the known
label anomalies `3312.jpg` and `3313.jpg` whose authoritative class remains
`folks`. Supplementary `excluding_known_label_anomalies` metrics remove exactly
those two images and no others. For the complete 230-image bank this produces a
228-image sensitivity subset; it is not the primary benchmark result.

## Outputs and resume behavior

The output directory contains:

- `rich_prompt_eval_predictions.csv`: one row per completed image, including
  image, authoritative class and GT, all three exact prompt strings, all three
  unrounded predictions, absolute errors, paired deltas, and improvement flags.
- `rich_prompt_eval_summary.json`: provenance plus both analysis subsets, with
  class/detailed/generalized MAE, RMSE, mean signed error, and paired
  diagnostics.
- `rich_prompt_eval_manifest.json`: the compatibility contract used by resume.

The predictions CSV is atomically rewritten after each completed image.
`--resume` skips only rows containing all three internally consistent finite
predictions. Resume fails if the checkpoint, prompt bank, protocol, metadata,
sample list, model configuration, device, patch batch size, patch geometry, or
other recorded inference configuration differs. Existing results otherwise
require either `--resume` or the explicit destructive choice `--overwrite`.

Any missing input, prompt/class mismatch, incompatible protocol, prediction
exception, or non-finite prediction terminates the run with a non-zero exit
status. Completed earlier rows remain available for a compatible resume; no
sample is silently skipped after failure.

## Command

Use the official paper checkpoint explicitly:

```powershell
python tools/evaluate_rich_prompts.py `
  --asset-root "D:\T2ICount-assets" `
  --metadata FSC-147-S.json `
  --prompt-bank prompts/fsc147s_gemma4_26b_v3.json `
  --checkpoint "D:\T2ICount-assets\checkpoints\official\best_model_paper.pth" `
  --output-dir results/rich_prompt_phase1b_official `
  --batch-size 16 `
  --device cuda
```

`--batch-size` controls patch inference batching, not the number of images loaded
together. Local Stable Diffusion and CLIP paths default to the normal locations
under `--asset-root`; `--sd-path`, `--clip-path`, and `--config` can override
them explicitly without changing checkpoint semantics.

Before loading the model, validate all 230 metadata entries, prompt entries,
classes, image paths, and required runtime assets with:

```powershell
python tools/evaluate_rich_prompts.py `
  --asset-root "D:\T2ICount-assets" `
  --metadata FSC-147-S.json `
  --prompt-bank prompts/fsc147s_gemma4_26b_v3.json `
  --checkpoint "D:\T2ICount-assets\checkpoints\official\best_model_paper.pth" `
  --output-dir results/rich_prompt_phase1b_official `
  --device cuda `
  --validate-only
```

Validation-only mode performs no model inference and writes no result files.
The full 230-image evaluation should be launched deliberately and is not part of
the automated test suite.
