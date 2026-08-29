# Rich-prompt research: Phase 2A train-bank preparation

## Scope

Phase 2A generates an offline rich-prompt bank for the exact deterministic
FSC147 training subset used by the limited-compute baseline. It is data
preparation only. It does not load the bank into T2ICount, implement RichCount
Stage 2 training, or change the model, loss, trainer, optimizer, inference,
checkpoint, density-map, or notebook behavior.

The first rich-prompt pilot uses `--train-samples 1000` and
`--train-subset-seed 3407` because its source images must match the baseline
1000x10 pilot exactly. Otherwise prompt quality and source-image selection
would change together and the comparison would not isolate the prompt input.

## Exact FSC147 subset and metadata

The generator reads only the official files below for train-bank selection and
class resolution:

```text
<asset-root>/datasets/FSC147/FSC_147/Train_Test_Val_FSC_147.json
<asset-root>/datasets/FSC147/FSC_147/ImageClasses_FSC147.txt
<asset-root>/datasets/FSC147/images_384_VarV2/<image filename>
```

It does not use `FSC-147-S.json`, annotation files, density maps, or ground-truth
counts. No GT count is sent to the MLLM or written to the prompt bank.

The trainer and generator both call
`utils/train_subset.py::select_train_subset_indices`. This helper preserves the
existing private `torch.Generator`, `manual_seed`, and ordered `torch.randperm`
selection semantics without advancing the global Torch RNG. For train-bank
generation, run in an environment where Torch is importable; the default
FSC-147-S Phase 1A path keeps Torch lazy and does not require it.

The prompt-bank metadata records:

- benchmark and split;
- requested train sample count and subset seed;
- effective selected sample count;
- generator model, Interactions API, and v3 protocol;
- the unchanged generation template and generalization rule; and
- `sha256:<hex>` over the UTF-8 compact JSON encoding of the ordered selected
  image-name list.

The fingerprint is an explicit hand-off contract for later training. A trainer
integration must recompute it from its selected subset and reject a mismatch.

## Protocol and concurrency

Phase 2A reuses the Phase 1A v3 prompt construction, exact-class validation,
target-absence validation, count-leakage validation, and deterministic
`class name -> object` generalization functions. Concurrency changes only how
independent offline API requests overlap; it is infrastructure, not a research
variable. Prompt contents, validation, and one-image-per-request semantics are
unchanged.

`--concurrency` defaults to `1`, preserving the existing sequential Phase 1A
path. Values above one use a bounded thread pool around the official synchronous
Interactions API adapter. Only `N` futures are submitted at a time, so no
unbounded 1000-task queue is created. Completed results are validated
independently, then the main thread updates and atomically replaces the JSON
file. The final prompt/failure mappings follow the deterministic selected-image
order even when requests finish out of order.

`--request-delay` is a global minimum interval between concurrent request
starts. This prevents all workers from starting in one burst while still
allowing slow requests to overlap. Per-sample retries retain exponential
backoff, and a numeric SDK `Retry-After` value is respected when available.
No RPM or RPD quota is assumed or hardcoded.

## Resume compatibility

Each successful result is atomically persisted. On rerun, `status=ok` entries
are skipped, while failed or missing entries are attempted again.
`--overwrite` regenerates successful entries.

Resume rejects a bank if any research-defining identity differs, including:

- benchmark or split;
- requested train sample count or subset seed;
- selected sample count or ordered-list fingerprint;
- generator model or Interactions API;
- v3 protocol or generation template; or
- generalization rule.

Concurrency and request delay are deliberately not compatibility fields because
they do not change prompt semantics.

## Commands

Three-sample dry-run (no API key, no API request, no output write):

```powershell
python tools/generate_rich_prompt_bank.py `
  --asset-root "D:\T2ICount-assets" `
  --split train `
  --train-samples 3 `
  --train-subset-seed 3407 `
  --model gemma-4-26b-a4b-it `
  --output prompts/fsc147_train3_seed3407_gemma4_v3.json `
  --dry-run
```

Ten-sample real concurrency smoke test:

```powershell
$env:GEMINI_API_KEY = "<your-key>"
python tools/generate_rich_prompt_bank.py `
  --asset-root "D:\T2ICount-assets" `
  --split train `
  --train-samples 10 `
  --train-subset-seed 3407 `
  --model gemma-4-26b-a4b-it `
  --output prompts/fsc147_train10_seed3407_gemma4_v3.json `
  --concurrency 4 `
  --request-delay 2 `
  --max-retries 3
```

Full exact 1000-sample bank:

```powershell
$env:GEMINI_API_KEY = "<your-key>"
python tools/generate_rich_prompt_bank.py `
  --asset-root "D:\T2ICount-assets" `
  --split train `
  --train-samples 1000 `
  --train-subset-seed 3407 `
  --model gemma-4-26b-a4b-it `
  --output prompts/fsc147_train1000_seed3407_gemma4_v3.json `
  --concurrency 4 `
  --request-delay 2 `
  --max-retries 3
```

The dry-run resolves every selected image path and class name, prints the first
five image/class pairs, selected count, and ordered-list fingerprint. Repeat
dry-runs with the same official metadata, sample count, seed, and Torch subset
semantics must print the same ordered-list fingerprint.
