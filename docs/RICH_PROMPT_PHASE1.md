# Rich-prompt research: Phase 1A

## Scope and goal

Phase 1A produces image-aware rich descriptions offline for the FSC-147-S
samples. It does not run T2ICount inference, train a model, change a checkpoint,
or implement the RichCount architecture.

For each sample, the generator sends its FSC147 image and authoritative class
name to `gemini-2.5-flash`. The stored output contains:

- `detailed`: Gemini's one-sentence response after whitespace cleanup only.
- `generalized`: the detailed response with every case-insensitive exact class
  occurrence replaced deterministically by `object`.

The protocol is inspired by RichCount, but it is not an exact RichCount
reproduction. Phase 1A uses Gemini 2.5 Flash and this repository's fixed
description protocol rather than the paper's GPT-4 setup.

## Why prompts are generated once

Generation is an offline preprocessing step because API output, availability,
latency, and cost should not become part of the later counting evaluation. The
JSON prompt bank records the model name, full prompt template, protocol version,
and deterministic generalization rule. Reusing that stored bank keeps later
comparisons on the same prompt inputs.

The bank intentionally stores no FSC-147-S ground-truth counts, API key, raw API
response objects, chain-of-thought, or image bytes.

## Count-leakage controls

The generator rejects empty or multiline responses, responses that are clearly
more than one sentence, missing exact class names, digits, English number words
from zero through twenty, hundred/thousand, and the documented quantity terms in
`COUNT_LEAKAGE_TERMS`. Validation failures are regenerated within the configured
retry limit. A sample that still fails is written only to the bank's `failures`
object; it never receives a fabricated fallback prompt.

Generalization makes no second API request and performs no paraphrasing. If the
exact class cannot be found case-insensitively, validation fails instead of
inventing generalized text.

## Environment and API key

Install the optional preprocessing dependency in a separate Python 3.10+
environment. This keeps the modern Google SDK out of the legacy T2ICount
training runtime:

```powershell
python -m pip install -r requirements-rich-prompt.txt
```

Only the `GEMINI_API_KEY` environment variable is read. Do not put the key in a
tracked `.env` file, command-line argument, output JSON, or log:

```powershell
$env:GEMINI_API_KEY = "<your-key>"
```

The implementation uses the current official `google-genai` SDK, not the legacy
`google-generativeai` package. Normal generation fails before client creation
when the key is absent. `--dry-run` deliberately needs no key and imports no
Gemini SDK code.

## Dry-run and generation

The repository's external asset convention is:

```text
<asset-root>/datasets/FSC147/images_384_VarV2/<image filename>
```

Validate the first three metadata entries and their real image paths without an
API call or output write:

```powershell
python tools/generate_rich_prompt_bank.py `
  --asset-root "D:\T2ICount-assets" `
  --metadata FSC-147-S.json `
  --output prompts/fsc147s_gemini25flash.json `
  --max-samples 3 `
  --dry-run
```

Run a 30-sample pilot:

```powershell
python tools/generate_rich_prompt_bank.py `
  --asset-root "D:\T2ICount-assets" `
  --metadata FSC-147-S.json `
  --output prompts/fsc147s_gemini25flash.json `
  --max-samples 30 `
  --request-delay 2 `
  --max-retries 3
```

`--max-retries` is the number of retries after the initial request. The delay is
applied between requests, and retries use exponential backoff.

## Incremental persistence and resume

Every successful description is saved immediately with a temporary file plus
atomic replacement. Exhausted failures are also persisted so they remain
auditable. On rerun, `status=ok` entries are skipped by default, while failed or
missing entries are attempted again. `--overwrite` explicitly regenerates
successful entries. An interrupted run therefore does not require regenerating
the whole benchmark.

The tool refuses to append to a bank whose generator, protocol, template, or
generalization rule differs from the current constants. This prevents silently
mixing incompatible prompt protocols.

## Intentionally deferred

Loading this bank into T2ICount and evaluating detailed versus generalized
prompts are separate later phases. Phase 1A has zero effect on baseline model,
dataset, training, inference, optimizer, or checkpoint behavior.
