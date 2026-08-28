# Rich-prompt research: Phase 1A

## Scope and goal

Phase 1A produces image-aware rich descriptions offline for the FSC-147-S
samples. It does not run T2ICount inference, train a model, change a checkpoint,
or implement the RichCount architecture.

For each sample, the generator sends its FSC147 image and authoritative class
name to the model selected by `--model` through the Google Gen AI Interactions
API. Phase 1A v2 accepts an explicit model argument; the current experimental
default is `gemma-4-26b-a4b-it`. The stored output contains:

- `detailed`: the selected generator's one-sentence response after whitespace
  cleanup only.
- `generalized`: the detailed response with every case-insensitive exact class
  occurrence replaced deterministically by `target object` or `target objects`.

The RichCount paper used GPT-4. This protocol is inspired by RichCount, but it
is not an exact RichCount reproduction: Phase 1A uses a separately selected
Google Gen AI generator and this repository's fixed description protocol.

Gemini 3.6 Flash was used during an earlier pilot but encountered a very low
free-tier daily request quota. Gemma 4 26B A4B IT is now being evaluated as the
practical free generator. The successful three-image Gemma smoke test confirms
the request path, not prompt quality or superiority over Gemini. Outputs from
different generators are not assumed to be directly equivalent, so exact model
provenance is retained and their prompt banks must remain separate.

## Why protocol v2 exists

The initial three-image Gemini 3.6 pilot exposed an implicit count cue that the
v1 validator did not catch. For the `people` class, the generated wording
included `a ... individual`, which implies one visible target instance without
using a digit or number word. The same pilot showed that replacing a plural
class with plain `object` could produce awkward wording such as `The object ...
consist`.

Protocol `rich-prompt-phase1-v2` is this experiment's refinement, not a
requirement taken from the RichCount paper. It strengthens the fixed generation
prompt against implicit instance cues, adds a conservative implicit-leakage
validator, and makes deterministic generalization plurality-aware. Selecting a
different generator does not alter the v2 prompt, validation, or deterministic
generalization protocol. The Interactions API transport remains unchanged.

## Why prompts are generated once

Generation is an offline preprocessing step because API output, availability,
latency, and cost should not become part of the later counting evaluation. The
JSON prompt bank records the model name, Interactions API transport, full prompt
template, protocol version, and deterministic generalization rule. Reusing that
stored bank keeps later comparisons on the same prompt inputs.

The bank intentionally stores no FSC-147-S ground-truth counts, API key, raw API
response objects, chain-of-thought, or image bytes.

## Count-leakage controls

The generator rejects empty or multiline responses, responses that are clearly
more than one sentence, missing exact class names, digits, English number words
from zero through twenty, hundred/thousand, and the documented quantity terms in
`COUNT_LEAKAGE_TERMS`. Protocol v2 also rejects the standalone tokens in
`IMPLICIT_COUNT_LEAKAGE_TERMS`, including `a`, `an`, `single`, `individual`,
`pair`, `couple`, `group`, `crowd`, and `cluster` plus documented plural forms.

The implicit filter is deliberately conservative for this pilot: even an
indefinite article used for scene context is rejected because it can imply an
instance count. Matching uses word boundaries, so substrings inside words such
as `metal`, `handle`, and `orange` are not rejected. Validation failures retain
the existing retry behavior. A sample that still fails is written only to the
bank's `failures` object; it never receives a fabricated fallback prompt.

Generalization makes no second API request and performs no paraphrasing. The v2
heuristic treats `people` and classes whose normalized names end in `s` as
plural-like, replacing them with `target objects`; all other classes become
`target object`. Replacement remains case-insensitive and exact. This small
heuristic intentionally avoids an NLP dependency, so singular words ending in
`s` can be classified as plural-like and irregular plurals other than `people`
can be classified as singular-like. If the exact class cannot be found,
validation fails instead of inventing generalized text.

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

The implementation uses the current official `google-genai` SDK and
`client.interactions.create()`, not the legacy `google-generativeai` package or
the Generate Content transport. Each request has one text protocol block and
one base64-encoded local image block with its MIME type. `store=False` avoids
retaining the interaction for later turns, and only `interaction.output_text`
is passed to validation; response steps, thoughts, and tool data are not stored.
Normal generation fails before client creation when the key is absent.
`--dry-run` deliberately needs no key and imports no Google Gen AI SDK code.

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
  --model gemma-4-26b-a4b-it `
  --max-samples 3 `
  --dry-run
```

When `--output` is omitted, the filename is derived from a filesystem-safe form
of the selected model. With the default model, the path is
`prompts/fsc147s_gemma_4_26b_a4b_it_v2.json`. An explicit `--output` always
takes precedence.

Run a 30-sample pilot:

```powershell
python tools/generate_rich_prompt_bank.py `
  --asset-root "D:\T2ICount-assets" `
  --metadata FSC-147-S.json `
  --output prompts/fsc147s_gemma4_26b_v2.json `
  --model gemma-4-26b-a4b-it `
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

The tool refuses to append to a bank whose recorded generator differs from the
selected `--model`, or whose API transport, protocol, template, or
generalization rule differs from the fixed v2 values. In particular, a Gemma 4
bank cannot be resumed with Gemini 3.6, a Gemini 3.6 bank cannot be resumed with
Gemma 4, and a v2 run cannot append to a v1 bank. This prevents model outputs or
generation/validation/generalization protocols from being silently mixed.

## Intentionally deferred

Loading this bank into T2ICount and evaluating detailed versus generalized
prompts are separate later phases. Phase 1A has zero effect on baseline model,
dataset, training, inference, optimizer, or checkpoint behavior.
