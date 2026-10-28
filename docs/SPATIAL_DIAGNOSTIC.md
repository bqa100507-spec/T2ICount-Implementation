# FSC147-S spatial diagnostic v2 - all 230 images

Use `notebooks/rich_semantic_spatial_diagnostic_230.ipynb` in VS Code with a
Colab kernel, or directly in Colab. The source is cloned from the GitHub branch
`rich-prompt-alignment` after the researcher commits/pushes the reviewed files.
No manually uploaded source ZIP is needed. Nothing is committed or pushed by
this implementation. Existing preliminary/20-image artifacts are preserved.

## Research definition and unchanged controls

- **Question:** Does RP-003 differ spatially from BASE-002 under class-name
  inference, and does RP-009 share those differences?
- **Hypothesis:** RP-003 may show different attention/HSCM/density responses;
  some may also occur in RP-009. This remains a hypothesis.
- **Changed scope:** The researcher approved expanding metric coverage from
  20 to all 230 FSC147-S images. The original 20 remain the visualization set.
- **Controls:** Original checkpoint files, saved class/detailed prompts, seed
  3407, model, float32, RGB [-1,1], 384 patch/stride, tokenizer/masks, evaluator,
  density reassembly/scaling, and comparison formulas are unchanged.
- **Metrics:** Counting error/MAE/RMSE and descriptive normalized map L1/Pearson.
  Dot localization metrics remain unavailable: original FSC147 dots may refer
  to another target. No pseudo annotations or additional dataset protocol.
- **Decision rule:** Report paired observations for researcher review; no causal,
  localization-accuracy, transfer-mechanism, or scientific conclusion is automatic.

No training, architecture/loss change, ALIGN-004 FFN, prompt generation, weight
update, or edit to PROJECT_STATE.md / EXPERIMENT_LEDGER.md is performed.

## Coverage and storage

| Artifact | Coverage |
| --- | --- |
| `results.csv` | 230 x 3 checkpoints x 2 prompts = **1,380 rows** |
| `map_comparisons.csv` | 230 x 9 condition pairs x 7 map types = **14,490 rows** |
| `summary.json` | Six counting aggregates (n=230), 63 paired-map aggregates |
| Detailed PNGs and `maps/*.npz` | Original fixed 20 images only; **120 NPZs** |
| Capture/no-capture equality | `3319.jpg` only, every model/prompt: **six checks** |

The seven map types are fused U-Net attention, HSCM sim_x2/sim_x1, native density,
and separate attention means at 12/24/48. Native token channels, similarities,
patch xywh coordinates, and evaluator-reassembled density are retained for the
original 20. Other images cache only the seven comparison arrays on local disk.
Models load sequentially through `build_t2icount` and the trusted strict loader;
no .tar fallback or simultaneous GPU models.

The deterministic order is the existing seed-3407/count-bin/class-diversity rule.
All 230 appear once; its first 20 remain exactly those frozen in
`docs/SPATIAL_DIAGNOSTIC_SELECTION_V1.json` (unchanged). Preflight checks bank and
metadata SHA256 plus that selection. The local dataset has **443 patches** across
230 images; the original 20 still have 40. Actual geometry is recorded at runtime.

Only density uses the existing reassembly: x8 bilinear /64, overlap mean, /60;
count sums that full map. Intermediate maps remain patch-local, including
shifted overlapping edge patches. Temporary hooks read existing outputs; no
alternative attention or HSCM function is introduced. Content masks use the
validated CLIP helper, exclude BOS/EOS/padding, and retain original BPE positions.

Comparisons keep the original formulas: per-patch minmax with eps=1e-7,
concatenate patch cells, then L1/Pearson within the **same map type**. Constant-map
correlation is null. Overlap cells count as separate patch observations; this is
not image-area-weighted localization accuracy. Different prompts have different
semantic content; their token means are descriptive perturbations.

A private `TemporaryDirectory` uses `/content` on Colab (OS temp locally), outside
Drive and the output/checkpoint root. Nonretained files are deleted as each image
is compared; the directory is cleaned on success and Python exceptions. Completion
requires it removed. A hard runtime/process termination cannot execute cleanup;
local scratch disappears on runtime reset. Partial Drive outputs remain for
inspection, never resume or overwrite. Use a new output directory for another run.

## Notebook workflow

1. Configure ASSET_ROOT, OUTPUT_DIR, NUM_IMAGES=230, MODEL_SELECTION, PROMPT_SELECTION,
   and the two inference flags. Both flags default to **False**. Default full output
   is `outputs/spatial_diagnostic_v2_all230` under Drive assets; preliminary remains
   `outputs/spatial_diagnostic_v1_preliminary`.
2. Mount Drive; inspect existing reports. Completed saved results can be viewed
   without source setup, dependency installs, model assets, or GPU inference.
3. For a new run, clone the GitHub branch into a fresh local REPO. A clean existing
   matching checkout may fast-forward; dirty/unrelated checkouts are refused and
   preserved. Empty EXPECTED_COMMIT resolves/pins the remote tip for this session;
   optionally supply a reviewed full SHA. The manifest records the exact commit.
4. Reuse `requirements-colab.txt`, pinned taming-transformers/import workaround,
   and existing local asset extraction. Source ZIP upload is removed; the large
   **asset** ZIP remains an optional external dataset/pretrained-model source.
5. Validate checkpoints, the existing saved bank, metadata, all images, tokens,
   geometry, and fixed visualization selection. Nothing loads a counting model.
6. Reuse a valid preliminary. If absent, enable RUN_PRELIMINARY, complete the
   designated one-image run, and inspect it. Partial directories are refused.
7. Enable RUN_DIAGNOSTIC. Full coverage uses 1,380 conditions but still only six
   designated-image equivalence checks. Progress streams through Python, no tee.
8. Inspect all-230 counting/map summaries and the original-20 detailed figures.

`tools/spatial_diagnostic_checks.py` is standard-library-only and shared verbatim
with the notebook, which can validate old preliminary reports before cloning.
It checks exact row/condition identities, aggregate coverage, retained NPZ/figure
identities, six equality checks, and cache cleanup. v1 preliminary reports remain
compatible. Full completion never requires 1,380 equivalence records.

## External prerequisites (keep out of Git)

ASSET_ROOT defaults to `/content/drive/MyDrive/T2ICount-assets`. The saved bank
must exist at `prompts/fsc147s_gemma4_26b_v3.json` there (or configure PROMPT_BANK).
It is the existing 230-entry Gemma v3 bank, not a regenerated copy; the selection
JSON records its exact SHA256. Copy the existing asset to Drive if missing.

The requested weights-only files under ASSET_ROOT are unchanged:

- BASE-002: `checkpoints/baseline_retrain/pilot_1000x10/baseline_1000x10/best_model_7.pth`
- RP-003: `checkpoints/rich_prompt/pilot_1000x10_lambda0/rich_prompt_1000x10_lambda0/best_model_7.pth`
- RP-009: `checkpoints/rich_prompt/rp009/template_diversity_1000x10_v1/best_model_7.pth`

Runtime assets use existing `pretrained/` and `datasets/FSC147/images_384_VarV2`.
Weights, datasets, saved/generated bank JSONs, NPZ/NPY/TAR/ZIP, output directories,
and private scratch are ignored by Git. The `datasets/` Python code stays tracked.

## Exactly what to commit before GitHub setup can run

Commit these eight files together, then push **rich-prompt-alignment** yourself:

```text
.gitignore
tools/spatial_diagnostic.py
tools/spatial_diagnostic_checks.py
tests/test_spatial_diagnostic.py
tests/test_spatial_diagnostic_notebook.py
notebooks/rich_semantic_spatial_diagnostic_230.ipynb
docs/SPATIAL_DIAGNOSTIC.md
docs/SPATIAL_DIAGNOSTIC_SELECTION_V1.json
```

The selection JSON is unchanged but still untracked in this checkout, so it must
be included. Required inference utilities, configs, requirements-colab.txt and
FSC-147-S.json are already tracked; no edits to them are needed. Do not include
unrelated notebooks, Drive assets, prompt banks, generated results, source bundles,
or temporary build scripts in the commit. Until this source is pushed, the old
remote branch lacks the new files and the notebook intentionally refuses it.

## Local verification and limitations

CPU tests cover native capture/no-capture output equality, density scaling,
patch mapping, full-230 selection, selective retention, metric parity with the
old permanent-map path, sequential model release, success/failure cleanup,
1,380/14,490-row completion coverage, legacy preliminary reuse, Git ignore rules,
and notebook execution order/default and saved-result paths with mocked GitHub.

```powershell
$env:SSLKEYLOGFILE = ''
& C:\Users\MSl\miniconda3\envs\t2itest\python.exe -m pytest tests/test_spatial_diagnostic.py tests/test_spatial_diagnostic_notebook.py tests/test_rich_prompt_evaluator.py tests/test_t2icount_checkpoint_compatibility.py -q
```

Real local metadata/bank/tokenizer/image preflight passed for all 230 images,
with unavailable checkpoint path checks explicitly excluded. This is not a
checkpoint-load or GPU inference test. No real-model GPU diagnostic, full Colab
setup, live GitHub clone of the unpushed changes, or scientific interpretation
was performed during this implementation.
