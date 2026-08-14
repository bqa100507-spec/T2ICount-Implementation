# Repository provenance audit

Audit date: 2026-08-14. This is a factual repository-hygiene record, not legal
advice or a legal conclusion about permission.

## Audit basis

The current `main` branch preserves the complete Git history of
[`cha15yq/T2ICount`](https://github.com/cha15yq/T2ICount). Upstream commit
`289d3fb95d435a3d19d2687c02211a40e4477e31` is both the local `upstream/main`
tip and the merge-base with this repository; implementation commit
`2addc6723c749022198a0a9912e92f46a614c187` follows it directly.

The audit used Git blob comparisons, upstream/current history, source comments,
imports, package locations, and authoritative repository metadata. “Exact”
below means an identical Git blob, not an independent legal conclusion.

## A. T2ICount-derived files

| Scope | Provenance assessment |
| --- | --- |
| `FSC-147-S.json`, `asset/`, `configs/` | Byte-identical to the corresponding upstream T2ICount files. |
| `models/decoder.py`, most existing `utils/` modules | Directly inherited from upstream. |
| `datasets/carpk.py`, `datasets/dataset.py`, `models/diff_unet.py`, `models/reg_model.py`, `simple_subset_test.py`, `test.py`, `train.py`, `utils/regression_trainer.py`, `visualize.py` | Upstream T2ICount files modified for portability, additional evaluation support, shared construction/inference, or resume persistence. These are mixed upstream/maintainer-authored files. |
| `README.md`, `environment.yaml` | Upstream documents/specifications with substantial repository-specific edits. |

No explicit T2ICount license grant or source-file copyright/license headers were
found. The upstream README identifies the official implementation and retains
the paper citation, but does not state redistribution terms. The lack of an
explicit grant makes permission for the T2ICount-derived portions unclear.

## B. Third-party-derived or installed components

The upstream T2ICount commit contains 18 tracked files under `ldm/`. At that
commit, 14 are exact blob matches to
[`CompVis/latent-diffusion`](https://github.com/CompVis/latent-diffusion), and
15 are exact matches to
[`CompVis/stable-diffusion`](https://github.com/CompVis/stable-diffusion).
`ldm/modules/encoders/modules.py` matches Stable Diffusion but not the Latent
Diffusion tree. Three files (`ddim.py`, `ddpm.py`, and `attention.py`) do not
exactly match either current authoritative tree and should be treated as
adapted/uncertain rather than assigned a more specific origin.

The included LDM tree also contains preserved source comments crediting OpenAI
CLIP/guided-diffusion code, while the authoritative Latent Diffusion README
credits OpenAI ADM and lucidrains' `x-transformers`. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for license links and
notices.

OpenAI CLIP and taming-transformers are installed dependencies, not tracked
vendored code. Ignored local editable checkouts under `src/` supply the verified
Windows environment; they are excluded from this repository's Git index.

## C. Repository-specific additions

The following files were introduced after the upstream tip and are attributable
to this implementation work, subject to any third-party APIs or snippets they
call:

- `.gitignore`;
- `models/build.py`;
- `utils/clip.py`, `utils/inference.py`, and `utils/paths.py`;
- `scripts/check_assets.py`;
- `tests/test_infrastructure.py` and `tests/test_checkpoint_resume.py`;
- `notebooks/train_colab.ipynb` and `notebooks/visualize_results.ipynb`;
- `requirements-colab.txt`;
- the files under `docs/` and the attribution/notices created by this audit.

Clearly separated new files such as these could receive file-specific
authorship or licensing treatment later. No repository-wide copyright or
license claim should be applied to inherited or mixed files without authority.

## Non-code assets and annotations

`asset/teaser.jpg`, `asset/visualization.jpg`, `asset/FSC-147-S-v1.json`, and
root `FSC-147-S.json` are exact copies from upstream T2ICount. Upstream displays
or links them but does not document a separate license, creator, or
redistribution permission for the images/annotations. A software license would
not automatically settle image rights unless its scope said so.

Conservative options are to obtain written permission or clarification from
the upstream authors, or remove the copies and link to their upstream locations.
This audit leaves them unchanged to avoid making an unsupported deletion or
substitution decision.

## Overall status

**Category C: upstream redistribution rights are unclear because no explicit
T2ICount license was found.** Third-party notices improve attribution but do not
resolve that missing upstream permission. Conservative next steps include
keeping the repository private while seeking clarification, asking upstream to
add a license, obtaining explicit permission, or publishing only original
patches/extensions rather than a full upstream copy.
