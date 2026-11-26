# Claude Code notes for T2ICount-Implementation (ZSC / ZSOC research)

@AGENTS.md

The research protocol in `AGENTS.md` is mandatory. The notes below add current context and Claude-specific limits.

## What this repository is

An independent experimental extension of T2ICount (CVPR 2025, upstream `cha15yq/T2ICount`), not the official implementation. Keep attribution in `docs/provenance.md` and `THIRD_PARTY_NOTICES.md`. Counting path: image → SD v1.5 VAE latent → one-step U-Net → diffusion features / cross-attention → HSCM → counter → density map. The CLIP image encoder is not in the counting path.

Research branch: `rich-prompt-alignment` (not `main`). Assets live outside Git under `T2ICOUNT_ASSET_ROOT` (Windows: `D:\T2ICount-assets`; Colab: `/content/T2ICount-assets`, persistent copies on Drive `MyDrive/T2ICount-assets/`). Keep `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; offline loading must fail fast.

## Current state (audited 2026-11-26)

`PROJECT_STATE.md` and `EXPERIMENT_LEDGER.md` were last updated on 2026-10-29 and do not yet record RP-010, RP-011 or RP-012. Until the owner reviews and appends them, treat these as the latest raw facts:

| ID | Status | Evidence location (Drive) |
|---|---|---|
| RP-010A template diversity, 1000×10, seed 3407 | training + evaluation complete | `outputs/rp010/…/pair_v1/RP-010A/` (`epoch_records.json`, `selection.json`, `evaluation_class/summary.json`) |
| RP-010B rich supervision, matched config | training + evaluation complete | `outputs/rp010/…/pair_v1/RP-010B/` (same files) |
| RP-011 HNW vs control, 2-epoch warm start from RP-010B `best_epoch9.pth` | both arms complete, 2,000 steps each | `outputs/rp011/…/pair_v1/control/`, `…/pair_v1/hnw/` |
| RP-012 official-checkpoint one-epoch warm start | prepared only (`configs/rp012.json`, `tools/train_rp012.py`, `notebooks/RP012_Colab.ipynb`) | not run |

RP-011 and RP-012 source files (2026-11-08/09) were created after the last local commit `9f673ce` and may be uncommitted. Check `git status` before editing.

## Limits for Claude

- Do not launch RP-012 or any training. Its notebook keeps `RUN_TRAINING = False` until the owner reviews the frozen baseline and GPU preflight.
- Do not finalize scientific conclusions in the state/ledger; draft them and ask the owner to review (AGENTS.md rules 10–12).
- RP-010 compares two supervision regimes at one seed; do not call it a causal or replicated effect.
- DUMLO is parked, not deleted. Do not mix DUMLO and rich-prompt changes.
- `.tar` = full resumable state; `.pth` = weights only. Never resume from `.pth`.
