# T2ICount Implementation / Experimental Extension

This is an independent research implementation and experimental extension
based substantially on the official T2ICount codebase for the CVPR 2025 paper
**"T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting."**

- Original paper: [arXiv:2502.20625](https://arxiv.org/abs/2502.20625)
- Official upstream repository: [cha15yq/T2ICount](https://github.com/cha15yq/T2ICount)
- Original authors: Yifei Qian, Zhongliang Guo, Bowen Deng, Chun Tong Lei,
  Shuai Zhao, Chun Pong Lau, Xiaopeng Hong, and Michael P. Pound

The original authors and paper remain fully credited. This repository contains
implementation modifications and experimental infrastructure maintained
independently by the current repository maintainer. It is not the official
T2ICount repository and is not affiliated with or endorsed by the original
authors.

![Original upstream T2ICount teaser](asset/teaser.jpg)

## Attribution, provenance, and license status

Substantial portions of this repository are inherited or modified from the
upstream T2ICount implementation. As of the 2026-08-14 audit, no explicit
`LICENSE`, `LICENSE.txt`, `LICENSE.md`, or README license grant was present in
the upstream repository, and GitHub reported no detected repository license.
This repository therefore does **not** claim to grant additional rights over
the upstream T2ICount code, annotations, or visual assets. Public availability
on GitHub is not treated here as a redistribution license.

Third-party components remain subject to their own terms. See
[the provenance audit](docs/provenance.md) and
[third-party notices](THIRD_PARTY_NOTICES.md). This documentation is repository
hygiene, not legal advice.

## Changes in this repository

Relative to upstream T2ICount, this implementation adds or extends:

- a portable external asset layout and fail-fast path resolution;
- fully offline CLIP/model loading and shared model/inference APIs;
- FSC-147, CARPK, and IDCIA evaluation integration, including IDCIA prompt and
  autocontrast diagnostics;
- Google Colab/Drive and VSCode-terminal orchestration;
- full-state training checkpoint/resume support;
- infrastructure tests, asset validation, and a visualization notebook.

These descriptions identify implementation work in this repository; they do
not claim ownership of the original T2ICount method or upstream code.

## RichCount-inspired full-image alignment

`tools/train_rich_alignment.py` is a standalone Stage 1 visual-text alignment
experiment for T2ICount. It is inspired by RichCount, but it is not an exact
RichCount reproduction: it uses the project's local CLIP ViT-L/14 model and one
full stored FSC147 384 image from `images_384_VarV2/` per sample. The image is
processed only by the standard local `CLIPProcessor`; no T2ICount crop, resize,
mosaic, flip, or prompt augmentation is used, and no visual prompts or cropped
regions are invented.

The experiment uses the exact deterministic 1000-image train subset and prompt
bank with seed 3407, then makes a separate deterministic 900/100 alignment
train/validation split. It first freezes all CLIP parameters and trains a
768-dimensional visual FFN. It then freezes CLIP and the trained FFN and trains
a residual token-wise `[B,T,768] -> [B,T,768]` text adapter. All class,
detailed, and generalized prompts are positive pairs; deterministic negatives
use the same prompt mode from a different FSC147 class. A different class is
not guaranteed to be absent from an image, so these negatives may contain
label noise.

Validation compares frozen CLIP, the trained FFN, and the trained adapter on
the held-out 100 samples. It reports positive and negative configured Euclidean
distance, separation gap, pairwise accuracy, margin violations, cosine
diagnostics, and Euclidean retrieval R@1. Duplicate class prompts use
class-aware retrieval correctness. FSC-147-S is not used for training or model
selection. This stage does not construct or train the counting model, density
maps, RichPrompt Phase 2 loss, or DUMLO, and it makes no claim about improved
counting performance.

Raw Euclidean distance remains the default. In the first raw-distance smoke,
projected ViT-L/14 validation distances were approximately 20--23, so no
negative violated margin 1 and the negative hinge term was inactive. The
opt-in `--normalize-embeddings` follow-up L2-normalizes only the final image and
text embeddings immediately before every Euclidean loss, metric, and retrieval
calculation. This maps their Euclidean distance to `[0,2]` while leaving CLIP
hidden states, FFN inputs, adapter token outputs, architectures, learning rates,
and margin unchanged. It is an experimental adaptation for this repository,
not an exact or undocumented RichCount implementation detail.

Validate the fixed data contract and run a real one-image/three-prompt offline
CLIP forward without training or output writes:

```bash
export RICH_PROMPT_BANK="/path/to/fsc147_train1000_seed3407_gemma4_v3.json"
python tools/train_rich_alignment.py \
  --asset-root "$T2ICOUNT_ASSET_ROOT" \
  --prompt-bank "$RICH_PROMPT_BANK" \
  --device cpu \
  --validate-only
```

## Portable offline setup

Clone this implementation repository:

```bash
git clone https://github.com/bqa100507-spec/T2ICount-Implementation.git
cd T2ICount-Implementation
```

Code and small configuration files live in this repository. Read-only model,
dataset, and official-checkpoint assets live in one external directory:

```text
T2ICount-assets/
|-- pretrained/
|   |-- clip-vit-large-patch14/
|   `-- sd-v1-5/v1-5-pruned-emaonly.ckpt
|-- datasets/
|   |-- FSC147/
|   |-- CARPK/
|   `-- IDCIA/
`-- checkpoints/
    `-- official/best_model_paper.pth
```

`FSC147/` retains `images_384_VarV2/`,
`gt_density_map_adaptive_384_VarV2/`, and `FSC_147/`. `CARPK/` retains
`Images/`, `Annotations/`, and `ImageSets/`. IDCIA retains `images/`,
`ground_truth/`, and the official `test.csv`.

### Windows PowerShell

```powershell
& "C:\Users\MSl\miniconda3\shell\condabin\conda-hook.ps1"
conda activate t2itest
Remove-Item Env:SSLKEYLOGFILE -ErrorAction SilentlyContinue
$env:T2ICOUNT_ASSET_ROOT = "D:\T2ICount-assets"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python scripts/check_assets.py --asset-root "D:\T2ICount-assets" --data fsc147 --check-offline-load
python test.py --asset-root "D:\T2ICount-assets" --data fsc147 --batch-size 1 --max-samples 1
```

The historical environment remains pinned in `environment.yaml` because the
upstream SD/T2ICount checkpoint path is sensitive to framework changes. See
`docs/dependency_audit.md` before upgrading it.

### Colab with Google Drive

Open `notebooks/train_colab.ipynb`. It mounts Drive, clones/opens this repo,
copies `/content/drive/MyDrive/T2ICount-assets.zip` to local Colab storage when
the required runtime assets are incomplete, and extracts it as
`/content/T2ICount-assets`. It sets
`T2ICOUNT_ASSET_ROOT=/content/T2ICount-assets`, validates the four required
local assets, and runs a one-image smoke test from that local root. The notebook
contains orchestration only and does not start full training automatically.

Long-running Colab runs should use an explicit Drive save directory, such as
`/content/drive/MyDrive/T2ICount-assets/checkpoints/baseline_retrain/run_01`,
because `/content` is temporary. New `.tar` checkpoints contain model weights,
optimizer state, the next epoch, best validation metrics, and RNG state. This
project has no LR scheduler or AMP scaler, so there is no scheduler/scaler state
to restore. Legacy `.tar` checkpoints remain loadable; `.pth` files are
weights-only and are not accepted by `--resume`.

The runtime asset root is treated as read-only. Evaluation CSVs default to the
repository-local `results/` directory unless `--results-path` is supplied, and
training requires an explicit writable `--save-dir`.

IDCIA result analysis and density-map visualization are in
`notebooks/visualize_results.ipynb`; this notebook uses the same external asset,
model-building, and inference APIs as `test.py`.

### Explicit paths without the environment variable

```powershell
python test.py --data fsc147 `
  --dataset-root "D:\T2ICount-assets\datasets\FSC147" `
  --clip-path "D:\T2ICount-assets\pretrained\clip-vit-large-patch14" `
  --sd-path "D:\T2ICount-assets\pretrained\sd-v1-5\v1-5-pruned-emaonly.ckpt" `
  --model-path "D:\T2ICount-assets\checkpoints\official\best_model_paper.pth" `
  --batch-size 16
```

Every Hugging Face load in the active T2ICount path is local-only. Missing
assets fail with the exact missing path instead of falling back to the Hub. See
`docs/offline_asset_audit.md` for the loader audit.

Heavy datasets, model weights, checkpoints, logs, and generated outputs are
intentionally excluded from Git. The small original `asset/` directory remains
part of the repository.

## Upstream dataset and model references

The original environment can be created with Anaconda:
```
conda env create -f environment.yaml
```
**Data:** The upstream T2ICount project reports experiments over three datasets.
The datasets can be obtained from their respective sources: [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything) | [CARPK](https://lafi.github.io/LPN/).
Notice that you have to download the annoations of FSC-147 separately from [their repo](https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master/data).

Keep the downloaded dataset structure below, but place it under the external
`datasets/` directory described above. FSC-147 is required for training.
```
data
├─CARPK/
│  ├─Annotations/
│  ├─Images/
│  ├─ImageSets/
│
├─FSC/    
│  ├─gt_density_map_adaptive_384_VarV2/
│  ├─images_384_VarV2/
│  ├─FSC_147/
│  │  ├─ImageClasses_FSC147.txt
│  │  ├─Train_Test_Val_FSC_147.json
│  │  ├─ annotation_FSC147_384.json
```
**Stable Diffusion:** The upstream T2ICount model was developed by fine-tuning Stable Diffusion v1.5. The referenced weights can be obtained [from their source](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt), subject to the terms applicable there.
Store this checkpoint under external `pretrained/sd-v1-5/`, not in `configs/`.

## FSC-147-S-v2

The original T2ICount authors report that FSC-147-S-v2 was introduced after a
review-stage reassessment and contains 230 images. The statistics and results
below are upstream-reported results, not claims of authorship by this repository
maintainer. The redistributed [FSC-147-S.json](FSC-147-S.json) and
[FSC-147-S-v1.json](asset/FSC-147-S-v1.json) are byte-identical to the files in
the [upstream repository](https://github.com/cha15yq/T2ICount); their licensing
status follows the upstream uncertainty described above.

| Medthod     |      MAE     |     RMSE     | 
|-------------|--------------|--------------|
| CLIP-Count  |    45.59     |    98.96     | 
| CountX      |    28.67     |    89.18     | 
| VLCounter   |    33.10     |    69.34     | 
| PseCo       |    30.53     |    43.92     | 
| DAVE        |    46.36     |    97.11     | 
| T2ICount (upstream result) | 5.99 | 10.55 |

---
## Train
Once you have prepared the data and the pretrained weights of SD1.5, you can train the model using the following command. 

The upstream authors report reproducibility results and provide an
[upstream training log](https://github.com/cha15yq/T2ICount/blob/main/logs/train.log)
and [reproduced model](https://drive.google.com/file/d/1VN5uI9F0XjKQ-JwjOpMYYJF5ku92znO_/view?usp=sharing).
```
python train.py --asset-root "$T2ICOUNT_ASSET_ROOT" --save-dir "/path/to/writable/checkpoints" --content exp --crop-size 384 --concat-size 224 --batch-size 16 --lr 5e-5 --weight-decay 5e-5
```
---
## Evaluation and the pretrained model

The upstream project provides a [pre-trained checkpoint](https://drive.google.com/file/d/1lw5LgpYP7vTazaMWTgNa6nFoZ63j-st9/view?usp=sharing)
identified there as the model used for the paper results. It is not distributed
in this Git repository.
| FSC val MAE | FSC val RMSE | FSC test MAE |  FSC test RMSE | CARPK MAE | CARPK RMSE |
|-------------|--------------|--------------|----------------|-----------|------------|
| 13.78       | 58.78        | 11.76        | 97.86          | 8.61      | 13.47      |

| FSC S-v2 MAE | FSC S-v2 MSE | 
|--------------|--------------|
| 5.99       | 10.55        |
```
python test.py --data fsc147 --batch-size 16
```
---
## Gallery
![more](asset/visualization.jpg)
## Citation

Please cite the original T2ICount paper when using the method or upstream code:
```
@inproceedings{qian2025t2icount,
               title={T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting}, 
               author={Qian, Yifei and Guo, Zhongliang and Deng, Bowen and Lei, Chun Tong and Zhao, Shuai and Lau, Chun Pong and Hong, Xiaopeng and Pound, Michael P},
               year={2025},
               booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition}
}
```
