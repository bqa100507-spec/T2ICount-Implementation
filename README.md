# T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting (CVPR2025)
## [Paper (ArXiv)](https://arxiv.org/abs/2502.20625) 

Official Implementation for CVPR 2025 paper T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting.
![teaser](asset/teaser.jpg)

## Portable offline setup

Clone this implementation repository:

```bash
git clone https://github.com/bqa100507-spec/T2ICount-Implementation.git
cd T2ICount-Implementation
```

Code and small configuration files live in this repository. Models, datasets,
checkpoints, and generated outputs live in one external directory:

```text
T2ICount-assets/
|-- pretrained/
|   |-- clip-vit-large-patch14/
|   `-- sd-v1-5/v1-5-pruned-emaonly.ckpt
|-- datasets/
|   |-- FSC147/
|   |-- CARPK/
|   `-- IDCIA/
|-- checkpoints/
|   |-- official/best_model_paper.pth
|   |-- baseline_retrain/
|   `-- dumlo/
`-- outputs/
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
official SD/T2ICount checkpoint path is sensitive to framework changes. See
`docs/dependency_audit.md` before upgrading it.

### Colab with Google Drive

Open `notebooks/train_colab.ipynb`. It mounts Drive, clones/opens this repo,
installs dependencies, sets
`T2ICOUNT_ASSET_ROOT=/content/drive/MyDrive/T2ICount-assets`, validates assets,
and runs a one-image smoke test. It also documents the separate environment
exports required by a VSCode terminal connected to the Colab runtime. The
notebook contains orchestration only and does not start full training
automatically.

Long-running Colab runs should use an explicit Drive save directory, such as
`/content/drive/MyDrive/T2ICount-assets/checkpoints/baseline_retrain/run_01`,
because `/content` is temporary. New `.tar` checkpoints contain model weights,
optimizer state, the next epoch, best validation metrics, and RNG state. This
project has no LR scheduler or AMP scaler, so there is no scheduler/scaler state
to restore. Legacy `.tar` checkpoints remain loadable; `.pth` files are
weights-only and are not accepted by `--resume`.

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

## Original dataset and model references

The original environment can be created with Anaconda:
```
conda env create -f environment.yaml
```
**Data:** We conduct experiments over three datasets, you can download and use whichever you would like to test.
The three dataset could be downloaded at: [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything) | [CARPK](https://lafi.github.io/LPN/).
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
**Stable Diffusion:** Our model is developed by fine-tuning Stable Diffusion v1.5, whose original weights can be downloaded from [here](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt).
Store this checkpoint under external `pretrained/sd-v1-5/`, not in `configs/`.

## FSC-147-S-v2
During the review process, the reviewers raised concerns regarding the dataset. In response, we conducted a thorough reassessment and introduced a revised version, which we named FSC-147-S-v2. This updated version includes an additional set of images, bringing the total to 230. As a result, the statistics of v2 differ from those originally reported in the paper. In this new subset, the objects originally annotated in these images from FSC-147 had an average count of 44.98, while the newly annotated objects have an average count of 3.96. The results from the baseline methods and our method are provided here. For the updated dataset (v2), please refer to [**FSC-147-S.json**](https://github.com/cha15yq/T2ICount/blob/main/FSC-147-S.json). As for the original subset used in the paper, you can download it [here](https://github.com/cha15yq/T2ICount/blob/main/asset/FSC-147-S-v1.json). We sincerely apologize for any confusion caused.

| Medthod     |      MAE     |     RMSE     | 
|-------------|--------------|--------------|
| CLIP-Count  |    45.59     |    98.96     | 
| CountX      |    28.67     |    89.18     | 
| VLCounter   |    33.10     |    69.34     | 
| PseCo       |    30.53     |    43.92     | 
| DAVE        |    46.36     |    97.11     | 
| T2ICount (Ours)    |    5.99     |    10.55     | 

We hope that this small subset can serve as an evaluation set to verify whether a model is truly performing zero-shot object counting.

---
## Train
Once you have prepared the data and the pretrained weights of SD1.5, you can train the model using the following command. 

We have tested the reproducibility of this code and obtained consistent results, the [training log](https://github.com/cha15yq/T2ICount/blob/main/logs/train.log) is provided along with the [reproduced model](https://drive.google.com/file/d/1VN5uI9F0XjKQ-JwjOpMYYJF5ku92znO_/view?usp=sharing).
```
python train.py --content exp --crop-size 384 --concat-size 224 --batch-size 16 --lr 5e-5 --weight-decay 5e-5
```
---
## Evaluation and the pretrained model

We provide a [pre-trained ckpt](https://drive.google.com/file/d/1lw5LgpYP7vTazaMWTgNa6nFoZ63j-st9/view?usp=sharing) of our full model, which is the exact model used to demonstrate the performance results presented in the paper.
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
Consider cite us if you find our paper is useful in your research :).
```
@inproceedings{qian2025t2icount,
               title={T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting}, 
               author={Qian, Yifei and Guo, Zhongliang and Deng, Bowen and Lei, Chun Tong and Zhao, Shuai and Lau, Chun Pong and Hong, Xiaopeng and Pound, Michael P},
               year={2025},
               booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition}
}
```
