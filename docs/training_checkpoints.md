# Training checkpoints

`train.py --save-dir DIR --content RUN` writes logs and checkpoints to
`DIR/RUN`. Point `DIR` at Google Drive for Colab training; files under
`/content` disappear when the runtime is recycled.

Periodic `*_ckpt.tar` files are full-state resume checkpoints. Version 2 stores:

- model state;
- AdamW optimizer state;
- completed and next epoch;
- best validation MAE and RMSE;
- Python, NumPy, PyTorch, and CUDA RNG state.

Checkpoint replacement is atomic within the destination directory. The training
loop does not create an LR scheduler or AMP scaler, so neither has state to
save. Older `.tar` files containing only `epoch`, `model_state_dict`, and
`optimizer_state_dict` remain supported; missing best metrics default to
infinity and missing RNG state is skipped.

Best-model `.pth` files remain weights-only inference artifacts. They are not
accepted by `--resume`, because they cannot restore optimizer or epoch state.

Example:

```bash
python train.py \
  --asset-root "/content/drive/MyDrive/T2ICount-assets" \
  --save-dir "/content/drive/MyDrive/T2ICount-assets/checkpoints/baseline_retrain" \
  --content run_01 \
  --resume "/content/drive/MyDrive/T2ICount-assets/checkpoints/baseline_retrain/run_01/50_ckpt.tar"
```
