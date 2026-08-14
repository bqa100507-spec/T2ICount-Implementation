import torch
import numpy as np
from models.build import build_t2icount
from utils.inference import build_prompt_attention_mask, predict_density
from utils.paths import AssetPaths
from utils.regression_trainer import setup_seed
from torchvision import transforms
from PIL import Image
import json


with open('FSC-147-S.json', 'r') as f:
    data = json.load(f)


setup_seed(15)

config = 'configs/v1-inference.yaml'
assets = AssetPaths.from_sources()
crop_size = 384
model = build_t2icount(
    config, assets.sd_checkpoint, assets.clip_dir,
    checkpoint_path=assets.official_checkpoint, device='cuda', mode='eval'
)
tokenizer = model.clip.tokenizer

error = []
wrong_percent_list = []
for step, img_file in enumerate(data.keys()):
    gt = data[img_file]['count']
    cls_name = data[img_file]['class']
    prompt_attn_mask = build_prompt_attention_mask(tokenizer, cls_name)
    img_path = assets.dataset_dir('fsc147') / 'images_384_VarV2' / img_file

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    im = Image.open(img_path).convert('RGB')
    im = transform(im).unsqueeze(0).to('cuda')

    with torch.no_grad():
        results = predict_density(
            model, im, cls_name, prompt_attn_mask,
            batch_size=4, patch_size=384, stride=384,
        ).detach().cpu().squeeze(0).squeeze(0)

        pred_count = results.sum().item()
        wrong_percent = abs(gt - pred_count) / gt * 100
        print(
            f"[{step + 1}/{len(data)}] "
            f"{img_file} | prompt={cls_name} | "
            f"GT={gt} | Pred={pred_count:.2f} | "
            f"Error={abs(gt - pred_count):.2f}"
            f" | Wrong Percent={wrong_percent:.2f}%"
        )

        error.append(abs(gt - pred_count))
        wrong_percent_list.append(wrong_percent)

mae = np.array(error).mean()
mse = np.sqrt(np.mean(np.square(error)))
avg_wrong_percent = np.mean(wrong_percent_list)
print('MAE:', mae, 'MSE:', mse, 'Avg Wrong Percent:', avg_wrong_percent)
