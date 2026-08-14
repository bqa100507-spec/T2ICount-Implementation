import matplotlib.pyplot as plt
import torch
import numpy as np
from models.build import build_t2icount
from utils.inference import build_prompt_attention_mask, predict_density
from utils.paths import AssetPaths
from utils.regression_trainer import setup_seed
from torchvision import transforms
from PIL import Image
import cv2


setup_seed(15)

config = 'configs/v1-inference.yaml'
assets = AssetPaths.from_sources()
crop_size = 384
model = build_t2icount(
    config, assets.sd_checkpoint, assets.clip_dir,
    checkpoint_path=assets.official_checkpoint, device='cuda', mode='eval'
)
tokenizer = model.clip.tokenizer

img_path = assets.dataset_dir('fsc147') / 'images_384_VarV2' / '2143.jpg'
cls_name = 'baskets'

attention_mask = build_prompt_attention_mask(tokenizer, cls_name)



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
im = Image.open(img_path).convert('RGB')
im = transform(im).unsqueeze(0).to('cuda')

with torch.set_grad_enabled(False):
    results = predict_density(
        model, im, cls_name, attention_mask,
        batch_size=4, patch_size=384, stride=384,
    ).detach().cpu().squeeze(0).squeeze(0)
    pred_density = results.numpy()
    print('Predicted Number:', pred_density.sum())
    pred_density = pred_density / pred_density.max()
    pred_density_write = 1. - pred_density
    pred_density_write = cv2.applyColorMap(np.uint8(255 * pred_density_write), cv2.COLORMAP_JET)
    pred_density_write = pred_density_write / 255
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    heatmap_pred = 0.33 * img + 0.67 * pred_density_write
    heatmap_pred = heatmap_pred / heatmap_pred.max()
    cv2.imwrite('den.jpg', cv2.cvtColor((heatmap_pred * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

