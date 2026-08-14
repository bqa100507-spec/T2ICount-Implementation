from typing import Mapping, Optional

import torch
from omegaconf import OmegaConf

from models.reg_model import Count
from utils.checkpoints import load_trusted_legacy_checkpoint
from utils.paths import require_directory, require_file


DEFAULT_UNET_CONFIG = {
    "base_size": 384,
    "max_attn_size": 384 // 8,
    "attn_selector": "down_cross+up_cross",
}


def load_t2icount_checkpoint(model, checkpoint_path, strict=True):
    checkpoint = require_file(checkpoint_path, "T2ICount checkpoint")
    state = load_trusted_legacy_checkpoint(
        str(checkpoint), map_location="cpu"
    )
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=strict)
    return model


def build_t2icount(
    config_path,
    sd_checkpoint,
    clip_path,
    checkpoint_path=None,
    device="cuda",
    mode="eval",
    unet_config: Optional[Mapping] = None,
):
    """Construct the unchanged T2ICount architecture from explicit local assets."""
    config_file = require_file(config_path, "Stable Diffusion config")
    sd_file = require_file(sd_checkpoint, "Stable Diffusion checkpoint")
    clip_dir = require_directory(clip_path, "CLIP model")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")
    if mode not in ("eval", "train"):
        raise ValueError("mode must be 'eval' or 'train'.")

    config = OmegaConf.load(str(config_file))
    cond_stage = config.model.params.cond_stage_config
    if "params" not in cond_stage:
        cond_stage.params = {}
    cond_stage.params.version = str(clip_dir)
    cond_stage.params.device = str(torch_device)

    model = Count(
        config,
        str(sd_file),
        unet_config=dict(unet_config or DEFAULT_UNET_CONFIG),
    )
    if checkpoint_path is not None:
        load_t2icount_checkpoint(model, checkpoint_path)
    model = model.to(torch_device)
    if mode == "eval":
        model.set_eval()
    else:
        model.set_train()
    return model
