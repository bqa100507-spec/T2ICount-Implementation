from pathlib import Path

from transformers import CLIPTokenizer

from utils.paths import require_directory


def load_clip_tokenizer(clip_path):
    """Load the Stable Diffusion CLIP tokenizer without Hub fallback."""
    local_path = require_directory(clip_path, "CLIP model")
    return CLIPTokenizer.from_pretrained(str(local_path), local_files_only=True)
