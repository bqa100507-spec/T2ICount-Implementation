import torch


def load_trusted_legacy_checkpoint(path, map_location="cpu"):
    """Load a trusted legacy research checkpoint across PyTorch versions."""
    try:
        # Intentional for trusted legacy research checkpoints: PyTorch >=2.6
        # changed torch.load's default to weights_only=True.
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # PyTorch 1.11 does not expose the weights_only argument.
        return torch.load(path, map_location=map_location)
