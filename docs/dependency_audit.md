# Dependency modernization audit

No dependency versions were changed in the verified Windows environment. The
verified local environment is Python 3.8.5, PyTorch 1.11.0 + CUDA 11.3,
Torchvision 0.12.0, Transformers 4.19.2, PyTorch Lightning 1.4.2,
OmegaConf 2.1.1, Kornia 0.6.0, and NumPy 1.24.4.

## Risk assessment

- Low-risk candidates: packaging and notebook-only tools, provided project
  imports and offline smoke tests are rerun.
- Medium-risk candidates: NumPy, Pillow, OpenCV, h5py, and OmegaConf. Dataset
  transforms, image resize behavior, and YAML mutation should be regression
  tested.
- High-risk candidates: Python, PyTorch/Torchvision/CUDA, Transformers,
  PyTorch Lightning, Kornia, and the Stable Diffusion/taming/OpenAI-CLIP
  snapshots. Their serialized checkpoints and internal APIs are coupled to the
  current implementation.

Known upgrade hazards include Lightning Trainer API changes, Transformers CLIP
state-dict/loading changes, removed private Transformers symbols imported by
`models/diff_unet.py`, Torch checkpoint/runtime differences, and altered image
resize/antialias behavior in Torchvision/Kornia.

Current Colab runtimes use Python 3.12 and PyTorch 2.x, so the Python 3.8-era
pins cannot be installed unchanged. `requirements-colab.txt` is deliberately a
separate compatibility bridge: it keeps Colab's PyTorch/Torchvision, stays on
Transformers 4.x, and uses Lightning 2.x. Two source-only compatibility edits
support both environments: an unused private Transformers import was removed,
and the Lightning `rank_zero_only` import accepts both old and new locations.

The Windows checkpoint regression validates only the preserved environment.
Colab must run the same one-image smoke comparison before its dependency bridge
is accepted for training; framework-level numerical equivalence is not assumed.
