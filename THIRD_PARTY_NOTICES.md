# Third-party notices

This file records major third-party provenance and license information. It does
not license T2ICount-derived code and does not imply that a dependency's license
applies to the repository as a whole.

## Code physically included through the upstream `ldm/` tree

| Component | Evidence and scope | License / notice |
| --- | --- | --- |
| [CompVis Latent Diffusion](https://github.com/CompVis/latent-diffusion) | Fourteen of the 18 upstream T2ICount `ldm/` files are exact Git-blob matches; other LDM files appear adapted. | MIT; copyright 2022 Machine Vision and Learning Group, LMU Munich. A copy is preserved at [`third_party/licenses/latent-diffusion-LICENSE.txt`](third_party/licenses/latent-diffusion-LICENSE.txt). |
| [CompVis Stable Diffusion](https://github.com/CompVis/stable-diffusion) | Fifteen upstream `ldm/` files match, including `ldm/modules/encoders/modules.py`, which does not match the Latent Diffusion tree. Model weights are external and are not distributed by this Git repository. | [CreativeML Open RAIL-M](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE); copyright 2022 Robin Rombach, Patrick Esser, and contributors. The authoritative text is linked rather than paraphrased here. |
| [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion), [improved-diffusion](https://github.com/openai/improved-diffusion), and [CLIP](https://github.com/openai/CLIP) | Source comments in `ldm/` credit guided-diffusion and CLIP; the authoritative Latent Diffusion project also credits OpenAI ADM. | MIT; copyright 2021 OpenAI. The common notice is preserved at [`third_party/licenses/openai-MIT-LICENSE.txt`](third_party/licenses/openai-MIT-LICENSE.txt). |
| [lucidrains/x-transformers](https://github.com/lucidrains/x-transformers) | `ldm/modules/x_transformer.py` matches the CompVis LDM copy; the authoritative LDM README identifies its transformer encoder as coming from x-transformers. | MIT; copyright 2020 Phil Wang. A copy is preserved at [`third_party/licenses/x-transformers-LICENSE.txt`](third_party/licenses/x-transformers-LICENSE.txt). |

The three upstream files that do not exactly match the current CompVis trees
(`ldm/models/diffusion/ddim.py`, `ldm/models/diffusion/ddpm.py`, and
`ldm/modules/attention.py`) retain their existing source comments. Their precise
change provenance was not guessed.

## Installed or external dependencies

These projects are dependencies and are not tracked as vendored source in this
repository:

| Dependency | Resolution used here | Authoritative license |
| --- | --- | --- |
| [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers) | Git dependency pinned to `3ba01b241669f5ade541ce990f7650a3b8f65318` | [MIT](https://github.com/CompVis/taming-transformers/blob/master/License.txt) |
| [OpenAI CLIP](https://github.com/openai/CLIP) | Git dependency pinned to `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` | [MIT](https://github.com/openai/CLIP/blob/main/LICENSE) |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Python dependency | [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) | Python dependency | [Apache-2.0](https://github.com/Lightning-AI/pytorch-lightning/blob/master/LICENSE) |
| [Kornia](https://github.com/kornia/kornia) | Python dependency | [Apache-2.0](https://github.com/kornia/kornia/blob/main/LICENSE) |

Other packages remain subject to the license distributed by their respective
authors. External datasets, CLIP/Stable Diffusion weights, and the upstream
T2ICount checkpoint are deliberately excluded from Git; users must review the
terms at their acquisition sources. In particular, a model-weight license must
not be confused with permission for the T2ICount source code or non-code assets.
