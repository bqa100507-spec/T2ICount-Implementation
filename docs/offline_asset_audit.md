# Offline asset-loading audit

The active construction chain is:

`models.build.build_t2icount` -> `models.reg_model.Count` ->
`ldm.models.diffusion.ddpm.LatentDiffusion` ->
`ldm.modules.encoders.modules.FrozenCLIPEmbedder`.

`build_t2icount` validates the local SD checkpoint and CLIP directory, loads the
repository YAML, and overrides the conditioning-stage `version` with that local
CLIP directory before LDM instantiation. `FrozenCLIPEmbedder` always passes
`local_files_only=True`; it has no remote model-ID fallback.

FSC-147 and CARPK reuse the tokenizer held by the constructed model. IDCIA
prompt masks use the same tokenizer. This removes the independent remote-ID
tokenizer calls previously made by dataset and evaluation code.

The upstream `BERTTokenizer`, `FrozenCLIPTextEmbedder`, and
`FrozenClipImageEmbedder` classes remain in `ldm/modules/encoders/modules.py`.
They contain potential network-capable loaders, but the shipped
`configs/v1-inference.yaml` does not instantiate them and T2ICount does not call
them. They were deliberately left unchanged to avoid unrelated LDM rewrites.

The vendored OpenAI CLIP checkout under `src/clip` also contains upstream model
download helpers. It is an installation dependency, not an active T2ICount
checkpoint-loading path.
