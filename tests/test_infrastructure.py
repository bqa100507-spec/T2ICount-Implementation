import tempfile
import unittest
from pathlib import Path

import torch

from utils.inference import build_prompt_attention_mask, predict_count
from utils.paths import AssetPathError, AssetPaths, require_directory


class _Tokenizer:
    def __call__(self, prompt, add_special_tokens=False, return_tensors='pt'):
        return {'input_ids': torch.tensor([[10, 11]])}


class _ConstantDensityModel:
    def __call__(self, images, prompts, prompt_mask):
        batch = images.size(0)
        # reassemble_patches divides by 64 and this project then divides by 60.
        # 3840 therefore produces a final density of one per image pixel.
        density = torch.full((batch, 1, 2, 2), 3840.0, device=images.device)
        return density, None, None, None


class InfrastructureTests(unittest.TestCase):
    def test_asset_layout(self):
        root = Path('external-assets').absolute()
        paths = AssetPaths(root)
        self.assertEqual(
            paths.clip_dir,
            root / 'pretrained' / 'clip-vit-large-patch14',
        )
        self.assertEqual(paths.dataset_dir('idcia'), root / 'datasets' / 'IDCIA')

    def test_missing_directory_error_is_absolute(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / 'missing-clip'
            with self.assertRaises(AssetPathError) as context:
                require_directory(missing, 'CLIP model')
            self.assertIn(str(missing.absolute()), str(context.exception))

    def test_prompt_mask_semantics(self):
        mask = build_prompt_attention_mask(_Tokenizer(), 'two tokens')
        self.assertEqual(mask.shape, (77,))
        self.assertEqual(mask.nonzero().flatten().tolist(), [1, 2])

    def test_original_density_scaling_is_preserved(self):
        inputs = torch.zeros(1, 3, 16, 16)
        mask = build_prompt_attention_mask(_Tokenizer(), 'object')
        count = predict_count(
            _ConstantDensityModel(),
            inputs,
            'object',
            mask,
            batch_size=1,
            patch_size=16,
            stride=16,
        )
        self.assertEqual(count, 256.0)


if __name__ == '__main__':
    unittest.main()
