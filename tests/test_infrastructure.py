import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from utils.checkpoints import load_trusted_legacy_checkpoint
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
    def test_trusted_legacy_checkpoint_disables_weights_only(self):
        expected = object()
        with mock.patch(
            'utils.checkpoints.torch.load', return_value=expected
        ) as torch_load:
            result = load_trusted_legacy_checkpoint(
                'checkpoint.ckpt', map_location='cpu'
            )

        self.assertIs(result, expected)
        torch_load.assert_called_once_with(
            'checkpoint.ckpt', map_location='cpu', weights_only=False
        )

    def test_trusted_legacy_checkpoint_falls_back_for_old_pytorch(self):
        expected = object()
        with mock.patch(
            'utils.checkpoints.torch.load',
            side_effect=[TypeError('unsupported argument'), expected],
        ) as torch_load:
            result = load_trusted_legacy_checkpoint(
                'checkpoint.ckpt', map_location='cpu'
            )

        self.assertIs(result, expected)
        self.assertEqual(
            torch_load.call_args_list,
            [
                mock.call(
                    'checkpoint.ckpt',
                    map_location='cpu',
                    weights_only=False,
                ),
                mock.call('checkpoint.ckpt', map_location='cpu'),
            ],
        )

    def test_repository_local_package_imports(self):
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                (
                    'from pathlib import Path; '
                    'import datasets, models, utils; '
                    'from datasets.carpk import CARPK; '
                    'from datasets.dataset import ObjectCount; '
                    'from models.build import build_t2icount; '
                    'from utils.paths import AssetPaths; '
                    'root = Path.cwd().resolve(); '
                    'assert Path(datasets.__file__).resolve() == '
                    'root / "datasets" / "__init__.py"; '
                    'assert Path(models.__file__).resolve() == '
                    'root / "models" / "__init__.py"; '
                    'assert Path(utils.__file__).resolve() == '
                    'root / "utils" / "__init__.py"; '
                    'print(datasets.__file__)'
                ),
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(repo_root / 'datasets' / '__init__.py'), result.stdout)

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
