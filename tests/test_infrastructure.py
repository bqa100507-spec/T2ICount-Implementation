import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from utils.checkpoints import load_trusted_legacy_checkpoint
from utils.inference import build_prompt_attention_mask, predict_count
from utils.paths import AssetPathError, AssetPaths, require_directory
from scripts import check_assets
from test import resolve_cli_paths
from train import resolve_training_paths


def _create_runtime_assets(root, include_official_checkpoint=True):
    clip_dir = root / 'pretrained' / 'clip-vit-large-patch14'
    fsc147_dir = root / 'datasets' / 'FSC147'
    clip_dir.mkdir(parents=True)
    fsc147_dir.mkdir(parents=True)

    sd_checkpoint = (
        root / 'pretrained' / 'sd-v1-5' / 'v1-5-pruned-emaonly.ckpt'
    )
    sd_checkpoint.parent.mkdir(parents=True)
    sd_checkpoint.touch()

    official_checkpoint = (
        root / 'checkpoints' / 'official' / 'best_model_paper.pth'
    )
    if include_official_checkpoint:
        official_checkpoint.parent.mkdir(parents=True)
        official_checkpoint.touch()

    return AssetPaths(root)


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

    def test_test_cli_asset_root_overrides_stale_environment_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            runtime_root = temp_root / 'runtime-assets'
            stale_drive_root = temp_root / 'drive-assets'
            assets = _create_runtime_assets(runtime_root)
            stale_drive_root.mkdir()
            config = temp_root / 'v1-inference.yaml'
            config.touch()
            args = SimpleNamespace(
                asset_root=str(runtime_root),
                clip_path=None,
                sd_path=None,
                model_path=None,
                data='fsc147',
                idcia_root=None,
                dataset_root=None,
                config=str(config),
                results_path=None,
            )

            with mock.patch.dict(
                os.environ,
                {'T2ICOUNT_ASSET_ROOT': str(stale_drive_root)},
            ):
                resolved = resolve_cli_paths(args)

            self.assertEqual(resolved[1], assets.sd_checkpoint)
            self.assertEqual(resolved[2], assets.clip_dir)
            self.assertEqual(resolved[3], assets.official_checkpoint)
            self.assertEqual(resolved[4], assets.dataset_dir('fsc147'))
            self.assertEqual(
                resolved[5], Path('results/idcia_predictions.csv').resolve()
            )

    def test_training_save_root_does_not_change_runtime_asset_reads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            runtime_root = temp_root / 'runtime-assets'
            drive_root = temp_root / 'drive-assets'
            assets = _create_runtime_assets(runtime_root)
            config = temp_root / 'v1-inference.yaml'
            config.touch()
            save_dir = drive_root / 'checkpoints' / 'baseline_retrain'
            args = SimpleNamespace(
                asset_root=str(runtime_root),
                config=str(config),
                sd_path=None,
                clip_path=None,
                data_dir=None,
                save_dir=str(save_dir),
                resume='',
            )

            resolved = resolve_training_paths(args)

            self.assertEqual(Path(resolved.sd_path), assets.sd_checkpoint)
            self.assertEqual(Path(resolved.clip_path), assets.clip_dir)
            self.assertEqual(
                Path(resolved.data_dir), assets.dataset_dir('fsc147')
            )
            self.assertEqual(Path(resolved.save_dir), save_dir)

    def test_training_requires_explicit_writable_save_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            runtime_root = temp_root / 'runtime-assets'
            _create_runtime_assets(runtime_root)
            config = temp_root / 'v1-inference.yaml'
            config.touch()
            args = SimpleNamespace(
                asset_root=str(runtime_root),
                config=str(config),
                sd_path=None,
                clip_path=None,
                data_dir=None,
                save_dir=None,
                resume='',
            )

            with self.assertRaisesRegex(ValueError, 'runtime asset root is read-only'):
                resolve_training_paths(args)

    def test_asset_check_requires_official_checkpoint_from_runtime_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_root = Path(temp_dir) / 'runtime-assets'
            _create_runtime_assets(
                runtime_root, include_official_checkpoint=False
            )
            args = SimpleNamespace(
                asset_root=str(runtime_root),
                clip_path=None,
                sd_path=None,
                data='fsc147',
                dataset_root=None,
                model_path=None,
                device='cpu',
                check_offline_load=False,
            )

            with mock.patch.object(check_assets, 'parse_args', return_value=args):
                self.assertEqual(check_assets.main(), 1)

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
