import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.paths import (  # noqa: E402
    AssetPaths,
    AssetPathError,
    require_directory,
    require_file,
    resolve_asset_root,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate external T2ICount assets.")
    parser.add_argument('--asset-root', default=None)
    parser.add_argument('--clip-path', default=None)
    parser.add_argument('--sd-path', default=None)
    parser.add_argument('--data', default='fsc147', choices=['fsc147', 'carpk', 'idcia'])
    parser.add_argument('--dataset-root', default=None)
    parser.add_argument('--model-path', default=None,
                        help='Optional T2ICount checkpoint override to validate.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--check-offline-load', action='store_true',
                        help='Also instantiate the local CLIP tokenizer and text encoder.')
    return parser.parse_args()


def report_ok(label, path):
    print('[OK] {}: {}'.format(label, path))


def report_missing(label, path):
    print('[MISSING] {}:\n{}'.format(label, path))


def main():
    args = parse_args()
    failures = []

    try:
        root = resolve_asset_root(args.asset_root)
        assets = AssetPaths(root)
        report_ok('Asset root', root)
    except AssetPathError as exc:
        print('[MISSING] Asset root:\n{}'.format(exc))
        return 1

    checks = [
        ('CLIP directory', args.clip_path or assets.clip_dir, require_directory),
        ('Stable Diffusion checkpoint', args.sd_path or assets.sd_checkpoint, require_file),
        ('{} dataset'.format(args.data.upper()),
         args.dataset_root or assets.dataset_dir(args.data), require_directory),
    ]
    if args.model_path:
        checks.append(('T2ICount checkpoint', args.model_path, require_file))
    elif assets.official_checkpoint.exists():
        checks.append(('T2ICount checkpoint', assets.official_checkpoint, require_file))

    validated = {}
    for label, path, validator in checks:
        try:
            validated[label] = validator(path, label)
            report_ok(label, validated[label])
        except AssetPathError:
            missing = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
            report_missing(label, missing)
            failures.append(label)

    try:
        import torch
        if args.device.startswith('cuda'):
            if torch.cuda.is_available():
                report_ok('CUDA', torch.cuda.get_device_name(0))
            else:
                print('[MISSING] CUDA: torch.cuda.is_available() is False')
                failures.append('CUDA')
        else:
            report_ok('Torch device', args.device)
    except Exception as exc:
        print('[ERROR] Torch device check: {}'.format(exc))
        failures.append('Torch device')

    if args.check_offline_load and 'CLIP directory' in validated:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        try:
            from transformers import CLIPTextModel
            from utils.clip import load_clip_tokenizer
            clip_path = str(validated['CLIP directory'])
            load_clip_tokenizer(clip_path)
            CLIPTextModel.from_pretrained(clip_path, local_files_only=True)
            report_ok('Offline CLIP load', clip_path)
        except Exception as exc:
            print('[ERROR] Offline CLIP load: {}'.format(exc))
            failures.append('Offline CLIP load')

    if failures:
        print('\nNot ready. Failed checks: {}'.format(', '.join(failures)))
        return 1
    print('\nReady.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
