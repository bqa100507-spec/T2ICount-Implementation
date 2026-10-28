#!/usr/bin/env python
"""Read-only FSC147-S spatial diagnostic; no training or evaluator changes.

Native intermediate maps remain patch-local. Only density is reassembled,
by utils.inference.predict_density (the production evaluator).
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from tools.evaluate_rich_prompts import load_evaluation_samples
from utils.inference import load_image_tensor, predict_density, prepare_image_patches
from utils.paths import AssetPaths, require_directory, require_file

CHECKPOINTS = {
    'BASE-002': 'checkpoints/baseline_retrain/pilot_1000x10/baseline_1000x10/best_model_7.pth',
    'RP-003': 'checkpoints/rich_prompt/pilot_1000x10_lambda0/rich_prompt_1000x10_lambda0/best_model_7.pth',
    'RP-009': 'checkpoints/rich_prompt/rp009/template_diversity_1000x10_v1/best_model_7.pth',
}
MAPS = ('fused_attention', 'sim_x2', 'sim_x1', 'density_native',
        'attention_12', 'attention_24', 'attention_48')
CAPTURE_IMAGE = '3319.jpg'
DOT_REASON = ('unavailable: no verified FSC147-S target dot annotations; '
              'original FSC147 dots can refer to a different target')
SELECTION_RULE = (
    'Sort image IDs lexicographically. Bin GT counts: [0,1], [2,5], [6,20], [21,+inf). '
    'Use random.Random(3407) to shuffle each bin once in bin order. Round-robin over '
    'nonempty bins; within each bin prefer the remaining image whose class has been '
    'selected least often globally, breaking ties by shuffled order. No predictions used.'
)
DESIGN = {
    'question': 'Does RP-003 differ spatially from BASE-002 under class inference, and does RP-009 share those differences?',
    'hypothesis': 'RP-003 may have different attention/HSCM/density response patterns; RP-009 may share some patterns. These are hypotheses, not causal claims.',
    'changed_variables': ['checkpoint identity (primary)', 'class versus saved detailed prompt (secondary, paired)', 'Approved diagnostic coverage extension from 20 to all 230 FSC147-S images; original 20 retained for visualization'],
    'controls': 'Same saved prompts, seed 3407, RGB [-1,1], 384 patch/stride, original model/tokenizer/evaluator, float32, no augmentation or weights updates.',
    'metrics': 'GT/predicted count, absolute error, FSC147-S MAE/RMSE; normalized within-map L1 and Pearson correlation describe representation changes only. Dot metrics unavailable.',
    'decision_rule': 'Report paired observations only. Researcher review is required before choosing a future transfer location; no causality, localization-accuracy claim, statistical equivalence, or automatic scientific conclusion.',
}


def write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def fingerprint(path, emit=None):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    size, read, last_report = path.stat().st_size, 0, time.monotonic()
    if emit and size > 64 * 1024**2:
        emit('Hashing read-only asset: {} ({:.1f} MiB)'.format(path, size / 1024**2))
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            read += len(chunk)
            if emit and time.monotonic() - last_report >= 15:
                emit('Hashing {}: {:.1f}/{:.1f} MiB'.format(path.name, read / 1024**2, size / 1024**2))
                last_report = time.monotonic()
    return {'path': str(path), 'size_bytes': size, 'sha256': digest.hexdigest()}


def checkpoint_paths(asset_root, models):
    if not models or len(set(models)) != len(models) or set(models) - set(CHECKPOINTS):
        raise ValueError('Choose unique known model IDs: ' + ', '.join(CHECKPOINTS))
    paths = {name: require_file(Path(asset_root) / CHECKPOINTS[name], name + ' weights-only checkpoint')
             for name in models}
    for name, path in paths.items():
        if path.stat().st_size == 0:
            raise ValueError('Empty checkpoint: ' + name)
    return paths


def discover_bank(asset_root, explicit=None):
    if explicit:
        return require_file(explicit, 'FSC147-S prompt bank')
    name = 'fsc147s_gemma4_26b_v3.json'
    candidates = [p for p in (Path(asset_root) / 'prompts' / name, ROOT / 'prompts' / name) if p.is_file()]
    if not candidates:
        raise FileNotFoundError('Saved 230-image prompt bank missing; configure --prompt-bank. Expected ' + name)
    if len({fingerprint(p)['sha256'] for p in candidates}) != 1:
        raise ValueError('Drive/repository prompt banks differ; inspect and pass --prompt-bank explicitly.')
    return candidates[0].resolve()


def select_samples(samples, number=20, seed=3407):
    if not 1 <= number <= len(samples):
        raise ValueError('num-images must be between 1 and dataset size')
    bins = [[] for _ in range(4)]
    for sample in sorted(samples, key=lambda s: s.image_name):
        count = sample.gt_count
        if count < 0:
            raise ValueError('Negative GT count')
        bins[0 if count <= 1 else 1 if count <= 5 else 2 if count <= 20 else 3].append(sample)
    rng = random.Random(seed)
    for group in bins:
        rng.shuffle(group)
    chosen, class_counts = [], {}
    while len(chosen) < number:
        for group in bins:
            if not group or len(chosen) == number:
                continue
            index = min(range(len(group)), key=lambda i: class_counts.get(group[i].class_name, 0))
            sample = group.pop(index)
            chosen.append(sample)
            class_counts[sample.class_name] = class_counts.get(sample.class_name, 0) + 1
    return chosen


def patch_positions(height, width, patch_size=384, stride=384):
    # Exact row-major coordinates from utils.tools.extract_patches, including
    # edge-shifted overlapping patches. Small images are rejected, not padded.
    if min(height, width) < patch_size or stride > patch_size or stride < 1:
        raise ValueError('Unsupported patch geometry; no resizing/padding fallback')
    nh = (height - patch_size + stride - 1) // stride + 1
    nw = (width - patch_size + stride - 1) // stride + 1
    return [[min(j * stride, width - patch_size), min(i * stride, height - patch_size),
             patch_size, patch_size] for i in range(nh) for j in range(nw)]


def prompt_tokens(tokenizer, prompt, mode):
    # Reuse the validated training mask for rich text. For class prompts also
    # assert equality to the original evaluator's mask (no silent mask changes).
    from datasets.dataset import build_rich_prompt_attention_mask
    from utils.inference import build_prompt_attention_mask
    mask = build_rich_prompt_attention_mask(tokenizer, prompt)
    if mode == 'class' and not torch.equal(mask, build_prompt_attention_mask(tokenizer, prompt)):
        raise ValueError('Class mask differs from standard evaluation')
    ids = tokenizer(prompt, truncation=True, max_length=77, padding='max_length',
                    return_tensors='pt')['input_ids'][0].tolist()
    indices = mask.nonzero().flatten().tolist()
    raw_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids'][0].tolist()
    if (len(ids) != 77 or ids[0] != tokenizer.bos_token_id
            or ids[len(indices) + 1] != tokenizer.eos_token_id
            or [ids[i] for i in indices] != raw_ids[:75]
            or mask[0] != 0 or mask[76] != 0):
        raise ValueError('CLIP BOS/EOS/padding/content alignment failed')
    return mask, {'prompt': prompt, 'input_ids': ids,
                  'tokens': tokenizer.convert_ids_to_tokens(ids),
                  'active_token_indices': indices, 'mask': mask.tolist(),
                  'raw_token_count': len(raw_ids), 'truncated': len(raw_ids) > 75}


class SpatialCapture:
    """Non-mutating hooks on existing outputs; do not patch attention math.

    UNetWrapper already averages attention heads and selected down/up layers.
    Capture its native token maps before Count.forward resizes/averages them.
    Individual heads/layers are not claimed to be retained.
    """
    def __init__(self, model, token_indices):
        self.model, self.indices = model, token_indices
        self.chunks, self.pending, self.handles = [], None, []

    def _unet(self, module, inputs, output):
        if self.pending is not None:
            raise RuntimeError('Unmatched U-Net/Count forwards')
        self.pending = {}
        for size, value in zip((12, 24, 48), output[1:]):
            if value is None or tuple(value.shape[1:]) != (77, size, size):
                raise ValueError('Unexpected native cross-attention shape at ' + str(size))
            self.pending['tokens_' + str(size)] = value[:, self.indices].detach().float().cpu().clone()

    def _count(self, module, inputs, output):
        if self.pending is None or len(output) != 4:
            raise RuntimeError('Expected production Count output')
        for name, value in zip(('density_native', 'sim_x2', 'sim_x1', 'fused_attention'), output):
            if tuple(value.shape[1:]) != (1, 48, 48):
                raise ValueError('Unexpected Count map shape: ' + name)
            self.pending[name] = value.detach().float().cpu().clone()
        self.chunks.append(self.pending)
        self.pending = None

    def __enter__(self):
        self.handles = [self.model.unet.register_forward_hook(self._unet),
                        self.model.register_forward_hook(self._count)]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()

    def arrays(self):
        if not self.chunks or self.pending is not None:
            raise RuntimeError('Incomplete capture')
        arrays = {key: torch.cat([chunk[key] for chunk in self.chunks]).numpy()
                  for key in self.chunks[0]}
        for size in (12, 24, 48):
            arrays['attention_' + str(size)] = arrays['tokens_' + str(size)].mean(1, keepdims=True)
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Nonfinite intermediate map')
        return arrays


def capture_prediction(model, inputs, prompt, mask, batch_size=1):
    prepared = prepare_image_patches(inputs, 384, 384)
    positions = patch_positions(inputs.shape[-2], inputs.shape[-1])
    if len(positions) != len(prepared[0]):
        raise ValueError('Patch coordinate/count mismatch')
    with SpatialCapture(model, mask.nonzero().flatten().tolist()) as capture:
        # Density/count come from exactly the production path, including overlap
        # averaging, x8 bilinear /64 in reassembly, and final /60.
        density = predict_density(model, inputs, prompt, mask, batch_size=batch_size,
                                  patch_size=384, stride=384, prepared_patches=prepared)
    arrays = capture.arrays()
    if any(len(v) != len(positions) for v in arrays.values()):
        raise ValueError('Capture lost patch ordering')
    arrays['density_full'] = density.detach().float().cpu().numpy()
    arrays['patch_xywh'] = np.asarray(positions, dtype=np.int32)
    if not np.isfinite(arrays['density_full']).all():
        raise ValueError('Nonfinite density')
    return arrays, float(density.sum().item())


def normalize_map(value):
    value = np.asarray(value, dtype=np.float64)
    low, high = value.min(), value.max()
    return (value - low) / (high - low + 1e-7)


def map_difference(first, second):
    """Per-patch minmax, same map type/coordinates only; not accuracy."""
    if first.shape != second.shape:
        raise ValueError('Cannot compare different map shapes')
    a = np.concatenate([normalize_map(v).ravel() for v in first])
    b = np.concatenate([normalize_map(v).ravel() for v in second])
    corr = None if a.std() < 1e-12 or b.std() < 1e-12 else float(np.corrcoef(a, b)[0, 1])
    return {'normalized_l1': float(np.abs(a - b).mean()), 'pearson': corr}


def artifact_stem(image, model, mode):
    if Path(image).name != image or model not in CHECKPOINTS or mode not in ('class', 'detailed'):
        raise ValueError('Unsafe artifact identity')
    return '{}__{}__{}'.format(Path(image).stem, model, mode)


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    for name in ('figures', 'maps'):
        (path / name).mkdir()
    return path


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preflight(args, tokenizer=None):
    assets = AssetPaths.from_sources(args.asset_root)
    checkpoints = checkpoint_paths(args.checkpoint_root or assets.root, args.models)
    if not args.prompts or len(set(args.prompts)) != len(args.prompts) or set(args.prompts) - {'class', 'detailed'}:
        raise ValueError('Choose unique class/detailed prompt modes')
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    if Path(args.output_dir).exists():
        raise FileExistsError('Output exists; choose a new OUTPUT_DIR: ' + str(args.output_dir))
    bank = discover_bank(args.checkpoint_root or assets.root, args.prompt_bank)
    metadata = require_file(args.metadata, '230-image FSC147-S metadata')
    config = require_file(args.config, 'SD config')
    require_file(assets.sd_checkpoint, 'SD checkpoint')
    require_directory(assets.clip_dir, 'CLIP directory')
    samples, provenance = load_evaluation_samples(metadata, bank, assets.dataset_dir('fsc147') / 'images_384_VarV2')
    bank_doc = json.loads(bank.read_text(encoding='utf-8'))
    if (len(samples) != 230 or set(bank_doc['prompts']) != {s.image_name for s in samples}
            or bank_doc['metadata'].get('benchmark') != 'FSC-147-S'):
        raise ValueError('Require exact 230-image FSC147-S bank/metadata coverage')
    selected = select_samples(samples, args.num_images)
    snapshot = json.loads((ROOT / 'docs/SPATIAL_DIAGNOSTIC_SELECTION_V1.json').read_text(encoding='utf-8'))
    if (fingerprint(metadata)['sha256'] != snapshot['metadata_sha256']
            or fingerprint(bank)['sha256'] != snapshot['bank_sha256']):
        raise ValueError('Metadata/bank differs from original reviewed selection')
    original_ids = [r['image'] for r in snapshot['images']]
    if [s.image_name for s in select_samples(samples, 20)] != original_ids:
        raise ValueError('Original 20-image selection changed')
    if args.capture_image not in {s.image_name for s in selected}:
        raise ValueError('Designated preliminary capture image absent from evaluated images')
    if args.visualize_first_n != 20:
        raise ValueError('Retain exactly the original 20-image visualization selection')
    cache_parent(args)  # Fail before model/output creation for remote cache paths.
    if tokenizer is None:
        from utils.clip import load_clip_tokenizer
        tokenizer = load_clip_tokenizer(assets.clip_dir)
    records = []
    for sample in selected:
        with Image.open(sample.image_path) as image:
            width, height = image.size
        records.append({'image': sample.image_name, 'class': sample.class_name,
                        'gt_count': sample.gt_count, 'image_file': fingerprint(sample.image_path),
                        'image_hw': [height, width], 'patch_xywh': patch_positions(height, width),
                        'prompts': {mode: prompt_tokens(tokenizer, sample.class_prompt if mode == 'class'
                                                       else sample.detailed_prompt, mode)[1]
                                    for mode in args.prompts}})
    return assets, checkpoints, selected, records, bank, metadata, config, provenance


def source_provenance():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    files = (
        'tools/spatial_diagnostic.py', 'models/reg_model.py', 'models/diff_unet.py',
        'models/decoder.py', 'models/build.py', 'utils/inference.py', 'utils/tools.py',
        'utils/checkpoints.py', 'datasets/dataset.py', 'tools/evaluate_rich_prompts.py',
        'ldm/modules/encoders/modules.py', 'utils/clip.py', 'utils/paths.py',
        'tools/spatial_diagnostic_checks.py', 'docs/SPATIAL_DIAGNOSTIC_SELECTION_V1.json')
    return {'commit': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
            'status': git('status', '--porcelain'),
            'tracked_inference_diff': git('diff', 'HEAD', '--', *files),
            'files': [fingerprint(ROOT / name) for name in files]}


def environment():
    packages = {}
    for name in ('torch', 'torchvision', 'transformers', 'numpy', 'Pillow', 'matplotlib',
                 'omegaconf', 'einops', 'pytorch-lightning', 'taming-transformers'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'platform': sys.platform, 'packages': packages,
            'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cudnn': torch.backends.cudnn.version(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}


def paired_maps(output, records, models, modes, temporary_maps=None, emit=None):
    rows = []
    conditions = [(model, mode) for model in models for mode in modes]
    for record in records:
        image = record['image']
        if emit:
            emit('Comparing maps: ' + image)
        def map_path(condition):
            name = artifact_stem(image, *condition) + '.npz'
            permanent = output / 'maps' / name
            return permanent if permanent.is_file() else Path(temporary_maps) / name
        for left_index, left in enumerate(conditions):
            for right in conditions[left_index + 1:]:
                # Only same-prompt checkpoint pairs or within-model prompt pairs.
                if left[0] != right[0] and left[1] != right[1]:
                    continue
                with np.load(map_path(left)) as a, np.load(map_path(right)) as b:
                    for kind in MAPS:
                        rows.append({'image': image, 'left_model': left[0], 'left_prompt': left[1],
                                     'right_model': right[0], 'right_prompt': right[1], 'map_type': kind,
                                     **map_difference(a[kind], b[kind])})
        # Only delete files created in this run's private local cache.
        if temporary_maps is not None:
            for condition in conditions:
                path = Path(temporary_maps) / (artifact_stem(image, *condition) + '.npz')
                if path.is_file():
                    path.unlink()
    write_csv(output / 'map_comparisons.csv', rows)
    aggregates = []
    keys = ('left_model', 'left_prompt', 'right_model', 'right_prompt', 'map_type')
    for key in sorted({tuple(row[k] for k in keys) for row in rows}):
        group = [row for row in rows if tuple(row[k] for k in keys) == key]
        correlations = [r['pearson'] for r in group if r['pearson'] is not None]
        aggregates.append({**dict(zip(keys, key)), 'n': len(group),
                           'mean_normalized_l1': float(np.mean([r['normalized_l1'] for r in group])),
                           'mean_pearson': float(np.mean(correlations)) if correlations else None,
                           'pearson_valid_n': len(correlations)})
    return aggregates


def cache_parent(args):
    parent = Path(args.temp_dir or ('/content' if Path('/content').is_dir() else tempfile.gettempdir())).resolve()
    require_directory(parent, 'local temporary cache parent')
    excluded = [Path('/content/drive').resolve(), Path(args.output_dir).resolve()]
    if args.checkpoint_root:
        excluded.append(Path(args.checkpoint_root).resolve())
    for remote in excluded:
        if parent == remote or remote in parent.parents:
            raise ValueError('Temporary map cache must use local storage outside Drive/output/checkpoint root')
    return parent


def summarize(rows):
    metrics = []
    for model, mode in dict.fromkeys((row['model'], row['prompt_type']) for row in rows):
        group = [r for r in rows if (r['model'], r['prompt_type']) == (model, mode)]
        errors = [r['abs_error'] for r in group]
        metrics.append({'model': model, 'prompt_type': mode, 'n': len(group),
                        'mae': float(np.mean(errors)), 'rmse': math.sqrt(float(np.mean(np.square(errors))))})
    return metrics


def plot_image(output, record, rows, models, modes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conditions = [(m, p) for m in models for p in modes]
    arrays = {}
    for condition in conditions:
        with np.load(output / 'maps' / (artifact_stem(record['image'], *condition) + '.npz')) as saved:
            arrays[condition] = {k: saved[k] for k in saved.files}
    with Image.open(record['image_file']['path']) as im:
        rgb = np.asarray(im.convert('RGB'))
    image_rows = {(r['model'], r['prompt_type']): r for r in rows if r['image'] == record['image']}
    def label(condition):
        row = image_rows[condition]
        return '{} / {}\nimage GT={:g}\nimage pred={:.2f} |err|={:.2f}'.format(*condition, row['gt_count'], row['pred_count'], row['abs_error'])
    # True evaluator-reassembled density; common physical density scale across
    # all conditions of this image. No reconstructed intermediate feature maps.
    vmax = max(float(a['density_full'].max()) for a in arrays.values()) or 1.0
    fig, axes = plt.subplots(len(conditions), 2, figsize=(9, 2.5 * len(conditions)), squeeze=False)
    for i, condition in enumerate(conditions):
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(label(condition), fontsize=9)
        axes[i, 1].imshow(arrays[condition]['density_full'][0, 0], vmin=0, vmax=vmax, cmap='magma')
        axes[i, 1].set_title('Full-image density (count / pixel)\nshared scale [0, {:.4g}]'.format(vmax), fontsize=9)
    for ax in axes.flat:
        ax.axis('off')
    fig.suptitle(record['image'] + ' | target: ' + record['class'] + '\nNo target dots; localization accuracy unavailable')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / 'figures' / (Path(record['image']).stem + '__full_density.png'), dpi=110)
    plt.close(fig)
    # Each patch occupies the exact original coordinates shown in the title.
    # All display maps independently minmax normalized; raw numbers stay in NPZ.
    kinds = ('fused_attention', 'sim_x2', 'sim_x1', 'density_native')
    for patch_index, (x, y, w, h) in enumerate(record['patch_xywh']):
        fig, axes = plt.subplots(len(conditions), 5, figsize=(15, 2.3 * len(conditions)), squeeze=False)
        for i, condition in enumerate(conditions):
            axes[i, 0].imshow(rgb[y:y+h, x:x+w])
            axes[i, 0].set_title(label(condition), fontsize=9)
            for j, kind in enumerate(kinds, 1):
                axes[i, j].imshow(normalize_map(arrays[condition][kind][patch_index, 0]), vmin=0, vmax=1, cmap='viridis')
                axes[i, j].set_title(kind + ' | native 48x48', fontsize=9)
        for ax in axes.flat:
            ax.axis('off')
        fig.suptitle('{} | target={} | patch {} xywh={} | each display map normalized [0,1]\nDetailed token means differ in semantics; heatmaps are not localization accuracy'.format(record['image'], record['class'], patch_index, (x, y, w, h)))
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(output / 'figures' / (Path(record['image']).stem + '__patch{:02d}.png'.format(patch_index)), dpi=110)
        plt.close(fig)
        fig, axes = plt.subplots(len(conditions), 3, figsize=(9, 2.3 * len(conditions)), squeeze=False)
        for i, condition in enumerate(conditions):
            for j, size in enumerate((12, 24, 48)):
                axes[i, j].imshow(normalize_map(arrays[condition]['attention_' + str(size)][patch_index, 0]),
                                  vmin=0, vmax=1, cmap='viridis')
                axes[i, j].set_title('{} / {} | {}x{} token mean'.format(*condition, size, size), fontsize=9)
        for ax in axes.flat:
            ax.axis('off')
        fig.suptitle('{} | target={} | patch {} xywh={}\nNative resolutions retained separately; per-map display normalization'.format(record['image'], record['class'], patch_index, (x, y, w, h)))
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(output / 'figures' / (Path(record['image']).stem + '__attention_patch{:02d}.png'.format(patch_index)), dpi=110)
        plt.close(fig)


def plot_token(output_dir, image, model, mode, token_position, patch_index=0):
    """Notebook helper: original CLIP sequence position, not word index."""
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    record = next(r for r in manifest['selected_images'] if r['image'] == image)
    tokens = record['prompts'][mode]
    offset = tokens['active_token_indices'].index(token_position)  # rejects BOS/EOS/padding
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    with np.load(output / 'maps' / (artifact_stem(image, model, mode) + '.npz')) as data:
        for ax, size in zip(axes, (12, 24, 48)):
            ax.imshow(normalize_map(data['tokens_' + str(size)][patch_index, offset]), vmin=0, vmax=1)
            ax.set_title('{}x{}'.format(size, size))
    fig.suptitle('{} / {} / {} | patch {} xywh={}\nCLIP position {} token {!r} (a BPE token, not necessarily a whole word)'.format(
        image, model, mode, patch_index, record['patch_xywh'][patch_index], token_position, tokens['tokens'][token_position]))
    fig.tight_layout()
    return fig


def run(args):
    assets, checkpoints, selected, records, bank, metadata, config, provenance = preflight(args)
    print('Validated FSC147-S: 230 images; selected:', [s.image_name for s in selected], flush=True)
    print('Prompt bank:', bank, '|', provenance, flush=True)
    print('Dot metrics:', DOT_REASON, flush=True)
    if not args.run:
        print('Validation only. No model loaded or outputs created. Use --run for inference.', flush=True)
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use the Colab GPU runtime')
    source = source_provenance()
    if source['branch'] != 'rich-prompt-alignment':
        raise ValueError('Require rich-prompt-alignment branch')
    output = new_output(args.output_dir)
    log = logging.getLogger('spatial_diagnostic')
    log.setLevel(logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(output / 'run.log', encoding='utf-8')]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        log.addHandler(handler)
    model = None
    temporary = None
    original_ids = [r['image'] for r in json.loads(
        (ROOT / 'docs/SPATIAL_DIAGNOSTIC_SELECTION_V1.json').read_text(encoding='utf-8'))['images']]
    visualization_records = [r for r in records if r['image'] in original_ids]
    visualization_ids = [r['image'] for r in visualization_records]
    try:
        temporary = tempfile.TemporaryDirectory(prefix='t2icount_spatial_', dir=str(cache_parent(args)))
        temporary_maps = Path(temporary.name)
        log.info('Recording exact asset hashes; read-only hashing may take time on Drive')
        manifest = {'schema_version': 'fsc147s-spatial-v2', 'status': 'running',
                    'created_at': datetime.now(timezone.utc).isoformat(), 'design': DESIGN,
                    'config': vars(args), 'source': source, 'environment': environment(),
                    'selection': {'seed': 3407, 'rule': SELECTION_RULE, 'candidate_count': 230},
                    'selected_images': records, 'prompt_bank': fingerprint(bank),
                    'visualization': {'count': len(visualization_ids), 'selected_images': visualization_ids,
                                      'rule': 'Intersection with original immutable 20-image selection'},
                    'capture_equivalence_image': args.capture_image,
                    'storage': {'ephemeral_cache': {'path': str(temporary_maps), 'status': 'active'},
                                'nonvisualized_maps': 'Only comparison map types cached locally; no permanent tokens/full density'},
                    'prompt_bank_metadata': json.loads(bank.read_text(encoding='utf-8'))['metadata'],
                    'metadata': fingerprint(metadata), 'sd_config': fingerprint(config),
                    'sd_checkpoint': fingerprint(assets.sd_checkpoint, log.info),
                    'clip_files': [fingerprint(p, log.info) for p in sorted(assets.clip_dir.iterdir()) if p.is_file()],
                    'checkpoints': {m: fingerprint(p, log.info) for m, p in checkpoints.items()},
                    'inference': {'patch_size': 384, 'stride': 384, 'precision': 'float32',
                                  'density_scaling': 'production x8 bilinear /64, overlap mean, /60',
                                  'attention_selector': 'down_cross+up_cross',
                                  'fused_attention': 'Count.forward output: 0.1 norm(attn48) + 0.3 norm(up2 attn24) + 0.6 norm(up4 attn12); content-token mean',
                                  'detailed_mask': 'validated rich mask; BOS/EOS/padding excluded; truncated to 75 content tokens'},
                    'dot_metrics': {'status': 'unavailable', 'reason': DOT_REASON},
                    'numerical_storage': 'float32 raw native patch maps; active token channels only; NPZ is lossless compression',
                    'map_comparison': 'Per-patch minmax eps=1e-7; concatenate row-major cells, mean absolute difference and Pearson. Same map type only. Descriptive, not localization accuracy.',
                    'capture_equivalence': []}
        write_json(output / 'manifest.json', manifest)
        random.seed(3407)
        np.random.seed(3407)
        torch.manual_seed(3407)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(3407)
        rows = []
        from models.build import build_t2icount
        for name, checkpoint in checkpoints.items():
            log.info('Loading %s sequentially: %s', name, checkpoint)
            model = build_t2icount(config, assets.sd_checkpoint, assets.clip_dir,
                                  checkpoint_path=checkpoint, device=args.device, mode='eval')
            model.eval()
            for index, sample in enumerate(selected):
                inputs = load_image_tensor(sample.image_path, args.device)
                for mode in args.prompts:
                    prompt = sample.class_prompt if mode == 'class' else sample.detailed_prompt
                    mask, token_record = prompt_tokens(model.clip.tokenizer, prompt, mode)
                    if token_record != records[index]['prompts'][mode]:
                        raise ValueError('Model tokenizer differs from preflight tokenizer')
                    arrays, count = capture_prediction(model, inputs, prompt, mask, args.batch_size)
                    # One image/prompt per model: fail closed on any changed output.
                    # This is an on-runtime check, not a claim of local GPU testing.
                    if sample.image_name == args.capture_image:
                        normal = predict_density(model, inputs, prompt, mask, batch_size=args.batch_size)
                        equal = np.array_equal(normal.detach().float().cpu().numpy(), arrays['density_full'])
                        manifest['capture_equivalence'].append({'model': name, 'image': sample.image_name,
                                                                 'prompt': mode, 'bitwise_equal': equal})
                        write_json(output / 'manifest.json', manifest)
                        del normal
                        if not equal:
                            raise RuntimeError('Capture/no-capture density differs; stop for inspection')
                    archive_name = artifact_stem(sample.image_name, name, mode) + '.npz'
                    if sample.image_name in visualization_ids:
                        np.savez_compressed(output / 'maps' / archive_name, **arrays)
                    else:
                        # Retain only the seven arrays required for paired metrics.
                        # Active tokens/full-image density are discarded after counting.
                        np.savez_compressed(temporary_maps / archive_name,
                                            **{kind: arrays[kind] for kind in MAPS})
                    rows.append({'image': sample.image_name, 'class': sample.class_name, 'model': name,
                                 'prompt_type': mode, 'prompt': prompt, 'gt_count': sample.gt_count,
                                 'pred_count': count, 'abs_error': abs(count - sample.gt_count),
                                 'patch_count': len(arrays['patch_xywh']),
                                 'dot_metrics_status': 'unavailable', 'dot_response': None,
                                 'dot_background_contrast': None, 'dot_metrics_reason': DOT_REASON})
                    write_csv(output / 'results.csv', rows)
                    log.info('[%s %d/%d %s] %s GT=%g pred=%.4f abs_error=%.4f patches=%d',
                             name, index+1, len(selected), mode, sample.image_name,
                             sample.gt_count, count, abs(count-sample.gt_count), len(arrays['patch_xywh']))
                    del arrays
                del inputs
            del model
            model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info('Released %s; next checkpoint can load', name)
        comparisons = paired_maps(output, records, args.models, args.prompts, temporary_maps, log.info)
        temporary.cleanup()
        manifest['storage']['ephemeral_cache']['status'] = 'removed'
        summary = {'schema_version': 'fsc147s-spatial-v2', 'status': 'complete',
                   'counting': summarize(rows), 'map_comparisons': comparisons,
                   'dot_metrics': {'status': 'unavailable', 'reason': DOT_REASON},
                   'evaluation_images': len(records),
                   'interpretation': ('All 230 FSC147-S images. ' if len(records) == 230 else 'Configured FSC147-S subset. ') + 'No localization accuracy or causal interpretation. Class inference is primary, detailed inference diagnostic only. Not standard FSC147 test results.'}
        for record in visualization_records:
            log.info('Rendering %s', record['image'])
            plot_image(output, record, rows, args.models, args.prompts)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ('mae', 'rmse')):
            ax.bar([r['model'] + '\n' + r['prompt_type'] for r in summary['counting']],
                   [r[metric] for r in summary['counting']])
            ax.set_title('FSC147-S n={} '.format(len(records)) + metric.upper())
            ax.tick_params(axis='x', labelsize=7)
        fig.tight_layout()
        fig.savefig(output / 'figures' / 'summary.png', dpi=120)
        plt.close(fig)
        write_json(output / 'summary.json', summary)
        write_json(output / 'manifest.json', manifest)
        from tools.spatial_diagnostic_checks import saved_run_state
        ok, reason = saved_run_state(output, len(records), args.models, args.prompts,
                                     [r['image'] for r in records], original_ids,
                                     args.capture_image, require_complete=False)
        if not ok:
            raise RuntimeError('Completion validation failed: ' + reason)
        manifest['status'] = 'complete'
        write_json(output / 'manifest.json', manifest)
        log.info('Complete: %s', output)
    except BaseException:
        log.error('FAILED\n%s', traceback.format_exc())
        if 'manifest' in locals():
            manifest['status'] = 'failed'
            write_json(output / 'manifest.json', manifest)
        if 'summary' in locals():
            summary['status'] = 'failed'
            write_json(output / 'summary.json', summary)
        raise
    finally:
        if temporary is not None:
            temporary.cleanup()
            if 'manifest' in locals():
                manifest['storage']['ephemeral_cache']['status'] = 'removed'
                write_json(output / 'manifest.json', manifest)
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for handler in handlers:
            log.removeHandler(handler)
            handler.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', required=True)
    parser.add_argument('--checkpoint-root', help='Optional Drive root for checkpoints/bank when runtime assets are local')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--prompt-bank')
    parser.add_argument('--metadata', default=str(ROOT / 'FSC-147-S.json'))
    parser.add_argument('--config', default=str(ROOT / 'configs/v1-inference.yaml'))
    parser.add_argument('--models', nargs='+', default=list(CHECKPOINTS), choices=list(CHECKPOINTS))
    parser.add_argument('--prompts', nargs='+', default=['class', 'detailed'], choices=['class', 'detailed'])
    parser.add_argument('--num-images', type=int, default=230)
    parser.add_argument('--visualize-first-n', type=int, default=20,
                        help='Fixed original 20 only; fewer are retained for a preliminary subset')
    parser.add_argument('--ephemeral-map-cache', action='store_true', default=True,
                        help='Compatibility flag: nonvisualized maps always use a temporary local cache')
    parser.add_argument('--temp-dir', help='Local cache parent (default /content on Colab, else OS temp)')
    parser.add_argument('--capture-image', default=CAPTURE_IMAGE,
                        help='Designated preliminary image for capture/no-capture equivalence only')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--run', action='store_true', help='Explicitly enable inference; default validates only')
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
