"""RP-010 controls; no model, loss or evaluator replacement."""
import hashlib
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from utils.rich_prompt_training import (
    build_class_templates, load_fsc147_train_metadata, load_rich_prompt_bank,
    ordered_image_fingerprint,
)
from utils.train_subset import select_train_subset_indices

SCHEMA = 'rp010-paired-v1'
MODES = {'RP-010A': 'template-diversity', 'RP-010B': 'rich'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return 'sha256:' + digest.hexdigest()


def digest_json(value):
    return 'sha256:' + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_object)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_config(path, asset_root, seed=None):
    config = read_json(path)
    if seed is not None:
        config['seed'] = seed
    if config.get('schema') != SCHEMA or MODES.get(config.get('experiment_id')) != config.get('supervision_mode'):
        raise ValueError('Invalid RP-010 arm/schema')
    required = {
        'train_samples': 1000, 'train_subset_seed': 3407, 'epochs': 10,
        'batch_size': 1, 'num_workers': 0, 'crop_size': 384, 'stride': 384,
        'concat_size': 224, 'downsample_ratio': 8, 'lr': 5e-5,
        'weight_decay': 5e-4, 'consistency_weight': 0.0,
        'start_val': 7, 'val_epoch': 1, 'checkpoint_metric': 'val_mae_plus_rmse',
        'scheduler': None, 'gradient_accumulation': 1, 'amp': False,
        'schedule': 'epoch-per-anchor-v1', 'donor_pool': 'official_full_train',
        'optimizer': 'AdamW', 'betas': [0.9, 0.999], 'eps': 1e-8,
        'unet_lr_factor': 0.1, 'unet_weight_decay_factor': 0.1,
        'cudnn_benchmark': False, 'cudnn_deterministic': True,
        'resume_policy': 'same-arm-completed-epoch',
    }
    for key, expected in required.items():
        if key not in config or config[key] != expected:
            raise ValueError('Protocol mismatch: ' + key)
    if not isinstance(config['seed'], int) or not 0 <= config['seed'] < 2**31:
        raise ValueError('Seed must be an explicit nonnegative integer')
    for key in ('dataset', 'prompt_bank', 'sd_checkpoint', 'clip_dir',
                'initialization_checkpoint', 'output_dir'):
        config[key] = str(Path(config[key].format(
            ASSET_ROOT=str(Path(asset_root).resolve()), seed=config['seed'],
        )).resolve())
    return config


def shared_config(config):
    return {key: value for key, value in config.items()
            if key not in ('experiment_id', 'supervision_mode', 'output_dir')}


def validate_pair(a, b):
    if a['experiment_id'] != 'RP-010A' or b['experiment_id'] != 'RP-010B':
        raise ValueError('Expected A then B')
    if shared_config(a) != shared_config(b):
        raise ValueError('Paired controls differ')
    if Path(a['output_dir']) == Path(b['output_dir']):
        raise ValueError('Arms must use separate output directories')


def data_contract(config, tokenizer):
    root = Path(config['dataset'])
    train, classes = load_fsc147_train_metadata(root)
    splits_path = root / 'FSC_147/Train_Test_Val_FSC_147.json'
    splits = read_json(splits_path)
    for name in ('train', 'val', 'test'):
        if len(set(splits[name])) != len(splits[name]):
            raise ValueError('Duplicate split image: ' + name)
    if any(set(splits[a]) & set(splits[b]) for a, b in (
            ('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Official splits overlap')
    indices = select_train_subset_indices(len(train), config['train_samples'], config['train_subset_seed'])
    names = [train[index] for index in indices]
    raw_bank = read_json(config['prompt_bank'])
    if set(raw_bank['prompts']) != set(names):
        raise ValueError('Bank must contain exactly selected training images')
    bank = load_rich_prompt_bank(
        config['prompt_bank'], config['train_samples'], config['train_subset_seed'], names, classes)
    class_lines = (root / 'FSC_147/ImageClasses_FSC147.txt').read_text(encoding='utf-8').splitlines()
    class_ids = [line.split('\t')[0] for line in class_lines]
    if len(class_ids) != len(set(class_ids)):
        raise ValueError('Duplicate class-to-image mapping')
    # Donor images/densities can come from any train anchor. Validate the full pool.
    payload_stats = []
    for name in train + splits['val'] + splits['test']:
        path = root / 'images_384_VarV2' / name
        if not path.is_file():
            raise FileNotFoundError('Missing RGB: ' + name)
        stat = path.stat()
        payload_stats.append([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns])
    for name in train:
        path = root / 'gt_density_map_adaptive_384_VarV2' / Path(name).with_suffix('.npy')
        if not path.is_file():
            raise FileNotFoundError('Missing train/donor density: ' + name)
        stat = path.stat()
        payload_stats.append([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns])
    token_rows = []
    for name in names:
        record = bank.records[name]
        texts = [('class', record.class_name)] + list(zip(
            ('generic_1', 'generic_2'), build_class_templates(record.class_name))) + [
                ('detailed', record.detailed), ('generalized', record.generalized)]
        for slot, text in texts:
            count = int(tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].shape[1])
            if count < 1 or (slot == 'class' and count > 75):
                raise ValueError('Invalid class/content token count: ' + name)
            token_rows.append({'image': name, 'slot': slot, 'prompt': text,
                               'raw_tokens': count, 'retained_tokens': min(count, 75),
                               'truncated': count > 75})
    return {
        'selected_images': names, 'subset_indices': indices,
        'selected_image_fingerprint': ordered_image_fingerprint(names),
        'prompt_bank_sha256': bank.file_fingerprint,
        'prompt_bank_metadata': raw_bank['metadata'],
        'split_sha256': sha256(splits_path),
        'class_sha256': sha256(root / 'FSC_147/ImageClasses_FSC147.txt'),
        'annotations_sha256': sha256(root / 'FSC_147/annotation_FSC147_384.json'),
        'payload_stat_fingerprint': digest_json(payload_stats),
        'payload_identity_method': 'path/size/mtime_ns; immutable dataset assumed, not payload byte hashes',
        'split_counts': {key: len(splits[key]) for key in ('train', 'val', 'test')},
        'donor_pool_size': len(train), 'token_rows': token_rows,
    }


def epoch_order(size, seed, epoch):
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def sample_seed(seed, epoch, index):
    return int(hashlib.sha256('{}:{}:{}'.format(seed, epoch, index).encode()).hexdigest()[:8], 16)


@contextmanager
def augmentation_rng(seed):
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        # Do not call torch.manual_seed: that also mutates CUDA model RNG.
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)


class EpochDataset(torch.utils.data.Dataset):
    """Preserve ObjectCount augmentation math, isolate its random stream."""
    def __init__(self, dataset, seed, epoch):
        self.dataset, self.seed, self.epoch = dataset, seed, epoch
        self.order = epoch_order(len(dataset), seed, epoch)
        self.trace = hashlib.sha256()
        self.compatible = 0

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        anchor = self.order[index]
        with augmentation_rng(sample_seed(self.seed, self.epoch, anchor)):
            sample = self.dataset[anchor]
        # Hash only identical controls, never the intentionally changed auxiliary text.
        self.trace.update(str(anchor).encode() + b'\n')
        for item in sample[:5]:
            if torch.is_tensor(item):
                self.trace.update(item.contiguous().numpy().tobytes())
            else:
                self.trace.update(str(item).encode())
        compatible = bool(sample[5]['rich_compatible'])
        self.trace.update(bytes([compatible]))
        self.compatible += int(compatible)
        return sample


def parameter_contract(model):
    return [{'name': name, 'shape': list(p.shape), 'dtype': str(p.dtype),
             'trainable': p.requires_grad, 'bytes': p.numel() * p.element_size()}
            for name, p in model.named_parameters()]


def parameter_memory_estimate(parameters):
    # A conservative budget, not a strict lower bound or a measured peak:
    # some trainable parameters (e.g. unused SD output head) can have no grad.
    weights = sum(p['bytes'] for p in parameters)
    trainable = sum(p['bytes'] for p in parameters if p['trainable'])
    return {'parameter_bytes': weights, 'trainable_bytes': trainable,
            'weights_gradients_adam_estimate_bytes': weights + 3 * trainable,
            'assumption': 'gradient and two Adam moments for every trainable parameter; unused tensors may allocate neither',
            'excludes': 'activations of three forwards, buffers, optimizer temporaries, CUDA workspace'}


def steps_per_epoch(config):
    return math.ceil(config['train_samples'] / config['batch_size'])


def validate_resume(checkpoint, guard, expected_steps):
    if checkpoint.get('rp010_guard') != guard:
        raise ValueError('Resume rejected: different arm/seed/init/data/config/source/runtime')
    next_epoch = checkpoint.get('next_epoch')
    if not isinstance(next_epoch, int) or not 1 <= next_epoch <= guard['config']['epochs']:
        raise ValueError('Resume requires a completed epoch boundary')
    if checkpoint.get('optimizer_steps') != next_epoch * expected_steps:
        raise ValueError('Resume step/epoch accounting mismatch')
    if len(checkpoint.get('epoch_records', [])) != next_epoch:
        raise ValueError('Missing epoch schedule records')
    for epoch, record in enumerate(checkpoint['epoch_records']):
        if record.get('epoch') != epoch or record.get('steps') != expected_steps:
            raise ValueError('Resume epoch records disagree with step accounting')


def new_output(path, resume=False):
    path = Path(path)
    if resume:
        if not (path / 'latest.tar').is_file() or not (path / 'manifest.json').is_file():
            raise ValueError('No matching completed-epoch state to resume')
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def better_validation(mae, rmse, best_mae, best_rmse):
    if not math.isfinite(mae) or not math.isfinite(rmse):
        raise ValueError('Nonfinite validation metrics')
    return mae + rmse < best_mae + best_rmse


def counting_summary(rows):
    if not rows:
        raise ValueError('No evaluation rows')
    errors = np.asarray([row['signed_error'] for row in rows], dtype=float)
    if not np.isfinite(errors).all():
        raise ValueError('Nonfinite predictions')
    return {'n': len(rows), 'mae': float(np.abs(errors).mean()),
            'rmse': float(np.sqrt(np.square(errors).mean())), 'bias': float(errors.mean()),
            'overcount': int((errors > 1e-9).sum()), 'undercount': int((errors < -1e-9).sum()),
            'ties': int((np.abs(errors) <= 1e-9).sum())}


def gt_group(count):
    return '1' if count == 1 else '2-5' if 2 <= count <= 5 else '6-20' if 6 <= count <= 20 else '>20' if count > 20 else 'other'
