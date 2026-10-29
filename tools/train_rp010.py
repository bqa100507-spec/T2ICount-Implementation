#!/usr/bin/env python
"""Isolated RP-010 lifecycle around the production trainer and predict_count."""
import argparse
import csv
import gc
import ctypes
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from utils.rp010 import (
    SCHEMA, EpochDataset, better_validation, counting_summary, data_contract,
    digest_json, gt_group, load_config, parameter_memory_estimate, new_output,
    parameter_contract, read_json, sha256, shared_config, steps_per_epoch,
    validate_pair, validate_resume, write_json,
)


def source_identity():
    names = subprocess.check_output(
        ['git', 'ls-files', '*.py', 'configs/*'], cwd=ROOT, text=True).splitlines()
    names += ['utils/rp010.py', 'tools/train_rp010.py',
              'configs/rp010a.json', 'configs/rp010b.json']
    files = {name: sha256(ROOT / name) for name in sorted(set(names)) if (ROOT / name).is_file()}
    return {'commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'files_sha256': digest_json(files), 'files': files}


def runtime_signature():
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'transformers', 'omegaconf', 'Pillow', 'opencv-python-headless'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.system(),
            'packages': packages, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot save empty results')
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_determinism(seed):
    from utils.regression_trainer import setup_seed
    setup_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build(config, device='cpu', checkpoint=None):
    from models.build import build_t2icount
    return build_t2icount(ROOT / config['sd_config'], config['sd_checkpoint'],
                         config['clip_dir'], checkpoint_path=checkpoint,
                         device=device, mode='train')


def init_manifest_path(config):
    return Path(config['initialization_checkpoint']).parent / 'initial_manifest.json'


def contract_identity(contract):
    return {key: value for key, value in contract.items() if key != 'token_rows'}


def get_contract(config):
    from utils.clip import load_clip_tokenizer
    tokenizer = load_clip_tokenizer(config['clip_dir'])
    return data_contract(config, tokenizer)


def verify_dataset_pair(a, b, contract, count=32):
    """No model: exercise actual augmentation on an identical prefix per arm."""
    from datasets.dataset import ObjectCount
    from utils.clip import load_clip_tokenizer
    from utils.rich_prompt_training import load_fsc147_train_metadata, load_rich_prompt_bank
    validate_pair(a, b)
    _, classes = load_fsc147_train_metadata(a['dataset'])
    bank = load_rich_prompt_bank(a['prompt_bank'], a['train_samples'],
                                a['train_subset_seed'], contract['selected_images'], classes)
    tokenizer = load_clip_tokenizer(a['clip_dir'])
    datasets = []
    for config in (a, b):
        base = ObjectCount(config['dataset'], 384, 8, concat_size=224,
                           tokenizer=tokenizer, rich_prompt_records=bank.records,
                           rich_supervision_mode=config['supervision_mode'])
        datasets.append(torch.utils.data.Subset(base, contract['subset_indices']))
    records = []
    for epoch in (0, 9):
        pair = [EpochDataset(dataset, a['seed'], epoch) for dataset in datasets]
        for index in range(min(count, len(pair[0]))):
            left, right = pair[0][index], pair[1][index]
            for x, y in zip(left[:5], right[:5]):
                if not (torch.equal(x, y) if torch.is_tensor(x) else x == y):
                    raise ValueError('Paired data/preprocessing changed')
            if left[5]['rich_compatible'] != right[5]['rich_compatible']:
                raise ValueError('Paired routing differs')
        if pair[0].trace.hexdigest() != pair[1].trace.hexdigest():
            raise ValueError('Paired dataset digest mismatch')
        records.append({'epoch': epoch, 'prefix_samples': min(count, len(pair[0])),
                        'input_target_routing_sha256': pair[0].trace.hexdigest(),
                        'compatible': pair[0].compatible,
                        'full_1000_order_sha256': digest_json(pair[0].order)})
    return records


def check_initialization(config, contract):
    manifest = read_json(init_manifest_path(config))
    expected = {'schema': SCHEMA, 'shared_config': shared_config(config),
                'source': source_identity(), 'runtime': runtime_signature(),
                'data': contract_identity(contract)}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError('Shared initialization mismatch: ' + key)
    if sha256(config['initialization_checkpoint']) != manifest['initialization_sha256']:
        raise ValueError('Shared initialization bytes changed')
    for asset in manifest['pretrained_assets']:
        stat = Path(asset['path']).stat()
        if stat.st_size != asset['bytes'] or stat.st_mtime_ns != asset['mtime_ns']:
            raise ValueError('Pretrained/tokenizer asset changed since preparation: ' + asset['path'])
    return manifest


def prepare(config):
    """CPU initialization only. Hash large assets once, preserve the cached manifest."""
    contract = get_contract(config)
    if init_manifest_path(config).exists():
        manifest = check_initialization(config, contract)
        print('Reusing validated shared initialization:', config['initialization_checkpoint'], flush=True)
        return manifest
    free = host_free_bytes()
    # A preparation safety threshold, not a measured model peak-memory estimate.
    if free is not None and free < 8 * 2**30:
        raise RuntimeError('Preparation pending: {:.2f} GiB host RAM free; require 8 GiB headroom before constructing the full CPU model. Use the L4 runtime; do not change precision/architecture.'.format(free / 2**30))
    directory = Path(config['initialization_checkpoint']).parent
    directory.mkdir(parents=True, exist_ok=False)
    write_csv(directory / 'prompt_tokens.csv', contract['token_rows'])
    configure_determinism(config['seed'])
    model = build(config)
    parameters = parameter_contract(model)
    for parameter in parameters:
        expected = parameter['name'].startswith(('unet.', 'decoder.'))
        if parameter['trainable'] != expected:
            raise ValueError('Unexpected trainable parameter: ' + parameter['name'])
    from utils.regression_trainer import _atomic_torch_save
    _atomic_torch_save(model.state_dict(), config['initialization_checkpoint'])
    clip = Path(config['clip_dir'])
    asset_files = [Path(config['sd_checkpoint'])] + sorted(
        p for p in clip.iterdir() if p.is_file() and p.suffix in ('.json', '.bin', '.txt', '.safetensors'))
    assets = [{'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
               'sha256': sha256(p)} for p in asset_files]
    manifest = {'schema': SCHEMA, 'shared_config': shared_config(config),
                'source': source_identity(), 'runtime': runtime_signature(),
                'data': contract_identity(contract), 'pretrained_assets': assets,
                'initialization_sha256': sha256(config['initialization_checkpoint']),
                'parameters': parameters, 'memory': parameter_memory_estimate(parameters)}
    write_json(init_manifest_path(config), manifest)
    print('Prepared one shared initialization; memory budget estimate:', manifest['memory'], flush=True)
    del model
    gc.collect()
    return manifest


def host_free_bytes():
    if sys.platform == 'win32':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ('total_phys', 'available_phys', 'total_page', 'available_page',
                 'total_virtual', 'available_virtual', 'extended')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available_phys)
    elif Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    return None


def make_args(config, output):
    from train import parse_arg
    options = {
        'content': output.name, 'save-dir': str(output.parent), 'seed': config['seed'],
        'data-dir': config['dataset'], 'config': str(ROOT / config['sd_config']),
        'sd-path': config['sd_checkpoint'], 'clip-path': config['clip_dir'],
        'rich-prompt-bank': config['prompt_bank'],
        'rich-supervision-mode': config['supervision_mode'],
        'rich-consistency-weight': config['consistency_weight'],
        'rich-loss-diagnostic-steps': 3, 'train-samples': config['train_samples'],
        'train-subset-seed': config['train_subset_seed'], 'smoke-train-samples': 0,
        'batch-size': config['batch_size'], 'num-workers': config['num_workers'],
        'crop-size': config['crop_size'], 'stride': config['stride'],
        'concat-size': config['concat_size'], 'downsample-ratio': config['downsample_ratio'],
        'epochs': config['epochs'], 'lr': config['lr'], 'weight-decay': config['weight_decay'],
        'start-epoch': 0, 'start-val': config['start_val'], 'val-epoch': config['val_epoch'],
        'checkpoint-interval': 1, 'max-num': config['epochs'], 'beta': 1e-4,
    }
    argv = [str(item) for key, value in options.items() for item in ('--' + key, value)]
    return parse_arg(argv)


# Importing this entry point remains CPU-safe; constructing this subclass requires CUDA.
from utils.regression_trainer import (  # noqa: E402
    Reg_Trainer, _atomic_torch_save, _capture_rng_state, _restore_rng_state, train_collate,
)


class ControlledTrainer(Reg_Trainer):
    def __init__(self, config, output, guard, initialization):
        self.control = config
        self.guard = guard
        self.initialization = initialization
        self.optimizer_steps = 0
        self.epoch_records = []
        self.best_epoch = None
        self.smoke = False
        super().__init__(make_args(config, output))

    def setup(self):
        super().setup()  # Original model, dataset, losses and optimizer.
        from models.build import load_t2icount_checkpoint
        load_t2icount_checkpoint(self.model, self.control['initialization_checkpoint'])
        if parameter_contract(self.model) != self.initialization['parameters']:
            raise ValueError('Trainable/frozen parameter contract changed')
        # Assert parent optimizer still agrees with the explicit controlled configuration.
        for group, factor in zip(self.optimizer.param_groups, (0.1, 1.0)):
            if (group['lr'] != self.control['lr'] * factor
                    or group['weight_decay'] != self.control['weight_decay'] * factor
                    or tuple(group['betas']) != tuple(self.control['betas'])
                    or group['eps'] != self.control['eps']):
                raise ValueError('Production optimizer differs from controlled config')
        configure_determinism(self.control['seed'])

    def _optimize_batch(self, loss):
        if not torch.isfinite(loss).all():
            raise ValueError('Nonfinite loss')
        if self.smoke:
            self.smoke_loss = float(loss.detach())
            self.optimizer.zero_grad()
            loss.backward()
            gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                raise ValueError('Missing/nonfinite smoke gradients')
            self.smoke_gradient_l2 = math.sqrt(sum(float(g.double().square().sum()) for g in gradients))
            if not self.smoke_gradient_l2 > 0:
                raise ValueError('Zero smoke gradient')
            self.optimizer.step()
        else:
            super()._optimize_batch(loss)
        self.optimizer_steps += 1

    def loader_for_epoch(self, epoch):
        dataset = EpochDataset(self.datasets['train'], self.control['seed'], epoch)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.control['batch_size'], shuffle=False,
            num_workers=0, pin_memory=True, collate_fn=train_collate,
            generator=torch.Generator().manual_seed(self.control['seed'] + epoch))
        return dataset, loader

    def train_epoch(self):
        # Dataset controls text content. Run the same production loss graph in
        # both arms, including rich's diagnostic consistency multiplied by zero.
        original = self.args.rich_supervision_mode
        try:
            self.args.rich_supervision_mode = 'rich'
            return super().train_epoch()
        finally:
            self.args.rich_supervision_mode = original

    def test_epoch(self):
        raise RuntimeError('RP-010 test is an explicit post-training operation only')

    def val_epoch(self):
        # Reuse original inference, prevent parent val_epoch's automatic test call.
        mae, rmse = self._evaluate_split('val')
        logging.info('RP010 epoch=%d validation MAE=%.6f RMSE=%.6f', self.epoch, mae, rmse)
        if better_validation(mae, rmse, self.best_mae, self.best_mse):
            self.best_mae, self.best_mse, self.best_epoch = float(mae), float(rmse), self.epoch
            path = Path(self.save_dir) / 'best_epoch{}.pth'.format(self.epoch)
            _atomic_torch_save(self.model.state_dict(), str(path))
            self.best_selection = {'epoch': self.epoch, 'val_mae': float(mae), 'val_rmse': float(rmse),
                                   'rule': self.control['checkpoint_metric'],
                                   'checkpoint': path.name, 'sha256': sha256(path)}
        return {'mae': float(mae), 'rmse': float(rmse)}

    def save_state(self, path, smoke=False):
        payload = {'rp010_guard': self.guard, 'epoch': self.epoch, 'next_epoch': self.epoch + 1,
                   'optimizer_steps': self.optimizer_steps, 'epoch_records': self.epoch_records,
                   'model_state_dict': self.model.state_dict(),
                   'optimizer_state_dict': self.optimizer.state_dict(),
                   'best_mae': self.best_mae, 'best_mse': self.best_mse,
                   'best_epoch': self.best_epoch, 'best_selection': getattr(self, 'best_selection', None),
                   'rng_state': _capture_rng_state(), 'smoke_only': smoke}
        _atomic_torch_save(payload, str(path))

    def load_state(self, path, smoke=False):
        from utils.checkpoints import load_trusted_legacy_checkpoint
        payload = load_trusted_legacy_checkpoint(str(path), 'cpu')
        if smoke:
            if payload.get('rp010_guard') != self.guard or payload.get('smoke_only') is not True:
                raise ValueError('Smoke resume provenance mismatch')
        else:
            if payload.get('smoke_only'):
                raise ValueError('Smoke checkpoint cannot resume training')
            validate_resume(payload, self.guard, steps_per_epoch(self.control))
        self.model.load_state_dict(payload['model_state_dict'], strict=True)
        self.optimizer.load_state_dict(payload['optimizer_state_dict'])
        self.start_epoch = payload['next_epoch']
        self.optimizer_steps = payload['optimizer_steps']
        self.epoch_records = payload['epoch_records']
        self.best_mae, self.best_mse = payload['best_mae'], payload['best_mse']
        self.best_epoch, self.best_selection = payload['best_epoch'], payload['best_selection']
        if not smoke and self.best_selection is not None:
            selected = Path(self.save_dir) / self.best_selection['checkpoint']
            if sha256(selected) != self.best_selection['sha256']:
                raise ValueError('Resume best-checkpoint artifact changed')
        _restore_rng_state(payload['rng_state'])
        return payload

    def train(self):
        for epoch in range(self.start_epoch, self.control['epochs']):
            self.epoch = epoch
            dataset, loader = self.loader_for_epoch(epoch)
            self.dataloaders['train'] = loader
            before = self.optimizer_steps
            # No changes to forward routing, targets, loss math or parent optimization.
            self.train_epoch()
            if self.optimizer_steps - before != steps_per_epoch(self.control):
                raise ValueError('Unexpected optimizer step count')
            record = {'epoch': epoch, 'steps': self.optimizer_steps - before,
                      'input_target_routing_sha256': dataset.trace.hexdigest(),
                      'order_sha256': digest_json(dataset.order), 'compatible': dataset.compatible,
                      'forward_calls': len(dataset) + 2 * dataset.compatible}
            if epoch >= self.control['start_val'] and epoch % self.control['val_epoch'] == 0:
                record['validation'] = self.val_epoch()
            self.epoch_records.append(record)
            self.save_state(Path(self.save_dir) / 'latest.tar')
            write_json(Path(self.save_dir) / 'epoch_records.json', self.epoch_records)
        if self.best_epoch is None or self.optimizer_steps != self.control['epochs'] * steps_per_epoch(self.control):
            raise ValueError('Incomplete training/selection')
        write_json(Path(self.save_dir) / 'selection.json', self.best_selection)
        write_json(Path(self.save_dir) / 'completed.json', {
            'guard': self.guard, 'optimizer_steps': self.optimizer_steps,
            'epochs': len(self.epoch_records), 'selection': self.best_selection})


def gpu_capacity(manifest):
    required = manifest['memory']['weights_gradients_adam_estimate_bytes']
    if not torch.cuda.is_available():
        return {'status': 'pending', 'reason': 'CUDA unavailable'}
    free, total = torch.cuda.mem_get_info()
    if free <= required:
        return {'status': 'pending', 'reason': 'GPU free memory below conservative all-trainable parameter/gradient/Adam budget; activations excluded',
                'gpu': torch.cuda.get_device_name(0), 'free_bytes': free, 'total_bytes': total,
                'estimated_budget_bytes': required}
    return None


def make_guard(config, initialization):
    return {'schema': SCHEMA, 'config': config, 'source': source_identity(),
            'runtime': runtime_signature(), 'data': initialization['data'],
            'initialization_sha256': initialization['initialization_sha256']}


def run_training(config, resume=False):
    contract = get_contract(config)
    initialization = check_initialization(config, contract)
    pending = gpu_capacity(initialization)
    if pending:
        raise RuntimeError(str(pending))
    guard = make_guard(config, initialization)
    output = Path(config['output_dir'])
    if resume:
        if read_json(output / 'manifest.json')['guard'] != guard:
            raise ValueError('Existing run provenance differs')
        if (output / 'completed.json').exists():
            raise ValueError('Run already completed; use evaluate')
    new_output(output, resume=resume)
    if not resume:
        write_json(output / 'manifest.json', {'guard': guard})
    trainer = ControlledTrainer(config, output, guard, initialization)
    trainer.setup()
    if resume:
        trainer.load_state(output / 'latest.tar')
    trainer.train()


def run_smoke(config):
    contract = get_contract(config)
    output = Path(config['output_dir']).with_name(config['experiment_id'] + '_smoke')
    new_output(output)
    if not init_manifest_path(config).is_file():
        result = {'status': 'pending', 'reason': 'Shared initialization not prepared; run prepare first',
                  'host_free_bytes': host_free_bytes(), 'runtime': runtime_signature(),
                  'data': contract_identity(contract)}
        write_json(output / 'smoke_result.json', result)
        print('Smoke pending:', result['reason'], '; report:', output / 'smoke_result.json', flush=True)
        return False
    initialization = check_initialization(config, contract)
    pending = gpu_capacity(initialization)
    if pending:
        write_json(output / 'smoke_result.json', pending)
        print(json.dumps(pending, indent=2), flush=True)
        return False
    guard = make_guard(config, initialization)
    write_json(output / 'manifest.json', {'guard': guard})
    trainer = ControlledTrainer(config, output, guard, initialization)
    trainer.setup()
    trainer.smoke = True
    dataset, _ = trainer.loader_for_epoch(0)
    for index in range(min(32, len(dataset))):
        sample = dataset[index]
        if sample[5]['rich_compatible']:
            break
    else:
        raise ValueError('No compatible smoke sample among first32; do not alter augmentation')
    batch = train_collate([sample])
    forward_prompts = []
    hook = trainer.model.register_forward_pre_hook(lambda model, inputs: forward_prompts.append(tuple(inputs[1])))
    trainer.epoch = 0
    trainer.dataloaders['train'] = [batch]
    parameter = next(trainer.model.decoder.parameters())
    before = parameter.detach().cpu().clone()
    torch.cuda.reset_peak_memory_stats()
    trainer.train_epoch()
    if len(forward_prompts) != 3 or torch.equal(before, parameter.detach().cpu()):
        raise ValueError('Smoke must execute three forwards and update decoder weights')
    hook.remove()
    loss, gradient = trainer.smoke_loss, trainer.smoke_gradient_l2
    # Validation inference uses the unchanged evaluator on one val image, not test.
    trainer.dataloaders['val'] = [next(iter(trainer.dataloaders['val']))]
    validation_before = trainer._evaluate_split('val')
    state = output / 'smoke_state.tar'
    trainer.save_state(state, smoke=True)
    # Alter one parameter and bookkeeping to prove the serialized state is restored.
    with torch.no_grad():
        parameter.add_(1)
    trainer.optimizer_steps = -1
    trainer.load_state(state, smoke=True)
    validation_after = trainer._evaluate_split('val')
    if trainer.optimizer_steps != 1 or not np.allclose(validation_before, validation_after, rtol=0, atol=1e-6):
        raise ValueError('Smoke save/load did not restore predictions/step count')
    trainer.train_epoch()
    if trainer.optimizer_steps != 2:
        raise ValueError('Resumed optimizer did not update')
    result = {'status': 'passed', 'guard': guard, 'sample': sample[5]['source_image_name'],
              'forward_prompts': forward_prompts, 'loss': loss, 'gradient_l2': gradient,
              'optimizer_steps': 2, 'validation_one_image_before': validation_before,
              'validation_one_image_after': validation_after,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'input_target_routing_sha256': dataset.trace.hexdigest()}
    write_json(output / 'smoke_result.json', result)
    # Smoke state is infrastructure only and may be several GB; retain result/log, clean its own state.
    state.unlink()
    print('Both-step smoke passed:', output, flush=True)
    return True


def evaluate(config, s_metadata, s_bank):
    from datasets.dataset import ObjectCount
    from tools.evaluate_rich_prompts import load_evaluation_samples
    from utils.inference import build_prompt_attention_mask, load_image_tensor, predict_count
    contract = get_contract(config)
    initialization = check_initialization(config, contract)
    output = Path(config['output_dir'])
    completed = read_json(output / 'completed.json')
    guard = make_guard(config, initialization)
    if completed['guard'] != guard or completed['optimizer_steps'] != config['epochs'] * steps_per_epoch(config):
        raise ValueError('Evaluate only this completed run')
    selection = read_json(output / 'selection.json')
    if selection != completed['selection'] or selection['rule'] != 'val_mae_plus_rmse':
        raise ValueError('Checkpoint must be selected using validation only')
    weights = output / selection['checkpoint']
    if sha256(weights) != selection['sha256']:
        raise ValueError('Selected weights changed')
    samples, provenance = load_evaluation_samples(
        Path(s_metadata), Path(s_bank), Path(config['dataset']) / 'images_384_VarV2')
    if len(samples) != 230 or len({s.image_name for s in samples}) != 230:
        raise ValueError('Require all230 FSC147-S samples')
    eval_dir = output / 'evaluation_class'
    new_output(eval_dir)
    write_json(eval_dir / 'manifest.json', {'guard': guard, 'selection': selection,
               's_metadata_sha256': sha256(s_metadata), 's_bank_sha256': sha256(s_bank),
               's_bank_provenance': vars(provenance), 'prompt': 'class',
               'patch_size': 384, 'stride': 384, 'density_scale': 60,
               's_train_image_overlap': len(set(contract['selected_images']) & {s.image_name for s in samples}),
               's_dot_metrics': 'unavailable; different targets from original annotations'})
    model = build(config, 'cuda', weights)
    model.set_eval()
    from tqdm import tqdm
    test = ObjectCount(config['dataset'], crop_size=384, downsample_ratio=8,
                       method='test', concat_size=224, tokenizer=model.clip.tokenizer)
    test_rows, s_rows = [], []
    def row(name, caption, gt, prediction):
        error = float(prediction) - float(gt)
        return {'image': name, 'class': caption, 'gt_count': float(gt), 'pred_count': float(prediction),
                'abs_error': abs(error), 'signed_error': error, 'gt_group': gt_group(gt)}
    for image, gt, caption, mask, name in tqdm(test, total=len(test), desc='FSC147 test class'):
        prediction = predict_count(model, image.unsqueeze(0).cuda(), caption, mask,
                                   batch_size=1, patch_size=384, stride=384)
        test_rows.append(row(name + '.jpg', caption, gt, prediction))
    for sample in tqdm(samples, desc='FSC147-S class'):
        inputs = load_image_tensor(sample.image_path, 'cuda')
        mask = build_prompt_attention_mask(model.clip.tokenizer, sample.class_prompt)
        prediction = predict_count(model, inputs, sample.class_prompt, mask,
                                   batch_size=1, patch_size=384, stride=384)
        s_rows.append(row(sample.image_name, sample.class_name, sample.gt_count, prediction))
    write_csv(eval_dir / 'fsc147_test_class.csv', test_rows)
    write_csv(eval_dir / 'fsc147s_class.csv', s_rows)
    summary = {'primary_fsc147_test': counting_summary(test_rows),
               'secondary_fsc147s': counting_summary(s_rows),
               'fsc147s_by_gt_group': {group: counting_summary([r for r in s_rows if r['gt_group'] == group])
                                     for group in sorted({r['gt_group'] for r in s_rows})}}
    write_json(eval_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def compare(a, b, output):
    validate_pair(a, b)
    manifests = [read_json(Path(c['output_dir']) / 'manifest.json')['guard'] for c in (a, b)]
    for config, guard in zip((a, b), manifests):
        if guard['config'] != config:
            raise ValueError('Run manifest belongs to a different arm/config')
        completed = read_json(Path(config['output_dir']) / 'completed.json')
        if (completed['guard'] != guard or completed['epochs'] != config['epochs']
                or completed['optimizer_steps'] != config['epochs'] * steps_per_epoch(config)):
            raise ValueError('Comparison requires two completed training runs')
        evaluation = read_json(Path(config['output_dir']) / 'evaluation_class/manifest.json')
        if evaluation['guard'] != guard or evaluation['selection'] != completed['selection']:
            raise ValueError('Evaluation provenance differs from selected completed run')
    for key in ('source', 'runtime', 'data', 'initialization_sha256'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError('Run controls differ: ' + key)
    schedules = [read_json(Path(c['output_dir']) / 'epoch_records.json') for c in (a, b)]
    controls = lambda records: [{k: v for k, v in r.items() if k != 'validation'} for r in records]
    if controls(schedules[0]) != controls(schedules[1]) or len(schedules[0]) != a['epochs']:
        raise ValueError('Paired input/target/routing schedules or steps differ')
    new_output(output)
    summaries = {}
    for filename in ('fsc147_test_class.csv', 'fsc147s_class.csv'):
        tables = []
        for c in (a, b):
            with (Path(c['output_dir']) / 'evaluation_class' / filename).open(encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            table = {r['image']: r for r in rows}
            if len(table) != len(rows):
                raise ValueError('Duplicate evaluation image')
            tables.append(table)
        if set(tables[0]) != set(tables[1]):
            raise ValueError('Evaluation images differ')
        expected_count = manifests[0]['data']['split_counts']['test'] if filename.startswith('fsc147_test') else 230
        if len(tables[0]) != expected_count:
            raise ValueError('Incomplete evaluation coverage')
        paired = []
        for name in sorted(tables[0]):
            left, right = (table[name] for table in tables)
            if (left['gt_count'], left['class']) != (right['gt_count'], right['class']):
                raise ValueError('Evaluation targets differ')
            paired.append({'image': name, 'gt_count': float(left['gt_count']),
                           'abs_error_a': float(left['abs_error']), 'abs_error_b': float(right['abs_error']),
                           'b_minus_a_abs_error': float(right['abs_error']) - float(left['abs_error'])})
        differences = np.asarray([r['b_minus_a_abs_error'] for r in paired])
        rng = np.random.default_rng(3407)
        bootstrap = [rng.choice(differences, len(differences), replace=True).mean() for _ in range(2000)]
        metrics = [read_json(Path(c['output_dir']) / 'evaluation_class/summary.json')[
            'primary_fsc147_test' if filename.startswith('fsc147_test') else 'secondary_fsc147s'] for c in (a, b)]
        summaries[filename] = {'a': metrics[0], 'b': metrics[1],
                              'b_minus_a': {k: metrics[1][k] - metrics[0][k] for k in ('mae', 'rmse', 'bias')},
                              'mae_delta_image_bootstrap95': np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                              'uncertainty': 'conditional on this trained pair; not training-seed uncertainty'}
        write_csv(Path(output) / filename, paired)
    write_json(Path(output) / 'summary.json', summaries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'smoke', 'train', 'evaluate', 'compare'))
    parser.add_argument('--config', default=str(ROOT / 'configs/rp010a.json'))
    parser.add_argument('--asset-root', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--pair-id', default='pair_v1', help='new paired output namespace; safe single filename component')
    parser.add_argument('--data-dir', help='explicit immutable dataset location, shared by both arms')
    parser.add_argument('--prompt-bank', help='existing train bank, shared by both arms')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-full-training', action='store_true')
    parser.add_argument('--s-metadata', default=str(ROOT / 'FSC-147-S.json'))
    parser.add_argument('--s-bank')
    parser.add_argument('--comparison-output')
    parser.add_argument('--validate-only-output', help='prepare: validate data/tokens on CPU without initializing a model; new directory')
    args = parser.parse_args(argv)
    config = load_config(args.config, args.asset_root, args.seed)
    pair = [load_config(ROOT / ('configs/rp010' + arm + '.json'), args.asset_root, args.seed) for arm in ('a', 'b')]
    if not args.pair_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.pair_id):
        parser.error('Invalid --pair-id')
    for item in [config] + pair:
        for option, key in ((args.data_dir, 'dataset'), (args.prompt_bank, 'prompt_bank')):
            if option:
                item[key] = str(Path(option).resolve())
        old = Path(item['output_dir'])
        item['output_dir'] = str(old.parent / args.pair_id / old.name)
    validate_pair(*pair)
    if config != pair[0 if config['experiment_id'] == 'RP-010A' else 1]:
        raise ValueError('Use the reviewed paired configs; shared path/seed overrides must use CLI options')
    if args.mode == 'prepare':
        if args.validate_only_output:
            directory = new_output(args.validate_only_output)
            contract = get_contract(config)
            write_csv(directory / 'prompt_tokens.csv', contract['token_rows'])
            write_json(directory / 'validation.json', {
                'config': config, 'data': contract_identity(contract),
                'source': source_identity(), 'runtime': runtime_signature(),
                'paired_dataset_prefix_checks': verify_dataset_pair(*pair, contract),
                'host_free_bytes': host_free_bytes(),
                'initialization': 'not performed', 'gpu_smoke': 'pending'})
            print('CPU data/token validation passed:', directory, flush=True)
        else:
            prepare(config)
    elif args.mode == 'smoke':
        return 0 if run_smoke(config) else 2
    elif args.mode == 'train':
        if not args.allow_full_training:
            parser.error('Full training requires --allow-full-training after review and both smokes')
        # Both-arm smoke reports must belong to this exact prepared pair/source/runtime.
        smoke_reports = []
        for arm in pair:
            report = read_json(Path(arm['output_dir']).with_name(arm['experiment_id'] + '_smoke') / 'smoke_result.json')
            initialization = read_json(init_manifest_path(arm))
            if report.get('status') != 'passed' or report.get('guard') != make_guard(arm, initialization):
                raise ValueError('Both paired GPU smokes must pass for current provenance')
            smoke_reports.append(report)
        if any(smoke_reports[0][key] != smoke_reports[1][key]
               for key in ('sample', 'input_target_routing_sha256', 'optimizer_steps')):
            raise ValueError('Paired smokes used different samples/routing/step budgets')
        run_training(config, args.resume)
    elif args.mode == 'evaluate':
        evaluate(config, args.s_metadata, args.s_bank or str(Path(args.asset_root) / 'prompts/fsc147s_gemma4_26b_v3.json'))
    else:
        if not args.comparison_output:
            parser.error('--comparison-output required; must be a new directory')
        compare(*pair, args.comparison_output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
