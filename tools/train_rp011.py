#!/usr/bin/env python
"""RP-011: two weight-only warm starts, differing only in opt-in RRCL HNW."""
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.build import load_t2icount_checkpoint
from tools import train_rp010 as base
from utils.inference import predict_count
from utils.regression_trainer import Reg_Trainer, _atomic_torch_save, train_collate
from utils.rp010 import (
    EpochDataset, better_validation, counting_summary, load_config,
    new_output, parameter_contract, parameter_memory_estimate, read_json, sha256,
    steps_per_epoch, write_json,
)

ARMS = {'control': False, 'hnw': True}
PATH_KEYS = {'dataset', 'prompt_bank', 'sd_checkpoint', 'clip_dir',
             'initialization_checkpoint', 'output_dir'}


def plan(asset_root, rp010_run, output, config_path=None, data_dir=None, prompt_bank=None):
    spec = read_json(config_path or ROOT / 'configs/rp011.json')
    required = {'schema': 'rp011-hnw-v1', 'base_config': 'configs/rp010b.json',
                'epochs': 2, 'start_val': 0, 'val_epoch': 1,
                'sampling_epoch_start': 0, 'source_epoch': 9, 'checkpoint': 'best_epoch9.pth'}
    for key, value in required.items():
        if spec.get(key) != value:
            raise ValueError('RP-011 protocol mismatch: ' + key)
    run, output = Path(rp010_run).resolve(), Path(output).resolve()
    if run == output or run in output.parents or output in run.parents:
        raise ValueError('RP-011 output must be separate from RP-010 artifacts')
    config = load_config(ROOT / spec['base_config'], asset_root)
    if data_dir:
        config['dataset'] = str(Path(data_dir).resolve())
    if prompt_bank:
        config['prompt_bank'] = str(Path(prompt_bank).resolve())
    config.update(epochs=spec['epochs'], start_val=spec['start_val'], val_epoch=spec['val_epoch'],
                  initialization_checkpoint=str(run / spec['checkpoint']))
    # Run names/paths are bookkeeping, outside the effective training controls.
    config.pop('output_dir')
    config.pop('experiment_id')
    config['resume_policy'] = 'weights-only-fresh-optimizer-no-resume'
    configs = {arm: dict(config, rrcl_hnw=flag) for arm, flag in ARMS.items()}
    differences = [key for key in config.keys() | {'rrcl_hnw'}
                   if configs['control'][key] != configs['hnw'][key]]
    if differences != ['rrcl_hnw']:
        raise ValueError('Effective controls differ beyond HNW')
    return spec, configs, run, output


def verify_source(spec, configs, run, verify_weights=True):
    manifest = read_json(run / 'manifest.json')
    selection = read_json(run / 'selection.json')
    historical = manifest['guard']['config']
    checked_in = read_json(ROOT / spec['base_config'])
    for key in historical.keys() | checked_in.keys():
        if key not in PATH_KEYS and historical.get(key) != checked_in.get(key):
            raise ValueError('RP-010B recorded configuration differs: ' + key)
    if historical.get('experiment_id') != 'RP-010B' or historical.get('rrcl_hnw', False):
        raise ValueError('Expected original RP-010B rich-prompt source')
    if (selection.get('epoch') != spec['source_epoch']
            or selection.get('checkpoint') != spec['checkpoint']
            or selection.get('rule') != historical['checkpoint_metric']
            or selection.get('sha256') != spec['checkpoint_sha256']):
        raise ValueError('Expected the pinned RP-010B best epoch-9 selection')
    checkpoint = configs['control']['initialization_checkpoint']
    if verify_weights and sha256(checkpoint) != selection['sha256']:
        raise ValueError('Warm-start checkpoint SHA256 mismatch')
    return manifest, selection


def verify_data(config, manifest):
    contract = base.get_contract(config)
    # Portable identities survive relocation. Payload path/size/mtime is also
    # pinned between this preflight and training, but is not a cross-host byte hash.
    keys = ('selected_images', 'subset_indices', 'selected_image_fingerprint',
            'prompt_bank_sha256', 'split_sha256', 'class_sha256', 'annotations_sha256',
            'split_counts', 'donor_pool_size')
    for key in keys:
        if contract[key] != manifest['guard']['data'][key]:
            raise ValueError('RP-010B data identity differs: ' + key)
    return base.contract_identity(contract)


def source_identity():
    identity = base.source_identity()
    identity['rp011_files'] = {name: sha256(ROOT / name) for name in (
        'tools/train_rp011.py', 'configs/rp011.json')}
    return identity


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update((name + str(tuple(tensor.shape)) + str(tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return 'sha256:' + digest.hexdigest()


class PilotTrainer(Reg_Trainer):
    def __init__(self, config, output):
        self.control = config
        args = base.make_args(config, output)
        args.rrcl_hnw = config['rrcl_hnw']
        super().__init__(args)
        self.optimizer_steps = 0
        self.smoke = False
        self.records = []
        self.best_epoch = None

    def setup(self):
        super().setup()
        # Deliberately bypass resume: no RP-010 optimizer, RNG, epoch or guard is loaded.
        load_t2icount_checkpoint(self.model, self.control['initialization_checkpoint'])
        if self.optimizer.state or self.start_epoch != 0 or self.args.resume:
            raise ValueError('Warm start must have fresh optimizer state and epoch zero')
        self.scheduler = None  # RP-010B scheduler=null; no scheduler state to restore.
        for group, factor in zip(self.optimizer.param_groups, (0.1, 1.0)):
            if (group['lr'] != self.control['lr'] * factor
                    or group['weight_decay'] != self.control['weight_decay'] * factor
                    or tuple(group['betas']) != tuple(self.control['betas'])
                    or group['eps'] != self.control['eps']):
                raise ValueError('Optimizer differs from RP-010B')
        # Loading and model construction cannot shift either arm's training RNG.
        base.configure_determinism(self.control['seed'])

    def loader(self, epoch):
        dataset = EpochDataset(self.datasets['train'], self.control['seed'], epoch)
        return dataset, torch.utils.data.DataLoader(
            dataset, batch_size=self.control['batch_size'], shuffle=False, num_workers=0,
            pin_memory=True, collate_fn=train_collate,
            generator=torch.Generator().manual_seed(self.control['seed'] + epoch))

    def _optimize_batch(self, loss):
        if not torch.isfinite(loss).all():
            raise ValueError('Nonfinite loss')
        if not self.smoke:
            super()._optimize_batch(loss)
            self.optimizer_steps += 1
            return
        self.optimizer.zero_grad()
        loss.backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise ValueError('Missing or nonfinite smoke gradients')
        norm = math.sqrt(sum(float(g.double().square().sum()) for g in gradients))
        if norm <= 0:
            raise ValueError('Zero smoke gradient')
        self.smoke_values = {'loss': float(loss.detach()), 'gradient_l2': norm,
                             'gradient_tensors': len(gradients), 'optimizer_steps': 0}

    def validate(self):
        self.model.set_eval()
        rows = []
        with torch.no_grad():
            for images, counts, captions, mask, names in self.dataloaders['val']:
                prediction = predict_count(
                    self.model, images.to(self.device), captions[0], mask,
                    batch_size=self.args.batch_size, patch_size=self.args.crop_size,
                    stride=self.args.stride)
                gt = float(counts[0])
                rows.append({'image': names[0], 'class': captions[0], 'gt_count': gt,
                             'pred_count': float(prediction), 'signed_error': float(prediction) - gt})
        summary = counting_summary(rows)
        base.write_csv(Path(self.save_dir) / ('val_epoch{}.csv'.format(self.epoch)), rows)
        return summary

    def train(self):
        for self.epoch in range(self.control['epochs']):
            dataset, self.dataloaders['train'] = self.loader(self.epoch)
            self.train_epoch()  # Existing three-forward RP-010B rich supervision.
            validation = self.validate()
            if better_validation(validation['mae'], validation['rmse'], self.best_mae, self.best_mse):
                self.best_mae, self.best_mse = validation['mae'], validation['rmse']
                self.best_epoch = self.epoch
                _atomic_torch_save(self.model.state_dict(), str(Path(self.save_dir) / 'best.pth'))
            self.records.append({'epoch': self.epoch, 'steps': steps_per_epoch(self.control),
                                 'input_target_routing_sha256': dataset.trace.hexdigest(),
                                 'compatible': dataset.compatible, 'validation': validation})
            write_json(Path(self.save_dir) / 'epoch_records.json', self.records)
        expected = steps_per_epoch(self.control) * self.control['epochs']
        if self.optimizer_steps != expected:
            raise ValueError('Pilot optimizer-step count mismatch')
        write_json(Path(self.save_dir) / 'completed.json', {
            'optimizer_steps': self.optimizer_steps, 'best_epoch': self.best_epoch,
            'records': self.records})


def smoke(config, output):
    trainer = PilotTrainer(config, output)
    try:
        trainer.setup()
        starting = state_fingerprint(trainer.model)
        parameters = parameter_memory_estimate(parameter_contract(trainer.model))
        free, _ = torch.cuda.mem_get_info()
        if free <= parameters['weights_gradients_adam_estimate_bytes'] - parameters['parameter_bytes']:
            raise RuntimeError('GPU free memory below gradient/Adam estimate before smoke; activations excluded')
        trainer.smoke = True
        dataset, _ = trainer.loader(0)
        for index in range(min(32, len(dataset))):
            sample = dataset[index]
            if sample[5]['rich_compatible']:
                break
        else:
            raise ValueError('No compatible smoke sample in fixed first 32 anchors')
        prompts = []
        hook = trainer.model.register_forward_pre_hook(lambda model, inputs: prompts.append(tuple(inputs[1])))
        try:
            trainer.epoch = 0
            trainer.dataloaders['train'] = [train_collate([sample])]
            torch.cuda.reset_peak_memory_stats()
            trainer.train_epoch()
            torch.cuda.synchronize()
        finally:
            hook.remove()
        if len(prompts) != 3 or trainer.optimizer.state or state_fingerprint(trainer.model) != starting:
            raise ValueError('Smoke must run three forwards without changing model/optimizer state')
        peak_reserved = torch.cuda.max_memory_reserved()
        return dict(trainer.smoke_values, starting_weights_sha256=starting,
                    strict_model_load=True, state_entries=len(trainer.model.state_dict()),
                    input_target_routing_sha256=dataset.trace.hexdigest(),
                    sample=sample[5]['source_image_name'], prompts=prompts,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=peak_reserved, parameter_memory=parameters,
                    training_estimate_bytes=peak_reserved + 2 * parameters['trainable_bytes'])
    finally:
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


def paired_smokes(reports):
    for key in ('starting_weights_sha256', 'input_target_routing_sha256', 'sample', 'prompts'):
        if reports['control'][key] != reports['hnw'][key]:
            raise ValueError('Paired preflight mismatch: ' + key)


def guard(spec, configs, run, data):
    return {'spec': spec, 'configs': configs, 'data': data,
            'selection_sha256': sha256(run / 'selection.json'),
            'manifest_sha256': sha256(run / 'manifest.json'),
            'source': source_identity(), 'runtime': base.runtime_signature()}


def preflight(spec, configs, run, output):
    directory = new_output(output / 'preflight')
    result = {'status': 'pending', 'effective_configs': configs,
              'effective_config_differences': ['rrcl_hnw'], 'arms': {},
              'expected_optimizer_steps_per_arm': steps_per_epoch(configs['control']) * spec['epochs'],
              'host_free_bytes': base.host_free_bytes(), 'runtime': base.runtime_signature()}
    try:
        manifest, selection = verify_source(spec, configs, run, verify_weights=False)
        result['source_selection'] = selection
        blockers = []
        if not Path(configs['control']['initialization_checkpoint']).is_file():
            blockers.append('Actual best_epoch9.pth is not available on this runtime')
        else:
            verify_source(spec, configs, run)
            result['checkpoint_sha256_verified'] = True
        data = verify_data(configs['control'], manifest)
        result['guard'] = guard(spec, configs, run, data)
        if not torch.cuda.is_available():
            blockers.append('CUDA unavailable')
        else:
            free, total = torch.cuda.mem_get_info()
            result.update(gpu_free_bytes=free, gpu_total_bytes=total)
        if result['host_free_bytes'] is not None and result['host_free_bytes'] < 8 * 2**30:
            blockers.append('Less than 8 GiB host RAM free for full model/checkpoint initialization')
        if blockers:
            result['blockers'] = blockers
            return result
        for arm, config in configs.items():
            result['arms'][arm] = smoke(config, directory / arm)
            write_json(directory / 'report.json', result)
        paired_smokes(result['arms'])
        estimate = max(item['training_estimate_bytes'] for item in result['arms'].values())
        result['estimated_training_bytes'] = estimate
        result['recommended_free_gpu_bytes'] = math.ceil(1.2 * estimate)
        result['memory_estimate_note'] = 'Smoke reserved peak + two Adam moments for all trainable parameters; 20% headroom recommended, not a measured training peak'
        if result['gpu_free_bytes'] < estimate:
            result['blockers'] = ['GPU free memory below conservative smoke-plus-Adam training estimate']
        else:
            result['status'] = 'passed'
        return result
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        result.update(status='failed', error=str(error))
        return result
    finally:
        write_json(directory / 'report.json', result)


def require_preflight(spec, configs, run, output):
    report = read_json(output / 'preflight/report.json')
    if report.get('status') != 'passed' or set(report.get('arms', {})) != set(ARMS):
        raise ValueError('Both RP-011 GPU smokes must pass before training')
    manifest, _ = verify_source(spec, configs, run)
    data = verify_data(configs['control'], manifest)
    if report['guard'] != guard(spec, configs, run, data):
        raise ValueError('Preflight provenance changed; use a new output directory and rerun preflight')
    paired_smokes(report['arms'])
    free, _ = torch.cuda.mem_get_info()
    if free < report['estimated_training_bytes']:
        raise ValueError('GPU free memory fell below preflight training estimate')
    return report


def train_arm(spec, configs, run, output, arm):
    report = require_preflight(spec, configs, run, output)
    directory = new_output(output / arm)
    trainer = PilotTrainer(configs[arm], directory)
    trainer.setup()
    starting = state_fingerprint(trainer.model)
    if starting != report['arms'][arm]['starting_weights_sha256']:
        raise ValueError('Training start differs from verified preflight weights')
    write_json(directory / 'manifest.json', {'guard': report['guard'], 'arm': arm,
                                           'starting_weights_sha256': starting})
    trainer.train()


def compare(output):
    manifests = {arm: read_json(output / arm / 'manifest.json') for arm in ARMS}
    if manifests['control']['guard'] != manifests['hnw']['guard']:
        raise ValueError('Pilot arm provenance differs')
    if manifests['control']['starting_weights_sha256'] != manifests['hnw']['starting_weights_sha256']:
        raise ValueError('Pilot starting weights differ')
    completed = {arm: read_json(output / arm / 'completed.json') for arm in ARMS}
    records = {arm: completed[arm]['records'] for arm in ARMS}
    if any(len(items) != 2 for items in records.values()) or any(
            item['optimizer_steps'] != 2000 for item in completed.values()):
        raise ValueError('Expected two epochs and 2000 updates per arm')
    rows = []
    for epoch, (control, hnw) in enumerate(zip(records['control'], records['hnw'])):
        for key in ('epoch', 'steps', 'input_target_routing_sha256', 'compatible'):
            if control[key] != hnw[key]:
                raise ValueError('Training schedule/data mismatch: ' + key)
        validation_rows = {}
        for arm in ARMS:
            with (output / arm / ('val_epoch{}.csv'.format(epoch))).open(newline='') as handle:
                items = list(csv.DictReader(handle))
            validation_rows[arm] = {item['image']: item for item in items}
            if len(validation_rows[arm]) != len(items):
                raise ValueError('Duplicate validation image')
            for item in items:
                if not math.isclose(float(item['pred_count']) - float(item['gt_count']),
                                    float(item['signed_error']), rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError('Signed error must be prediction minus GT')
            measured = counting_summary([{'signed_error': float(item['signed_error'])} for item in items])
            if measured['n'] != manifests[arm]['guard']['data']['split_counts']['val']:
                raise ValueError('Validation coverage differs from the fixed split')
            for metric in ('n', 'mae', 'rmse', 'bias'):
                if not math.isclose(measured[metric], records[arm][epoch]['validation'][metric], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError('Validation rows and summary disagree: ' + metric)
        if validation_rows['control'].keys() != validation_rows['hnw'].keys():
            raise ValueError('Validation image sets differ')
        for image, left in validation_rows['control'].items():
            right = validation_rows['hnw'][image]
            if left['class'] != right['class'] or float(left['gt_count']) != float(right['gt_count']):
                raise ValueError('Validation class/GT differs: ' + image)
        row = {'epoch': epoch}
        for metric in ('mae', 'rmse', 'bias'):
            row['control_' + metric] = control['validation'][metric]
            row['hnw_' + metric] = hnw['validation'][metric]
            row['hnw_minus_control_' + metric] = hnw['validation'][metric] - control['validation'][metric]
        rows.append(row)
    base.write_csv(output / 'paired_validation.csv', rows)
    write_json(output / 'paired_validation.json', {
        'difference': 'HNW minus control', 'signed_bias': 'prediction minus GT',
        'rows': rows, 'interpretation': 'raw observations; researcher review required'})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'train', 'compare'))
    parser.add_argument('--asset-root', required=True)
    parser.add_argument('--rp010-run', required=True, help='Read-only RP-010B directory with manifest, selection and best_epoch9.pth')
    parser.add_argument('--output-dir', required=True, help='Separate RP-011 output directory')
    parser.add_argument('--data-dir')
    parser.add_argument('--prompt-bank')
    parser.add_argument('--arm', choices=tuple(ARMS))
    parser.add_argument('--allow-training', action='store_true')
    args = parser.parse_args(argv)
    spec, configs, run, output = plan(args.asset_root, args.rp010_run, args.output_dir,
                                     data_dir=args.data_dir, prompt_bank=args.prompt_bank)
    if args.mode == 'preflight':
        result = preflight(spec, configs, run, output)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result['status'] == 'passed' else 2
    if args.mode == 'train':
        if not args.allow_training or args.arm is None:
            parser.error('train requires --arm and --allow-training after preflight review')
        train_arm(spec, configs, run, output, args.arm)
    else:
        print(json.dumps(compare(output), indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
