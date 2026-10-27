#!/usr/bin/env python
"""RP-008 loss/gradient probe using the production train_epoch, without updates."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from train import parse_arg, resolve_training_paths
from utils.regression_trainer import (
    Reg_Trainer, _capture_rng_state, _restore_rng_state,
)
from tools.validate_rich_prompt_training_bank import main as validate_bank


def rng_digest():
    state = _capture_rng_state()
    digest = hashlib.sha256(pickle.dumps((state['python'], state['numpy'])))
    for value in [state['torch']] + state.get('cuda', []):
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def tensors(model):
    # Includes nonpersistent buffers, which state_dict alone would miss.
    return dict(itertools.chain(model.named_parameters(), model.named_buffers()))


def reset_attention_cache(model):
    for module in model.modules():
        store = getattr(module, 'attention_store', None)
        if store is not None:
            store.reset()


class TensorSnapshot:
    """Stream tensors to local scratch to avoid another full model in GPU/RAM."""
    def __init__(self, model, directory):
        self.paths = {}
        for index, (name, value) in enumerate(tensors(model).items()):
            path = Path(directory) / ('state_{}.pt'.format(index))
            torch.save(value.detach().cpu(), path)
            self.paths[name] = path

    def restore(self, model):
        model.zero_grad(set_to_none=True)
        reset_attention_cache(model)
        with torch.no_grad():
            for name, value in tensors(model).items():
                value.copy_(torch.load(self.paths[name], map_location='cpu'))

    def changed(self, model):
        return [name for name, value in tensors(model).items()
                if not torch.equal(value.detach().cpu(), torch.load(
                    self.paths[name], map_location='cpu'))]


class ProbeTrainer(Reg_Trainer):
    def _optimize_batch(self, loss):
        self.model.zero_grad(set_to_none=True)
        self.probe_loss = loss.detach().item()
        loss.backward()  # Deliberately no optimizer.step(), validation, or save.


def gradient_comparison(model, directory, write_reference=False):
    ref_sq = diff_sq = current_sq = 0.0
    unused_mismatches = []
    per_parameter = {}
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        path = Path(directory) / ('grad_{}.pt'.format(index))
        grad = None if parameter.grad is None else parameter.grad.detach().cpu()
        if write_reference:
            torch.save(grad, path)
            if grad is not None:
                current_sq += grad.double().square().sum().item()
        else:
            reference = torch.load(path, map_location='cpu')
            if (reference is None) != (grad is None):
                unused_mismatches.append(name)
            if reference is None and grad is None:
                continue
            reference = torch.zeros_like(grad) if reference is None else reference
            grad = torch.zeros_like(reference) if grad is None else grad
            a, b = reference.double(), grad.double()
            r2 = a.square().sum().item()
            d2 = (b - a).square().sum().item()
            c2 = b.square().sum().item()
            ref_sq += r2
            diff_sq += d2
            current_sq += c2
            per_parameter[name] = {
                'difference_l2': math.sqrt(d2),
                'reference_l2': math.sqrt(r2),
                'current_l2': math.sqrt(c2),
            }
    if write_reference:
        ref_sq = current_sq
    return {
        'reference_norm': math.sqrt(ref_sq),
        'current_norm': math.sqrt(current_sq),
        'difference_norm': math.sqrt(diff_sq),
        'relative_difference': math.sqrt(diff_sq) / max(math.sqrt(ref_sq), 1e-30),
        'unused_gradient_mismatches': unused_mismatches,
        'per_parameter': per_parameter,
    }


def compare_batch(trainer, batch, snapshot, directory, atol=1e-6, rtol=1e-5):
    """One instance, same parameters/inputs/RNG; normal sequential extra forwards."""
    initial_rng = _capture_rng_state()
    training_flags = {name: m.training for name, m in trainer.model.named_modules()}
    results = {}
    try:
        for label in ('class_only', 'class_only_replay', 'repeat_class'):
            snapshot.restore(trainer.model)
            _restore_rng_state(initial_rng)
            probe = ProbeTrainer.__new__(ProbeTrainer)
            probe.model, probe.device = trainer.model, trainer.device
            probe.args = SimpleNamespace(**vars(trainer.args))
            probe.args.rich_supervision_mode = 'repeat-class'
            probe.args.rich_consistency_weight = 0.0
            probe.args.rich_loss_diagnostic_steps = 0
            probe.epoch = 0
            probe.rich_prompt_config = None
            probe._rich_diagnostic_steps_logged = 0
            probe.dataloaders = {'train': [batch if label == 'repeat_class' else batch[:5]]}
            forwards = []
            before = {}

            def pre_hook(model, inputs):
                before['rng'] = rng_digest()
                before['buffers'] = {n: b.detach().clone() for n, b in model.named_buffers()}
                before['versions'] = {n: p._version for n, p in model.named_parameters()}

            def post_hook(model, inputs, outputs):
                forwards.append({
                    'captions': list(inputs[1]),
                    'mask_sum': inputs[2].detach().sum().item(),
                    'rng_changed': before['rng'] != rng_digest(),
                    'buffers_changed': [n for n, b in model.named_buffers()
                                        if not torch.equal(b, before['buffers'][n])],
                    'parameter_versions_changed': [n for n, p in model.named_parameters()
                                                  if p._version != before['versions'][n]],
                })

            handles = [probe.model.register_forward_pre_hook(pre_hook),
                       probe.model.register_forward_hook(post_hook)]
            start_rng = rng_digest()
            try:
                probe.train_epoch()
            finally:
                for handle in handles:
                    handle.remove()
            end_rng = rng_digest()
            results[label] = {
                'loss': probe.probe_loss,
                'rng_changed': start_rng != end_rng,
                'end_rng': end_rng,
                'forwards': forwards,
                'registered_tensors_changed': snapshot.changed(probe.model),
                'gradients': gradient_comparison(
                    probe.model, directory, write_reference=(label == 'class_only')),
            }
        reference = results['class_only']
        checks = []
        for label in ('class_only_replay', 'repeat_class'):
            result = results[label]
            result['loss_difference'] = result['loss'] - reference['loss']
            g = result['gradients']
            # Check every parameter as well as the global gradient norm.
            gradients_close = (g['reference_norm'] > 0 and
                math.isfinite(g['reference_norm']) and math.isfinite(g['current_norm']) and
                not g['unused_gradient_mismatches'] and
                g['difference_norm'] <= atol + rtol * g['reference_norm'] and
                all(p['difference_l2'] <= atol + rtol * p['reference_l2']
                    for p in g['per_parameter'].values()))
            result['numerically_close'] = (
                math.isfinite(result['loss']) and math.isfinite(reference['loss'])
                and abs(result['loss_difference']) <= atol + rtol * abs(reference['loss'])
                and gradients_close)
            checks.append(result['numerically_close'] and
                          result['end_rng'] == reference['end_rng'] and
                          not result['registered_tensors_changed'] and
                          not any(f['buffers_changed'] or f['parameter_versions_changed']
                                  for f in result['forwards']))
        results['skip_candidate'] = (all(checks) and
            not reference['registered_tensors_changed'] and
            len(results['repeat_class']['forwards']) == 3)
        results['tolerances'] = {'absolute': atol, 'relative': rtol}
        return results
    finally:
        snapshot.restore(trainer.model)
        _restore_rng_state(initial_rng)
        for name, module in trainer.model.named_modules():
            module.training = training_flags[name]


def json_finite(value):
    """Keep nonfinite diagnostics visible without emitting invalid JSON numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(v) for v in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--output', required=True)
    parser.add_argument('--probe-batches', type=int, default=2)
    parser.add_argument('--max-scan-batches', type=int, default=32)
    parser.add_argument('--scratch-dir', default=None)
    options, training_argv = parser.parse_known_args(argv)
    args = resolve_training_paths(parse_arg(training_argv))
    if (args.rich_supervision_mode != 'repeat-class' or args.resume or args.seed < 0
            or args.batch_size != 1 or args.num_workers != 0):
        raise ValueError('Preflight requires repeat-class, a fixed seed, batch size 1, '
                         'num-workers 0, and fresh initialization (no resume).')
    if not 1 <= options.probe_batches <= options.max_scan_batches:
        raise ValueError('Require 1 <= probe-batches <= max-scan-batches.')
    output = Path(options.output)
    if output.exists():
        raise FileExistsError('Use a new preflight output path: {}'.format(output))
    # Fail on bank/subset mismatch BEFORE loading the GPU model.
    if validate_bank(['--rich-prompt-bank', args.rich_prompt_bank,
                      '--data-dir', args.data_dir, '--train-samples', str(args.train_samples),
                      '--train-subset-seed', str(args.train_subset_seed)]) != 0:
        return 1
    torch.backends.cudnn.benchmark = True  # Match train.py; do not disable dropout/RNG.
    trainer = Reg_Trainer(args)
    trainer.setup()
    required = sum(t.numel() * t.element_size() for t in tensors(trainer.model).values())
    required += sum(p.numel() * p.element_size() for p in trainer.model.parameters()
                    if p.requires_grad)
    scratch = options.scratch_dir or tempfile.gettempdir()
    if shutil.disk_usage(scratch).free < required + 1024**3:
        raise RuntimeError('Insufficient local scratch for streamed state/gradient snapshots.')
    records = []
    with tempfile.TemporaryDirectory(prefix='rp008_', dir=scratch) as directory:
        snapshot = TensorSnapshot(trainer.model, directory)
        iterator = iter(trainer.dataloaders['train'])
        for index in range(options.max_scan_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if not batch[5]['rich_compatible'].all().item():
                continue  # Never force augmentation branches or compatibility.
            result = compare_batch(trainer, batch, snapshot, directory)
            result['scan_batch_index'] = index
            result['source_images'] = list(batch[5]['source_image_name'])
            records.append(result)
            print('Probe {}: class={} repeat={} loss_diff={} grad_rel={}'.format(
                len(records), result['class_only']['loss'], result['repeat_class']['loss'],
                result['repeat_class']['loss_difference'],
                result['repeat_class']['gradients']['relative_difference']), flush=True)
            if len(records) == options.probe_batches:
                break
    enough = len(records) == options.probe_batches
    skip = enough and all(r['skip_candidate'] for r in records)
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(ROOT), *arguments], text=True).strip()
    source_files = ['train.py', 'utils/regression_trainer.py', 'utils/rich_prompt_training.py',
                    'datasets/dataset.py', 'models/reg_model.py', 'models/diff_unet.py',
                    'models/decoder.py', 'models/build.py', 'ldm/modules/attention.py',
                    'ldm/modules/diffusionmodules/util.py',
                    'tools/preflight_repeat_class.py', args.config]
    report = {
        'experiment': 'RP-008 preflight only', 'optimizer_steps': 0,
        'command': sys.argv if argv is None else argv,
        'args': vars(args), 'rich_prompt_config': trainer.rich_prompt_config,
        'git_head': git('rev-parse', 'HEAD'), 'git_branch': git('branch', '--show-current'),
        'git_status': git('status', '--short'),
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in source_files},
        'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'active_dropout': [n for n, m in trainer.model.named_modules()
                           if isinstance(m, torch.nn.modules.dropout._DropoutNd)
                           and m.training and m.p > 0],
        'activation_checkpoint_modules': [n for n, m in trainer.model.named_modules()
                                          if getattr(m, 'use_checkpoint', False)],
        'state_scope': 'All named parameters and buffers (including nonpersistent); '
                       'per-forward buffer values/parameter versions. T2ICount attention '
                       'cache is overwritten/reset each forward, not persistent learned state. '
                       'Arbitrary Python attributes are not exhaustively compared.',
        'historical_rp003_initial_weights_and_runtime_verified': False,
        'enough_compatible_batches': enough, 'batches': records,
        'recommendation': (
            'SKIP full 10-epoch RP-008: measured batch losses/gradients agree within '
            'tolerances, with no extra registered-state or RNG progression effect. '
            'Propose a template-diversity control for researcher review; not approved.'
            if skip else
            'REVIEW before full training: differences, state/RNG effects, nonfinite '
            'values, or insufficient compatible batches prevent an equivalence finding.'),
        'limitation': 'Finite batch diagnostic, not proof of trajectory equivalence '
                      'or a scientific conclusion about rich semantics.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_finite(report), indent=2, allow_nan=False), encoding='utf-8')
    print(report['recommendation'])
    print('Report: {}'.format(output))
    return 0 if enough else 2


if __name__ == '__main__':
    raise SystemExit(main())
