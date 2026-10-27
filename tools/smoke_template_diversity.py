#!/usr/bin/env python
"""One RP-009 batch through the real training loss/backward; zero optimizer steps."""

import argparse
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets.dataset import build_rich_prompt_attention_mask
from train import parse_arg, resolve_training_paths
from utils import regression_trainer as training
from utils.rich_prompt_training import build_class_templates
from tools.validate_rich_prompt_training_bank import main as validate_bank


def parameter_fingerprint(model):
    """sha256 named-parameters-v1: names/dtypes/shapes and exact raw CPU bytes."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        header = json.dumps([name, str(parameter.dtype), list(parameter.shape)],
                            separators=(',', ':')).encode('utf-8')
        digest.update(header + b'\n')
        raw = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return 'sha256:' + digest.hexdigest()


class SmokeTrainer(training.Reg_Trainer):
    def _optimize_batch(self, loss):
        # This override is diagnostic only. The production optimizer path is unchanged.
        self.smoke_loss = float(loss.detach())
        if not math.isfinite(self.smoke_loss):
            raise ValueError('Nonfinite smoke loss.')
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        squared_norm, used, elements = 0., 0, 0
        for parameter in self.model.parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if not torch.isfinite(gradient).all().item():
                raise ValueError('Nonfinite smoke gradient.')
            squared_norm += gradient.double().square().sum().item()
            used += 1
            elements += gradient.numel()
        if used == 0 or not math.isfinite(squared_norm) or squared_norm <= 0:
            raise ValueError('Smoke backward produced no finite nonzero gradient.')
        self.smoke_gradients = {'parameter_tensors_with_grad': used,
                               'gradient_elements': elements,
                               'gradient_l2': math.sqrt(squared_norm)}


def run_smoke_batch(trainer, batch, tokenizer):
    """Observe production forwards/criterion without replacing any loss math."""
    if len(batch) != 6 or batch[0].shape[0] != 1 or not batch[5]['rich_compatible'].all():
        raise ValueError('Smoke requires one compatible positive batch.')
    if (trainer.args.rich_supervision_mode != 'template-diversity'
            or trainer.args.rich_consistency_weight != 0):
        raise ValueError('Smoke requires template-diversity with zero consistency.')
    caption = batch[2][0]
    expected_captions = (caption,) + build_class_templates(caption)
    if not torch.equal(batch[3], build_rich_prompt_attention_mask(tokenizer, caption).unsqueeze(0)):
        raise ValueError('Incorrect original class attention mask.')
    expected_masks = [batch[3]] + [
        build_rich_prompt_attention_mask(tokenizer, text).unsqueeze(0)
        for text in expected_captions[1:]]
    for index, slot in enumerate(('detailed', 'generalized'), 1):
        if batch[5][slot + '_prompt'] != (expected_captions[index],):
            raise ValueError('Incorrect template caption: ' + slot)
        if not torch.equal(batch[5][slot + '_prompt_attn_mask'], expected_masks[index]):
            raise ValueError('Incorrect template attention mask: ' + slot)
    expected_images = batch[0].to(trainer.device)
    expected_density = batch[1].to(trainer.device) * 60
    expected_image_mask = batch[4].to(trainer.device)
    forwards, components = [], []

    def observe_forward(model, inputs):
        index = len(forwards)
        if index >= 3 or tuple(inputs[1]) != (expected_captions[index],):
            raise ValueError('Unexpected forward count/caption.')
        if not torch.equal(inputs[0], expected_images):
            raise ValueError('Forward changed batch images.')
        mask = inputs[2].detach().cpu()
        if tuple(mask.shape) != (1, 77, 1, 1) or not torch.equal(
                mask.flatten(1), expected_masks[index]):
            raise ValueError('Forward changed class/template attention mask.')
        forwards.append({'caption': inputs[1][0], 'mask': mask.flatten().tolist(),
                         'mask_tokens': mask.sum().item()})

    original_criterion = training._t2i_loss_components
    def observe_criterion(outputs, density, image_mask):
        if not torch.equal(density, expected_density) or not torch.equal(image_mask, expected_image_mask):
            raise ValueError('Supervised branch changed ground truth or image mask.')
        result = original_criterion(outputs, density, image_mask)
        values = {key: value.detach().item() for key, value in result.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError('Nonfinite supervised loss component.')
        components.append(values)
        return result

    initial = parameter_fingerprint(trainer.model)
    hook = trainer.model.register_forward_pre_hook(observe_forward)
    trainer.dataloaders = {'train': [batch]}
    trainer.epoch = 0
    try:
        with mock.patch.object(training, '_t2i_loss_components', side_effect=observe_criterion), \
             mock.patch.object(training, 'rich_consistency_loss',
                               side_effect=AssertionError('Consistency must not run.')), \
             mock.patch.object(trainer.optimizer, 'step',
                               side_effect=AssertionError('Optimizer step is forbidden.')) as step:
            trainer.train_epoch()  # Exactly this one preloaded batch; never trainer.train().
            assert step.call_count == 0
    finally:
        hook.remove()
    if len(forwards) != 3 or len(components) != 4:
        raise ValueError('Expected 3 forwards and 4 criterion calls (class all/subset + 2 auxiliaries).')
    final = parameter_fingerprint(trainer.model)
    if final != initial:
        raise ValueError('Model parameters changed during smoke.')
    return {'status': 'PASS', 'optimizer_steps': 0, 'backward_completed': True,
            'loss': trainer.smoke_loss, 'gradients': trainer.smoke_gradients,
            'forwards': forwards, 'supervised_components': components,
            'fingerprint_scheme': 'named-parameters-v1',
            'initial_parameter_fingerprint': initial, 'final_parameter_fingerprint': final,
            'source_images': list(batch[5]['source_image_name']),
            'limitation': 'Mechanical one-batch check only; no counting evaluation. '
                          'Forward buffers/caches can change in training mode; '
                          'parameter fingerprint excludes buffers.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-scan-batches', type=int, default=32)
    options, training_argv = parser.parse_known_args(argv)
    if not re.fullmatch('[0-9a-f]{40}', options.expected_commit):
        raise ValueError('--expected-commit must be the full pushed commit SHA.')
    actual_commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                            text=True).strip()
    if actual_commit != options.expected_commit:
        raise ValueError('Checkout does not match --expected-commit.')
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', 'HEAD', '--', '.'], check=True)
    args = resolve_training_paths(parse_arg(training_argv))
    if (args.rich_supervision_mode != 'template-diversity' or args.resume
            or args.batch_size != 1 or args.num_workers != 0 or args.seed != 3407
            or args.train_samples != 1000 or args.train_subset_seed != 3407):
        raise ValueError('Use the documented fresh RP-009 1000/3407, batch-1, worker-0 settings.')
    output = Path(options.output)
    if output.exists() or (Path(args.save_dir) / args.content).exists():
        raise FileExistsError('Use new smoke report and run-directory names; never overwrite.')
    if options.max_scan_batches < 1:
        raise ValueError('--max-scan-batches must be positive.')
    if validate_bank(['--rich-prompt-bank', args.rich_prompt_bank, '--data-dir', args.data_dir,
                      '--train-samples', str(args.train_samples),
                      '--train-subset-seed', str(args.train_subset_seed)]) != 0:
        return 1
    torch.backends.cudnn.benchmark = True  # Same as train.py; no new precision/RNG settings.
    trainer = SmokeTrainer(args)
    trainer.setup()  # Actual T2ICount, assets, tokenizer, dataset, optimizer configuration.
    for index, batch in enumerate(islice(trainer.dataloaders['train'], options.max_scan_batches)):
        if batch[5]['rich_compatible'].all().item():
            source = batch[5]['source_image_name'][0]
            if batch[2][0] != trainer.rich_prompt_bank.records[source].class_name:
                raise ValueError('Class caption differs from validated class provenance.')
            report = run_smoke_batch(trainer, batch, trainer.model.clip.tokenizer)
            report.update(git_commit=actual_commit, args=vars(args),
                          rich_prompt_config=trainer.rich_prompt_config,
                          torch=torch.__version__, cuda=torch.version.cuda,
                          gpu=torch.cuda.get_device_name(), scan_batch_index=index)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x', encoding='utf-8') as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
            print(json.dumps(report, indent=2, allow_nan=False))
            print('Smoke PASS; zero optimizer updates. Report: ' + str(output))
            return 0
    raise RuntimeError('No compatible batch within scan limit; no smoke result.')


if __name__ == '__main__':
    raise SystemExit(main())
