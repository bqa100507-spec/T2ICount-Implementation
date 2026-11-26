#!/usr/bin/env python
"""Evaluate the frozen official T2ICount, then run one gated warm-start epoch."""
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tools import train_rp010 as base
from tools.train_rp011 import PilotTrainer, smoke, state_fingerprint
from utils.rp010 import counting_summary, digest_json, load_config, new_output, read_json, sha256, write_json

SPEC_PATH = ROOT / 'configs/rp012.json'
BASELINE_CSV = Path('preflight/baseline/val_epoch0.csv')
REPORT = Path('preflight/report.json')
METRICS = ('mae', 'rmse', 'bias')


def plan(asset_root, checkpoint, output, data_dir=None, prompt_bank=None):
    spec = read_json(SPEC_PATH)
    source_config = ROOT / 'configs/rp010b.json'
    if digest_json(read_json(source_config)) != spec['rp010b_config_digest']:
        raise ValueError('RP-010B source configuration differs from the pinned protocol')
    config = load_config(source_config, asset_root)
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    protected = {'rp010', 'rp011', 'rp-010', 'rp-010a', 'rp-010b', 'rp-011'}
    if (output == checkpoint or output in checkpoint.parents
            or any(part.casefold() in protected for part in output.parts)):
        raise ValueError('Choose a separate RP-012 output directory')
    if config['experiment_id'] != 'RP-010B' or config['supervision_mode'] != 'rich':
        raise ValueError('RP-012 requires the RP-010B rich supervision configuration')
    config.update(
        experiment_id='RP-012', output_dir=str(output), epochs=1, start_val=0,
        initialization_checkpoint=str(checkpoint), rrcl_hnw=False,
        resume_policy='weights-only-fresh-optimizer-no-resume',
    )
    if data_dir is not None:
        config['dataset'] = str(Path(data_dir).resolve())
    if prompt_bank is not None:
        config['prompt_bank'] = str(Path(prompt_bank).resolve())
    return spec, config, output


def verify_inputs(spec, config):
    checkpoint_hash = sha256(config['initialization_checkpoint'])
    if checkpoint_hash != spec['official_checkpoint_sha256']:
        raise ValueError('Official best_model_paper.pth SHA-256 mismatch')
    contract = base.get_contract(config)
    for key, expected in spec['rp010b_data'].items():
        if contract[key] != expected:
            raise ValueError('RP-010B data identity differs: ' + key)
    return {'checkpoint_sha256': checkpoint_hash, 'data': base.contract_identity(contract)}


def make_guard(spec, config, inputs):
    source = base.source_identity()
    source['rp012_files'] = {
        name: sha256(ROOT / name)
        for name in ('tools/train_rp012.py', 'tools/train_rp011.py', 'configs/rp012.json')
    }
    clip_dir = Path(config['clip_dir'])
    tokenizer_files = {
        path.name: sha256(path) for path in sorted(clip_dir.iterdir())
        if path.is_file() and path.suffix in {'.json', '.txt'}
    }
    return {'spec': spec, 'config': config, 'inputs': inputs, 'source': source,
            'runtime': base.runtime_signature(), 'clip_text_assets': tokenizer_files}


def baseline_check(summary, spec):
    reference = spec['published_validation']
    tolerance = spec['baseline_relative_tolerance']
    relative_delta = {key: summary[key] / reference[key] - 1 for key in ('mae', 'rmse')}
    return {'reference': reference, 'relative_tolerance': tolerance,
            'relative_delta': relative_delta,
            'passed': all(abs(value) <= tolerance for value in relative_delta.values())}


def frozen_baseline(config, output):
    trainer = PilotTrainer(config, output)
    try:
        trainer.setup()
        trainer.model.requires_grad_(False)
        starting = state_fingerprint(trainer.model)
        trainer.epoch = 0
        summary = trainer.validate()
        if trainer.optimizer_steps or state_fingerprint(trainer.model) != starting:
            raise ValueError('Frozen baseline model changed during validation')
        return {'strict_model_load': True, 'starting_weights_sha256': starting,
                'optimizer_steps': 0, 'metrics': summary}
    finally:
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


def preflight(spec, config, output):
    directory = new_output(output / 'preflight')
    result = {'status': 'failed', 'optimizer_steps': 0}
    try:
        inputs = verify_inputs(spec, config)
        result['guard'] = make_guard(spec, config, inputs)
        host = base.host_free_bytes()
        if host is not None and host < 8 * 2**30:
            raise RuntimeError('Require at least 8 GiB available host RAM')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Require one CUDA GPU')
        free, total = torch.cuda.mem_get_info()
        result['gpu_before'] = {'free_bytes': free, 'total_bytes': total}
        if free < 21 * 2**30:
            raise RuntimeError('Require 21 GiB free VRAM for the full-precision preflight')

        result['baseline'] = frozen_baseline(config, directory / 'baseline')
        summary = result['baseline']['metrics']
        result['baseline_csv_sha256'] = sha256(output / BASELINE_CSV)
        result['baseline_plausibility'] = baseline_check(summary, spec)
        if summary['n'] != spec['rp010b_data']['split_counts']['val']:
            raise ValueError('Frozen baseline validation coverage differs from FSC147 val')
        if not result['baseline_plausibility']['passed']:
            raise ValueError('Frozen baseline outside the published validation tolerance')

        result['smoke'] = smoke(config, directory / 'smoke')
        if result['smoke']['starting_weights_sha256'] != result['baseline']['starting_weights_sha256']:
            raise ValueError('Smoke and frozen baseline must load identical model weights')
        if result['smoke']['training_estimate_bytes'] > free:
            raise RuntimeError('Estimated training memory exceeds free VRAM')
        result['status'] = 'passed'
        return result
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        write_json(output / REPORT, result)


def paired_validation(baseline_file, trained_file, expected_count):
    def read_predictions(path):
        predictions = {}
        with path.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                name = row['image']
                gt, prediction = float(row['gt_count']), float(row['pred_count'])
                error = prediction - gt
                if name in predictions or not math.isfinite(error):
                    raise ValueError('Duplicate or invalid validation prediction: ' + name)
                if not math.isclose(error, float(row['signed_error']), abs_tol=1e-10):
                    raise ValueError('Validation signed error differs from prediction minus GT: ' + name)
                predictions[name] = (row['class'], gt, prediction, error)
        if len(predictions) != expected_count:
            raise ValueError('Validation coverage differs from FSC147 val')
        return predictions

    baseline, trained = read_predictions(baseline_file), read_predictions(trained_file)
    if baseline.keys() != trained.keys():
        raise ValueError('Validation image IDs differ')
    rows = []
    counts = {'improved': 0, 'worsened': 0, 'tied': 0}
    for name, before in baseline.items():
        after = trained[name]
        if before[:2] != after[:2]:
            raise ValueError('Validation class or GT differs: ' + name)
        delta = abs(after[3]) - abs(before[3])
        if abs(delta) <= 1e-9:
            outcome = 'tied'
        else:
            outcome = 'improved' if delta < 0 else 'worsened'
        counts[outcome] += 1
        rows.append({'image': name, 'class': before[0], 'gt_count': before[1],
                     'official_prediction': before[2], 'trained_prediction': after[2],
                     'official_signed_error': before[3], 'trained_signed_error': after[3],
                     'delta_abs_error': delta, 'outcome': outcome})
    left, right = (counting_summary([{'signed_error': item[3]} for item in arm.values()])
                   for arm in (baseline, trained))
    summary = {'official': left, 'warm_start': right,
               'warm_start_minus_official': {key: right[key] - left[key] for key in METRICS},
               'per_image': counts, 'tie_tolerance': 1e-9,
               'validation_gate_met': right['mae'] < left['mae'] and right['rmse'] <= left['rmse']}
    return rows, summary


def train(spec, config, output):
    report = read_json(output / REPORT)
    if report.get('status') != 'passed':
        raise ValueError('Complete the official baseline and GPU preflight first')
    inputs = verify_inputs(spec, config)
    if make_guard(spec, config, inputs) != report['guard']:
        raise ValueError('Preflight input, source or runtime identity changed')
    if sha256(output / BASELINE_CSV) != report['baseline_csv_sha256']:
        raise ValueError('Frozen validation predictions changed after preflight')
    free, _ = torch.cuda.mem_get_info()
    if free < report['smoke']['training_estimate_bytes']:
        raise RuntimeError('Available VRAM fell below the preflight estimate')
    directory = new_output(output / 'warm_start')
    trainer = PilotTrainer(config, directory)
    trainer.setup()
    starting = state_fingerprint(trainer.model)
    if starting != report['baseline']['starting_weights_sha256']:
        raise ValueError('Warm start differs from the frozen official weights')
    write_json(directory / 'manifest.json', {'guard': report['guard'],
               'starting_weights_sha256': starting, 'baseline_csv_sha256': report['baseline_csv_sha256']})
    trainer.train()
    if trainer.optimizer_steps != 1000:
        raise ValueError('Expected exactly 1,000 optimizer updates')
    rows, summary = paired_validation(output / BASELINE_CSV, directory / 'val_epoch0.csv',
                                      spec['rp010b_data']['split_counts']['val'])
    base.write_csv(output / 'paired_validation.csv', rows)
    summary['optimizer_steps'] = trainer.optimizer_steps
    write_json(output / 'paired_validation.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('verify', 'preflight', 'train'))
    parser.add_argument('--asset-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-dir')
    parser.add_argument('--prompt-bank')
    parser.add_argument('--allow-training', action='store_true')
    args = parser.parse_args(argv)
    spec, config, output = plan(args.asset_root, args.checkpoint, args.output_dir,
                                args.data_dir, args.prompt_bank)
    if args.mode == 'verify':
        inputs = verify_inputs(spec, config)
        result = {'checkpoint_sha256': inputs['checkpoint_sha256'],
                  'anchors': len(inputs['data']['selected_images']),
                  'split_counts': inputs['data']['split_counts'],
                  'expected_optimizer_steps': 1000, 'config': config}
    elif args.mode == 'preflight':
        report = preflight(spec, config, output)
        result = {key: value for key, value in report.items() if key != 'guard'}
    else:
        if not args.allow_training:
            parser.error('train requires --allow-training after preflight review')
        result = train(spec, config, output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
