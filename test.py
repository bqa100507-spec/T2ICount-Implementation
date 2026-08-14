import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from datasets.carpk import CARPK
from datasets.dataset import IDCIA, ObjectCount
from models.build import build_t2icount
from utils.inference import (
    build_prompt_attention_mask,
    predict_count,
    prepare_image_patches,
)
from utils.paths import (
    AssetPaths,
    require_file,
    resolve_required_directory,
    resolve_required_file,
)


IDCIA_PROMPTS = {
    'DAPI': 'cell nuclei',
    'TuJ1': 'immature neurons',
    'MAP2ab': 'maturing neurons',
    'RIP': 'oligodendrocytes',
    'GFAP': 'astrocytes',
    'Nestin': 'neural stem cells',
    'Ki67': 'proliferating cells',
}
IDCIA_STAINING_ORDER = tuple(IDCIA_PROMPTS.keys())


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-root', default=None,
                        help='External asset root; overrides T2ICOUNT_ASSET_ROOT.')
    parser.add_argument('--model-path', default=None,
                        help='T2ICount checkpoint; defaults to checkpoints/official under the asset root.')
    parser.add_argument('--clip-path', default=None,
                        help='Local CLIP directory; defaults to pretrained/clip-vit-large-patch14.')
    parser.add_argument('--sd-path', default=None,
                        help='Local SD v1.5 checkpoint; defaults under the asset root.')
    parser.add_argument('--config', default='configs/v1-inference.yaml',
                        help='Stable Diffusion YAML config stored in this repository.')
    parser.add_argument('--data', default='carpk', choices=['carpk', 'fsc147', 'idcia'])
    parser.add_argument('--dataset-root', default=None,
                        help='Selected dataset directory; defaults under the asset root.')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--prompt-mode', default='both', choices=['generic', 'specific', 'both'],
                        help='IDCIA prompt strategy; ignored for FSC-147 and CARPK.')
    parser.add_argument('--idcia-root', default=None, type=str,
                        help='Deprecated IDCIA-specific alias for --dataset-root.')
    parser.add_argument(
        '--idcia-preprocess',
        default='raw',
        choices=['raw', 'autocontrast'],
        help='IDCIA image preprocessing; ignored for FSC-147 and CARPK.'
    )
    parser.add_argument('--results-path', default=None, type=str,
                        help='IDCIA CSV path; defaults under the asset output directory.')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--max-samples', default=None, type=int,
                        help='Optional smoke-test limit applied to the selected dataset.')
    return parser.parse_args()


def evaluate_legacy(model, data, batch_size, crop_size, device):
    dataloader = torch.utils.data.DataLoader(data, batch_size=1)
    epoch_res = []
    wrong_percent_list = []

    for step, (img, gt_counts, prompts, prompt_attn_mask, name) in enumerate(dataloader):
        inputs = img.to(device)
        pred_count = predict_count(
            model,
            inputs,
            prompts[0],
            prompt_attn_mask,
            batch_size=batch_size,
            patch_size=crop_size,
        )
        gt_count = gt_counts[0].item()
        wrong_percent = abs(gt_count - pred_count) / gt_count * 100

        print(
            f'[{step + 1}/{len(dataloader)}] '
            f'{name[0]} | prompt={prompts[0]} | '
            f'GT={gt_count} | '
            f'Pred={pred_count:.2f}'
            f' | Wrong Percent={wrong_percent:.2f}%'
        )
        epoch_res.append(gt_count - pred_count)
        wrong_percent_list.append(wrong_percent)

    epoch_res = np.array(epoch_res)
    mse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))
    avg_wrong_percent = np.mean(wrong_percent_list)
    print(f'Test, MAE:{mae}, MSE:{mse}, Avg Wrong Percent:{avg_wrong_percent:.2f}%')


def compute_metrics(records, prediction_key):
    gt = np.array([record['gt'] for record in records], dtype=np.float64)
    predictions = np.array([record[prediction_key] for record in records], dtype=np.float64)
    absolute_errors = np.abs(predictions - gt)
    positive_gt = gt > 0
    ape = absolute_errors[positive_gt] / gt[positive_gt] * 100
    gt_sum = np.sum(gt)

    return {
        'mae': np.mean(absolute_errors),
        'rmse': np.sqrt(np.mean(np.square(predictions - gt))),
        'avg_wrong_percent': np.mean(ape) if ape.size else np.nan,
        'wape': np.sum(absolute_errors) / gt_sum * 100 if gt_sum > 0 else np.nan,
        'median_ape': np.median(ape) if ape.size else np.nan,
        'excluded_zero_gt': int(np.sum(~positive_gt)),
    }


def print_metric_summary(title, metrics):
    print(f'\n{title}')
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Avg Wrong Percent (GT > 0): {metrics['avg_wrong_percent']:.2f}%")
    print(f"GT=0 samples excluded from percentage metric: {metrics['excluded_zero_gt']}")
    print(f"WAPE: {metrics['wape']:.2f}%")
    print(f"Median APE (GT > 0): {metrics['median_ape']:.2f}%")


def print_staining_results(records, prompt_mode):
    print('\nIDCIA per-staining results')
    if prompt_mode == 'both':
        print('Staining | N | Mean GT | Generic MAE | Specific MAE')
    elif prompt_mode == 'generic':
        print('Staining | N | Mean GT | Generic MAE')
    else:
        print('Staining | N | Mean GT | Specific MAE')

    for staining in IDCIA_STAINING_ORDER:
        staining_records = [record for record in records if record['staining'] == staining]
        if not staining_records:
            continue
        mean_gt = np.mean([record['gt'] for record in staining_records])
        values = [staining, str(len(staining_records)), f'{mean_gt:.2f}']
        if prompt_mode in ('generic', 'both'):
            generic_metrics = compute_metrics(staining_records, 'pred_generic')
            values.append(f"{generic_metrics['mae']:.4f}")
        if prompt_mode in ('specific', 'both'):
            specific_metrics = compute_metrics(staining_records, 'pred_specific')
            values.append(f"{specific_metrics['mae']:.4f}")
        print(' | '.join(values))

    if prompt_mode == 'both':
        generic_mae = compute_metrics(records, 'pred_generic')['mae']
        specific_mae = compute_metrics(records, 'pred_specific')['mae']
        print(f'\nSpecific MAE improvement: {generic_mae - specific_mae:.6f}')
        print('(Positive means the staining-specific prompt performs better.)')


def save_idcia_results(records, prompt_mode, results_path):
    fields = ['image', 'staining', 'gt']
    if prompt_mode in ('generic', 'both'):
        fields.extend([
            'generic_prompt', 'pred_generic', 'abs_error_generic', 'ape_generic'
        ])
    if prompt_mode in ('specific', 'both'):
        fields.extend([
            'specific_prompt', 'pred_specific', 'abs_error_specific', 'ape_specific'
        ])

    output_path = Path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in ('image', 'staining', 'gt')}
            if prompt_mode in ('generic', 'both'):
                generic_error = abs(record['pred_generic'] - record['gt'])
                row.update({
                    'generic_prompt': record['generic_prompt'],
                    'pred_generic': f"{record['pred_generic']:.6f}",
                    'abs_error_generic': f'{generic_error:.6f}',
                    'ape_generic': '' if record['gt'] == 0 else f'{generic_error / record["gt"] * 100:.6f}',
                })
            if prompt_mode in ('specific', 'both'):
                specific_error = abs(record['pred_specific'] - record['gt'])
                row.update({
                    'specific_prompt': record['specific_prompt'],
                    'pred_specific': f"{record['pred_specific']:.6f}",
                    'abs_error_specific': f'{specific_error:.6f}',
                    'ape_specific': '' if record['gt'] == 0 else f'{specific_error / record["gt"] * 100:.6f}',
                })
            writer.writerow(row)
    return output_path


def evaluate_idcia(model, data, tokenizer, prompt_mode, batch_size, crop_size, device, results_path):
    dataloader = torch.utils.data.DataLoader(data, batch_size=1)
    prompts = {'cell'}
    if prompt_mode in ('specific', 'both'):
        prompts.update(IDCIA_PROMPTS.values())
    prompt_masks = {
        prompt: build_prompt_attention_mask(tokenizer, prompt)
        for prompt in prompts
    }
    records = []

    for step, (img, gt_counts, image_names, stainings) in enumerate(dataloader):
        inputs = img.to(device)
        patch_info = prepare_image_patches(inputs, crop_size)
        gt_count = gt_counts[0].item()
        image_name = image_names[0]
        staining = stainings[0]
        record = {'image': image_name, 'staining': staining, 'gt': gt_count}
        progress = [
            f'[{step + 1}/{len(dataloader)}] {image_name}',
            f'staining={staining}',
            f'GT={gt_count}',
        ]

        if prompt_mode in ('generic', 'both'):
            generic_prompt = 'cell'
            record['generic_prompt'] = generic_prompt
            record['pred_generic'] = predict_count(
                model, inputs, generic_prompt, prompt_masks[generic_prompt],
                batch_size=batch_size, patch_size=crop_size,
                prepared_patches=patch_info,
            )
            progress.append(f'{generic_prompt}={record["pred_generic"]:.2f}')

        if prompt_mode in ('specific', 'both'):
            specific_prompt = IDCIA_PROMPTS[staining]
            record['specific_prompt'] = specific_prompt
            record['pred_specific'] = predict_count(
                model, inputs, specific_prompt, prompt_masks[specific_prompt],
                batch_size=batch_size, patch_size=crop_size,
                prepared_patches=patch_info,
            )
            progress.append(f'{specific_prompt}={record["pred_specific"]:.2f}')

        print(' | '.join(progress))
        records.append(record)

    if prompt_mode in ('generic', 'both'):
        print_metric_summary(
            'IDCIA - Generic prompt ("cell")',
            compute_metrics(records, 'pred_generic'),
        )
    if prompt_mode in ('specific', 'both'):
        print_metric_summary(
            'IDCIA - Specific prompts',
            compute_metrics(records, 'pred_specific'),
        )
    print_staining_results(records, prompt_mode)
    output_path = save_idcia_results(records, prompt_mode, results_path)
    print(f'\nPer-image predictions saved to: {output_path}')


def resolve_cli_paths(args):
    assets = AssetPaths.from_sources(args.asset_root, required=False)
    clip_path = resolve_required_directory(
        args.clip_path, assets.clip_dir if assets else None, 'CLIP model'
    )
    sd_path = resolve_required_file(
        args.sd_path, assets.sd_checkpoint if assets else None,
        'Stable Diffusion checkpoint'
    )
    model_path = resolve_required_file(
        args.model_path, assets.official_checkpoint if assets else None,
        'T2ICount checkpoint'
    )
    dataset_override = args.idcia_root if args.data == 'idcia' and args.idcia_root else args.dataset_root
    dataset_root = resolve_required_directory(
        dataset_override, assets.dataset_dir(args.data) if assets else None,
        '{} dataset'.format(args.data.upper())
    )
    config_path = require_file(args.config, 'Stable Diffusion config')
    if args.results_path:
        results_path = Path(args.results_path).expanduser().resolve()
    elif assets:
        results_path = assets.output_dir / 'idcia_predictions.csv'
    else:
        results_path = Path('results/idcia_predictions.csv').resolve()
    return config_path, sd_path, clip_path, model_path, dataset_root, results_path


def main():
    args = parse_arg()
    if args.batch_size < 1:
        raise ValueError('--batch-size must be at least 1.')
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError('--max-samples must be at least 1.')

    crop_size = 384
    device = torch.device(args.device)
    config_path, sd_path, clip_path, model_path, dataset_root, results_path = resolve_cli_paths(args)

    model = build_t2icount(
        config_path,
        sd_path,
        clip_path,
        checkpoint_path=model_path,
        device=device,
        mode='eval',
    )
    tokenizer = model.clip.tokenizer

    if args.data == 'carpk':
        data = CARPK(
            dataset_root,
            dataset_root / 'ImageSets' / 'test.txt',
            tokenizer=tokenizer,
        )
    elif args.data == 'fsc147':
        data = ObjectCount(
            dataset_root, crop_size, 8, 'test', 224, tokenizer=tokenizer
        )
    else:
        data = IDCIA(
            dataset_root,
            split='test',
            preprocess=args.idcia_preprocess,
        )

    if args.max_samples is not None:
        data = torch.utils.data.Subset(data, range(min(args.max_samples, len(data))))

    if args.data == 'idcia':
        evaluate_idcia(
            model,
            data,
            tokenizer,
            args.prompt_mode,
            args.batch_size,
            crop_size,
            device,
            results_path,
        )
    else:
        evaluate_legacy(model, data, args.batch_size, crop_size, device)


if __name__ == '__main__':
    main()
