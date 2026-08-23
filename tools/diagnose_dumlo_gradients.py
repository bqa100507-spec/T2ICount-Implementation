"""Measure DUMLO output-gradient contributions without updating the model."""

import argparse
import math
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

GRADIENT_STAT_NAMES = (
    'l2',
    'rms',
    'mean_abs',
    'max_abs',
    'fraction_positive',
    'fraction_negative',
)


def gradient_statistics(gradient):
    """Return scalar descriptive statistics for one output gradient."""
    detached = gradient.detach()
    squared = detached * detached
    return {
        'l2': torch.sqrt(squared.sum()).item(),
        'rms': torch.sqrt(squared.mean()).item(),
        'mean_abs': detached.abs().mean().item(),
        'max_abs': detached.abs().max().item(),
        'fraction_positive': (detached > 0).to(detached.dtype).mean().item(),
        'fraction_negative': (detached < 0).to(detached.dtype).mean().item(),
    }


def gradient_cosine(first, second):
    """Return cosine similarity, or NaN when either gradient is zero."""
    first_flat = first.detach().reshape(-1)
    second_flat = second.detach().reshape(-1)
    denominator = torch.linalg.vector_norm(first_flat) * torch.linalg.vector_norm(
        second_flat
    )
    if denominator.item() == 0.0:
        return float('nan')
    return torch.dot(first_flat, second_flat).div(denominator).item()


def gradient_cancellation_ratio(component_gradients):
    """Return total L2 divided by the sum of component L2 norms."""
    gradients = tuple(component_gradients)
    total = torch.stack(gradients).sum(dim=0)
    denominator = sum(torch.linalg.vector_norm(item).item() for item in gradients)
    if denominator == 0.0:
        return float('nan')
    return torch.linalg.vector_norm(total).item() / denominator


def expected_count_gradient_abs(lambda_count):
    """Expected non-kink Count-gradient magnitude for pred_den / 60."""
    return abs(float(lambda_count)) / 60.0


def construct_weighted_components(dumlo_diagnostics, dumlo_loss, gt_count):
    """Construct the exact batch-1 weighted terms from DUMLO graph tensors."""
    return {
        'count': (
            dumlo_loss.lambda_count * dumlo_diagnostics['count_loss']
        ),
        'ot': dumlo_loss.lambda_ot * dumlo_diagnostics['ot_loss'],
        'tv': (
            dumlo_loss.lambda_tv
            * float(gt_count)
            * dumlo_diagnostics['tv_loss']
        ),
    }


def component_output_gradients(weighted_components, pred_den):
    """Differentiate each weighted component only with respect to pred_den."""
    count_gradient = torch.autograd.grad(
        weighted_components['count'], pred_den, retain_graph=True
    )[0]
    ot_gradient = torch.autograd.grad(
        weighted_components['ot'], pred_den, retain_graph=True
    )[0]
    tv_gradient = torch.autograd.grad(
        weighted_components['tv'], pred_den
    )[0]
    return {
        'count': count_gradient,
        'ot': ot_gradient,
        'tv': tv_gradient,
    }


def aggregate_scalars(records):
    """Compute population mean/std for each finite scalar metric."""
    summaries = {}
    for name in sorted(records[0]):
        finite_values = [
            float(record[name]) for record in records
            if math.isfinite(float(record[name]))
        ]
        if not finite_values:
            summaries[name] = {
                'mean': float('nan'),
                'std': float('nan'),
                'valid': 0,
            }
            continue
        values = torch.tensor(finite_values, dtype=torch.float64)
        summaries[name] = {
            'mean': values.mean().item(),
            'std': values.std(unbiased=False).item(),
            'valid': len(finite_values),
        }
    return summaries


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Diagnose weighted DUMLO Count/OT/TV gradients with respect to '
            'the T2ICount density output. No model updates are performed.'
        )
    )
    parser.add_argument('--asset-root', default=None)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-samples', default=1000, type=int)
    parser.add_argument('--train-subset-seed', default=3407, type=int)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--num-samples', default=8, type=int)
    parser.add_argument('--config', default='configs/v1-inference.yaml')
    parser.add_argument('--sd-path', default=None)
    parser.add_argument('--clip-path', default=None)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--crop-size', default=384, type=int)
    parser.add_argument('--concat-size', default=224, type=int)
    parser.add_argument('--downsample-ratio', default=8, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dumlo-lambda-count', default=1.0, type=float)
    parser.add_argument('--dumlo-lambda-ot', default=0.1, type=float)
    parser.add_argument('--dumlo-lambda-tv', default=0.01, type=float)
    parser.add_argument('--dumlo-epsilon', default=10.0, type=float)
    parser.add_argument('--dumlo-iters', default=100, type=int)
    parser.add_argument('--dumlo-aug-points', default=10, type=int)
    parser.add_argument('--dumlo-radius-factor', default=0.5, type=float)
    parser.add_argument('--dumlo-sampling-seed', default=3407, type=int)
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.train_samples <= 0:
        raise ValueError('--train-samples must be positive for this diagnostic.')
    if args.num_samples <= 0:
        raise ValueError('--num-samples must be positive.')
    if Path(args.checkpoint).suffix.casefold() != '.pth':
        raise ValueError('--checkpoint must be a model-only .pth checkpoint.')
    return args


def resolve_paths(args):
    from utils.paths import (
        AssetPaths,
        require_file,
        resolve_required_directory,
        resolve_required_file,
    )

    assets = AssetPaths.from_sources(args.asset_root, required=False)
    args.config = str(require_file(args.config, 'Stable Diffusion config'))
    args.checkpoint = str(require_file(args.checkpoint, 'model-only checkpoint'))
    args.sd_path = str(resolve_required_file(
        args.sd_path,
        assets.sd_checkpoint if assets else None,
        'Stable Diffusion checkpoint',
    ))
    args.clip_path = str(resolve_required_directory(
        args.clip_path,
        assets.clip_dir if assets else None,
        'CLIP model',
    ))
    args.data_dir = str(resolve_required_directory(
        args.data_dir,
        assets.dataset_dir('fsc147') if assets else None,
        'FSC147 dataset',
    ))
    return args


def _format_value(value):
    return '{:.9g}'.format(value)


def _print_gradient_stats(label, stats):
    print('{} {}'.format(
        label,
        ' '.join(
            '{}={}'.format(name, _format_value(stats[name]))
            for name in GRADIENT_STAT_NAMES
        ),
    ))


def _flatten_sample_metrics(component_stats, cosine_stats, cancellation_ratio,
                            expected_count_abs, count_error):
    record = {}
    for component, stats in component_stats.items():
        for name, value in stats.items():
            record['{}.{}'.format(component, name)] = value
    for name, value in cosine_stats.items():
        record['cosine.{}'.format(name)] = value
    record['cancellation_ratio'] = cancellation_ratio
    record['expected_count_grad_abs'] = expected_count_abs
    record['count_grad_sanity_abs_error'] = (
        abs(component_stats['count']['mean_abs'] - expected_count_abs)
        if count_error != 0.0 else float('nan')
    )
    return record


def run_diagnostics(args):
    from torch.utils.data import DataLoader, Subset

    from datasets.dataset import ObjectCount
    from losses.dumlo import DUMLOLoss
    from models.build import build_t2icount
    from utils.regression_trainer import (
        apply_train_sample_subset,
        compute_regression_loss,
        setup_seed,
        train_collate,
    )

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device requested, but CUDA is unavailable.')

    setup_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    model = build_t2icount(
        args.config,
        args.sd_path,
        args.clip_path,
        checkpoint_path=args.checkpoint,
        device=device,
        mode='train',
        unet_config={
            'base_size': args.crop_size,
            'max_attn_size': args.crop_size // args.downsample_ratio,
            'attn_selector': 'down_cross+up_cross',
        },
    )
    model.set_train()

    datasets = {
        'train': ObjectCount(
            args.data_dir,
            crop_size=args.crop_size,
            downsample_ratio=args.downsample_ratio,
            method='train',
            concat_size=args.concat_size,
            tokenizer=model.clip.tokenizer,
            return_points=True,
        )
    }
    apply_train_sample_subset(
        datasets, args.train_samples, args.train_subset_seed
    )
    selected_train_dataset = datasets['train']
    diagnostic_count = min(args.num_samples, len(selected_train_dataset))
    if diagnostic_count < args.num_samples:
        print(
            'Requested {} diagnostic samples; only {} are available.'.format(
                args.num_samples, diagnostic_count
            )
        )
    diagnostic_dataset = Subset(
        selected_train_dataset, range(diagnostic_count)
    )
    dataloader = DataLoader(
        diagnostic_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=train_collate,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    dumlo_loss = DUMLOLoss(
        lambda_count=args.dumlo_lambda_count,
        lambda_ot=args.dumlo_lambda_ot,
        lambda_tv=args.dumlo_lambda_tv,
        epsilon=args.dumlo_epsilon,
        num_iters=args.dumlo_iters,
        augmentation_points=args.dumlo_aug_points,
        radius_factor=args.dumlo_radius_factor,
        sampling_seed=args.dumlo_sampling_seed,
    )
    original_indices = list(selected_train_dataset.indices)
    print(
        'diagnostic_config batch_size=1 train_samples={} subset_seed={} '
        'seed={} num_samples={} sampling_seed={} device={}'.format(
            len(selected_train_dataset),
            args.train_subset_seed,
            args.seed,
            diagnostic_count,
            args.dumlo_sampling_seed,
            device,
        )
    )
    print('diagnostic_dataset_indices={}'.format(original_indices[:diagnostic_count]))

    # Model construction can consume RNG. Reset immediately before the fixed,
    # single-worker data/forward sequence so augmentations and stochastic model
    # operations are reproducible for the same CLI seeds.
    setup_seed(args.seed)
    records = []
    for step, batch in enumerate(dataloader):
        (inputs, den_maps, captions, prompt_attn_masks,
         _img_attn_masks, point_sets) = batch
        inputs = inputs.to(device)
        gt_den_maps = den_maps.to(device) * 60
        gt_prompt_attn_masks = (
            prompt_attn_masks.to(device).unsqueeze(2).unsqueeze(3)
        )

        pred_den, _sim_x2, _sim_x1, _fused_cross_attn = model(
            inputs, captions, gt_prompt_attn_masks
        )
        dumlo_total, dumlo_diagnostics = compute_regression_loss(
            'dumlo',
            pred_den,
            gt_den_maps,
            point_sets=point_sets,
            dumlo_loss=dumlo_loss,
            input_h=args.crop_size,
            input_w=args.crop_size,
            epoch=0,
            step=step,
        )
        gt_count = len(point_sets[0])
        weighted_components = construct_weighted_components(
            dumlo_diagnostics, dumlo_loss, gt_count
        )
        reconstructed_total = sum(weighted_components.values())
        if not torch.allclose(dumlo_total, reconstructed_total):
            raise RuntimeError(
                'Weighted diagnostic components do not reconstruct DUMLO loss.'
            )
        gradients = component_output_gradients(weighted_components, pred_den)
        gradients['total'] = sum(gradients.values())
        component_stats = {
            name: gradient_statistics(gradient)
            for name, gradient in gradients.items()
        }
        cosine_stats = {
            'count_ot': gradient_cosine(gradients['count'], gradients['ot']),
            'count_tv': gradient_cosine(gradients['count'], gradients['tv']),
            'ot_tv': gradient_cosine(gradients['ot'], gradients['tv']),
        }
        cancellation_ratio = gradient_cancellation_ratio((
            gradients['count'], gradients['ot'], gradients['tv']
        ))
        expected_count_abs = expected_count_gradient_abs(
            dumlo_loss.lambda_count
        )
        count_error = dumlo_diagnostics['mean_signed_count_error'].item()

        print(
            '\nsample={} dataset_index={} gt_count={} pred_count={} '
            'signed_count_error={}'.format(
                step,
                original_indices[step],
                gt_count,
                _format_value(dumlo_diagnostics['mean_pred_count'].item()),
                _format_value(count_error),
            )
        )
        print('raw_losses count={} ot={} tv={} total={}'.format(
            _format_value(dumlo_diagnostics['count_loss'].item()),
            _format_value(dumlo_diagnostics['ot_loss'].item()),
            _format_value(dumlo_diagnostics['tv_loss'].item()),
            _format_value(dumlo_total.item()),
        ))
        print('weighted_losses count={} ot={} tv={}'.format(
            _format_value(weighted_components['count'].item()),
            _format_value(weighted_components['ot'].item()),
            _format_value(weighted_components['tv'].item()),
        ))
        for component in ('count', 'ot', 'tv', 'total'):
            _print_gradient_stats(component, component_stats[component])
        print('cos(count,ot)={} cos(count,tv)={} cos(ot,tv)={}'.format(
            _format_value(cosine_stats['count_ot']),
            _format_value(cosine_stats['count_tv']),
            _format_value(cosine_stats['ot_tv']),
        ))
        print('cancellation_ratio={}'.format(_format_value(cancellation_ratio)))
        print(
            'count_gradient_sanity applicable={} '
            'expected_count_grad_abs={} measured_count_mean_abs={}'.format(
                'yes' if count_error != 0.0 else 'no',
                _format_value(expected_count_abs),
                _format_value(component_stats['count']['mean_abs']),
            )
        )
        records.append(_flatten_sample_metrics(
            component_stats,
            cosine_stats,
            cancellation_ratio,
            expected_count_abs,
            count_error,
        ))
        del pred_den, dumlo_total, dumlo_diagnostics, weighted_components

    parameters_with_grad = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if parameters_with_grad:
        raise RuntimeError(
            'Diagnostic unexpectedly accumulated model parameter gradients: {}'
            .format(', '.join(parameters_with_grad))
        )

    print('\naggregate population_mean_std samples={}'.format(len(records)))
    for name, summary in aggregate_scalars(records).items():
        print('{} mean={} std={} valid={}/{}'.format(
            name,
            _format_value(summary['mean']),
            _format_value(summary['std']),
            summary['valid'],
            len(records),
        ))
    print('model_parameter_gradients_accumulated=0')


def main(argv=None):
    args = resolve_paths(parse_args(argv))
    run_diagnostics(args)


if __name__ == '__main__':
    main()
