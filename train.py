import argparse
import torch
from utils.regression_trainer import Reg_Trainer, validate_train_sample_options
from utils.paths import (
    AssetPaths,
    require_file,
    resolve_required_directory,
    resolve_required_file,
)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-root', default=None,
                        help='External asset root; overrides T2ICOUNT_ASSET_ROOT.')
    parser.add_argument('--content', default="test", type=str,
                        help='what is it?')
    parser.add_argument('--seed', default=-1, type=int, help='if not using seed, please set as -1')
    parser.add_argument('--crop-size', default=384, type=int,
                        help='the cropped size of the training data')
    parser.add_argument('--concat-size', default=224, type=int,
                        help='the concat size of the training data')
    parser.add_argument('--downsample-ratio', default=8, type=int,
                        help='the downsample ratio of the model')
    parser.add_argument('--data-dir', default=None,
                        help='FSC-147 directory; defaults under the asset root')
    parser.add_argument('--config', default='configs/v1-inference.yaml',
                        help='the config of the ldm model')
    parser.add_argument('--sd-path', default=None,
                        help='local Stable Diffusion checkpoint; defaults under the asset root')
    parser.add_argument('--clip-path', default=None,
                        help='local CLIP directory; defaults under the asset root')

    parser.add_argument('--save-dir', default=None,
                        help='required writable directory for checkpoints/logs')

    parser.add_argument('--max-num', default=2, type=int,
                        help='the maximum number of saved models ')
    parser.add_argument('--resume', default="",
                        help='the path of the resume training model')
    parser.add_argument('--batch-size', default=4, type=int,
                        help='the number of samples in a batch')
    parser.add_argument('--smoke-train-samples', default=0, type=int,
                        help='limit only the training split for infrastructure smoke tests; 0 disables')
    parser.add_argument('--train-samples', default=0, type=int,
                        help='randomly subset the training split; 0 uses the full split')
    parser.add_argument('--train-subset-seed', default=3407, type=int,
                        help='seed for deterministic training subset selection')
    parser.add_argument('--loss-mode', choices=('baseline', 'dumlo'),
                        default='baseline',
                        help='opt-in training regression loss')
    parser.add_argument('--dumlo-lambda-count', default=1.0, type=float,
                        help='DUMLO count-loss weight.')
    parser.add_argument('--dumlo-lambda-ot', default=0.1, type=float,
                        help='DUMLO optimal-transport loss weight')
    parser.add_argument('--dumlo-lambda-tv', default=0.01, type=float,
                        help='DUMLO total-variation loss weight')
    parser.add_argument(
        '--dumlo-lambda-ssim', default=0.0, type=float,
        help=(
            'weight for adding only the original T2ICount SSIM component '
            'to DUMLO training'
        ),
    )
    parser.add_argument('--dumlo-epsilon', default=10.0, type=float,
                        help='DUMLO entropic regularization epsilon')
    parser.add_argument('--dumlo-iters', default=100, type=int,
                        help='number of DUMLO Trihorn iterations')
    parser.add_argument('--dumlo-aug-points', default=10, type=int,
                        help='additional adaptive samples per annotation')
    parser.add_argument('--dumlo-radius-factor', default=0.5, type=float,
                        help='nearest-neighbor sampling radius multiplier')
    parser.add_argument('--dumlo-sampling-seed', default=3407, type=int,
                        help='private deterministic DUMLO sampling seed')
    parser.add_argument('--stride', default=384, type=int,
                        help='the stride for patchify')
    parser.add_argument('--beta', default=1e-4, type=float,
                        help='the initialization value of beta')

    # Optimizer
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='the learning rate')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='the number of workers')

    parser.add_argument('--start-epoch', default=0, type=int,
                        help='the number of starting epoch')
    parser.add_argument('--epochs', default=300, type=int,
                        help='the maximum number of training epoch')
    parser.add_argument('--start-val', default=50, type=int,
                        help='the starting epoch for validation')
    parser.add_argument('--val-epoch', default=1, type=int,
                        help='the number of epoch between validation')

    args = parser.parse_args()
    validate_train_sample_options(args)
    return args


def resolve_training_paths(args):
    assets = AssetPaths.from_sources(args.asset_root, required=False)
    args.config = str(require_file(args.config, 'Stable Diffusion config'))
    args.sd_path = str(resolve_required_file(
        args.sd_path, assets.sd_checkpoint if assets else None,
        'Stable Diffusion checkpoint'
    ))
    args.clip_path = str(resolve_required_directory(
        args.clip_path, assets.clip_dir if assets else None, 'CLIP model'
    ))
    args.data_dir = str(resolve_required_directory(
        args.data_dir, assets.dataset_dir('fsc147') if assets else None,
        'FSC147 dataset'
    ))
    if args.save_dir is None:
        raise ValueError(
            '--save-dir is required; the runtime asset root is read-only.'
        )
    if args.resume:
        args.resume = str(require_file(args.resume, 'resume checkpoint'))
    return args


if __name__ == '__main__':
    args = resolve_training_paths(parse_arg())
    torch.backends.cudnn.benchmark = True
    trainer = Reg_Trainer(args)
    trainer.setup()
    trainer.train()
