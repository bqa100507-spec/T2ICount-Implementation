#!/usr/bin/env python
"""Validate a Phase 2B training prompt bank without loading a model or GPU."""

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.rich_prompt_training import (
    RichPromptTrainingError,
    load_fsc147_train_metadata,
    load_rich_prompt_bank,
)
from utils.train_subset import select_train_subset_indices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Statically validate an FSC147 Phase 2B prompt bank.'
    )
    parser.add_argument('--rich-prompt-bank', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--train-samples', required=True, type=int)
    parser.add_argument('--train-subset-seed', default=3407, type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        train_names, classes = load_fsc147_train_metadata(args.data_dir)
        indices = select_train_subset_indices(
            len(train_names), args.train_samples, args.train_subset_seed
        )
        selected_names = [train_names[index] for index in indices]
        bank = load_rich_prompt_bank(
            args.rich_prompt_bank,
            train_samples=args.train_samples,
            train_subset_seed=args.train_subset_seed,
            selected_image_names=selected_names,
            class_by_image=classes,
        )
    except (RichPromptTrainingError, ValueError) as exc:
        print('ERROR: {}'.format(exc), file=sys.stderr)
        return 1

    print('Rich prompt bank validation passed.')
    print('selected_sample_count={}'.format(len(selected_names)))
    print(
        'selected_image_fingerprint={}'.format(
            bank.selected_image_fingerprint
        )
    )
    print('prompt_bank_fingerprint={}'.format(bank.file_fingerprint))
    print('protocol_version={}'.format(bank.protocol_version))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
