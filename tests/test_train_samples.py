import unittest
from unittest import mock

import torch
from torch.utils.data import Subset

from train import parse_arg
from utils.regression_trainer import apply_train_sample_subset


def _datasets(train_size=12):
    return {
        'train': ['train-{}'.format(index) for index in range(train_size)],
        'val': ['val-0', 'val-1'],
        'test': ['test-0', 'test-1', 'test-2'],
    }


class TrainSampleSubsetTests(unittest.TestCase):
    def test_default_disabled_uses_full_training_dataset(self):
        datasets = _datasets()
        original_train = datasets['train']
        with mock.patch('sys.argv', ['train.py']):
            args = parse_arg()

        self.assertEqual(args.train_samples, 0)
        self.assertEqual(args.train_subset_seed, 3407)
        result = apply_train_sample_subset(
            datasets, args.train_samples, args.train_subset_seed
        )

        self.assertIs(result, datasets)
        self.assertIs(datasets['train'], original_train)

    def test_same_size_and_seed_select_same_indices(self):
        first = _datasets()
        second = _datasets()

        apply_train_sample_subset(first, 5, 3407)
        apply_train_sample_subset(second, 5, 3407)

        self.assertIsInstance(first['train'], Subset)
        self.assertEqual(
            list(first['train'].indices), list(second['train'].indices)
        )
        self.assertEqual(len(set(first['train'].indices)), 5)

    def test_different_seeds_select_different_indices(self):
        first = _datasets()
        second = _datasets()

        apply_train_sample_subset(first, 5, 3407)
        apply_train_sample_subset(second, 5, 3408)

        self.assertNotEqual(
            list(first['train'].indices), list(second['train'].indices)
        )

    def test_oversized_limit_selects_each_training_sample_once(self):
        datasets = _datasets(train_size=4)

        apply_train_sample_subset(datasets, 100, 3407)

        self.assertIsInstance(datasets['train'], Subset)
        self.assertEqual(len(datasets['train']), 4)
        self.assertEqual(sorted(datasets['train'].indices), [0, 1, 2, 3])

    def test_validation_and_test_datasets_are_unchanged(self):
        datasets = _datasets()
        original_val = datasets['val']
        original_test = datasets['test']

        apply_train_sample_subset(datasets, 5, 3407)

        self.assertIs(datasets['val'], original_val)
        self.assertIs(datasets['test'], original_test)

    def test_selection_does_not_advance_global_torch_rng(self):
        datasets = _datasets()
        torch.manual_seed(123)
        expected_state = torch.get_rng_state().clone()

        apply_train_sample_subset(datasets, 5, 3407)

        self.assertTrue(torch.equal(torch.get_rng_state(), expected_state))

    def test_train_and_smoke_limits_are_mutually_exclusive(self):
        with mock.patch(
            'sys.argv',
            [
                'train.py',
                '--train-samples', '5',
                '--smoke-train-samples', '2',
            ],
        ):
            with self.assertRaisesRegex(
                ValueError,
                '--train-samples and --smoke-train-samples cannot be used '
                'simultaneously',
            ):
                parse_arg()


if __name__ == '__main__':
    unittest.main()
