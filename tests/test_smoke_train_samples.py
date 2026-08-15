import unittest
from unittest import mock

from torch.utils.data import Subset

from train import parse_arg
from utils.regression_trainer import apply_smoke_train_sample_limit


def _datasets():
    return {
        'train': ['train-0', 'train-1', 'train-2', 'train-3'],
        'val': ['val-0', 'val-1'],
        'test': ['test-0', 'test-1', 'test-2'],
    }


class SmokeTrainSampleLimitTests(unittest.TestCase):
    def test_default_zero_does_not_limit_training_dataset(self):
        datasets = _datasets()
        original_train = datasets['train']
        with mock.patch('sys.argv', ['train.py']):
            args = parse_arg()

        self.assertEqual(args.smoke_train_samples, 0)
        result = apply_smoke_train_sample_limit(
            datasets, args.smoke_train_samples
        )

        self.assertIs(result, datasets)
        self.assertIs(datasets['train'], original_train)

    def test_positive_limit_wraps_only_first_n_training_samples(self):
        datasets = _datasets()
        original_train = datasets['train']

        apply_smoke_train_sample_limit(datasets, 2)

        self.assertIsInstance(datasets['train'], Subset)
        self.assertIs(datasets['train'].dataset, original_train)
        self.assertEqual(list(datasets['train'].indices), [0, 1])
        self.assertEqual(
            [datasets['train'][index] for index in range(2)],
            ['train-0', 'train-1'],
        )

    def test_limit_larger_than_training_dataset_is_safe(self):
        datasets = _datasets()

        apply_smoke_train_sample_limit(datasets, 100)

        self.assertIsInstance(datasets['train'], Subset)
        self.assertEqual(len(datasets['train']), 4)
        self.assertEqual(list(datasets['train'].indices), [0, 1, 2, 3])

    def test_validation_and_test_datasets_are_unchanged(self):
        datasets = _datasets()
        original_val = datasets['val']
        original_test = datasets['test']

        apply_smoke_train_sample_limit(datasets, 1)

        self.assertIs(datasets['val'], original_val)
        self.assertIs(datasets['test'], original_test)


if __name__ == '__main__':
    unittest.main()
