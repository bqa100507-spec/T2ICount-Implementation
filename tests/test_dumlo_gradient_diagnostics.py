import math
import unittest

import torch

from losses.dumlo import DUMLOLoss
from tools.diagnose_dumlo_gradients import (
    aggregate_scalars,
    component_output_gradients,
    construct_weighted_components,
    expected_count_gradient_abs,
    gradient_cancellation_ratio,
    gradient_cosine,
    gradient_statistics,
    parse_args,
)


class DUMLOGradientDiagnosticTests(unittest.TestCase):
    def test_cli_diagnostic_defaults(self):
        args = parse_args(['--checkpoint', 'model.pth'])

        self.assertEqual(args.train_samples, 1000)
        self.assertEqual(args.train_subset_seed, 3407)
        self.assertEqual(args.seed, 3407)
        self.assertEqual(args.num_samples, 8)
        self.assertEqual(args.dumlo_lambda_count, 1.0)
        self.assertEqual(args.dumlo_lambda_ot, 0.1)
        self.assertEqual(args.dumlo_lambda_tv, 0.01)
        self.assertEqual(args.dumlo_sampling_seed, 3407)

    def test_gradient_statistics(self):
        gradient = torch.tensor([-2.0, 0.0, 1.0, 3.0])

        stats = gradient_statistics(gradient)

        self.assertAlmostEqual(stats['l2'], math.sqrt(14.0), places=6)
        self.assertAlmostEqual(stats['rms'], math.sqrt(3.5), places=6)
        self.assertAlmostEqual(stats['mean_abs'], 1.5)
        self.assertAlmostEqual(stats['max_abs'], 3.0)
        self.assertAlmostEqual(stats['fraction_positive'], 0.5)
        self.assertAlmostEqual(stats['fraction_negative'], 0.25)

    def test_cosines_and_cancellation(self):
        horizontal = torch.tensor([1.0, 0.0])
        vertical = torch.tensor([0.0, 2.0])
        opposite = torch.tensor([-1.0, 0.0])

        self.assertAlmostEqual(gradient_cosine(horizontal, vertical), 0.0)
        self.assertAlmostEqual(gradient_cosine(horizontal, opposite), -1.0)
        self.assertAlmostEqual(
            gradient_cancellation_ratio((horizontal, opposite, vertical)),
            0.5,
        )
        self.assertTrue(math.isnan(
            gradient_cosine(horizontal, torch.zeros(2))
        ))

    def test_aggregate_uses_population_std_and_ignores_nan(self):
        summary = aggregate_scalars([
            {'metric': 1.0, 'optional': float('nan')},
            {'metric': 3.0, 'optional': 5.0},
        ])

        self.assertEqual(summary['metric']['mean'], 2.0)
        self.assertEqual(summary['metric']['std'], 1.0)
        self.assertEqual(summary['metric']['valid'], 2)
        self.assertEqual(summary['optional']['mean'], 5.0)
        self.assertEqual(summary['optional']['std'], 0.0)
        self.assertEqual(summary['optional']['valid'], 1)

    def test_nonzero_count_error_gradient_matches_lambda_over_60(self):
        lambda_count = 0.3
        pred_den = torch.ones(1, 1, 2, 3, requires_grad=True)
        criterion = DUMLOLoss(
            lambda_count=lambda_count,
            num_iters=2,
            augmentation_points=0,
        )
        total, diagnostics = criterion(
            pred_den, [torch.empty(0, 2)], input_h=16, input_w=24
        )
        self.assertNotEqual(
            diagnostics['mean_signed_count_error'].item(), 0.0
        )

        weighted = construct_weighted_components(
            diagnostics, criterion, gt_count=0
        )
        gradients = component_output_gradients(weighted, pred_den)
        measured = gradient_statistics(gradients['count'])['mean_abs']

        self.assertAlmostEqual(
            measured,
            expected_count_gradient_abs(lambda_count),
            places=7,
        )
        self.assertTrue(torch.equal(total, sum(weighted.values())))
        self.assertIsNone(pred_den.grad)


if __name__ == '__main__':
    unittest.main()
