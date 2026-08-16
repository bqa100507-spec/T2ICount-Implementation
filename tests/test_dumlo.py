import unittest
from unittest import mock

import torch

from losses.dumlo import (
    DUMLOLoss,
    adaptive_augment_points,
    analytical_ot_loss,
    generate_discrete_map,
    prediction_grid_coordinates,
    trihorn,
)
from train import parse_arg
from utils.regression_trainer import compute_regression_loss, get_reg_loss


class DUMLOCoreTests(unittest.TestCase):
    def test_cli_defaults_leave_baseline_enabled(self):
        with mock.patch('sys.argv', ['train.py']):
            args = parse_arg()
        self.assertEqual(args.loss_mode, 'baseline')
        self.assertEqual(args.dumlo_lambda_ot, 0.1)
        self.assertEqual(args.dumlo_lambda_tv, 0.01)
        self.assertEqual(args.dumlo_epsilon, 10.0)
        self.assertEqual(args.dumlo_iters, 100)
        self.assertEqual(args.dumlo_aug_points, 10)
        self.assertEqual(args.dumlo_radius_factor, 0.5)
        self.assertEqual(args.dumlo_sampling_seed, 3407)

    def test_discrete_map_preserves_colliding_point_mass(self):
        points = torch.tensor([[1.0, 1.0], [1.5, 1.5], [7.9, 7.9]])
        result = generate_discrete_map(points, 2, 2, 8, 8)
        self.assertEqual(result.shape, (2, 2))
        self.assertAlmostEqual(result.sum().item(), 3.0)
        self.assertEqual(result[0, 0].item(), 2.0)

    def test_adaptive_sampling_is_private_and_deterministic(self):
        points = torch.tensor([[4.0, 4.0], [12.0, 12.0]])
        torch.manual_seed(123)
        state = torch.get_rng_state().clone()
        first = adaptive_augment_points(
            points, 16, 16, 3, 0.5, 3407, 2, 5, 1
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        second = adaptive_augment_points(
            points, 16, 16, 3, 0.5, 3407, 2, 5, 1
        )
        different = adaptive_augment_points(
            points, 16, 16, 3, 0.5, 3408, 2, 5, 1
        )
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))
        self.assertEqual(first.shape, (8, 2))
        self.assertTrue((first >= 0).all())
        self.assertTrue((first[:, 0] < 16).all())
        self.assertTrue((first[:, 1] < 16).all())
        self.assertTrue(torch.equal(first[[0, 4]], points))

    def test_single_point_sampling_uses_bounded_fallback(self):
        point = torch.tensor([[0.0, 0.0]])
        result = adaptive_augment_points(
            point, 32, 64, 20, 0.5, 1, 0, 0, 0
        )
        distances = torch.linalg.vector_norm(result[1:] - point, dim=1)
        self.assertTrue((distances <= 4.0 + 1e-5).all())

    def test_trihorn_dimensions_are_finite_and_detached(self):
        original = torch.tensor([[2.0, 2.0], [6.0, 6.0]], requires_grad=True)
        augmented = torch.tensor([
            [2.0, 2.0], [3.0, 2.0], [6.0, 6.0], [5.0, 6.0]
        ], requires_grad=True)
        grid = prediction_grid_coordinates(2, 2, 8, 8, 'cpu', torch.float32)
        prediction = torch.full((4,), 0.25, requires_grad=True)
        result = trihorn(
            original, augmented, prediction, grid,
            epsilon=10.0, num_iters=20
        )
        self.assertEqual(result.beta_hat.shape, (4,))
        self.assertEqual(result.z_tilde.shape, (4,))
        self.assertEqual(result.u.shape, (2,))
        self.assertEqual(result.u_hat.shape, (4,))
        for value in result:
            self.assertTrue(torch.isfinite(value).all())
            self.assertFalse(value.requires_grad)

    def test_analytical_ot_backward_matches_returned_gradient(self):
        pred_mass = torch.tensor(
            [[0.4, 0.1], [0.2, 0.3]], requires_grad=True
        )
        original = torch.tensor([[2.0, 2.0]])
        augmented = original.clone()
        grid = prediction_grid_coordinates(2, 2, 8, 8, 'cpu', torch.float32)
        loss, gradient, _ = analytical_ot_loss(
            pred_mass, original, augmented, grid,
            epsilon=10.0, num_iters=20
        )
        loss.backward()
        self.assertEqual(gradient.shape, (4,))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(pred_mass.grad).all())
        self.assertTrue(torch.allclose(
            pred_mass.grad.reshape(-1), gradient, atol=1e-6
        ))

    def test_prediction_perturbation_changes_ot_and_gradient(self):
        original = torch.tensor([[2.0, 2.0]])
        grid = prediction_grid_coordinates(2, 2, 8, 8, 'cpu', torch.float32)
        near = torch.tensor([[1.0, 0.01], [0.01, 0.01]], requires_grad=True)
        far = torch.tensor([[0.01, 0.01], [0.01, 1.0]], requires_grad=True)
        near_loss, near_grad, _ = analytical_ot_loss(
            near, original, original, grid, num_iters=30
        )
        far_loss, far_grad, _ = analytical_ot_loss(
            far, original, original, grid, num_iters=30
        )
        self.assertLess(near_loss.item(), far_loss.item())
        self.assertFalse(torch.allclose(near_grad, far_grad))

    def test_count_tv_and_total_composition(self):
        pred_den = torch.zeros(1, 1, 2, 2, requires_grad=True)
        with torch.no_grad():
            pred_den[0, 0, 0, 0] = 60.0
            pred_den[0, 0, 1, 1] = 60.0
        points = [torch.tensor([[1.0, 1.0], [7.0, 7.0]])]
        criterion = DUMLOLoss(
            lambda_ot=0.1, lambda_tv=0.01, num_iters=10,
            augmentation_points=0
        )
        total, diagnostics = criterion(pred_den, points, 8, 8)
        expected = (
            diagnostics['count_loss']
            + 0.1 * diagnostics['ot_loss']
            + 0.01 * 2 * diagnostics['tv_loss']
        )
        self.assertAlmostEqual(diagnostics['count_loss'].item(), 0.0)
        self.assertAlmostEqual(diagnostics['tv_loss'].item(), 0.0, places=6)
        self.assertTrue(torch.allclose(total, expected))
        total.backward()
        self.assertTrue(torch.isfinite(pred_den.grad).all())

    def test_count_and_tv_follow_predicted_distribution(self):
        pred_den = torch.zeros(1, 1, 2, 2)
        pred_den[0, 0, 1, 1] = 120.0
        criterion = DUMLOLoss(
            lambda_ot=0.0, lambda_tv=0.0, num_iters=5,
            augmentation_points=0
        )
        _, diagnostics = criterion(
            pred_den, [torch.tensor([[1.0, 1.0]])], 8, 8
        )
        self.assertAlmostEqual(diagnostics['count_loss'].item(), 1.0)
        self.assertAlmostEqual(diagnostics['tv_loss'].item(), 1.0)

    def test_zero_point_negative_has_only_absolute_count_loss(self):
        pred_den = torch.ones(1, 1, 2, 2, requires_grad=True)
        criterion = DUMLOLoss(num_iters=5, augmentation_points=0)
        total, diagnostics = criterion(
            pred_den, [torch.empty(0, 2)], 8, 8
        )
        expected_count = pred_den.sum() / 60.0
        self.assertTrue(torch.allclose(total, expected_count))
        self.assertEqual(diagnostics['ot_loss'].item(), 0.0)
        self.assertEqual(diagnostics['tv_loss'].item(), 0.0)
        self.assertEqual(diagnostics['mean_gt_count'].item(), 0.0)
        total.backward()
        self.assertTrue(torch.isfinite(pred_den.grad).all())

    def test_baseline_regression_path_is_unchanged(self):
        pred = torch.rand(1, 1, 8, 8)
        gt = torch.rand(1, 1, 8, 8)
        expected = get_reg_loss(pred, gt, threshold=1e-3 * 60)
        actual, diagnostics = compute_regression_loss(
            'baseline', pred, gt,
            point_sets=[torch.tensor([[1.0, 1.0]])],
            dumlo_loss=mock.Mock(side_effect=AssertionError('not called')),
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertIsNone(diagnostics)

    def test_dumlo_loss_has_no_model_parameters(self):
        self.assertEqual(list(DUMLOLoss().parameters()), [])


if __name__ == '__main__':
    unittest.main()
