import math
from typing import NamedTuple

import torch
import torch.nn as nn


class TrihornResult(NamedTuple):
    beta_hat: torch.Tensor
    z_tilde: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    u_hat: torch.Tensor
    v_hat: torch.Tensor
    transport_cost: torch.Tensor


def _validate_points(points, name):
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('{} must have shape [N, 2]'.format(name))


def generate_discrete_map(points, height, width, input_h, input_w,
                          device=None, dtype=torch.float32):
    """Project full-resolution [x, y] annotations onto a density grid."""
    if min(height, width, input_h, input_w) <= 0:
        raise ValueError('map and input dimensions must be positive')
    points = torch.as_tensor(points, dtype=dtype, device=device)
    if points.numel() == 0:
        points = points.reshape(0, 2)
    _validate_points(points, 'points')

    discrete = torch.zeros(height * width, dtype=dtype,
                           device=points.device)
    if points.shape[0] == 0:
        return discrete.reshape(height, width)

    x = torch.floor(points[:, 0] * width / float(input_w)).long()
    y = torch.floor(points[:, 1] * height / float(input_h)).long()
    x = x.clamp(0, width - 1)
    y = y.clamp(0, height - 1)
    indices = y * width + x
    discrete.scatter_add_(
        0, indices, torch.ones_like(indices, dtype=dtype)
    )
    return discrete.reshape(height, width)


def prediction_grid_coordinates(height, width, input_h, input_w, device,
                                dtype):
    """Return prediction-cell centers in full-resolution [x, y] units."""
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5)
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5)
    y = y * (float(input_h) / height)
    x = x * (float(input_w) / width)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)


def derive_sampling_seed(base_seed, epoch, step, sample_index):
    """Derive a stable private seed without reading or changing global RNG."""
    modulus = 2 ** 63 - 1
    return int(
        (int(base_seed)
         + 1000003 * int(epoch)
         + 10007 * int(step)
         + 97 * int(sample_index)) % modulus
    )


def adaptive_augment_points(points, input_h, input_w, additional_points,
                            radius_factor, base_seed, epoch, step,
                            sample_index):
    """Sample extra points uniformly in per-annotation adaptive disks.

    The paper's accessible text does not define the single-point bandwidth.
    With no neighbor, one quarter of the shorter crop side is used as a
    bounded surrogate nearest-neighbor distance before applying radius_factor.
    """
    if additional_points < 0:
        raise ValueError('additional_points must be non-negative')
    if radius_factor < 0:
        raise ValueError('radius_factor must be non-negative')
    if min(input_h, input_w) <= 0:
        raise ValueError('input dimensions must be positive')

    points = torch.as_tensor(points)
    if points.numel() == 0:
        return points.reshape(0, 2).clone()
    _validate_points(points, 'points')

    source_device = points.device
    source_dtype = points.dtype
    cpu_points = points.detach().to(device='cpu', dtype=torch.float32)
    count = cpu_points.shape[0]
    if count == 1:
        nearest = torch.full(
            (1,), min(input_h, input_w) / 4.0, dtype=torch.float32
        )
    else:
        distances = torch.cdist(cpu_points, cpu_points, p=2)
        distances.fill_diagonal_(float('inf'))
        nearest = distances.min(dim=1)[0]
    radii = radius_factor * nearest

    generator = torch.Generator(device='cpu')
    generator.manual_seed(derive_sampling_seed(
        base_seed, epoch, step, sample_index
    ))
    augmented = []
    for point, radius in zip(cpu_points, radii):
        augmented.append(point.unsqueeze(0))
        if additional_points == 0:
            continue
        angles = torch.rand(additional_points, generator=generator)
        angles = angles * (2.0 * math.pi)
        distances = torch.sqrt(torch.rand(
            additional_points, generator=generator
        )) * radius
        offsets = torch.stack((
            torch.cos(angles) * distances,
            torch.sin(angles) * distances,
        ), dim=1)
        sampled = point.unsqueeze(0) + offsets
        sampled[:, 0].clamp_(0.0, float(input_w) - 1e-6)
        sampled[:, 1].clamp_(0.0, float(input_h) - 1e-6)
        augmented.append(sampled)
    return torch.cat(augmented, dim=0).to(
        device=source_device, dtype=source_dtype
    )


def _kernel_matvec(kernel, vector, transpose=False):
    if transpose:
        return torch.matmul(kernel.transpose(0, 1), vector)
    return torch.matmul(kernel, vector)


def _transport_cost(left_points, right_points, kernel, left_scale,
                    right_scale, chunk_rows=64):
    """Compute <diag(left) K diag(right), C> without a transport plan."""
    total = kernel.new_zeros(())
    for start in range(0, left_points.shape[0], chunk_rows):
        end = min(start + chunk_rows, left_points.shape[0])
        cost_chunk = torch.cdist(
            left_points[start:end], right_points, p=2
        )
        total = total + (
            left_scale[start:end, None]
            * kernel[start:end]
            * right_scale[None, :]
            * cost_chunk
        ).sum()
    return total


def trihorn(original_points, augmented_points, normalized_prediction,
            grid_coordinates, epsilon=10.0, num_iters=100,
            numerical_eps=1e-8):
    """Run DUMLO Trihorn scaling with Gauss-Seidel updates."""
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    if num_iters <= 0:
        raise ValueError('num_iters must be positive')
    _validate_points(original_points, 'original_points')
    _validate_points(augmented_points, 'augmented_points')
    _validate_points(grid_coordinates, 'grid_coordinates')
    if original_points.shape[0] == 0:
        raise ValueError('Trihorn requires at least one original point')
    if augmented_points.shape[0] == 0:
        raise ValueError('Trihorn requires at least one augmented point')
    if normalized_prediction.ndim != 1:
        raise ValueError('normalized_prediction must have shape [M]')
    if grid_coordinates.shape[0] != normalized_prediction.shape[0]:
        raise ValueError('prediction and grid-coordinate dimensions differ')
    if original_points.device != augmented_points.device:
        raise ValueError('point tensors must be on the same device')
    if original_points.device != grid_coordinates.device:
        raise ValueError('points and grid coordinates must share a device')

    with torch.no_grad():
        original_points = original_points.detach()
        augmented_points = augmented_points.detach()
        normalized_prediction = normalized_prediction.detach()
        grid_coordinates = grid_coordinates.detach()
        # Convert costs to kernels in place. The transport-cost diagnostic
        # recomputes distance chunks later, avoiding a second retained Q x M
        # matrix and never materializing either transport plan.
        kernel = torch.cdist(original_points, augmented_points, p=2)
        kernel.div_(-epsilon).exp_()
        kernel_hat = torch.cdist(augmented_points, grid_coordinates, p=2)
        kernel_hat.div_(-epsilon).exp_()
        z = torch.full(
            (original_points.shape[0],),
            1.0 / original_points.shape[0],
            device=original_points.device,
            dtype=original_points.dtype,
        )
        u_hat = torch.ones(
            augmented_points.shape[0], device=original_points.device,
            dtype=original_points.dtype
        )

        for _ in range(num_iters):
            v = 1.0 / u_hat.clamp_min(numerical_eps)
            u = z / _kernel_matvec(kernel, v).clamp_min(numerical_eps)
            v_hat = normalized_prediction / _kernel_matvec(
                kernel_hat, u_hat, transpose=True
            ).clamp_min(numerical_eps)
            u_hat = (
                v * _kernel_matvec(kernel, u, transpose=True)
                / _kernel_matvec(kernel_hat, v_hat).clamp_min(numerical_eps)
            )

        v = 1.0 / u_hat.clamp_min(numerical_eps)
        u = z / _kernel_matvec(kernel, v).clamp_min(numerical_eps)
        z_tilde = v * _kernel_matvec(kernel, u, transpose=True)
        v_hat = normalized_prediction / _kernel_matvec(
            kernel_hat, u_hat, transpose=True
        ).clamp_min(numerical_eps)
        beta_hat = -epsilon * torch.log(v_hat.clamp_min(numerical_eps))
        transport_cost = _transport_cost(
            original_points, augmented_points, kernel, u, v
        )
        transport_cost = transport_cost + _transport_cost(
            augmented_points, grid_coordinates, kernel_hat, u_hat, v_hat
        )

    return TrihornResult(
        beta_hat=beta_hat,
        z_tilde=z_tilde,
        u=u,
        v=v,
        u_hat=u_hat,
        v_hat=v_hat,
        transport_cost=transport_cost,
    )


def analytical_ot_loss(pred_mass, original_points, augmented_points,
                       grid_coordinates, epsilon=10.0, num_iters=100,
                       numerical_eps=1e-8):
    """Return an OT surrogate with the paper's closed-form backward gradient."""
    pred_flat = pred_mass.reshape(-1)
    if pred_flat.shape[0] != grid_coordinates.shape[0]:
        raise ValueError('predicted mass and grid-coordinate dimensions differ')
    detached_mass = pred_flat.detach()
    normalized_prediction = (detached_mass + numerical_eps)
    normalized_prediction = normalized_prediction / normalized_prediction.sum()
    result = trihorn(
        original_points, augmented_points, normalized_prediction,
        grid_coordinates, epsilon=epsilon, num_iters=num_iters,
        numerical_eps=numerical_eps,
    )
    beta_hat = result.beta_hat.detach()
    mass_sum = detached_mass.sum().clamp_min(numerical_eps)
    gradient = (
        -beta_hat / mass_sum
        + (beta_hat * detached_mass).sum() / (mass_sum ** 2)
    )
    surrogate = (pred_flat * gradient.detach()).sum()
    ot_loss = (
        surrogate - surrogate.detach()
        + result.transport_cost.detach().to(pred_mass.dtype)
    )
    return ot_loss, gradient, result


class DUMLOLoss(nn.Module):
    def __init__(self, lambda_ot=0.1, lambda_tv=0.01, epsilon=10.0,
                 num_iters=100, augmentation_points=10,
                 radius_factor=0.5, sampling_seed=3407,
                 numerical_eps=1e-8, lambda_count=1.0):
        super(DUMLOLoss, self).__init__()
        self.lambda_count = lambda_count
        self.lambda_ot = lambda_ot
        self.lambda_tv = lambda_tv
        self.epsilon = epsilon
        self.num_iters = num_iters
        self.augmentation_points = augmentation_points
        self.radius_factor = radius_factor
        self.sampling_seed = sampling_seed
        self.numerical_eps = numerical_eps

    def forward(self, pred_den, point_sets, input_h, input_w, epoch=0,
                step=0):
        if pred_den.ndim != 4 or pred_den.shape[1] != 1:
            raise ValueError('pred_den must have shape [B, 1, H, W]')
        if len(point_sets) != pred_den.shape[0]:
            raise ValueError('one variable-length point tensor is required per image')
        height, width = pred_den.shape[-2:]
        grid_coordinates = prediction_grid_coordinates(
            height, width, input_h, input_w, pred_den.device,
            pred_den.dtype
        )
        totals = []
        count_losses = []
        ot_losses = []
        tv_losses = []
        gt_counts = []
        pred_counts = []
        signed_errors = []

        for sample_index, raw_points in enumerate(point_sets):
            points = torch.as_tensor(
                raw_points, device=pred_den.device, dtype=pred_den.dtype
            )
            if points.numel() == 0:
                points = points.reshape(0, 2)
            _validate_points(points, 'point_sets[{}]'.format(sample_index))
            gt_count = points.shape[0]
            pred_mass = pred_den[sample_index, 0] / 60.0
            pred_count = pred_mass.sum()

            # The paper's printed count/TV formulas use Z_tilde, but its
            # derivatives use Z_hat and the text follows DM-Count. Using
            # Z_tilde would also make count loss zero by mass conservation.
            count_loss = torch.abs(pred_count - float(gt_count))
            if gt_count == 0:
                ot_loss = pred_count * 0.0
                tv_loss = pred_count * 0.0
            else:
                discrete = generate_discrete_map(
                    points, height, width, input_h, input_w,
                    device=pred_den.device, dtype=pred_den.dtype
                )
                target_probability = discrete / float(gt_count)
                predicted_probability = pred_mass / (
                    pred_count + self.numerical_eps
                )
                tv_loss = 0.5 * torch.abs(
                    target_probability - predicted_probability
                ).sum()
                augmented_points = adaptive_augment_points(
                    points, input_h, input_w,
                    self.augmentation_points, self.radius_factor,
                    self.sampling_seed, epoch, step, sample_index,
                )
                ot_loss, _, _ = analytical_ot_loss(
                    pred_mass, points, augmented_points, grid_coordinates,
                    epsilon=self.epsilon, num_iters=self.num_iters,
                    numerical_eps=self.numerical_eps,
                )

            total = (
                self.lambda_count * count_loss
                + self.lambda_ot * ot_loss
                + self.lambda_tv * float(gt_count) * tv_loss
            )
            totals.append(total)
            count_losses.append(count_loss)
            ot_losses.append(ot_loss)
            tv_losses.append(tv_loss)
            gt_counts.append(pred_count.new_tensor(float(gt_count)))
            pred_counts.append(pred_count)
            signed_errors.append(pred_count - float(gt_count))

        diagnostics = {
            'count_loss': torch.stack(count_losses).mean(),
            'ot_loss': torch.stack(ot_losses).mean(),
            'tv_loss': torch.stack(tv_losses).mean(),
            'mean_gt_count': torch.stack(gt_counts).mean(),
            'mean_pred_count': torch.stack(pred_counts).mean(),
            'mean_signed_count_error': torch.stack(signed_errors).mean(),
        }
        return torch.stack(totals).mean(), diagnostics
