"""CPU-only HNW math, production routing and legacy checkpoint compatibility."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from models.build import load_t2icount_checkpoint
from tools import train_rp010 as rp010_runner
from train import parse_arg
from utils.checkpoints import load_trusted_legacy_checkpoint
from utils.helper import SaveHandler
from utils.regression_trainer import RRC_loss, Reg_Trainer, _t2i_loss_components
from utils.rp010 import load_config


def _map(values, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype).reshape(1, 1, 1, -1)


def _original_loss(simi, ambiguous, positive):
    # Frozen pre-HNW implementation: default-off values AND gradients must match.
    pos = (1 - simi) * positive
    neg = torch.clamp(simi, min=0) * (ambiguous == 0) * (positive == 0)
    pos_num = positive.flatten(1).sum(dim=1)
    neg_num = ((ambiguous == 0) * (positive == 0)).flatten(1).sum(dim=1)
    loss = 2 * pos.flatten(1).sum(dim=1) / (pos_num + 1e-7) + neg.flatten(1).sum(dim=1) / (neg_num + 1e-7)
    return loss.mean()


def test_weighted_formula_uses_selected_weights_and_per_image_means():
    simi = torch.tensor([
        [-0.4, 0.2, 0.8, 0.5, -0.3, 0.9],
        [0.6, 0.9, -0.2, 0.4, 0.7, 0.1],
    ], dtype=torch.float64).reshape(2, 1, 1, 6)
    positive = torch.tensor([
        [1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0],
    ], dtype=torch.bool).reshape_as(simi)
    ambiguous = torch.tensor([
        [1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1],
    ], dtype=torch.bool).reshape_as(simi)
    # Image 1 negatives: .2, .8, -.3, .9. Image 2: .4 only.
    first = 2 * 1.4 / (1 + 1e-7) + (1.2 * .2 + 1.8 * .8 + 1.9 * .9) / (1.2 + 1.8 + 1 + 1.9 + 1e-7)
    second = 2 * (.4 + .1) / (2 + 1e-7) + 1.4 * .4 / (1.4 + 1e-7)
    torch.testing.assert_close(
        RRC_loss(simi, ambiguous, positive, hnw=True),
        torch.tensor((first + second) / 2, dtype=simi.dtype),
    )


def test_gradients_detach_both_weights_and_the_weighted_denominator():
    simi = _map([-.5, .2, .8, .4, -.3, .9]).requires_grad_()
    positive = _map([1, 0, 0, 0, 0, 0]).bool()
    ambiguous = _map([1, 0, 0, 1, 0, 0]).bool()
    RRC_loss(simi, ambiguous, positive, hnw=True).backward()
    denominator = 1.2 + 1.8 + 1 + 1.9 + 1e-7
    # The unchanged positive count + epsilon follows PyTorch's default FP32 promotion.
    positive_denominator = float(torch.tensor(1) + 1e-7)
    expected = _map([-2 / positive_denominator, 1.2 / denominator,
                     1.8 / denominator, 0, 0, 1.9 / denominator])
    torch.testing.assert_close(simi.grad, expected, rtol=1e-12, atol=1e-12)
    assert simi.grad.flatten()[2] > simi.grad.flatten()[1]


@pytest.mark.parametrize('hnw', [False, True])
@pytest.mark.parametrize('all_positive', [False, True])
def test_empty_negative_mask_is_finite_and_keeps_positive_gradient(hnw, all_positive):
    simi = _map([.2, .8, -.4]).requires_grad_()
    positive = torch.full_like(simi, all_positive, dtype=torch.bool)
    ambiguous = torch.ones_like(positive)
    loss = RRC_loss(simi, ambiguous, positive, hnw=hnw)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(simi.grad).all()
    expected = 2 * (1 - simi).sum() / (3 + 1e-7) if all_positive else simi.new_zeros(())
    torch.testing.assert_close(loss, expected)
    gradient = -2 / (3 + 1e-7) if all_positive else 0.
    torch.testing.assert_close(simi.grad, torch.full_like(simi, gradient))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_all_nonpositive_negatives_have_zero_loss_and_preserve_hinge_boundary(dtype):
    simi = _map([-.8, -.2, 0.], dtype=dtype).requires_grad_()
    empty = torch.zeros_like(simi, dtype=torch.bool)
    loss = RRC_loss(simi, empty, empty, hnw=True)
    reference = simi.detach().clone().requires_grad_()
    original = _original_loss(reference, empty, empty)
    loss.backward()
    original.backward()
    assert loss.item() == 0.
    # torch.clamp has its existing boundary derivative at exactly zero.
    torch.testing.assert_close(simi.grad, reference.grad, rtol=1e-7, atol=0)
    assert torch.equal(simi.grad.flatten()[:2], torch.zeros(2, dtype=dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_disabled_is_bitwise_original_including_gradients(dtype):
    generator = torch.Generator().manual_seed(3407)
    values = torch.rand(3, 1, 4, 5, generator=generator, dtype=dtype) * 2 - 1
    positive = torch.rand(values.shape, generator=generator) > .7
    ambiguous = torch.rand(values.shape, generator=generator) > .5
    baseline = values.clone().requires_grad_()
    expected = _original_loss(baseline, ambiguous, positive)
    expected.backward()
    for options in ({}, {'hnw': False}):
        candidate = values.clone().requires_grad_()
        actual = RRC_loss(candidate, ambiguous, positive, **options)
        actual.backward()
        assert torch.equal(actual, expected)
        assert torch.equal(candidate.grad, baseline.grad)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_negative_accumulation_is_finite(dtype):
    simi = torch.full((1, 1, 256, 256), .5, dtype=dtype, requires_grad=True)
    empty = torch.zeros_like(simi, dtype=torch.bool)
    positive = empty.clone()
    positive[..., 0, 0] = True
    loss = RRC_loss(simi, empty, positive, hnw=True)
    loss.backward()
    assert loss.item() == pytest.approx(1.5)
    assert torch.isfinite(simi.grad).all()


@pytest.mark.parametrize('enabled', [False, True])
def test_both_stages_receive_same_masks_and_keep_coefficients(enabled):
    gt = _map([.06, .059, 0., 0.])
    image_mask = _map([1., 1., 1., 0.])
    attention = _map([.9, .3, .299, .9])
    outputs = (gt.clone(), _map([.8, .7, .6, .5]),
               _map([.4, .3, .2, .1]), attention)
    with mock.patch('utils.regression_trainer.get_reg_loss', return_value=torch.tensor(2.)), \
            mock.patch('utils.regression_trainer.RRC_loss', wraps=RRC_loss) as criterion:
        parts = _t2i_loss_components(outputs, gt, image_mask, rrcl_hnw=enabled)
    assert criterion.call_count == 2
    for call, similarity in zip(criterion.call_args_list, outputs[1:3]):
        assert call.args[0] is similarity
        assert torch.equal(call.args[1], _map([1, 1, 0, 0]).bool())
        assert torch.equal(call.args[2], _map([1, 0, 0, 0]).bool())
        assert call.kwargs == {'hnw': enabled}
    torch.testing.assert_close(parts['total'], parts['reg'] + .01 * parts['rrc1'] + .01 * parts['rrc2'])


class _TinyOutputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.similarity = torch.nn.Parameter(torch.tensor([[[[.2, .8], [-.4, .5]]]]))

    def set_train(self):
        self.train()

    def forward(self, inputs, captions, mask):
        simi = self.similarity.expand(inputs.shape[0], -1, -1, -1)
        return simi.sigmoid(), simi, simi * .5, torch.zeros_like(simi)


@pytest.mark.parametrize('enabled', [False, True])
def test_flag_routes_to_all_supervised_prompt_branches_without_optimizer_step(enabled):
    trainer = Reg_Trainer.__new__(Reg_Trainer)
    trainer.args = SimpleNamespace(rrcl_hnw=enabled, rich_consistency_weight=0.,
                                   rich_loss_diagnostic_steps=0, rich_supervision_mode='rich')
    trainer.device = torch.device('cpu')
    trainer.model = _TinyOutputs()
    trainer.epoch = 0
    trainer._rich_diagnostic_steps_logged = 0
    trainer._optimize_batch = lambda loss: loss.backward()  # No optimizer or parameter update.
    rich = {'rich_compatible': torch.tensor([True]),
            'detailed_prompt': ('red apples',), 'generalized_prompt': ('red objects',),
            'detailed_prompt_attn_mask': torch.ones(1, 77),
            'generalized_prompt_attn_mask': torch.ones(1, 77)}
    trainer.dataloaders = {'train': [(torch.ones(1, 3, 2, 2), torch.zeros(1, 1, 2, 2),
                                    ('apples',), torch.ones(1, 77), torch.ones(1, 1, 2, 2), rich)]}
    before = trainer.model.similarity.detach().clone()
    with mock.patch('utils.regression_trainer.get_reg_loss', return_value=torch.tensor(2.)), \
            mock.patch('utils.regression_trainer.RRC_loss', wraps=RRC_loss) as criterion:
        trainer.train_epoch()
    assert criterion.call_count == 8  # Class batch, compatible class slice, two auxiliary branches.
    assert all(call.kwargs == {'hnw': enabled} for call in criterion.call_args_list)
    assert torch.isfinite(trainer.model.similarity.grad).all()
    assert torch.equal(before, trainer.model.similarity)


def test_cli_default_and_explicit_enable_disable():
    assert parse_arg([]).rrcl_hnw is False
    assert parse_arg(['--rrcl-hnw']).rrcl_hnw is True
    assert parse_arg(['--no-rrcl-hnw']).rrcl_hnw is False
    with pytest.raises(SystemExit):
        parse_arg(['--rrcl-hnw', '--no-rrcl-hnw'])


def _checkpoint_trainer(directory, enabled=False):
    trainer = Reg_Trainer.__new__(Reg_Trainer)
    trainer.args = SimpleNamespace(rrcl_hnw=enabled)
    trainer.model = torch.nn.Linear(2, 1)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
    trainer.save_dir = str(directory)
    trainer.save_list = SaveHandler(num=2)
    trainer.epoch = 0
    trainer.start_epoch = 0
    trainer.best_mae = 1.
    trainer.best_mse = 2.
    return trainer


@pytest.mark.parametrize('enabled', [False, True])
def test_checkpoint_persists_flag_and_rejects_cross_loss_resume(tmp_path, enabled):
    source = _checkpoint_trainer(tmp_path, enabled)
    source._save_training_checkpoint()
    path = tmp_path / '0_ckpt.tar'
    checkpoint = load_trusted_legacy_checkpoint(path, 'cpu')
    assert checkpoint['rrcl_hnw'] is enabled
    target = _checkpoint_trainer(tmp_path, enabled)
    target._load_training_checkpoint(path)
    assert target.start_epoch == 1
    for key, value in source.model.state_dict().items():
        assert torch.equal(value, target.model.state_dict()[key])
    other = _checkpoint_trainer(tmp_path, not enabled)
    before = copy.deepcopy(other.model.state_dict())
    with pytest.raises(ValueError, match='HNW configuration mismatch'):
        other._load_training_checkpoint(path)
    assert all(torch.equal(value, other.model.state_dict()[key]) for key, value in before.items())


def test_legacy_generic_checkpoint_without_flag_loads_only_with_original_loss(tmp_path):
    source = _checkpoint_trainer(tmp_path)
    path = tmp_path / 'legacy.tar'
    torch.save({'epoch': 7, 'model_state_dict': source.model.state_dict(),
                'optimizer_state_dict': source.optimizer.state_dict()}, path)
    target = _checkpoint_trainer(tmp_path)
    target._load_training_checkpoint(path)
    assert target.start_epoch == 8
    with pytest.raises(ValueError, match='HNW configuration mismatch'):
        _checkpoint_trainer(tmp_path, True)._load_training_checkpoint(path)


@pytest.mark.parametrize('arm', ['a', 'b'])
def test_legacy_rp010_checkpoint_schema_loads_without_hnw_metadata(tmp_path, arm):
    # Exercise the actual RP-010 loaders with the pre-HNW payload schema.
    # This fixture is a tiny CPU model, not a historical trained SD checkpoint.
    config = load_config(Path(__file__).resolve().parents[1] / ('configs/rp010' + arm + '.json'), tmp_path)
    source = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(source.parameters())
    guard = {'config': config, 'source': 'historical-source-fixture'}
    payload = {'rp010_guard': guard, 'epoch': 0, 'next_epoch': 1,
               'optimizer_steps': 1000, 'epoch_records': [{'epoch': 0, 'steps': 1000}],
               'model_state_dict': source.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'best_mae': 1., 'best_mse': 2., 'best_epoch': None, 'best_selection': None,
               'rng_state': {'torch': torch.get_rng_state()}, 'smoke_only': False}
    path = tmp_path / 'legacy_rp010.tar'
    torch.save(payload, path)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    trainer = rp010_runner.ControlledTrainer.__new__(rp010_runner.ControlledTrainer)
    trainer.args = rp010_runner.make_args(config, tmp_path)
    trainer.control, trainer.guard = config, guard
    trainer.model = torch.nn.Linear(2, 1)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
    trainer.save_dir = str(tmp_path)
    assert trainer.args.rrcl_hnw is False
    trainer.load_state(path)
    assert trainer.start_epoch == 1
    assert trainer.optimizer_steps == 1000
    assert all(torch.equal(value, trainer.model.state_dict()[key]) for key, value in source.state_dict().items())
    # Raw best weights and wrapped latest.tar remain accepted by the production loader.
    best = tmp_path / 'best_epoch0.pth'
    torch.save(source.state_dict(), best)
    for checkpoint_path in (best, path):
        target = torch.nn.Linear(2, 1)
        load_t2icount_checkpoint(target, checkpoint_path)
        assert all(torch.equal(value, target.state_dict()[key]) for key, value in source.state_dict().items())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    trainer.guard = dict(guard, source='changed-source')
    with pytest.raises(ValueError, match='Resume rejected'):
        trainer.load_state(path)
