"""CPU checks for RP-011; mocked smoke hardware is never a real GPU pass."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tools import train_rp011 as runner
from utils.regression_trainer import Reg_Trainer
from utils.rp010 import counting_summary, read_json, sha256, write_json


@pytest.fixture(autouse=True)
def quiet_file_logging(monkeypatch):
    monkeypatch.setattr('utils.trainer.logger', lambda path: None)


@pytest.fixture
def planned(tmp_path):
    return runner.plan(tmp_path / 'assets', tmp_path / 'RP-010B', tmp_path / 'RP-011')


def source_fixture(planned):
    spec, configs, run, output = planned
    run.mkdir()
    torch.save(TinyCount().state_dict(), run / spec['checkpoint'])
    spec['checkpoint_sha256'] = sha256(run / spec['checkpoint'])
    selection = {'epoch': 9, 'rule': 'val_mae_plus_rmse', 'checkpoint': spec['checkpoint'],
                 'sha256': spec['checkpoint_sha256']}
    manifest = {'guard': {'config': read_json(runner.ROOT / spec['base_config']), 'data': {}}}
    write_json(run / 'manifest.json', manifest)
    write_json(run / 'selection.json', selection)
    return manifest


class TinyCount(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = torch.nn.Conv2d(3, 1, 1)
        self.decoder = torch.nn.Conv2d(1, 1, 1)

    def set_train(self):
        self.train()

    def set_eval(self):
        self.eval()

    def forward(self, image, captions, mask):
        feat = self.unet(torch.nn.functional.adaptive_avg_pool2d(image, 8))
        return self.decoder(feat).sigmoid(), feat.tanh(), feat.tanh(), torch.zeros_like(feat)


def fake_setup(trainer):
    trainer.model = TinyCount()
    trainer.device = torch.device('cpu')
    c = trainer.control
    trainer.optimizer = torch.optim.AdamW([
        {'params': trainer.model.unet.parameters(), 'lr': c['lr'] * .1,
         'weight_decay': c['weight_decay'] * .1},
        {'params': trainer.model.decoder.parameters(), 'lr': c['lr'],
         'weight_decay': c['weight_decay']}])
    trainer.start_epoch = 0
    trainer.best_mae = trainer.best_mse = float('inf')
    trainer._rich_diagnostic_steps_logged = 0
    trainer.datasets = {'train': Samples()}
    trainer.dataloaders = {}


class Samples(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        mask = torch.zeros(77)
        mask[1] = 1
        density = torch.zeros(1, 8, 8)
        density[0, 3, 3] = 1
        return (torch.ones(3, 16, 16) * (index + 1), density, 'apples', mask,
                torch.ones(1, 8, 8), {'source_image_name': str(index), 'rich_compatible': True,
                'detailed_prompt': 'red apples', 'generalized_prompt': 'red objects',
                'detailed_prompt_attn_mask': mask, 'generalized_prompt_attn_mask': mask})


def test_effective_controls_and_runtime_arguments_differ_only_in_hnw(planned):
    spec, configs, run, output = planned
    left, right = configs.values()
    assert [key for key in left if left[key] != right[key]] == ['rrcl_hnw']
    assert left['epochs'] == 2 and left['start_val'] == 0 and left['val_epoch'] == 1
    assert left['scheduler'] is None and left['supervision_mode'] == 'rich'
    assert runner.steps_per_epoch(left) * left['epochs'] == 2000
    args = [runner.PilotTrainer(config, output / arm).args for arm, config in configs.items()]
    differences = {key for key in vars(args[0]) if getattr(args[0], key) != getattr(args[1], key)}
    assert differences == {'content', 'rrcl_hnw'}
    assert all(arg.resume == '' and arg.start_epoch == 0 for arg in args)


@pytest.mark.parametrize('relation', ['same', 'child', 'parent'])
def test_output_cannot_overlap_historical_run(tmp_path, relation):
    run = tmp_path / 'RP-010B'
    output = {'same': run, 'child': run / 'new', 'parent': tmp_path}[relation]
    with pytest.raises(ValueError, match='separate'):
        runner.plan(tmp_path, run, output)


def test_source_hash_and_historical_configuration_are_checked(planned):
    spec, configs, run, _ = planned
    manifest = source_fixture(planned)
    runner.verify_source(spec, configs, run)
    manifest['guard']['config']['lr'] = .01
    write_json(run / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='configuration differs: lr'):
        runner.verify_source(spec, configs, run)
    manifest['guard']['config']['lr'] = 5e-5
    write_json(run / 'manifest.json', manifest)
    with (run / spec['checkpoint']).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        runner.verify_source(spec, configs, run)


def test_selection_must_be_epoch_nine(planned):
    spec, configs, run, _ = planned
    source_fixture(planned)
    selection = read_json(run / 'selection.json')
    selection['epoch'] = 7
    write_json(run / 'selection.json', selection)
    with pytest.raises(ValueError, match='epoch-9'):
        runner.verify_source(spec, configs, run)


def test_both_weight_only_starts_are_equal_with_fresh_optimizer(planned):
    _, configs, run, output = planned
    source_fixture(planned)
    original = sha256(run / 'best_epoch9.pth')
    trainers = []
    with mock.patch.object(Reg_Trainer, 'setup', fake_setup):
        for arm, config in configs.items():
            trainer = runner.PilotTrainer(config, output / arm)
            trainer.setup()
            assert not trainer.optimizer.state and trainer.scheduler is None
            assert trainer.start_epoch == 0
            trainers.append(trainer)
    assert runner.state_fingerprint(trainers[0].model) == runner.state_fingerprint(trainers[1].model)
    assert sha256(run / 'best_epoch9.pth') == original
    # The real strict loader must reject an incompatible learned weight shape.
    state = trainers[0].model.state_dict()
    state['decoder.weight'] = torch.ones(5, 5, 1, 1)
    bad = output / 'bad.pth'
    torch.save(state, bad)
    with pytest.raises(RuntimeError, match='size mismatch'):
        runner.load_t2icount_checkpoint(trainers[0].model, bad)


def test_epoch_order_and_augmented_inputs_match_both_arms(planned):
    _, configs, _, output = planned
    pairs = [runner.PilotTrainer(config, output / arm) for arm, config in configs.items()]
    for trainer in pairs:
        trainer.datasets = {'train': Samples()}
    for epoch in (0, 1):
        data = [trainer.loader(epoch)[0] for trainer in pairs]
        assert data[0].order == data[1].order
        for i in range(len(data[0])):
            a, b = data[0][i], data[1][i]
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]
        assert data[0].trace.hexdigest() == data[1].trace.hexdigest()


def test_smoke_contract_runs_three_cpu_forwards_and_backward_without_updates(planned):
    _, configs, _, output = planned
    source_fixture(planned)
    reports = {}
    with mock.patch.object(Reg_Trainer, 'setup', fake_setup), \
            mock.patch('torch.cuda.reset_peak_memory_stats'), \
            mock.patch('torch.cuda.synchronize'), \
            mock.patch('torch.cuda.empty_cache'), \
            mock.patch('torch.cuda.mem_get_info', return_value=(10**9, 10**9)), \
            mock.patch('torch.cuda.max_memory_allocated', return_value=1000), \
            mock.patch('torch.cuda.max_memory_reserved', return_value=2000):
        for arm, config in configs.items():
            reports[arm] = runner.smoke(config, output / arm)
            assert reports[arm]['optimizer_steps'] == 0
            assert reports[arm]['gradient_l2'] > 0
            assert len(reports[arm]['prompts']) == 3
    runner.paired_smokes(reports)
    reports['hnw']['starting_weights_sha256'] = 'different'
    with pytest.raises(ValueError, match='starting_weights'):
        runner.paired_smokes(reports)


def test_missing_resources_record_pending_without_model_build(planned):
    spec, configs, run, output = planned
    source_fixture(planned)
    (run / spec['checkpoint']).unlink()
    with mock.patch.object(runner.base, 'host_free_bytes', return_value=2**30), \
            mock.patch('torch.cuda.is_available', return_value=False), \
            mock.patch.object(runner, 'verify_data', return_value={}), \
            mock.patch.object(runner, 'smoke') as smoke:
        result = runner.preflight(spec, configs, run, output)
    assert result['status'] == 'pending' and len(result['blockers']) == 3
    smoke.assert_not_called()
    assert read_json(output / 'preflight/report.json')['status'] == 'pending'
    with pytest.raises(ValueError, match='must pass'):
        runner.require_preflight(spec, configs, run, output)


def test_validation_reports_prediction_minus_gt_and_uses_class_inference(planned):
    _, configs, _, output = planned
    trainer = runner.PilotTrainer(configs['control'], output / 'control')
    trainer.model = TinyCount()
    trainer.device = torch.device('cpu')
    trainer.epoch = 0
    trainer.dataloaders = {'val': [
        (torch.ones(1, 3, 16, 16), torch.tensor([5]), ('apples',), torch.ones(1, 77), ('a',)),
        (torch.ones(1, 3, 16, 16), torch.tensor([3]), ('pears',), torch.ones(1, 77), ('b',))]}
    with mock.patch.object(runner, 'predict_count', side_effect=[3., 4.]) as predict:
        summary = trainer.validate()
    assert summary['mae'] == 1.5 and summary['bias'] == -.5
    assert summary['rmse'] == pytest.approx((5 / 2) ** .5)
    assert [call.args[2] for call in predict.call_args_list] == ['apples', 'pears']
    assert (output / 'control/val_epoch0.csv').is_file()


def test_two_epochs_validate_each_time_and_keep_mae_plus_rmse_selection(planned):
    _, configs, _, output = planned
    trainer = runner.PilotTrainer(configs['control'], output / 'control')
    trainer.model = TinyCount()
    trainer.dataloaders = {}
    trainer.best_mae = trainer.best_mse = float('inf')
    trainer.loader = mock.Mock(return_value=(SimpleNamespace(trace=hashlib.sha256(b'data'), compatible=500), []))
    trainer.train_epoch = lambda: setattr(trainer, 'optimizer_steps', trainer.optimizer_steps + 1000)
    trainer.validate = mock.Mock(side_effect=[{'mae': 3., 'rmse': 4., 'bias': -.5},
                                             {'mae': 2., 'rmse': 6., 'bias': 0.}])
    trainer.train()
    assert trainer.optimizer_steps == 2000 and trainer.validate.call_count == 2
    assert trainer.best_epoch == 0  # MAE alone cannot choose epoch 1.
    assert [item['epoch'] for item in trainer.records] == [0, 1]


def paired_outputs(output):
    common = {'data': {'split_counts': {'val': 2}}}
    for arm, errors in [('control', [2., -1.]), ('hnw', [1., -1.])]:
        directory = output / arm
        directory.mkdir(parents=True)
        write_json(directory / 'manifest.json', {'guard': common, 'starting_weights_sha256': 'same'})
        records = []
        for epoch in (0, 1):
            rows = [{'image': str(i), 'class': 'apples', 'gt_count': 5., 'pred_count': 5 + error,
                     'signed_error': error} for i, error in enumerate(errors)]
            runner.base.write_csv(directory / ('val_epoch{}.csv'.format(epoch)), rows)
            records.append({'epoch': epoch, 'steps': 1000, 'compatible': 500,
                            'input_target_routing_sha256': str(epoch), 'validation': counting_summary(rows)})
        write_json(directory / 'completed.json', {'optimizer_steps': 2000, 'records': records})


def test_compare_reports_paired_deltas_and_rejects_changed_sampling(planned):
    *_, output = planned
    paired_outputs(output)
    rows = runner.compare(output)
    assert len(rows) == 2
    assert rows[0]['hnw_minus_control_mae'] == -.5
    assert rows[0]['hnw_minus_control_bias'] == -.5
    assert rows[0]['hnw_minus_control_rmse'] == pytest.approx(1 - (2.5 ** .5))
    completed = read_json(output / 'hnw/completed.json')
    completed['records'][0]['input_target_routing_sha256'] = 'changed'
    write_json(output / 'hnw/completed.json', completed)
    with pytest.raises(ValueError, match='schedule/data mismatch'):
        runner.compare(output)


def test_compare_rejects_different_validation_images(planned):
    *_, output = planned
    paired_outputs(output)
    path = output / 'hnw/val_epoch0.csv'
    text = path.read_text().replace('0,apples', 'other,apples')
    path.write_text(text)
    with pytest.raises(ValueError, match='image sets differ'):
        runner.compare(output)


def test_train_command_is_explicitly_gated(planned):
    _, _, run, output = planned
    argv = ['train', '--asset-root', str(output.parent), '--rp010-run', str(run), '--output-dir', str(output)]
    with pytest.raises(SystemExit):
        runner.main(argv)
    with pytest.raises(FileNotFoundError):
        runner.main(argv + ['--arm', 'control', '--allow-training'])
