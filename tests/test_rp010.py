"""CPU controls, real dataset routing and production loss on a tiny mock model."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
import torch

from datasets.dataset import ObjectCount, build_rich_prompt_attention_mask
from tools import train_rp010 as runner
from utils.regression_trainer import Reg_Trainer, train_collate
from utils.rich_prompt_training import load_rich_prompt_bank, ordered_image_fingerprint
from utils.rp010 import (
    EpochDataset, augmentation_rng, better_validation, counting_summary,
    data_contract, epoch_order, load_config, parameter_memory_estimate, new_output,
    parameter_contract, read_json, sample_seed, shared_config, steps_per_epoch,
    validate_pair, validate_resume, write_json,
)
from utils.train_subset import select_train_subset_indices

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    def __call__(self, text, **kwargs):
        count = len(text.split())
        if kwargs.get('truncation'):
            count = min(count, kwargs['max_length'])
        return {'input_ids': torch.arange(count).reshape(1, -1)}


class TinyCount(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = torch.nn.Conv2d(3, 1, 1)
        self.decoder = torch.nn.Conv2d(1, 1, 1)
        self.forwards = []

    def set_train(self):
        self.train()

    def set_eval(self):
        self.eval()

    def forward(self, image, captions, mask):
        self.forwards.append((tuple(captions), mask.detach().clone()))
        feat = self.unet(torch.nn.functional.adaptive_avg_pool2d(image, 48))
        density = self.decoder(feat).sigmoid()
        return density, feat.tanh(), feat.tanh(), feat.sigmoid().detach()


class RP010Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a = load_config(ROOT / 'configs/rp010a.json', self.root)
        self.b = load_config(ROOT / 'configs/rp010b.json', self.root)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self):
        root = self.root / 'dataset'
        for folder in ('FSC_147', 'images_384_VarV2', 'gt_density_map_adaptive_384_VarV2'):
            (root / folder).mkdir(parents=True)
        train = ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg']
        splits = {'train': train, 'val': ['v.jpg'], 'test': ['t.jpg']}
        names = train + ['v.jpg', 't.jpg']
        classes = {name: ('red mugs' if i % 2 else 'apples') for i, name in enumerate(names)}
        write_json(root / 'FSC_147/Train_Test_Val_FSC_147.json', splits)
        write_json(root / 'FSC_147/annotation_FSC147_384.json', {n: {'points': [[100, 100]]} for n in names})
        (root / 'FSC_147/ImageClasses_FSC147.txt').write_text(''.join(n + '\t' + classes[n] + '\n' for n in names))
        for i, name in enumerate(names):
            Image.new('RGB', (384, 384), (i * 20, 20, 40)).save(root / 'images_384_VarV2' / name)
            np.save(root / 'gt_density_map_adaptive_384_VarV2' / Path(name).with_suffix('.npy'),
                    np.ones((384, 384), dtype=np.float32) / 384**2)
        order = select_train_subset_indices(4, 4, 3407)
        selected = [train[i] for i in order]
        bank = self.root / 'bank.json'
        write_json(bank, {'metadata': {
            'benchmark': 'FSC147', 'split': 'train', 'train_samples': 4, 'train_subset_seed': 3407,
            'protocol_version': 'rich-prompt-phase1-v3', 'selected_sample_count': 4,
            'selected_image_fingerprint': ordered_image_fingerprint(selected)},
            'prompts': {n: {'image': n, 'class': classes[n], 'status': 'ok',
                            'detailed': classes[n] + ' with vivid appearance on the surface',
                            'generalized': 'objects with vivid appearance on the surface'} for n in train}})
        config = dict(self.a, dataset=str(root), prompt_bank=str(bank), train_samples=4)
        return config, selected, classes

    def test_pair_explicit_controls_and_separate_outputs(self):
        validate_pair(self.a, self.b)
        self.assertEqual(shared_config(self.a), shared_config(self.b))
        self.assertEqual(self.a['initialization_checkpoint'], self.b['initialization_checkpoint'])
        self.assertNotEqual(self.a['output_dir'], self.b['output_dir'])
        self.assertEqual(steps_per_epoch(self.a) * self.a['epochs'], 10000)
        self.b['lr'] *= 2
        with self.assertRaisesRegex(ValueError, 'controls'):
            validate_pair(self.a, self.b)

    def test_independent_seed_changes_init_but_not_subset(self):
        extra = load_config(ROOT / 'configs/rp010a.json', self.root, 3408)
        self.assertEqual(extra['train_subset_seed'], self.a['train_subset_seed'])
        self.assertNotEqual(extra['initialization_checkpoint'], self.a['initialization_checkpoint'])

    def test_config_missing_or_changed_protocol_setting_rejected(self):
        raw = json.loads((ROOT / 'configs/rp010a.json').read_text())
        for key in ('batch_size', 'amp', 'checkpoint_metric', 'resume_policy'):
            changed = dict(raw)
            del changed[key]
            path = self.root / 'config.json'
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, key):
                load_config(path, self.root)

    def test_bank_coverage_mapping_and_token_report(self):
        config, selected, _ = self.fixture()
        result = data_contract(config, Tokenizer())
        self.assertEqual(result['selected_images'], selected)
        self.assertEqual(len(result['token_rows']), 20)
        self.assertEqual(result['donor_pool_size'], 4)
        self.assertEqual({r['slot'] for r in result['token_rows']},
                         {'class', 'generic_1', 'generic_2', 'detailed', 'generalized'})
        bank = read_json(config['prompt_bank'])
        bank['prompts'][selected[0]]['class'] = 'wrong target'
        write_json(config['prompt_bank'], bank)
        with self.assertRaisesRegex(ValueError, 'class mismatch'):
            data_contract(config, Tokenizer())

    def test_missing_or_extra_bank_description_rejected(self):
        config, selected, _ = self.fixture()
        bank = read_json(config['prompt_bank'])
        bank['prompts']['unselected.jpg'] = bank['prompts'][selected[0]]
        write_json(config['prompt_bank'], bank)
        with self.assertRaisesRegex(ValueError, 'exactly'):
            data_contract(config, Tokenizer())

    def test_duplicate_json_keys_rejected(self):
        path = self.root / 'duplicated.json'
        path.write_text('{"prompts":{"a":1,"a":2}}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_json(path)

    def test_split_leakage_and_missing_donor_fail(self):
        config, _, _ = self.fixture()
        path = Path(config['dataset']) / 'FSC_147/Train_Test_Val_FSC_147.json'
        split = read_json(path)
        split['val'] = ['a.jpg']
        write_json(path, split)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            data_contract(config, Tokenizer())

    def test_token_truncation_reported_without_text_transformation(self):
        config, selected, _ = self.fixture()
        bank = read_json(config['prompt_bank'])
        original = ' '.join(['appearance'] * 90)
        bank['prompts'][selected[0]]['detailed'] = original
        write_json(config['prompt_bank'], bank)
        result = data_contract(config, Tokenizer())
        row = next(r for r in result['token_rows'] if r['image'] == selected[0] and r['slot'] == 'detailed')
        self.assertEqual((row['raw_tokens'], row['retained_tokens'], row['prompt']), (90, 75, original))
        mask = build_rich_prompt_attention_mask(Tokenizer(), original)
        self.assertEqual(mask.sum().item(), 75)
        self.assertEqual(mask[[0, 76]].sum().item(), 0)

    def test_augmentation_rng_restores_python_numpy_torch(self):
        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        expected = random.random(), np.random.rand(), torch.rand(1)
        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        with augmentation_rng(100):
            random.random(), np.random.rand(), torch.rand(10)
        actual = random.random(), np.random.rand(), torch.rand(1)
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_real_objectcount_paired_augmentation_and_routing(self):
        config, names, classes = self.fixture()
        bank = load_rich_prompt_bank(config['prompt_bank'], 4, 3407, names, classes)
        datasets = [ObjectCount(config['dataset'], 384, 8, tokenizer=Tokenizer(),
                                rich_prompt_records=bank.records, rich_supervision_mode=mode)
                    for mode in ('template-diversity', 'rich')]
        subsets = [torch.utils.data.Subset(d, select_train_subset_indices(4, 4, 3407)) for d in datasets]
        pair = [EpochDataset(d, 3407, 2) for d in subsets]
        for i in range(4):
            left, right = pair[0][i], pair[1][i]
            for x, y in zip(left[:5], right[:5]):
                self.assertTrue(torch.equal(x, y) if torch.is_tensor(x) else x == y)
            self.assertEqual(left[5]['rich_compatible'], right[5]['rich_compatible'])
        self.assertEqual(pair[0].trace.hexdigest(), pair[1].trace.hexdigest())
        self.assertEqual(pair[0].compatible, pair[1].compatible)
        self.assertEqual(epoch_order(1000, 3407, 0), epoch_order(1000, 3407, 0))
        self.assertNotEqual(epoch_order(1000, 3407, 0), epoch_order(1000, 3407, 1))

    def make_mock_trainer(self, arm='A'):
        trainer = runner.ControlledTrainer.__new__(runner.ControlledTrainer)
        trainer.control = self.a if arm == 'A' else self.b
        trainer.guard = {'schema': 'test', 'config': trainer.control}
        trainer.model = TinyCount()
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=5e-5)
        trainer.optimizer_steps, trainer.epoch_records = 0, []
        trainer.best_mae = trainer.best_mse = float('inf')
        trainer.best_epoch = None
        trainer.smoke = False
        trainer.epoch, trainer.start_epoch = 0, 0
        trainer.device = torch.device('cpu')
        trainer.args = SimpleNamespace(rich_supervision_mode=trainer.control['supervision_mode'],
                                      rich_consistency_weight=0., rich_loss_diagnostic_steps=0)
        trainer._rich_diagnostic_steps_logged = 0
        trainer.rich_prompt_config = {'test': True}
        trainer.save_dir = str(self.root)
        return trainer

    def test_production_three_forwards_backward_and_one_optimizer_step_both_arms(self):
        for arm in ('A', 'B'):
            trainer = self.make_mock_trainer(arm)
            image = torch.ones(3, 384, 384)
            mask = torch.zeros(77)
            mask[1] = 1
            meta = {'source_image_name': 'a.jpg', 'rich_compatible': True,
                    'detailed_prompt': 'auxiliary one', 'generalized_prompt': 'auxiliary two',
                    'detailed_prompt_attn_mask': mask, 'generalized_prompt_attn_mask': mask}
            batch = train_collate([(image, torch.ones(1, 48, 48) / 2304, 'apples', mask,
                                    torch.ones(1, 48, 48), meta)])
            trainer.dataloaders = {'train': [batch]}
            before = next(trainer.model.parameters()).detach().clone()
            trainer.train_epoch()
            self.assertEqual(trainer.optimizer_steps, 1)
            self.assertEqual(len(trainer.model.forwards), 3)
            self.assertEqual(trainer.args.rich_supervision_mode, trainer.control['supervision_mode'])
            self.assertFalse(torch.equal(before, next(trainer.model.parameters()).detach()))

    def test_checkpoint_roundtrip_optimizer_and_rng_and_cross_arm_guard(self):
        trainer = self.make_mock_trainer()
        trainer.optimizer.zero_grad()
        sum(p.square().sum() for p in trainer.model.parameters()).backward()
        trainer.optimizer.step()
        trainer.optimizer_steps = 1000
        trainer.epoch_records = [{'epoch': 0, 'steps': 1000}]
        expected = copy.deepcopy(trainer.model.state_dict())
        path = self.root / 'latest.tar'
        trainer.save_state(path)
        expected_random = torch.rand(5)
        with torch.no_grad():
            next(trainer.model.parameters()).add_(10)
        trainer.load_state(path)
        self.assertEqual(trainer.start_epoch, 1)
        self.assertEqual(trainer.optimizer_steps, 1000)
        self.assertTrue(torch.equal(expected_random, torch.rand(5)))
        for name, value in trainer.model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]))
        self.assertEqual(len(trainer.optimizer.state), len(list(trainer.model.parameters())))
        other = self.make_mock_trainer('B')
        with self.assertRaisesRegex(ValueError, 'Resume rejected'):
            other.load_state(path)

    def test_interrupted_epoch_boundary_resume_matches_uninterrupted_cpu_updates(self):
        # Small model only; exercise the real shared loss and RP-010 lifecycle.
        class OneSample(torch.utils.data.Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, index):
                mask = torch.zeros(77)
                mask[1] = 1
                return (torch.ones(3, 384, 384), torch.ones(1, 48, 48) / 2304,
                        'apples', mask, torch.ones(1, 48, 48), {
                            'source_image_name': 'a.jpg', 'rich_compatible': True,
                            'detailed_prompt': 'red apples', 'generalized_prompt': 'red objects',
                            'detailed_prompt_attn_mask': mask, 'generalized_prompt_attn_mask': mask})
        def trainer_at(directory):
            torch.manual_seed(55)
            trainer = self.make_mock_trainer()
            trainer.control = dict(self.a, epochs=2, train_samples=1, start_val=0)
            trainer.guard = {'schema': 'test', 'config': trainer.control}
            trainer.datasets = {'train': OneSample()}
            trainer.dataloaders = {}
            trainer.save_dir = str(directory)
            directory.mkdir()
            trainer._evaluate_split = mock.Mock(return_value=(1., 2.))
            return trainer
        complete = trainer_at(self.root / 'complete')
        complete.train()
        interrupted = trainer_at(self.root / 'interrupted')
        original_save = interrupted.save_state
        def save_then_interrupt(path, smoke=False):
            original_save(path, smoke)
            raise InterruptedError('simulated runtime interruption after committed epoch')
        interrupted.save_state = save_then_interrupt
        with self.assertRaises(InterruptedError):
            interrupted.train()
        resumed = trainer_at(self.root / 'new_trainer')
        resumed.save_dir = interrupted.save_dir
        resumed.load_state(Path(interrupted.save_dir) / 'latest.tar')
        resumed.train()
        self.assertEqual(resumed.optimizer_steps, complete.optimizer_steps)
        self.assertEqual(resumed.epoch_records, complete.epoch_records)
        for name, value in complete.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[name]))
        self.assertEqual(read_json(Path(resumed.save_dir) / 'completed.json')['epochs'], 2)

    def test_resume_step_mismatch_and_smoke_not_training(self):
        trainer = self.make_mock_trainer()
        path = self.root / 'state.tar'
        trainer.save_state(path, smoke=True)
        with self.assertRaisesRegex(ValueError, 'cannot resume training'):
            trainer.load_state(path)
        checkpoint = {'rp010_guard': trainer.guard, 'next_epoch': 1,
                      'optimizer_steps': 999, 'epoch_records': [{}]}
        with self.assertRaisesRegex(ValueError, 'step/epoch'):
            validate_resume(checkpoint, trainer.guard, 1000)

    def test_validation_rule_strict_and_no_test_call(self):
        trainer = self.make_mock_trainer()
        trainer._evaluate_split = mock.Mock(return_value=(10., 20.))
        trainer.test_epoch = mock.Mock(side_effect=AssertionError('test peeking'))
        with mock.patch.object(runner, '_atomic_torch_save') as save, mock.patch.object(runner, 'sha256', return_value='hash'):
            trainer.val_epoch()
            trainer.val_epoch()  # equal sum must not replace the first selected checkpoint
        trainer._evaluate_split.assert_called_with('val')
        self.assertEqual(save.call_count, 1)
        trainer.test_epoch.assert_not_called()
        with self.assertRaises(ValueError):
            better_validation(float('nan'), 1, 2, 3)

    def test_output_no_overwrite_and_resume_requires_state(self):
        path = self.root / 'run'
        new_output(path)
        with self.assertRaises(FileExistsError):
            new_output(path)
        with self.assertRaises(ValueError):
            new_output(path, resume=True)

    def test_memory_contract_and_gpu_preflight(self):
        parameters = parameter_contract(TinyCount())
        memory = parameter_memory_estimate(parameters)
        self.assertEqual(memory['weights_gradients_adam_estimate_bytes'], 4 * memory['parameter_bytes'])
        with mock.patch('torch.cuda.is_available', return_value=True), \
                mock.patch('torch.cuda.mem_get_info', return_value=(1, 2)), \
                mock.patch('torch.cuda.get_device_name', return_value='tiny GPU'):
            self.assertEqual(runner.gpu_capacity({'memory': memory})['status'], 'pending')

    def test_low_host_memory_blocks_initialization_before_build(self):
        with mock.patch.object(runner, 'get_contract', return_value={}), \
                mock.patch.object(runner, 'host_free_bytes', return_value=2**30), \
                mock.patch.object(runner, 'build') as build:
            with self.assertRaisesRegex(RuntimeError, 'host RAM'):
                runner.prepare(self.a)
            build.assert_not_called()

    def test_initialization_hash_and_pretrained_stat_guards(self):
        path = Path(self.a['initialization_checkpoint'])
        path.parent.mkdir(parents=True)
        path.write_bytes(b'only a fixture; not model weights')
        asset = self.root / 'tokenizer.json'
        asset.write_text('{}')
        contract = {'mock_data': 'fixed'}
        manifest = {'schema': 'rp010-paired-v1', 'shared_config': shared_config(self.a),
                    'source': 'source', 'runtime': 'runtime', 'data': contract,
                    'initialization_sha256': runner.sha256(path),
                    'pretrained_assets': [{'path': str(asset), 'bytes': asset.stat().st_size,
                                           'mtime_ns': asset.stat().st_mtime_ns}]}
        write_json(runner.init_manifest_path(self.a), manifest)
        with mock.patch.object(runner, 'source_identity', return_value='source'), \
                mock.patch.object(runner, 'runtime_signature', return_value='runtime'):
            runner.check_initialization(self.a, contract)
            asset.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'asset changed'):
                runner.check_initialization(self.a, contract)
            path.write_bytes(b'changed initialization')
            with self.assertRaisesRegex(ValueError, 'initialization bytes changed'):
                runner.check_initialization(self.a, contract)

    def test_count_metrics_signed_bias(self):
        summary = counting_summary([{'signed_error': 3}, {'signed_error': -1}, {'signed_error': 0}])
        self.assertAlmostEqual(summary['bias'], 2 / 3)
        self.assertAlmostEqual(summary['mae'], 4 / 3)
        self.assertEqual((summary['overcount'], summary['undercount'], summary['ties']), (1, 1, 1))

    def test_completed_pair_comparison_joins_images_and_checks_arm_identity(self):
        for config, shift in ((self.a, 2), (self.b, 1)):
            output = new_output(config['output_dir'])
            evaluation = new_output(output / 'evaluation_class')
            guard = {'config': config, 'source': 'same', 'runtime': 'same',
                     'data': {'split_counts': {'test': 2}}, 'initialization_sha256': 'same'}
            selection = {'epoch': 7, 'rule': 'val_mae_plus_rmse', 'checkpoint': 'best_epoch7.pth'}
            write_json(output / 'manifest.json', {'guard': guard})
            write_json(output / 'completed.json', {'guard': guard, 'epochs': 10,
                                                   'optimizer_steps': 10000, 'selection': selection})
            write_json(output / 'epoch_records.json', [{'epoch': i, 'steps': 1000,
                                                       'input_target_routing_sha256': str(i)} for i in range(10)])
            write_json(evaluation / 'manifest.json', {'guard': guard, 'selection': selection})
            summaries = {}
            for name, count, key in [('fsc147_test_class.csv', 2, 'primary_fsc147_test'),
                                     ('fsc147s_class.csv', 230, 'secondary_fsc147s')]:
                rows = [{'image': '{}.jpg'.format(i), 'class': 'apples', 'gt_count': 1,
                         'pred_count': 1 + shift, 'abs_error': shift, 'signed_error': shift} for i in range(count)]
                runner.write_csv(evaluation / name, rows if shift == 2 else list(reversed(rows)))
                summaries[key] = counting_summary(rows)
            write_json(evaluation / 'summary.json', summaries)
        directory = self.root / 'paired'
        runner.compare(self.a, self.b, directory)
        summary = read_json(directory / 'summary.json')['fsc147_test_class.csv']
        self.assertEqual(summary['b_minus_a']['mae'], -1)
        self.assertEqual(summary['mae_delta_image_bootstrap95'], [-1, -1])
        path = Path(self.b['output_dir']) / 'manifest.json'
        changed = read_json(path)
        changed['guard']['config'] = self.a
        write_json(path, changed)
        with self.assertRaisesRegex(ValueError, 'different arm'):
            runner.compare(self.a, self.b, self.root / 'invalid_comparison')

    def test_train_flag_required_before_any_training(self):
        with mock.patch.object(runner, 'run_training') as train, self.assertRaises(SystemExit):
            runner.main(['train', '--asset-root', str(self.root)])
        train.assert_not_called()

    def test_notebook_executes_all_cells_in_safe_default_order(self):
        notebook = json.loads((ROOT / 'notebooks/RP010_controlled_supervision.ipynb').read_text(encoding='utf-8'))
        namespace = {'__name__': '__notebook_test__'}
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                self.assertFalse(cell['outputs'])
                exec(compile(''.join(cell['source']), 'cell_{}'.format(index), 'exec'), namespace)
        self.assertFalse(namespace['RUN_STAGE'])
        self.assertFalse(namespace['ENABLE_FULL_TRAINING'])
        sources = '\n'.join(''.join(c['source']) for c in notebook['cells'])
        self.assertNotIn('SOURCE_ZIP', sources)
        self.assertNotIn('%%bash', sources)
        self.assertLess(sources.index("'clone'"), sources.index("command = [sys.executable"))


if __name__ == '__main__':
    unittest.main()
