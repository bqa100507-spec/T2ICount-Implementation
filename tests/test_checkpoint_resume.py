import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from train import parse_arg
from utils.helper import SaveHandler
from utils.regression_trainer import (
    Reg_Trainer,
    should_save_training_checkpoint,
)
from utils.trainer import Trainer


def _trainer(model, optimizer, directory):
    trainer = Reg_Trainer.__new__(Reg_Trainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.device = torch.device('cpu')
    trainer.save_dir = str(directory)
    trainer.save_list = SaveHandler(num=2)
    trainer.epoch = 3
    trainer.start_epoch = 0
    trainer.best_mae = 4.5
    trainer.best_mse = 6.75
    return trainer


def _run_epochs(checkpoint_interval, epochs):
    trainer = Reg_Trainer.__new__(Reg_Trainer)
    trainer.args = SimpleNamespace(
        epochs=epochs,
        start_val=epochs + 1,
        val_epoch=1,
        checkpoint_interval=checkpoint_interval,
    )
    trainer.start_epoch = 0
    trainer.train_epoch = mock.Mock()
    trainer.val_epoch = mock.Mock()
    saved_epochs = []
    trainer._save_training_checkpoint = lambda: saved_epochs.append(
        trainer.epoch
    )

    trainer.train()
    return saved_epochs


class CheckpointIntervalTests(unittest.TestCase):
    def test_default_interval_reproduces_existing_epoch_cadence(self):
        with mock.patch('sys.argv', ['train.py']):
            args = parse_arg()

        self.assertEqual(args.checkpoint_interval, 5)
        self.assertEqual(
            _run_epochs(args.checkpoint_interval, 16), [0, 5, 10, 15]
        )

    def test_interval_one_saves_after_every_epoch(self):
        self.assertEqual(_run_epochs(1, 6), [0, 1, 2, 3, 4, 5])

    def test_invalid_intervals_are_rejected_by_cli(self):
        for invalid_interval in (0, -1):
            with self.subTest(checkpoint_interval=invalid_interval):
                with mock.patch(
                    'sys.argv',
                    ['train.py', '--checkpoint-interval', str(invalid_interval)],
                ):
                    with self.assertRaisesRegex(
                        ValueError, '--checkpoint-interval must be >= 1'
                    ):
                        parse_arg()

    def test_checkpoint_interval_is_in_startup_config_log(self):
        args = SimpleNamespace(
            content='test-run',
            save_dir=None,
            checkpoint_interval=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            args.save_dir = temp_dir
            with mock.patch('utils.trainer.logger'), mock.patch(
                'utils.trainer.logging.info'
            ) as log_info:
                Trainer(args)

        logged_messages = [call.args[0] for call in log_info.call_args_list]
        self.assertIn('checkpoint_interval:1', logged_messages)

    def test_checkpoint_interval_does_not_change_best_model_selection(self):
        trainer = Reg_Trainer.__new__(Reg_Trainer)
        trainer.args = SimpleNamespace(checkpoint_interval=1)
        trainer.epoch = 7
        trainer.best_mae = 5.0
        trainer.best_mse = 5.0
        trainer.save_dir = 'checkpoints'
        model_state = {'weight': object()}
        trainer.model = mock.Mock()
        trainer.model.state_dict.return_value = model_state
        trainer._evaluate_split = mock.Mock(
            side_effect=[(3.0, 4.0), (4.0, 4.0)]
        )
        trainer.test_epoch = mock.Mock()

        with mock.patch(
            'utils.regression_trainer._atomic_torch_save'
        ) as atomic_save:
            trainer.val_epoch()
            trainer.val_epoch()

        self.assertEqual((trainer.best_mae, trainer.best_mse), (3.0, 4.0))
        atomic_save.assert_called_once()
        saved_payload, saved_path = atomic_save.call_args.args
        self.assertIs(saved_payload, model_state)
        self.assertEqual(
            saved_path, os.path.join('checkpoints', 'best_model_7.pth')
        )
        trainer.test_epoch.assert_called_once_with()

    def test_pure_cadence_helper_matches_epoch_index_convention(self):
        saved_epochs = [
            epoch for epoch in range(12)
            if should_save_training_checkpoint(epoch, 5)
        ]
        self.assertEqual(saved_epochs, [0, 5, 10])


class CheckpointResumeTests(unittest.TestCase):
    def test_cuda_resume_loads_checkpoint_on_cpu_and_restores_full_state(self):
        source_model = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.AdamW(source_model.parameters())
        source_model(torch.ones(1, 2)).sum().backward()
        source_optimizer.step()
        cpu_rng_state = torch.get_rng_state().clone()
        checkpoint = {
            'epoch': 4,
            'next_epoch': 5,
            'model_state_dict': source_model.state_dict(),
            'optimizer_state_dict': source_optimizer.state_dict(),
            'best_mae': 1.25,
            'best_mse': 2.5,
            'rng_state': {'torch': cpu_rng_state},
        }

        parameter_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        target_model = torch.nn.Linear(2, 1).to(parameter_device)
        target_optimizer = torch.optim.AdamW(target_model.parameters())
        trainer = _trainer(target_model, target_optimizer, '.')
        trainer.device = torch.device('cuda')

        with mock.patch(
            'utils.regression_trainer.load_trusted_legacy_checkpoint',
            return_value=checkpoint,
        ) as checkpoint_loader, mock.patch(
            'utils.regression_trainer.torch.set_rng_state'
        ) as set_rng_state:
            trainer._load_training_checkpoint('resume.tar')

        checkpoint_loader.assert_called_once_with('resume.tar', 'cpu')
        set_rng_state.assert_called_once()
        restored_rng_state = set_rng_state.call_args.args[0]
        self.assertEqual(restored_rng_state.device.type, 'cpu')
        self.assertEqual(restored_rng_state.dtype, torch.uint8)
        self.assertTrue(torch.equal(restored_rng_state, cpu_rng_state))

        self.assertEqual(trainer.start_epoch, 5)
        self.assertEqual(trainer.best_mae, 1.25)
        self.assertEqual(trainer.best_mse, 2.5)
        for source, target in zip(
                source_model.parameters(), target_model.parameters()):
            self.assertTrue(torch.equal(source, target.detach().cpu()))

        target_parameters = set(target_model.parameters())
        self.assertEqual(set(target_optimizer.state), target_parameters)
        for parameter, state in target_optimizer.state.items():
            self.assertIn(parameter, target_parameters)
            for key in ('exp_avg', 'exp_avg_sq'):
                self.assertEqual(state[key].device, parameter.device)
        source_states = list(source_optimizer.state.values())
        target_states = list(target_optimizer.state.values())
        self.assertEqual(len(source_states), len(target_states))
        for source_state, target_state in zip(source_states, target_states):
            self.assertEqual(set(source_state), set(target_state))
            for key, source_value in source_state.items():
                target_value = target_state[key]
                if torch.is_tensor(source_value):
                    self.assertTrue(torch.equal(
                        source_value.cpu(), target_value.cpu()
                    ))
                else:
                    self.assertEqual(source_value, target_value)

    def test_full_state_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_model = torch.nn.Linear(2, 1)
            source_optimizer = torch.optim.AdamW(source_model.parameters())
            source_model(torch.ones(1, 2)).sum().backward()
            source_optimizer.step()
            source_trainer = _trainer(
                source_model, source_optimizer, Path(temp_dir)
            )

            random.seed(10)
            np.random.seed(11)
            torch.manual_seed(12)
            source_trainer._save_training_checkpoint()
            expected_random = random.random()
            expected_numpy = np.random.rand()
            expected_torch = torch.rand(1)

            target_model = torch.nn.Linear(2, 1)
            target_optimizer = torch.optim.AdamW(target_model.parameters())
            target_trainer = _trainer(
                target_model, target_optimizer, Path(temp_dir)
            )
            target_trainer.best_mae = np.inf
            target_trainer.best_mse = np.inf
            checkpoint = Path(temp_dir) / '3_ckpt.tar'
            target_trainer._load_training_checkpoint(str(checkpoint))

            self.assertEqual(target_trainer.start_epoch, 4)
            self.assertEqual(target_trainer.best_mae, 4.5)
            self.assertEqual(target_trainer.best_mse, 6.75)
            self.assertTrue(target_optimizer.state)
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(np.random.rand(), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(1), expected_torch))
            for source, target in zip(
                    source_model.parameters(), target_model.parameters()):
                self.assertTrue(torch.equal(source, target))

    def test_legacy_checkpoint_defaults_missing_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            checkpoint = Path(temp_dir) / 'legacy.tar'
            torch.save({
                'epoch': 7,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, str(checkpoint))

            trainer = _trainer(model, optimizer, Path(temp_dir))
            trainer.best_mae = np.inf
            trainer.best_mse = np.inf
            trainer._load_training_checkpoint(str(checkpoint))
            self.assertEqual(trainer.start_epoch, 8)
            self.assertTrue(np.isinf(trainer.best_mae))
            self.assertTrue(np.isinf(trainer.best_mse))


if __name__ == '__main__':
    unittest.main()
