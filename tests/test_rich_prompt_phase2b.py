import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

from datasets.dataset import ObjectCount, build_rich_prompt_attention_mask
from train import parse_arg
from utils.regression_trainer import (
    Reg_Trainer,
    _t2i_loss_components,
    combine_rich_sample_weighted_loss,
    log_rich_loss_diagnostic,
    rich_consistency_loss,
    train_collate,
)
from utils.rich_prompt_training import (
    RICH_PROMPT_PROTOCOL,
    RichPromptRecord,
    RichPromptTrainingError,
    ValidatedRichPromptBank,
    build_rich_checkpoint_config,
    load_rich_prompt_bank,
    ordered_image_fingerprint,
    validate_prompt_bank_metadata,
    validate_resume_rich_config,
)


PILOT_FINGERPRINT = (
    'sha256:dd96b36bf15013e194b1a8ece06452a19822aae028fc00c0de019cbb7a311f24'
)


class _Tokenizer:
    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
        return_overflowing_tokens=False,
        return_tensors='pt',
    ):
        length = len(text.split())
        if truncation and max_length is not None:
            length = min(length, max_length)
        return {'input_ids': torch.arange(length).reshape(1, length)}


def _metadata(names, train_samples=None, seed=3407, fingerprint=None):
    return {
        'benchmark': 'FSC147',
        'split': 'train',
        'train_samples': len(names) if train_samples is None else train_samples,
        'train_subset_seed': seed,
        'protocol_version': RICH_PROMPT_PROTOCOL,
        'selected_sample_count': len(names),
        'selected_image_fingerprint': (
            fingerprint or ordered_image_fingerprint(names)
        ),
    }


def _prompt(image_name, class_name):
    return {
        'image': image_name,
        'class': class_name,
        'detailed': 'Detailed {} appearance.'.format(class_name),
        'generalized': 'Generalized object appearance.',
        'status': 'ok',
    }


def _write_bank(path, names, classes, metadata=None, prompts=None):
    payload = {
        'metadata': metadata or _metadata(names),
        'prompts': (
            {name: _prompt(name, classes[name]) for name in names}
            if prompts is None else prompts
        ),
        'failures': {},
    }
    path.write_text(json.dumps(payload), encoding='utf-8')


class PromptBankValidationTests(unittest.TestCase):
    def test_exact_1000_3407_pilot_provenance_is_accepted(self):
        validate_prompt_bank_metadata(
            {
                'benchmark': 'FSC147',
                'split': 'train',
                'train_samples': 1000,
                'train_subset_seed': 3407,
                'protocol_version': 'rich-prompt-phase1-v3',
                'selected_sample_count': 1000,
                'selected_image_fingerprint': PILOT_FINGERPRINT,
            },
            train_samples=1000,
            train_subset_seed=3407,
            selected_sample_count=1000,
            selected_image_fingerprint=PILOT_FINGERPRINT,
        )

    def test_wrong_seed_is_rejected(self):
        names = ['a.jpg']
        with self.assertRaisesRegex(
            RichPromptTrainingError, 'train_subset_seed'
        ):
            validate_prompt_bank_metadata(
                _metadata(names, seed=999),
                train_samples=1,
                train_subset_seed=3407,
                selected_sample_count=1,
                selected_image_fingerprint=ordered_image_fingerprint(names),
            )

    def test_wrong_fingerprint_is_rejected(self):
        names = ['a.jpg']
        with self.assertRaisesRegex(
            RichPromptTrainingError, 'selected_image_fingerprint'
        ):
            validate_prompt_bank_metadata(
                _metadata(names, fingerprint='sha256:' + ('0' * 64)),
                train_samples=1,
                train_subset_seed=3407,
                selected_sample_count=1,
                selected_image_fingerprint=ordered_image_fingerprint(names),
            )

    def test_missing_prompt_is_rejected(self):
        names = ['a.jpg']
        classes = {'a.jpg': 'apples'}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'bank.json'
            _write_bank(path, names, classes, prompts={})
            with self.assertRaisesRegex(
                RichPromptTrainingError, 'missing selected prompt'
            ):
                load_rich_prompt_bank(path, 1, 3407, names, classes)

    def test_missing_prompt_text_is_rejected(self):
        names = ['a.jpg']
        classes = {'a.jpg': 'apples'}
        prompt = _prompt('a.jpg', 'apples')
        prompt['generalized'] = '  '
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'bank.json'
            _write_bank(path, names, classes, prompts={'a.jpg': prompt})
            with self.assertRaisesRegex(
                RichPromptTrainingError, 'missing generalized prompt'
            ):
                load_rich_prompt_bank(path, 1, 3407, names, classes)

    def test_class_mismatch_is_rejected(self):
        names = ['a.jpg']
        classes = {'a.jpg': 'apples'}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'bank.json'
            _write_bank(
                path,
                names,
                classes,
                prompts={'a.jpg': _prompt('a.jpg', 'pears')},
            )
            with self.assertRaisesRegex(
                RichPromptTrainingError, 'class mismatch'
            ):
                load_rich_prompt_bank(path, 1, 3407, names, classes)

    def test_valid_bank_is_loaded_once_into_records_with_file_hash(self):
        names = ['a.jpg', 'b.jpg']
        classes = {'a.jpg': 'apples', 'b.jpg': 'pears'}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'bank.json'
            _write_bank(path, names, classes)
            bank = load_rich_prompt_bank(path, 2, 3407, names, classes)

        self.assertEqual(list(bank.records), names)
        self.assertRegex(bank.file_fingerprint, r'^sha256:[0-9a-f]{64}$')


def _build_dataset(root, rich=True):
    names = ['source.jpg', 'same.jpg', 'other.jpg', 'fourth.jpg']
    classes = {
        'source.jpg': 'apples',
        'same.jpg': 'apples',
        'other.jpg': 'pears',
        'fourth.jpg': 'plums',
    }
    (root / 'FSC_147').mkdir(parents=True)
    (root / 'images_384_VarV2').mkdir()
    (root / 'gt_density_map_adaptive_384_VarV2').mkdir()
    (root / 'FSC_147' / 'Train_Test_Val_FSC_147.json').write_text(
        json.dumps({'train': names, 'val': names[:1], 'test': names[:1]}),
        encoding='utf-8',
    )
    (root / 'FSC_147' / 'annotation_FSC147_384.json').write_text(
        json.dumps({name: {'points': [[1, 1]]} for name in names}),
        encoding='utf-8',
    )
    (root / 'FSC_147' / 'ImageClasses_FSC147.txt').write_text(
        ''.join('{}\t{}\n'.format(name, classes[name]) for name in names),
        encoding='utf-8',
    )
    for index, name in enumerate(names):
        Image.new('RGB', (8, 8), color=(index, index, index)).save(
            root / 'images_384_VarV2' / name
        )
        np.save(
            root / 'gt_density_map_adaptive_384_VarV2' / name.replace(
                '.jpg', '.npy'
            ),
            np.ones((8, 8), dtype=np.float32),
        )

    records = None
    if rich:
        records = {
            name: RichPromptRecord(
                image_name=name,
                class_name=classes[name],
                detailed='red round {}'.format(classes[name]),
                generalized='red objects',
            )
            for name in names
        }
    dataset = ObjectCount(
        str(root),
        crop_size=8,
        downsample_ratio=8,
        method='train',
        concat_size=4,
        tokenizer=_Tokenizer(),
        rich_prompt_records=records,
    )
    dataset.train_transform_density = mock.Mock(
        return_value=(
            torch.zeros(3, 8, 8),
            torch.zeros(1, 1, 1),
            torch.ones(1, 1, 1),
        )
    )
    return dataset


class DatasetCompatibilityTests(unittest.TestCase):
    def test_original_positive_is_rich_compatible_with_own_masks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = _build_dataset(Path(temp_dir))
            with mock.patch(
                'datasets.dataset.random.random', return_value=0.9
            ):
                sample = dataset[0]

        metadata = sample[5]
        self.assertTrue(metadata['rich_compatible'])
        self.assertEqual(metadata['source_image_name'], 'source.jpg')
        self.assertEqual(
            metadata['detailed_prompt_attn_mask'].sum().item(), 3
        )
        self.assertEqual(
            metadata['generalized_prompt_attn_mask'].sum().item(), 2
        )

    def test_different_class_random_negative_is_not_rich_compatible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = _build_dataset(Path(temp_dir))
            other_path = dataset.im_list[2]
            with mock.patch(
                'datasets.dataset.random.random', side_effect=[0.1, 0.9]
            ), mock.patch(
                'datasets.dataset.random.sample', return_value=[other_path]
            ):
                sample = dataset[0]

        self.assertEqual(sample[2], 'pears')
        self.assertFalse(sample[5]['rich_compatible'])
        self.assertEqual(
            sample[5]['detailed_prompt_attn_mask'].sum().item(), 0
        )

    def test_same_class_unchanged_random_branch_is_rich_compatible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = _build_dataset(Path(temp_dir))
            same_path = dataset.im_list[1]
            with mock.patch(
                'datasets.dataset.random.random', side_effect=[0.1, 0.9]
            ), mock.patch(
                'datasets.dataset.random.sample', return_value=[same_path]
            ):
                sample = dataset[0]

        self.assertEqual(sample[2], 'apples')
        self.assertTrue(sample[5]['rich_compatible'])

    def test_mosaic_is_not_rich_compatible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = _build_dataset(Path(temp_dir))
            mosaic_paths = dataset.im_list[1:4]
            with mock.patch(
                'datasets.dataset.random.random', side_effect=[0.1, 0.1]
            ), mock.patch(
                'datasets.dataset.random.sample', return_value=mosaic_paths
            ), mock.patch('datasets.dataset.random.shuffle'):
                sample = dataset[0]

        self.assertFalse(sample[5]['rich_compatible'])

    def test_no_rich_records_preserve_five_field_dataset_return(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = _build_dataset(Path(temp_dir), rich=False)
            with mock.patch(
                'datasets.dataset.random.random', return_value=0.9
            ):
                sample = dataset[0]

        self.assertEqual(len(sample), 5)

    def test_rich_mask_truncation_is_explicit_and_deterministic(self):
        prompt = ' '.join('token{}'.format(index) for index in range(100))
        first = build_rich_prompt_attention_mask(_Tokenizer(), prompt)
        second = build_rich_prompt_attention_mask(_Tokenizer(), prompt)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.sum().item(), 75)
        self.assertEqual(first[0].item(), 0)
        self.assertEqual(first[76].item(), 0)


class _Progress(list):
    def set_postfix(self, **kwargs):
        return None


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def set_train(self):
        self.train()

    def forward(self, inputs, captions, prompt_mask):
        self.calls.append((inputs.detach().clone(), tuple(captions)))
        density = inputs[:, :1] * self.scale
        similarity = density
        fused = torch.zeros_like(density).detach()
        return density, similarity, similarity, fused


def _fake_t2i_components(outputs, gt_den_maps, gt_img_attn_mask):
    total = outputs[0].mean()
    zero = total * 0
    return {'reg': total, 'rrc1': zero, 'rrc2': zero, 'total': total}


def _training_batch(compatible=None):
    inputs = torch.stack(
        [torch.ones(1, 2, 2), torch.full((1, 2, 2), 2.0)]
    )
    baseline = (
        inputs,
        torch.zeros(2, 1, 2, 2),
        ('class-a', 'class-b'),
        torch.ones(2, 77),
        torch.ones(2, 1, 2, 2),
    )
    if compatible is None:
        return baseline
    rich_batch = {
        'source_image_name': ('a.jpg', 'b.jpg'),
        'rich_compatible': torch.tensor(compatible, dtype=torch.bool),
        'detailed_prompt': ('detailed-a', 'detailed-b'),
        'generalized_prompt': ('general-a', 'general-b'),
        'detailed_prompt_attn_mask': torch.ones(2, 77),
        'generalized_prompt_attn_mask': torch.ones(2, 77),
    }
    return baseline + (rich_batch,)


def _run_training_batch(compatible=None, diagnostic_steps=0):
    trainer = Reg_Trainer.__new__(Reg_Trainer)
    trainer.args = SimpleNamespace(
        rich_consistency_weight=0.0,
        rich_loss_diagnostic_steps=diagnostic_steps,
    )
    trainer.device = torch.device('cpu')
    trainer.model = _DummyModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.dataloaders = {'train': [_training_batch(compatible)]}
    trainer.epoch = 0
    trainer.rich_prompt_config = (
        None if compatible is None else {'enabled': True}
    )
    trainer._rich_diagnostic_steps_logged = 0
    with mock.patch(
        'utils.regression_trainer.progress_dataloader',
        side_effect=lambda dataloader, description: _Progress(dataloader),
    ), mock.patch(
        'utils.regression_trainer._t2i_loss_components',
        side_effect=_fake_t2i_components,
    ):
        trainer.train_epoch()
    return trainer


class ForwardAndLossTests(unittest.TestCase):
    def test_baseline_batch_keeps_one_forward_and_original_collate_shape(self):
        sample = (
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 2),
            'class',
            torch.ones(77),
            torch.ones(1, 2, 2),
        )
        self.assertEqual(len(train_collate([sample])), 5)
        trainer = _run_training_batch()
        self.assertEqual(len(trainer.model.calls), 1)

    def test_rich_collate_keeps_each_prompt_and_its_own_mask(self):
        base = (
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 2),
            'class',
            torch.ones(77),
            torch.ones(1, 2, 2),
        )
        rich = {
            'source_image_name': 'a.jpg',
            'rich_compatible': True,
            'detailed_prompt': 'detailed text',
            'generalized_prompt': 'general text',
            'detailed_prompt_attn_mask': torch.ones(77),
            'generalized_prompt_attn_mask': torch.zeros(77),
        }
        batch = train_collate([base + (rich,)])
        self.assertEqual(len(batch), 6)
        self.assertEqual(batch[5]['detailed_prompt'], ('detailed text',))
        self.assertEqual(batch[5]['generalized_prompt'], ('general text',))
        self.assertEqual(
            batch[5]['detailed_prompt_attn_mask'].sum().item(), 77
        )
        self.assertEqual(
            batch[5]['generalized_prompt_attn_mask'].sum().item(), 0
        )

    def test_each_branch_uses_original_t2i_supervised_equation(self):
        outputs = tuple(torch.ones(1, 1, 2, 2) for _ in range(4))
        with mock.patch(
            'utils.regression_trainer.get_reg_loss',
            return_value=torch.tensor(2.0),
        ), mock.patch(
            'utils.regression_trainer.RRC_loss',
            side_effect=[torch.tensor(3.0), torch.tensor(4.0)],
        ):
            components = _t2i_loss_components(
                outputs,
                torch.ones(1, 1, 2, 2),
                torch.ones(1, 1, 2, 2),
            )
        self.assertAlmostEqual(components['total'].item(), 2.07, places=6)

    def test_compatible_sample_receives_all_three_supervised_forwards(self):
        trainer = _run_training_batch([True, False])
        self.assertEqual(len(trainer.model.calls), 3)
        self.assertEqual(trainer.model.calls[0][1], ('class-a', 'class-b'))
        self.assertEqual(trainer.model.calls[1][1], ('detailed-a',))
        self.assertEqual(trainer.model.calls[2][1], ('general-a',))
        self.assertTrue(torch.equal(
            trainer.model.calls[1][0], torch.ones(1, 1, 2, 2)
        ))

    def test_incompatible_samples_receive_class_forward_only(self):
        trainer = _run_training_batch([False, False])
        self.assertEqual(len(trainer.model.calls), 1)
        self.assertEqual(trainer.model.calls[0][1], ('class-a', 'class-b'))

    def test_consistency_is_mean_of_three_pairwise_elementwise_mses(self):
        terms = rich_consistency_loss(
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([3.0]),
        )
        self.assertAlmostEqual(terms['class_detailed'].item(), 1.0)
        self.assertAlmostEqual(terms['class_generalized'].item(), 9.0)
        self.assertAlmostEqual(terms['detailed_generalized'].item(), 4.0)
        self.assertAlmostEqual(terms['mean'].item(), 14.0 / 3.0, places=6)

    def test_zero_consistency_weight_adds_no_consistency_gradient(self):
        consistency_only = torch.tensor(2.0, requires_grad=True)
        combined, _ = combine_rich_sample_weighted_loss(
            torch.tensor(3.0, requires_grad=True),
            torch.tensor(2.0, requires_grad=True),
            torch.tensor(5.0, requires_grad=True),
            torch.tensor(8.0, requires_grad=True),
            consistency_only.square(),
            rich_count=1,
            total_count=2,
            consistency_weight=0.0,
        )
        combined.backward()
        self.assertEqual(consistency_only.grad.item(), 0.0)

    def test_sample_count_weighting_does_not_triple_positive_weight(self):
        combined, rich_supervised = combine_rich_sample_weighted_loss(
            torch.tensor(3.0),
            torch.tensor(2.0),
            torch.tensor(5.0),
            torch.tensor(8.0),
            torch.tensor(0.0),
            rich_count=1,
            total_count=2,
            consistency_weight=0.0,
        )
        self.assertEqual(rich_supervised.item(), 5.0)
        self.assertEqual(combined.item(), 4.5)

    def test_diagnostic_logging_returns_same_optimization_loss_object(self):
        loss = torch.tensor(4.25, requires_grad=True)
        with mock.patch('utils.regression_trainer.logging.info'):
            returned = log_rich_loss_diagnostic(
                1, 0, 2, None, None, None, None, loss
            )
        self.assertIs(returned, loss)


class ResumeAndCliTests(unittest.TestCase):
    def test_rich_cli_is_opt_in_with_zero_weight_and_diagnostic_defaults(self):
        with mock.patch('sys.argv', ['train.py']):
            args = parse_arg()
        self.assertIsNone(args.rich_prompt_bank)
        self.assertEqual(args.rich_consistency_weight, 0.0)
        self.assertEqual(args.rich_loss_diagnostic_steps, 0)

    def test_resume_rejects_incompatible_rich_configuration(self):
        current = {
            'prompt_bank_fingerprint': 'sha256:a',
            'selected_image_fingerprint': 'sha256:b',
            'protocol_version': RICH_PROMPT_PROTOCOL,
            'rich_consistency_weight': 0.0,
            'train_samples': 1000,
            'train_subset_seed': 3407,
        }
        checkpoint = dict(current)
        checkpoint['train_subset_seed'] = 999
        with self.assertRaisesRegex(
            RichPromptTrainingError, 'train_subset_seed'
        ):
            validate_resume_rich_config(current, checkpoint)

    def test_checkpoint_provenance_contains_required_rich_fields(self):
        bank = ValidatedRichPromptBank(
            path=str(Path('prompt-bank.json').absolute()),
            file_fingerprint='sha256:bank',
            selected_image_fingerprint=PILOT_FINGERPRINT,
            protocol_version=RICH_PROMPT_PROTOCOL,
            records={},
        )
        config = build_rich_checkpoint_config(
            bank,
            consistency_weight=0.25,
            train_samples=1000,
            train_subset_seed=3407,
        )
        self.assertEqual(config['prompt_bank_filename'], 'prompt-bank.json')
        self.assertEqual(config['prompt_bank_fingerprint'], 'sha256:bank')
        self.assertEqual(
            config['selected_image_fingerprint'], PILOT_FINGERPRINT
        )
        self.assertEqual(config['protocol_version'], RICH_PROMPT_PROTOCOL)
        self.assertEqual(config['rich_consistency_weight'], 0.25)
        self.assertEqual(config['train_samples'], 1000)
        self.assertEqual(config['train_subset_seed'], 3407)

    def test_resume_rejects_before_mutating_model_or_optimizer(self):
        trainer = Reg_Trainer.__new__(Reg_Trainer)
        trainer.rich_prompt_config = {
            'prompt_bank_fingerprint': 'sha256:current',
            'selected_image_fingerprint': 'sha256:subset',
            'protocol_version': RICH_PROMPT_PROTOCOL,
            'rich_consistency_weight': 0.0,
            'train_samples': 1000,
            'train_subset_seed': 3407,
        }
        trainer.model = mock.Mock()
        trainer.optimizer = mock.Mock()
        checkpoint_config = dict(trainer.rich_prompt_config)
        checkpoint_config['rich_consistency_weight'] = 0.5
        checkpoint = {
            'rich_prompt_config': checkpoint_config,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'epoch': 0,
        }
        with mock.patch(
            'utils.regression_trainer.load_trusted_legacy_checkpoint',
            return_value=checkpoint,
        ):
            with self.assertRaisesRegex(
                RichPromptTrainingError, 'rich_consistency_weight'
            ):
                trainer._load_training_checkpoint('resume.tar')
        trainer.model.load_state_dict.assert_not_called()
        trainer.optimizer.load_state_dict.assert_not_called()


if __name__ == '__main__':
    unittest.main()
