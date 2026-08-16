import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from datasets.dataset import (
    ObjectCount,
    assemble_2x2_points,
    crop_points,
    horizontal_flip_points,
    resize_points,
)
from losses.dumlo import generate_discrete_map
from utils.regression_trainer import train_collate


class _Tokenizer:
    def __call__(self, prompt, add_special_tokens=False, return_tensors='pt'):
        return {'input_ids': torch.tensor([[1]])}


def _build_fsc147(root):
    fsc = root / 'FSC_147'
    images = root / 'images_384_VarV2'
    density = root / 'gt_density_map_adaptive_384_VarV2'
    fsc.mkdir()
    images.mkdir()
    density.mkdir()
    names = ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg']
    classes = {'a.jpg': 'cat', 'b.jpg': 'cat', 'c.jpg': 'dog', 'd.jpg': 'cat'}
    annotations = {
        'a.jpg': {'points': [[1.0, 1.0]]},
        'b.jpg': {'points': [[2.0, 1.0]]},
        'c.jpg': {'points': [[3.0, 3.0]]},
        'd.jpg': {'points': [[1.0, 2.0]]},
    }
    (fsc / 'Train_Test_Val_FSC_147.json').write_text(json.dumps({
        'train': names, 'val': names, 'test': names
    }))
    (fsc / 'annotation_FSC147_384.json').write_text(json.dumps(annotations))
    (fsc / 'ImageClasses_FSC147.txt').write_text(''.join(
        '{}\t{}\n'.format(name, classes[name]) for name in names
    ))
    for index, name in enumerate(names):
        Image.new('RGB', (8, 8), color=(index, index, index)).save(images / name)
        np.save(density / name.replace('.jpg', '.npy'), np.ones((8, 8)))
    return images


class DUMLODatasetTests(unittest.TestCase):
    def test_point_coordinate_helpers(self):
        points = torch.tensor([[2.0, 3.0], [7.0, 7.0], [8.0, 8.0]])
        cropped = crop_points(points, top=2, left=1, height=6, width=7)
        self.assertTrue(torch.equal(
            cropped, torch.tensor([[1.0, 1.0], [6.0, 5.0]])
        ))
        resized = resize_points(cropped, 7, 6, 14, 12)
        self.assertTrue(torch.equal(
            resized, torch.tensor([[2.0, 2.0], [12.0, 10.0]])
        ))
        flipped = horizontal_flip_points(resized, 14)
        self.assertTrue(torch.equal(
            flipped, torch.tensor([[11.0, 2.0], [1.0, 10.0]])
        ))

    def test_2x2_translation_follows_shuffled_tile_order(self):
        tiles = [
            {'points': torch.tensor([[1.0, 2.0]])},
            {'points': torch.empty(0, 2)},
            {'points': torch.tensor([[2.0, 1.0]])},
            {'points': torch.tensor([[1.0, 1.0]])},
        ]
        result = assemble_2x2_points(tiles, 4)
        expected = torch.tensor([[1.0, 2.0], [2.0, 5.0], [5.0, 5.0]])
        self.assertTrue(torch.equal(result, expected))

    def test_collate_retains_variable_length_points(self):
        common = (
            torch.zeros(3, 8, 8), torch.zeros(1, 4, 4), 'cat',
            torch.zeros(77), torch.zeros(1, 1, 1)
        )
        result = train_collate([
            common + (torch.tensor([[1.0, 1.0]]),),
            common + (torch.empty(0, 2),),
        ])
        self.assertEqual(len(result), 6)
        self.assertIsInstance(result[5], list)
        self.assertEqual([len(points) for points in result[5]], [1, 0])

    def test_train_transform_applies_resize_crop_and_flip_to_points(self):
        dataset = ObjectCount.__new__(ObjectCount)
        dataset.crop_size = 8
        dataset.down_ratio = 2
        dataset.transform = lambda image: torch.from_numpy(
            np.asarray(image).copy()
        )
        image = Image.new('RGB', (8, 8))
        density = np.ones((8, 8))
        attention = np.ones((8, 8))
        with mock.patch(
            'datasets.dataset.random.random',
            side_effect=[1.0, 1.0, 1.0],
        ), mock.patch(
            'datasets.dataset.random_crop',
            return_value=(2, 4, 8, 8),
        ):
            transformed = dataset.train_transform_density(
                image, density, attention,
                torch.tensor([[4.0, 3.0]])
            )
        self.assertTrue(torch.equal(
            transformed[3], torch.tensor([[3.0, 4.0]])
        ))

    def test_negative_prompt_has_no_points_and_same_class_preserves_points(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = _build_fsc147(root)
            dataset = ObjectCount(
                str(root), crop_size=8, downsample_ratio=2,
                method='train', concat_size=4, tokenizer=_Tokenizer(),
                return_points=True,
            )
            with mock.patch(
                'datasets.dataset.random.random',
                side_effect=[0.0, 0.9, 0.0, 0.0],
            ), mock.patch(
                'datasets.dataset.random.sample',
                return_value=[str(images / 'c.jpg')],
            ), mock.patch('datasets.dataset.random.randint', return_value=0):
                negative = dataset[0]
            self.assertEqual(negative[2], 'dog')
            self.assertEqual(negative[5].shape, (0, 2))

            with mock.patch(
                'datasets.dataset.random.random',
                side_effect=[0.0, 0.9, 0.0, 0.0],
            ), mock.patch(
                'datasets.dataset.random.sample',
                return_value=[str(images / 'b.jpg')],
            ), mock.patch('datasets.dataset.random.randint', return_value=0):
                same_class = dataset[0]
            self.assertEqual(same_class[2], 'cat')
            self.assertTrue(torch.equal(
                same_class[5], torch.tensor([[1.0, 1.0]])
            ))

    def test_full_concat_tracks_only_target_class_tiles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = _build_fsc147(root)
            dataset = ObjectCount(
                str(root), crop_size=8, downsample_ratio=2,
                method='train', concat_size=4, tokenizer=_Tokenizer(),
                return_points=True,
            )
            crop_results = [
                (0, 0, 4, 4), (0, 0, 4, 4), (0, 0, 4, 4),
                (0, 0, 4, 4), (0, 0, 8, 8),
            ]
            with mock.patch(
                'datasets.dataset.random.random',
                side_effect=[0.0, 0.0, 0.0, 0.0],
            ), mock.patch(
                'datasets.dataset.random.sample',
                return_value=[
                    str(images / 'b.jpg'), str(images / 'c.jpg'),
                    str(images / 'd.jpg'),
                ],
            ), mock.patch(
                'datasets.dataset.random_crop', side_effect=crop_results
            ), mock.patch(
                'datasets.dataset.random.shuffle',
                side_effect=lambda values: values.reverse(),
            ):
                sample = dataset[0]
            expected = torch.tensor([[1.0, 2.0], [2.0, 5.0], [5.0, 5.0]])
            self.assertTrue(torch.equal(sample[5], expected))
            discrete = generate_discrete_map(sample[5], 4, 4, 8, 8)
            self.assertEqual(discrete.sum().item(), len(sample[5]))

    def test_point_tracking_does_not_change_python_rng_sequence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_fsc147(root)
            baseline = ObjectCount(
                str(root), 8, 2, method='train', concat_size=4,
                tokenizer=_Tokenizer(), return_points=False,
            )
            dumlo = ObjectCount(
                str(root), 8, 2, method='train', concat_size=4,
                tokenizer=_Tokenizer(), return_points=True,
            )
            random.seed(1234)
            np.random.seed(5678)
            torch.manual_seed(9012)
            baseline_sample = baseline[0]
            baseline_state = random.getstate()
            baseline_numpy_state = np.random.get_state()
            baseline_torch_state = torch.get_rng_state().clone()
            random.seed(1234)
            np.random.seed(5678)
            torch.manual_seed(9012)
            dumlo_sample = dumlo[0]
            dumlo_state = random.getstate()
            dumlo_numpy_state = np.random.get_state()
            dumlo_torch_state = torch.get_rng_state()
            self.assertEqual(baseline_state, dumlo_state)
            self.assertEqual(
                baseline_numpy_state[1].tolist(),
                dumlo_numpy_state[1].tolist(),
            )
            self.assertTrue(torch.equal(
                baseline_torch_state, dumlo_torch_state
            ))
            for baseline_value, dumlo_value in zip(
                    baseline_sample, dumlo_sample[:5]):
                if torch.is_tensor(baseline_value):
                    self.assertTrue(torch.equal(
                        baseline_value, dumlo_value
                    ))
                else:
                    self.assertEqual(baseline_value, dumlo_value)


if __name__ == '__main__':
    unittest.main()
