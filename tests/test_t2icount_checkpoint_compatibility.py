import tempfile
import unittest
from pathlib import Path

import torch

from models.build import (
    LEGACY_NONPERSISTENT_BUFFER_KEYS,
    load_t2icount_checkpoint,
)
from utils.checkpoints import load_trusted_legacy_checkpoint


LEGACY_POSITION_IDS_KEY = (
    "clip.transformer.text_model.embeddings.position_ids"
)


class _TinyCount(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.learned_weight = torch.nn.Parameter(torch.tensor([0.0]))


class _LegacyTinyCount(_TinyCount):
    def __init__(self):
        super().__init__()
        self.clip = torch.nn.Module()
        self.clip.transformer = torch.nn.Module()
        self.clip.transformer.text_model = torch.nn.Module()
        self.clip.transformer.text_model.embeddings = torch.nn.Module()
        self.clip.transformer.text_model.embeddings.register_buffer(
            "position_ids", torch.arange(77).unsqueeze(0)
        )


class T2ICountCheckpointCompatibilityTests(unittest.TestCase):
    def _save_checkpoint(self, directory, state_dict, wrapped=False):
        path = Path(directory) / "tiny_t2icount.pth"
        payload = {"model_state_dict": state_dict} if wrapped else state_dict
        torch.save(payload, str(path))
        return path

    def test_old_checkpoint_key_is_removed_for_modern_model(self):
        self.assertEqual(
            LEGACY_NONPERSISTENT_BUFFER_KEYS,
            frozenset({LEGACY_POSITION_IDS_KEY}),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source = _LegacyTinyCount()
            source.learned_weight.data.fill_(7.0)
            state = source.state_dict()
            checkpoint = self._save_checkpoint(temp_dir, state)

            target = _TinyCount()
            load_t2icount_checkpoint(target, checkpoint)

            self.assertTrue(
                torch.equal(target.learned_weight, source.learned_weight)
            )
            on_disk = load_trusted_legacy_checkpoint(
                str(checkpoint), map_location="cpu"
            )
            self.assertIn(LEGACY_POSITION_IDS_KEY, on_disk)

    def test_modern_checkpoint_key_is_populated_for_legacy_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = _TinyCount()
            source.learned_weight.data.fill_(5.0)
            checkpoint = self._save_checkpoint(
                temp_dir, source.state_dict(), wrapped=True
            )

            target = _LegacyTinyCount()
            expected_position_ids = target.state_dict()[
                LEGACY_POSITION_IDS_KEY
            ].clone()
            load_t2icount_checkpoint(target, checkpoint)

            self.assertTrue(
                torch.equal(target.learned_weight, source.learned_weight)
            )
            self.assertTrue(torch.equal(
                target.state_dict()[LEGACY_POSITION_IDS_KEY],
                expected_position_ids,
            ))
            on_disk = load_trusted_legacy_checkpoint(
                str(checkpoint), map_location="cpu"
            )
            self.assertNotIn(
                LEGACY_POSITION_IDS_KEY, on_disk["model_state_dict"]
            )

    def test_checkpoint_value_is_kept_when_both_have_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = _LegacyTinyCount()
            source.clip.transformer.text_model.embeddings.position_ids.fill_(
                42
            )
            checkpoint = self._save_checkpoint(temp_dir, source.state_dict())

            target = _LegacyTinyCount()
            load_t2icount_checkpoint(target, checkpoint)

            self.assertTrue(torch.equal(
                target.state_dict()[LEGACY_POSITION_IDS_KEY],
                source.state_dict()[LEGACY_POSITION_IDS_KEY],
            ))

    def test_neither_checkpoint_nor_model_has_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = _TinyCount()
            source.learned_weight.data.fill_(3.0)
            checkpoint = self._save_checkpoint(temp_dir, source.state_dict())

            target = _TinyCount()
            load_t2icount_checkpoint(target, checkpoint)

            self.assertTrue(
                torch.equal(target.learned_weight, source.learned_weight)
            )

    def test_strict_load_rejects_unrelated_unexpected_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = _TinyCount().state_dict()
            state["unrelated.unexpected_key"] = torch.tensor([1.0])
            checkpoint = self._save_checkpoint(temp_dir, state)

            with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                load_t2icount_checkpoint(_TinyCount(), checkpoint)

    def test_strict_load_rejects_missing_learned_weight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = {
                LEGACY_POSITION_IDS_KEY: torch.arange(77).unsqueeze(0),
            }
            checkpoint = self._save_checkpoint(temp_dir, state)

            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_t2icount_checkpoint(_TinyCount(), checkpoint)


if __name__ == "__main__":
    unittest.main()
