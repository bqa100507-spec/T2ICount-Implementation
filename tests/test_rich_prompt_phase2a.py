import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from tools import generate_rich_prompt_bank as rich


_CLASSES = (
    "apples",
    "pears",
    "oranges",
    "bananas",
    "grapes",
    "melons",
    "cherries",
    "peaches",
)


def _create_train_fixture(root):
    asset_root = root / "assets"
    dataset_root = asset_root / "datasets" / "FSC147"
    metadata_root = dataset_root / "FSC_147"
    image_root = dataset_root / "images_384_VarV2"
    metadata_root.mkdir(parents=True)
    image_root.mkdir(parents=True)

    train_images = ["train-{}.jpg".format(index) for index in range(8)]
    val_images = ["val-0.jpg", "val-1.jpg"]
    test_images = ["test-0.jpg"]
    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }
    (metadata_root / "Train_Test_Val_FSC_147.json").write_text(
        json.dumps(splits),
        encoding="utf-8",
    )

    all_images = train_images + val_images + test_images
    class_names = {}
    class_lines = []
    for index, image_name in enumerate(all_images):
        class_name = _CLASSES[index % len(_CLASSES)]
        class_names[image_name] = class_name
        class_lines.append("{}\t{}".format(image_name, class_name))
        (image_root / image_name).write_bytes(image_name.encode("ascii"))
    (metadata_root / "ImageClasses_FSC147.txt").write_text(
        "\n".join(class_lines) + "\n",
        encoding="utf-8",
    )
    return asset_root, splits, class_names


def _config(root, asset_root, **overrides):
    values = {
        "asset_root": asset_root,
        "metadata_path": root / "unused-fsc147s.json",
        "output_path": root / "prompts" / "train-bank.json",
        "model": rich.DEFAULT_MODEL,
        "max_samples": None,
        "max_retries": 0,
        "request_delay": 0.0,
        "overwrite": False,
        "dry_run": False,
        "split": "train",
        "train_samples": 5,
        "train_subset_seed": 3407,
        "concurrency": 1,
    }
    values.update(overrides)
    return rich.GenerationConfig(**values)


def _valid_description(class_name):
    return "{} appear glossy and colorful beside green leaves.".format(
        class_name.capitalize()
    )


class _RecordingClient:
    def __init__(self, failures=None, delays=None):
        self.failures = set(failures or ())
        self.delays = dict(delays or {})
        self.calls = []
        self.completions = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def generate(self, model, image_bytes, mime_type, prompt):
        image_name = image_bytes.decode("ascii")
        with self._lock:
            self.calls.append(image_name)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delays.get(image_name, 0.01))
            if image_name in self.failures:
                raise ValueError("private failure details")
            class_name = prompt.split('"', 2)[1]
            return _valid_description(class_name)
        finally:
            with self._lock:
                self.active -= 1
                self.completions.append(image_name)


def _contains_exact_key(value, forbidden_key):
    if isinstance(value, dict):
        return forbidden_key in value or any(
            _contains_exact_key(item, forbidden_key) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_exact_key(item, forbidden_key) for item in value)
    return False


class RichPromptPhase2ATests(unittest.TestCase):
    def test_same_sample_count_and_seed_give_identical_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, _, _ = _create_train_fixture(root)

            first = rich.load_fsc147_split_samples(asset_root, "train", 5, 3407)
            second = rich.load_fsc147_split_samples(asset_root, "train", 5, 3407)

            self.assertEqual(first, second)

    def test_different_seed_changes_selected_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, _, _ = _create_train_fixture(root)

            first = rich.load_fsc147_split_samples(asset_root, "train", 5, 3407)
            second = rich.load_fsc147_split_samples(asset_root, "train", 5, 3408)

            self.assertNotEqual(first, second)

    def test_generator_and_trainer_use_exact_shared_selector(self):
        from utils import regression_trainer

        self.assertIs(
            rich.select_train_subset_indices,
            regression_trainer.select_train_subset_indices,
        )

    def test_only_official_train_split_images_are_selected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, splits, _ = _create_train_fixture(root)

            samples = rich.load_fsc147_split_samples(asset_root, "train", 8, 3407)
            names = {sample.image_name for sample in samples}

            self.assertEqual(names, set(splits["train"]))
            self.assertTrue(names.isdisjoint(splits["val"] + splits["test"]))

    def test_class_names_come_from_official_class_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, _, class_names = _create_train_fixture(root)

            samples = rich.load_fsc147_split_samples(asset_root, "train", 8, 3407)

            self.assertEqual(
                [sample.class_name for sample in samples],
                [class_names[sample.image_name] for sample in samples],
            )

    def test_train_bank_has_provenance_and_no_ground_truth_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, _, _ = _create_train_fixture(root)
            config = _config(root, asset_root, train_samples=3)
            client = _RecordingClient(delays={})

            summary = rich.run_generation(config, client=client, emit=lambda _: None)
            bank = json.loads(config.output_path.read_text(encoding="utf-8"))
            metadata = bank["metadata"]

            self.assertEqual(summary.generated, 3)
            self.assertEqual(metadata["benchmark"], "FSC147")
            self.assertEqual(metadata["split"], "train")
            self.assertEqual(metadata["train_samples"], 3)
            self.assertEqual(metadata["train_subset_seed"], 3407)
            self.assertEqual(metadata["generator"], rich.DEFAULT_MODEL)
            self.assertEqual(metadata["api"], "interactions")
            self.assertEqual(metadata["protocol_version"], rich.PROTOCOL_VERSION)
            self.assertEqual(metadata["generalization_rule"], rich.GENERALIZATION_RULE)
            self.assertEqual(metadata["selected_sample_count"], 3)
            self.assertRegex(metadata["selected_image_fingerprint"], r"^sha256:[0-9a-f]{64}$")
            self.assertFalse(_contains_exact_key(bank, "count"))

    def test_fingerprint_is_deterministic_and_order_sensitive(self):
        samples = [rich.Sample("first.jpg", "apples"), rich.Sample("second.jpg", "pears")]

        first = rich.ordered_image_fingerprint(samples)
        second = rich.ordered_image_fingerprint(list(samples))
        reversed_fingerprint = rich.ordered_image_fingerprint(list(reversed(samples)))

        self.assertEqual(first, second)
        self.assertNotEqual(first, reversed_fingerprint)

    def test_resume_rejects_different_subset_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, _, _ = _create_train_fixture(root)
            original = _config(root, asset_root, train_samples=3)
            rich.run_generation(original, client=_RecordingClient(), emit=lambda _: None)
            changed = _config(
                root,
                asset_root,
                train_samples=3,
                train_subset_seed=3408,
            )
            client = _RecordingClient()

            with self.assertRaisesRegex(rich.PromptBankError, "incompatible train_subset_seed"):
                rich.run_generation(changed, client=client, emit=lambda _: None)
            self.assertEqual(client.calls, [])

    def test_resume_rejects_different_selected_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, _, _ = _create_train_fixture(root)
            config = _config(root, asset_root, train_samples=3)
            rich.run_generation(config, client=_RecordingClient(), emit=lambda _: None)
            bank = json.loads(config.output_path.read_text(encoding="utf-8"))
            bank["metadata"]["selected_image_fingerprint"] = "sha256:" + ("0" * 64)
            config.output_path.write_text(json.dumps(bank), encoding="utf-8")
            client = _RecordingClient()

            with self.assertRaisesRegex(
                rich.PromptBankError,
                "incompatible selected_image_fingerprint",
            ):
                rich.run_generation(config, client=client, emit=lambda _: None)
            self.assertEqual(client.calls, [])

    def test_concurrency_one_preserves_selected_request_order_and_delay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, _, _ = _create_train_fixture(root)
            config = _config(
                root,
                asset_root,
                train_samples=3,
                request_delay=0.25,
                concurrency=1,
            )
            selected = rich.load_fsc147_split_samples(asset_root, "train", 3, 3407)
            client = _RecordingClient(delays={name.image_name: 0 for name in selected})
            sleeps = []

            rich.run_generation(
                config,
                client=client,
                sleep=sleeps.append,
                emit=lambda _: None,
            )

            self.assertEqual(client.calls, [sample.image_name for sample in selected])
            self.assertEqual(client.max_active, 1)
            self.assertEqual(sleeps, [0.25, 0.25])

    def test_concurrency_never_exceeds_configured_in_flight_maximum(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, _, _ = _create_train_fixture(root)
            config = _config(root, asset_root, concurrency=3)
            client = _RecordingClient()

            rich.run_generation(config, client=client, emit=lambda _: None)

            self.assertGreater(client.max_active, 1)
            self.assertLessEqual(client.max_active, 3)

    def test_out_of_order_completion_persists_selected_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, _, _ = _create_train_fixture(root)
            config = _config(root, asset_root, train_samples=3, concurrency=3)
            selected = rich.load_fsc147_split_samples(asset_root, "train", 3, 3407)
            selected_names = [sample.image_name for sample in selected]
            delays = {
                selected_names[0]: 0.08,
                selected_names[1]: 0.01,
                selected_names[2]: 0.02,
            }
            client = _RecordingClient(delays=delays)

            rich.run_generation(config, client=client, emit=lambda _: None)
            bank = json.loads(config.output_path.read_text(encoding="utf-8"))

            self.assertNotEqual(client.completions, selected_names)
            self.assertEqual(list(bank["prompts"]), selected_names)

    def test_failed_request_does_not_discard_successful_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, _, _ = _create_train_fixture(root)
            config = _config(root, asset_root, train_samples=3, concurrency=3)
            selected = rich.load_fsc147_split_samples(asset_root, "train", 3, 3407)
            failed_name = selected[1].image_name
            client = _RecordingClient(failures={failed_name})

            summary = rich.run_generation(config, client=client, emit=lambda _: None)
            bank = json.loads(config.output_path.read_text(encoding="utf-8"))

            self.assertEqual(summary.generated, 2)
            self.assertEqual(summary.failed, 1)
            self.assertEqual(set(bank["failures"]), {failed_name})
            self.assertEqual(len(bank["prompts"]), 2)
            self.assertNotIn("private failure details", json.dumps(bank))

    def test_train_dry_run_has_no_key_calls_or_output_writes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, _, _ = _create_train_fixture(root)
            output_path = root / "prompts" / "dry-run.json"
            factory = mock.Mock()
            stdout = io.StringIO()

            with mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(stdout):
                return_code = rich.main(
                    [
                        "--asset-root",
                        str(asset_root),
                        "--split",
                        "train",
                        "--train-samples",
                        "3",
                        "--train-subset-seed",
                        "3407",
                        "--output",
                        str(output_path),
                        "--dry-run",
                    ],
                    client_factory=factory,
                )

            self.assertEqual(return_code, 0)
            factory.assert_not_called()
            self.assertFalse(output_path.exists())
            self.assertIn("selected_sample_count=3", stdout.getvalue())
            self.assertIn("ordered_selected_image_fingerprint=sha256:", stdout.getvalue())
            self.assertIn("zero API calls, no output writes", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
