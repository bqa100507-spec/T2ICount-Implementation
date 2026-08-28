import base64
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import generate_rich_prompt_bank as rich


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, model, image_bytes, mime_type, prompt):
        self.calls.append((model, image_bytes, mime_type, prompt))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _RateLimitError(Exception):
    status_code = 429


class _ConnectError(Exception):
    pass


class _RecordingInteractions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="The Stew pot is dark and centered on the stove.",
            steps=[{"type": "thought", "content": "must not be extracted"}],
        )


class _UnavailableModelError(Exception):
    status_code = 404


class _FailingInteractions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise _UnavailableModelError("selected model unavailable")


def _create_fixture(root, entries=None):
    entries = entries or {
        "20.jpg": {"class": "Stew pot", "count": 17},
    }
    asset_root = root / "assets"
    image_dir = asset_root / "datasets" / "FSC147" / "images_384_VarV2"
    image_dir.mkdir(parents=True)
    for image_name in entries:
        (image_dir / image_name).write_bytes(b"image-bytes-" + image_name.encode())
    metadata_path = root / "FSC-147-S.json"
    metadata_path.write_text(json.dumps(entries), encoding="utf-8")
    return asset_root, metadata_path


def _config(root, asset_root, metadata_path, **overrides):
    values = {
        "asset_root": asset_root,
        "metadata_path": metadata_path,
        "output_path": root / "prompts" / "bank.json",
        "model": rich.DEFAULT_MODEL,
        "max_samples": None,
        "max_retries": 0,
        "request_delay": 0.0,
        "overwrite": False,
        "dry_run": False,
    }
    values.update(overrides)
    return rich.GenerationConfig(**values)


class RichPromptGeneratorTests(unittest.TestCase):
    def test_cli_default_model_and_model_safe_output(self):
        args = rich.parse_args([])

        self.assertEqual(args.model, "gemma-4-26b-a4b-it")
        self.assertEqual(
            args.output,
            "prompts/fsc147s_gemma_4_26b_a4b_it_v2.json",
        )

    def test_cli_parses_explicit_model_and_derives_matching_output(self):
        args = rich.parse_args(["--model", "gemini-3.6-flash"])

        self.assertEqual(args.model, "gemini-3.6-flash")
        self.assertEqual(
            args.output,
            "prompts/fsc147s_gemini_3_6_flash_v2.json",
        )

    def test_metadata_loading_preserves_order_and_discards_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entries = {
                "20.jpg": {"class": "Stew pot", "count": 17},
                "27.jpg": {"class": "Stew pot", "count": 3},
            }
            _, metadata_path = _create_fixture(root, entries)

            samples = rich.load_fsc147s_metadata(metadata_path)

            self.assertEqual(
                samples,
                [
                    rich.Sample("20.jpg", "Stew pot"),
                    rich.Sample("27.jpg", "Stew pot"),
                ],
            )
            self.assertFalse(hasattr(samples[0], "count"))

    def test_image_path_resolution_uses_external_fsc147_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, _ = _create_fixture(root)

            image_path, mime_type = rich.resolve_image_path(asset_root, "20.jpg")

            self.assertEqual(
                image_path,
                asset_root
                / "datasets"
                / "FSC147"
                / "images_384_VarV2"
                / "20.jpg",
            )
            self.assertEqual(mime_type, "image/jpeg")

    def test_generation_prompt_contains_exact_class_and_fixed_protocol(self):
        prompt = rich.build_generation_prompt("Stew pot")

        self.assertIn('The target category is: "Stew pot".', prompt)
        self.assertIn('exact category name "Stew pot" verbatim', prompt)
        self.assertIn("Do not state or imply how many", prompt)
        self.assertEqual(prompt.count("Stew pot"), 2)

    def test_interactions_request_has_exact_model_text_and_image_payload(self):
        interactions = _RecordingInteractions()
        adapter = rich.GoogleGenAIClient.__new__(rich.GoogleGenAIClient)
        adapter._client = SimpleNamespace(interactions=interactions)
        image_bytes = b"local-image-bytes"
        protocol = rich.build_generation_prompt("Stew pot")

        selected_model = "gemini-3.6-flash"
        result = adapter.generate(
            selected_model,
            image_bytes,
            "image/png",
            protocol,
        )

        self.assertEqual(
            result,
            "The Stew pot is dark and centered on the stove.",
        )
        self.assertEqual(len(interactions.calls), 1)
        request = interactions.calls[0]
        self.assertEqual(request["model"], selected_model)
        self.assertIs(request["store"], False)
        inputs = request["input"]
        text_inputs = [item for item in inputs if item.get("type") == "text"]
        image_inputs = [item for item in inputs if item.get("type") == "image"]
        self.assertEqual(text_inputs, [{"type": "text", "text": protocol}])
        self.assertEqual(len(image_inputs), 1)
        self.assertEqual(image_inputs[0]["mime_type"], "image/png")
        self.assertEqual(
            image_inputs[0]["data"],
            base64.b64encode(image_bytes).decode("ascii"),
        )
        self.assertEqual(len(inputs), 2)

    def test_unavailable_selected_model_has_no_automatic_fallback(self):
        interactions = _FailingInteractions()
        adapter = rich.GoogleGenAIClient.__new__(rich.GoogleGenAIClient)
        adapter._client = SimpleNamespace(interactions=interactions)
        selected_model = "gemini-3.6-flash"

        with self.assertRaises(_UnavailableModelError):
            adapter.generate(
                selected_model,
                b"local-image-bytes",
                "image/jpeg",
                rich.build_generation_prompt("Stew pot"),
            )

        self.assertEqual(len(interactions.calls), 1)
        self.assertEqual(interactions.calls[0]["model"], selected_model)

    def test_prompt_bank_metadata_records_model_and_interactions_api(self):
        selected_model = "gemini-3.6-flash"
        metadata = rich.new_prompt_bank(
            selected_model,
            "2026-08-28T00:00:00Z",
        )["metadata"]

        self.assertEqual(metadata["generator"], selected_model)
        self.assertEqual(metadata["api"], "interactions")
        self.assertEqual(metadata["protocol_version"], "rich-prompt-phase1-v2")
        self.assertEqual(
            metadata["generalization_rule"],
            (
                "case-insensitive exact class replacement: singular-like class "
                "-> target object; plural-like class -> target objects"
            ),
        )

    def test_missing_api_key_fails_before_client_creation(self):
        factory = mock.Mock()
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(stderr):
            return_code = rich.main(
                [
                    "--asset-root",
                    "unused",
                    "--metadata",
                    "unused.json",
                    "--output",
                    "unused-output.json",
                ],
                client_factory=factory,
            )

        self.assertEqual(return_code, 2)
        self.assertIn("GEMINI_API_KEY", stderr.getvalue())
        factory.assert_not_called()

    def test_cli_selected_model_reaches_client_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, metadata_path = _create_fixture(root)
            output_path = root / "prompts" / "bank.json"
            selected_model = "gemini-3.6-flash"
            client = _FakeClient(
                ["The Stew pot is dark and centered on the stove."]
            )

            with mock.patch.dict(
                os.environ,
                {"GEMINI_API_KEY": "test-key"},
                clear=True,
            ):
                return_code = rich.main(
                    [
                        "--asset-root",
                        str(asset_root),
                        "--metadata",
                        str(metadata_path),
                        "--output",
                        str(output_path),
                        "--model",
                        selected_model,
                        "--request-delay",
                        "0",
                        "--max-retries",
                        "0",
                    ],
                    client_factory=lambda _: client,
                )

            saved = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(return_code, 0)
            self.assertEqual(client.calls[0][0], selected_model)
            self.assertEqual(
                saved["metadata"]["generator"],
                selected_model,
            )

    def test_dry_run_needs_no_key_makes_no_calls_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, metadata_path = _create_fixture(root)
            output_path = root / "prompts" / "bank.json"
            factory = mock.Mock()
            stdout = io.StringIO()

            with mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(stdout):
                return_code = rich.main(
                    [
                        "--asset-root",
                        str(asset_root),
                        "--metadata",
                        str(metadata_path),
                        "--output",
                        str(output_path),
                        "--max-samples",
                        "1",
                        "--dry-run",
                    ],
                    client_factory=factory,
                )

            self.assertEqual(return_code, 0)
            factory.assert_not_called()
            self.assertFalse(output_path.exists())
            self.assertIn("zero API calls, no output writes", stdout.getvalue())

    def test_valid_detailed_prompt_passes_with_whitespace_cleanup(self):
        detailed = rich.validate_detailed_description(
            "  The   Stew pot is dark, round, and centered on the stove.  ",
            "Stew pot",
        )

        self.assertEqual(
            detailed,
            "The Stew pot is dark, round, and centered on the stove.",
        )

    def test_digits_are_rejected(self):
        with self.assertRaisesRegex(rich.DescriptionValidationError, "digit"):
            rich.validate_detailed_description(
                "The Stew pot has 2 dark handles.",
                "Stew pot",
            )

    def test_number_words_are_rejected(self):
        for word in ("zero", "three", "twenty", "hundred", "thousand"):
            with self.subTest(word=word), self.assertRaisesRegex(
                rich.DescriptionValidationError,
                "quantity leakage",
            ):
                rich.validate_detailed_description(
                    "The Stew pot sits beside {} plates.".format(word),
                    "Stew pot",
                )

    def test_vague_quantity_words_are_rejected(self):
        for word in ("many", "several", "few", "numerous", "multiple"):
            with self.subTest(word=word), self.assertRaisesRegex(
                rich.DescriptionValidationError,
                "quantity leakage",
            ):
                rich.validate_detailed_description(
                    "The {} Stew pot shapes are dark and round.".format(word),
                    "Stew pot",
                )

    def test_indefinite_article_and_individual_cues_are_rejected(self):
        for description in (
            "The people include a dark-haired individual near the peppers.",
            "The people include an isolated individual near the doorway.",
        ):
            with self.subTest(description=description), self.assertRaisesRegex(
                rich.DescriptionValidationError,
                "implicit count leakage",
            ):
                rich.validate_detailed_description(description, "people")

    def test_implicit_instance_count_terms_are_rejected(self):
        for term in (
            "single",
            "pair",
            "couple",
            "group",
            "crowd",
            "cluster",
        ):
            with self.subTest(term=term), self.assertRaisesRegex(
                rich.DescriptionValidationError,
                "implicit count leakage: {}".format(term),
            ):
                rich.validate_detailed_description(
                    "The people appear as {} silhouettes near the doorway."
                    .format(term),
                    "people",
                )

    def test_implicit_leakage_matching_respects_word_boundaries(self):
        detailed = rich.validate_detailed_description(
            "The Spatula shows metal, handle contours, and orange coloring.",
            "Spatula",
        )

        self.assertEqual(
            detailed,
            "The Spatula shows metal, handle contours, and orange coloring.",
        )

    def test_missing_target_class_is_rejected(self):
        with self.assertRaisesRegex(
            rich.DescriptionValidationError,
            "target class missing",
        ):
            rich.validate_detailed_description(
                "The dark cookware is centered on the stove.",
                "Stew pot",
            )

    def test_multiline_and_multiple_sentences_are_rejected(self):
        with self.assertRaisesRegex(rich.DescriptionValidationError, "multiline"):
            rich.validate_detailed_description(
                "The Stew pot is dark.\nIt is centered.",
                "Stew pot",
            )
        with self.assertRaisesRegex(
            rich.DescriptionValidationError,
            "multiple sentences",
        ):
            rich.validate_detailed_description(
                "The Stew pot is dark. It is centered.",
                "Stew pot",
            )

    def test_generalization_is_deterministic(self):
        detailed = "The Stew pot is dark and the Stew pot has curved handles."

        first = rich.generalize_description(detailed, "Stew pot")
        second = rich.generalize_description(detailed, "Stew pot")

        self.assertEqual(
            first,
            (
                "The target object is dark and the target object has curved "
                "handles."
            ),
        )
        self.assertEqual(first, second)

    def test_generalization_uses_plurality_aware_target_phrase(self):
        cases = (
            ("Spatula", "The Spatula is metallic.", "The target object is metallic."),
            (
                "Stew pot",
                "The Stew pot has steel surfaces.",
                "The target object has steel surfaces.",
            ),
            (
                "people",
                "The people have dark clothing.",
                "The target objects have dark clothing.",
            ),
            (
                "strawberries",
                "The strawberries have red surfaces.",
                "The target objects have red surfaces.",
            ),
        )
        for class_name, detailed, expected in cases:
            with self.subTest(class_name=class_name):
                self.assertEqual(
                    rich.generalize_description(detailed, class_name),
                    expected,
                )

    def test_generalization_replacement_is_case_insensitive(self):
        self.assertEqual(
            rich.generalize_description(
                "The STEW POT is dark and round.",
                "Stew pot",
            ),
            "The target object is dark and round.",
        )

    def test_v1_prompt_bank_is_rejected_by_v2_compatibility_check(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "v1-bank.json"
            bank = rich.new_prompt_bank(
                rich.DEFAULT_MODEL,
                "2026-08-28T00:00:00Z",
            )
            bank["metadata"]["protocol_version"] = "rich-prompt-phase1-v1"
            rich.atomic_save_prompt_bank(bank, output_path)

            with self.assertRaisesRegex(
                rich.PromptBankError,
                "incompatible protocol_version",
            ):
                rich.load_prompt_bank(output_path, rich.DEFAULT_MODEL)

    def test_gemma_bank_rejects_gemini_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "gemma-bank.json"
            bank = rich.new_prompt_bank(
                "gemma-4-26b-a4b-it",
                "2026-08-28T00:00:00Z",
            )
            rich.atomic_save_prompt_bank(bank, output_path)

            with self.assertRaisesRegex(
                rich.PromptBankError,
                "incompatible generator",
            ):
                rich.load_prompt_bank(output_path, "gemini-3.6-flash")

    def test_gemini_bank_rejects_gemma_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "gemini-bank.json"
            bank = rich.new_prompt_bank(
                "gemini-3.6-flash",
                "2026-08-28T00:00:00Z",
            )
            rich.atomic_save_prompt_bank(bank, output_path)

            with self.assertRaisesRegex(
                rich.PromptBankError,
                "incompatible generator",
            ):
                rich.load_prompt_bank(output_path, "gemma-4-26b-a4b-it")

    def test_resume_skips_successful_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, metadata_path = _create_fixture(root)
            config = _config(root, asset_root, metadata_path)
            bank = rich.new_prompt_bank(
                config.model,
                "2026-08-28T00:00:00Z",
            )
            bank["prompts"]["20.jpg"] = {
                "class": "Stew pot",
                "detailed": "The Stew pot is dark and round.",
                "generalized": "The target object is dark and round.",
                "status": "ok",
                "attempts": 1,
            }
            rich.atomic_save_prompt_bank(bank, config.output_path)
            client = _FakeClient([])

            summary = rich.run_generation(config, client=client, emit=lambda _: None)

            self.assertEqual(summary.skipped, 1)
            self.assertEqual(client.calls, [])

    def test_overwrite_regenerates_successful_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, metadata_path = _create_fixture(root)
            config = _config(root, asset_root, metadata_path, overwrite=True)
            bank = rich.new_prompt_bank(
                config.model,
                "2026-08-28T00:00:00Z",
            )
            bank["prompts"]["20.jpg"] = {
                "class": "Stew pot",
                "detailed": "The Stew pot was previously described.",
                "generalized": "The target object was previously described.",
                "status": "ok",
                "attempts": 1,
            }
            rich.atomic_save_prompt_bank(bank, config.output_path)
            client = _FakeClient(
                ["The Stew pot is glossy, round, and centered on the stove."]
            )

            summary = rich.run_generation(config, client=client, emit=lambda _: None)
            saved = json.loads(config.output_path.read_text(encoding="utf-8"))

            self.assertEqual(summary.generated, 1)
            self.assertEqual(len(client.calls), 1)
            self.assertIn("glossy", saved["prompts"]["20.jpg"]["detailed"])

    def test_generalization_introduces_no_second_gemini_request(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, metadata_path = _create_fixture(root)
            config = _config(root, asset_root, metadata_path)
            client = _FakeClient(
                ["The Stew pot has glossy steel surfaces and curved handles."]
            )

            summary = rich.run_generation(config, client=client, emit=lambda _: None)
            saved = json.loads(config.output_path.read_text(encoding="utf-8"))

            self.assertEqual(summary.generated, 1)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(
                saved["prompts"]["20.jpg"]["generalized"],
                "The target object has glossy steel surfaces and curved handles.",
            )

    def test_failed_generation_never_creates_fallback_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, metadata_path = _create_fixture(root)
            config = _config(
                root,
                asset_root,
                metadata_path,
                max_retries=1,
            )
            client = _FakeClient(
                [
                    "Many Stew pot shapes are visible.",
                    "Several Stew pot shapes are visible.",
                ]
            )

            summary = rich.run_generation(config, client=client, emit=lambda _: None)
            saved = json.loads(config.output_path.read_text(encoding="utf-8"))

            self.assertEqual(summary.failed, 1)
            self.assertNotIn("20.jpg", saved["prompts"])
            self.assertEqual(saved["failures"]["20.jpg"]["attempts"], 2)
            self.assertNotIn("detailed", saved["failures"]["20.jpg"])

    def test_output_contains_neither_gt_count_nor_api_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, metadata_path = _create_fixture(root)
            output_path = root / "prompts" / "bank.json"
            client = _FakeClient(
                ["The Stew pot is glossy, round, and centered on the stove."]
            )
            secret = "test-secret-that-must-not-be-persisted"
            stdout = io.StringIO()
            stderr = io.StringIO()

            with mock.patch.dict(
                os.environ,
                {"GEMINI_API_KEY": secret},
                clear=True,
            ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                return_code = rich.main(
                    [
                        "--asset-root",
                        str(asset_root),
                        "--metadata",
                        str(metadata_path),
                        "--output",
                        str(output_path),
                        "--request-delay",
                        "0",
                        "--max-retries",
                        "0",
                    ],
                    client_factory=lambda _: client,
                )

            saved = json.loads(output_path.read_text(encoding="utf-8"))
            serialized = json.dumps(saved)

            self.assertEqual(return_code, 0)
            self.assertNotIn(secret, serialized)
            self.assertNotIn(secret, stdout.getvalue())
            self.assertNotIn(secret, stderr.getvalue())
            self.assertNotIn("count", saved["prompts"]["20.jpg"])
            self.assertNotIn(17, saved["prompts"]["20.jpg"].values())

    def test_every_success_is_saved_atomically_and_no_temp_file_remains(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entries = {
                "20.jpg": {"class": "Stew pot", "count": 17},
                "27.jpg": {"class": "Stew pot", "count": 3},
            }
            asset_root, metadata_path = _create_fixture(root, entries)
            config = _config(root, asset_root, metadata_path)
            client = _FakeClient(
                [
                    "The Stew pot is dark and centered on the stove.",
                    "The Stew pot is silver and positioned on the counter.",
                ]
            )

            with mock.patch.object(
                rich,
                "atomic_save_prompt_bank",
                wraps=rich.atomic_save_prompt_bank,
            ) as save:
                summary = rich.run_generation(
                    config,
                    client=client,
                    emit=lambda _: None,
                )

            saved = json.loads(config.output_path.read_text(encoding="utf-8"))
            self.assertEqual(summary.generated, 2)
            self.assertEqual(save.call_count, 2)
            self.assertEqual(list(saved["prompts"]), ["20.jpg", "27.jpg"])
            self.assertEqual(list(config.output_path.parent.glob("*.tmp")), [])

    def test_transient_api_error_is_retried(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset_root, metadata_path = _create_fixture(root)
            config = _config(
                root,
                asset_root,
                metadata_path,
                max_retries=1,
            )
            client = _FakeClient(
                [
                    _RateLimitError("sensitive response detail"),
                    "The Stew pot is dark and centered on the stove.",
                ]
            )
            progress = []

            summary = rich.run_generation(config, client=client, emit=progress.append)

            self.assertEqual(summary.generated, 1)
            self.assertEqual(len(client.calls), 2)
            self.assertIn("HTTP 429", progress[0])
            self.assertNotIn("sensitive response detail", "\n".join(progress))

    def test_network_error_class_is_transient(self):
        self.assertTrue(rich.is_transient_api_error(_ConnectError()))


if __name__ == "__main__":
    unittest.main()
