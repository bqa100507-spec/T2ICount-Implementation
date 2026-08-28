import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import evaluate_rich_prompts as evaluator


def _write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _create_inputs(root, protocol=evaluator.REQUIRED_PROTOCOL):
    image_dir = root / "assets" / "datasets" / "FSC147" / "images_384_VarV2"
    image_dir.mkdir(parents=True)
    (image_dir / "214.jpg").write_bytes(b"synthetic-image")
    metadata_path = root / "FSC-147-S.json"
    prompt_bank_path = root / "prompt-bank.json"
    _write_json(
        metadata_path,
        {"214.jpg": {"class": "bottles", "count": 3}},
    )
    _write_json(
        prompt_bank_path,
        {
            "metadata": {
                "generator": "test-generator",
                "protocol_version": protocol,
            },
            "prompts": {
                "214.jpg": {
                    "class": "bottles",
                    "detailed": "  Dark bottles stand behind grapes.  ",
                    "generalized": "  Dark object stand behind grapes.  ",
                    "status": "ok",
                    "attempts": 1,
                    "count": 999,
                }
            },
            "failures": {},
        },
    )
    return metadata_path, prompt_bank_path, image_dir


def _sample(image_name="sample.jpg", gt_count=10.0):
    return evaluator.EvaluationSample(
        image_name=image_name,
        class_name="bottles",
        gt_count=gt_count,
        class_prompt="bottles",
        detailed_prompt="Dark bottles stand behind grapes.",
        generalized_prompt="Dark object stand behind grapes.",
        image_path=Path(image_name),
    )


class RichPromptEvaluatorTests(unittest.TestCase):
    def test_v3_prompt_bank_loads_and_prompts_are_exact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path, prompt_bank_path, image_dir = _create_inputs(root)

            samples, provenance = evaluator.load_evaluation_samples(
                metadata_path,
                prompt_bank_path,
                image_dir,
            )

            self.assertEqual(len(samples), 1)
            self.assertEqual(provenance.generator, "test-generator")
            self.assertEqual(
                provenance.protocol_version,
                "rich-prompt-phase1-v3",
            )
            self.assertEqual(samples[0].class_prompt, "bottles")
            self.assertEqual(
                samples[0].detailed_prompt,
                "  Dark bottles stand behind grapes.  ",
            )
            self.assertEqual(
                samples[0].generalized_prompt,
                "  Dark object stand behind grapes.  ",
            )
            self.assertEqual(samples[0].gt_count, 3.0)

    def test_protocol_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path, prompt_bank_path, image_dir = _create_inputs(
                root,
                protocol="rich-prompt-phase1-v2",
            )

            with self.assertRaisesRegex(
                evaluator.RichPromptEvaluationError,
                "Incompatible prompt-bank protocol",
            ):
                evaluator.load_evaluation_samples(
                    metadata_path,
                    prompt_bank_path,
                    image_dir,
                )

    def test_metadata_prompt_bank_class_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path, prompt_bank_path, image_dir = _create_inputs(root)
            bank = json.loads(prompt_bank_path.read_text(encoding="utf-8"))
            bank["prompts"]["214.jpg"]["class"] = "cups"
            _write_json(prompt_bank_path, bank)

            with self.assertRaisesRegex(
                evaluator.RichPromptEvaluationError,
                "Class mismatch for 214.jpg",
            ):
                evaluator.load_evaluation_samples(
                    metadata_path,
                    prompt_bank_path,
                    image_dir,
                )

    def test_image_is_prepared_once_and_same_patches_reused_three_times(self):
        sample = _sample()
        model = mock.Mock()
        inputs = object()
        prepared_patches = object()
        image_loader = mock.Mock(return_value=inputs)
        patch_preparer = mock.Mock(return_value=prepared_patches)
        mask_builder = mock.Mock(side_effect=lambda _, prompt: "mask:" + prompt)
        predictions_by_prompt = {
            sample.class_prompt: 9.0,
            sample.detailed_prompt: 10.0,
            sample.generalized_prompt: 11.0,
        }
        prediction_calls = []

        def predictor(model_arg, inputs_arg, prompt, mask, **kwargs):
            prediction_calls.append(
                (model_arg, inputs_arg, prompt, mask, kwargs)
            )
            return predictions_by_prompt[prompt]

        persist = mock.Mock()
        records = evaluator.evaluate_samples(
            [sample],
            model,
            tokenizer=object(),
            device="cpu",
            batch_size=4,
            persist=persist,
            emit=lambda _: None,
            image_loader=image_loader,
            patch_preparer=patch_preparer,
            mask_builder=mask_builder,
            predictor=predictor,
        )

        model.eval.assert_called_once_with()
        image_loader.assert_called_once_with(sample.image_path, "cpu")
        patch_preparer.assert_called_once_with(
            inputs,
            evaluator.PATCH_SIZE,
            evaluator.PATCH_STRIDE,
        )
        self.assertEqual(len(prediction_calls), 3)
        self.assertEqual(
            [call[2] for call in prediction_calls],
            [
                sample.class_prompt,
                sample.detailed_prompt,
                sample.generalized_prompt,
            ],
        )
        for _, inputs_arg, _, _, kwargs in prediction_calls:
            self.assertIs(inputs_arg, inputs)
            self.assertIs(kwargs["prepared_patches"], prepared_patches)
        self.assertEqual(records[0]["pred_class"], 9.0)
        self.assertEqual(records[0]["pred_detailed"], 10.0)
        self.assertEqual(records[0]["pred_generalized"], 11.0)
        persist.assert_called_once()

    def test_non_finite_prediction_fails_with_image_and_mode(self):
        sample = _sample()
        predictions = iter((9.0, float("nan")))

        with self.assertRaisesRegex(
            evaluator.RichPromptEvaluationError,
            "sample.jpg mode=detailed",
        ):
            evaluator.evaluate_samples(
                [sample],
                mock.Mock(),
                tokenizer=object(),
                device="cpu",
                batch_size=1,
                emit=lambda _: None,
                image_loader=lambda *_: object(),
                patch_preparer=lambda *_: object(),
                mask_builder=lambda *_: object(),
                predictor=lambda *args, **kwargs: next(predictions),
            )

    def test_mae_and_rmse_are_computed_correctly(self):
        records = [
            evaluator.make_prediction_record(
                _sample("a.jpg", gt_count=1.0),
                {"class": 2.0, "detailed": 1.0, "generalized": 1.0},
            ),
            evaluator.make_prediction_record(
                _sample("b.jpg", gt_count=3.0),
                {"class": 5.0, "detailed": 3.0, "generalized": 3.0},
            ),
        ]

        metrics = evaluator.compute_mode_metrics(records, "class")

        self.assertEqual(metrics["mae"], 1.5)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(2.5))
        self.assertEqual(metrics["mean_signed_error"], 1.5)

    def test_paired_deltas_and_outcome_counts_are_correct(self):
        predictions = (
            (12.0, 11.0),
            (12.0, 13.0),
            (11.0, 11.0),
        )
        records = [
            evaluator.make_prediction_record(
                _sample("{}.jpg".format(index), gt_count=10.0),
                {
                    "class": class_prediction,
                    "detailed": detailed_prediction,
                    "generalized": 10.0,
                },
            )
            for index, (class_prediction, detailed_prediction) in enumerate(
                predictions
            )
        ]

        paired = evaluator.compute_paired_diagnostics(records, "detailed")

        self.assertEqual(paired["mean_delta_abs_error"], 0.0)
        self.assertEqual(paired["median_delta_abs_error"], 0.0)
        self.assertEqual(paired["improved"], 1)
        self.assertEqual(paired["worsened"], 1)
        self.assertEqual(paired["tied"], 1)

    def test_anomaly_sensitivity_removes_only_two_ids(self):
        records = [
            {"image": "{}.jpg".format(index)}
            for index in range(228)
        ] + [
            {"image": "3312.jpg"},
            {"image": "3313.jpg"},
        ]

        subsets = evaluator.build_analysis_subsets(records)

        self.assertEqual(len(subsets["all_samples"]), 230)
        self.assertEqual(
            {record["image"] for record in subsets["all_samples"]}
            & evaluator.KNOWN_LABEL_ANOMALIES,
            evaluator.KNOWN_LABEL_ANOMALIES,
        )
        excluded = subsets["excluding_known_label_anomalies"]
        self.assertEqual(len(excluded), 228)
        self.assertFalse(
            {record["image"] for record in excluded}
            & evaluator.KNOWN_LABEL_ANOMALIES
        )

    def test_csv_and_summary_schemas(self):
        record = evaluator.make_prediction_record(
            _sample(),
            {"class": 9.0, "detailed": 10.0, "generalized": 11.0},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / evaluator.PREDICTIONS_FILENAME
            summary_path = root / evaluator.SUMMARY_FILENAME

            evaluator.write_predictions_csv([record], csv_path)
            summary = evaluator.build_summary(
                [record],
                {"generator": "test-generator"},
            )
            evaluator._atomic_json_write(summary, summary_path)

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(tuple(reader.fieldnames), evaluator.CSV_FIELDS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["class_prompt"], "bottles")
            self.assertIn("all_samples", saved_summary["subsets"])
            self.assertIn(
                "excluding_known_label_anomalies",
                saved_summary["subsets"],
            )
            self.assertIn(
                "rmse",
                saved_summary["subsets"]["all_samples"]["metrics"]["class"],
            )

    def test_resume_reuses_only_compatible_complete_records(self):
        sample = _sample()
        record = evaluator.make_prediction_record(
            sample,
            {"class": 9.0, "detailed": 10.0, "generalized": 11.0},
        )
        manifest = {"schema_version": evaluator.EVALUATION_SCHEMA_VERSION}
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"
            paths = evaluator.initialize_outputs(
                output_dir,
                manifest,
                [sample],
                resume=False,
                overwrite=False,
            )
            evaluator.write_predictions_csv([record], paths[0])

            resumed = evaluator.initialize_outputs(
                output_dir,
                manifest,
                [sample],
                resume=True,
                overwrite=False,
            )[3]

            self.assertEqual(set(resumed), {sample.image_name})
            self.assertEqual(resumed[sample.image_name]["pred_class"], 9.0)

            with self.assertRaisesRegex(
                evaluator.RichPromptEvaluationError,
                "Resume manifest is incompatible",
            ):
                evaluator.initialize_outputs(
                    output_dir,
                    {"schema_version": "different"},
                    [sample],
                    resume=True,
                    overwrite=False,
                )

    def test_model_loader_uses_eval_mode_and_no_training_module(self):
        source = Path(evaluator.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from train import", source)
        self.assertNotIn("import train", source)
        model = mock.Mock()
        model.clip = SimpleNamespace(tokenizer=object())
        paths = SimpleNamespace(
            config=Path("config.yaml"),
            sd_checkpoint=Path("sd.ckpt"),
            clip_dir=Path("clip"),
            checkpoint=Path("official.pth"),
        )

        with mock.patch(
            "models.build.build_t2icount",
            return_value=model,
        ) as build:
            loaded = evaluator.load_evaluation_model(paths, "cpu")

        self.assertIs(loaded, model)
        build.assert_called_once_with(
            paths.config,
            paths.sd_checkpoint,
            paths.clip_dir,
            checkpoint_path=paths.checkpoint,
            device="cpu",
            mode="eval",
        )
        model.eval.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
