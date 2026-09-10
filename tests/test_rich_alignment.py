import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from utils.rich_alignment import (
    RichAlignmentError,
    TokenTextAdapter,
    VisualAlignmentFFN,
    build_alignment_split,
    build_negative_indices,
    build_stage_metrics,
    compute_alignment_distances,
    compute_distance_diagnostics,
    compute_distance_distribution,
    compute_mode_alignment_metrics,
    configure_stage_a,
    configure_stage_b,
    pool_eot_tokens,
    prepare_distance_embedding,
    richcount_contrastive_loss,
    validate_resume_provenance,
)
from tools.train_rich_alignment import (
    build_stage0_diagnostics_document,
    capture_parameter_state,
    main as alignment_main,
    parse_args,
    run_init_diagnostics_only,
    run_stage0_diagnostics_only,
    validate_args,
)


class NormalizationFlagTests(unittest.TestCase):
    def test_flag_defaults_to_false(self):
        self.assertFalse(parse_args([]).normalize_embeddings)

    def test_flag_enables_normalization(self):
        self.assertTrue(
            parse_args(["--normalize-embeddings"]).normalize_embeddings
        )

    def test_stage0_diagnostics_only_flag_defaults_off_and_is_opt_in(self):
        self.assertFalse(parse_args([]).stage0_diagnostics_only)
        self.assertTrue(
            parse_args(["--stage0-diagnostics-only"]).stage0_diagnostics_only
        )

    def test_init_diagnostics_only_flag_defaults_off_and_is_opt_in(self):
        self.assertFalse(parse_args([]).init_diagnostics_only)
        self.assertTrue(
            parse_args(["--init-diagnostics-only"]).init_diagnostics_only
        )

    def test_diagnostic_only_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(RichAlignmentError, "mutually exclusive"):
            validate_args(
                parse_args(
                    ["--stage0-diagnostics-only", "--init-diagnostics-only"]
                )
            )


class ContrastiveLossTests(unittest.TestCase):
    def test_formula_matches_balanced_paper_form(self):
        image = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        positive = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        negative = torch.tensor([[0.5, 0.0], [2.0, 0.0]])
        result = richcount_contrastive_loss(image, positive, negative, margin=1.0)

        positive_term = (1.0 ** 2 + 2.0 ** 2) / 2.0
        negative_term = ((1.0 - 0.5) ** 2 + 0.0) / 2.0
        expected = 0.5 * (positive_term + negative_term)
        self.assertAlmostEqual(result["loss"].item(), expected)

    def test_margin_hinge_is_zero_at_and_above_margin(self):
        image = torch.zeros(2, 1)
        positive = torch.zeros(2, 1)
        negative = torch.tensor([[1.0], [1.5]])
        result = richcount_contrastive_loss(image, positive, negative, margin=1.0)
        self.assertEqual(result["negative_loss"].item(), 0.0)

    def test_margin_hinge_penalizes_below_margin(self):
        result = richcount_contrastive_loss(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.tensor([[0.25]]),
            margin=1.0,
        )
        self.assertAlmostEqual(result["negative_loss"].item(), 0.75 ** 2)

    def test_raw_mode_is_unchanged(self):
        image = torch.tensor([[2.0, 0.0]])
        positive = torch.tensor([[4.0, 0.0]])
        negative = torch.tensor([[2.5, 0.0]])
        implicit = richcount_contrastive_loss(
            image, positive, negative, margin=1.0
        )
        explicit = richcount_contrastive_loss(
            image,
            positive,
            negative,
            margin=1.0,
            normalize_embeddings=False,
        )
        self.assertIs(prepare_distance_embedding(image, False), image)
        self.assertTrue(torch.equal(implicit["loss"], explicit["loss"]))
        self.assertAlmostEqual(implicit["positive_distance"].item(), 2.0)
        self.assertAlmostEqual(implicit["negative_distance"].item(), 0.5)

    def test_normalized_embeddings_have_unit_norm(self):
        embeddings = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
        normalized = prepare_distance_embedding(embeddings, True)
        self.assertTrue(
            torch.allclose(
                torch.linalg.norm(normalized, dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )

    def test_normalized_euclidean_distance_is_bounded_by_two(self):
        result = richcount_contrastive_loss(
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            torch.tensor([[-1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0.0, -1.0], [-1.0, 0.0]]),
            margin=1.0,
            normalize_embeddings=True,
        )
        distances = torch.cat(
            [result["positive_distance"], result["negative_distance"]]
        )
        self.assertGreaterEqual(distances.min().item(), 0.0)
        self.assertLessEqual(distances.max().item(), 2.0)

    def test_positive_scaling_does_not_change_normalized_distance(self):
        image = torch.tensor([[1.0, 2.0]])
        text = torch.tensor([[2.0, -1.0]])
        first = richcount_contrastive_loss(
            image,
            text,
            -text,
            normalize_embeddings=True,
        )["positive_distance"]
        scaled = richcount_contrastive_loss(
            image * 7.0,
            text * 3.0,
            -text * 5.0,
            normalize_embeddings=True,
        )["positive_distance"]
        self.assertTrue(torch.allclose(first, scaled, atol=1e-6))

    def test_contrastive_loss_uses_normalized_final_embeddings(self):
        image = torch.tensor([[10.0, 0.0]])
        positive = torch.tensor([[1.0, 0.0]])
        negative = torch.tensor([[0.0, 1.0]])
        raw = richcount_contrastive_loss(image, positive, negative)
        normalized = richcount_contrastive_loss(
            image, positive, negative, normalize_embeddings=True
        )
        self.assertGreater(raw["loss"].item(), 1.0)
        self.assertAlmostEqual(normalized["positive_distance"].item(), 0.0)
        self.assertAlmostEqual(
            normalized["negative_distance"].item(), 2.0 ** 0.5, places=6
        )
        self.assertAlmostEqual(normalized["loss"].item(), 0.0)


class SplitAndNegativeTests(unittest.TestCase):
    def test_alignment_split_is_deterministic_and_rng_independent(self):
        names = ["{}.jpg".format(index) for index in range(10)]
        torch.manual_seed(1)
        first = build_alignment_split(names, 3407, train_size=8, val_size=2)
        torch.manual_seed(999)
        second = build_alignment_split(names, 3407, train_size=8, val_size=2)
        self.assertEqual(first, second)

    def test_alignment_split_is_disjoint_and_complete(self):
        names = ["{}.jpg".format(index) for index in range(10)]
        split = build_alignment_split(names, 3407, train_size=8, val_size=2)
        self.assertTrue(set(split.train_images).isdisjoint(split.val_images))
        self.assertEqual(set(split.train_images).union(split.val_images), set(names))

    def test_negative_sampling_never_uses_same_image_or_class(self):
        classes = ["apple", "apple", "pear", "orange"]
        negatives = build_negative_indices(classes, 3407)
        for anchor, negative in enumerate(negatives):
            self.assertNotEqual(anchor, negative)
            self.assertNotEqual(classes[anchor], classes[negative])

    def test_negative_sampling_is_reproducible_and_rng_independent(self):
        classes = ["apple", "pear", "orange", "banana"]
        torch.manual_seed(12)
        first = build_negative_indices(classes, 3407, "epoch")
        torch.manual_seed(98)
        second = build_negative_indices(classes, 3407, "epoch")
        self.assertEqual(first, second)


class ModuleAndFreezeTests(unittest.TestCase):
    def test_token_adapter_preserves_b_t_768(self):
        adapter = TokenTextAdapter(dropout=0.0)
        inputs = torch.randn(2, 77, 768)
        self.assertEqual(adapter(inputs).shape, inputs.shape)

    def test_visual_ffn_outputs_768(self):
        ffn = VisualAlignmentFFN(dropout=0.0)
        self.assertEqual(ffn(torch.randn(3, 768)).shape, (3, 768))

    def test_stage_a_updates_only_ffn(self):
        torch.manual_seed(3)
        clip = nn.Linear(4, 4)
        ffn = VisualAlignmentFFN(embedding_dim=4, dropout=0.0)
        adapter = TokenTextAdapter(embedding_dim=4, dropout=0.0)
        configure_stage_a(clip, ffn, adapter)
        clip_before = [parameter.detach().clone() for parameter in clip.parameters()]
        ffn_before = [parameter.detach().clone() for parameter in ffn.parameters()]
        optimizer = torch.optim.SGD(ffn.parameters(), lr=0.1)
        with torch.no_grad():
            frozen_features = clip(torch.randn(2, 4))
        loss = ffn(frozen_features).pow(2).mean()
        loss.backward()
        optimizer.step()

        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(clip_before, clip.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(ffn_before, ffn.parameters())
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in clip.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in adapter.parameters()))

    def test_stage_b_updates_only_adapter(self):
        torch.manual_seed(4)
        clip = nn.Linear(4, 4)
        ffn = VisualAlignmentFFN(embedding_dim=4, dropout=0.0)
        adapter = TokenTextAdapter(embedding_dim=4, dropout=0.0)
        projection = nn.Linear(4, 4, bias=False)
        configure_stage_b(clip, ffn, adapter)
        projection.requires_grad_(False)
        clip_before = [parameter.detach().clone() for parameter in clip.parameters()]
        ffn_before = [parameter.detach().clone() for parameter in ffn.parameters()]
        adapter_before = [parameter.detach().clone() for parameter in adapter.parameters()]
        optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
        input_ids = torch.tensor([[1, 2, 9, 0], [1, 9, 0, 0]])
        with torch.no_grad():
            frozen_tokens = clip(torch.randn(2, 4, 4))
        adapted = adapter(frozen_tokens)
        pooled = pool_eot_tokens(adapted, input_ids)
        loss = projection(pooled).pow(2).mean()
        loss.backward()
        optimizer.step()

        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(clip_before, clip.parameters())
            )
        )
        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(ffn_before, ffn.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(adapter_before, adapter.parameters())
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in clip.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in ffn.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in projection.parameters()))


class MetricTests(unittest.TestCase):
    def test_distance_distribution_uses_torch_quantiles_on_toy_values(self):
        distances = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        distribution = compute_distance_distribution(distances)

        self.assertEqual(
            set(distribution),
            {
                "min",
                "p01",
                "p05",
                "p10",
                "p25",
                "median",
                "p75",
                "p90",
                "p95",
                "p99",
                "max",
                "mean",
                "std",
            },
        )
        expected_quantiles = torch.quantile(
            distances,
            torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]),
        )
        for key, expected in zip(
            ("p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99"),
            expected_quantiles,
        ):
            self.assertEqual(distribution[key], expected.item())
        self.assertEqual(distribution["min"], 0.0)
        self.assertEqual(distribution["max"], 4.0)
        self.assertEqual(distribution["mean"], 2.0)
        self.assertAlmostEqual(distribution["std"], 2.0 ** 0.5)

    def test_negative_thresholds_and_margin_use_strict_less_than(self):
        diagnostics = compute_distance_diagnostics(
            torch.zeros(6),
            torch.tensor([0.8, 0.9, 1.0, 1.2, 1.25, 1.5]),
            margin=1.2,
        )

        self.assertEqual(diagnostics["fraction_below_0_8"], 0.0)
        self.assertAlmostEqual(diagnostics["fraction_below_0_9"], 1.0 / 6.0)
        self.assertAlmostEqual(diagnostics["fraction_below_1_0"], 2.0 / 6.0)
        self.assertAlmostEqual(diagnostics["fraction_below_1_2"], 3.0 / 6.0)
        self.assertAlmostEqual(diagnostics["fraction_below_1_25"], 4.0 / 6.0)
        self.assertAlmostEqual(diagnostics["fraction_below_1_5"], 5.0 / 6.0)
        self.assertEqual(diagnostics["negative_count_below_margin"], 3)
        self.assertAlmostEqual(
            diagnostics["negative_fraction_below_margin"], 3.0 / 6.0
        )

    def test_diagnostic_loss_components_match_training_formula(self):
        positive = torch.tensor([1.0, 2.0])
        negative = torch.tensor([0.5, 2.0])
        diagnostics = compute_distance_diagnostics(positive, negative, margin=1.0)

        expected_positive = (1.0 ** 2 + 2.0 ** 2) / 2.0
        expected_negative = ((1.0 - 0.5) ** 2 + 0.0) / 2.0
        expected_total = 0.5 * (expected_positive + expected_negative)
        self.assertAlmostEqual(diagnostics["positive_loss"], expected_positive)
        self.assertAlmostEqual(diagnostics["negative_loss"], expected_negative)
        self.assertAlmostEqual(
            diagnostics["total_contrastive_loss"], expected_total
        )

    def test_alignment_metrics_match_toy_values(self):
        images = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        positives = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
        negatives = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
        metrics = compute_mode_alignment_metrics(
            images,
            positives,
            negatives,
            ["a", "b"],
            margin=1.0,
            retrieval_target="sample",
        )
        self.assertAlmostEqual(metrics["mean_positive_euclidean_distance"], 0.5)
        self.assertAlmostEqual(metrics["mean_negative_euclidean_distance"], 2.0)
        self.assertAlmostEqual(metrics["separation_gap"], 1.5)
        self.assertEqual(metrics["pairwise_alignment_accuracy"], 1.0)
        self.assertEqual(metrics["margin_violation_rate"], 0.0)

    def test_class_retrieval_accepts_duplicate_class_prompt_candidate(self):
        images = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        texts = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        metrics = compute_mode_alignment_metrics(
            images,
            texts,
            texts.flip(0),
            ["apple", "apple", "pear"],
            margin=1.0,
            retrieval_target="class",
        )
        self.assertEqual(metrics["retrieval_class_aware_r_at_1"], 1.0)
        self.assertEqual(metrics["retrieval_r_at_1"], 1.0)
        self.assertLess(metrics["retrieval_sample_r_at_1"], 1.0)

    def test_normalized_margin_violation_can_be_active(self):
        metrics = compute_mode_alignment_metrics(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.1]]),
            ["apple"],
            margin=1.0,
            retrieval_target="sample",
            normalize_embeddings=True,
        )
        self.assertEqual(metrics["margin_violation_rate"], 1.0)
        self.assertGreater(metrics["contrastive_loss"], 0.0)

    def test_retrieval_and_pairwise_metrics_use_normalized_convention(self):
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        positives = torch.tensor([[10.0, 0.0], [0.0, 0.1]])
        negatives = positives.flip(0)
        raw = compute_mode_alignment_metrics(
            images,
            positives,
            negatives,
            ["apple", "pear"],
            margin=1.0,
            retrieval_target="sample",
            normalize_embeddings=False,
        )
        normalized = compute_mode_alignment_metrics(
            images,
            positives,
            negatives,
            ["apple", "pear"],
            margin=1.0,
            retrieval_target="sample",
            normalize_embeddings=True,
        )
        self.assertEqual(raw["retrieval_r_at_1"], 0.5)
        self.assertEqual(raw["pairwise_alignment_accuracy"], 0.5)
        self.assertEqual(normalized["retrieval_r_at_1"], 1.0)
        self.assertEqual(normalized["pairwise_alignment_accuracy"], 1.0)

    def test_raw_and_normalized_metrics_share_distance_diagnostics_path(self):
        images = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        positives = torch.tensor([[4.0, 0.0], [0.0, 1.0]])
        negatives = positives.flip(0)
        with patch(
            "utils.rich_alignment.compute_alignment_distances",
            wraps=compute_alignment_distances,
        ) as shared_distances:
            raw = compute_mode_alignment_metrics(
                images,
                positives,
                negatives,
                ["a", "b"],
                margin=1.0,
                retrieval_target="sample",
                normalize_embeddings=False,
            )
            normalized = compute_mode_alignment_metrics(
                images,
                positives,
                negatives,
                ["a", "b"],
                margin=1.0,
                retrieval_target="sample",
                normalize_embeddings=True,
            )

        self.assertEqual(shared_distances.call_count, 2)
        for metrics in (raw, normalized):
            self.assertEqual(
                metrics["positive_distance_distribution"]["mean"],
                metrics["mean_positive_euclidean_distance"],
            )
            self.assertEqual(
                metrics["negative_distance_distribution"]["mean"],
                metrics["mean_negative_euclidean_distance"],
            )
            self.assertEqual(
                metrics["total_contrastive_loss"], metrics["contrastive_loss"]
            )

    def test_stage_metrics_have_required_layout(self):
        embeddings = torch.eye(3)
        by_mode = {
            "class": embeddings,
            "detailed": embeddings,
            "generalized": embeddings,
        }
        negatives = {
            mode: embeddings.roll(1, 0) for mode in by_mode
        }
        metrics = build_stage_metrics(
            embeddings, by_mode, negatives, ["a", "b", "c"], margin=1.0
        )
        self.assertEqual(
            set(metrics), {"overall", "class", "detailed", "generalized"}
        )

    def test_overall_distribution_concatenates_all_mode_distances(self):
        images = torch.zeros(2, 1)
        positives = {
            "class": torch.tensor([[0.0], [2.0]]),
            "detailed": torch.tensor([[4.0], [6.0]]),
            "generalized": torch.tensor([[8.0], [10.0]]),
        }
        negatives = {mode: value + 20.0 for mode, value in positives.items()}
        metrics = build_stage_metrics(
            images, positives, negatives, ["a", "b"], margin=1.0
        )

        overall = metrics["overall"]
        self.assertEqual(
            overall["distance_distribution_aggregation"],
            "concatenated_class_detailed_generalized_distance_tensors",
        )
        self.assertEqual(overall["positive_distance_distribution"]["mean"], 5.0)
        self.assertEqual(overall["positive_distance_distribution"]["median"], 5.0)
        self.assertEqual(overall["sample_count"], 6)

    def test_stage_metrics_are_deterministic_for_deterministic_toy_input(self):
        images = torch.eye(3)
        positives = {
            "class": images,
            "detailed": images.roll(1, 0),
            "generalized": images.roll(2, 0),
        }
        negatives = {mode: value.flip(0) for mode, value in positives.items()}

        first = build_stage_metrics(
            images, positives, negatives, ["a", "b", "c"], margin=1.2
        )
        second = build_stage_metrics(
            images, positives, negatives, ["a", "b", "c"], margin=1.2
        )
        self.assertEqual(first, second)


class Stage0DiagnosticsOnlyTests(unittest.TestCase):
    @staticmethod
    def _stage_metrics():
        embeddings = torch.eye(3)
        positives = {mode: embeddings for mode in ("class", "detailed", "generalized")}
        negatives = {mode: embeddings.roll(1, 0) for mode in positives}
        return build_stage_metrics(
            embeddings, positives, negatives, ["a", "b", "c"], margin=1.2
        )

    @staticmethod
    def _provenance():
        return {
            "source_subset_fingerprint": "subset",
            "prompt_bank_fingerprint": "bank",
            "split_fingerprint": "split",
            "config_fingerprint": "config",
        }

    def test_stage0_diagnostics_only_evaluates_clip_validation_without_training(self):
        args = SimpleNamespace(
            margin=1.2,
            normalize_embeddings=True,
            output_dir=None,
        )
        split = SimpleNamespace(val_images=("v1.jpg", "v2.jpg", "v3.jpg"))
        with patch(
            "tools.train_rich_alignment.evaluate_stage",
            return_value=self._stage_metrics(),
        ) as evaluate, patch(
            "tools.train_rich_alignment.train_stage"
        ) as train, patch(
            "tools.train_rich_alignment.torch.optim.Adam"
        ) as optimizer:
            document = run_stage0_diagnostics_only(
                object(),
                object(),
                object(),
                object(),
                split,
                object(),
                object(),
                object(),
                args,
                self._provenance(),
            )

        self.assertEqual(evaluate.call_args.args[0], "clip")
        self.assertEqual(evaluate.call_args.args[5], list(split.val_images))
        train.assert_not_called()
        optimizer.assert_not_called()
        self.assertEqual(document["sample_count"], 3)
        self.assertTrue(document["normalize_embeddings"])
        self.assertEqual(document["training_distance"], "l2_normalized_euclidean")

    def test_diagnostics_document_is_deterministic(self):
        args = SimpleNamespace(margin=1.2, normalize_embeddings=False)
        metrics = self._stage_metrics()
        first = build_stage0_diagnostics_document(
            metrics, args, self._provenance(), 3
        )
        second = build_stage0_diagnostics_document(
            metrics, args, self._provenance(), 3
        )
        self.assertEqual(first, second)
        self.assertEqual(first["training_distance"], "raw_euclidean_l2")
        self.assertEqual(
            first["overall_distribution_aggregation"],
            "concatenated_class_detailed_generalized_distance_tensors",
        )


class InitializationDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def _parameters_match(module, expected):
        actual = dict(module.named_parameters())
        return set(actual) == set(expected) and all(
            torch.equal(value, actual[name].detach().cpu())
            for name, value in expected.items()
        )

    def test_init_diagnostics_only_uses_same_unchanged_ffn_without_optimizer(self):
        ffn = VisualAlignmentFFN(embedding_dim=2, dropout=0.0)
        adapter = TokenTextAdapter(embedding_dim=2, dropout=0.0)
        initial_ffn = capture_parameter_state(ffn)
        args = SimpleNamespace(output_dir=None)
        split = SimpleNamespace(val_images=("v1.jpg", "v2.jpg"))
        metrics = {"overall": {"contrastive_loss": 1.0}}

        def evaluate(stage, clip_arg, ffn_arg, adapter_arg, *unused):
            self.assertIs(ffn_arg, ffn)
            self.assertIs(adapter_arg, adapter)
            return metrics

        with patch(
            "tools.train_rich_alignment.evaluate_stage", side_effect=evaluate
        ) as evaluate_mock, patch(
            "tools.train_rich_alignment.train_stage"
        ) as train, patch(
            "tools.train_rich_alignment.torch.optim.Adam"
        ) as optimizer:
            result = run_init_diagnostics_only(
                object(),
                ffn,
                adapter,
                object(),
                split,
                object(),
                object(),
                object(),
                args,
            )

        self.assertEqual(
            [call.args[0] for call in evaluate_mock.call_args_list],
            ["clip", "ffn"],
        )
        self.assertTrue(self._parameters_match(ffn, initial_ffn))
        self.assertEqual(set(result), {"stage0_clip", "stage0_ffn_init"})
        train.assert_not_called()
        optimizer.assert_not_called()

    def test_normal_path_evaluates_init_parameters_before_same_modules_train(self):
        events = []
        ffn = VisualAlignmentFFN(embedding_dim=2, dropout=0.0)
        adapter = TokenTextAdapter(embedding_dim=2, dropout=0.0)
        initial_ffn = capture_parameter_state(ffn)
        initial_adapter = capture_parameter_state(adapter)

        class DummyClip(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    projection_dim=2,
                    text_config=SimpleNamespace(hidden_size=2),
                    vision_config=SimpleNamespace(hidden_size=2),
                )
                self.text_projection = nn.Linear(2, 2, bias=False)
                self.visual_projection = nn.Linear(2, 2, bias=False)

        clip = DummyClip()
        tokenizer = SimpleNamespace(vocab_size=10)
        bank = SimpleNamespace(
            selected_image_fingerprint="subset",
            file_fingerprint="bank",
        )
        split = SimpleNamespace(
            train_images=("train.jpg",),
            val_images=("val.jpg",),
            fingerprint="split",
        )
        config = {"research_config": {"training_distance": "raw_euclidean_l2"}}
        provenance = {
            "source_subset_fingerprint": "subset",
            "prompt_bank_fingerprint": "bank",
            "split_fingerprint": "split",
            "config_fingerprint": "config",
        }
        metrics = {"overall": {"contrastive_loss": 1.0}}
        ffn_evaluations = 0
        adapter_evaluations = 0

        def evaluate(stage, clip_arg, ffn_arg, adapter_arg, *unused):
            nonlocal ffn_evaluations, adapter_evaluations
            self.assertIs(clip_arg, clip)
            self.assertIs(ffn_arg, ffn)
            self.assertIs(adapter_arg, adapter)
            if stage == "clip":
                label = "stage0_clip"
            elif stage == "ffn":
                label = "stage0_ffn_init" if ffn_evaluations == 0 else "stage1_ffn"
                ffn_evaluations += 1
                if label == "stage0_ffn_init":
                    self.assertTrue(self._parameters_match(ffn, initial_ffn))
            else:
                label = (
                    "stage1_adapter_init"
                    if adapter_evaluations == 0
                    else "stage2_adapter"
                )
                adapter_evaluations += 1
                if label == "stage1_adapter_init":
                    self.assertTrue(
                        self._parameters_match(adapter, initial_adapter)
                    )
                    self.assertFalse(self._parameters_match(ffn, initial_ffn))
                    self.assertTrue(
                        all(
                            not parameter.requires_grad
                            for parameter in ffn.parameters()
                        )
                    )
            events.append(label)
            return metrics

        ffn_parameter_ids = {id(parameter) for parameter in ffn.parameters()}
        adapter_parameter_ids = {id(parameter) for parameter in adapter.parameters()}

        class FakeOptimizer:
            def __init__(self, label):
                self.label = label

            def step(self):
                events.append("{}_optimizer_step".format(self.label))

        def make_optimizer(parameters, lr):
            del lr
            parameter_ids = {id(parameter) for parameter in parameters}
            if parameter_ids == ffn_parameter_ids:
                label = "ffn"
            elif parameter_ids == adapter_parameter_ids:
                label = "adapter"
            else:
                self.fail("Optimizer received an unexpected parameter set")
            events.append("{}_optimizer_created".format(label))
            return FakeOptimizer(label)

        def train(stage, *call_args, **unused):
            ffn_arg = call_args[3]
            adapter_arg = call_args[4]
            optimizer = call_args[5]
            self.assertIs(ffn_arg, ffn)
            self.assertIs(adapter_arg, adapter)
            if stage == "ffn":
                self.assertTrue(self._parameters_match(ffn, initial_ffn))
                module = ffn
            else:
                self.assertTrue(self._parameters_match(adapter, initial_adapter))
                module = adapter
            events.append("{}_train".format(stage))
            optimizer.step()
            with torch.no_grad():
                next(module.parameters()).add_(1.0)
            state = {
                name: value.detach().cpu().clone()
                for name, value in module.state_dict().items()
            }
            return 0.5, 1, state

        with TemporaryDirectory() as temporary_directory, patch(
            "tools.train_rich_alignment.resolve_and_validate_data",
            return_value=(
                object(),
                Path("clip"),
                Path("images"),
                bank,
                split,
                {},
                {},
            ),
        ), patch(
            "tools.train_rich_alignment.load_offline_clip",
            return_value=(object(), tokenizer, clip, {}),
        ), patch(
            "tools.train_rich_alignment.VisualAlignmentFFN", return_value=ffn
        ), patch(
            "tools.train_rich_alignment.TokenTextAdapter", return_value=adapter
        ), patch(
            "tools.train_rich_alignment.build_configs",
            return_value=(config, provenance),
        ), patch(
            "tools.train_rich_alignment.evaluate_stage", side_effect=evaluate
        ), patch(
            "tools.train_rich_alignment.train_stage", side_effect=train
        ), patch(
            "tools.train_rich_alignment.torch.optim.Adam",
            side_effect=make_optimizer,
        ), patch(
            "tools.train_rich_alignment.atomic_json_write"
        ) as json_write, patch(
            "tools.train_rich_alignment.atomic_torch_save"
        ):
            return_code = alignment_main(
                ["--device", "cpu", "--output-dir", temporary_directory]
            )

        self.assertEqual(return_code, 0)
        self.assertEqual(
            events,
            [
                "stage0_clip",
                "stage0_ffn_init",
                "ffn_optimizer_created",
                "ffn_train",
                "ffn_optimizer_step",
                "stage1_ffn",
                "stage1_adapter_init",
                "adapter_optimizer_created",
                "adapter_train",
                "adapter_optimizer_step",
                "stage2_adapter",
            ],
        )
        written_names = [call.args[1].name for call in json_write.call_args_list]
        for required_name in (
            "stage0_clip_metrics.json",
            "stage0_ffn_init_metrics.json",
            "stage1_ffn_metrics.json",
            "stage1_adapter_init_metrics.json",
            "stage2_adapter_metrics.json",
            "alignment_summary.json",
        ):
            self.assertIn(required_name, written_names)
        summary_call = next(
            call
            for call in json_write.call_args_list
            if call.args[1].name == "alignment_summary.json"
        )
        self.assertTrue(
            {
                "stage0_clip",
                "stage0_ffn_init",
                "stage1_ffn",
                "stage1_adapter_init",
                "stage2_adapter",
            }.issubset(summary_call.args[0])
        )


class ResumeProvenanceTests(unittest.TestCase):
    def test_resume_rejects_provenance_mismatch(self):
        current = {
            "source_subset_fingerprint": "subset",
            "prompt_bank_fingerprint": "bank",
            "split_fingerprint": "split",
            "config_fingerprint": "config",
        }
        checkpoint = {"provenance": dict(current)}
        checkpoint["provenance"]["split_fingerprint"] = "wrong"
        with self.assertRaisesRegex(RichAlignmentError, "split_fingerprint"):
            validate_resume_provenance(current, checkpoint)

    def test_resume_rejects_normalization_mismatch(self):
        current = {
            "source_subset_fingerprint": "subset",
            "prompt_bank_fingerprint": "bank",
            "split_fingerprint": "split",
            "config_fingerprint": "same-for-focused-test",
            "normalize_embeddings": True,
            "training_distance": "l2_normalized_euclidean",
        }
        checkpoint = {
            "provenance": dict(
                current,
                normalize_embeddings=False,
                training_distance="raw_euclidean_l2",
            )
        }
        with self.assertRaisesRegex(RichAlignmentError, "normalize_embeddings"):
            validate_resume_provenance(current, checkpoint)


if __name__ == "__main__":
    unittest.main()
