import unittest

import torch
import torch.nn as nn

from utils.rich_alignment import (
    RichAlignmentError,
    TokenTextAdapter,
    VisualAlignmentFFN,
    build_alignment_split,
    build_negative_indices,
    build_stage_metrics,
    compute_mode_alignment_metrics,
    configure_stage_a,
    configure_stage_b,
    pool_eot_tokens,
    richcount_contrastive_loss,
    validate_resume_provenance,
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


if __name__ == "__main__":
    unittest.main()
