"""Standalone RichCount-inspired alignment components for T2ICount.

This module intentionally has no dependency on the counting model, density-map
training, DUMLO, or inference code.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PROMPT_MODES = ("class", "detailed", "generalized")
DISTANCE_QUANTILES = (
    ("p01", 0.01),
    ("p05", 0.05),
    ("p10", 0.10),
    ("p25", 0.25),
    ("median", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)
NEGATIVE_DISTANCE_THRESHOLDS = (
    ("fraction_below_0_8", 0.8),
    ("fraction_below_0_9", 0.9),
    ("fraction_below_1_0", 1.0),
    ("fraction_below_1_1", 1.1),
    ("fraction_below_1_2", 1.2),
    ("fraction_below_1_25", 1.25),
    ("fraction_below_1_3", 1.3),
    ("fraction_below_1_35", 1.35),
    ("fraction_below_1_4", 1.4),
    ("fraction_below_1_45", 1.45),
    ("fraction_below_1_5", 1.5),
)
EXPECTED_SUBSET_FINGERPRINT = (
    "sha256:dd96b36bf15013e194b1a8ece06452a19822aae028fc00c0de019cbb7a311f24"
)
EXPECTED_PROMPT_BANK_FINGERPRINT = (
    "sha256:c3ec587b37fd9f351cdedbce888b26e0ba7a020de956c4ef18202a1268e7e32f"
)


class RichAlignmentError(ValueError):
    """Raised when alignment configuration or provenance is incompatible."""


@dataclass(frozen=True)
class AlignmentSplit:
    train_images: Sequence[str]
    val_images: Sequence[str]
    split_seed: int
    fingerprint: str


class VisualAlignmentFFN(nn.Module):
    """Minimal full-width visual FFN operating in CLIP joint space."""

    def __init__(self, embedding_dim=768, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, inputs):
        return self.network(inputs)


class TokenTextAdapter(nn.Module):
    """Residual MLP applied independently to each CLIP text token."""

    def __init__(self, embedding_dim=768, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, inputs):
        return inputs + self.network(inputs)


def compact_json_fingerprint(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:{}".format(hashlib.sha256(payload).hexdigest())


def build_alignment_split(
    image_names: Sequence[str],
    split_seed: int,
    train_size: int = 900,
    val_size: int = 100,
) -> AlignmentSplit:
    """Split an ordered source subset with a private Torch generator."""
    names = list(image_names)
    if len(names) != train_size + val_size:
        raise RichAlignmentError(
            "Alignment split requires exactly {} images, got {}".format(
                train_size + val_size, len(names)
            )
        )
    if len(set(names)) != len(names):
        raise RichAlignmentError("Alignment source subset contains duplicate images")

    generator = torch.Generator()
    generator.manual_seed(split_seed)
    permutation = torch.randperm(len(names), generator=generator).tolist()
    train_images = [names[index] for index in permutation[:train_size]]
    val_images = [names[index] for index in permutation[train_size:]]
    fingerprint_payload = {
        "alignment_split_seed": split_seed,
        "train_images": train_images,
        "val_images": val_images,
    }
    split = AlignmentSplit(
        train_images=train_images,
        val_images=val_images,
        split_seed=split_seed,
        fingerprint=compact_json_fingerprint(fingerprint_payload),
    )
    validate_alignment_split(split, names)
    return split


def validate_alignment_split(
    split: AlignmentSplit, source_image_names: Sequence[str]
) -> None:
    train = list(split.train_images)
    val = list(split.val_images)
    source = list(source_image_names)
    if set(train).intersection(val):
        raise RichAlignmentError("Alignment train/validation split overlaps")
    if len(train) + len(val) != len(source):
        raise RichAlignmentError("Alignment split size does not match source subset")
    if set(train).union(val) != set(source):
        raise RichAlignmentError("Alignment split union does not match source subset")
    repeated = compact_json_fingerprint(
        {
            "alignment_split_seed": split.split_seed,
            "train_images": train,
            "val_images": val,
        }
    )
    if repeated != split.fingerprint:
        raise RichAlignmentError("Alignment split fingerprint mismatch")


def build_split_manifest(
    split: AlignmentSplit,
    source_subset_fingerprint: str,
    prompt_bank_fingerprint: str,
) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "alignment_split_seed": split.split_seed,
        "alignment_train_count": len(split.train_images),
        "alignment_validation_count": len(split.val_images),
        "train_images": list(split.train_images),
        "val_images": list(split.val_images),
        "source_subset_fingerprint": source_subset_fingerprint,
        "prompt_bank_fingerprint": prompt_bank_fingerprint,
        "split_fingerprint": split.fingerprint,
    }


def _derived_seed(seed: int, namespace: str) -> int:
    encoded = "{}:{}".format(seed, namespace).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def build_negative_indices(
    class_names: Sequence[str], seed: int, namespace: str = "default"
) -> Sequence[int]:
    """Choose one deterministic different-class negative for every sample."""
    classes = list(class_names)
    if len(classes) < 2:
        raise RichAlignmentError("Negative sampling requires at least two samples")
    generator = torch.Generator()
    generator.manual_seed(_derived_seed(seed, namespace))
    negatives = []
    for anchor, class_name in enumerate(classes):
        eligible = [
            index
            for index, candidate_class in enumerate(classes)
            if index != anchor and candidate_class != class_name
        ]
        if not eligible:
            raise RichAlignmentError(
                "No different-class negative is available for sample {}".format(anchor)
            )
        selected = int(
            torch.randint(len(eligible), (1,), generator=generator).item()
        )
        negatives.append(eligible[selected])
    validate_negative_indices(classes, negatives)
    return negatives


def validate_negative_indices(
    class_names: Sequence[str], negative_indices: Sequence[int]
) -> None:
    classes = list(class_names)
    if len(classes) != len(negative_indices):
        raise RichAlignmentError("Negative index count does not match sample count")
    for anchor, negative in enumerate(negative_indices):
        if negative < 0 or negative >= len(classes):
            raise RichAlignmentError("Negative index is out of bounds")
        if negative == anchor:
            raise RichAlignmentError("Negative sampler selected the same image")
        if classes[negative] == classes[anchor]:
            raise RichAlignmentError("Negative sampler selected the same class")


def prompt_for_mode(record, mode: str) -> str:
    if mode == "class":
        return record.class_name
    if mode == "detailed":
        return record.detailed
    if mode == "generalized":
        return record.generalized
    raise RichAlignmentError("Unknown prompt mode: {}".format(mode))


def prepare_distance_embedding(embeddings, normalize_embeddings=False):
    """Apply the configured convention only to final joint-space embeddings."""
    if normalize_embeddings:
        return F.normalize(embeddings, p=2, dim=-1, eps=1e-12)
    return embeddings


def compute_alignment_distances(
    image_embeddings,
    positive_text,
    negative_text,
    normalize_embeddings=False,
):
    """Return prepared embeddings and paired distances for loss and metrics."""
    if image_embeddings.shape != positive_text.shape:
        raise RichAlignmentError("Positive embedding shape mismatch")
    if image_embeddings.shape != negative_text.shape:
        raise RichAlignmentError("Negative embedding shape mismatch")
    distance_image = prepare_distance_embedding(
        image_embeddings, normalize_embeddings
    )
    distance_positive = prepare_distance_embedding(
        positive_text, normalize_embeddings
    )
    distance_negative = prepare_distance_embedding(
        negative_text, normalize_embeddings
    )
    positive_distance = torch.linalg.norm(
        distance_image - distance_positive, dim=-1
    )
    negative_distance = torch.linalg.norm(
        distance_image - distance_negative, dim=-1
    )
    return (
        distance_image,
        distance_positive,
        distance_negative,
        positive_distance,
        negative_distance,
    )


def _contrastive_loss_components(positive_distance, negative_distance, margin):
    if margin <= 0:
        raise RichAlignmentError("Contrastive margin must be greater than zero")
    if positive_distance.shape != negative_distance.shape:
        raise RichAlignmentError("Positive/negative distance shape mismatch")
    if positive_distance.numel() == 0:
        raise RichAlignmentError("Contrastive distances must not be empty")
    positive_loss = torch.mean(positive_distance.pow(2))
    negative_loss = torch.mean(F.relu(margin - negative_distance).pow(2))
    total_loss = 0.5 * (positive_loss + negative_loss)
    return positive_loss, negative_loss, total_loss


def compute_distance_distribution(distances) -> Dict[str, float]:
    """Summarize one actual distance tensor with Torch quantiles."""
    values = distances.reshape(-1)
    if values.numel() == 0:
        raise RichAlignmentError("Distance distribution must not be empty")
    quantile_values = torch.quantile(
        values,
        values.new_tensor([quantile for _, quantile in DISTANCE_QUANTILES]),
    )
    distribution = {"min": float(values.min().item())}
    distribution.update(
        {
            name: float(value.item())
            for (name, _), value in zip(DISTANCE_QUANTILES, quantile_values)
        }
    )
    distribution.update(
        {
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
        }
    )
    return distribution


def compute_distance_diagnostics(
    positive_distance, negative_distance, margin
) -> Dict[str, object]:
    """Compute distributions, strict thresholds, and shared loss components."""
    positive_values = positive_distance.reshape(-1)
    negative_values = negative_distance.reshape(-1)
    positive_loss, negative_loss, total_loss = _contrastive_loss_components(
        positive_values, negative_values, margin
    )
    diagnostics = {
        "positive_distance_distribution": compute_distance_distribution(
            positive_values
        ),
        "negative_distance_distribution": compute_distance_distribution(
            negative_values
        ),
    }
    diagnostics.update(
        {
            name: float((negative_values < threshold).float().mean().item())
            for name, threshold in NEGATIVE_DISTANCE_THRESHOLDS
        }
    )
    below_margin = negative_values < margin
    diagnostics.update(
        {
            "negative_count_below_margin": int(below_margin.sum().item()),
            "negative_fraction_below_margin": float(
                below_margin.float().mean().item()
            ),
            "positive_loss": float(positive_loss.item()),
            "negative_loss": float(negative_loss.item()),
            "total_contrastive_loss": float(total_loss.item()),
        }
    )
    return diagnostics


def richcount_contrastive_loss(
    image_embeddings,
    positive_text,
    negative_text,
    margin=1.0,
    normalize_embeddings=False,
):
    """Implement the balanced RichCount Euclidean contrastive objective."""
    (
        _,
        _,
        _,
        positive_distance,
        negative_distance,
    ) = compute_alignment_distances(
        image_embeddings,
        positive_text,
        negative_text,
        normalize_embeddings=normalize_embeddings,
    )
    positive_loss, negative_loss, total_loss = _contrastive_loss_components(
        positive_distance, negative_distance, margin
    )
    return {
        "loss": total_loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "positive_distance": positive_distance,
        "negative_distance": negative_distance,
    }


def pool_eot_tokens(token_embeddings, input_ids):
    """Match Transformers 4.19 CLIP pooling after adapting token features."""
    if token_embeddings.ndim != 3 or input_ids.ndim != 2:
        raise RichAlignmentError("EOT pooling expects [B,T,D] tokens and [B,T] ids")
    if token_embeddings.shape[:2] != input_ids.shape:
        raise RichAlignmentError("Token embedding/input id shape mismatch")
    batch_indices = torch.arange(token_embeddings.shape[0], device=input_ids.device)
    eot_indices = input_ids.argmax(dim=-1)
    return token_embeddings[batch_indices, eot_indices]


def freeze_module(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
        parameter.grad = None


def configure_stage_a(clip_model, ffn, adapter=None):
    freeze_module(clip_model)
    ffn.train()
    for parameter in ffn.parameters():
        parameter.requires_grad = True
    if adapter is not None:
        freeze_module(adapter)


def configure_stage_b(clip_model, ffn, adapter):
    freeze_module(clip_model)
    freeze_module(ffn)
    adapter.train()
    for parameter in adapter.parameters():
        parameter.requires_grad = True


def assert_no_parameter_gradients(module, label):
    offenders = [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    ]
    if offenders:
        raise RuntimeError(
            "{} unexpectedly received gradients: {}".format(
                label, ", ".join(offenders[:5])
            )
        )


def assert_parameter_gradients(module, label):
    if not any(parameter.grad is not None for parameter in module.parameters()):
        raise RuntimeError("{} did not receive gradients".format(label))


def _compute_mode_alignment_metrics_and_distances(
    image_embeddings,
    positive_text_embeddings,
    negative_text_embeddings,
    class_names: Sequence[str],
    margin: float,
    retrieval_target: str,
    normalize_embeddings: bool = False,
):
    """Compute paired distances, cosine diagnostics, and validation R@1."""
    if image_embeddings.ndim != 2:
        raise RichAlignmentError("Metric image embeddings must be [N,D]")
    if image_embeddings.shape != positive_text_embeddings.shape:
        raise RichAlignmentError("Metric positive embedding shape mismatch")
    if image_embeddings.shape != negative_text_embeddings.shape:
        raise RichAlignmentError("Metric negative embedding shape mismatch")
    if len(class_names) != image_embeddings.shape[0]:
        raise RichAlignmentError("Metric class count mismatch")
    if retrieval_target not in ("sample", "class"):
        raise RichAlignmentError("Unknown retrieval target: {}".format(retrieval_target))

    (
        distance_image,
        distance_positive,
        _,
        positive_distance,
        negative_distance,
    ) = compute_alignment_distances(
        image_embeddings,
        positive_text_embeddings,
        negative_text_embeddings,
        normalize_embeddings=normalize_embeddings,
    )
    positive_cosine = F.cosine_similarity(
        image_embeddings, positive_text_embeddings, dim=-1
    )
    negative_cosine = F.cosine_similarity(
        image_embeddings, negative_text_embeddings, dim=-1
    )
    retrieval_distances = torch.cdist(distance_image, distance_positive)
    nearest = retrieval_distances.argmin(dim=1).tolist()
    sample_correct = [nearest[index] == index for index in range(len(nearest))]
    class_correct = [
        class_names[nearest[index]] == class_names[index]
        for index in range(len(nearest))
    ]
    primary = class_correct if retrieval_target == "class" else sample_correct
    diagnostics = compute_distance_diagnostics(
        positive_distance, negative_distance, margin
    )
    metrics = {
        "sample_count": image_embeddings.shape[0],
        "contrastive_loss": diagnostics["total_contrastive_loss"],
        "mean_positive_euclidean_distance": float(positive_distance.mean().item()),
        "mean_negative_euclidean_distance": float(negative_distance.mean().item()),
        "separation_gap": float((negative_distance - positive_distance).mean().item()),
        "pairwise_alignment_accuracy": float(
            (positive_distance < negative_distance).float().mean().item()
        ),
        "margin_violation_rate": float(
            (negative_distance < margin).float().mean().item()
        ),
        "positive_cosine_similarity": float(positive_cosine.mean().item()),
        "negative_cosine_similarity": float(negative_cosine.mean().item()),
        "retrieval_r_at_1": sum(primary) / len(primary),
        "retrieval_sample_r_at_1": sum(sample_correct) / len(sample_correct),
        "retrieval_class_aware_r_at_1": sum(class_correct) / len(class_correct),
        "retrieval_target": retrieval_target,
    }
    metrics.update(diagnostics)
    return metrics, positive_distance, negative_distance


def compute_mode_alignment_metrics(
    image_embeddings,
    positive_text_embeddings,
    negative_text_embeddings,
    class_names: Sequence[str],
    margin: float,
    retrieval_target: str,
    normalize_embeddings: bool = False,
) -> Dict[str, object]:
    metrics, _, _ = _compute_mode_alignment_metrics_and_distances(
        image_embeddings,
        positive_text_embeddings,
        negative_text_embeddings,
        class_names,
        margin,
        retrieval_target,
        normalize_embeddings=normalize_embeddings,
    )
    return metrics


def build_stage_metrics(
    image_embeddings,
    positive_by_mode: Mapping[str, torch.Tensor],
    negative_by_mode: Mapping[str, torch.Tensor],
    class_names: Sequence[str],
    margin: float,
    normalize_embeddings: bool = False,
) -> Dict[str, object]:
    per_mode = {}
    positive_distances = []
    negative_distances = []
    for mode in PROMPT_MODES:
        metrics, positive_distance, negative_distance = (
            _compute_mode_alignment_metrics_and_distances(
                image_embeddings,
                positive_by_mode[mode],
                negative_by_mode[mode],
                class_names,
                margin,
                retrieval_target="class" if mode == "class" else "sample",
                normalize_embeddings=normalize_embeddings,
            )
        )
        per_mode[mode] = metrics
        positive_distances.append(positive_distance)
        negative_distances.append(negative_distance)
    mean_fields = (
        "contrastive_loss",
        "mean_positive_euclidean_distance",
        "mean_negative_euclidean_distance",
        "separation_gap",
        "pairwise_alignment_accuracy",
        "margin_violation_rate",
        "positive_cosine_similarity",
        "negative_cosine_similarity",
        "retrieval_r_at_1",
        "retrieval_sample_r_at_1",
        "retrieval_class_aware_r_at_1",
    )
    overall = {
        "sample_count": len(class_names) * len(PROMPT_MODES),
        "retrieval_aggregation": "mean_across_prompt_modes",
    }
    for field in mean_fields:
        overall[field] = sum(per_mode[mode][field] for mode in PROMPT_MODES) / len(
            PROMPT_MODES
        )
    overall.update(
        compute_distance_diagnostics(
            torch.cat(positive_distances, dim=0),
            torch.cat(negative_distances, dim=0),
            margin,
        )
    )
    overall["distance_distribution_aggregation"] = (
        "concatenated_class_detailed_generalized_distance_tensors"
    )
    return {
        "overall": overall,
        "class": per_mode["class"],
        "detailed": per_mode["detailed"],
        "generalized": per_mode["generalized"],
    }


def validate_resume_provenance(current: Mapping[str, object], checkpoint) -> None:
    if not isinstance(checkpoint, Mapping):
        raise RichAlignmentError("Resume checkpoint is not a mapping")
    checkpoint_provenance = checkpoint.get("provenance")
    if not isinstance(checkpoint_provenance, Mapping):
        raise RichAlignmentError("Resume checkpoint has no alignment provenance")
    for key in (
        "source_subset_fingerprint",
        "prompt_bank_fingerprint",
        "split_fingerprint",
        "config_fingerprint",
        "normalize_embeddings",
        "training_distance",
    ):
        if checkpoint_provenance.get(key) != current.get(key):
            raise RichAlignmentError(
                "Resume checkpoint provenance mismatch: {}".format(key)
            )
