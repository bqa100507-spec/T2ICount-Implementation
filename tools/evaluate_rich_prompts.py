#!/usr/bin/env python
"""Evaluate FSC-147-S class and rich prompts with fixed T2ICount inference."""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

import torch

from utils.inference import (
    DENSITY_SCALE,
    build_prompt_attention_mask,
    load_image_tensor,
    predict_count,
    prepare_image_patches,
)
from utils.paths import (
    AssetPathError,
    AssetPaths,
    require_directory,
    require_file,
    resolve_required_directory,
    resolve_required_file,
)


BENCHMARK = "FSC-147-S"
REQUIRED_PROTOCOL = "rich-prompt-phase1-v3"
EVALUATION_SCHEMA_VERSION = "rich-prompt-phase1b-v1"
PATCH_SIZE = 384
PATCH_STRIDE = 384
PREDICTIONS_FILENAME = "rich_prompt_eval_predictions.csv"
SUMMARY_FILENAME = "rich_prompt_eval_summary.json"
MANIFEST_FILENAME = "rich_prompt_eval_manifest.json"
PROMPT_MODES = ("class", "detailed", "generalized")
KNOWN_LABEL_ANOMALIES = frozenset(("3312.jpg", "3313.jpg"))
CSV_FIELDS = (
    "image",
    "class",
    "gt_count",
    "class_prompt",
    "detailed_prompt",
    "generalized_prompt",
    "pred_class",
    "pred_detailed",
    "pred_generalized",
    "abs_err_class",
    "abs_err_detailed",
    "abs_err_generalized",
    "delta_abs_err_detailed_vs_class",
    "delta_abs_err_generalized_vs_class",
    "detailed_better_than_class",
    "generalized_better_than_class",
)


class RichPromptEvaluationError(Exception):
    """Raised when Phase 1B cannot proceed without compromising the protocol."""


@dataclass(frozen=True)
class EvaluationSample:
    image_name: str
    class_name: str
    gt_count: float
    class_prompt: str
    detailed_prompt: str
    generalized_prompt: str
    image_path: Path


@dataclass(frozen=True)
class PromptBankProvenance:
    generator: str
    protocol_version: str


@dataclass(frozen=True)
class EvaluationPaths:
    asset_root: Path
    metadata: Path
    prompt_bank: Path
    checkpoint: Path
    output_dir: Path
    config: Path
    sd_checkpoint: Path
    clip_dir: Path
    image_dir: Path


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _absolute(path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _load_json_object(path: Path, label: str) -> Dict[str, object]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RichPromptEvaluationError(
            "Could not parse {}: {}".format(label, path)
        ) from exc
    if not isinstance(document, dict):
        raise RichPromptEvaluationError("{} must be a JSON object".format(label))
    return document


def load_evaluation_samples(
    metadata_path: Path,
    prompt_bank_path: Path,
    image_dir: Path,
    max_samples: Optional[int] = None,
) -> Tuple[List[EvaluationSample], PromptBankProvenance]:
    """Load authoritative counts/classes and exact v3 rich prompt strings."""
    metadata_file = require_file(metadata_path, "FSC-147-S metadata")
    prompt_bank_file = require_file(prompt_bank_path, "rich prompt bank")
    images = require_directory(image_dir, "FSC147 image")
    metadata = _load_json_object(metadata_file, "FSC-147-S metadata")
    prompt_bank = _load_json_object(prompt_bank_file, "rich prompt bank")

    bank_metadata = prompt_bank.get("metadata")
    bank_prompts = prompt_bank.get("prompts")
    if not isinstance(bank_metadata, dict):
        raise RichPromptEvaluationError("Prompt bank has invalid metadata")
    if not isinstance(bank_prompts, dict):
        raise RichPromptEvaluationError("Prompt bank has no prompts object")

    protocol_version = bank_metadata.get("protocol_version")
    if protocol_version != REQUIRED_PROTOCOL:
        raise RichPromptEvaluationError(
            "Incompatible prompt-bank protocol: expected {}, found {}".format(
                REQUIRED_PROTOCOL,
                protocol_version,
            )
        )
    generator = bank_metadata.get("generator")
    if not isinstance(generator, str) or not generator.strip():
        raise RichPromptEvaluationError("Prompt bank has no generator provenance")

    metadata_items = list(metadata.items())
    if max_samples is not None:
        metadata_items = metadata_items[:max_samples]

    samples = []
    for image_name, metadata_entry in metadata_items:
        if (
            not isinstance(image_name, str)
            or not image_name
            or Path(image_name).name != image_name
        ):
            raise RichPromptEvaluationError(
                "Invalid metadata image filename: {!r}".format(image_name)
            )
        if not isinstance(metadata_entry, dict):
            raise RichPromptEvaluationError(
                "Metadata entry for {} must be an object".format(image_name)
            )
        class_name = metadata_entry.get("class")
        gt_count = metadata_entry.get("count")
        if not isinstance(class_name, str) or not class_name.strip():
            raise RichPromptEvaluationError(
                "Metadata entry for {} has no valid class".format(image_name)
            )
        if (
            isinstance(gt_count, bool)
            or not isinstance(gt_count, (int, float))
            or not math.isfinite(float(gt_count))
        ):
            raise RichPromptEvaluationError(
                "Metadata entry for {} has no finite count".format(image_name)
            )

        prompt_entry = bank_prompts.get(image_name)
        if not isinstance(prompt_entry, dict):
            raise RichPromptEvaluationError(
                "Prompt entry missing for {}".format(image_name)
            )
        prompt_class = prompt_entry.get("class")
        if prompt_class != class_name:
            raise RichPromptEvaluationError(
                "Class mismatch for {}: metadata={!r}, prompt_bank={!r}".format(
                    image_name,
                    class_name,
                    prompt_class,
                )
            )
        if prompt_entry.get("status") != "ok":
            raise RichPromptEvaluationError(
                "Prompt entry for {} is not status=ok".format(image_name)
            )
        detailed = prompt_entry.get("detailed")
        generalized = prompt_entry.get("generalized")
        if not isinstance(detailed, str) or not detailed.strip():
            raise RichPromptEvaluationError(
                "Detailed prompt missing for {}".format(image_name)
            )
        if not isinstance(generalized, str) or not generalized.strip():
            raise RichPromptEvaluationError(
                "Generalized prompt missing for {}".format(image_name)
            )

        image_path = images / image_name
        if not image_path.is_file():
            raise RichPromptEvaluationError(
                "Image missing for {}: {}".format(image_name, image_path)
            )
        samples.append(
            EvaluationSample(
                image_name=image_name,
                class_name=class_name,
                gt_count=float(gt_count),
                class_prompt=class_name,
                detailed_prompt=detailed,
                generalized_prompt=generalized,
                image_path=image_path,
            )
        )

    if not samples:
        raise RichPromptEvaluationError("No FSC-147-S samples were selected")
    return samples, PromptBankProvenance(generator, protocol_version)


def make_prediction_record(
    sample: EvaluationSample,
    predictions: Mapping[str, float],
) -> Dict[str, object]:
    converted = {}
    for mode in PROMPT_MODES:
        try:
            prediction = float(predictions[mode])
        except (KeyError, TypeError, ValueError) as exc:
            raise RichPromptEvaluationError(
                "Missing valid {} prediction for {}".format(
                    mode,
                    sample.image_name,
                )
            ) from exc
        if not math.isfinite(prediction):
            raise RichPromptEvaluationError(
                "Non-finite prediction for {} mode={}".format(
                    sample.image_name,
                    mode,
                )
            )
        converted[mode] = prediction

    errors = {
        mode: abs(converted[mode] - sample.gt_count)
        for mode in PROMPT_MODES
    }
    detailed_delta = errors["detailed"] - errors["class"]
    generalized_delta = errors["generalized"] - errors["class"]
    return {
        "image": sample.image_name,
        "class": sample.class_name,
        "gt_count": sample.gt_count,
        "class_prompt": sample.class_prompt,
        "detailed_prompt": sample.detailed_prompt,
        "generalized_prompt": sample.generalized_prompt,
        "pred_class": converted["class"],
        "pred_detailed": converted["detailed"],
        "pred_generalized": converted["generalized"],
        "abs_err_class": errors["class"],
        "abs_err_detailed": errors["detailed"],
        "abs_err_generalized": errors["generalized"],
        "delta_abs_err_detailed_vs_class": detailed_delta,
        "delta_abs_err_generalized_vs_class": generalized_delta,
        "detailed_better_than_class": detailed_delta < 0.0,
        "generalized_better_than_class": generalized_delta < 0.0,
    }


def _progress_line(
    index: int,
    total: int,
    record: Mapping[str, object],
    resumed: bool = False,
) -> str:
    prefix = "[{}/{}] {}".format(index, total, record["image"])
    if resumed:
        prefix += " | resumed"
    return (
        "{} | gt={:g} | class={:.6f} | detailed={:.6f} | "
        "generalized={:.6f}"
    ).format(
        prefix,
        float(record["gt_count"]),
        float(record["pred_class"]),
        float(record["pred_detailed"]),
        float(record["pred_generalized"]),
    )


def evaluate_samples(
    samples: Sequence[EvaluationSample],
    model,
    tokenizer,
    device,
    batch_size: int,
    existing_records: Optional[Mapping[str, Mapping[str, object]]] = None,
    persist: Optional[Callable[[Sequence[Mapping[str, object]]], None]] = None,
    emit: Callable[[str], None] = print,
    image_loader: Callable = load_image_tensor,
    patch_preparer: Callable = prepare_image_patches,
    mask_builder: Callable = build_prompt_attention_mask,
    predictor: Callable = predict_count,
) -> List[Dict[str, object]]:
    """Evaluate three prompts while reusing one prepared patch tuple per image."""
    if batch_size < 1:
        raise RichPromptEvaluationError("--batch-size must be at least 1")
    model.eval()
    records_by_image = {
        image_name: dict(record)
        for image_name, record in (existing_records or {}).items()
    }
    total = len(samples)

    for index, sample in enumerate(samples, start=1):
        existing = records_by_image.get(sample.image_name)
        if existing is not None:
            emit(_progress_line(index, total, existing, resumed=True))
            continue

        try:
            inputs = image_loader(sample.image_path, device)
            prepared_patches = patch_preparer(
                inputs,
                PATCH_SIZE,
                PATCH_STRIDE,
            )
        except Exception as exc:
            raise RichPromptEvaluationError(
                "Image preparation failed for {} ({})".format(
                    sample.image_name,
                    type(exc).__name__,
                )
            ) from exc

        prompts = {
            "class": sample.class_prompt,
            "detailed": sample.detailed_prompt,
            "generalized": sample.generalized_prompt,
        }
        predictions = {}
        with torch.no_grad():
            for mode in PROMPT_MODES:
                prompt = prompts[mode]
                try:
                    prompt_mask = mask_builder(tokenizer, prompt)
                    prediction = predictor(
                        model,
                        inputs,
                        prompt,
                        prompt_mask,
                        batch_size=batch_size,
                        patch_size=PATCH_SIZE,
                        stride=PATCH_STRIDE,
                        prepared_patches=prepared_patches,
                    )
                except Exception as exc:
                    raise RichPromptEvaluationError(
                        "Prediction failed for {} mode={} ({})".format(
                            sample.image_name,
                            mode,
                            type(exc).__name__,
                        )
                    ) from exc
                try:
                    prediction = float(prediction)
                except (TypeError, ValueError) as exc:
                    raise RichPromptEvaluationError(
                        "Non-numeric prediction for {} mode={}".format(
                            sample.image_name,
                            mode,
                        )
                    ) from exc
                if not math.isfinite(prediction):
                    raise RichPromptEvaluationError(
                        "Non-finite prediction for {} mode={}".format(
                            sample.image_name,
                            mode,
                        )
                    )
                predictions[mode] = prediction

        record = make_prediction_record(sample, predictions)
        records_by_image[sample.image_name] = record
        ordered_records = [
            records_by_image[selected.image_name]
            for selected in samples
            if selected.image_name in records_by_image
        ]
        if persist is not None:
            persist(ordered_records)
        emit(_progress_line(index, total, record))

    return [records_by_image[sample.image_name] for sample in samples]


def compute_mode_metrics(
    records: Sequence[Mapping[str, object]],
    mode: str,
) -> Dict[str, object]:
    if mode not in PROMPT_MODES:
        raise ValueError("Unknown prompt mode: {}".format(mode))
    if not records:
        raise ValueError("Cannot compute metrics for an empty record set")
    errors = []
    signed_errors = []
    for record in records:
        prediction = float(record["pred_{}".format(mode)])
        gt_count = float(record["gt_count"])
        if not math.isfinite(prediction) or not math.isfinite(gt_count):
            raise RichPromptEvaluationError("Metrics received a non-finite value")
        difference = prediction - gt_count
        signed_errors.append(difference)
        errors.append(abs(difference))
    return {
        "sample_count": len(records),
        "mae": sum(errors) / len(errors),
        "rmse": math.sqrt(
            sum(error * error for error in signed_errors) / len(signed_errors)
        ),
        "mean_signed_error": sum(signed_errors) / len(signed_errors),
    }


def compute_paired_diagnostics(
    records: Sequence[Mapping[str, object]],
    comparison_mode: str,
) -> Dict[str, object]:
    if comparison_mode not in ("detailed", "generalized"):
        raise ValueError("Paired comparison must use detailed or generalized")
    if not records:
        raise ValueError("Cannot compute diagnostics for an empty record set")
    deltas = [
        float(record["abs_err_{}".format(comparison_mode)])
        - float(record["abs_err_class"])
        for record in records
    ]
    return {
        "sample_count": len(records),
        "mean_delta_abs_error": sum(deltas) / len(deltas),
        "median_delta_abs_error": statistics.median(deltas),
        "improved": sum(delta < 0.0 for delta in deltas),
        "worsened": sum(delta > 0.0 for delta in deltas),
        "tied": sum(delta == 0.0 for delta in deltas),
    }


def build_analysis_subsets(
    records: Sequence[Mapping[str, object]],
) -> Dict[str, List[Mapping[str, object]]]:
    return {
        "all_samples": list(records),
        "excluding_known_label_anomalies": [
            record
            for record in records
            if record["image"] not in KNOWN_LABEL_ANOMALIES
        ],
    }


def build_summary(
    records: Sequence[Mapping[str, object]],
    run_metadata: Mapping[str, object],
) -> Dict[str, object]:
    subsets = {}
    for subset_name, subset_records in build_analysis_subsets(records).items():
        if not subset_records:
            raise RichPromptEvaluationError(
                "Analysis subset is empty: {}".format(subset_name)
            )
        subsets[subset_name] = {
            "sample_count": len(subset_records),
            "metrics": {
                mode: compute_mode_metrics(subset_records, mode)
                for mode in PROMPT_MODES
            },
            "paired_diagnostics": {
                "detailed_vs_class": compute_paired_diagnostics(
                    subset_records,
                    "detailed",
                ),
                "generalized_vs_class": compute_paired_diagnostics(
                    subset_records,
                    "generalized",
                ),
            },
        }
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "metadata": dict(run_metadata),
        "subsets": subsets,
    }


def _atomic_json_write(document: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_predictions_csv(
    records: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(output_path.name),
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for record in records:
                writer.writerow({field: record[field] for field in CSV_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(output_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _parse_csv_bool(value: str, field: str, image_name: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise RichPromptEvaluationError(
        "Invalid {} value for resumed image {}".format(field, image_name)
    )


def load_resumed_records(
    predictions_path: Path,
    samples: Sequence[EvaluationSample],
) -> Dict[str, Dict[str, object]]:
    sample_by_image = {sample.image_name: sample for sample in samples}
    try:
        with predictions_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise RichPromptEvaluationError(
                    "Resume CSV has an incompatible schema"
                )
            rows = list(reader)
    except OSError as exc:
        raise RichPromptEvaluationError(
            "Could not read resume CSV: {}".format(predictions_path)
        ) from exc

    records = {}
    for row in rows:
        image_name = row.get("image")
        if image_name not in sample_by_image:
            raise RichPromptEvaluationError(
                "Resume CSV contains an unexpected image: {}".format(image_name)
            )
        if image_name in records:
            raise RichPromptEvaluationError(
                "Resume CSV contains duplicate image: {}".format(image_name)
            )
        sample = sample_by_image[image_name]
        expected_strings = {
            "class": sample.class_name,
            "class_prompt": sample.class_prompt,
            "detailed_prompt": sample.detailed_prompt,
            "generalized_prompt": sample.generalized_prompt,
        }
        for field, expected in expected_strings.items():
            if row.get(field) != expected:
                raise RichPromptEvaluationError(
                    "Resume CSV has incompatible {} for {}".format(
                        field,
                        image_name,
                    )
                )
        try:
            csv_gt = float(row["gt_count"])
            predictions = {
                mode: float(row["pred_{}".format(mode)])
                for mode in PROMPT_MODES
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RichPromptEvaluationError(
                "Resume CSV has incomplete predictions for {}".format(image_name)
            ) from exc
        if csv_gt != sample.gt_count:
            raise RichPromptEvaluationError(
                "Resume CSV has incompatible gt_count for {}".format(image_name)
            )
        record = make_prediction_record(sample, predictions)
        for field in (
            "abs_err_class",
            "abs_err_detailed",
            "abs_err_generalized",
            "delta_abs_err_detailed_vs_class",
            "delta_abs_err_generalized_vs_class",
        ):
            try:
                serialized_value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise RichPromptEvaluationError(
                    "Resume CSV has invalid {} for {}".format(field, image_name)
                ) from exc
            if serialized_value != float(record[field]):
                raise RichPromptEvaluationError(
                    "Resume CSV has inconsistent {} for {}".format(
                        field,
                        image_name,
                    )
                )
        for field in (
            "detailed_better_than_class",
            "generalized_better_than_class",
        ):
            if _parse_csv_bool(row[field], field, image_name) != record[field]:
                raise RichPromptEvaluationError(
                    "Resume CSV has inconsistent {} for {}".format(
                        field,
                        image_name,
                    )
                )
        records[image_name] = record
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path, include_hash: bool) -> Dict[str, object]:
    stat = path.stat()
    fingerprint = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        fingerprint["sha256"] = _sha256_file(path)
    return fingerprint


def build_resume_manifest(
    paths: EvaluationPaths,
    samples: Sequence[EvaluationSample],
    provenance: PromptBankProvenance,
    device: str,
    batch_size: int,
) -> Dict[str, object]:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "checkpoint": _file_fingerprint(paths.checkpoint, include_hash=True),
        "prompt_bank": _file_fingerprint(paths.prompt_bank, include_hash=True),
        "metadata": _file_fingerprint(paths.metadata, include_hash=True),
        "protocol_version": provenance.protocol_version,
        "generator": provenance.generator,
        "sample_images": [sample.image_name for sample in samples],
        "inference": {
            "config": _file_fingerprint(paths.config, include_hash=True),
            "sd_checkpoint": _file_fingerprint(
                paths.sd_checkpoint,
                include_hash=False,
            ),
            "clip_dir": str(paths.clip_dir),
            "device": device,
            "patch_batch_size": batch_size,
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "density_scale": DENSITY_SCALE,
        },
    }


def initialize_outputs(
    output_dir: Path,
    manifest: Mapping[str, object],
    samples: Sequence[EvaluationSample],
    resume: bool,
    overwrite: bool,
) -> Tuple[Path, Path, Path, Dict[str, Dict[str, object]]]:
    if resume and overwrite:
        raise RichPromptEvaluationError(
            "--resume and --overwrite cannot be used together"
        )
    predictions_path = output_dir / PREDICTIONS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    if resume:
        if not manifest_path.is_file() or not predictions_path.is_file():
            raise RichPromptEvaluationError(
                "Resume requires both {} and {}".format(
                    manifest_path,
                    predictions_path,
                )
            )
        existing_manifest = _load_json_object(manifest_path, "resume manifest")
        if existing_manifest != dict(manifest):
            raise RichPromptEvaluationError(
                "Resume manifest is incompatible with the current checkpoint, "
                "prompt bank, protocol, samples, or inference configuration"
            )
        existing_records = load_resumed_records(predictions_path, samples)
        return (
            predictions_path,
            summary_path,
            manifest_path,
            existing_records,
        )

    existing_outputs = [
        path
        for path in (predictions_path, summary_path, manifest_path)
        if path.exists()
    ]
    if existing_outputs and not overwrite:
        raise RichPromptEvaluationError(
            "Evaluation output already exists; use --resume or --overwrite: {}"
            .format(existing_outputs[0])
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(manifest, manifest_path)
    write_predictions_csv([], predictions_path)
    if overwrite and summary_path.exists():
        summary_path.unlink()
    return predictions_path, summary_path, manifest_path, {}


def build_run_metadata(
    paths: EvaluationPaths,
    provenance: PromptBankProvenance,
    sample_count: int,
    device: str,
    batch_size: int,
) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "evaluation": "zero-shot prompt compatibility",
        "checkpoint_path": str(paths.checkpoint),
        "prompt_bank_path": str(paths.prompt_bank),
        "protocol_version": provenance.protocol_version,
        "generator": provenance.generator,
        "sample_count": sample_count,
        "device": device,
        "patch_batch_size": batch_size,
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "known_label_anomalies": sorted(KNOWN_LABEL_ANOMALIES),
    }


def print_summary(summary: Mapping[str, object], emit=print) -> None:
    for subset_name, subset in summary["subsets"].items():
        emit("subset={}".format(subset_name))
        for mode in PROMPT_MODES:
            metrics = subset["metrics"][mode]
            emit(
                "{:<12} MAE={:.6f} RMSE={:.6f}".format(
                    mode,
                    metrics["mae"],
                    metrics["rmse"],
                )
            )
        for comparison in (
            "detailed_vs_class",
            "generalized_vs_class",
        ):
            paired = subset["paired_diagnostics"][comparison]
            emit(
                "{} improved={} worsened={} tied={} "
                "mean_delta_abs_error={:.6f} median_delta_abs_error={:.6f}"
                .format(
                    comparison,
                    paired["improved"],
                    paired["worsened"],
                    paired["tied"],
                    paired["mean_delta_abs_error"],
                    paired["median_delta_abs_error"],
                )
            )
        emit("")


def resolve_paths(args: argparse.Namespace) -> EvaluationPaths:
    assets = AssetPaths.from_sources(args.asset_root, required=True)
    metadata = require_file(args.metadata, "FSC-147-S metadata")
    prompt_bank = require_file(args.prompt_bank, "rich prompt bank")
    checkpoint = require_file(args.checkpoint, "T2ICount checkpoint")
    config = require_file(args.config, "Stable Diffusion config")
    sd_checkpoint = resolve_required_file(
        args.sd_path,
        assets.sd_checkpoint,
        "Stable Diffusion checkpoint",
    )
    clip_dir = resolve_required_directory(
        args.clip_path,
        assets.clip_dir,
        "CLIP model",
    )
    image_dir = require_directory(
        assets.dataset_dir("fsc147") / "images_384_VarV2",
        "FSC147 image",
    )
    output_dir = _absolute(args.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise RichPromptEvaluationError(
            "Output directory path is not a directory: {}".format(output_dir)
        )
    return EvaluationPaths(
        asset_root=assets.root,
        metadata=metadata,
        prompt_bank=prompt_bank,
        checkpoint=checkpoint,
        output_dir=output_dir,
        config=config,
        sd_checkpoint=sd_checkpoint,
        clip_dir=clip_dir,
        image_dir=image_dir,
    )


def load_evaluation_model(paths: EvaluationPaths, device: str):
    from models.build import build_t2icount

    model = build_t2icount(
        paths.config,
        paths.sd_checkpoint,
        paths.clip_dir,
        checkpoint_path=paths.checkpoint,
        device=device,
        mode="eval",
    )
    model.eval()
    return model


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FSC-147-S class, detailed, and generalized prompts "
            "with unchanged T2ICount inference."
        )
    )
    parser.add_argument(
        "--asset-root",
        default=None,
        help="External asset root; defaults to T2ICOUNT_ASSET_ROOT.",
    )
    parser.add_argument("--metadata", default="FSC-147-S.json")
    parser.add_argument("--prompt-bank", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="configs/v1-inference.yaml")
    parser.add_argument("--sd-path", default=None)
    parser.add_argument("--clip-path", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and sample coverage without loading the model.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.batch_size < 1:
            raise RichPromptEvaluationError("--batch-size must be at least 1")
        if args.max_samples is not None and args.max_samples < 1:
            raise RichPromptEvaluationError("--max-samples must be at least 1")
        paths = resolve_paths(args)
        samples, provenance = load_evaluation_samples(
            paths.metadata,
            paths.prompt_bank,
            paths.image_dir,
            max_samples=args.max_samples,
        )

        print("checkpoint path: {}".format(paths.checkpoint))
        print("prompt bank path: {}".format(paths.prompt_bank))
        print("protocol version: {}".format(provenance.protocol_version))
        print("generator: {}".format(provenance.generator))
        print("sample count: {}".format(len(samples)))
        print("device: {}".format(args.device))
        print("patch batch size: {}".format(args.batch_size))

        if args.validate_only:
            print("Validation complete: no model inference or output writes.")
            return 0

        manifest = build_resume_manifest(
            paths,
            samples,
            provenance,
            args.device,
            args.batch_size,
        )
        (
            predictions_path,
            summary_path,
            _,
            existing_records,
        ) = initialize_outputs(
            paths.output_dir,
            manifest,
            samples,
            args.resume,
            args.overwrite,
        )

        model = load_evaluation_model(paths, args.device)
        tokenizer = model.clip.tokenizer
        records = evaluate_samples(
            samples,
            model,
            tokenizer,
            torch.device(args.device),
            args.batch_size,
            existing_records=existing_records,
            persist=lambda completed: write_predictions_csv(
                completed,
                predictions_path,
            ),
        )
        run_metadata = build_run_metadata(
            paths,
            provenance,
            len(samples),
            args.device,
            args.batch_size,
        )
        summary = build_summary(records, run_metadata)
        _atomic_json_write(summary, summary_path)
        print_summary(summary)
        print("predictions CSV: {}".format(predictions_path))
        print("summary JSON: {}".format(summary_path))
        return 0
    except (RichPromptEvaluationError, AssetPathError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    except OSError as exc:
        print(
            "error: filesystem operation failed ({})".format(type(exc).__name__),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            "error: unexpected {}: {}".format(type(exc).__name__, exc),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
