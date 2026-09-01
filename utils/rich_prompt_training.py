"""Validation and provenance helpers for opt-in rich-prompt training."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence


FSC147_BENCHMARK = "FSC147"
TRAIN_SPLIT = "train"
RICH_PROMPT_PROTOCOL = "rich-prompt-phase1-v3"


class RichPromptTrainingError(ValueError):
    """Raised before training when a rich-prompt bank is incompatible."""


@dataclass(frozen=True)
class RichPromptRecord:
    image_name: str
    class_name: str
    detailed: str
    generalized: str


@dataclass(frozen=True)
class ValidatedRichPromptBank:
    path: str
    file_fingerprint: str
    selected_image_fingerprint: str
    protocol_version: str
    records: Dict[str, RichPromptRecord]


def ordered_image_fingerprint(image_names: Sequence[str]) -> str:
    """Match the Phase 2A compact-JSON ordered-image fingerprint."""
    serialized = json.dumps(
        list(image_names),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:{}".format(hashlib.sha256(serialized).hexdigest())


def validate_prompt_bank_metadata(
    metadata: Mapping[str, object],
    train_samples: int,
    train_subset_seed: int,
    selected_sample_count: int,
    selected_image_fingerprint: str,
) -> None:
    """Validate every Phase 2B research-defining metadata field."""
    if not isinstance(metadata, Mapping):
        raise RichPromptTrainingError("Rich prompt bank has invalid metadata")

    expected = {
        "benchmark": FSC147_BENCHMARK,
        "split": TRAIN_SPLIT,
        "train_samples": train_samples,
        "train_subset_seed": train_subset_seed,
        "protocol_version": RICH_PROMPT_PROTOCOL,
        "selected_sample_count": selected_sample_count,
        "selected_image_fingerprint": selected_image_fingerprint,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RichPromptTrainingError(
                "Rich prompt bank has incompatible {}: expected {!r}, got {!r}"
                .format(key, value, metadata.get(key))
            )


def _validated_prompt_records(
    prompts: object,
    selected_image_names: Sequence[str],
    class_by_image: Mapping[str, str],
) -> Dict[str, RichPromptRecord]:
    if not isinstance(prompts, Mapping):
        raise RichPromptTrainingError(
            "Rich prompt bank must contain a prompts object"
        )

    records = {}
    for image_name in selected_image_names:
        record = prompts.get(image_name)
        if not isinstance(record, Mapping):
            raise RichPromptTrainingError(
                "Rich prompt bank is missing selected prompt: {}".format(
                    image_name
                )
            )
        if record.get("status") != "ok":
            raise RichPromptTrainingError(
                "Rich prompt bank selected prompt is not status=ok: {}"
                .format(image_name)
            )
        if record.get("image") != image_name:
            raise RichPromptTrainingError(
                "Rich prompt bank image field mismatch for {}".format(
                    image_name
                )
            )

        expected_class = class_by_image.get(image_name)
        if not isinstance(expected_class, str) or not expected_class:
            raise RichPromptTrainingError(
                "FSC147 class metadata is missing selected image: {}".format(
                    image_name
                )
            )
        stored_class = record.get("class")
        if stored_class != expected_class:
            raise RichPromptTrainingError(
                "Rich prompt bank class mismatch for {}: expected {!r}, got {!r}"
                .format(image_name, expected_class, stored_class)
            )

        detailed = record.get("detailed")
        generalized = record.get("generalized")
        for label, prompt in (
            ("detailed", detailed),
            ("generalized", generalized),
        ):
            if not isinstance(prompt, str) or not prompt.strip():
                raise RichPromptTrainingError(
                    "Rich prompt bank has missing {} prompt for {}".format(
                        label, image_name
                    )
                )

        records[image_name] = RichPromptRecord(
            image_name=image_name,
            class_name=stored_class,
            detailed=detailed,
            generalized=generalized,
        )
    return records


def load_rich_prompt_bank(
    path,
    train_samples: int,
    train_subset_seed: int,
    selected_image_names: Sequence[str],
    class_by_image: Mapping[str, str],
) -> ValidatedRichPromptBank:
    """Load one bank once and validate it against the exact train subset."""
    bank_path = Path(path).expanduser().absolute()
    if not bank_path.is_file():
        raise RichPromptTrainingError(
            "Rich prompt bank file not found: {}".format(bank_path)
        )
    try:
        raw_bank = bank_path.read_bytes()
        bank = json.loads(raw_bank.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RichPromptTrainingError(
            "Could not parse rich prompt bank: {}".format(bank_path)
        ) from exc
    if not isinstance(bank, Mapping):
        raise RichPromptTrainingError("Rich prompt bank must be a JSON object")

    selected_image_names = list(selected_image_names)
    selected_fingerprint = ordered_image_fingerprint(selected_image_names)
    validate_prompt_bank_metadata(
        bank.get("metadata"),
        train_samples=train_samples,
        train_subset_seed=train_subset_seed,
        selected_sample_count=len(selected_image_names),
        selected_image_fingerprint=selected_fingerprint,
    )
    records = _validated_prompt_records(
        bank.get("prompts"), selected_image_names, class_by_image
    )
    return ValidatedRichPromptBank(
        path=str(bank_path),
        file_fingerprint="sha256:{}".format(
            hashlib.sha256(raw_bank).hexdigest()
        ),
        selected_image_fingerprint=selected_fingerprint,
        protocol_version=RICH_PROMPT_PROTOCOL,
        records=records,
    )


def build_rich_checkpoint_config(
    bank: ValidatedRichPromptBank,
    consistency_weight: float,
    train_samples: int,
    train_subset_seed: int,
) -> Dict[str, object]:
    """Return the auditable rich configuration stored in .tar checkpoints."""
    return {
        "prompt_bank_path": bank.path,
        "prompt_bank_filename": Path(bank.path).name,
        "prompt_bank_fingerprint": bank.file_fingerprint,
        "selected_image_fingerprint": bank.selected_image_fingerprint,
        "protocol_version": bank.protocol_version,
        "rich_consistency_weight": consistency_weight,
        "train_samples": train_samples,
        "train_subset_seed": train_subset_seed,
    }


def validate_resume_rich_config(current, checkpoint) -> None:
    """Reject baseline/rich or rich/rich resume provenance mismatches."""
    if current is None and checkpoint is None:
        return
    if current is None:
        raise RichPromptTrainingError(
            "Cannot resume a rich-prompt checkpoint without --rich-prompt-bank"
        )
    if checkpoint is None:
        raise RichPromptTrainingError(
            "Cannot resume rich-prompt training from a baseline checkpoint"
        )
    if not isinstance(checkpoint, Mapping):
        raise RichPromptTrainingError(
            "Resume checkpoint has invalid rich prompt configuration"
        )

    compatibility_keys = (
        "prompt_bank_fingerprint",
        "selected_image_fingerprint",
        "protocol_version",
        "rich_consistency_weight",
        "train_samples",
        "train_subset_seed",
    )
    for key in compatibility_keys:
        if checkpoint.get(key) != current.get(key):
            raise RichPromptTrainingError(
                "Resume checkpoint has incompatible rich configuration: {}"
                .format(key)
            )


def load_fsc147_train_metadata(data_dir):
    """Read official train filenames/classes for no-GPU bank validation."""
    dataset_root = Path(data_dir).expanduser().absolute()
    split_path = dataset_root / "FSC_147" / "Train_Test_Val_FSC_147.json"
    class_path = dataset_root / "FSC_147" / "ImageClasses_FSC147.txt"
    try:
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RichPromptTrainingError(
            "Could not read FSC147 split metadata: {}".format(split_path)
        ) from exc
    train_names = split_payload.get(TRAIN_SPLIT)
    if not isinstance(train_names, list) or not all(
        isinstance(name, str) and name for name in train_names
    ):
        raise RichPromptTrainingError("FSC147 train split is invalid")

    classes = {}
    try:
        with class_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.strip().split("\t")
                if len(fields) != 2 or not fields[0] or not fields[1]:
                    raise RichPromptTrainingError(
                        "FSC147 class metadata contains an invalid row"
                    )
                classes[fields[0]] = fields[1]
    except (OSError, UnicodeError) as exc:
        raise RichPromptTrainingError(
            "Could not read FSC147 class metadata: {}".format(class_path)
        ) from exc
    return train_names, classes
