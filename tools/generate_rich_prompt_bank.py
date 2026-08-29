#!/usr/bin/env python
"""Generate and persist FSC147 rich descriptions with Google Gen AI.

This is an offline preprocessing tool. It deliberately has no imports from the
T2ICount model, training, or inference stacks.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.train_subset import select_train_subset_indices


FSC147S_BENCHMARK = "FSC-147-S"
FSC147_BENCHMARK = "FSC147"
BENCHMARK = FSC147S_BENCHMARK
DEFAULT_MODEL = "gemma-4-26b-a4b-it"
API_NAME = "interactions"
PROTOCOL_VERSION = "rich-prompt-phase1-v3"
GENERALIZATION_RULE = (
    "case-insensitive exact class replacement: class name -> object"
)
ASSET_ROOT_ENV = "T2ICOUNT_ASSET_ROOT"
API_KEY_ENV = "GEMINI_API_KEY"

GENERATION_PROMPT_TEMPLATE = '''The target category is: "{class_name}".

Describe the visual appearance and scene context of the target category
in this image in one concise sentence.

Focus only on information useful for identifying the target, such as:
- appearance
- color
- shape
- material or state
- spatial arrangement
- location in the scene

Requirements:
- Include the exact category name "{class_name}" verbatim.
- Describe only visible evidence of the target category.
- Do not state that the target category is absent, missing, invisible, or not
  shown.
- Do not count the targets.
- Do not state or imply how many target instances are visible.
- Do not use digits, number words, or quantity words.
- Avoid instance-count cues such as a, an, single, individual, pair,
  pairs, couple, group, crowd, or cluster.
- Describe the target category rather than narrating one particular instance.
- Do not describe unrelated objects unless needed to identify the target's
  location.
- Output exactly one sentence and nothing else.'''

# Auditable count-leakage vocabulary. Multi-word expressions and whole words
# are matched case-insensitively. Additions should be reviewed as protocol
# changes because they alter which generated descriptions are accepted.
COUNT_LEAKAGE_TERMS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "hundred",
    "thousand",
    "many",
    "several",
    "few",
    "numerous",
    "dozen",
    "dozens",
    "multiple",
    "both",
    "double",
    "triple",
    "a lot",
    "lots",
    "a number of",
)

# Deliberately conservative pilot filter for implicit instance-count cues.
# Matching is token/word-boundary-aware, so article tokens such as ``a`` and
# ``an`` are rejected without matching the same letters inside other words.
IMPLICIT_COUNT_LEAKAGE_TERMS = (
    "a",
    "an",
    "single",
    "individual",
    "individuals",
    "pair",
    "pairs",
    "couple",
    "couples",
    "group",
    "groups",
    "crowd",
    "crowds",
    "cluster",
    "clusters",
)

# Auditable target-absence phrases. They are applied only in class-aware
# sentence patterns, so unrelated uses of words such as ``no`` are not rejected.
TARGET_ABSENCE_TERMS = (
    "not visible",
    "not present",
    "not shown",
    "not seen",
    "cannot be seen",
    "can't be seen",
    "cannot be found",
    "invisible",
    "absent",
    "missing from the scene",
)

SUPPORTED_IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

_COUNT_LEAKAGE_PATTERN = re.compile(
    r"(?<!\w)(?:{})(?!\w)".format(
        "|".join(
            re.escape(term)
            for term in sorted(COUNT_LEAKAGE_TERMS, key=len, reverse=True)
        )
    ),
    flags=re.IGNORECASE,
)
_IMPLICIT_COUNT_LEAKAGE_PATTERN = re.compile(
    r"(?<!\w)(?:{})(?!\w)".format(
        "|".join(
            re.escape(term)
            for term in sorted(
                IMPLICIT_COUNT_LEAKAGE_TERMS,
                key=len,
                reverse=True,
            )
        )
    ),
    flags=re.IGNORECASE,
)
_TARGET_ABSENCE_TERM_PATTERN = r"(?<!\w)(?:{})(?!\w)".format(
    "|".join(
        re.escape(term)
        for term in sorted(TARGET_ABSENCE_TERMS, key=len, reverse=True)
    )
)
_INTERNAL_SENTENCE_END_PATTERN = re.compile(
    r"[.!?]+(?:[\"')\]]*)\s+\S"
)


class RichPromptError(Exception):
    """Base exception for clear, expected pipeline failures."""


class ConfigurationError(RichPromptError):
    """Raised when CLI or environment configuration is invalid."""


class MetadataError(RichPromptError):
    """Raised when FSC-147-S metadata does not match the expected shape."""


class PromptBankError(RichPromptError):
    """Raised when an existing prompt bank cannot be resumed safely."""


class DescriptionValidationError(RichPromptError):
    """Raised when a generated description violates the fixed protocol."""


@dataclass(frozen=True)
class Sample:
    image_name: str
    class_name: str


@dataclass(frozen=True)
class GenerationConfig:
    asset_root: Path
    metadata_path: Path
    output_path: Path
    model: str
    max_samples: Optional[int] = None
    max_retries: int = 3
    request_delay: float = 2.0
    overwrite: bool = False
    dry_run: bool = False
    split: str = "fsc147s"
    train_samples: int = 0
    train_subset_seed: int = 3407
    concurrency: int = 1


@dataclass(frozen=True)
class RunSummary:
    selected: int
    generated: int
    skipped: int
    failed: int
    dry_run: bool
    selected_image_fingerprint: Optional[str] = None


@dataclass(frozen=True)
class SampleResult:
    index: int
    sample: Sample
    succeeded: bool
    attempts: int
    detailed: Optional[str] = None
    generalized: Optional[str] = None
    failure_reason: str = "generation did not complete"


class GoogleGenAIClient:
    """Small adapter around the optional official Google Gen AI SDK."""

    def __init__(self, api_key: str):
        try:
            from google import genai
        except ImportError as exc:
            raise ConfigurationError(
                "The optional google-genai SDK is not installed. Install "
                "requirements-rich-prompt.txt in a Python 3.10+ environment."
            ) from exc

        try:
            self._client = genai.Client(api_key=api_key)
        except Exception as exc:
            raise ConfigurationError(
                "Could not initialize the google-genai client."
            ) from exc

    def generate(
        self,
        model: str,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> str:
        interaction = self._client.interactions.create(
            model=model,
            store=False,
            input=[
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                    "mime_type": mime_type,
                },
            ],
        )
        try:
            text = interaction.output_text
        except Exception:
            return ""
        return text if isinstance(text, str) else ""


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_generation_prompt(class_name: str) -> str:
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("class_name must be a non-empty string")
    return GENERATION_PROMPT_TEMPLATE.format(class_name=class_name)


def _class_pattern(class_name: str) -> re.Pattern:
    return re.compile(
        r"(?<!\w){}(?!\w)".format(re.escape(class_name)),
        flags=re.IGNORECASE,
    )


def has_target_absence_claim(detailed: str, class_name: str) -> bool:
    """Return whether text makes an auditable class-specific absence claim."""
    escaped_class = re.escape(class_name)
    class_pattern = r"(?<!\w){}(?!\w)".format(escaped_class)
    class_then_absence = re.compile(
        r"{}\s+(?:(?:is|are|was|were)\s+)?{}".format(
            class_pattern,
            _TARGET_ABSENCE_TERM_PATTERN,
        ),
        flags=re.IGNORECASE,
    )
    no_class_observable = re.compile(
        (
            r"(?<!\w)no\s+{}\s+(?:"
            r"(?:(?:is|are|was|were)\s+)?"
            r"(?:visible|present|shown|seen)"
            r"|(?:can|could)\s+be\s+(?:seen|found)"
            r")(?!\w)"
        ).format(class_pattern),
        flags=re.IGNORECASE,
    )
    there_is_no_class = re.compile(
        r"(?<!\w)there\s+(?:is|are|was|were)\s+no\s+{}".format(
            class_pattern
        ),
        flags=re.IGNORECASE,
    )
    class_is_missing = re.compile(
        (
            r"{}\s+(?:(?:is|are|was|were)\s+)?"
            r"(?<!\w)missing(?!\w)\s*[.!?]?\s*$"
        ).format(class_pattern),
        flags=re.IGNORECASE,
    )
    return any(
        pattern.search(detailed)
        for pattern in (
            class_then_absence,
            no_class_observable,
            there_is_no_class,
            class_is_missing,
        )
    )


def validate_detailed_description(raw_description: str, class_name: str) -> str:
    """Validate Gemini text and return whitespace-cleaned text only."""
    if not isinstance(raw_description, str) or not raw_description.strip():
        raise DescriptionValidationError("empty output")
    if "\n" in raw_description or "\r" in raw_description:
        raise DescriptionValidationError("multiline output")

    detailed = " ".join(raw_description.split())
    if _INTERNAL_SENTENCE_END_PATTERN.search(detailed):
        raise DescriptionValidationError("multiple sentences")
    if any(character.isdigit() for character in detailed):
        raise DescriptionValidationError("digit leakage")

    leakage_match = _COUNT_LEAKAGE_PATTERN.search(detailed)
    if leakage_match:
        raise DescriptionValidationError(
            "quantity leakage: {}".format(leakage_match.group(0).casefold())
        )
    implicit_match = _IMPLICIT_COUNT_LEAKAGE_PATTERN.search(detailed)
    if implicit_match:
        raise DescriptionValidationError(
            "implicit count leakage: {}".format(
                implicit_match.group(0).casefold()
            )
        )
    if not _class_pattern(class_name).search(detailed):
        raise DescriptionValidationError("target class missing")
    if has_target_absence_claim(detailed, class_name):
        raise DescriptionValidationError("target absence claim")
    return detailed


def generalize_description(detailed: str, class_name: str) -> str:
    """Replace exact class occurrences with the literal v3 term ``object``."""
    pattern = _class_pattern(class_name)
    if not pattern.search(detailed):
        raise DescriptionValidationError("target class missing")
    return pattern.sub("object", detailed)


def load_fsc147s_metadata(metadata_path: Path) -> List[Sample]:
    path = Path(metadata_path).expanduser().absolute()
    if not path.is_file():
        raise MetadataError("Missing FSC-147-S metadata file: {}".format(path))
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw_metadata = json.load(handle)
    except (OSError, ValueError) as exc:
        raise MetadataError(
            "Could not parse FSC-147-S metadata: {}".format(path)
        ) from exc

    if not isinstance(raw_metadata, dict):
        raise MetadataError("FSC-147-S metadata must be a JSON object")

    samples = []
    for image_name, record in raw_metadata.items():
        if (
            not isinstance(image_name, str)
            or not image_name
            or Path(image_name).name != image_name
        ):
            raise MetadataError("Invalid FSC-147-S image filename")
        if not isinstance(record, dict):
            raise MetadataError(
                "Metadata entry for {} must be an object".format(image_name)
            )
        class_name = record.get("class")
        if not isinstance(class_name, str) or not class_name.strip():
            raise MetadataError(
                "Metadata entry for {} has no valid class".format(image_name)
            )
        # The source count is intentionally not copied into Sample or output.
        samples.append(Sample(image_name=image_name, class_name=class_name))
    return samples


def _load_json_object(path: Path, label: str) -> Dict[str, object]:
    if not path.is_file():
        raise MetadataError("Missing {} file: {}".format(label, path))
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise MetadataError("Could not parse {}: {}".format(label, path)) from exc
    if not isinstance(payload, dict):
        raise MetadataError("{} must be a JSON object".format(label))
    return payload


def _fsc147_dataset_root(asset_root: Path) -> Path:
    return Path(asset_root) / "datasets" / "FSC147"


def load_fsc147_split_samples(
    asset_root: Path,
    split: str,
    train_samples: int,
    train_subset_seed: int,
) -> List[Sample]:
    """Load official FSC147 metadata and apply the trainer's shared selector."""
    if split != "train":
        raise ConfigurationError(
            "Official FSC147 prompt generation currently supports --split train"
        )

    dataset_root = _fsc147_dataset_root(asset_root)
    metadata_root = dataset_root / "FSC_147"
    split_path = metadata_root / "Train_Test_Val_FSC_147.json"
    classes_path = metadata_root / "ImageClasses_FSC147.txt"
    split_metadata = _load_json_object(split_path, "FSC147 split metadata")

    image_names = split_metadata.get(split)
    if not isinstance(image_names, list) or not all(
        isinstance(image_name, str)
        and image_name
        and Path(image_name).name == image_name
        for image_name in image_names
    ):
        raise MetadataError("FSC147 split '{}' must be a list of filenames".format(split))
    if len(set(image_names)) != len(image_names):
        raise MetadataError("FSC147 split '{}' contains duplicate filenames".format(split))

    if not classes_path.is_file():
        raise MetadataError("Missing FSC147 class metadata file: {}".format(classes_path))
    classes = {}
    try:
        with classes_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                fields = line.split("\t")
                if len(fields) != 2 or not fields[0] or not fields[1].strip():
                    raise MetadataError(
                        "Invalid FSC147 class metadata at line {}".format(line_number)
                    )
                if fields[0] in classes:
                    raise MetadataError(
                        "Duplicate FSC147 class metadata for {}".format(fields[0])
                    )
                classes[fields[0]] = fields[1].strip()
    except OSError as exc:
        raise MetadataError(
            "Could not read FSC147 class metadata: {}".format(classes_path)
        ) from exc

    missing_classes = [name for name in image_names if name not in classes]
    if missing_classes:
        raise MetadataError(
            "FSC147 class metadata is missing {}".format(missing_classes[0])
        )

    indices = select_train_subset_indices(
        len(image_names),
        train_samples,
        train_subset_seed,
    )
    return [
        Sample(image_names[index], classes[image_names[index]])
        for index in indices
    ]


def ordered_image_fingerprint(samples: Sequence[Sample]) -> str:
    """Hash the ordered selected image-name list with an explicit encoding."""
    serialized = json.dumps(
        [sample.image_name for sample in samples],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:{}".format(hashlib.sha256(serialized).hexdigest())


def resolve_asset_root(explicit_root: Optional[Path]) -> Path:
    configured = explicit_root or os.environ.get(ASSET_ROOT_ENV)
    if not configured:
        raise ConfigurationError(
            "T2ICount asset root is not configured. Pass --asset-root or set "
            "{}.".format(ASSET_ROOT_ENV)
        )
    root = Path(configured).expanduser().absolute()
    if not root.is_dir():
        raise ConfigurationError("Missing asset root directory: {}".format(root))
    return root


def resolve_image_path(asset_root: Path, image_name: str) -> Tuple[Path, str]:
    suffix = Path(image_name).suffix.casefold()
    try:
        mime_type = SUPPORTED_IMAGE_MIME_TYPES[suffix]
    except KeyError as exc:
        raise ConfigurationError(
            "Unsupported image extension for {}. Supported: .jpg, .jpeg, .png"
            .format(image_name)
        ) from exc

    image_path = (
        Path(asset_root)
        / "datasets"
        / "FSC147"
        / "images_384_VarV2"
        / image_name
    ).absolute()
    if not image_path.is_file():
        raise ConfigurationError(
            "Missing FSC147 image file: {}".format(image_path)
        )
    return image_path, mime_type


def _expected_metadata(
    timestamp: str,
    model: str,
    subset_provenance: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    metadata = {
        "benchmark": (
            FSC147_BENCHMARK if subset_provenance else FSC147S_BENCHMARK
        ),
        "generator": model,
        "api": API_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "generation_prompt_template": GENERATION_PROMPT_TEMPLATE,
        "generalization_rule": GENERALIZATION_RULE,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    if subset_provenance:
        metadata.update(subset_provenance)
    return metadata


def new_prompt_bank(
    model: str,
    timestamp: Optional[str] = None,
    subset_provenance: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    created_at = timestamp or utc_timestamp()
    return {
        "metadata": _expected_metadata(
            created_at,
            model,
            subset_provenance,
        ),
        "prompts": {},
        "failures": {},
    }


def load_prompt_bank(
    output_path: Path,
    model: str,
    timestamp_factory: Callable[[], str] = utc_timestamp,
    subset_provenance: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    path = Path(output_path).expanduser().absolute()
    if not path.exists():
        return new_prompt_bank(
            model,
            timestamp_factory(),
            subset_provenance,
        )
    if not path.is_file():
        raise PromptBankError("Prompt-bank output is not a file: {}".format(path))
    try:
        with path.open("r", encoding="utf-8") as handle:
            bank = json.load(handle)
    except (OSError, ValueError) as exc:
        raise PromptBankError("Could not parse prompt bank: {}".format(path)) from exc

    if not isinstance(bank, dict):
        raise PromptBankError("Prompt bank must be a JSON object")
    metadata = bank.get("metadata")
    prompts = bank.get("prompts")
    failures = bank.get("failures")
    if not isinstance(metadata, dict):
        raise PromptBankError("Prompt bank has invalid metadata")
    if not isinstance(prompts, dict) or not isinstance(failures, dict):
        raise PromptBankError("Prompt bank must contain prompts and failures objects")

    expected = _expected_metadata(
        metadata.get("created_at", ""),
        model,
        subset_provenance,
    )
    compatibility_keys = [
        "benchmark",
        "generator",
        "api",
        "protocol_version",
        "generation_prompt_template",
        "generalization_rule",
    ]
    if subset_provenance:
        compatibility_keys.extend(
            [
                "split",
                "train_samples",
                "train_subset_seed",
                "selected_sample_count",
                "selected_image_fingerprint",
            ]
        )
    for key in compatibility_keys:
        if metadata.get(key) != expected[key]:
            raise PromptBankError(
                "Existing prompt bank has incompatible {}".format(key)
            )
    if not metadata.get("created_at"):
        raise PromptBankError("Existing prompt bank has no created_at timestamp")
    return bank


def atomic_save_prompt_bank(
    bank: Dict[str, object],
    output_path: Path,
    timestamp_factory: Callable[[], str] = utc_timestamp,
) -> None:
    path = Path(output_path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = bank["metadata"]
    if not isinstance(metadata, dict):
        raise PromptBankError("Prompt bank has invalid metadata")
    metadata["updated_at"] = timestamp_factory()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(bank, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _status_code(exc: Exception) -> Optional[int]:
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def is_transient_api_error(exc: Exception) -> bool:
    status_code = _status_code(exc)
    if status_code in (408, 409, 425, 429):
        return True
    if status_code is not None and 500 <= status_code <= 599:
        return True
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    class_name = type(exc).__name__.casefold()
    return any(
        marker in class_name
        for marker in (
            "timeout",
            "connection",
            "connecterror",
            "networkerror",
            "transporterror",
            "remoteprotocolerror",
            "readerror",
            "writeerror",
            "servererror",
            "serviceunavailable",
            "resourceexhausted",
            "toomanyrequests",
        )
    )


def safe_api_error_reason(exc: Exception) -> str:
    status_code = _status_code(exc)
    kind = "transient API error" if is_transient_api_error(exc) else "API error"
    if status_code is not None:
        return "{} (HTTP {})".format(kind, status_code)
    return "{} ({})".format(kind, type(exc).__name__)


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    candidates = [getattr(exc, "retry_after", None)]
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            candidates.append(headers.get("Retry-After"))
        except Exception:
            pass
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return None


class _RequestThrottle:
    """Space concurrent request starts by a configurable global interval."""

    def __init__(
        self,
        request_delay: float,
        sleep: Callable[[float], None],
        clock: Callable[[], float] = time.monotonic,
    ):
        self._request_delay = request_delay
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._last_started_at = None

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last_started_at is not None:
                remaining = self._last_started_at + self._request_delay - now
                if remaining > 0:
                    self._sleep(remaining)
                    now = self._clock()
            self._last_started_at = now


def _select_samples(
    config: GenerationConfig,
) -> Tuple[List[Sample], Optional[Dict[str, object]]]:
    if config.split == "train":
        samples = load_fsc147_split_samples(
            config.asset_root,
            config.split,
            config.train_samples,
            config.train_subset_seed,
        )
        provenance = {
            "split": config.split,
            "train_samples": config.train_samples,
            "train_subset_seed": config.train_subset_seed,
            "selected_sample_count": len(samples),
            "selected_image_fingerprint": ordered_image_fingerprint(samples),
        }
        return samples, provenance

    samples = load_fsc147s_metadata(config.metadata_path)
    if config.max_samples is not None:
        samples = samples[:config.max_samples]
    return samples, None


def _resolve_samples(config: GenerationConfig, samples: Sequence[Sample]):
    resolved_samples = []
    for sample in samples:
        image_path, mime_type = resolve_image_path(
            config.asset_root,
            sample.image_name,
        )
        generation_prompt = build_generation_prompt(sample.class_name)
        resolved_samples.append(
            (sample, image_path, mime_type, generation_prompt)
        )
    return resolved_samples


def _order_bank_mappings(
    prompts: Dict[str, object],
    failures: Dict[str, object],
    selected_names: Sequence[str],
) -> None:
    selected_name_set = set(selected_names)
    for mapping in (prompts, failures):
        ordered = {
            image_name: mapping[image_name]
            for image_name in selected_names
            if image_name in mapping
        }
        for image_name in sorted(set(mapping) - selected_name_set):
            ordered[image_name] = mapping[image_name]
        mapping.clear()
        mapping.update(ordered)


def _persist_result(
    result: SampleResult,
    bank: Dict[str, object],
    output_path: Path,
    timestamp_factory: Callable[[], str],
    include_image: bool,
    deterministic_names: Optional[Sequence[str]] = None,
) -> None:
    prompts = bank["prompts"]
    failures = bank["failures"]
    if not isinstance(prompts, dict) or not isinstance(failures, dict):
        raise PromptBankError("Prompt bank has invalid prompt structures")

    image_name = result.sample.image_name
    if result.succeeded:
        entry = {
            "class": result.sample.class_name,
            "detailed": result.detailed,
            "generalized": result.generalized,
            "status": "ok",
            "attempts": result.attempts,
        }
        if include_image:
            entry = {"image": image_name, **entry}
        prompts[image_name] = entry
        failures.pop(image_name, None)
    else:
        prompts.pop(image_name, None)
        failures[image_name] = {
            "image": image_name,
            "class": result.sample.class_name,
            "reason": result.failure_reason,
            "attempts": result.attempts,
        }

    if deterministic_names is not None:
        _order_bank_mappings(prompts, failures, deterministic_names)
    atomic_save_prompt_bank(bank, output_path, timestamp_factory)


def _validate_config(config: GenerationConfig) -> None:
    if not isinstance(config.model, str) or not config.model.strip():
        raise ConfigurationError("--model must be a non-empty string")
    if config.max_samples is not None and config.max_samples <= 0:
        raise ConfigurationError("--max-samples must be greater than zero")
    if config.max_retries < 0:
        raise ConfigurationError("--max-retries must be zero or greater")
    if config.request_delay < 0:
        raise ConfigurationError("--request-delay must be zero or greater")
    if config.concurrency < 1:
        raise ConfigurationError("--concurrency must be one or greater")
    if config.split not in ("fsc147s", "train"):
        raise ConfigurationError("--split must be fsc147s or train")
    if config.train_samples < 0:
        raise ConfigurationError("--train-samples must be zero or greater")
    if config.split == "train" and config.max_samples is not None:
        raise ConfigurationError(
            "--max-samples is only supported for the FSC-147-S path"
        )
    if config.split == "fsc147s" and config.train_samples:
        raise ConfigurationError("--train-samples requires --split train")


def _run_sequential_generation(
    config: GenerationConfig,
    resolved_samples,
    bank: Dict[str, object],
    client,
    sleep: Callable[[float], None],
    emit: Callable[[str], None],
    timestamp_factory: Callable[[], str],
) -> RunSummary:
    prompts = bank["prompts"]
    if not isinstance(prompts, dict):
        raise PromptBankError("Prompt bank has invalid prompt structures")

    selected = len(resolved_samples)
    generated = 0
    skipped = 0
    failed = 0
    request_count = 0

    for index, (sample, image_path, mime_type, generation_prompt) in enumerate(
        resolved_samples,
        start=1,
    ):
        prefix = "[{}/{}] {} | class={}".format(
            index, selected, sample.image_name, sample.class_name
        )
        existing = prompts.get(sample.image_name)
        if (
            isinstance(existing, dict)
            and existing.get("status") == "ok"
            and not config.overwrite
        ):
            emit("{} | skipped".format(prefix))
            skipped += 1
            continue

        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            raise ConfigurationError(
                "Could not read FSC147 image file: {}".format(image_path)
            ) from exc

        attempts = 0
        failure_reason = "generation did not complete"
        succeeded = False
        retry_after = None
        for attempt_index in range(1, config.max_retries + 2):
            if request_count:
                delay_multiplier = (
                    2 ** (attempt_index - 2) if attempt_index > 1 else 1
                )
                sleep(max(
                    config.request_delay * delay_multiplier,
                    retry_after or 0.0,
                ))
            retry_after = None
            request_count += 1
            attempts += 1

            try:
                raw_description = client.generate(
                    config.model,
                    image_bytes,
                    mime_type,
                    generation_prompt,
                )
            except Exception as exc:
                failure_reason = safe_api_error_reason(exc)
                can_retry = (
                    is_transient_api_error(exc)
                    and attempt_index <= config.max_retries
                )
                if can_retry:
                    retry_after = _retry_after_seconds(exc)
                    emit(
                        "{} | retry {}: {}".format(
                            prefix, attempt_index, failure_reason
                        )
                    )
                    continue
                break

            try:
                detailed = validate_detailed_description(
                    raw_description,
                    sample.class_name,
                )
                generalized = generalize_description(
                    detailed,
                    sample.class_name,
                )
            except DescriptionValidationError as exc:
                failure_reason = str(exc)
                if attempt_index <= config.max_retries:
                    emit(
                        "{} | retry {}: {}".format(
                            prefix, attempt_index, failure_reason
                        )
                    )
                    continue
                break

            result = SampleResult(
                index=index,
                sample=sample,
                succeeded=True,
                attempts=attempts,
                detailed=detailed,
                generalized=generalized,
            )
            _persist_result(
                result,
                bank,
                config.output_path,
                timestamp_factory,
                include_image=(config.split == "train"),
            )
            emit("{} | generated".format(prefix))
            generated += 1
            succeeded = True
            break

        if succeeded:
            continue

        result = SampleResult(
            index=index,
            sample=sample,
            succeeded=False,
            attempts=attempts,
            failure_reason=failure_reason,
        )
        _persist_result(
            result,
            bank,
            config.output_path,
            timestamp_factory,
            include_image=(config.split == "train"),
        )
        emit("{} | failed: {}".format(prefix, failure_reason))
        failed += 1

    fingerprint = bank["metadata"].get("selected_image_fingerprint")
    return RunSummary(selected, generated, skipped, failed, False, fingerprint)


def _generate_one_concurrently(
    index: int,
    selected: int,
    resolved_sample,
    config: GenerationConfig,
    client,
    throttle: _RequestThrottle,
    sleep: Callable[[float], None],
    emit: Callable[[str], None],
) -> SampleResult:
    sample, image_path, mime_type, generation_prompt = resolved_sample
    prefix = "[{}/{}] {} | class={}".format(
        index, selected, sample.image_name, sample.class_name
    )
    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        return SampleResult(
            index=index,
            sample=sample,
            succeeded=False,
            attempts=0,
            failure_reason="image read error ({})".format(type(exc).__name__),
        )

    attempts = 0
    failure_reason = "generation did not complete"
    for attempt_index in range(1, config.max_retries + 2):
        throttle.wait()
        attempts += 1
        retry_after = None
        try:
            raw_description = client.generate(
                config.model,
                image_bytes,
                mime_type,
                generation_prompt,
            )
        except Exception as exc:
            failure_reason = safe_api_error_reason(exc)
            can_retry = (
                is_transient_api_error(exc)
                and attempt_index <= config.max_retries
            )
            if not can_retry:
                break
            retry_after = _retry_after_seconds(exc)
        else:
            try:
                detailed = validate_detailed_description(
                    raw_description,
                    sample.class_name,
                )
                generalized = generalize_description(
                    detailed,
                    sample.class_name,
                )
            except DescriptionValidationError as exc:
                failure_reason = str(exc)
                if attempt_index > config.max_retries:
                    break
            else:
                return SampleResult(
                    index=index,
                    sample=sample,
                    succeeded=True,
                    attempts=attempts,
                    detailed=detailed,
                    generalized=generalized,
                )

        emit(
            "{} | retry {}: {}".format(
                prefix, attempt_index, failure_reason
            )
        )
        backoff = config.request_delay * (2 ** (attempt_index - 1))
        delay = max(backoff, retry_after or 0.0)
        if delay:
            sleep(delay)

    return SampleResult(
        index=index,
        sample=sample,
        succeeded=False,
        attempts=attempts,
        failure_reason=failure_reason,
    )


def _run_concurrent_generation(
    config: GenerationConfig,
    resolved_samples,
    bank: Dict[str, object],
    client,
    sleep: Callable[[float], None],
    emit: Callable[[str], None],
    timestamp_factory: Callable[[], str],
) -> RunSummary:
    prompts = bank["prompts"]
    if not isinstance(prompts, dict):
        raise PromptBankError("Prompt bank has invalid prompt structures")

    selected = len(resolved_samples)
    generated = 0
    skipped = 0
    failed = 0
    work_items = []
    for index, resolved_sample in enumerate(resolved_samples, start=1):
        sample = resolved_sample[0]
        existing = prompts.get(sample.image_name)
        prefix = "[{}/{}] {} | class={}".format(
            index, selected, sample.image_name, sample.class_name
        )
        if (
            isinstance(existing, dict)
            and existing.get("status") == "ok"
            and not config.overwrite
        ):
            emit("{} | skipped".format(prefix))
            skipped += 1
        else:
            work_items.append((index, resolved_sample))

    selected_names = [item[0].image_name for item in resolved_samples]
    throttle = _RequestThrottle(config.request_delay, sleep)
    work_iterator = iter(work_items)

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        pending = {}

        def submit_next():
            try:
                index, resolved_sample = next(work_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                _generate_one_concurrently,
                index,
                selected,
                resolved_sample,
                config,
                client,
                throttle,
                sleep,
                emit,
            )
            pending[future] = (index, resolved_sample[0])
            return True

        for _ in range(config.concurrency):
            if not submit_next():
                break

        while pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda item: pending[item][0]):
                index, sample = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = SampleResult(
                        index=index,
                        sample=sample,
                        succeeded=False,
                        attempts=0,
                        failure_reason="generation worker error ({})".format(
                            type(exc).__name__
                        ),
                    )

                _persist_result(
                    result,
                    bank,
                    config.output_path,
                    timestamp_factory,
                    include_image=(config.split == "train"),
                    deterministic_names=selected_names,
                )
                prefix = "[{}/{}] {} | class={}".format(
                    result.index,
                    selected,
                    result.sample.image_name,
                    result.sample.class_name,
                )
                if result.succeeded:
                    emit("{} | generated".format(prefix))
                    generated += 1
                else:
                    emit("{} | failed: {}".format(prefix, result.failure_reason))
                    failed += 1
                submit_next()

    fingerprint = bank["metadata"].get("selected_image_fingerprint")
    return RunSummary(selected, generated, skipped, failed, False, fingerprint)


def run_generation(
    config: GenerationConfig,
    client=None,
    sleep: Callable[[float], None] = time.sleep,
    emit: Callable[[str], None] = print,
    timestamp_factory: Callable[[], str] = utc_timestamp,
) -> RunSummary:
    """Run generation with an injected client; dry-run never calls or writes."""
    _validate_config(config)
    samples, subset_provenance = _select_samples(config)
    resolved_samples = _resolve_samples(config, samples)
    selected = len(resolved_samples)

    if config.dry_run and subset_provenance:
        for index, (sample, _, _, _) in enumerate(
            resolved_samples[:5],
            start=1,
        ):
            emit(
                "[{}/{}] {} | class={} | dry-run ready".format(
                    index, selected, sample.image_name, sample.class_name
                )
            )
        emit("selected_sample_count={}".format(selected))
        emit(
            "ordered_selected_image_fingerprint={}".format(
                subset_provenance["selected_image_fingerprint"]
            )
        )
        return RunSummary(
            selected,
            0,
            0,
            0,
            True,
            subset_provenance["selected_image_fingerprint"],
        )

    bank = load_prompt_bank(
        config.output_path,
        config.model,
        timestamp_factory,
        subset_provenance,
    )
    prompts = bank["prompts"]
    failures = bank["failures"]
    if not isinstance(prompts, dict) or not isinstance(failures, dict):
        raise PromptBankError("Prompt bank has invalid prompt structures")

    if config.dry_run:
        for index, (sample, _, _, _) in enumerate(resolved_samples, start=1):
            existing = prompts.get(sample.image_name)
            action = "would skip" if (
                isinstance(existing, dict)
                and existing.get("status") == "ok"
                and not config.overwrite
            ) else "ready"
            emit(
                "[{}/{}] {} | class={} | dry-run {}".format(
                    index,
                    selected,
                    sample.image_name,
                    sample.class_name,
                    action,
                )
            )
        return RunSummary(selected, 0, 0, 0, True)

    if client is None:
        raise ConfigurationError(
            "A Google Gen AI client is required outside dry-run mode"
        )
    if config.concurrency == 1:
        return _run_sequential_generation(
            config,
            resolved_samples,
            bank,
            client,
            sleep,
            emit,
            timestamp_factory,
        )
    return _run_concurrent_generation(
        config,
        resolved_samples,
        bank,
        client,
        sleep,
        emit,
        timestamp_factory,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an offline FSC147 rich-prompt bank through the "
            "Google Gen AI Interactions API."
        )
    )
    parser.add_argument(
        "--asset-root",
        default=None,
        help="External T2ICount asset root; defaults to T2ICOUNT_ASSET_ROOT.",
    )
    parser.add_argument("--metadata", default="FSC-147-S.json")
    parser.add_argument(
        "--split",
        choices=("fsc147s", "train"),
        default="fsc147s",
        help=(
            "Source selection: existing FSC-147-S metadata or the official "
            "FSC147 training split."
        ),
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=0,
        help="Deterministically subset the official train split; 0 uses all.",
    )
    parser.add_argument(
        "--train-subset-seed",
        type=int,
        default=3407,
        help="Seed for the exact shared training-subset selector.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Generator model (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Prompt-bank path; defaults to a model-derived filename under "
            "prompts/."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries after the first request for each sample.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=2.0,
        help="Base delay in seconds between requests; retries back off exponentially.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Maximum API requests in flight; starts are globally spaced by "
            "--request-delay."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.model.strip():
        parser.error("--model must be a non-empty string")
    if args.output is None:
        model_slug = re.sub(
            r"[^a-z0-9]+",
            "_",
            args.model.casefold(),
        ).strip("_")
        if args.split == "train":
            args.output = (
                "prompts/fsc147_train{}_seed{}_{}_v3.json".format(
                    args.train_samples,
                    args.train_subset_seed,
                    model_slug,
                )
            )
        else:
            args.output = "prompts/fsc147s_{}_v3.json".format(model_slug)
    return args


def main(
    argv: Optional[Sequence[str]] = None,
    client_factory: Callable[[str], object] = GoogleGenAIClient,
) -> int:
    args = parse_args(argv)
    try:
        api_key = None
        if not args.dry_run:
            api_key = os.environ.get(API_KEY_ENV)
            if not api_key or not api_key.strip():
                raise ConfigurationError(
                    "GEMINI_API_KEY is required outside --dry-run mode."
                )

        asset_root = resolve_asset_root(args.asset_root)
        config = GenerationConfig(
            asset_root=asset_root,
            metadata_path=Path(args.metadata).expanduser().absolute(),
            output_path=Path(args.output).expanduser().absolute(),
            model=args.model,
            max_samples=args.max_samples,
            max_retries=args.max_retries,
            request_delay=args.request_delay,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            split=args.split,
            train_samples=args.train_samples,
            train_subset_seed=args.train_subset_seed,
            concurrency=args.concurrency,
        )
        _validate_config(config)

        client = None
        if not config.dry_run:
            client = client_factory(api_key)

        summary = run_generation(config, client=client)
    except RichPromptError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    except OSError as exc:
        print(
            "error: filesystem operation failed ({})".format(type(exc).__name__),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print("error: unexpected {}".format(type(exc).__name__), file=sys.stderr)
        return 2

    if summary.dry_run:
        print(
            "Dry-run complete: {} sample(s), zero API calls, no output writes."
            .format(summary.selected)
        )
    else:
        print(
            "Complete: generated={}, skipped={}, failed={}.".format(
                summary.generated,
                summary.skipped,
                summary.failed,
            )
        )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
