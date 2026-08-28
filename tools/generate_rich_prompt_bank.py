#!/usr/bin/env python
"""Generate and persist FSC-147-S rich descriptions with Google Gen AI.

This is an offline preprocessing tool. It deliberately has no imports from the
T2ICount model, training, or inference stacks.
"""

import argparse
import base64
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


BENCHMARK = "FSC-147-S"
DEFAULT_MODEL = "gemma-4-26b-a4b-it"
API_NAME = "interactions"
PROTOCOL_VERSION = "rich-prompt-phase1-v2"
GENERALIZATION_RULE = (
    "case-insensitive exact class replacement: singular-like class -> "
    "target object; plural-like class -> target objects"
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

# Small deterministic plurality heuristic. It intentionally avoids an NLP
# dependency; known false classifications are documented with the protocol.
IRREGULAR_PLURAL_LIKE_CLASSES = frozenset(("people",))

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


@dataclass(frozen=True)
class RunSummary:
    selected: int
    generated: int
    skipped: int
    failed: int
    dry_run: bool


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
    return detailed


def is_plural_like_class(class_name: str) -> bool:
    """Return the deterministic v2 plurality classification for a class."""
    normalized = class_name.strip().casefold()
    if not normalized:
        raise ValueError("class_name must be a non-empty string")
    return (
        normalized in IRREGULAR_PLURAL_LIKE_CLASSES
        or normalized.endswith("s")
    )


def generalize_description(detailed: str, class_name: str) -> str:
    """Replace exact class occurrences with a plurality-aware target phrase."""
    pattern = _class_pattern(class_name)
    if not pattern.search(detailed):
        raise DescriptionValidationError("target class missing")
    replacement = (
        "target objects" if is_plural_like_class(class_name) else "target object"
    )
    return pattern.sub(replacement, detailed)


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


def _expected_metadata(timestamp: str, model: str) -> Dict[str, str]:
    return {
        "benchmark": BENCHMARK,
        "generator": model,
        "api": API_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "generation_prompt_template": GENERATION_PROMPT_TEMPLATE,
        "generalization_rule": GENERALIZATION_RULE,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def new_prompt_bank(
    model: str,
    timestamp: Optional[str] = None,
) -> Dict[str, object]:
    created_at = timestamp or utc_timestamp()
    return {
        "metadata": _expected_metadata(created_at, model),
        "prompts": {},
        "failures": {},
    }


def load_prompt_bank(
    output_path: Path,
    model: str,
    timestamp_factory: Callable[[], str] = utc_timestamp,
) -> Dict[str, object]:
    path = Path(output_path).expanduser().absolute()
    if not path.exists():
        return new_prompt_bank(model, timestamp_factory())
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

    expected = _expected_metadata(metadata.get("created_at", ""), model)
    for key in (
        "benchmark",
        "generator",
        "api",
        "protocol_version",
        "generation_prompt_template",
        "generalization_rule",
    ):
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


def _validate_config(config: GenerationConfig) -> None:
    if not isinstance(config.model, str) or not config.model.strip():
        raise ConfigurationError("--model must be a non-empty string")
    if config.max_samples is not None and config.max_samples <= 0:
        raise ConfigurationError("--max-samples must be greater than zero")
    if config.max_retries < 0:
        raise ConfigurationError("--max-retries must be zero or greater")
    if config.request_delay < 0:
        raise ConfigurationError("--request-delay must be zero or greater")


def run_generation(
    config: GenerationConfig,
    client=None,
    sleep: Callable[[float], None] = time.sleep,
    emit: Callable[[str], None] = print,
    timestamp_factory: Callable[[], str] = utc_timestamp,
) -> RunSummary:
    """Run generation with an injected client; dry-run never calls or writes."""
    _validate_config(config)
    samples = load_fsc147s_metadata(config.metadata_path)
    if config.max_samples is not None:
        samples = samples[:config.max_samples]

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

    bank = load_prompt_bank(
        config.output_path,
        config.model,
        timestamp_factory,
    )
    prompts = bank["prompts"]
    failures = bank["failures"]
    if not isinstance(prompts, dict) or not isinstance(failures, dict):
        raise PromptBankError("Prompt bank has invalid prompt structures")

    selected = len(resolved_samples)
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

    generated = 0
    skipped = 0
    failed = 0
    request_count = 0

    for index, (sample, image_path, mime_type, generation_prompt) in enumerate(
        resolved_samples,
        start=1,
    ):
        prefix = "[{}/{}] {} | class={}".format(
            index,
            selected,
            sample.image_name,
            sample.class_name,
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
        for attempt_index in range(1, config.max_retries + 2):
            if request_count:
                delay_multiplier = (
                    2 ** (attempt_index - 2) if attempt_index > 1 else 1
                )
                sleep(config.request_delay * delay_multiplier)
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
                    emit(
                        "{} | retry {}: {}".format(
                            prefix,
                            attempt_index,
                            failure_reason,
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
                            prefix,
                            attempt_index,
                            failure_reason,
                        )
                    )
                    continue
                break

            prompts[sample.image_name] = {
                "class": sample.class_name,
                "detailed": detailed,
                "generalized": generalized,
                "status": "ok",
                "attempts": attempts,
            }
            failures.pop(sample.image_name, None)
            atomic_save_prompt_bank(
                bank,
                config.output_path,
                timestamp_factory,
            )
            emit("{} | generated".format(prefix))
            generated += 1
            succeeded = True
            break

        if succeeded:
            continue

        prompts.pop(sample.image_name, None)
        failures[sample.image_name] = {
            "image": sample.image_name,
            "class": sample.class_name,
            "reason": failure_reason,
            "attempts": attempts,
        }
        atomic_save_prompt_bank(
            bank,
            config.output_path,
            timestamp_factory,
        )
        emit("{} | failed: {}".format(prefix, failure_reason))
        failed += 1

    return RunSummary(selected, generated, skipped, failed, False)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an offline FSC-147-S rich-prompt bank through the "
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
        args.output = "prompts/fsc147s_{}_v2.json".format(model_slug)
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
