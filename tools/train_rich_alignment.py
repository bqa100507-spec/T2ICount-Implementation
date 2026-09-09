#!/usr/bin/env python
"""Train RichCount-inspired full-image alignment modules for T2ICount.

This standalone entry point never constructs or trains the T2ICount counting
model. It uses frozen local CLIP ViT-L/14 features and the fixed FSC147 train
subset/prompt bank defined by the rich-prompt experiment.
"""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

from utils.checkpoints import load_trusted_legacy_checkpoint
from utils.helper import SaveHandler
from utils.paths import AssetPaths, require_directory
from utils.rich_alignment import (
    EXPECTED_PROMPT_BANK_FINGERPRINT,
    EXPECTED_SUBSET_FINGERPRINT,
    PROMPT_MODES,
    RichAlignmentError,
    TokenTextAdapter,
    VisualAlignmentFFN,
    assert_no_parameter_gradients,
    assert_parameter_gradients,
    build_alignment_split,
    build_negative_indices,
    build_split_manifest,
    build_stage_metrics,
    compact_json_fingerprint,
    configure_stage_a,
    configure_stage_b,
    pool_eot_tokens,
    prompt_for_mode,
    richcount_contrastive_loss,
    validate_resume_provenance,
)
from utils.rich_prompt_training import (
    RichPromptTrainingError,
    load_fsc147_train_metadata,
    load_rich_prompt_bank,
)
from utils.train_subset import select_train_subset_indices


DEFAULT_PROMPT_BANK = (
    REPOSITORY_ROOT / "prompts" / "fsc147_train1000_seed3407_gemma4_v3.json"
)


class RichAlignmentDataset(Dataset):
    """Ordered names only; image processing happens in the collator."""

    def __init__(self, image_root, image_names, class_by_image):
        self.image_root = Path(image_root)
        self.image_names = list(image_names)
        self.class_by_image = class_by_image

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        return {
            "index": index,
            "image_name": image_name,
            "class_name": self.class_by_image[image_name],
            "image_path": str(self.image_root / image_name),
        }


class RichAlignmentCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, samples):
        images = []
        try:
            for sample in samples:
                with Image.open(sample["image_path"]) as image:
                    images.append(image.convert("RGB"))
            pixel_values = self.processor(
                images=images, return_tensors="pt"
            )["pixel_values"]
        finally:
            for image in images:
                image.close()
        return {
            "indices": torch.tensor(
                [sample["index"] for sample in samples], dtype=torch.long
            ),
            "image_names": [sample["image_name"] for sample in samples],
            "class_names": [sample["class_name"] for sample in samples],
            "pixel_values": pixel_values,
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "RichCount-inspired full-image visual-text alignment for T2ICount; "
            "no counting model is trained."
        )
    )
    parser.add_argument("--asset-root")
    parser.add_argument("--prompt-bank", default=str(DEFAULT_PROMPT_BANK))
    parser.add_argument("--output-dir")
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--subset-seed", type=int, default=3407)
    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--ffn-epochs", type=int, default=10)
    parser.add_argument("--adapter-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-ffn", type=float, default=1e-4)
    parser.add_argument("--lr-adapter", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help=(
            "L2-normalize final projected image/text embeddings immediately "
            "before every Euclidean distance computation."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--resume")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--stage0-diagnostics-only",
        action="store_true",
        help=(
            "Evaluate frozen-CLIP Stage 0 distance distributions on the fixed "
            "100-image validation split, then exit without training."
        ),
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Diagnostic-only cap per epoch; 0 uses every alignment train batch.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.train_samples != 1000:
        raise RichAlignmentError("This experiment requires --train-samples 1000")
    if args.subset_seed != 3407:
        raise RichAlignmentError("This experiment requires --subset-seed 3407")
    if args.split_seed != 3407:
        raise RichAlignmentError("This experiment requires --split-seed 3407")
    if args.ffn_epochs < 1 or args.adapter_epochs < 1:
        raise RichAlignmentError("Both stage epoch counts must be at least one")
    if args.batch_size < 1:
        raise RichAlignmentError("--batch-size must be at least one")
    if args.lr_ffn <= 0 or args.lr_adapter <= 0:
        raise RichAlignmentError("Learning rates must be greater than zero")
    if args.margin <= 0:
        raise RichAlignmentError("--margin must be greater than zero")
    if args.dropout < 0 or args.dropout >= 1:
        raise RichAlignmentError("--dropout must be in [0,1)")
    if args.num_workers < 0:
        raise RichAlignmentError("--num-workers must be zero or greater")
    if args.max_train_batches < 0:
        raise RichAlignmentError("--max-train-batches must be zero or greater")
    if args.checkpoint_interval != 1:
        raise RichAlignmentError(
            "Alignment checkpoints are required every epoch; "
            "--checkpoint-interval must be 1"
        )
    if args.validate_only and args.stage0_diagnostics_only:
        raise RichAlignmentError(
            "--validate-only and --stage0-diagnostics-only are mutually exclusive"
        )
    if (
        not args.validate_only
        and not args.stage0_diagnostics_only
        and not args.output_dir
    ):
        raise RichAlignmentError("--output-dir is required for training")


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_json_write(document, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp".format(path.name))
    try:
        torch.save(payload, str(temporary))
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def cpu_state_dict(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def resolve_and_validate_data(args):
    assets = AssetPaths.from_sources(args.asset_root)
    clip_path = require_directory(assets.clip_dir, "CLIP model")
    dataset_root = require_directory(assets.dataset_dir("fsc147"), "FSC147 dataset")
    image_root = require_directory(
        dataset_root / "images_384_VarV2", "FSC147 stored 384 images"
    )
    train_names, class_by_image = load_fsc147_train_metadata(dataset_root)
    indices = select_train_subset_indices(
        len(train_names), args.train_samples, args.subset_seed
    )
    selected_names = [train_names[index] for index in indices]
    bank = load_rich_prompt_bank(
        args.prompt_bank,
        train_samples=args.train_samples,
        train_subset_seed=args.subset_seed,
        selected_image_names=selected_names,
        class_by_image=class_by_image,
    )
    if bank.selected_image_fingerprint != EXPECTED_SUBSET_FINGERPRINT:
        raise RichAlignmentError(
            "Unexpected source subset fingerprint: {}".format(
                bank.selected_image_fingerprint
            )
        )
    if bank.file_fingerprint != EXPECTED_PROMPT_BANK_FINGERPRINT:
        raise RichAlignmentError(
            "Unexpected prompt-bank fingerprint: {}".format(bank.file_fingerprint)
        )
    missing_images = [
        name for name in selected_names if not (image_root / name).is_file()
    ]
    if missing_images:
        raise RichAlignmentError(
            "Missing selected FSC147 image: {}".format(missing_images[0])
        )
    split = build_alignment_split(selected_names, args.split_seed)
    repeated = build_alignment_split(selected_names, args.split_seed)
    if split != repeated:
        raise RichAlignmentError("Repeated alignment split construction changed")
    manifest = build_split_manifest(
        split, bank.selected_image_fingerprint, bank.file_fingerprint
    )
    for namespace, names in (
        ("alignment_train_validation", split.train_images),
        ("alignment_validation", split.val_images),
    ):
        classes = [bank.records[name].class_name for name in names]
        build_negative_indices(classes, args.seed, namespace=namespace)
    return assets, clip_path, image_root, bank, split, manifest, class_by_image


def load_offline_clip(clip_path, device):
    processor = CLIPProcessor.from_pretrained(
        str(clip_path), local_files_only=True
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        str(clip_path), local_files_only=True
    )
    model, loading_info = CLIPModel.from_pretrained(
        str(clip_path), local_files_only=True, output_loading_info=True
    )
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(field):
            raise RichAlignmentError(
                "Offline CLIP load reported {}: {}".format(
                    field, loading_info[field]
                )
            )
    text_hidden = model.config.text_config.hidden_size
    vision_hidden = model.config.vision_config.hidden_size
    projection_dim = model.config.projection_dim
    if text_hidden != 768 or projection_dim != 768:
        raise RichAlignmentError(
            "T2ICount-compatible adapter requires text/projection dimensions "
            "768/768, got {}/{}".format(text_hidden, projection_dim)
        )
    if tuple(model.text_projection.weight.shape) != (projection_dim, text_hidden):
        raise RichAlignmentError("Unexpected CLIP text projection shape")
    if tuple(model.visual_projection.weight.shape) != (
        projection_dim,
        vision_hidden,
    ):
        raise RichAlignmentError("Unexpected CLIP visual projection shape")
    model.eval().requires_grad_(False)
    model.to(device)
    return processor, tokenizer, model, loading_info


def make_loader(
    image_root,
    image_names,
    class_by_image,
    processor,
    args,
    shuffle,
    namespace,
):
    generator = torch.Generator()
    generator.manual_seed(
        int(compact_json_fingerprint([args.seed, namespace]).split(":", 1)[1][:16], 16)
    )
    return DataLoader(
        RichAlignmentDataset(image_root, image_names, class_by_image),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=RichAlignmentCollator(processor),
        generator=generator,
        pin_memory=str(args.device).startswith("cuda"),
    )


def tokenize(processor, texts, device):
    encoded = processor(
        text=list(texts),
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if key in ("input_ids", "attention_mask")
    }


def encode_image(clip_model, pixel_values):
    outputs = clip_model.vision_model(pixel_values=pixel_values)
    return clip_model.visual_projection(outputs.pooler_output)


def encode_original_text(clip_model, encoded):
    outputs = clip_model.text_model(**encoded)
    return clip_model.text_projection(outputs.pooler_output)


def encode_adapted_text(clip_model, adapter, encoded):
    with torch.no_grad():
        outputs = clip_model.text_model(**encoded)
    adapted_tokens = adapter(outputs.last_hidden_state)
    pooled = pool_eot_tokens(adapted_tokens, encoded["input_ids"])
    return clip_model.text_projection(pooled)


def build_batch_prompts(batch_indices, records, negative_indices):
    positives = {mode: [] for mode in PROMPT_MODES}
    negatives = {mode: [] for mode in PROMPT_MODES}
    for index_tensor in batch_indices:
        index = int(index_tensor.item())
        positive = records[index]
        negative = records[negative_indices[index]]
        for mode in PROMPT_MODES:
            positives[mode].append(prompt_for_mode(positive, mode))
            negatives[mode].append(prompt_for_mode(negative, mode))
    return positives, negatives


def stage_forward(
    stage,
    clip_model,
    ffn,
    adapter,
    pixel_values,
    positives,
    negatives,
    processor,
    device,
    margin,
    normalize_embeddings=False,
):
    if stage == "ffn":
        with torch.no_grad():
            base_image = encode_image(clip_model, pixel_values)
        image_embeddings = ffn(base_image)
    elif stage == "adapter":
        with torch.no_grad():
            image_embeddings = ffn(encode_image(clip_model, pixel_values))
    else:
        raise RichAlignmentError("Unknown training stage: {}".format(stage))

    losses = {}
    for mode in PROMPT_MODES:
        positive_inputs = tokenize(processor, positives[mode], device)
        negative_inputs = tokenize(processor, negatives[mode], device)
        if stage == "ffn":
            with torch.no_grad():
                positive_text = encode_original_text(clip_model, positive_inputs)
                negative_text = encode_original_text(clip_model, negative_inputs)
        else:
            positive_text = encode_adapted_text(
                clip_model, adapter, positive_inputs
            )
            negative_text = encode_adapted_text(
                clip_model, adapter, negative_inputs
            )
        losses[mode] = richcount_contrastive_loss(
            image_embeddings,
            positive_text,
            negative_text,
            margin,
            normalize_embeddings=normalize_embeddings,
        )["loss"]
    losses["overall"] = torch.stack(
        [losses[mode] for mode in PROMPT_MODES]
    ).mean()
    return losses


def train_epoch(
    stage,
    epoch_index,
    clip_model,
    ffn,
    adapter,
    optimizer,
    image_root,
    names,
    bank,
    class_by_image,
    processor,
    args,
):
    if stage == "ffn":
        configure_stage_a(clip_model, ffn, adapter)
        trainable = ffn
    else:
        configure_stage_b(clip_model, ffn, adapter)
        trainable = adapter
    records = [bank.records[name] for name in names]
    classes = [record.class_name for record in records]
    negative_indices = build_negative_indices(
        classes,
        args.seed,
        namespace="{}_epoch_{}".format(stage, epoch_index + 1),
    )
    loader = make_loader(
        image_root,
        names,
        class_by_image,
        processor,
        args,
        shuffle=True,
        namespace="{}_loader_epoch_{}".format(stage, epoch_index + 1),
    )
    totals = {mode: 0.0 for mode in PROMPT_MODES}
    totals["overall"] = 0.0
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        if args.max_train_batches and batch_index >= args.max_train_batches:
            break
        optimizer.zero_grad()
        pixel_values = batch["pixel_values"].to(args.device, non_blocking=True)
        positives, negatives = build_batch_prompts(
            batch["indices"], records, negative_indices
        )
        losses = stage_forward(
            stage,
            clip_model,
            ffn,
            adapter,
            pixel_values,
            positives,
            negatives,
            processor,
            args.device,
            args.margin,
            args.normalize_embeddings,
        )
        losses["overall"].backward()
        assert_no_parameter_gradients(clip_model, "CLIP")
        if stage == "adapter":
            assert_no_parameter_gradients(ffn, "visual FFN")
        assert_parameter_gradients(trainable, stage)
        optimizer.step()
        count = pixel_values.shape[0]
        sample_count += count
        for mode in tuple(PROMPT_MODES) + ("overall",):
            totals[mode] += float(losses[mode].detach().item()) * count
    if sample_count == 0:
        raise RichAlignmentError("Training epoch processed no samples")
    return {
        "sample_count": sample_count,
        "loss": {key: value / sample_count for key, value in totals.items()},
    }


def evaluate_stage(
    stage,
    clip_model,
    ffn,
    adapter,
    image_root,
    names,
    bank,
    class_by_image,
    processor,
    args,
):
    clip_model.eval()
    ffn.eval()
    adapter.eval()
    records = [bank.records[name] for name in names]
    classes = [record.class_name for record in records]
    negative_indices = build_negative_indices(
        classes, args.seed, namespace="alignment_validation"
    )
    loader = make_loader(
        image_root,
        names,
        class_by_image,
        processor,
        args,
        shuffle=False,
        namespace="alignment_validation_loader",
    )
    image_parts = []
    positive_parts = {mode: [] for mode in PROMPT_MODES}
    negative_parts = {mode: [] for mode in PROMPT_MODES}
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixel_values"].to(args.device, non_blocking=True)
            base_image = encode_image(clip_model, pixels)
            image_embeddings = base_image if stage == "clip" else ffn(base_image)
            positives, negatives = build_batch_prompts(
                batch["indices"], records, negative_indices
            )
            image_parts.append(image_embeddings.cpu())
            for mode in PROMPT_MODES:
                positive_inputs = tokenize(processor, positives[mode], args.device)
                negative_inputs = tokenize(processor, negatives[mode], args.device)
                if stage == "adapter":
                    positive_text = encode_adapted_text(
                        clip_model, adapter, positive_inputs
                    )
                    negative_text = encode_adapted_text(
                        clip_model, adapter, negative_inputs
                    )
                else:
                    positive_text = encode_original_text(
                        clip_model, positive_inputs
                    )
                    negative_text = encode_original_text(
                        clip_model, negative_inputs
                    )
                positive_parts[mode].append(positive_text.cpu())
                negative_parts[mode].append(negative_text.cpu())
    images = torch.cat(image_parts, dim=0)
    positives = {
        mode: torch.cat(positive_parts[mode], dim=0) for mode in PROMPT_MODES
    }
    negatives = {
        mode: torch.cat(negative_parts[mode], dim=0) for mode in PROMPT_MODES
    }
    return build_stage_metrics(
        images,
        positives,
        negatives,
        classes,
        args.margin,
        normalize_embeddings=args.normalize_embeddings,
    )


def run_forward_smoke(
    clip_model, processor, ffn, adapter, image_root, split, bank, args
):
    names = list(split.train_images)
    records = [bank.records[name] for name in names]
    classes = [record.class_name for record in records]
    negatives = build_negative_indices(classes, args.seed, "validate_only_smoke")
    with Image.open(image_root / names[0]) as image:
        pixel_values = processor(
            images=image.convert("RGB"), return_tensors="pt"
        )["pixel_values"].to(args.device)
    batch_indices = torch.tensor([0])
    positive_texts, negative_texts = build_batch_prompts(
        batch_indices, records, negatives
    )

    configure_stage_a(clip_model, ffn, adapter)
    stage_a = stage_forward(
        "ffn",
        clip_model,
        ffn,
        adapter,
        pixel_values,
        positive_texts,
        negative_texts,
        processor,
        args.device,
        args.margin,
        args.normalize_embeddings,
    )
    stage_a["overall"].backward()
    assert_no_parameter_gradients(clip_model, "CLIP")
    assert_parameter_gradients(ffn, "visual FFN")
    ffn.zero_grad()

    configure_stage_b(clip_model, ffn, adapter)
    stage_b = stage_forward(
        "adapter",
        clip_model,
        ffn,
        adapter,
        pixel_values,
        positive_texts,
        negative_texts,
        processor,
        args.device,
        args.margin,
        args.normalize_embeddings,
    )
    stage_b["overall"].backward()
    assert_no_parameter_gradients(clip_model, "CLIP")
    assert_no_parameter_gradients(ffn, "visual FFN")
    assert_parameter_gradients(adapter, "text adapter")
    with torch.no_grad():
        token_inputs = tokenize(processor, positive_texts["class"], args.device)
        tokens = clip_model.text_model(**token_inputs).last_hidden_state
        adapted = adapter(tokens)
    if adapted.shape != tokens.shape or adapted.shape[-1] != 768:
        raise RichAlignmentError("Token adapter smoke shape invariant failed")
    return {
        "image": names[0],
        "pixel_values_shape": list(pixel_values.shape),
        "token_input_shape": list(tokens.shape),
        "token_output_shape": list(adapted.shape),
        "stage_a_loss": float(stage_a["overall"].detach().item()),
        "stage_b_loss": float(stage_b["overall"].detach().item()),
    }


def build_configs(args, clip_path, bank, split):
    training_distance = (
        "l2_normalized_euclidean"
        if args.normalize_embeddings
        else "raw_euclidean_l2"
    )
    research_config = {
        "experiment_name": "RichCount-inspired full-image alignment for T2ICount",
        "train_samples": args.train_samples,
        "subset_seed": args.subset_seed,
        "alignment_train_samples": len(split.train_images),
        "alignment_validation_samples": len(split.val_images),
        "alignment_split_seed": args.split_seed,
        "seed": args.seed,
        "ffn_epochs": args.ffn_epochs,
        "adapter_epochs": args.adapter_epochs,
        "batch_size": args.batch_size,
        "lr_ffn": args.lr_ffn,
        "lr_adapter": args.lr_adapter,
        "margin": args.margin,
        "dropout": args.dropout,
        "max_train_batches": args.max_train_batches,
        "normalize_embeddings": args.normalize_embeddings,
        "training_distance": training_distance,
        "l2_normalization": args.normalize_embeddings,
        "prompt_modes": list(PROMPT_MODES),
        "best_checkpoint_criterion": "minimum_validation_overall_contrastive_loss",
    }
    config_fingerprint = compact_json_fingerprint(research_config)
    provenance = {
        "source_subset_fingerprint": bank.selected_image_fingerprint,
        "prompt_bank_fingerprint": bank.file_fingerprint,
        "split_fingerprint": split.fingerprint,
        "config_fingerprint": config_fingerprint,
        "normalize_embeddings": args.normalize_embeddings,
        "training_distance": training_distance,
    }
    document = {
        "schema_version": 1,
        "research_config": research_config,
        "config_fingerprint": config_fingerprint,
        "provenance": provenance,
        "paths": {
            "clip": str(clip_path),
            "prompt_bank": bank.path,
            "image_source": "<asset-root>/datasets/FSC147/images_384_VarV2/",
        },
        "implementation_choices": {
            "visual_input": "full stored FSC147 384 image",
            "visual_preprocessing": "standard local CLIPProcessor only",
            "visual_ffn": "Linear(768,768)-ReLU-Dropout-Linear(768,768)",
            "text_adapter": (
                "token-wise residual Linear(768,768)-ReLU-Dropout-"
                "Linear(768,768)"
            ),
            "class_prompt": "exact FSC147 class string",
            "validation_negatives": "fixed across stage0/stage1/stage2",
            "negative_label_noise_limitation": (
                "A different FSC147 class is not guaranteed to be visually "
                "absent from the anchor image."
            ),
            "richcount_reproduction": False,
        },
        "runtime": {
            "device": args.device,
            "num_workers": args.num_workers,
            "checkpoint_interval": args.checkpoint_interval,
            "offline_environment": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            },
        },
    }
    return document, provenance


def checkpoint_handler(output_dir, stage):
    handler = SaveHandler(num=2)
    existing = sorted(Path(output_dir).glob("{}_epoch_*.pt".format(stage)))[-2:]
    handler.save_list.extend(str(path) for path in existing)
    return handler


def save_epoch_checkpoint(
    output_dir,
    stage,
    epoch_index,
    trainable,
    optimizer,
    config_document,
    provenance,
    best_metric,
    best_epoch,
    best_state,
    handler,
    ffn=None,
):
    path = Path(output_dir) / "{}_epoch_{:04d}.pt".format(stage, epoch_index + 1)
    payload = {
        "format_version": 1,
        "stage": stage,
        "epoch": epoch_index + 1,
        "next_epoch": epoch_index + 1,
        "trainable_state_dict": cpu_state_dict(trainable),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "normalize_embeddings": config_document["research_config"][
            "normalize_embeddings"
        ],
        "training_distance": config_document["research_config"][
            "training_distance"
        ],
        "config": config_document,
        "provenance": dict(provenance),
        "best_validation_overall_contrastive_loss": best_metric,
        "best_epoch": best_epoch,
        "best_state_dict": best_state,
    }
    if ffn is not None:
        payload["ffn_state_dict"] = cpu_state_dict(ffn)
    atomic_torch_save(payload, path)
    handler.append(str(path))


def train_stage(
    stage,
    start_epoch,
    epochs,
    clip_model,
    ffn,
    adapter,
    optimizer,
    image_root,
    train_names,
    val_names,
    bank,
    class_by_image,
    processor,
    args,
    output_dir,
    config_document,
    provenance,
    best_metric=float("inf"),
    best_epoch=None,
    best_state=None,
):
    trainable = ffn if stage == "ffn" else adapter
    handler = checkpoint_handler(output_dir, stage)
    for epoch_index in range(start_epoch, epochs):
        training_metrics = train_epoch(
            stage,
            epoch_index,
            clip_model,
            ffn,
            adapter,
            optimizer,
            image_root,
            train_names,
            bank,
            class_by_image,
            processor,
            args,
        )
        validation_metrics = evaluate_stage(
            stage,
            clip_model,
            ffn,
            adapter,
            image_root,
            val_names,
            bank,
            class_by_image,
            processor,
            args,
        )
        selection_metric = validation_metrics["overall"]["contrastive_loss"]
        if selection_metric < best_metric:
            best_metric = selection_metric
            best_epoch = epoch_index + 1
            best_state = cpu_state_dict(trainable)
        save_epoch_checkpoint(
            output_dir,
            stage,
            epoch_index,
            trainable,
            optimizer,
            config_document,
            provenance,
            best_metric,
            best_epoch,
            best_state,
            handler,
            ffn=ffn if stage == "adapter" else None,
        )
        print(
            json.dumps(
                {
                    "stage": stage,
                    "epoch": epoch_index + 1,
                    "training": training_metrics,
                    "validation": validation_metrics,
                    "best_epoch": best_epoch,
                    "best_validation_overall_contrastive_loss": best_metric,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if best_state is None:
        raise RichAlignmentError("No best state is available for {}".format(stage))
    trainable.load_state_dict(best_state, strict=True)
    return best_metric, best_epoch, best_state


def build_stage0_diagnostics_document(
    stage0_metrics, args, provenance, validation_sample_count
):
    training_distance = (
        "l2_normalized_euclidean"
        if args.normalize_embeddings
        else "raw_euclidean_l2"
    )
    return {
        "schema_version": 1,
        "diagnostic": "frozen_clip_stage0_distance_distribution",
        "margin": args.margin,
        "normalize_embeddings": args.normalize_embeddings,
        "training_distance": training_distance,
        "sample_count": validation_sample_count,
        "provenance_fingerprints": {
            key: provenance[key]
            for key in (
                "source_subset_fingerprint",
                "prompt_bank_fingerprint",
                "split_fingerprint",
                "config_fingerprint",
            )
        },
        "per_mode_diagnostics": {
            mode: stage0_metrics[mode] for mode in PROMPT_MODES
        },
        "overall_diagnostics": stage0_metrics["overall"],
        "overall_distribution_aggregation": (
            "concatenated_class_detailed_generalized_distance_tensors"
        ),
        "standard_deviation_convention": "population_std_unbiased_false",
    }


def print_stage0_diagnostics_summary(document):
    print("Stage 0 distance diagnostics (frozen CLIP validation only)")
    print(
        "mode | d_pos mean/min/p25/median/p75/max | "
        "d_neg mean/min/p25/median/p75/max | frac d_neg < margin | "
        "negative_loss | pairwise accuracy | retrieval R@1"
    )
    rows = list(PROMPT_MODES) + ["overall"]
    for mode in rows:
        metrics = (
            document["overall_diagnostics"]
            if mode == "overall"
            else document["per_mode_diagnostics"][mode]
        )
        positive = metrics["positive_distance_distribution"]
        negative = metrics["negative_distance_distribution"]
        print(
            "{} | {:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f} | "
            "{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f} | "
            "{:.6f} | {:.12g} | {:.6f} | {:.6f}".format(
                mode,
                positive["mean"],
                positive["min"],
                positive["p25"],
                positive["median"],
                positive["p75"],
                positive["max"],
                negative["mean"],
                negative["min"],
                negative["p25"],
                negative["median"],
                negative["p75"],
                negative["max"],
                metrics["negative_fraction_below_margin"],
                metrics["negative_loss"],
                metrics["pairwise_alignment_accuracy"],
                metrics["retrieval_r_at_1"],
            )
        )


def run_stage0_diagnostics_only(
    clip_model,
    ffn,
    adapter,
    image_root,
    split,
    bank,
    class_by_image,
    processor,
    args,
    provenance,
):
    val_names = list(split.val_images)
    stage0_metrics = evaluate_stage(
        "clip",
        clip_model,
        ffn,
        adapter,
        image_root,
        val_names,
        bank,
        class_by_image,
        processor,
        args,
    )
    document = build_stage0_diagnostics_document(
        stage0_metrics, args, provenance, len(val_names)
    )
    print_stage0_diagnostics_summary(document)
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().absolute()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "stage0_distance_diagnostics.json"
        atomic_json_write(document, output_path)
        print("stage0_distance_diagnostics={}".format(output_path))
    print(
        "Stage-0 diagnostics completed without FFN/adapter training or optimizer state."
    )
    return document


def main(argv=None):
    args = parse_args(argv)
    try:
        validate_args(args)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        set_reproducible_seed(args.seed)
        (
            assets,
            clip_path,
            image_root,
            bank,
            split,
            split_manifest,
            class_by_image,
        ) = resolve_and_validate_data(args)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RichAlignmentError("CUDA was requested but is unavailable")
        processor, tokenizer, clip_model, loading_info = load_offline_clip(
            clip_path, device
        )
        ffn = VisualAlignmentFFN(
            clip_model.config.projection_dim, args.dropout
        ).to(device)
        adapter = TokenTextAdapter(
            clip_model.config.text_config.hidden_size, args.dropout
        ).to(device)
        config_document, provenance = build_configs(args, clip_path, bank, split)
        parameter_counts = {
            "frozen_clip": sum(p.numel() for p in clip_model.parameters()),
            "trainable_ffn": sum(p.numel() for p in ffn.parameters()),
            "trainable_adapter": sum(p.numel() for p in adapter.parameters()),
        }
        config_document["parameter_counts"] = parameter_counts
        print("Offline CLIP preflight passed.")
        print("CLIPProcessor=OK")
        print("CLIPTokenizer=OK vocab_size={}".format(tokenizer.vocab_size))
        print("CLIPModel=OK loading_info={}".format(json.dumps(loading_info)))
        print(
            "CLIPTextEncoder=OK hidden_size={}".format(
                clip_model.config.text_config.hidden_size
            )
        )
        print(
            "CLIPVisionEncoder=OK hidden_size={}".format(
                clip_model.config.vision_config.hidden_size
            )
        )
        print(
            "CLIPTextProjection=OK shape={}".format(
                tuple(clip_model.text_projection.weight.shape)
            )
        )
        print(
            "CLIPVisualProjection=OK shape={}".format(
                tuple(clip_model.visual_projection.weight.shape)
            )
        )
        print("parameter_counts={}".format(json.dumps(parameter_counts)))
        print("source_subset_fingerprint={}".format(bank.selected_image_fingerprint))
        print("prompt_bank_fingerprint={}".format(bank.file_fingerprint))
        print("split_fingerprint={}".format(split.fingerprint))
        print("alignment_train_count={}".format(len(split.train_images)))
        print("alignment_validation_count={}".format(len(split.val_images)))
        print(
            "normalize_embeddings={}".format(
                str(args.normalize_embeddings).lower()
            )
        )
        print(
            "training_distance={}".format(
                config_document["research_config"]["training_distance"]
            )
        )

        if args.validate_only:
            smoke = run_forward_smoke(
                clip_model,
                processor,
                ffn,
                adapter,
                image_root,
                split,
                bank,
                args,
            )
            print("forward_smoke={}".format(json.dumps(smoke, sort_keys=True)))
            print("Validation-only completed without training or output writes.")
            return 0

        if args.stage0_diagnostics_only:
            run_stage0_diagnostics_only(
                clip_model,
                ffn,
                adapter,
                image_root,
                split,
                bank,
                class_by_image,
                processor,
                args,
                provenance,
            )
            return 0

        output_dir = Path(args.output_dir).expanduser().absolute()
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_write(config_document, output_dir / "alignment_config.json")
        atomic_json_write(
            split_manifest, output_dir / "alignment_split_manifest.json"
        )
        val_names = list(split.val_images)
        train_names = list(split.train_images)
        stage0 = evaluate_stage(
            "clip",
            clip_model,
            ffn,
            adapter,
            image_root,
            val_names,
            bank,
            class_by_image,
            processor,
            args,
        )
        atomic_json_write(
            {"stage0_clip": stage0}, output_dir / "stage0_clip_metrics.json"
        )

        resume_checkpoint = None
        resume_stage = None
        if args.resume:
            resume_checkpoint = load_trusted_legacy_checkpoint(args.resume, "cpu")
            validate_resume_provenance(provenance, resume_checkpoint)
            resume_stage = resume_checkpoint.get("stage")
            if resume_stage not in ("ffn", "adapter"):
                raise RichAlignmentError("Resume checkpoint has invalid stage")

        ffn_start = 0
        ffn_best_metric = float("inf")
        ffn_best_epoch = None
        ffn_best_state = None
        ffn_optimizer = torch.optim.Adam(ffn.parameters(), lr=args.lr_ffn)
        if resume_stage == "ffn":
            ffn.load_state_dict(
                resume_checkpoint["trainable_state_dict"], strict=True
            )
            ffn_optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            ffn_start = int(resume_checkpoint["next_epoch"])
            ffn_best_metric = float(
                resume_checkpoint["best_validation_overall_contrastive_loss"]
            )
            ffn_best_epoch = resume_checkpoint["best_epoch"]
            ffn_best_state = resume_checkpoint["best_state_dict"]
            restore_rng_state(resume_checkpoint.get("rng_state"))
        elif resume_stage == "adapter":
            ffn.load_state_dict(resume_checkpoint["ffn_state_dict"], strict=True)
        if resume_stage != "adapter":
            ffn_best_metric, ffn_best_epoch, ffn_best_state = train_stage(
                "ffn",
                ffn_start,
                args.ffn_epochs,
                clip_model,
                ffn,
                adapter,
                ffn_optimizer,
                image_root,
                train_names,
                val_names,
                bank,
                class_by_image,
                processor,
                args,
                output_dir,
                config_document,
                provenance,
                ffn_best_metric,
                ffn_best_epoch,
                ffn_best_state,
            )
        atomic_torch_save(cpu_state_dict(ffn), output_dir / "ffn_best.pt")
        stage1 = evaluate_stage(
            "ffn",
            clip_model,
            ffn,
            adapter,
            image_root,
            val_names,
            bank,
            class_by_image,
            processor,
            args,
        )
        atomic_json_write(
            {"stage1_ffn": stage1}, output_dir / "stage1_ffn_metrics.json"
        )

        adapter_start = 0
        adapter_best_metric = float("inf")
        adapter_best_epoch = None
        adapter_best_state = None
        adapter_optimizer = torch.optim.Adam(
            adapter.parameters(), lr=args.lr_adapter
        )
        if resume_stage == "adapter":
            adapter.load_state_dict(
                resume_checkpoint["trainable_state_dict"], strict=True
            )
            adapter_optimizer.load_state_dict(
                resume_checkpoint["optimizer_state_dict"]
            )
            adapter_start = int(resume_checkpoint["next_epoch"])
            adapter_best_metric = float(
                resume_checkpoint["best_validation_overall_contrastive_loss"]
            )
            adapter_best_epoch = resume_checkpoint["best_epoch"]
            adapter_best_state = resume_checkpoint["best_state_dict"]
            restore_rng_state(resume_checkpoint.get("rng_state"))
        (
            adapter_best_metric,
            adapter_best_epoch,
            adapter_best_state,
        ) = train_stage(
            "adapter",
            adapter_start,
            args.adapter_epochs,
            clip_model,
            ffn,
            adapter,
            adapter_optimizer,
            image_root,
            train_names,
            val_names,
            bank,
            class_by_image,
            processor,
            args,
            output_dir,
            config_document,
            provenance,
            adapter_best_metric,
            adapter_best_epoch,
            adapter_best_state,
        )
        atomic_torch_save(
            cpu_state_dict(adapter), output_dir / "adapter_best.pt"
        )
        stage2 = evaluate_stage(
            "adapter",
            clip_model,
            ffn,
            adapter,
            image_root,
            val_names,
            bank,
            class_by_image,
            processor,
            args,
        )
        atomic_json_write(
            {"stage2_adapter": stage2},
            output_dir / "stage2_adapter_metrics.json",
        )
        summary = {
            "stage0_clip": stage0,
            "stage1_ffn": stage1,
            "stage2_adapter": stage2,
            "best_checkpoints": {
                "ffn": {
                    "epoch": ffn_best_epoch,
                    "validation_overall_contrastive_loss": ffn_best_metric,
                },
                "adapter": {
                    "epoch": adapter_best_epoch,
                    "validation_overall_contrastive_loss": adapter_best_metric,
                },
            },
            "provenance": provenance,
            "parameter_counts": parameter_counts,
            "normalize_embeddings": args.normalize_embeddings,
            "training_distance": config_document["research_config"][
                "training_distance"
            ],
        }
        atomic_json_write(summary, output_dir / "alignment_summary.json")
        print("Alignment training completed: {}".format(output_dir))
        return 0
    except (
        RichAlignmentError,
        RichPromptTrainingError,
        FileNotFoundError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
    ) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
