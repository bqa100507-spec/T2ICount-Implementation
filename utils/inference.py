import torch
from PIL import Image
from torchvision import transforms

from utils.tools import extract_patches, reassemble_patches


DENSITY_SCALE = 60.0
INFERENCE_IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_image_tensor(image_path, device):
    """Load one RGB image with the unchanged T2ICount test transform."""
    with Image.open(image_path) as image:
        tensor = INFERENCE_IMAGE_TRANSFORM(image.convert("RGB"))
    return tensor.unsqueeze(0).to(device)


def build_prompt_attention_mask(tokenizer, prompt, max_length=77):
    mask = torch.zeros(max_length)
    tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_length = tokens["input_ids"].shape[1]
    mask[1:1 + prompt_length] = 1
    return mask


def prepare_image_patches(inputs, patch_size=384, stride=None):
    stride = patch_size if stride is None else stride
    patches, num_h, num_w = extract_patches(
        inputs, patch_size=patch_size, stride=stride
    )
    return patches, num_h, num_w, inputs.size(2), inputs.size(3)


def predict_density(
    model,
    inputs,
    prompt,
    prompt_attention_mask,
    batch_size=16,
    patch_size=384,
    stride=None,
    prepared_patches=None,
):
    """Run the original patch inference, reassembly, and /60 scaling path."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    stride = patch_size if stride is None else stride
    patch_info = prepared_patches or prepare_image_patches(inputs, patch_size, stride)
    patches, num_h, num_w, image_height, image_width = patch_info
    device = inputs.device

    mask = prompt_attention_mask.to(device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(2).unsqueeze(3)

    outputs = []
    with torch.no_grad():
        for start in range(0, patches.size(0), batch_size):
            end = min(start + batch_size, patches.size(0))
            chunk_size = end - start
            output = model(
                patches[start:end],
                [prompt] * chunk_size,
                mask.repeat(chunk_size, 1, 1, 1),
            )[0]
            outputs.append(output)

        density = reassemble_patches(
            torch.cat(outputs, dim=0),
            num_h,
            num_w,
            image_height,
            image_width,
            patch_size=patch_size,
            stride=stride,
        ) / DENSITY_SCALE
    return density


def predict_count(*args, **kwargs):
    return torch.sum(predict_density(*args, **kwargs)).item()
