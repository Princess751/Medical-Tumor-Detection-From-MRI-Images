"""MRI image preprocessing for inference and training."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ml.config import IMAGE_SIZE, MEAN, STD


def _to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "L":
        return Image.merge("RGB", (pil_image, pil_image, pil_image))
    if pil_image.mode != "RGB":
        return pil_image.convert("RGB")
    return pil_image


def load_image_from_bytes(data: bytes) -> Image.Image:
    return _to_rgb(Image.open(io.BytesIO(data)))


def load_image_from_path(path: Path | str) -> Image.Image:
    return _to_rgb(Image.open(path))


def apply_clahe(pil_image: Image.Image) -> Image.Image:
    """Contrast-limited adaptive histogram equalization — common MRI preprocessing."""
    gray = np.array(pil_image.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)
    return Image.fromarray(rgb)


def get_train_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def get_eval_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def preprocess_for_model(pil_image: Image.Image, use_clahe: bool = True) -> torch.Tensor:
    if use_clahe:
        pil_image = apply_clahe(pil_image)
    tensor = get_eval_transforms()(pil_image)
    return tensor.unsqueeze(0)
