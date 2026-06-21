"""
Generate synthetic MRI-like images for lab/demo training when real data is unavailable.

Real deployments should replace this with authorized hospital or public datasets
(e.g. Kaggle Brain Tumor MRI, BraTS) using scripts/download_dataset.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ml.config import CLASS_NAMES, DATA_DIR


def _base_brain(size: int = 224) -> np.ndarray:
    rng = np.random.default_rng()
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size // 2 + rng.integers(-8, 9), size // 2 + rng.integers(-8, 9)
    brain = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (size * 0.32) ** 2))
    noise = rng.normal(0, 0.04, (size, size))
    img = np.clip(brain + noise, 0, 1)
    return (img * 200 + 30).astype(np.uint8)


def _add_lesion(img: np.ndarray, lesion_type: str) -> np.ndarray:
    rng = np.random.default_rng()
    out = img.copy()
    h, w = out.shape
    lx = rng.integers(w // 4, 3 * w // 4)
    ly = rng.integers(h // 4, 3 * h // 4)
    radius = rng.integers(8, 22)

    yy, xx = np.mgrid[0:h, 0:w]
    mask = (xx - lx) ** 2 + (yy - ly) ** 2 <= radius**2

    if lesion_type == "glioma":
        out[mask] = np.clip(out[mask] * 0.55 + 40, 0, 255)
    elif lesion_type == "meningioma":
        out[mask] = np.clip(out[mask] * 1.25 + 25, 0, 255)
    elif lesion_type == "pituitary":
        # Sellar region — lower center
        ly = int(h * 0.62) + rng.integers(-6, 7)
        lx = w // 2 + rng.integers(-10, 11)
        mask = (xx - lx) ** 2 + (yy - ly) ** 2 <= (radius - 2) ** 2
        out[mask] = np.clip(out[mask] * 1.35 + 35, 0, 255)
    return out


def _to_pil(gray: np.ndarray) -> Image.Image:
    pil = Image.fromarray(gray, mode="L").convert("RGB")
    return pil.filter(ImageFilter.GaussianBlur(radius=0.6))


def generate_split(split: str, per_class: int, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_NAMES:
        dest = out_root / split / class_name
        dest.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            base = _base_brain()
            if class_name != "no_tumor":
                base = _add_lesion(base, class_name)
            pil = _to_pil(base)
            # Slight annotation ring for meningioma (skull-adjacent hint)
            if class_name == "meningioma":
                draw = ImageDraw.Draw(pil)
                draw.ellipse([30, 30, 194, 194], outline=(180, 180, 180), width=1)
            pil.save(dest / f"{class_name}_{split}_{i:04d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic MRI training images")
    parser.add_argument("--train-per-class", type=int, default=80)
    parser.add_argument("--val-per-class", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    generate_split("train", args.train_per_class, args.output)
    generate_split("val", args.val_per_class, args.output)
    print(f"Synthetic dataset written to {args.output}")


if __name__ == "__main__":
    main()
