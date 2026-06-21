"""PyTorch Dataset for folder-structured MRI images."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from ml.config import CLASS_NAMES
from ml.preprocess import get_train_transforms, get_eval_transforms


class MRITumorDataset(Dataset):
    """
    Expected layout:
      data/train/<class_name>/*.jpg|png
      data/val/<class_name>/*.jpg|png
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, root: Path, train: bool = True):
        self.root = Path(root)
        self.train = train
        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

        for class_name in CLASS_NAMES:
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            label = self.class_to_idx[class_name]
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((path, label))

        self.transform = get_train_transforms() if train else get_eval_transforms()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), label
