"""Inference service for MRI tumor classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ml.config import CLASS_LABELS, CLASS_NAMES, DEFAULT_WEIGHTS
from ml.gradcam import GradCAM, heatmap_to_png_bytes, overlay_heatmap
from ml.model import load_model
from ml.preprocess import load_image_from_bytes, preprocess_for_model


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    tumor_detected: bool
    heatmap_png: bytes | None
    model_version: str
    limitations: list[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("heatmap_png", None)
        return d


class TumorDetector:
    def __init__(self, weights_path: Path | None = None, device: str | None = None):
        self.weights_path = Path(weights_path or DEFAULT_WEIGHTS)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self._gradcam: GradCAM | None = None
        self.model_version = "unloaded"

    def load(self) -> None:
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.weights_path}. "
                "Run scripts/setup_lab.py or scripts/train_model.py first."
            )
        self.model = load_model(self.weights_path, self.device)
        self._gradcam = GradCAM(self.model, self.model.layer4)
        checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            self.model_version = f"resnet18-epoch-{checkpoint.get('epoch', '?')}-acc-{checkpoint.get('val_acc', 0):.2f}"
        else:
            self.model_version = "resnet18"

    def predict(self, image_bytes: bytes, include_heatmap: bool = True) -> PredictionResult:
        if self.model is None:
            self.load()

        pil_image = load_image_from_bytes(image_bytes)
        tensor = preprocess_for_model(pil_image).to(self.device)
        tensor.requires_grad_(include_heatmap)

        with torch.set_grad_enabled(include_heatmap):
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

        class_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        predicted_class = CLASS_NAMES[class_idx]
        confidence = probs[class_idx]

        heatmap_png = None
        if include_heatmap and self._gradcam is not None:
            cam = self._gradcam.generate(tensor, class_idx)
            overlay = overlay_heatmap(pil_image, cam)
            heatmap_png = heatmap_to_png_bytes(overlay)

        prob_map = {CLASS_NAMES[i]: round(probs[i], 4) for i in range(len(CLASS_NAMES))}

        return PredictionResult(
            predicted_class=predicted_class,
            predicted_label=CLASS_LABELS[predicted_class],
            confidence=round(confidence, 4),
            probabilities=prob_map,
            tumor_detected=predicted_class != "no_tumor",
            heatmap_png=heatmap_png,
            model_version=self.model_version,
            limitations=[
                "Decision-support only — not a standalone diagnostic device.",
                "Model must be validated on hospital-specific scanners and protocols before clinical use.",
                "Grad-CAM highlights model attention, not confirmed tumor boundaries.",
                "False negatives and false positives occur; radiologist review is required.",
            ],
        )
