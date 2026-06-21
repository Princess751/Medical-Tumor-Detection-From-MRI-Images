from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO / "weights"
DATA_DIR = REPO / "data"
OUTPUT_DIR = REPO / "output"

IMAGE_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ("glioma", "meningioma", "no_tumor", "pituitary")
CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No tumor detected",
    "pituitary": "Pituitary tumor",
}

# ImageNet normalization (applied after converting grayscale MRI to 3-channel)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEFAULT_WEIGHTS = WEIGHTS_DIR / "tumor_classifier.pt"
