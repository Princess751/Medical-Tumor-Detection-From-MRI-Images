"""Verify all required packages are importable."""

import sys


def main() -> int:
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        import torch
        import torchvision
        import cv2
        import PIL
    except ImportError as exc:
        print(f"FAIL: missing package - {exc}")
        return 1

    print("All packages OK")
    print(f"  torch {torch.__version__}")
    print(f"  fastapi {fastapi.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
