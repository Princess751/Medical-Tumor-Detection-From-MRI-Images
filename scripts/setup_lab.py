"""
End-to-end lab setup: synthetic data -> train -> verify weights.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    subprocess.check_call(cmd, cwd=REPO, env=env)


def main() -> None:
    py = sys.executable
    run([py, str(REPO / "scripts" / "generate_synthetic_data.py")])
    run([py, str(REPO / "scripts" / "train_model.py"), "--epochs", "10"])
    weights = REPO / "weights" / "tumor_classifier.pt"
    if weights.exists():
        print(f"\nLab ready. Weights: {weights}")
        print("Start the app: .\\run.bat")
        print("Or: .venv\\Scripts\\uvicorn app.main:app --host 127.0.0.1 --port 8780")
    else:
        raise SystemExit("Training did not produce weights.")


if __name__ == "__main__":
    main()
