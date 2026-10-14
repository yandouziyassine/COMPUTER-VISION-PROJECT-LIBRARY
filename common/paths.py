"""Shared path helpers.

Data, model weights and caches live outside the repo in a sibling ``cv-data`` folder,
so the repo stays small and nothing large is ever committed. Override with the
``CV_DATA`` environment variable.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("CV_DATA", REPO_ROOT.parent / "cv-data"))
CACHE_ROOT = DATA_ROOT / ".cache"

# Keep model downloads off the (small) system drive.
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
# Ultralytics keeps its settings (and default dataset/runs folders) here instead of C:.
# Import this module *before* ultralytics for it to take effect.
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_ROOT / "ultralytics"))


def data_dir(project: str) -> Path:
    """Folder for one project's datasets, e.g. data_dir("solar")."""
    path = DATA_ROOT / project
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(project: str) -> Path:
    """Folder for one project's weights, logs and other large outputs."""
    path = DATA_ROOT / project / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path
