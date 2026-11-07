"""Load a prepared four-plane target as a normalized NumPy array."""

from pathlib import Path

import numpy as np

from evaluate import load_grayscale_png
from metrics import normalize_amplitude


def load_quadruplet(dataset_dir: Path, record: dict) -> np.ndarray:
    """Return float32 amplitudes shaped (4, 128, 128), in depth order."""
    if len(record["images"]) != 4:
        raise ValueError("A sample must contain exactly four image paths.")
    return np.stack([
        normalize_amplitude(load_grayscale_png(dataset_dir / image_path))
        for image_path in record["images"]
    ]).astype(np.float32)
