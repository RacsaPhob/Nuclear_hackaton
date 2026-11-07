"""Reconstruct four planes from an 8-bit DOE and evaluate their amplitudes."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fresnel import DISTANCES_MM, IMAGE_SIZE, propagate_fresnel
from metrics import normalize_amplitude, pearson_correlation, structural_similarity


def load_grayscale_png(path: Path) -> np.ndarray:
    """Read a single-frame 128×128 8-bit grayscale PNG."""
    with path.open("rb") as source:
        # The IHDR bit-depth byte distinguishes 8-bit PNGs from 2/4-bit
        # grayscale PNGs, which Pillow also loads in mode L.
        source.seek(24)
        bit_depth = source.read(1)
        with Image.open(source) as image:
            if (
                image.format != "PNG"
                or bit_depth != b"\x08"
                or image.mode != "L"
                or image.size != (IMAGE_SIZE, IMAGE_SIZE)
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ValueError(f"{path}: expected a 128×128 8-bit grayscale PNG.")
            return np.array(image)


def evaluate(phase_path: Path, target_paths: list[Path], output: Path) -> list[dict]:
    if len(target_paths) != len(DISTANCES_MM):
        raise ValueError("Exactly four target images are required.")

    phase = load_grayscale_png(phase_path) * (2 * np.pi / 255) - np.pi
    targets = [normalize_amplitude(load_grayscale_png(path)) for path in target_paths]
    output.mkdir(parents=True)

    scores = []
    for distance_mm, target, target_path in zip(DISTANCES_MM, targets, target_paths):
        field = propagate_fresnel(phase, distance_mm / 1000)
        restored = normalize_amplitude(np.abs(field))
        name = f"reconstruction_{distance_mm:.1f}mm.png"
        Image.fromarray(np.rint(255 * restored).astype(np.uint8)).save(output / name)
        scores.append(
            {
                "distance_mm": distance_mm,
                "target": str(target_path),
                "reconstruction": name,
                "pearson": pearson_correlation(target, restored),
                "ssim": structural_similarity(target, restored),
            }
        )
    (output / "metrics.json").write_text(
        json.dumps(scores, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, type=Path, help="8-bit DOE phase PNG")
    parser.add_argument(
        "--targets",
        required=True,
        nargs=4,
        type=Path,
        help="Target PNGs for 24.1, 25.3, 26.5, and 27.7 mm, in that order",
    )
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    args = parser.parse_args()
    try:
        scores = evaluate(args.phase, args.targets, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print("Depth (mm)    Pearson       SSIM")
    for score in scores:
        pearson = score["pearson"]
        correlation = "undefined" if pearson is None else f"{pearson:.6f}"
        print(f"{score['distance_mm']:>10.1f}    {correlation:>9}   {score['ssim']:>8.6f}")


if __name__ == "__main__":
    main()
