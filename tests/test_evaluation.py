import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import zlib

import numpy as np
from PIL import Image

from evaluate import load_grayscale_png
from fresnel import DISTANCES_MM, PIXEL_PITCH_M, WAVELENGTH_M, propagate_fresnel
from metrics import normalize_amplitude, pearson_correlation, structural_similarity


def direct_fresnel(phase, distance_m):
    """Reference sampled Fresnel integral using the coordinate differences."""
    size = phase.shape[0]
    scale = WAVELENGTH_M * distance_m
    input_axis = (np.arange(size) - size // 2) * PIXEL_PITCH_M
    output_axis = (np.arange(size) - size // 2) * scale / (size * PIXEL_PITCH_M)
    input_x, input_y = np.meshgrid(input_axis, input_axis)
    result = np.empty_like(phase, dtype=complex)
    for row, y in enumerate(output_axis):
        for column, x in enumerate(output_axis):
            kernel = np.pi * ((x - input_x)**2 + (y - input_y)**2) / scale
            result[row, column] = np.sum(np.exp(1j * (phase + kernel)))
    carrier = np.exp(2j * np.pi * distance_m / WAVELENGTH_M)
    return result * carrier * PIXEL_PITCH_M**2 / (1j * scale)


class FresnelTests(unittest.TestCase):
    def test_complex_field_matches_direct_sum(self):
        rng = np.random.default_rng(42)
        for size in (5, 6):
            phase = rng.uniform(-np.pi, np.pi, size=(size, size))
            for distance_mm in DISTANCES_MM:
                with self.subTest(size=size, distance_mm=distance_mm):
                    distance_m = distance_mm / 1000
                    np.testing.assert_allclose(
                        propagate_fresnel(phase, distance_m),
                        direct_fresnel(phase, distance_m),
                        rtol=1e-12,
                        atol=1e-14,
                    )

    def test_tilted_lens_focuses_at_expected_pixel(self):
        size = 128
        distance_m = 24.1e-3
        axis = np.arange(size) - size // 2
        x, y = np.meshgrid(axis, axis)
        lens = -np.pi * PIXEL_PITCH_M**2 * (x**2 + y**2) / (WAVELENGTH_M * distance_m)
        tilt = 2 * np.pi * (5 * x - 3 * y) / size
        amplitude = np.abs(propagate_fresnel(lens + tilt, distance_m))
        peak = np.unravel_index(amplitude.argmax(), amplitude.shape)
        self.assertEqual(peak, (size // 2 - 3, size // 2 + 5))
        self.assertAlmostEqual(amplitude.sum() / amplitude.max(), 1, places=10)


class MetricTests(unittest.TestCase):
    def test_normalization_and_identical_nonconstant_images(self):
        a = np.arange(144, dtype=float).reshape(12, 12) + 2
        original = normalize_amplitude(a)
        reconstructed = normalize_amplitude(9 * a)
        self.assertEqual(original.max(), 1)
        self.assertGreater(original.min(), 0)
        self.assertAlmostEqual(pearson_correlation(original, reconstructed), 1)
        self.assertAlmostEqual(structural_similarity(original, reconstructed), 1)
        self.assertAlmostEqual(pearson_correlation(original, 1 - original), -1)

    def test_constant_images(self):
        zeros = np.zeros((11, 11))
        ones = np.ones_like(zeros)
        np.testing.assert_array_equal(normalize_amplitude(zeros), zeros)
        self.assertIsNone(pearson_correlation(zeros, ones))
        self.assertAlmostEqual(structural_similarity(zeros, zeros), 1)
        self.assertAlmostEqual(structural_similarity(zeros, ones), 0.0001 / 1.0001)

    def test_ssim_matches_direct_weighted_windows(self):
        rng = np.random.default_rng(17)
        a = rng.random((12, 13))
        b = 0.7 * a + 0.3 * rng.random(a.shape)
        y, x = np.mgrid[-5:6, -5:6]
        weights = np.exp(-(x**2 + y**2) / (2 * 1.5**2))
        weights /= weights.sum()
        scores = []
        for row in range(2):
            for column in range(3):
                window = np.s_[row:row + 11, column:column + 11]
                left, right = a[window], b[window]
                mean_left = np.sum(left * weights)
                mean_right = np.sum(right * weights)
                centered_left, centered_right = left - mean_left, right - mean_right
                variance = np.sum((centered_left**2 + centered_right**2) * weights)
                covariance = np.sum(centered_left * centered_right * weights)
                scores.append(
                    (2 * mean_left * mean_right + 0.01**2)
                    * (2 * covariance + 0.03**2)
                    / ((mean_left**2 + mean_right**2 + 0.01**2) * (variance + 0.03**2))
                )
        self.assertAlmostEqual(structural_similarity(a, b), np.mean(scores), places=13)


class EvaluationTests(unittest.TestCase):
    def test_cli_reconstructs_quantized_mask_on_four_planes(self):
        rng = np.random.default_rng(12)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            phase_path = root / "doe.png"
            pixels = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
            Image.fromarray(pixels).save(phase_path)
            phase = np.pi * (2 * pixels.astype(float) / 255 - 1)
            targets = []
            for index, distance_mm in enumerate(DISTANCES_MM):
                target = root / f"target_{index}.png"
                amplitude = np.abs(propagate_fresnel(phase, distance_mm / 1000))
                expected = np.rint(255 * amplitude / amplitude.max()).astype(np.uint8)
                Image.fromarray(expected).save(target)
                targets.append(target)

            script = Path(__file__).resolve().parents[1] / "evaluate.py"
            output = root / "output"
            result = subprocess.run(
                [
                    sys.executable, "-B", str(script), "--phase", str(phase_path),
                    "--targets", *map(str, targets), "--output", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Pearson", result.stdout)
            scores = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual([score["distance_mm"] for score in scores], list(DISTANCES_MM))
            for score, target in zip(scores, targets):
                self.assertEqual(score["target"], str(target))
                np.testing.assert_array_equal(
                    load_grayscale_png(output / score["reconstruction"]),
                    load_grayscale_png(target),
                )
                self.assertGreater(score["pearson"], 0.999)
                self.assertGreater(score["ssim"], 0.999)

    def test_rgb_input_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.png"
            Image.new("RGB", (128, 128)).save(path)
            with self.assertRaisesRegex(ValueError, "8-bit grayscale PNG"):
                load_grayscale_png(path)

    def test_four_bit_grayscale_input_is_rejected(self):
        def png_chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(
                ">I", zlib.crc32(kind + data)
            )

        header = struct.pack(">IIBBBBB", 128, 128, 4, 0, 0, 0, 0)
        pixels = zlib.compress(bytes(128 * (1 + 128 // 2)))
        png = b"\x89PNG\r\n\x1a\n" + (
            png_chunk(b"IHDR", header) + png_chunk(b"IDAT", pixels) + png_chunk(b"IEND", b"")
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gray4.png"
            path.write_bytes(png)
            with Image.open(path) as image:
                self.assertEqual(image.mode, "L")
            with self.assertRaisesRegex(ValueError, "8-bit grayscale PNG"):
                load_grayscale_png(path)


if __name__ == "__main__":
    unittest.main()
