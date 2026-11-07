"""Image metrics for amplitudes normalized to their own maximum."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03

_positions = np.arange(SSIM_WINDOW_SIZE) - SSIM_WINDOW_SIZE // 2
_gaussian = np.exp(-0.5 * (_positions / SSIM_SIGMA) ** 2)
_gaussian /= _gaussian.sum()


def normalize_amplitude(amplitude: np.ndarray) -> np.ndarray:
    """Scale a nonnegative image to peak 1, preserving all-zero images."""
    amplitude = np.asarray(amplitude, dtype=np.float64)
    peak = amplitude.max()
    return amplitude / peak if peak > 0 else amplitude.copy()


def _gaussian_mean(image: np.ndarray) -> np.ndarray:
    horizontal = sliding_window_view(image, SSIM_WINDOW_SIZE, axis=1)
    filtered = np.einsum("...k,k->...", horizontal, _gaussian)
    vertical = sliding_window_view(filtered, SSIM_WINDOW_SIZE, axis=0)
    return np.einsum("...k,k->...", vertical, _gaussian)


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Return Pearson's r, or None when either image is constant."""
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return None
    return float(np.clip(np.dot(a, b) / denominator, -1, 1))


def structural_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM for equal-size 2D float arrays in [0, 1]."""
    if a.shape != b.shape or a.ndim != 2 or min(a.shape) < SSIM_WINDOW_SIZE:
        raise ValueError("SSIM requires equal-size 2D images of at least 11×11 pixels.")

    mean_a = _gaussian_mean(a)
    mean_b = _gaussian_mean(b)
    variance_a = _gaussian_mean(a * a) - mean_a**2
    variance_b = _gaussian_mean(b * b) - mean_b**2
    covariance = _gaussian_mean(a * b) - mean_a * mean_b
    c1, c2 = SSIM_K1**2, SSIM_K2**2
    luminance = (2 * mean_a * mean_b + c1) / (mean_a**2 + mean_b**2 + c1)
    structure = (2 * covariance + c2) / (variance_a + variance_b + c2)
    return float(np.mean(luminance * structure))
