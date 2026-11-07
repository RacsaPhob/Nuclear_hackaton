"""Single-FFT Fresnel propagation for the hackathon's phase-only DOE."""

import numpy as np


IMAGE_SIZE = 128
WAVELENGTH_M = 532e-9
PIXEL_PITCH_M = 10e-6
DISTANCES_MM = (24.1, 25.3, 26.5, 27.7)


def propagate_fresnel(
    phase: np.ndarray,
    distance_m: float,
    *,
    wavelength_m: float = WAVELENGTH_M,
    pixel_pitch_m: float = PIXEL_PITCH_M,
) -> np.ndarray:
    """Return the complex field on the native, centered output grid.

    The input is a square phase array in radians under unit-amplitude plane
    illumination. Row and column correspond to y and x. Output sampling is
    wavelength_m * distance_m / (N * pixel_pitch_m) on both axes.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2 or phase.shape[0] != phase.shape[1] or not phase.size:
        raise ValueError("Phase must be a nonempty square array.")
    if not np.isfinite(phase).all():
        raise ValueError("Phase must contain finite values.")
    if any(
        not np.isfinite(value) or value <= 0
        for value in (distance_m, wavelength_m, pixel_pitch_m)
    ):
        raise ValueError("Distance, wavelength, and pixel pitch must be positive and finite.")

    size = phase.shape[0]
    indices = np.arange(size) - size // 2
    radius_squared = indices[:, None] ** 2 + indices[None, :] ** 2
    scale = wavelength_m * distance_m
    output_pitch = scale / (size * pixel_pitch_m)
    input_chirp = np.pi * pixel_pitch_m**2 / scale * radius_squared
    output_chirp = np.pi * output_pitch**2 / scale * radius_squared

    input_field = np.exp(1j * (phase + input_chirp))
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(input_field)))
    carrier = np.exp(2j * np.pi * distance_m / wavelength_m)
    prefactor = carrier * pixel_pitch_m**2 / (1j * scale)
    return prefactor * np.exp(1j * output_chirp) * spectrum
