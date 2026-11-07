"""Simultaneous four-plane Gerchberg-Saxton phase retrieval."""

import torch
from torch import nn

from fresnel import IMAGE_SIZE
from optics_torch import FresnelPlanes, centered_fft


class MultiPlaneGS(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        self.optics = FresnelPlanes()
        generator = torch.Generator().manual_seed(seed)
        phase = torch.rand((4, IMAGE_SIZE, IMAGE_SIZE), generator=generator) * (2 * torch.pi)
        self.register_buffer("target_phasor", torch.exp(1j * phase))

    def forward(self, target, iterations=100, initial_phase=None):
        """Project onto all four target amplitudes and the common phase-only DOE."""
        if iterations < 0:
            raise ValueError("The number of GS iterations must be nonnegative.")
        phase = initial_phase
        if phase is None:
            fields = self.optics.backproject(target * self.target_phasor)
            rms = fields.abs().square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
            phase = torch.angle((fields / rms).mean(dim=1))
        target_rms = target / target.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        for _ in range(iterations):
            wave = centered_fft(torch.exp(1j * phase[:, None]) * self.optics.chirp)
            constrained = torch.polar(target_rms, torch.angle(wave))
            phase = torch.angle(self.optics.backproject(constrained).mean(dim=1))
        return phase
