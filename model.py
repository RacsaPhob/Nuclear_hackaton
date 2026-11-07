"""Four target amplitudes to a single phase-only DOE."""

import torch
from torch import nn
from torch.nn import functional as F

from fresnel import IMAGE_SIZE
from optics_torch import FresnelPlanes, centered_fft


def conv_block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(outputs, outputs, 3, padding=1),
        nn.SiLU(),
    )


class PhaseUNet(nn.Module):
    def __init__(self, width=16, refinement_steps=6):
        super().__init__()
        self.width = width
        self.refinement_steps = refinement_steps
        if refinement_steps:
            self.refinement_logits = nn.Parameter(torch.zeros(refinement_steps))
        else:
            self.register_buffer("refinement_logits", torch.empty(0))
        self.optics = FresnelPlanes()
        generator = torch.Generator().manual_seed(42)
        phase = torch.rand((4, IMAGE_SIZE, IMAGE_SIZE), generator=generator) * (2 * torch.pi)
        self.register_buffer("target_phasor", torch.exp(1j * phase))
        self.encoder1 = conv_block(8, width)
        self.encoder2 = conv_block(width, width * 2)
        self.encoder3 = conv_block(width * 2, width * 4)
        self.bridge = conv_block(width * 4, width * 8)
        self.decoder3 = conv_block(width * 12, width * 4)
        self.decoder2 = conv_block(width * 6, width * 2)
        self.decoder1 = conv_block(width * 3, width)
        self.output = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, target):
        fields = self.optics.backproject(target * self.target_phasor)
        scale = fields.abs().square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        fields = fields / scale
        features = torch.cat([fields.real, fields.imag], dim=1)
        e1 = self.encoder1(features)
        e2 = self.encoder2(F.avg_pool2d(e1, 2))
        e3 = self.encoder3(F.avg_pool2d(e2, 2))
        center = self.bridge(F.avg_pool2d(e3, 2))
        d3 = self.decoder3(torch.cat([F.interpolate(center, scale_factor=2, mode="nearest"), e3], dim=1))
        d2 = self.decoder2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], dim=1))
        d1 = self.decoder1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], dim=1))
        residual = self.output(d1)
        field = fields.mean(dim=1) + torch.complex(residual[:, 0], residual[:, 1])
        phase = torch.angle(field)
        target_rms = target / target.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        for step in 2 * torch.sigmoid(self.refinement_logits):
            wave = centered_fft(torch.exp(1j * phase[:, None]) * self.optics.chirp)
            magnitude = wave.abs().clamp_min(1e-6)
            constrained = wave + step * (target_rms - magnitude) * wave / magnitude
            phase = torch.angle(self.optics.backproject(constrained).mean(dim=1))
        return phase
