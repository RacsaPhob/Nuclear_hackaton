"""Differentiable version of the validated four-plane Fresnel simulator."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fresnel import DISTANCES_MM, IMAGE_SIZE, PIXEL_PITCH_M, WAVELENGTH_M
from metrics import SSIM_K1, SSIM_K2, SSIM_SIGMA, SSIM_WINDOW_SIZE


def centered_fft(field, inverse=False):
    transform = torch.fft.ifft2 if inverse else torch.fft.fft2
    return torch.fft.fftshift(
        transform(torch.fft.ifftshift(field, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def quantize_phase(phase, straight_through=False):
    pixels = torch.round((phase + torch.pi) * (255 / (2 * torch.pi))).clamp(0, 255)
    quantized = pixels * (2 * torch.pi / 255) - torch.pi
    if straight_through:
        return phase + (quantized - phase).detach()
    return quantized


class FresnelPlanes(nn.Module):
    def __init__(self):
        super().__init__()
        axis = np.arange(IMAGE_SIZE) - IMAGE_SIZE // 2
        radius = axis[:, None] ** 2 + axis[None, :] ** 2
        distances = np.array(DISTANCES_MM)[:, None, None] / 1000
        chirp = np.exp(1j * np.pi * PIXEL_PITCH_M**2 * radius / (WAVELENGTH_M * distances))
        self.register_buffer("chirp", torch.from_numpy(chirp.astype(np.complex64)))

    def forward(self, phase):
        field = torch.exp(1j * phase[:, None]) * self.chirp
        amplitude = centered_fft(field).abs()
        # Output chirp has unit modulus and prefactor cancels in peak normalization.
        return amplitude / amplitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    def backproject(self, fields):
        return centered_fft(fields, inverse=True) * self.chirp.conj()


class AmplitudeObjective(nn.Module):
    def __init__(self):
        super().__init__()
        axis = torch.arange(SSIM_WINDOW_SIZE) - SSIM_WINDOW_SIZE // 2
        kernel = torch.exp(-0.5 * (axis / SSIM_SIGMA)**2)
        kernel = kernel / kernel.sum()
        self.register_buffer("horizontal", kernel.reshape(1, 1, 1, -1).repeat(20, 1, 1, 1))
        self.register_buffer("vertical", kernel.reshape(1, 1, -1, 1).repeat(20, 1, 1, 1))

    def metrics(self, reconstructed, target):
        moments = torch.cat(
            [reconstructed, target, reconstructed.square(), target.square(), reconstructed * target],
            dim=1,
        )
        moments = F.conv2d(moments, self.horizontal, groups=20)
        moments = F.conv2d(moments, self.vertical, groups=20)
        ma, mb, aa, bb, ab = moments.chunk(5, dim=1)
        c1, c2 = SSIM_K1**2, SSIM_K2**2
        ssim = ((2 * ma * mb + c1) * (2 * (ab - ma * mb) + c2)) / (
            (ma.square() + mb.square() + c1)
            * (aa - ma.square() + bb - mb.square() + c2)
        )
        a = reconstructed - reconstructed.mean(dim=(-2, -1), keepdim=True)
        b = target - target.mean(dim=(-2, -1), keepdim=True)
        denominator = (a.square().sum((-2, -1)) * b.square().sum((-2, -1))).sqrt()
        correlation = (a * b).sum((-2, -1)) / denominator.clamp_min(1e-8)
        return ssim.mean((-2, -1)), correlation

    def depth_separation(self, reconstructed, target):
        a = reconstructed.flatten(2)
        b = target.flatten(2)
        a = F.normalize(a - a.mean(-1, keepdim=True), dim=-1)
        b = F.normalize(b - b.mean(-1, keepdim=True), dim=-1)
        cross = a @ b.transpose(1, 2)
        target_similarity = b @ b.transpose(1, 2)
        own = cross.diagonal(dim1=1, dim2=2)[:, :, None]
        margin = 0.25 * (1 - target_similarity)
        off_diagonal = 1 - torch.eye(4, device=target.device, dtype=target.dtype)
        penalty = (F.relu(cross + margin - own) * off_diagonal).sum(-1) / 3
        correct = cross.argmax(-1) == torch.arange(4, device=target.device)
        return penalty, correct

    def forward(self, reconstructed, target):
        ssim, correlation = self.metrics(reconstructed, target)
        mse = F.mse_loss(reconstructed, target)
        separation, _ = self.depth_separation(reconstructed, target)
        loss = mse + 0.5 * (1 - ssim.mean()) + 0.25 * (1 - correlation.mean()) + 2 * separation.mean()
        return loss, ssim.mean(), correlation.mean()
