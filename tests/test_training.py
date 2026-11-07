import unittest

import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest("Install requirements-training.txt to run the neural-network checks.")

from fresnel import DISTANCES_MM, propagate_fresnel
from metrics import pearson_correlation, structural_similarity
from model import PhaseUNet
from optics_torch import AmplitudeObjective, FresnelPlanes, quantize_phase


class DifferentiableOpticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_matches_numpy_after_phase_quantization(self):
        phase = torch.tensor(np.random.default_rng(8).uniform(-np.pi, np.pi, (1, 128, 128)), dtype=torch.float32)
        phase = quantize_phase(phase)
        reconstructed = FresnelPlanes()(phase).numpy()[0]
        for index, distance_mm in enumerate(DISTANCES_MM):
            expected = np.abs(propagate_fresnel(phase[0].numpy(), distance_mm / 1000))
            expected /= expected.max()
            np.testing.assert_allclose(reconstructed[index], expected, atol=2e-6, rtol=2e-5)

    def test_metrics_match_reference(self):
        rng = np.random.default_rng(3)
        a = rng.random((1, 4, 128, 128))
        b = 0.7 * a + 0.3 * rng.random(a.shape)
        objective = AmplitudeObjective().double()
        ssim, correlation = objective.metrics(torch.from_numpy(a), torch.from_numpy(b))
        for plane in range(4):
            self.assertAlmostEqual(ssim[0, plane].item(), structural_similarity(a[0, plane], b[0, plane]), places=7)
            self.assertAlmostEqual(correlation[0, plane].item(), pearson_correlation(a[0, plane], b[0, plane]), places=10)

    def test_propagation_gradient_matches_finite_difference(self):
        torch.manual_seed(5)
        phase = torch.rand(1, 128, 128, dtype=torch.float64, requires_grad=True)
        direction = torch.randn_like(phase)
        target = torch.rand(1, 4, 128, 128, dtype=torch.float64)
        optics = FresnelPlanes()
        loss = (optics(phase) - target).square().mean()
        loss.backward()
        analytic = (phase.grad * direction).sum().item()
        step = 1e-5
        with torch.no_grad():
            upper = (optics(phase + step * direction) - target).square().mean()
            lower = (optics(phase - step * direction) - target).square().mean()
        numeric = ((upper - lower) / (2 * step)).item()
        self.assertAlmostEqual(analytic, numeric, places=7)

    def test_network_receives_gradient_through_quantization(self):
        torch.manual_seed(6)
        model = PhaseUNet(width=4)
        target = torch.rand(2, 4, 128, 128)
        phase = model(target)
        self.assertEqual(tuple(phase.shape), (2, 128, 128))
        self.assertLessEqual(phase.abs().max().item(), np.pi)
        restored = model.optics(quantize_phase(phase, straight_through=True))
        loss, _, _ = AmplitudeObjective()(restored, target)
        loss.backward()
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())
        self.assertGreater(model.output.weight.grad.abs().max().item(), 0)

    def test_depth_penalty_detects_identical_reconstructions(self):
        torch.manual_seed(19)
        targets = torch.rand(2, 4, 128, 128)
        objective = AmplitudeObjective()
        penalty, correct = objective.depth_separation(targets, targets)
        self.assertLess(penalty.max().item(), 1e-6)
        self.assertTrue(correct.all())
        mixed = targets.mean(1, keepdim=True).expand_as(targets)
        penalty, correct = objective.depth_separation(mixed, targets)
        self.assertGreater(penalty.mean().item(), 0.2)
        self.assertAlmostEqual(correct.float().mean().item(), 0.25)


if __name__ == "__main__":
    unittest.main()
