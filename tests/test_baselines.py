import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Install requirements-training.txt to check GS.")

from baselines import MultiPlaneGS
from model import PhaseUNet


class GerchbergSaxtonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.gs = MultiPlaneGS()

    def test_known_consistent_phase_is_a_fixed_point(self):
        torch.manual_seed(23)
        phase = torch.rand(2, 128, 128) * (2 * torch.pi) - torch.pi
        target = self.gs.optics(phase)
        restored_phase = self.gs(target, iterations=5, initial_phase=phase)
        torch.testing.assert_close(
            torch.exp(1j * restored_phase), torch.exp(1j * phase), atol=5e-6, rtol=1e-5
        )

    def test_zero_iterations_preserve_initial_phase(self):
        phase = torch.rand(1, 128, 128)
        target = torch.rand(1, 4, 128, 128)
        torch.testing.assert_close(self.gs(target, 0, phase), phase, atol=0, rtol=0)

    def test_initialization_matches_untrained_network_without_refinement(self):
        target = torch.rand(1, 4, 128, 128)
        model = PhaseUNet(width=4, refinement_steps=0)
        with torch.no_grad():
            torch.testing.assert_close(self.gs(target, 0), model(target), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
