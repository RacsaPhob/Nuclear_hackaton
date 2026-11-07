import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Install requirements.txt to check the submission.")

from evaluate import load_grayscale_png


class SubmissionTests(unittest.TestCase):
    def test_five_sets_and_two_level_averaging(self):
        project = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            subprocess.run([
                sys.executable, "-B", str(project / "synthesize.py"),
                "--sets", str(project / "examples" / "sets"),
                "--output", str(output), "--device", "cpu",
            ], check=True, capture_output=True, text=True)
            report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["set_count"], 5)
            self.assertEqual(report["method"], "neural")
            for case in report["sets"]:
                path = output / case["doe"]
                self.assertEqual(load_grayscale_png(path).shape, (128, 128))
                header = path.read_bytes()[:29]
                self.assertEqual(header[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(struct.unpack(">II", header[16:24]), (128, 128))
                self.assertEqual(header[24:26], bytes([8, 0]))
                self.assertEqual(len(case["per_plane"]), 4)
                self.assertGreater(case["synthesis_ms"], 0)
                for metric in ("ssim", "pearson"):
                    self.assertAlmostEqual(case[metric], np.mean([p[metric] for p in case["per_plane"]]))
            for metric in ("ssim", "pearson"):
                self.assertAlmostEqual(report[metric], np.mean([case[metric] for case in report["sets"]]))


if __name__ == "__main__":
    unittest.main()
