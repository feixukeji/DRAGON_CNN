import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils import asinh_normalize, load_asinh_stats
from data_preprocessing import load_or_compute_asinh_stats


class _FakeDataset:
    def __init__(self):
        self.labels = np.array([0, 1])
        self.images = [
            torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]),
            torch.tensor([[[4.0, 5.0], [6.0, 7.0]]]),
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], torch.tensor(int(self.labels[index]))


class AsinhNormalizationTest(unittest.TestCase):
    def test_fixed_limits_match_euclid_formula(self):
        image = torch.tensor([[[[-5.0, 0.0, 5.0, 10.0, 15.0]]]])
        actual = asinh_normalize(image, softening=0.1, vmin=[0.0], vmax=[10.0])

        clipped = np.clip(image.numpy(), 0.0, 10.0)
        x = clipped / (10.0 + 1e-6)
        expected = np.arcsinh(x / 0.1) / np.arcsinh(1.0 / 0.1)

        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(actual.min().item(), 0.0)
        self.assertLessEqual(actual.max().item(), 1.0)

    def test_per_cutout_per_channel_limits_are_independent(self):
        base = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
        batch = torch.cat([base, base * 10.0 + 100.0], dim=0)

        actual = asinh_normalize(batch, low_pct=0.0, high_pct=100.0)

        torch.testing.assert_close(actual[0], actual[1])

    def test_constant_cutout_and_nonfinite_values_are_finite(self):
        constant = torch.full((1, 1, 3, 3), 7.0)
        normalized_constant = asinh_normalize(constant)
        self.assertEqual(torch.count_nonzero(normalized_constant), 0)

        image = torch.tensor([[[[float("nan"), float("-inf"), float("inf")]]]])
        normalized = asinh_normalize(image, vmin=[0.0], vmax=[10.0])
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertEqual(normalized[0, 0, 0, 0], 0)
        self.assertEqual(normalized[0, 0, 0, 1], 0)
        self.assertGreater(normalized[0, 0, 0, 2], 0.99)

    def test_load_stats_validates_channel_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stats.json"
            path.write_text(json.dumps({"vmin": [0.0, 1.0], "vmax": [2.0, 3.0]}))

            self.assertEqual(load_asinh_stats(path, channels=2)["vmax"], [2.0, 3.0])
            with self.assertRaisesRegex(ValueError, "expected 1"):
                load_asinh_stats(path, channels=1)

    def test_missing_stats_are_computed_once_and_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing_stats.json"
            dataset = _FakeDataset()
            kwargs = {
                "low_pct": 0.0,
                "high_pct": 100.0,
                "sample_per_image": 0,
                "max_samples_per_channel": 0,
                "seed": 42,
                "show_progress": False,
            }

            computed_stats, computed = load_or_compute_asinh_stats(
                path, dataset, channels=1, **kwargs
            )
            loaded_stats, computed_again = load_or_compute_asinh_stats(
                path, dataset, channels=1, **kwargs
            )

            self.assertTrue(computed)
            self.assertFalse(computed_again)
            self.assertTrue(path.is_file())
            self.assertEqual(computed_stats["vmin"], [0.0])
            self.assertEqual(computed_stats["vmax"], [7.0])
            self.assertEqual(loaded_stats["vmax"], [7.0])


if __name__ == "__main__":
    unittest.main()
