import unittest

import torch
import torch.nn as nn

from utils import build_optimizer


class OptimizerFactoryTest(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))

    def test_builds_sgd_with_momentum_and_nesterov(self):
        optimizer = build_optimizer(
            self.model,
            "sgd",
            lr=1e-3,
            weight_decay=1e-4,
            momentum=0.9,
            nesterov=True,
        )

        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.defaults["lr"], 1e-3)
        self.assertEqual(optimizer.defaults["momentum"], 0.9)
        self.assertTrue(optimizer.defaults["nesterov"])

    def test_builds_adamw_without_decay_on_bias_or_norm(self):
        optimizer = build_optimizer(
            self.model,
            "adamw",
            lr=3e-5,
            weight_decay=1e-4,
            adamw_beta1=0.8,
            adamw_beta2=0.95,
        )

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["betas"], (0.8, 0.95))
        groups_by_decay = {
            group["weight_decay"]: group["params"] for group in optimizer.param_groups
        }
        self.assertEqual(len(groups_by_decay[1e-4]), 1)
        self.assertEqual(len(groups_by_decay[0.0]), 3)

        all_optimizer_parameters = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(
            all_optimizer_parameters,
            {id(parameter) for parameter in self.model.parameters()},
        )

    def test_rejects_unknown_optimizer(self):
        with self.assertRaisesRegex(ValueError, "Unsupported optimizer"):
            build_optimizer(self.model, "unknown", lr=1e-3)


if __name__ == "__main__":
    unittest.main()
